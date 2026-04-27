"""
All-in-One Anchor Evaluation for NeurIPS 2026 (AFM Version)
Dataset-Specific Neural Circuits in Medical LLMs

This script performs comprehensive evaluation of anchor neurons across three medical datasets:
- MedQA (4-way multiple choice)
- MedMCQA (4-way multiple choice)  
- PubMedQA (binary yes/no)

Outputs:
- Anchor vs random comparisons with statistical significance
- Per-anchor cross-dataset effect profiles
- K-means clustering of anchors by effect patterns
- Full reproducibility metadata

Author: [Your Name]
Date: March 2026
"""

import os
import json
import random
import hashlib
import numpy as np
from typing import Dict, List, Any, Tuple, Optional
from dataclasses import dataclass, asdict
from datetime import datetime

import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from scipy import stats
from sklearn.metrics import silhouette_score

# Optional local module path
_mednsq_lib_dir = os.getenv("MEDNSQ_LIB_DIR")
if _mednsq_lib_dir and _mednsq_lib_dir not in os.sys.path:
    os.sys.path.insert(0, _mednsq_lib_dir)

from mednsq_data import build_adversarial_pairs, load_mcq_dataset
from mednsq_probe import MedNSQProbe


# ============================================================================
# CONFIGURATION
# ============================================================================

@dataclass
class Config:
    """Immutable configuration for reproducibility."""
    # Model
    model_id: str = "openmed-community/AFM-4.5B-OpenMed-RL-CoT"
    anchor_file: str = "anchors_final.json"  # Or AFM-specific anchors if available
    
    # Dataset sizes (160 is optimal: enough power, not too costly)
    n_medqa: int = 400
    n_medmcqa: int = 400
    n_pubmedqa: int = 400
    
    # Anchor selection
    k_top: int = 64  # Number of top anchors to evaluate
    
    # Clustering
    k_clusters: int = 4  # Number of clusters (based on elbow method)
    
    # Random baseline
    n_random_trials: int = 1  # Only 1 seed, no ablation sampling
    random_seed: int = 42
    
    # Data caching (AFM base model)
    medqa_cache: str = "base_afm_medqa_pairs.txt"
    medmcqa_cache: str = "base_afm_medmcqa_pairs.txt"
    pubmedqa_cache: str = "base_afm_pubmedqa_pairs.txt"
    
    # Output
    output_file: str = "afm_anchor_complete_results.json"
    
    # PubMedQA settings
    max_contexts: int = 3
    max_context_chars: int = 2200
    
    def __post_init__(self):
        """Validate configuration."""
        assert self.k_top > 0, "k_top must be positive"
        assert self.k_clusters >= 2, "k_clusters must be at least 2"
        assert self.n_random_trials >= 1, "Need at least 1 trial for baseline"
        
    @property
    def config_dict(self) -> Dict:
        """Return config as dictionary for logging."""
        return asdict(self)
    
    @property
    def config_hash(self) -> str:
        """Return hash of config for reproducibility."""
        return hashlib.sha256(
            json.dumps(self.config_dict, sort_keys=True).encode()
        ).hexdigest()[:8]


CONFIG = Config()


# ============================================================================
# HARDCODED AFM ANCHORS - Sorted by Descending Mean Drop
# ============================================================================

AFM_ANCHORS = [
    (23, 9366), (13, 6927), (25, 10762), (14, 6171), (22, 5708),
    (22, 7310), (10, 9418), (13, 8175), (22, 10214), (10, 7547),
    (11, 14111), (17, 14058), (11, 15880), (16, 7083), (11, 1731),
    (14, 2176), (27, 7796), (25, 8241), (10, 1234), (13, 3492),
    (11, 818), (13, 6924), (23, 8318), (17, 15709), (17, 588),
    (17, 15839), (27, 15203), (20, 878), (10, 14742), (16, 15686),
    (22, 513), (27, 11037), (11, 553), (23, 1557), (10, 15578),
    (11, 12815), (10, 4760), (19, 8779), (13, 17388), (27, 8933),
    (17, 3696), (23, 17140), (20, 7699), (17, 11698), (25, 2244),
    (11, 464), (11, 10962), (25, 13915), (13, 11202), (23, 13125),
    (11, 22898), (23, 6331), (16, 18209), (19, 15377), (25, 9794),
    (27, 5909), (11, 5746), (19, 1830), (27, 12556), (25, 8216),
    (11, 16295), (25, 17726), (13, 14162), (14, 17780), (11, 17563),
    (17, 2594), (14, 15406), (14, 4433), (13, 14695), (20, 1056),
    (19, 12552), (20, 6598), (19, 15777), (19, 4300), (13, 18220),
    (11, 4637), (19, 17338), (14, 1531), (17, 2424), (25, 5417),
    (16, 10261), (27, 14033), (25, 3755), (14, 12940), (14, 11486),
    (23, 18170), (14, 11451), (23, 2716), (27, 2656), (23, 5522),
    (11, 3559),
]


# ============================================================================
# TOKEN UTILITIES
# ============================================================================

def get_single_token_id(tokenizer, text: str) -> int:
    """
    Robust token ID resolution for single tokens.
    
    Args:
        tokenizer: HuggingFace tokenizer
        text: Target text (e.g., "yes", "A")
        
    Returns:
        token_id: Integer token ID
        
    Raises:
        RuntimeError: If token cannot be resolved uniquely
    """
    # Try with space prefix (common for single tokens)
    for candidate in [f" {text}", text]:
        ids = tokenizer.encode(candidate, add_special_tokens=False)
        if len(ids) == 1:
            return int(ids[0])
    
    # Fallback: take first token if multiple
    ids = tokenizer.encode(f" {text}", add_special_tokens=False)
    if not ids:
        raise RuntimeError(f"Cannot resolve token id for: '{text}'")
    
    print(f"Warning: '{text}' split into {len(ids)} tokens, using first")
    return int(ids[0])


def get_letter_token_ids(tokenizer) -> Dict[str, int]:
    """Get token IDs for A, B, C, D."""
    return {l: get_single_token_id(tokenizer, l) for l in ["A", "B", "C", "D"]}


# ============================================================================
# DATA CACHING
# ============================================================================

def save_pairs(path: str, pairs: List[Dict[str, Any]]) -> None:
    """
    Save adversarial pairs to cache with metadata.
    
    Format: JSONL with one pair per line, includes metadata header.
    """
    with open(path, "w", encoding="utf-8") as f:
        # Write metadata header
        metadata = {
            "version": "1.0",
            "n_pairs": len(pairs),
            "created": datetime.now().isoformat(),
            "config_hash": CONFIG.config_hash
        }
        f.write("#" + json.dumps(metadata) + "\n")
        
        # Write pairs
        for p in pairs:
            row = {
                "input_ids": p["input_ids"][0].tolist(),
                "attention_mask": p["attention_mask"][0].tolist(),
                "safe_input_ids": p["safe_input_ids"][0].tolist(),
                "safe_attention_mask": p["safe_attention_mask"][0].tolist(),
                "pos_id": int(p["pos_id"]),
                "neg_id": int(p["neg_id"]),
            }
            f.write(json.dumps(row) + "\n")


def load_pairs(path: str) -> List[Dict[str, Any]]:
    """
    Load adversarial pairs from cache, skipping metadata.
    """
    pairs: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            row = json.loads(line)
            pairs.append({
                "input_ids": torch.tensor([row["input_ids"]], dtype=torch.long),
                "attention_mask": torch.tensor([row["attention_mask"]], dtype=torch.long),
                "safe_input_ids": torch.tensor([row["safe_input_ids"]], dtype=torch.long),
                "safe_attention_mask": torch.tensor([row["safe_attention_mask"]], dtype=torch.long),
                "pos_id": int(row["pos_id"]),
                "neg_id": int(row["neg_id"]),
            })
    return pairs


# ============================================================================
# DATASET LOADERS
# ============================================================================

def get_medqa_pairs(model, tokenizer, n_total: int) -> List[Dict[str, Any]]:
    """Load or build MedQA adversarial pairs."""
    if os.path.exists(CONFIG.medqa_cache):
        print(f"Loading MedQA from cache: {CONFIG.medqa_cache}")
        return load_pairs(CONFIG.medqa_cache)[:n_total]
    
    print(f"Building MedQA pairs (n={n_total})...")
    ds = load_mcq_dataset(n_total=n_total)
    pairs = build_adversarial_pairs(
        model=model, 
        tokenizer=tokenizer, 
        dataset=ds, 
        n_calib=len(ds)
    )
    save_pairs(CONFIG.medqa_cache, pairs)
    return pairs


def make_pubmed_prompt(question: str, contexts: List[str]) -> str:
    """Format PubMedQA prompt with context truncation."""
    ctx = "\n".join(f"- {c}" for c in contexts[:CONFIG.max_contexts] if c)
    if len(ctx) > CONFIG.max_context_chars:
        ctx = ctx[:CONFIG.max_context_chars]
    return (
        "You are answering a biomedical yes/no question.\n"
        f"Question: {question}\n"
        f"Context:\n{ctx}\n\n"
        "Answer with one word (yes or no).\n"
        "Answer:"
    )


def get_pubmedqa_pairs(tokenizer, n_total: int) -> List[Dict[str, Any]]:
    """Load or build PubMedQA adversarial pairs."""
    if os.path.exists(CONFIG.pubmedqa_cache):
        print(f"Loading PubMedQA from cache: {CONFIG.pubmedqa_cache}")
        return load_pairs(CONFIG.pubmedqa_cache)[:n_total]

    print(f"Building PubMedQA pairs (n={n_total})...")
    yes_id = get_single_token_id(tokenizer, "yes")
    no_id = get_single_token_id(tokenizer, "no")
    ds = load_dataset("pubmed_qa", "pqa_labeled", split="train")

    pairs = []
    for row in ds:
        gold = str(row.get("final_decision", "")).strip().lower()
        if gold not in {"yes", "no"}:
            continue
            
        question = str(row.get("question", "")).strip()
        ctx_obj = row.get("context", {})
        contexts = ctx_obj.get("contexts", []) if isinstance(ctx_obj, dict) else []
        
        if not question:
            continue

        enc = tokenizer(
            make_pubmed_prompt(question, contexts), 
            return_tensors="pt"
        )
        pos_id = yes_id if gold == "yes" else no_id
        neg_id = no_id if gold == "yes" else yes_id

        pairs.append({
            "input_ids": enc["input_ids"],
            "attention_mask": enc["attention_mask"],
            "safe_input_ids": enc["input_ids"].clone(),
            "safe_attention_mask": enc["attention_mask"].clone(),
            "pos_id": int(pos_id),
            "neg_id": int(neg_id),
        })
        
        if len(pairs) >= n_total:
            break

    if not pairs:
        raise RuntimeError("No PubMedQA pairs built.")
    
    save_pairs(CONFIG.pubmedqa_cache, pairs)
    return pairs


def make_medmcqa_prompt(row: Dict[str, Any]) -> str:
    """Format MedMCQA prompt."""
    return (
        f"Question: {str(row.get('question', '')).strip()}\n"
        f"Options: (A) {str(row.get('opa', '')).strip()} "
        f"(B) {str(row.get('opb', '')).strip()} "
        f"(C) {str(row.get('opc', '')).strip()} "
        f"(D) {str(row.get('opd', '')).strip()}\n"
        "Answer: ("
    )


def get_medmcqa_pairs(model, tokenizer, n_total: int) -> List[Dict[str, Any]]:
    """Load or build MedMCQA adversarial pairs."""
    if os.path.exists(CONFIG.medmcqa_cache):
        print(f"Loading MedMCQA from cache: {CONFIG.medmcqa_cache}")
        return load_pairs(CONFIG.medmcqa_cache)[:n_total]

    print(f"Building MedMCQA pairs (n={n_total})...")
    ds = load_dataset("openlifescienceai/medmcqa", split="train")
    letter_ids = get_letter_token_ids(tokenizer)
    idx_to_letter = {0: "A", 1: "B", 2: "C", 3: "D"}
    device = next(model.parameters()).device

    pairs = []
    for row in ds:
        try:
            cop = int(row.get("cop", -1))
        except (TypeError, ValueError):
            continue
            
        if cop not in idx_to_letter:
            continue

        enc = tokenizer(make_medmcqa_prompt(row), return_tensors="pt")
        
        with torch.no_grad():
            logits = model(
                input_ids=enc["input_ids"].to(device),
                attention_mask=enc["attention_mask"].to(device),
            ).logits[0, -1, :].float()

        correct = idx_to_letter[cop]
        pos_id = int(letter_ids[correct])
        wrong_ids = [int(letter_ids[l]) for l in ["A", "B", "C", "D"] if l != correct]
        wrong_logits = logits[torch.tensor(wrong_ids, device=logits.device)]
        neg_id = int(wrong_ids[int(torch.argmax(wrong_logits).item())])

        pairs.append({
            "input_ids": enc["input_ids"],
            "attention_mask": enc["attention_mask"],
            "safe_input_ids": enc["input_ids"].clone(),
            "safe_attention_mask": enc["attention_mask"].clone(),
            "pos_id": pos_id,
            "neg_id": neg_id,
        })
        
        if len(pairs) >= n_total:
            break

    if not pairs:
        raise RuntimeError("No MedMCQA pairs built.")
    
    save_pairs(CONFIG.medmcqa_cache, pairs)
    return pairs


# ============================================================================
# ABLATION METHOD
# ============================================================================

ABLATION_METHOD = "crush"  # Zero out weights (only method for EMS)





# ============================================================================
# CORE METRICS
# ============================================================================

def mean_drop_for_neuron(
    probe: MedNSQProbe,
    pairs: List[Dict[str, Any]],
    baseline: torch.Tensor,
    layer: int,
    col: int,
) -> float:
    """
    Compute mean margin drop for a single neuron (crush ablation).
    
    Args:
        probe: MedNSQProbe instance
        pairs: Adversarial pairs
        baseline: Pre-computed baseline margins
        layer: Layer index
        col: Column index
        
    Returns:
        mean_drop: Average margin drop (baseline - ablated)
    """
    # Apply crush ablation
    orig = probe.simulate_column_crush(layer, col)
    
    try:
        ablated = probe.compute_per_sample_margins(pairs)
    finally:
        probe.restore_column(layer, col, orig)
    
    if baseline.numel() == 0:
        return 0.0
    
    drops = baseline - ablated
    return float(drops.mean().item())


def mean_drop_for_set(
    probe: MedNSQProbe,
    pairs: List[Dict[str, Any]],
    neurons: List[Tuple[int, int]],
    baseline: Optional[torch.Tensor] = None,
) -> Tuple[float, float, List[float]]:
    """
    Compute mean margin drop for a set of neurons (crush ablation).
    
    Args:
        probe: MedNSQProbe instance
        pairs: Adversarial pairs
        neurons: List of (layer, column) tuples
        baseline: Pre-computed baseline margins
        
    Returns:
        mean_drop: Average margin drop
        std_drop: Standard deviation of drops
        drops: Per-sample drops (for later analysis)
    """
    if baseline is None:
        baseline = probe.compute_per_sample_margins(pairs)
    originals = []
    
    try:
        for l, c in neurons:
            orig = probe.simulate_column_crush(l, c)
            originals.append((l, c, orig))
        
        ablated = probe.compute_per_sample_margins(pairs)
        drops = (baseline - ablated).cpu().numpy()
        
        return float(drops.mean()), float(drops.std()), drops.tolist()
    
    finally:
        for l, c, orig in originals:
            probe.restore_column(l, c, orig)


# ============================================================================
# CLUSTERING
# ============================================================================

def perform_kmeans(
    features: torch.Tensor,
    k: int,
    seed: int = 42,
    max_iters: int = 100
) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]:
    """
    Perform k-means clustering with quality metrics.
    
    Args:
        features: [n_samples, n_dims] tensor
        k: Number of clusters
        seed: Random seed
        max_iters: Maximum iterations
        
    Returns:
        assignments: Cluster assignments
        centers: Cluster centers
        metrics: Quality metrics (silhouette, inertia)
    """
    n, d = features.shape
    k = max(2, min(k, n))  # Ensure at least 2 clusters
    
    # Initialize with k-means++
    rng = np.random.RandomState(seed)
    n_samples = features.shape[0]
    
    # Convert to numpy for scikit-learn compatibility
    X = features.numpy()
    
    # Simple k-means implementation with metrics
    from sklearn.cluster import KMeans
    
    kmeans = KMeans(
        n_clusters=k,
        random_state=seed,
        n_init=10,
        max_iter=max_iters
    )
    
    assignments = kmeans.fit_predict(X)
    centers = torch.tensor(kmeans.cluster_centers_)
    
    # Compute silhouette score if enough samples
    silhouette = silhouette_score(X, assignments) if n > k else 0.0
    
    metrics = {
        "inertia": float(kmeans.inertia_),
        "silhouette_score": float(silhouette),
        "n_iter": int(kmeans.n_iter_)
    }
    
    return torch.tensor(assignments), centers, metrics


# ============================================================================
# STATISTICAL TESTS
# ============================================================================

def compare_anchor_vs_random(
    anchor_drops: List[float],
    random_drops: List[float],
    name: str
) -> Dict[str, Any]:
    """
    Statistical comparison between anchor and random neuron drops.
    
    Returns:
        Dictionary with statistics and p-values
    """
    anchor_mean = np.mean(anchor_drops)
    anchor_std = np.std(anchor_drops, ddof=1)
    random_mean = np.mean(random_drops)
    random_std = np.std(random_drops, ddof=1)
    
    # Welch's t-test (unequal variance)
    t_stat, p_value = stats.ttest_ind(
        anchor_drops, random_drops, 
        equal_var=False, alternative='greater'
    )
    
    # Effect size (Cohen's d)
    pooled_std = np.sqrt((anchor_std**2 + random_std**2) / 2)
    cohens_d = (anchor_mean - random_mean) / pooled_std if pooled_std > 0 else 0
    
    return {
        "dataset": name,
        "anchor_mean": float(anchor_mean),
        "anchor_std": float(anchor_std),
        "random_mean": float(random_mean),
        "random_std": float(random_std),
        "difference": float(anchor_mean - random_mean),
        "t_statistic": float(t_stat),
        "p_value": float(p_value),
        "cohens_d": float(cohens_d),
        "n_anchor": len(anchor_drops),
        "n_random": len(random_drops)
    }


# ============================================================================
# MAIN
# ============================================================================

def main():
    """Main execution with full reproducibility."""
    
    # Set all seeds
    random.seed(CONFIG.random_seed)
    np.random.seed(CONFIG.random_seed)
    torch.manual_seed(CONFIG.random_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(CONFIG.random_seed)
    
    print("=" * 60)
    print("ALL-IN-ONE ANCHOR EVALUATION (AFM VERSION)")
    print("=" * 60)
    print(f"Config hash: {CONFIG.config_hash}")
    print(f"Model: {CONFIG.model_id}")
    print(f"Top anchors: {CONFIG.k_top}")
    print(f"Random trials: {CONFIG.n_random_trials}")
    print("=" * 60)
    
    # ------------------------------------------------------------------------
    # Load model and probe
    # ------------------------------------------------------------------------
    print("\n[1/6] Loading model...")
    tokenizer = AutoTokenizer.from_pretrained(CONFIG.model_id, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        CONFIG.model_id,
        dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    probe = MedNSQProbe(model)
    
    # ------------------------------------------------------------------------
    # Load datasets
    # ------------------------------------------------------------------------
    print("\n[2/6] Preparing dataset pairs...")
    medqa = get_medqa_pairs(model, tokenizer, CONFIG.n_medqa)
    medmcqa = get_medmcqa_pairs(model, tokenizer, CONFIG.n_medmcqa)
    pubmedqa = get_pubmedqa_pairs(tokenizer, CONFIG.n_pubmedqa)
    
    datasets = {
        "medqa": medqa,
        "medmcqa": medmcqa,
        "pubmedqa": pubmedqa
    }
    
    print(f"  MedQA: {len(medqa)} pairs")
    print(f"  MedMCQA: {len(medmcqa)} pairs")
    print(f"  PubMedQA: {len(pubmedqa)} pairs")
    
    # ------------------------------------------------------------------------
    # Load anchors
    # ------------------------------------------------------------------------
    print("\n[3/6] Loading anchors...")
    # Use hardcoded AFM anchors instead of loading from file
    anchors = [{"layer": l, "column": c, "drop": 1.0} for l, c in AFM_ANCHORS[:CONFIG.k_top]]
    # Assume they are already sorted by importance
    anchor_neurons = [(int(a["layer"]), int(a["column"])) for a in anchors]
    print(f"  Loaded {len(anchor_neurons)} anchors from hardcoded AFM_ANCHORS")
    
    # ------------------------------------------------------------------------
    # Anchor vs random comparison
    # ------------------------------------------------------------------------
    print("\n[4/6] Running anchor vs random comparison...")

    print("  Precomputing dataset baselines...")
    baseline = {k: probe.compute_per_sample_margins(v) for k, v in datasets.items()}
    
    # Get all possible neurons for random sampling
    dims = probe.get_model_dims()
    n_layers = int(dims["num_layers"])
    n_cols = int(dims["intermediate_size"])
    anchor_set = set(anchor_neurons)
    candidate_neurons = [
        (l, c)
        for l in range(n_layers)
        for c in range(n_cols)
        if (l, c) not in anchor_set
    ]
    
    # Generate random trials from non-anchor pool (single seed, no sampling loop).
    random_trial_rng = random.Random(CONFIG.random_seed)
    random_trials = [random_trial_rng.sample(candidate_neurons, len(anchor_neurons))]
    
    # Compute drops (crush ablation only)
    anchor_vs_random = {}
    
    for name, pairs in datasets.items():
        print(f"  Processing {name}...")
        
        # Anchor drops
        anchor_mean, anchor_std, anchor_drops = mean_drop_for_set(
            probe, pairs, anchor_neurons, baseline=baseline[name]
        )
        
        # Random drops (single trial)
        random_drops_list = []
        for i, trial in enumerate(random_trials):
            if (i + 1) % 10 == 0:
                print(f"    Random trial {i+1}/{CONFIG.n_random_trials}")
            mean_drop, _, _ = mean_drop_for_set(
                probe,
                pairs,
                trial,
                baseline=baseline[name],
            )
            random_drops_list.append(mean_drop)
        
        # Statistical comparison
        stats_result = compare_anchor_vs_random(
            [anchor_mean] * CONFIG.n_random_trials,  # Broadcast for t-test
            random_drops_list,
            name
        )
        
        anchor_vs_random[name] = {
            "anchor_mean": anchor_mean,
            "anchor_std": anchor_std,
            "random_mean": np.mean(random_drops_list),
            "random_std": np.std(random_drops_list, ddof=1),
            "random_drops": random_drops_list,
            "difference": anchor_mean - np.mean(random_drops_list),
            "t_statistic": stats_result["t_statistic"],
            "p_value": stats_result["p_value"],
            "cohens_d": stats_result["cohens_d"],
            "n_random_trials": CONFIG.n_random_trials
        }
        
        sig = "***" if stats_result["p_value"] < 0.001 else "**" if stats_result["p_value"] < 0.01 else "*" if stats_result["p_value"] < 0.05 else "ns"
        print(f"    {name}: anchor={anchor_mean:.4f} ± {anchor_std:.4f}, "
              f"random={anchor_vs_random[name]['random_mean']:.4f} ± {anchor_vs_random[name]['random_std']:.4f}, "
              f"diff={anchor_vs_random[name]['difference']:.4f}, p={stats_result['p_value']:.4f} {sig}")
    
    # ------------------------------------------------------------------------
    # Per-anchor cross-dataset effects (crush ablation)
    # ------------------------------------------------------------------------
    print("\n[5/6] Computing per-anchor cross-dataset effects...")
    
    rows = []
    for i, (l, c) in enumerate(anchor_neurons, start=1):
        row_data = {"layer": l, "column": c}
        
        # Compute drops across datasets
        d1 = mean_drop_for_neuron(probe, medqa, baseline["medqa"], l, c)
        d2 = mean_drop_for_neuron(probe, medmcqa, baseline["medmcqa"], l, c)
        d3 = mean_drop_for_neuron(probe, pubmedqa, baseline["pubmedqa"], l, c)
        
        row_data["drop_medqa"] = d1
        row_data["drop_medmcqa"] = d2
        row_data["drop_pubmedqa"] = d3
        
        rows.append(row_data)
        
        print(f"  [{i:03d}/{len(anchor_neurons)}] L{l} C{c}: mqa={d1:+.4f}, mmcqa={d2:+.4f}, pmqa={d3:+.4f}")
    
    # Clustering
    print(f"\n[6/6] Performing cluster analysis...")
    
    feat = torch.tensor([
        [r["drop_medqa"], r["drop_medmcqa"], r["drop_pubmedqa"]] 
        for r in rows
    ], dtype=torch.float32)
    
    # Z-score normalization (focus on pattern, not magnitude)
    mu = feat.mean(dim=0)
    sd = feat.std(dim=0, unbiased=False)
    sd = torch.where(sd < 1e-8, torch.ones(3), sd)
    z = (feat - mu) / sd
    
    # K-means clustering with quality metrics
    assignments, centers, cluster_metrics = perform_kmeans(
        z, 
        k=CONFIG.k_clusters,
        seed=CONFIG.random_seed
    )
    
    # Summarize clusters
    cluster_summary = {}
    for cid in range(int(assignments.max().item()) + 1):
        mask = assignments == cid
        if not mask.any():
            continue
        
        idx = torch.where(mask)[0]
        raw_center = feat[idx].mean(dim=0)
        
        cluster_summary[int(cid)] = {
            "count": int(idx.numel()),
            "mean_drop_medqa": float(raw_center[0].item()),
            "mean_drop_medmcqa": float(raw_center[1].item()),
            "mean_drop_pubmedqa": float(raw_center[2].item()),
            "std_drop_medqa": float(feat[idx, 0].std().item()),
            "std_drop_medmcqa": float(feat[idx, 1].std().item()),
            "std_drop_pubmedqa": float(feat[idx, 2].std().item()),
        }
    
    print("\n  Cluster Summary:")
    for cid, summary in sorted(cluster_summary.items()):
        print(f"    Cluster {cid} (n={summary['count']}): "
              f"medqa={summary['mean_drop_medqa']:+.4f} ± {summary['std_drop_medqa']:.4f}, "
              f"medmcqa={summary['mean_drop_medmcqa']:+.4f} ± {summary['std_drop_medmcqa']:.4f}, "
              f"pubmedqa={summary['mean_drop_pubmedqa']:+.4f} ± {summary['std_drop_pubmedqa']:.4f}")
    
    print(f"\n  Cluster quality: silhouette={cluster_metrics['silhouette_score']:.4f}, "
          f"inertia={cluster_metrics['inertia']:.4f}")
    
    # Add cluster assignments to rows
    for i, r in enumerate(rows):
        r["cluster"] = int(assignments[i].item())
    
    # Save results
    # ------------------------------------------------------------------------
    print(f"\nSaving results to {CONFIG.output_file}...")
    
    out = {
        "metadata": {
            "config": CONFIG.config_dict,
            "config_hash": CONFIG.config_hash,
            "timestamp": datetime.now().isoformat(),
            "random_seed": CONFIG.random_seed,
            "torch_version": torch.__version__,
            "ablation_method": ABLATION_METHOD,
        },
        "dataset_sizes": {k: len(v) for k, v in datasets.items()},
        "anchor_vs_random": anchor_vs_random,
        "cluster_metrics": cluster_metrics,
        "cluster_summary": cluster_summary,
        "anchors": rows,
    }
    
    with open(CONFIG.output_file, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    
    print(f"\n✅ Done! Results saved to {CONFIG.output_file}")
    print("=" * 60)


if __name__ == "__main__":
    main()