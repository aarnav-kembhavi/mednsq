"""
AlphaMed-8B-Instruct-RL anchor-vs-random evaluation (MedNSQ-style).

Evaluates discovered anchors against random neuron groups on:
- MedQA (4-way multiple choice)
- MedMCQA (4-way multiple choice)
- PubMedQA (binary yes/no)

Outputs paper-style anchor vs random statistics (Welch t-test, Cohen's d) and
per-anchor cross-dataset margin drops. Loads model locally only (no Hub download).
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
    model_path: str = "/workspace/alphamed"
    anchor_file: str = "anchors_alphamed_8b_instruct_rl.json"

    n_medqa: int = 400
    n_medmcqa: int = 400
    n_pubmedqa: int = 400

    # If set, only the first `anchor_limit` anchors from JSON (order preserved).
    anchor_limit: Optional[int] = None

    n_random_trials: int = 500
    random_seed: int = 42

    medqa_cache: str = "alphamed_medqa_pairs.txt"
    medmcqa_cache: str = "alphamed_medmcqa_pairs.txt"
    pubmedqa_cache: str = "alphamed_pubmedqa_pairs.txt"

    output_file: str = "alphamed_anchor_results.json"

    max_contexts: int = 3
    max_context_chars: int = 2200

    def __post_init__(self):
        assert self.n_random_trials >= 2, "Need at least 2 random trials for variance"
        if self.anchor_limit is not None:
            assert self.anchor_limit > 0, "anchor_limit must be positive when set"

    @property
    def config_dict(self) -> Dict:
        return asdict(self)

    @property
    def config_hash(self) -> str:
        return hashlib.sha256(
            json.dumps(self.config_dict, sort_keys=True, default=str).encode()
        ).hexdigest()[:8]


CONFIG = Config()


# ============================================================================
# TOKEN UTILITIES
# ============================================================================

def get_single_token_id(tokenizer, text: str) -> int:
    """Robust token ID resolution for single tokens."""
    for candidate in [f" {text}", text]:
        ids = tokenizer.encode(candidate, add_special_tokens=False)
        if len(ids) == 1:
            return int(ids[0])

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
    """Save adversarial pairs to cache with metadata."""
    with open(path, "w", encoding="utf-8") as f:
        metadata = {
            "version": "1.0",
            "n_pairs": len(pairs),
            "created": datetime.now().isoformat(),
            "config_hash": CONFIG.config_hash,
        }
        f.write("#" + json.dumps(metadata) + "\n")

        for p in pairs:
            safe_ids = p.get("safe_input_ids", p["input_ids"])
            safe_mask = p.get("safe_attention_mask", p["attention_mask"])
            row = {
                "input_ids": p["input_ids"][0].tolist(),
                "attention_mask": p["attention_mask"][0].tolist(),
                "safe_input_ids": safe_ids[0].tolist(),
                "safe_attention_mask": safe_mask[0].tolist(),
                "pos_id": int(p["pos_id"]),
                "neg_id": int(p["neg_id"]),
            }
            f.write(json.dumps(row) + "\n")


def load_pairs(path: str) -> List[Dict[str, Any]]:
    """Load adversarial pairs from cache, skipping metadata."""
    pairs: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            row = json.loads(line)
            sid = row.get("safe_input_ids", row["input_ids"])
            sm = row.get("safe_attention_mask", row["attention_mask"])
            pairs.append({
                "input_ids": torch.tensor([row["input_ids"]], dtype=torch.long),
                "attention_mask": torch.tensor([row["attention_mask"]], dtype=torch.long),
                "safe_input_ids": torch.tensor([sid], dtype=torch.long),
                "safe_attention_mask": torch.tensor([sm], dtype=torch.long),
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
        n_calib=len(ds),
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
            return_tensors="pt",
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

ABLATION_METHOD = "crush"


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
    """Mean margin drop for a single neuron (column crush)."""
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
    """Mean/std margin drop for a set of neurons (simultaneous crush)."""
    if baseline is None:
        baseline = probe.compute_per_sample_margins(pairs)
    originals = []

    try:
        for l, c in neurons:
            orig = probe.simulate_column_crush(l, c)
            originals.append((l, c, orig))

        ablated = probe.compute_per_sample_margins(pairs)
        drops = (baseline - ablated).cpu().numpy()

        return float(drops.mean()), float(drops.std(ddof=1)), drops.tolist()

    finally:
        for l, c, orig in originals:
            probe.restore_column(l, c, orig)


def bootstrap_mean_drops(
    per_sample_drops: np.ndarray,
    n_trials: int,
    seed: int,
) -> np.ndarray:
    """Bootstrap sample means of per-sample drops (one scalar per trial)."""
    rng = np.random.default_rng(seed)
    n = len(per_sample_drops)
    if n == 0:
        return np.zeros(n_trials)
    out = np.empty(n_trials, dtype=np.float64)
    for i in range(n_trials):
        idx = rng.integers(0, n, size=n)
        out[i] = float(per_sample_drops[idx].mean())
    return out


def compare_two_samples_welch(
    sample_a: np.ndarray,
    sample_b: np.ndarray,
    alternative: str = "greater",
) -> Dict[str, Any]:
    """Welch t-test and Cohen's d between two independent samples."""
    m1 = float(np.mean(sample_a))
    s1 = float(np.std(sample_a, ddof=1))
    m2 = float(np.mean(sample_b))
    s2 = float(np.std(sample_b, ddof=1))

    t_stat, p_value = stats.ttest_ind(
        sample_a, sample_b, equal_var=False, alternative=alternative
    )

    pooled_std = np.sqrt((s1 ** 2 + s2 ** 2) / 2)
    cohens_d = (m1 - m2) / pooled_std if pooled_std > 0 else 0.0

    return {
        "mean_a": m1,
        "std_a": s1,
        "mean_b": m2,
        "std_b": s2,
        "t_statistic": float(t_stat),
        "p_value": float(p_value),
        "cohens_d": float(cohens_d),
        "n_a": int(len(sample_a)),
        "n_b": int(len(sample_b)),
    }


def format_p_value(p: float) -> str:
    if p < 1e-300:
        return "p<1e-300"
    if p < 0.001:
        return "p<0.001"
    return f"p={p:.4g}"


# ============================================================================
# MAIN
# ============================================================================

def load_anchor_neurons(path: str) -> List[Tuple[int, int]]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    anchors = data["anchors"]
    neurons = [(int(a["layer"]), int(a["column"])) for a in anchors]
    if CONFIG.anchor_limit is not None:
        neurons = neurons[: CONFIG.anchor_limit]
    return neurons


def main():
    random.seed(CONFIG.random_seed)
    np.random.seed(CONFIG.random_seed)
    torch.manual_seed(CONFIG.random_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(CONFIG.random_seed)

    model_path = CONFIG.model_path

    print("=" * 60)
    print("ALPHAMED-8B ANCHOR VS RANDOM (MedNSQ)")
    print("=" * 60)
    print(f"Config hash: {CONFIG.config_hash}")
    print(f"Model path: {model_path}")
    print(f"Anchor file: {CONFIG.anchor_file}")
    print(f"Random trials: {CONFIG.n_random_trials}")
    print("=" * 60)

    print("\n[1/4] Loading model (local only)...")
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True,
        use_fast=True,
        local_files_only=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
        local_files_only=True,
    )
    model.eval()
    probe = MedNSQProbe(model)

    print("\n[2/4] Preparing dataset pairs...")
    medqa = get_medqa_pairs(model, tokenizer, CONFIG.n_medqa)
    medmcqa = get_medmcqa_pairs(model, tokenizer, CONFIG.n_medmcqa)
    pubmedqa = get_pubmedqa_pairs(tokenizer, CONFIG.n_pubmedqa)

    datasets = {
        "medqa": medqa,
        "medmcqa": medmcqa,
        "pubmedqa": pubmedqa,
    }

    print(f"  MedQA: {len(medqa)} pairs")
    print(f"  MedMCQA: {len(medmcqa)} pairs")
    print(f"  PubMedQA: {len(pubmedqa)} pairs")

    print("\n[3/4] Loading anchors from JSON...")
    if not os.path.isfile(CONFIG.anchor_file):
        raise FileNotFoundError(
            f"Anchor file not found: {CONFIG.anchor_file} "
            "(expected keys: anchors[].layer, anchors[].column)"
        )
    anchor_neurons = load_anchor_neurons(CONFIG.anchor_file)
    if not anchor_neurons:
        raise RuntimeError("No anchors loaded from JSON.")
    print(f"  Using {len(anchor_neurons)} anchors (JSON order preserved)")

    n_layers = len(probe.layers)
    n_cols = probe.intermediate_size
    anchor_set = set(anchor_neurons)
    candidate_neurons = [
        (l, c)
        for l in range(n_layers)
        for c in range(n_cols)
        if (l, c) not in anchor_set
    ]
    k_group = len(anchor_neurons)
    if len(candidate_neurons) < k_group:
        raise RuntimeError(
            f"Not enough non-anchor neurons: need {k_group}, have {len(candidate_neurons)}"
        )

    print("\n[4/4] Anchor vs random (group crush)...")
    baseline = {name: probe.compute_per_sample_margins(pairs) for name, pairs in datasets.items()}

    random_trial_rng = random.Random(CONFIG.random_seed + 911)

    anchor_vs_random: Dict[str, Any] = {}
    rows: List[Dict[str, Any]] = []

    for name, pairs in datasets.items():
        print(f"  {name}...")

        anchor_mean, anchor_std, anchor_drop_list = mean_drop_for_set(
            probe, pairs, anchor_neurons, baseline=baseline[name]
        )
        anchor_per_sample = np.asarray(anchor_drop_list, dtype=np.float64)

        boot_seed = CONFIG.random_seed + {"medqa": 11, "medmcqa": 17, "pubmedqa": 23}[name]
        anchor_bootstrap_means = bootstrap_mean_drops(
            anchor_per_sample,
            n_trials=CONFIG.n_random_trials,
            seed=boot_seed,
        )

        random_trial_means: List[float] = []
        for trial_idx in range(CONFIG.n_random_trials):
            if (trial_idx + 1) % 100 == 0:
                print(f"    random trials {trial_idx + 1}/{CONFIG.n_random_trials}")
            trial = random_trial_rng.sample(candidate_neurons, k_group)
            mean_drop, _, _ = mean_drop_for_set(
                probe,
                pairs,
                trial,
                baseline=baseline[name],
            )
            random_trial_means.append(mean_drop)

        random_arr = np.asarray(random_trial_means, dtype=np.float64)
        stat = compare_two_samples_welch(
            anchor_bootstrap_means,
            random_arr,
            alternative="greater",
        )

        random_mean = float(np.mean(random_arr))
        random_std = float(np.std(random_arr, ddof=1))

        anchor_vs_random[name] = {
            "anchor_mean": anchor_mean,
            "anchor_std": anchor_std,
            "anchor_bootstrap_mean_of_means": stat["mean_a"],
            "anchor_bootstrap_std": stat["std_a"],
            "random_mean": random_mean,
            "random_std": random_std,
            "random_trial_means": random_trial_means,
            "t_statistic": stat["t_statistic"],
            "p_value": stat["p_value"],
            "cohens_d": stat["cohens_d"],
            "n_random_trials": CONFIG.n_random_trials,
            "welch_compares": "bootstrap_anchor_means_vs_random_trial_means",
        }

        p_str = format_p_value(stat["p_value"])
        print(
            f"{name}: anchor={anchor_mean:+.4f} ± {anchor_std:.4f}, "
            f"random={random_mean:+.4f} ± {random_std:.4f}, "
            f"d={stat['cohens_d']:+.2f}, {p_str}"
        )

    print("\nPer-anchor cross-dataset margin drops...")
    for i, (l, c) in enumerate(anchor_neurons, start=1):
        row_data = {"layer": l, "column": c}
        d1 = mean_drop_for_neuron(probe, medqa, baseline["medqa"], l, c)
        d2 = mean_drop_for_neuron(probe, medmcqa, baseline["medmcqa"], l, c)
        d3 = mean_drop_for_neuron(probe, pubmedqa, baseline["pubmedqa"], l, c)
        row_data["drop_medqa"] = d1
        row_data["drop_medmcqa"] = d2
        row_data["drop_pubmedqa"] = d3
        rows.append(row_data)
        print(f"  [{i:03d}/{len(anchor_neurons)}] L{l} C{c}: mqa={d1:+.4f}, mmcqa={d2:+.4f}, pmqa={d3:+.4f}")

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
        "anchors": rows,
    }

    with open(CONFIG.output_file, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)

    print(f"\nDone. Results saved to {CONFIG.output_file}")
    print("=" * 60)


if __name__ == "__main__":
    main()
