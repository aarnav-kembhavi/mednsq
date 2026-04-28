"""
MED42 cross-dataset anchor profiling.

For each anchor neuron, measures margin drop across:
  - MedQA (4-option MCQ)
  - MedMCQA (4-option MCQ)
  - PubMedQA (binary yes/no, native format)

Clusters anchors by their cross-dataset effect profile.

No anchor-vs-random comparison, no group ablation sweep.
"""

import argparse
import gc
import hashlib
import json
import os
import random
from dataclasses import dataclass, asdict
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from datasets import load_dataset
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from transformers import AutoModelForCausalLM, AutoTokenizer

from mednsq_data import build_adversarial_pairs, load_mcq_dataset
from mednsq_probe import MedNSQProbe

import torch._dynamo
torch._dynamo.config.suppress_errors = True
# =====================================================================
# MODEL + ANCHORS
# =====================================================================
# Med42 anchors from validated discovery pipeline (ablation_med42_8b.json,
# z=259 at K=32 on MedQA). Top 64, sorted by drop_val descending.
MEDITRON_ANCHORS: List[Tuple[int, int]] = [
    (17,3451), (14,18336), (12,8109), (20,13290), (18,11555), (19,18147),
    (17,16475), (14,9924), (13,14316), (19,8750), (24,12896), (17,7652),
    (15,485), (19,9166), (23,5298), (22,9786), (24,10538), (17,4105),
    (11,3558), (19,8257), (20,9629), (21,2951), (15,10749), (17,1105),
    (11,10155), (22,1969), (20,18853), (24,16669), (19,10109), (12,11452),
    (15,3642), (15,9396), (14,8180), (23,12030), (12,6127), (15,10782),
    (15,16588), (16,4570), (24,13642), (15,10943), (15,3905), (13,14562),
    (13,1151), (15,1639)
]

MED42_ANCHORS = MEDITRON_ANCHORS  # Keep variable name for compatibility

MODEL_ID = "Qwen/Qwen2.5-8B"
MODEL_KEY: str = "qwen25_8b"


@dataclass
class EvalConfig:
    n_medqa: int = 400
    n_medmcqa: int = 400
    n_pubmedqa: int = 400
    random_seed: int = 42
    # Cluster K is auto-selected via silhouette in [2..k_clusters_max]
    k_clusters_max: int = 6
    output_dir: str = "results"
    cache_dir: str = "."
    max_contexts: int = 3
    max_context_chars: int = 2200


# =====================================================================
# TOKEN UTILITIES
# =====================================================================
def get_single_token_id(tokenizer, text: str) -> int:
    """Resolve a single-token id, trying both space-prefixed and bare."""
    for candidate in [f" {text}", text]:
        ids = tokenizer.encode(candidate, add_special_tokens=False)
        if len(ids) == 1:
            return int(ids[0])
    ids = tokenizer.encode(f" {text}", add_special_tokens=False)
    if not ids:
        raise RuntimeError(f"Cannot resolve token id for: '{text}'")
    return int(ids[0])


def get_letter_token_ids(tokenizer) -> Dict[str, int]:
    return {l: get_single_token_id(tokenizer, l) for l in ["A", "B", "C", "D"]}


def get_yes_no_token_ids(tokenizer) -> Tuple[int, int]:
    """Token ids for ' yes' and ' no' (PubMedQA native format)."""
    return get_single_token_id(tokenizer, "yes"), get_single_token_id(tokenizer, "no")


# =====================================================================
# PAIR CACHING
# =====================================================================
def _prompt_hash(*strs: str) -> str:
    h = hashlib.sha256()
    for s in strs:
        h.update(s.encode("utf-8"))
    return h.hexdigest()[:12]


def save_pairs(path: str, pairs: List[Dict[str, Any]], metadata: Dict[str, Any]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        f.write("#" + json.dumps(metadata) + "\n")
        for p in pairs:
            row = {
                "input_ids": p["input_ids"][0].tolist(),
                "attention_mask": p["attention_mask"][0].tolist(),
                "pos_id": int(p["pos_id"]),
                "neg_id": int(p["neg_id"]),
            }
            f.write(json.dumps(row) + "\n")


def load_pairs(path: str) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Returns (pairs, metadata)."""
    pairs: List[Dict[str, Any]] = []
    metadata: Dict[str, Any] = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith("#"):
                try:
                    metadata = json.loads(line[1:])
                except Exception:
                    metadata = {}
                continue
            row = json.loads(line)
            pairs.append({
                "input_ids": torch.tensor([row["input_ids"]], dtype=torch.long),
                "attention_mask": torch.tensor([row["attention_mask"]], dtype=torch.long),
                "pos_id": int(row["pos_id"]),
                "neg_id": int(row["neg_id"]),
            })
    return pairs, metadata


def move_pairs_to_device(pairs: List[Dict[str, Any]], device: torch.device) -> List[Dict[str, Any]]:
    for p in pairs:
        p["input_ids"] = p["input_ids"].to(device)
        p["attention_mask"] = p["attention_mask"].to(device)
    return pairs


# =====================================================================
# DATASET LOADERS (with baseline-correctness filtering)
# =====================================================================
MEDMCQA_PROMPT_TEMPLATE = (
    "Question: {q}\n"
    "Options: (A) {a} (B) {b} (C) {c} (D) {d}\n"
    "Answer: ("
)

PUBMED_BINARY_PROMPT_TEMPLATE = (
    "You are answering a biomedical research question with either yes or no.\n"
    "Question: {q}\n"
    "Context:\n{ctx}\n"
    "Answer:"
)


def get_medqa_pairs(model, tokenizer, cache_path: str, n_total: int) -> List[Dict[str, Any]]:
    """MedQA from mednsq_data.build_adversarial_pairs (already filters correct-at-baseline)."""
    if os.path.exists(cache_path):
        cached, meta = load_pairs(cache_path)
        if len(cached) >= n_total:
            print(f"Loading MedQA from cache: {cache_path} (n={len(cached)})")
            return cached[:n_total]

    print(f"Building MedQA pairs (target n={n_total})...")
    # Pull more raw samples than n_total since the filter drops some
    ds = load_mcq_dataset(n_total=int(n_total * 1.8))
    pairs = build_adversarial_pairs(model=model, tokenizer=tokenizer, dataset=ds, n_calib=len(ds))
    pairs = pairs[:n_total]
    save_pairs(cache_path, pairs, {
        "dataset": "medqa",
        "n_pairs": len(pairs),
        "created": datetime.now().isoformat(),
        "filter": "correct_at_baseline",
    })
    print(f"  Built {len(pairs)} MedQA pairs")
    return pairs


def get_medmcqa_pairs(model, tokenizer, cache_path: str, n_total: int) -> List[Dict[str, Any]]:
    """MedMCQA, filtered to correct-at-baseline."""
    prompt_hash = _prompt_hash(MEDMCQA_PROMPT_TEMPLATE)

    if os.path.exists(cache_path):
        cached, meta = load_pairs(cache_path)
        if (len(cached) >= n_total
                and meta.get("prompt_hash") == prompt_hash
                and meta.get("filter") == "correct_at_baseline"):
            print(f"Loading MedMCQA from cache: {cache_path} (n={len(cached)})")
            return cached[:n_total]
        else:
            print(f"MedMCQA cache invalid, rebuilding...")

    print(f"Building MedMCQA pairs (target n={n_total}, filtered)...")
    ds = load_dataset("openlifescienceai/medmcqa", split="train")
    letter_ids = get_letter_token_ids(tokenizer)
    letter_id_tensor_template = list(letter_ids.values())
    idx_to_letter = {0: "A", 1: "B", 2: "C", 3: "D"}
    device = next(model.parameters()).device

    pairs: List[Dict[str, Any]] = []
    n_seen = 0
    n_wrong_at_baseline = 0

    for row in ds:
        try:
            cop = int(row.get("cop", -1))
        except (TypeError, ValueError):
            continue
        if cop not in idx_to_letter:
            continue

        prompt = MEDMCQA_PROMPT_TEMPLATE.format(
            q=str(row.get("question", "")).strip(),
            a=str(row.get("opa", "")).strip(),
            b=str(row.get("opb", "")).strip(),
            c=str(row.get("opc", "")).strip(),
            d=str(row.get("opd", "")).strip(),
        )
        enc = tokenizer(prompt, return_tensors="pt")
        n_seen += 1

        with torch.no_grad():
            logits = model(
                input_ids=enc["input_ids"].to(device),
                attention_mask=enc["attention_mask"].to(device),
            ).logits[0, -1, :].float()

        # Baseline correctness check
        letter_logits = logits[torch.tensor(letter_id_tensor_template, device=logits.device)]
        pred_letter_idx = int(torch.argmax(letter_logits).item())
        pred_letter = ["A", "B", "C", "D"][pred_letter_idx]
        correct = idx_to_letter[cop]
        if pred_letter != correct:
            n_wrong_at_baseline += 1
            continue

        pos_id = int(letter_ids[correct])
        wrong_ids = [int(letter_ids[l]) for l in ["A", "B", "C", "D"] if l != correct]
        wrong_logits = logits[torch.tensor(wrong_ids, device=logits.device)]
        neg_id = int(wrong_ids[int(torch.argmax(wrong_logits).item())])

        pairs.append({
            "input_ids": enc["input_ids"],
            "attention_mask": enc["attention_mask"],
            "pos_id": pos_id,
            "neg_id": neg_id,
        })
        if len(pairs) >= n_total:
            break

    if not pairs:
        raise RuntimeError("No MedMCQA pairs built.")

    print(f"  MedMCQA: kept {len(pairs)}/{n_seen} (dropped {n_wrong_at_baseline} wrong-at-baseline, "
          f"frac={n_wrong_at_baseline/max(n_seen,1):.2f})")

    save_pairs(cache_path, pairs, {
        "dataset": "medmcqa",
        "n_pairs": len(pairs),
        "created": datetime.now().isoformat(),
        "filter": "correct_at_baseline",
        "prompt_hash": prompt_hash,
    })
    return pairs


def make_pubmed_binary_prompt(question: str, contexts: List[str], cfg: EvalConfig) -> str:
    ctx = "\n".join(f"- {c}" for c in contexts[: cfg.max_contexts] if c)
    if len(ctx) > cfg.max_context_chars:
        ctx = ctx[: cfg.max_context_chars]
    return PUBMED_BINARY_PROMPT_TEMPLATE.format(q=question, ctx=ctx)


def get_pubmedqa_pairs(model, tokenizer, cache_path: str, n_total: int, cfg: EvalConfig) -> List[Dict[str, Any]]:
    """PubMedQA in NATIVE binary format. pos_id/neg_id are ' yes' and ' no' tokens.
    Filtered to correct-at-baseline.
    """
    prompt_hash = _prompt_hash(PUBMED_BINARY_PROMPT_TEMPLATE)
    yes_id, no_id = get_yes_no_token_ids(tokenizer)

    if os.path.exists(cache_path):
        cached, meta = load_pairs(cache_path)
        cache_ok = (
            len(cached) >= n_total
            and meta.get("prompt_hash") == prompt_hash
            and meta.get("format") == "binary_yes_no"
            and meta.get("filter") == "correct_at_baseline"
        )
        if cache_ok:
            # Double-check sample tokens
            sample = cached[: min(8, len(cached))]
            tokens_ok = all(int(p["pos_id"]) in {yes_id, no_id}
                            and int(p["neg_id"]) in {yes_id, no_id}
                            and int(p["pos_id"]) != int(p["neg_id"])
                            for p in sample)
            if tokens_ok:
                print(f"Loading PubMedQA from cache: {cache_path} (n={len(cached)})")
                return cached[:n_total]
        print(f"PubMedQA cache invalid (format/hash/tokens mismatch), rebuilding...")

    print(f"Building PubMedQA pairs (binary yes/no, target n={n_total}, filtered)...")
    ds = load_dataset("pubmed_qa", "pqa_labeled", split="train")
    device = next(model.parameters()).device

    pairs: List[Dict[str, Any]] = []
    n_seen = 0
    n_wrong_at_baseline = 0

    for row in ds:
        gold = str(row.get("final_decision", "")).strip().lower()
        if gold not in {"yes", "no"}:
            continue
        question = str(row.get("question", "")).strip()
        if not question:
            continue

        ctx_obj = row.get("context", {})
        contexts = ctx_obj.get("contexts", []) if isinstance(ctx_obj, dict) else []
        prompt = make_pubmed_binary_prompt(question, contexts, cfg)
        enc = tokenizer(prompt, return_tensors="pt")
        n_seen += 1

        with torch.no_grad():
            logits = model(
                input_ids=enc["input_ids"].to(device),
                attention_mask=enc["attention_mask"].to(device),
            ).logits[0, -1, :].float()

        # Baseline correctness on yes/no
        yes_logit = logits[yes_id].item()
        no_logit = logits[no_id].item()
        pred = "yes" if yes_logit > no_logit else "no"
        if pred != gold:
            n_wrong_at_baseline += 1
            continue

        pos_id = yes_id if gold == "yes" else no_id
        neg_id = no_id if gold == "yes" else yes_id

        pairs.append({
            "input_ids": enc["input_ids"],
            "attention_mask": enc["attention_mask"],
            "pos_id": int(pos_id),
            "neg_id": int(neg_id),
        })
        if len(pairs) >= n_total:
            break

    if not pairs:
        raise RuntimeError("No PubMedQA pairs built.")

    print(f"  PubMedQA: kept {len(pairs)}/{n_seen} (dropped {n_wrong_at_baseline} wrong-at-baseline, "
          f"frac={n_wrong_at_baseline/max(n_seen,1):.2f})")

    save_pairs(cache_path, pairs, {
        "dataset": "pubmedqa",
        "format": "binary_yes_no",
        "n_pairs": len(pairs),
        "created": datetime.now().isoformat(),
        "filter": "correct_at_baseline",
        "prompt_hash": prompt_hash,
    })
    return pairs


# =====================================================================
# CORE METRICS
# =====================================================================
def mean_drop_for_neuron(probe: MedNSQProbe, pairs: List[Dict[str, Any]],
                         baseline: torch.Tensor, layer: int, col: int) -> float:
    """Single-neuron crush, return mean (baseline - ablated) margin drop."""
    orig = probe.simulate_column_crush(layer, col)
    try:
        ablated = probe.compute_per_sample_margins(pairs)
    finally:
        probe.restore_column(layer, col, orig)
    if baseline.numel() == 0:
        return 0.0
    return float((baseline - ablated).mean().item())


# =====================================================================
# CLUSTERING
# =====================================================================
def auto_kmeans(features: np.ndarray, k_max: int, seed: int = 42) -> Tuple[np.ndarray, int, Dict[str, Any]]:
    """K-means with k chosen by best silhouette in [2..k_max]."""
    n = features.shape[0]
    k_max = min(k_max, n - 1)

    best = None
    best_k = 2
    best_sil = -1.0
    best_metrics = {}

    for k in range(2, k_max + 1):
        km = KMeans(n_clusters=k, random_state=seed, n_init=10, max_iter=200)
        assign = km.fit_predict(features)
        if len(set(assign)) < 2:
            continue
        sil = silhouette_score(features, assign)
        if sil > best_sil:
            best_sil = sil
            best_k = k
            best = assign
            best_metrics = {
                "inertia": float(km.inertia_),
                "silhouette_score": float(sil),
                "n_iter": int(km.n_iter_),
                "k_chosen": k,
            }

    if best is None:
        # Fallback: k=2
        km = KMeans(n_clusters=2, random_state=seed, n_init=10)
        best = km.fit_predict(features)
        best_k = 2
        best_metrics = {"inertia": float(km.inertia_), "silhouette_score": 0.0,
                        "n_iter": int(km.n_iter_), "k_chosen": 2}

    return best, best_k, best_metrics


# =====================================================================
# MAIN EVAL
# =====================================================================
def evaluate(cfg: EvalConfig) -> Dict[str, Any]:
    print("=" * 72)
    print(f"MODEL: {MODEL_KEY} ({MODEL_ID})")
    print(f"Anchors: {len(MED42_ANCHORS)}")
    print("=" * 72)

    random.seed(cfg.random_seed)
    np.random.seed(cfg.random_seed)
    torch.manual_seed(cfg.random_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cfg.random_seed)

    print("\n[1/4] Loading model...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    # Qwen2.5 specific: ensure chat template doesn't interfere
    tokenizer.chat_template = None  # Use base template for our prompts
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.bfloat16,  # Changed from float16 to bfloat16 for better 8B support
        device_map="auto",
        trust_remote_code=True,
        low_cpu_mem_usage=True,  # Added
    )
    model.eval()
    probe = MedNSQProbe(model)

    n_layers = len(probe.layers)
    n_cols = int(probe.intermediate_size)
    print(f"  Architecture: layers={n_layers}, intermediate={n_cols}")

    os.makedirs(cfg.output_dir, exist_ok=True)
    medqa_cache = os.path.join(cfg.cache_dir, f"{MODEL_KEY}_medqa_pairs.txt")
    medmcqa_cache = os.path.join(cfg.cache_dir, f"{MODEL_KEY}_medmcqa_pairs.txt")
    pubmedqa_cache = os.path.join(cfg.cache_dir, f"{MODEL_KEY}_pubmedqa_binary_pairs.txt")

    print("\n[2/4] Building/loading dataset pairs...")
    medqa = get_medqa_pairs(model, tokenizer, medqa_cache, cfg.n_medqa)
    medmcqa = get_medmcqa_pairs(model, tokenizer, medmcqa_cache, cfg.n_medmcqa)
    pubmedqa = get_pubmedqa_pairs(model, tokenizer, pubmedqa_cache, cfg.n_pubmedqa, cfg)

    device = next(model.parameters()).device
    medqa = move_pairs_to_device(medqa, device)
    medmcqa = move_pairs_to_device(medmcqa, device)
    pubmedqa = move_pairs_to_device(pubmedqa, device)

    datasets = {"medqa": medqa, "medmcqa": medmcqa, "pubmedqa": pubmedqa}

    # Baseline margins per dataset
    print("\n[3/4] Computing baselines + per-anchor cross-dataset effects...")
    baseline = {}
    baseline_summary = {}
    for name, pairs in datasets.items():
        m = probe.compute_per_sample_margins(pairs)
        baseline[name] = m
        baseline_summary[name] = {
            "mean": float(m.mean().item()),
            "std": float(m.std().item()),
            "frac_neg": float((m < 0).float().mean().item()),
            "n": len(pairs),
        }
        print(f"  {name}: n={len(pairs)} mean_margin={m.mean().item():.3f} "
              f"std={m.std().item():.3f} frac_neg={baseline_summary[name]['frac_neg']:.2f}")

    # Filter anchors to valid (in-range) ones
    anchor_neurons = [(l, c) for (l, c) in MED42_ANCHORS
                      if 0 <= l < n_layers and 0 <= c < n_cols]
    if len(anchor_neurons) < len(MED42_ANCHORS):
        print(f"  Note: {len(MED42_ANCHORS) - len(anchor_neurons)} anchors out-of-range, dropped.")

    # Per-anchor drops on each dataset
    rows: List[Dict[str, Any]] = []
    for i, (l, c) in enumerate(anchor_neurons, start=1):
        d_mqa = mean_drop_for_neuron(probe, medqa, baseline["medqa"], l, c)
        d_mmcqa = mean_drop_for_neuron(probe, medmcqa, baseline["medmcqa"], l, c)
        d_pmqa = mean_drop_for_neuron(probe, pubmedqa, baseline["pubmedqa"], l, c)

        # Relative drops (drop / baseline_margin) for fair cross-dataset comparison
        rel_mqa = d_mqa / max(baseline_summary["medqa"]["mean"], 1e-6)
        rel_mmcqa = d_mmcqa / max(baseline_summary["medmcqa"]["mean"], 1e-6)
        rel_pmqa = d_pmqa / max(baseline_summary["pubmedqa"]["mean"], 1e-6)

        rows.append({
            "layer": l,
            "column": c,
            "drop_medqa": d_mqa,
            "drop_medmcqa": d_mmcqa,
            "drop_pubmedqa": d_pmqa,
            "rel_drop_medqa": rel_mqa,
            "rel_drop_medmcqa": rel_mmcqa,
            "rel_drop_pubmedqa": rel_pmqa,
        })
        print(f"  [{i:02d}/{len(anchor_neurons)}] L{l:02d} C{c:5d}: "
              f"mqa={d_mqa:+.4f} ({rel_mqa:+.3f}), "
              f"mmcqa={d_mmcqa:+.4f} ({rel_mmcqa:+.3f}), "
              f"pmqa={d_pmqa:+.4f} ({rel_pmqa:+.3f})")

    # =====================================================================
    # CLUSTERING on relative drops (auto-pick k by silhouette)
    # =====================================================================
    print("\n[4/4] Clustering anchors by cross-dataset effect profile...")

    # Use relative drops for clustering (handles different baseline scales)
    feat_relative = np.array([[r["rel_drop_medqa"],
                               r["rel_drop_medmcqa"],
                               r["rel_drop_pubmedqa"]] for r in rows], dtype=np.float32)

    # Z-score normalize each axis (focus on pattern, not magnitude)
    mu = feat_relative.mean(axis=0)
    sd = feat_relative.std(axis=0)
    sd = np.where(sd < 1e-8, 1.0, sd)
    feat_z = (feat_relative - mu) / sd

    assignments, k_chosen, cluster_metrics = auto_kmeans(
        feat_z, k_max=cfg.k_clusters_max, seed=cfg.random_seed
    )
    print(f"  K chosen by silhouette: {k_chosen} (silhouette={cluster_metrics['silhouette_score']:.3f})")

    # Cluster summary using RAW drops for interpretation
    cluster_summary: Dict[str, Any] = {}
    feat_raw = np.array([[r["drop_medqa"], r["drop_medmcqa"], r["drop_pubmedqa"]] for r in rows])
    for cid in range(int(assignments.max()) + 1):
        mask = assignments == cid
        if not mask.any():
            continue
        cluster_summary[str(cid)] = {
            "count": int(mask.sum()),
            "mean_drop_medqa": float(feat_raw[mask, 0].mean()),
            "mean_drop_medmcqa": float(feat_raw[mask, 1].mean()),
            "mean_drop_pubmedqa": float(feat_raw[mask, 2].mean()),
            "mean_rel_medqa": float(feat_relative[mask, 0].mean()),
            "mean_rel_medmcqa": float(feat_relative[mask, 1].mean()),
            "mean_rel_pubmedqa": float(feat_relative[mask, 2].mean()),
            "std_drop_medqa": float(feat_raw[mask, 0].std()),
            "std_drop_medmcqa": float(feat_raw[mask, 1].std()),
            "std_drop_pubmedqa": float(feat_raw[mask, 2].std()),
        }

    print("\n  Cluster summary (raw drops, mean ± std):")
    for cid in sorted(cluster_summary.keys(), key=int):
        s = cluster_summary[cid]
        print(f"    C{cid} (n={s['count']:2d}): "
              f"mqa={s['mean_drop_medqa']:+.4f}±{s['std_drop_medqa']:.3f}, "
              f"mmcqa={s['mean_drop_medmcqa']:+.4f}±{s['std_drop_medmcqa']:.3f}, "
              f"pmqa={s['mean_drop_pubmedqa']:+.4f}±{s['std_drop_pubmedqa']:.3f}")

    for i, r in enumerate(rows):
        r["cluster"] = int(assignments[i])

    # =====================================================================
    # SAVE
    # =====================================================================
    out = {
        "metadata": {
            "model_key": MODEL_KEY,
            "model_id": MODEL_ID,
            "timestamp": datetime.now().isoformat(),
            "config": asdict(cfg),
            "torch_version": torch.__version__,
            "ablation_method": "column_crush_1bit",
            "anchor_count": len(anchor_neurons),
            "anchors_input": MED42_ANCHORS,
            "anchors_used": anchor_neurons,
            "pubmedqa_format": "binary_yes_no",
            "clustering_features": "z-scored relative drops (drop / baseline_margin)",
        },
        "dataset_baselines": baseline_summary,
        "anchors": rows,
        "cluster_metrics": cluster_metrics,
        "cluster_summary": cluster_summary,
    }
    out_path = os.path.join(cfg.output_dir, f"{MODEL_KEY}_cross_dataset_results.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved: {out_path}")
    return out


# =====================================================================
# CLI
# =====================================================================
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="MED42 cross-dataset anchor profiling")
    p.add_argument("--n-medqa", type=int, default=400)
    p.add_argument("--n-medmcqa", type=int, default=400)
    p.add_argument("--n-pubmedqa", type=int, default=400)
    p.add_argument("--k-clusters-max", type=int, default=6)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output-dir", type=str, default="results")
    p.add_argument("--cache-dir", type=str, default=".")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = EvalConfig(
        n_medqa=args.n_medqa,
        n_medmcqa=args.n_medmcqa,
        n_pubmedqa=args.n_pubmedqa,
        k_clusters_max=args.k_clusters_max,
        random_seed=args.seed,
        output_dir=args.output_dir,
        cache_dir=args.cache_dir,
    )
    evaluate(cfg)
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()