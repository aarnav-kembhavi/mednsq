"""
End-to-end EMS pipeline for OpenMeditron/Meditron3-Qwen2.5-7B.

Steps:
  1. Load model in bf16, load MedQA train/test splits.
  2. Build adversarial calibration pairs from train.
  3. For each layer in MIDDLE_LAYERS:
       a. Forward-Taylor proxy ranks columns -> top STAGE1_TOPK candidates.
       b. Stage 1: full-calibration column crush, keep columns with mean_drop > random baseline.
       c. Stage 2: re-evaluate survivors on validation pairs (train held-out subset)
          -> select anchors with z > Z_THRESHOLD.
  4. Save anchors_meditron_qwen25_7b.json.
  5. Ablation sweep on K in [1,2,4,8,16,32,64]:
       - Anchor run: crush top-K anchors, measure margin drop on calib + accuracy on test.
       - Random control: 30 trials, same-layer-distribution sampling, same crush.
       - Report z-score, accuracy translation rate.
  6. Save ablation_meditron_qwen25_7b.json.

Designed for ~3-4 hours on a single A100 80GB.
"""

import hashlib
import inspect
import json
import math
import os
import random
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
from scipy.stats import norm
from transformers import AutoModelForCausalLM, AutoTokenizer

from mednsq_data import (
    build_adversarial_pairs,
    format_prompt,
    load_mcq_dataset,
)
from mednsq_eval import evaluate_model
from mednsq_probe import MedNSQProbe


# =====================================================================
# CONFIG
# =====================================================================
@dataclass
class Config:
    model_name: str = "m42-health/Llama3-Med42-8B"
    # Middle layers — Llama-3-8B has 32 layers; middle = 12..27 inclusive
    middle_layers: Tuple[int, ...] = tuple(range(14, 27))
    seed: int = 42

    # Calibration / validation / test sizes
    calib_size: int = 400         # adversarial pairs from MedQA train
    val_size: int = 200           # held-out adversarial pairs (still from train, no overlap)
    test_size: int = 300          # MedQA test split, used for ablation accuracy

    # Discovery
    stage1_topk: int = 64        # forward-Taylor top-k per layer
    stage1_eval_pairs: int = 80  # cheaper subset for stage 1
    stage2_eval_pairs: int = 400  # full calib for stage 2
    random_baseline_cols: int = 32
    z_threshold: float = 4.0
    max_anchors: int = 64

    # Ablation
    k_values: Tuple[int, ...] = (1, 2, 4, 8, 16, 32, 64)
    n_random_trials: int = 30
    ablation_eval_pairs: int = 200  # margin drop measured on this subset of calib
    ablation_test_size: int = 300   # accuracy on test split

    # Output
    discovery_file: str = "anchors_med42_8b.json"
    ablation_file: str = "ablation_med42_8b.json"
    log_file: str = "experiment_med42_8b.log"

    intervention_type: str = "column_crush_1bit"


CFG = Config()


def log(msg: str) -> None:
    line = f"[{datetime.now().strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(CFG.log_file, "a", encoding="utf-8") as f:
        f.write(line + "\n")


# =====================================================================
# SETUP
# =====================================================================
def setup_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def setup_perf() -> None:
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision("high")


# =====================================================================
# DISCOVERY
# =====================================================================
def discover_anchors(
    model,
    tokenizer,
    probe: MedNSQProbe,
    calib_pairs: List[Dict[str, Any]],
    val_pairs: List[Dict[str, Any]],
    pad_id: int,
) -> Tuple[List[Dict[str, Any]], Dict[int, Dict[str, float]]]:
    """Returns (anchors, layer_stats)."""
    log(f"Discovery start. Layers={list(CFG.middle_layers)} calib={len(calib_pairs)} val={len(val_pairs)}")

    # Cache calib + val baseline margins
    calib_baseline = probe.compute_per_sample_margins(calib_pairs, pad_id=pad_id)
    val_baseline = probe.compute_per_sample_margins(val_pairs, pad_id=pad_id)
    log(f"Calib baseline margin mean={calib_baseline.mean():.3f} std={calib_baseline.std():.3f} "
        f"frac_neg={(calib_baseline < 0).float().mean():.2f}")
    log(f"Val baseline margin mean={val_baseline.mean():.3f} std={val_baseline.std():.3f}")

    # Sanity: round-trip a column, verify margins identical after restore
    sanity_layer = CFG.middle_layers[0]
    log(f"Round-trip sanity: crushing layer={sanity_layer} col=0, restoring, comparing margins...")
    pre = probe.compute_per_sample_margins(calib_pairs[:32], pad_id=pad_id)
    orig = probe.simulate_column_crush(sanity_layer, 0)
    probe.restore_column(sanity_layer, 0, orig)
    post = probe.compute_per_sample_margins(calib_pairs[:32], pad_id=pad_id)
    diff = (pre - post).abs().max().item()
    if diff > 1e-3:
        raise RuntimeError(f"Round-trip failed: max margin diff after restore = {diff}")
    log(f"Round-trip OK (max diff = {diff:.2e})")

    layer_stats: Dict[int, Dict[str, float]] = {}
    all_candidates: List[Dict[str, Any]] = []

    rng = random.Random(CFG.seed + 7)

    for layer_idx in CFG.middle_layers:
        t0 = time.time()
        log(f"--- Layer {layer_idx} ---")

        # Stage 0: forward-Taylor scoring
        scores = probe.forward_taylor_scores(
            layer_idx=layer_idx,
            adv_pairs=calib_pairs,
            n_pairs=CFG.stage1_eval_pairs,
            pad_id=pad_id,
        )
        top_vals, top_idx = torch.topk(scores, k=min(CFG.stage1_topk, scores.numel()))
        candidate_cols = [int(c) for c in top_idx.tolist()]
        log(f"  Taylor top-{len(candidate_cols)} cols. score range=[{top_vals[-1]:.4f}, {top_vals[0]:.4f}]")

        # Random baseline columns (from full population, NOT excluding top — overlap is ~1%)
        rand_cols = rng.sample(range(probe.intermediate_size), CFG.random_baseline_cols)

        # Stage 1: evaluate candidates and randoms on a calib subset
        stage1_pairs = calib_pairs[:CFG.stage1_eval_pairs]
        stage1_baseline = calib_baseline[:CFG.stage1_eval_pairs]

        def eval_cols_batched(cols: List[int], pairs, baseline) -> List[float]:
            """Crush each column, measure margin drop, restore. batch_size=32 keeps GPU fed."""
            drops = []
            for c in cols:
                orig = probe.simulate_column_crush(layer_idx, c)
                try:
                    m = probe.compute_per_sample_margins(pairs, batch_size=32, pad_id=pad_id)
                    drops.append((baseline - m).mean().item())
                finally:
                    probe.restore_column(layer_idx, c, orig)
            return drops

        rand_drops = eval_cols_batched(rand_cols, stage1_pairs, stage1_baseline)
        mu_r = float(np.mean(rand_drops))
        sigma_r = float(np.std(rand_drops))
        log(f"  Random baseline: mu={mu_r:.5f} sigma={sigma_r:.5f}")

        cand_drops_stage1 = list(zip(
            candidate_cols,
            eval_cols_batched(candidate_cols, stage1_pairs, stage1_baseline),
        ))

        # Keep candidates with positive drop above random mean
        survivors = [(c, d) for c, d in cand_drops_stage1 if d > mu_r]
        survivors.sort(key=lambda x: x[1], reverse=True)
        survivors = survivors[:32]  # cap before stage 2
        log(f"  Stage 1 survivors: {len(survivors)} (above mu_rand)")

        # Stage 2: re-evaluate survivors on validation pairs
        survivor_cols = [c for c, _ in survivors]
        val_drops = []
        for c in survivor_cols:
            orig = probe.simulate_column_crush(layer_idx, c)
            try:
                m_val = probe.compute_per_sample_margins(val_pairs, batch_size=32, pad_id=pad_id)
                val_drops.append((val_baseline - m_val).mean().item())
            finally:
                probe.restore_column(layer_idx, c, orig)

        calib_drop_lookup = dict(cand_drops_stage1)
        stage2_results = []
        for c, drop_val in zip(survivor_cols, val_drops):
            z = (drop_val - mu_r) / (sigma_r + 1e-8)
            stage2_results.append({
                "layer": layer_idx,
                "column": c,
                "drop_calib": calib_drop_lookup.get(c, 0.0),
                "drop_val": drop_val,
                "z_score": z,
                "p_value": float(1.0 - norm.cdf(z)),
            })

        # Anchors: pass z threshold on VALIDATION drop
        layer_anchors = [r for r in stage2_results if r["z_score"] > CFG.z_threshold]
        layer_anchors.sort(key=lambda r: r["drop_val"], reverse=True)
        all_candidates.extend(layer_anchors)

        layer_stats[layer_idx] = {
            "mu_rand": mu_r,
            "sigma_rand": sigma_r,
            "n_survivors": len(survivors),
            "n_anchors": len(layer_anchors),
            "elapsed_sec": time.time() - t0,
        }
        log(f"  Layer {layer_idx} done in {layer_stats[layer_idx]['elapsed_sec']:.1f}s. "
            f"Anchors (z>{CFG.z_threshold}): {len(layer_anchors)}")

    all_candidates.sort(key=lambda r: r["drop_val"], reverse=True)
    anchors = all_candidates[:CFG.max_anchors]
    log(f"Discovery complete. Total anchors: {len(anchors)}")
    return anchors, layer_stats


# =====================================================================
# ABLATION SWEEP
# =====================================================================
def crush_many(probe: MedNSQProbe, neurons: List[Tuple[int, int]]) -> List[Tuple[int, int, torch.Tensor]]:
    saved = []
    for layer, col in neurons:
        orig = probe.simulate_column_crush(layer, col)
        saved.append((layer, col, orig))
    return saved


def restore_many(probe: MedNSQProbe, saved: List[Tuple[int, int, torch.Tensor]]) -> None:
    for layer, col, orig in saved:
        probe.restore_column(layer, col, orig)


def sample_random_neurons(
    k: int,
    anchors_subset: List[Tuple[int, int]],
    intermediate_size: int,
    rng: random.Random,
) -> List[Tuple[int, int]]:
    """Match per-layer count of anchors_subset, exclude collisions."""
    from collections import defaultdict
    layer_counts: Dict[int, int] = defaultdict(int)
    for l, _ in anchors_subset:
        layer_counts[l] += 1
    anchor_set = set(anchors_subset)
    chosen: List[Tuple[int, int]] = []
    chosen_set = set()
    for layer, count in layer_counts.items():
        added = 0
        attempts = 0
        while added < count and attempts < count * 200:
            c = rng.randrange(intermediate_size)
            t = (layer, c)
            if t not in anchor_set and t not in chosen_set:
                chosen.append(t)
                chosen_set.add(t)
                added += 1
            attempts += 1
    return chosen


def run_ablation(
    model,
    tokenizer,
    probe: MedNSQProbe,
    anchors: List[Dict[str, Any]],
    calib_pairs: List[Dict[str, Any]],
    test_samples: List[Dict[str, Any]],
    pad_id: int,
) -> Dict[str, Any]:
    log("=== Ablation sweep ===")
    anchor_tuples = [(a["layer"], a["column"]) for a in anchors]

    # Baseline
    eval_pairs = calib_pairs[:CFG.ablation_eval_pairs]
    base_margins = probe.compute_per_sample_margins(eval_pairs, batch_size=32, pad_id=pad_id)
    base_margin = float(base_margins.mean().item())
    base_test = evaluate_model(model, tokenizer, test_samples)
    base_acc = base_test["accuracy"]
    log(f"Baseline margin={base_margin:.4f} accuracy={base_acc:.4f}")

    rng = random.Random(CFG.seed + 99)
    results: Dict[int, Dict[str, Any]] = {}

    for K in CFG.k_values:
        if K > len(anchor_tuples):
            log(f"Skipping K={K} (only {len(anchor_tuples)} anchors)")
            continue
        log(f"--- K={K} ---")
        subset = anchor_tuples[:K]

        # Anchor run
        saved = crush_many(probe, subset)
        try:
            anchor_margins = probe.compute_per_sample_margins(eval_pairs, batch_size=32, pad_id=pad_id)
            anchor_margin = float(anchor_margins.mean().item())
            anchor_test = evaluate_model(model, tokenizer, test_samples)
            anchor_acc = anchor_test["accuracy"]
        finally:
            restore_many(probe, saved)

        anchor_margin_drop = base_margin - anchor_margin
        anchor_acc_drop = base_acc - anchor_acc

        # Random trials
        rand_margin_drops: List[float] = []
        rand_acc_drops: List[float] = []
        for trial in range(CFG.n_random_trials):
            rand_neurons = sample_random_neurons(K, subset, probe.intermediate_size, rng)
            saved_r = crush_many(probe, rand_neurons)
            try:
                rm = probe.compute_per_sample_margins(eval_pairs, batch_size=32, pad_id=pad_id)
                rand_margin = float(rm.mean().item())
                # Test accuracy is expensive — only run for a sub-sample of trials to save time
                if trial < min(10, CFG.n_random_trials):
                    rt = evaluate_model(model, tokenizer, test_samples)
                    rand_acc = rt["accuracy"]
                    rand_acc_drops.append(base_acc - rand_acc)
            finally:
                restore_many(probe, saved_r)
            rand_margin_drops.append(base_margin - rand_margin)

        rmd = np.array(rand_margin_drops)
        rad = np.array(rand_acc_drops) if rand_acc_drops else np.array([0.0])
        z = (anchor_margin_drop - rmd.mean()) / (rmd.std() + 1e-8)

        translation = (anchor_acc_drop / anchor_margin_drop) if abs(anchor_margin_drop) > 1e-6 else 0.0

        log(f"  Anchor margin drop: {anchor_margin_drop:+.4f}")
        log(f"  Random margin drop: {rmd.mean():+.4f} ± {rmd.std():.4f}  (z = {z:+.2f})")
        log(f"  Anchor accuracy drop: {anchor_acc_drop:+.4f}")
        log(f"  Random accuracy drop: {rad.mean():+.4f} ± {rad.std():.4f}  (n={len(rad)})")
        log(f"  Translation rate (acc_drop / margin_drop): {translation:+.3f}")

        results[K] = {
            "K": K,
            "anchor_margin_drop": anchor_margin_drop,
            "anchor_acc_drop": anchor_acc_drop,
            "rand_margin_drop_mean": float(rmd.mean()),
            "rand_margin_drop_std": float(rmd.std()),
            "rand_acc_drop_mean": float(rad.mean()),
            "rand_acc_drop_std": float(rad.std()),
            "z_margin": float(z),
            "translation_rate": float(translation),
            "n_random_trials": CFG.n_random_trials,
        }

    return {
        "baseline_margin": base_margin,
        "baseline_accuracy": base_acc,
        "per_k": results,
    }


# =====================================================================
# MAIN
# =====================================================================
def main():
    setup_perf()
    setup_seeds(CFG.seed)
    log(f"Config: {asdict(CFG)}")

    log("Loading tokenizer + model (bf16, device_map=auto)...")
    tokenizer = AutoTokenizer.from_pretrained(CFG.model_name, trust_remote_code=True, use_fast=False)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    pad_id = tokenizer.pad_token_id

    model = AutoModelForCausalLM.from_pretrained(
        CFG.model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()
    probe = MedNSQProbe(model)

    # Sanity assert: anchors will fit
    log(f"Model intermediate_size = {probe.intermediate_size}, layers = {len(probe.layers)}")

    # Load data
    log("Loading MedQA train + test splits...")
    train_pool = load_mcq_dataset(n_total=CFG.calib_size + CFG.val_size + 100, split="train")
    random.Random(CFG.seed).shuffle(train_pool)
    calib_samples = train_pool[:CFG.calib_size]
    val_samples = train_pool[CFG.calib_size:CFG.calib_size + CFG.val_size]
    test_samples = load_mcq_dataset(n_total=CFG.test_size, split="test")
    log(f"Calib={len(calib_samples)} Val={len(val_samples)} Test={len(test_samples)}")

    # Build adversarial pairs
    log("Building adversarial pairs (calib)...")
    calib_pairs = build_adversarial_pairs(model, tokenizer, calib_samples, n_calib=len(calib_samples))
    assert all(p["pos_id"] != p["neg_id"] for p in calib_pairs), "Bad adversarial pair (pos==neg)"
    log(f"Built {len(calib_pairs)} calib pairs")

    log("Building adversarial pairs (val)...")
    val_pairs = build_adversarial_pairs(model, tokenizer, val_samples, n_calib=len(val_samples))
    log(f"Built {len(val_pairs)} val pairs")

    # Discovery
    anchors, layer_stats = discover_anchors(model, tokenizer, probe, calib_pairs, val_pairs, pad_id)

    if len(anchors) == 0:
        log("WARNING: No anchors discovered. Aborting before ablation.")
        return

    prompt_hash = hashlib.sha256(inspect.getsource(format_prompt).encode("utf-8")).hexdigest()[:16]

    discovery_output = {
        "metadata": {
            "timestamp": datetime.now().isoformat(),
            "model_name": CFG.model_name,
            "config": asdict(CFG),
            "dtype": "torch.bfloat16",
            "prompt_template_hash": prompt_hash,
            "intervention_type": CFG.intervention_type,
            "selection_rule": f"z_score(val) > {CFG.z_threshold}",
        },
        "layer_stats": layer_stats,
        "anchors": anchors,
    }
    with open(CFG.discovery_file, "w", encoding="utf-8") as f:
        json.dump(discovery_output, f, indent=2)
    log(f"Saved discovery to {CFG.discovery_file} ({len(anchors)} anchors)")

    # Ablation
    ablation = run_ablation(model, tokenizer, probe, anchors, calib_pairs, test_samples[:CFG.ablation_test_size], pad_id)

    ablation_output = {
        "metadata": {
            "timestamp": datetime.now().isoformat(),
            "model_name": CFG.model_name,
            "config": asdict(CFG),
            "prompt_template_hash": prompt_hash,
            "intervention_type": CFG.intervention_type,
            "anchors_used": [(a["layer"], a["column"]) for a in anchors],
        },
        "results": ablation,
    }
    with open(CFG.ablation_file, "w", encoding="utf-8") as f:
        json.dump(ablation_output, f, indent=2)
    log(f"Saved ablation to {CFG.ablation_file}")

    # Summary
    log("\n=== FINAL SUMMARY ===")
    log(f"Baseline: margin={ablation['baseline_margin']:.4f}, accuracy={ablation['baseline_accuracy']:.4f}")
    log(f"{'K':>4}  {'AnchMarΔ':>10}  {'RandMarΔ':>10}  {'zMar':>6}  {'AnchAccΔ':>10}  {'TransRate':>10}")
    for K in CFG.k_values:
        r = ablation["per_k"].get(K)
        if r is None:
            continue
        log(f"{K:>4}  "
            f"{r['anchor_margin_drop']:>+10.4f}  "
            f"{r['rand_margin_drop_mean']:>+10.4f}  "
            f"{r['z_margin']:>+6.2f}  "
            f"{r['anchor_acc_drop']:>+10.4f}  "
            f"{r['translation_rate']:>+10.3f}")


if __name__ == "__main__":
    main()