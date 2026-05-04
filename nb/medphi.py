"""
End-to-end EMS pipeline for openmed-community/AFM-4.5B-OpenMed-RL-CoT.

NOTE on architecture:
  - AFM-4.5B uses ReLU^2 activation (not SwiGLU) and grouped-query attention.
  - It is a custom architecture (model_type="arcee") requiring trust_remote_code.
  - The MLP likely has up_proj + down_proj (no gate). The intervention is still
    1-bit column crush on down_proj.weight — semantically the same as for
    SwiGLU models (zeroing one neuron's contribution to the residual).
  - LAYER RANGE BELOW IS A GUESS. Once the probe prints `layers=N` at startup,
    edit `middle_layers` to cover the middle ~50% of layers.
"""

import hashlib
import inspect
import json
import os
import random
import time
from dataclasses import asdict, dataclass
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
    model_name: str = "microsoft/MediPhi-Instruct"
    # Phi-3.5-mini has 32 layers (0-31). Middle 60% = layers 6-25.
    middle_layers: Tuple[int, ...] = tuple(range(12, 24))
    seed: int = 42

    # Calibration / validation / test sizes
    calib_size: int = 400
    val_size: int = 200
    test_size: int = 300

    # Discovery
    stage1_topk: int = 64
    stage1_eval_pairs: int = 80
    stage2_eval_pairs: int = 200
    random_baseline_cols: int = 32
    z_threshold: float = 2.0
    max_anchors: int = 100

    # Ablation
    k_values: Tuple[int, ...] = (1, 2, 4, 8, 16, 32, 64)
    n_random_trials: int = 5
    ablation_eval_pairs: int = 200
    ablation_test_size: int = 300

    # Output
    discovery_file: str = "anchors_medphi_instruct.json"
    ablation_file: str = "ablation_medphi_instruct.json"
    log_file: str = "experiment_medphi_instruct.log"

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


def report_architecture(model, probe: MedNSQProbe) -> None:
    """Print enough about the model to verify probe will work correctly."""
    log("=" * 60)
    log("ARCHITECTURE REPORT — MediPhi-Instruct (Phi-3.5-mini)")
    log("=" * 60)
    log(f"Model class: {type(model).__name__}")
    log(f"Config model_type: {model.config.model_type}")
    log(f"Hidden size: {probe.hidden_size}")
    log(f"Intermediate size (= MLP neurons per layer): {probe.intermediate_size}")
    log(f"Num layers: {len(probe.layers)}")
    log(f"Num attention heads: {probe.num_heads}")

    # Phi-3.5-mini MLP uses gate_up_proj (fused) + down_proj
    # gate_up_proj.weight shape: [2 * intermediate_size, hidden_size]
    # down_proj.weight shape:    [hidden_size, intermediate_size]
    layer0 = probe.layers[0]
    mlp0   = layer0.mlp
    down0  = mlp0.down_proj
    log(f"Layer 0 MLP submodules: {[name for name, _ in mlp0.named_children()]}")
    log(f"Layer 0 down_proj weight shape: {list(down0.weight.shape)}")
    log(f"Configured middle_layers: {list(CFG.middle_layers)}")

    if max(CFG.middle_layers) >= len(probe.layers):
        log(f"FATAL: middle_layers max={max(CFG.middle_layers)} "
            f"but model has only {len(probe.layers)} layers. Fix CFG.")
    log("=" * 60)


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
    log(f"Discovery start. Layers={list(CFG.middle_layers)} calib={len(calib_pairs)} val={len(val_pairs)}")

    calib_baseline = probe.compute_per_sample_margins(calib_pairs, pad_id=pad_id)
    val_baseline = probe.compute_per_sample_margins(val_pairs, pad_id=pad_id)
    log(f"Calib baseline margin mean={calib_baseline.mean():.3f} std={calib_baseline.std():.3f} "
        f"frac_neg={(calib_baseline < 0).float().mean():.2f}")
    log(f"Val baseline margin mean={val_baseline.mean():.3f} std={val_baseline.std():.3f}")

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

        scores = probe.forward_taylor_scores(
            layer_idx=layer_idx,
            adv_pairs=calib_pairs,
            n_pairs=CFG.stage1_eval_pairs,
            pad_id=pad_id,
        )
        top_vals, top_idx = torch.topk(scores, k=min(CFG.stage1_topk, scores.numel()))
        candidate_cols = [int(c) for c in top_idx.tolist()]
        log(f"  Taylor top-{len(candidate_cols)} cols. score range=[{top_vals[-1]:.4f}, {top_vals[0]:.4f}]")

        rand_cols = rng.sample(range(probe.intermediate_size), CFG.random_baseline_cols)

        stage1_pairs = calib_pairs[:CFG.stage1_eval_pairs]
        stage1_baseline = calib_baseline[:CFG.stage1_eval_pairs]

        def eval_cols_batched(cols: List[int], pairs, baseline) -> List[float]:
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

        survivors = [(c, d) for c, d in cand_drops_stage1 if d > mu_r]
        survivors.sort(key=lambda x: x[1], reverse=True)
        survivors = survivors[:32]
        log(f"  Stage 1 survivors: {len(survivors)} (above mu_rand)")

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

        rand_margin_drops: List[float] = []
        rand_acc_drops: List[float] = []
        for trial in range(CFG.n_random_trials):
            rand_neurons = sample_random_neurons(K, subset, probe.intermediate_size, rng)
            saved_r = crush_many(probe, rand_neurons)
            try:
                rm = probe.compute_per_sample_margins(eval_pairs, batch_size=32, pad_id=pad_id)
                rand_margin = float(rm.mean().item())
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
    # MediPhi-Instruct is standard transformers, no trust_remote_code needed
    tokenizer = AutoTokenizer.from_pretrained(CFG.model_name, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    pad_id = tokenizer.pad_token_id

    model = AutoModelForCausalLM.from_pretrained(
        CFG.model_name,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto",
        low_cpu_mem_usage=True,
    )

    # ---- Manual architecture inspection for MediPhi (Phi-3.5-mini) ----
    # Do NOT use getattr chains — inspect directly to avoid silent failures
    log("Inspecting MediPhi architecture manually...")
    _layers = model.model.layers          # nn.ModuleList, 32 layers
    _layer0 = _layers[0]
    _mlp0   = _layer0.mlp
    # Phi-3.5-mini MLP: gate_up_proj (fused, dim = 2 * intermediate_size) + down_proj
    # down_proj.weight shape: [hidden_size, intermediate_size]
    _down0  = _mlp0.down_proj
    _hidden_size       = _down0.weight.shape[0]   # 3072
    _intermediate_size = _down0.weight.shape[1]   # 8192
    _num_layers        = len(_layers)              # 32
    _num_heads         = model.config.num_attention_heads  # 32

    log(f"  num_layers        = {_num_layers}")
    log(f"  hidden_size       = {_hidden_size}")
    log(f"  intermediate_size = {_intermediate_size}")
    log(f"  num_heads         = {_num_heads}")
    log(f"  down_proj shape   = {list(_down0.weight.shape)}")
    # Verify expected values; abort if wrong to avoid silent bugs
    assert _num_layers == 32,        f"Expected 32 layers, got {_num_layers}"
    assert _hidden_size == 3072,     f"Expected hidden 3072, got {_hidden_size}"
    assert _intermediate_size == 8192, f"Expected intermediate 8192, got {_intermediate_size}"
    log("  Architecture assertions passed.")

    probe = MedNSQProbe(model)
    # Override probe fields with manually verified values
    probe.layers           = _layers
    probe.hidden_size      = _hidden_size
    probe.intermediate_size = _intermediate_size
    probe.num_heads        = _num_heads

    # Print full architecture so user can verify before discovery starts
    report_architecture(model, probe)

    # Validate middle_layers against actual layer count
    if max(CFG.middle_layers) >= len(probe.layers):
        log(f"FATAL: middle_layers max={max(CFG.middle_layers)} but model has {len(probe.layers)} layers")
        log("Edit CFG.middle_layers in this script and rerun. Aborting.")
        return

    log("Loading MedQA train + test splits...")
    train_pool = load_mcq_dataset(n_total=CFG.calib_size + CFG.val_size + 100, split="train")
    random.Random(CFG.seed).shuffle(train_pool)
    calib_samples = train_pool[:CFG.calib_size]
    val_samples = train_pool[CFG.calib_size:CFG.calib_size + CFG.val_size]
    test_samples = load_mcq_dataset(n_total=CFG.test_size, split="test")
    log(f"Calib={len(calib_samples)} Val={len(val_samples)} Test={len(test_samples)}")

    log("Building adversarial pairs (calib)...")
    calib_pairs = build_adversarial_pairs(model, tokenizer, calib_samples, n_calib=len(calib_samples))
    assert all(p["pos_id"] != p["neg_id"] for p in calib_pairs), "Bad adversarial pair (pos==neg)"
    log(f"Built {len(calib_pairs)} calib pairs")

    log("Building adversarial pairs (val)...")
    val_pairs = build_adversarial_pairs(model, tokenizer, val_samples, n_calib=len(val_samples))
    log(f"Built {len(val_pairs)} val pairs")

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

    ablation = run_ablation(model, tokenizer, probe, anchors, calib_pairs,
                            test_samples[:CFG.ablation_test_size], pad_id)

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