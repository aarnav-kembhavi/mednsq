"""
MedGemma-4B ablation sweep using pre-discovered anchors.
Compatible with the same pipeline as Meditron / Med42 / AFM-OpenMed.

Notes on architecture:
  - MedGemma-4B-it is multimodal (Gemma3ForConditionalGeneration).
  - We use the TEXT decoder only — visual encoder is unused for MedQA.
  - Layers are accessed via model.model.language_model.layers (Gemma3 nesting).
  - MLP block has gate_proj + up_proj + down_proj (SwiGLU), same as Llama/Qwen.
  - Probe handles this since _get_layers walks model.model.layers — we may need
    a small adjustment for Gemma's nested language_model.
"""

import hashlib
import inspect
import json
import os
import random
import time
from collections import defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from mednsq_data import build_adversarial_pairs, format_prompt, load_mcq_dataset
from mednsq_eval import evaluate_model
from mednsq_probe import MedNSQProbe


# =====================================================================
# CONFIG
# =====================================================================
@dataclass
class Config:
    # If load fails, try "google/medgemma-4b-pt" or "google/medgemma-27b-text-it"
    model_name: str = "google/medgemma-4b-it"
    seed: int = 42

    calib_size: int = 400
    val_size: int = 200
    test_size: int = 300

    # Ablation
    k_values: Tuple[int, ...] = (1, 2, 4, 8, 16, 32, 64)
    n_random_trials: int = 30
    ablation_eval_pairs: int = 200
    ablation_test_size: int = 300

    # Output
    ablation_file: str = "ablation_medgemma_4b.json"
    log_file: str = "experiment_medgemma_4b.log"

    intervention_type: str = "column_crush_1bit"


CFG = Config()

# Pre-discovered MedGemma anchors, ranked by mean drop (high → low)
MEDGEMMA_ANCHORS: List[Tuple[int, int]] = [
    (20, 8153), (22, 4572), (18, 1539), (17, 6857), (19, 8682),
    (17, 4970), (15, 1192), (9, 10132), (18, 10081), (12, 8419),
    (19, 747),  (13, 1545), (19, 757),  (9, 8314),  (21, 1332),
    (15, 95),   (11, 4480), (17, 2697), (15, 9541), (18, 1583),
    (16, 3433), (8, 3347),  (21, 9373), (15, 1827), (18, 5631),
    (14, 7080), (14, 4189), (19, 9222), (8, 3977),  (16, 1761),
    (16, 4146), (12, 1855), (17, 7170), (14, 2521), (17, 10058),
    (24, 1444), (20, 4459), (14, 4132), (16, 2056), (9, 832),
    (16, 4273), (8, 5386),  (13, 2345), (9, 8192),  (21, 988),
    (12, 1995), (10, 1826), (8, 2877),  (17, 7513), (9, 6724),
    (13, 945),  (16, 8664), (10, 3530), (8, 1577),  (10, 183),
    (17, 6148), (15, 209),  (21, 6065), (10, 5422), (17, 5353),
    (18, 1277), (20, 6146), (12, 4067), (16, 6845), (15, 1780),
    (10, 8392),
]


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
    log("=" * 60)
    log("ARCHITECTURE REPORT")
    log("=" * 60)
    log(f"Model class: {type(model).__name__}")
    log(f"Config model_type: {getattr(model.config, 'model_type', 'unknown')}")
    log(f"Hidden size: {probe.hidden_size}")
    log(f"Intermediate size: {probe.intermediate_size}")
    log(f"Num layers: {len(probe.layers)}")

    layer0 = probe.layers[0]
    if hasattr(layer0, "mlp"):
        log(f"Layer 0 MLP submodules: {[name for name, _ in layer0.mlp.named_children()]}")
    log("=" * 60)

    # Validate anchors against architecture
    n_layers = len(probe.layers)
    n_cols = probe.intermediate_size
    valid = [(l, c) for (l, c) in MEDGEMMA_ANCHORS if 0 <= l < n_layers and 0 <= c < n_cols]
    invalid = [(l, c) for (l, c) in MEDGEMMA_ANCHORS if not (0 <= l < n_layers and 0 <= c < n_cols)]
    log(f"Anchor validity: {len(valid)} valid, {len(invalid)} out-of-range")
    if invalid:
        log(f"Invalid anchors (will be skipped): {invalid}")
    return valid


# =====================================================================
# ABLATION HELPERS
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


# =====================================================================
# ABLATION SWEEP
# =====================================================================
def run_ablation(
    model,
    tokenizer,
    probe: MedNSQProbe,
    anchors: List[Tuple[int, int]],
    calib_pairs: List[Dict[str, Any]],
    test_samples: List[Dict[str, Any]],
    pad_id: int,
) -> Dict[str, Any]:
    log("=== Ablation sweep ===")

    eval_pairs = calib_pairs[:CFG.ablation_eval_pairs]
    base_margins = probe.compute_per_sample_margins(eval_pairs, batch_size=32, pad_id=pad_id)
    base_margin = float(base_margins.mean().item())
    base_test = evaluate_model(model, tokenizer, test_samples)
    base_acc = base_test["accuracy"]
    log(f"Baseline margin={base_margin:.4f} accuracy={base_acc:.4f}")

    rng = random.Random(CFG.seed + 99)
    results: Dict[int, Dict[str, Any]] = {}

    for K in CFG.k_values:
        if K > len(anchors):
            log(f"Skipping K={K} (only {len(anchors)} anchors)")
            continue
        log(f"--- K={K} ---")
        subset = anchors[:K]

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
    tokenizer = AutoTokenizer.from_pretrained(CFG.model_name, trust_remote_code=True, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    pad_id = tokenizer.pad_token_id

    # MedGemma is multimodal (Gemma3ForConditionalGeneration). AutoModelForCausalLM
    # may complain — fall back to the conditional generation class if needed.
    try:
        model = AutoModelForCausalLM.from_pretrained(
            CFG.model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
        )
    except (ValueError, KeyError) as e:
        log(f"AutoModelForCausalLM failed: {e}. Trying Gemma3ForConditionalGeneration...")
        from transformers import Gemma3ForConditionalGeneration
        model = Gemma3ForConditionalGeneration.from_pretrained(
            CFG.model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
        )

    model.eval()

    # Locate the text decoder layers manually for multimodal Gemma3.
    # IMPORTANT: probe needs the full model for forward passes (lm_head lives there)
    # but layer access goes through the language_model submodule.
    if hasattr(model, "language_model") and hasattr(model.language_model, "layers"):
        log("Detected multimodal Gemma3 — layers at model.language_model.layers")
        layers_module = model.language_model
    elif hasattr(model, "model") and hasattr(model.model, "language_model"):
        log("Detected nested multimodal Gemma3 — layers at model.model.language_model.layers")
        layers_module = model.model.language_model
    else:
        layers_module = model

    # Build probe against the full model (for forwards) but patch its layer access
    probe = MedNSQProbe(model)
    # Override the layers reference to point at the text decoder
    probe.layers = layers_module.layers
    # Re-probe dimensions from the actual decoder layer
    from mednsq_probe import _get_mlp_down_proj
    sample = probe.layers[0]
    down = _get_mlp_down_proj(sample)
    probe.hidden_size = down.weight.shape[0]
    probe.intermediate_size = down.weight.shape[1]
    log(f"Probe rebound: hidden={probe.hidden_size} intermediate={probe.intermediate_size} layers={len(probe.layers)}")

    valid_anchors = report_architecture(model, probe)

    if len(valid_anchors) == 0:
        log("FATAL: No valid anchors after bounds check. Aborting.")
        return

    if len(valid_anchors) < len(MEDGEMMA_ANCHORS):
        log(f"WARNING: {len(MEDGEMMA_ANCHORS) - len(valid_anchors)} anchors out-of-range, dropped.")

    # Trim k_values to fit
    max_k = len(valid_anchors)
    cfg_k = tuple(k for k in CFG.k_values if k <= max_k)
    if cfg_k != CFG.k_values:
        log(f"Trimming k_values to {cfg_k} (only {max_k} valid anchors)")

    log("Loading MedQA train + test splits...")
    train_pool = load_mcq_dataset(n_total=CFG.calib_size + CFG.val_size + 100, split="train")
    random.Random(CFG.seed).shuffle(train_pool)
    calib_samples = train_pool[:CFG.calib_size]
    test_samples = load_mcq_dataset(n_total=CFG.test_size, split="test")
    log(f"Calib={len(calib_samples)} Test={len(test_samples)}")

    log("Building adversarial pairs (calib)...")
    calib_pairs = build_adversarial_pairs(model, tokenizer, calib_samples, n_calib=len(calib_samples))
    assert all(p["pos_id"] != p["neg_id"] for p in calib_pairs), "Bad adversarial pair (pos==neg)"
    log(f"Built {len(calib_pairs)} calib pairs")

    ablation = run_ablation(
        model, tokenizer, probe, valid_anchors, calib_pairs,
        test_samples[:CFG.ablation_test_size], pad_id,
    )

    prompt_hash = hashlib.sha256(inspect.getsource(format_prompt).encode("utf-8")).hexdigest()[:16]

    ablation_output = {
        "metadata": {
            "timestamp": datetime.now().isoformat(),
            "model_name": CFG.model_name,
            "config": asdict(CFG),
            "prompt_template_hash": prompt_hash,
            "intervention_type": CFG.intervention_type,
            "anchors_used": [list(a) for a in valid_anchors],
            "anchors_input_count": len(MEDGEMMA_ANCHORS),
            "anchors_valid_count": len(valid_anchors),
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