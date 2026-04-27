"""
Meditron3-Qwen2.5-7B anchor ablation validator.

Matches discovery intervention by using MedNSQProbe.simulate_column_crush /
restore_column directly (no forward hooks).
"""

import hashlib
import inspect
import json
import random
from collections import defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
from datasets import load_dataset

import os
os.environ.setdefault("USE_TF", "0")

from transformers import AutoModelForCausalLM, AutoTokenizer

from mednsq_data import build_adversarial_pairs, format_prompt, load_mcq_dataset
from mednsq_eval import evaluate_model
from mednsq_probe import MedNSQProbe


@dataclass
class AblationConfig:
    model_name: str = "OpenMeditron/Meditron3-Qwen2.5-7B"
    anchor_file: str = "anchors_meditron_qwen25_7b.json"
    output_file: str = "ablation_meditron_qwen25_7b.json"
    seed: int = 42
    calibration_pairs: int = 200
    eval_samples: int = 200
    k_values: Tuple[int, ...] = (1, 2, 4, 8, 16, 32, 64)
    random_trials: int = 30
    intervention_type: str = "column_crush_1bit"
    dtype: str = "torch.bfloat16"


def _load_anchor_bundle(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _verify_anchor_config(bundle: Dict[str, Any], cfg: AblationConfig) -> None:
    if "metadata" not in bundle or "anchors" not in bundle:
        raise RuntimeError("Anchor file missing required keys: metadata / anchors.")

    meta = bundle["metadata"]
    cfg_block = meta.get("config", {})
    required_checks = {
        "model_name": (cfg_block.get("model_name"), cfg.model_name),
        "intervention_type": (meta.get("intervention_type"), cfg.intervention_type),
        "dtype": (meta.get("dtype"), cfg.dtype),
    }

    prompt_hash_current = hashlib.sha256(
        inspect.getsource(format_prompt).encode("utf-8")
    ).hexdigest()[:16]
    prompt_hash_saved = meta.get("prompt_template_hash")
    if prompt_hash_saved != prompt_hash_current:
        raise RuntimeError(
            "Prompt template hash mismatch between discovery output and current code. "
            f"saved={prompt_hash_saved} current={prompt_hash_current}"
        )

    for key, (saved, expected) in required_checks.items():
        if saved != expected:
            raise RuntimeError(
                f"Config mismatch for '{key}': saved='{saved}' expected='{expected}'."
            )


def _load_medqa_test_subset(n: int, seed: int) -> List[Dict[str, Any]]:
    ds = load_dataset("openlifescienceai/medqa")
    split = "test" if "test" in ds else "validation" if "validation" in ds else "train"
    rows = ds[split]
    idxs = list(range(len(rows)))
    rng = random.Random(seed)
    rng.shuffle(idxs)
    out: List[Dict[str, Any]] = []
    for i in idxs:
        row = rows[i]
        q = row["data"]["Question"]
        opts = row["data"]["Options"]
        options = [opts["A"], opts["B"], opts["C"], opts["D"]]
        correct_letter = row["data"]["Correct Option"]
        correct_index = "ABCD".index(correct_letter)
        out.append({"question": q, "options": options, "correct_index": correct_index})
        if len(out) >= n:
            break
    return out


def _margin_and_flip_metrics(
    baseline_margins: torch.Tensor,
    crushed_margins: torch.Tensor,
) -> Tuple[float, float]:
    drops = baseline_margins - crushed_margins
    mean_drop = float(drops.mean().item()) if drops.numel() else 0.0
    # Flip rate: sign flip of margin relative to baseline.
    flips = ((baseline_margins > 0) & (crushed_margins <= 0)).float()
    flip_rate = float(flips.mean().item()) if flips.numel() else 0.0
    return mean_drop, flip_rate


def _apply_crush_set(
    probe: MedNSQProbe,
    neurons: List[Tuple[int, int]],
) -> List[Tuple[int, int, torch.Tensor]]:
    saved: List[Tuple[int, int, torch.Tensor]] = []
    for layer, col in neurons:
        original = probe.simulate_column_crush(layer, col)
        saved.append((layer, col, original))
    return saved


def _restore_crush_set(probe: MedNSQProbe, saved: List[Tuple[int, int, torch.Tensor]]) -> None:
    for layer, col, original in reversed(saved):
        probe.restore_column(layer, col, original)


def _sample_random_matched(
    k: int,
    anchor_subset: List[Tuple[int, int]],
    n_layers: int,
    n_cols: int,
    rng: random.Random,
) -> List[Tuple[int, int]]:
    layer_counts = defaultdict(int)
    anchor_set = set(anchor_subset)
    for layer, _ in anchor_subset:
        layer_counts[layer] += 1

    chosen: List[Tuple[int, int]] = []
    chosen_set = set()
    for layer, count in layer_counts.items():
        added = 0
        attempts = 0
        while added < count and attempts < count * 1000:
            c = rng.randrange(n_cols)
            cand = (layer, c)
            if cand not in anchor_set and cand not in chosen_set:
                chosen.append(cand)
                chosen_set.add(cand)
                added += 1
            attempts += 1
        if added < count:
            # fallback across all layers if specific layer exhausted
            while added < count:
                cand = (rng.randrange(n_layers), rng.randrange(n_cols))
                if cand not in anchor_set and cand not in chosen_set:
                    chosen.append(cand)
                    chosen_set.add(cand)
                    added += 1
    return chosen[:k]


def main() -> None:
    cfg = AblationConfig()
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision("high")

    anchor_bundle = _load_anchor_bundle(cfg.anchor_file)
    _verify_anchor_config(anchor_bundle, cfg)
    anchors = [(int(a["layer"]), int(a["column"])) for a in anchor_bundle["anchors"]]

    tokenizer = AutoTokenizer.from_pretrained(
        cfg.model_name,
        trust_remote_code=True,
        use_fast=False,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        cfg.model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()
    probe = MedNSQProbe(model)

    # Calibration: MedQA train, adversarial pairs.
    calib_ds = load_mcq_dataset(n_total=cfg.calibration_pairs)
    adv_pairs = build_adversarial_pairs(model, tokenizer, calib_ds, n_calib=cfg.calibration_pairs)

    # Held-out eval: MedQA test split. Different split => no overlap.
    eval_ds = _load_medqa_test_subset(cfg.eval_samples, cfg.seed)

    baseline_margins = probe.compute_per_sample_margins(adv_pairs)
    baseline_eval = evaluate_model(model, tokenizer, eval_ds)
    base_acc = float(baseline_eval["accuracy"])

    dims = probe.get_layer_weight(0).shape
    n_layers = len(probe.layers)
    n_cols = int(dims[1])
    rng = random.Random(cfg.seed)

    per_k_results: Dict[int, Dict[str, Any]] = {}

    for k in cfg.k_values:
        if k > len(anchors):
            continue
        anchor_subset = anchors[:k]

        # Anchor run.
        saved = []
        try:
            saved = _apply_crush_set(probe, anchor_subset)
            anchor_margins = probe.compute_per_sample_margins(adv_pairs)
            anchor_margin_drop, anchor_flip_rate = _margin_and_flip_metrics(
                baseline_margins, anchor_margins
            )
            anchor_eval = evaluate_model(model, tokenizer, eval_ds)
            anchor_acc = float(anchor_eval["accuracy"])
        finally:
            _restore_crush_set(probe, saved)

        anchor_acc_drop = base_acc - anchor_acc

        # Random control.
        rand_margin_drops: List[float] = []
        rand_acc_drops: List[float] = []
        rand_flip_rates: List[float] = []
        for _ in range(cfg.random_trials):
            rand_neurons = _sample_random_matched(k, anchor_subset, n_layers, n_cols, rng)
            saved_rand = []
            try:
                saved_rand = _apply_crush_set(probe, rand_neurons)
                rand_margins = probe.compute_per_sample_margins(adv_pairs)
                r_margin_drop, r_flip_rate = _margin_and_flip_metrics(
                    baseline_margins, rand_margins
                )
                rand_eval = evaluate_model(model, tokenizer, eval_ds)
                r_acc_drop = base_acc - float(rand_eval["accuracy"])
            finally:
                _restore_crush_set(probe, saved_rand)
            rand_margin_drops.append(r_margin_drop)
            rand_acc_drops.append(r_acc_drop)
            rand_flip_rates.append(r_flip_rate)

        rand_margin_mean = float(np.mean(rand_margin_drops))
        rand_margin_std = float(np.std(rand_margin_drops))
        rand_acc_mean = float(np.mean(rand_acc_drops))
        rand_acc_std = float(np.std(rand_acc_drops))
        z = (anchor_margin_drop - rand_margin_mean) / (rand_margin_std + 1e-8)
        translation_rate = anchor_acc_drop / anchor_margin_drop if abs(anchor_margin_drop) > 1e-8 else 0.0

        print(f"\n=== K = {k} ===")
        print(f"Anchor margin drop:    {anchor_margin_drop:.4f}")
        print(f"Random margin drop:    {rand_margin_mean:.4f} ± {rand_margin_std:.4f}  (z = {z:.2f})")
        print(f"Anchor accuracy drop:  {anchor_acc_drop:.4f}")
        print(f"Random accuracy drop:  {rand_acc_mean:.4f} ± {rand_acc_std:.4f}")
        print(f"Translation rate:      {translation_rate:.2f}  (accuracy drop per unit margin drop)")

        per_k_results[k] = {
            "anchor_margin_drop": anchor_margin_drop,
            "anchor_accuracy_drop": anchor_acc_drop,
            "anchor_flip_rate": anchor_flip_rate,
            "random_margin_drop_mean": rand_margin_mean,
            "random_margin_drop_std": rand_margin_std,
            "random_accuracy_drop_mean": rand_acc_mean,
            "random_accuracy_drop_std": rand_acc_std,
            "random_flip_rate_mean": float(np.mean(rand_flip_rates)),
            "random_flip_rate_std": float(np.std(rand_flip_rates)),
            "z_score_margin_drop": float(z),
            "translation_rate": float(translation_rate),
            "random_margin_drops": rand_margin_drops,
            "random_accuracy_drops": rand_acc_drops,
            "random_flip_rates": rand_flip_rates,
        }

    output = {
        "metadata": {
            "timestamp": datetime.now().isoformat(),
            "config": asdict(cfg),
            "baseline_accuracy": base_acc,
            "baseline_mean_margin": float(baseline_margins.mean().item()) if baseline_margins.numel() else 0.0,
            "anchor_source_file": cfg.anchor_file,
        },
        "results_by_k": per_k_results,
    }
    with open(cfg.output_file, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved ablation output: {cfg.output_file}")


if __name__ == "__main__":
    main()
