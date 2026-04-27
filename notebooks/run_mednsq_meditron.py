"""
Meditron3-Qwen2.5-7B EMS discovery pipeline.

Produces anchors_meditron_qwen25_7b.json with full config metadata.
"""

import hashlib
import inspect
import json
import os
import random
from dataclasses import dataclass, asdict
from datetime import datetime
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
from scipy.stats import norm

# Avoid optional TensorFlow import path issues inside transformers utils.
os.environ.setdefault("USE_TF", "0")

from transformers import AutoModelForCausalLM, AutoTokenizer

from mednsq_data import build_adversarial_pairs, format_prompt, load_mcq_dataset
from mednsq_eval import evaluate_model, _get_letter_token_ids
from mednsq_probe import MedNSQProbe


RANDOM_BASELINE_COLS = 256
Z_THRESHOLD = 4.0
BATCH_SIZE = 32


@dataclass
class EMSConfig:
    model_name: str = "OpenMeditron/Meditron3-Qwen2.5-7B"
    layers_to_test: Tuple[int, ...] = tuple(range(8, 23))
    seeds: Tuple[int, ...] = (1, 2)
    calibration_size: int = 800
    eval_size: int = 60
    stage1_top_k: int = 256
    stage1_samples: int = 200
    stage2_top_k: int = 128
    stage2_samples: int = 800
    random_baseline_cols: int = RANDOM_BASELINE_COLS
    z_threshold: float = Z_THRESHOLD
    intervention_type: str = "column_crush_1bit"
    output_file: str = "anchors_meditron_qwen25_7b.json"


def log_progress(text: str) -> None:
    with open("experiment_progress_meditron_qwen25_7b.log", "a", encoding="utf-8") as f:
        f.write(text + "\n")


def save_checkpoint(data: Dict[str, Any]) -> None:
    torch.save(data, "ems_checkpoint_meditron_qwen25_7b.pt")


def load_checkpoint() -> Any:
    if not os.path.exists("ems_checkpoint_meditron_qwen25_7b.pt"):
        return None
    try:
        return torch.load("ems_checkpoint_meditron_qwen25_7b.pt", weights_only=False)
    except TypeError:
        return torch.load("ems_checkpoint_meditron_qwen25_7b.pt")


def benjamini_hochberg(p_values: List[float], q: float = 0.05) -> List[bool]:
    m = len(p_values)
    order = sorted(range(m), key=lambda i: p_values[i])
    sorted_p = [p_values[i] for i in order]
    passed = [False] * m
    max_i = -1
    for i, p in enumerate(sorted_p):
        if p <= ((i + 1) / m) * q:
            max_i = i
    if max_i >= 0:
        for i in range(max_i + 1):
            passed[order[i]] = True
    return passed


def _evaluate_column_ems(
    model,
    probe: MedNSQProbe,
    layer_idx: int,
    col_idx: int,
    adv_pairs: List[Dict[str, Any]],
    baseline_margins: torch.Tensor,
    letter_token_ids: torch.Tensor,
    early_stop_mu: float | None = None,
    early_stop_sigma: float | None = None,
    max_samples: int | None = None,
) -> Tuple[torch.Tensor, float]:
    if max_samples is not None:
        adv_pairs = adv_pairs[:max_samples]
        baseline_margins = baseline_margins[:max_samples]
    if not adv_pairs:
        return torch.zeros(0, dtype=torch.float32), 0.0

    original_col = probe.simulate_column_crush(layer_idx, col_idx)
    crushed: List[float] = []
    processed = 0
    try:
        with torch.no_grad():
            for pair in adv_pairs:
                logits = model(
                    input_ids=pair["input_ids"],
                    attention_mask=pair["attention_mask"],
                ).logits[:, -1, :]
                margin = (logits[0, pair["pos_id"]] - logits[0, pair["neg_id"]]).item()
                crushed.append(margin)
                processed += 1
                if early_stop_mu is not None and processed % 10 == 0:
                    drops_so_far = baseline_margins[:processed] - torch.tensor(
                        crushed[:processed], dtype=torch.float32
                    )
                    mean_drop = float(drops_so_far.mean().item())
                    threshold = early_stop_mu if early_stop_sigma is None else (early_stop_mu + 0.25 * early_stop_sigma)
                    if mean_drop < threshold:
                        break
    finally:
        probe.restore_column(layer_idx, col_idx, original_col)

    crushed_tensor = torch.tensor(crushed, dtype=torch.float32)
    drops = baseline_margins[: len(crushed_tensor)] - crushed_tensor
    return drops, float(drops.mean().item()) if drops.numel() > 0 else 0.0


def _run_ems_for_layer(
    model,
    tokenizer,
    probe: MedNSQProbe,
    adv_pairs: List[Dict[str, Any]],
    layer_idx: int,
    cfg: EMSConfig,
    baseline_margins_all: torch.Tensor,
) -> Dict[str, Any]:
    jacobian_scores = probe.compute_contrastive_jacobian(layer_idx, adv_pairs)
    num_cols = jacobian_scores.numel()
    stage1_top_k = min(cfg.stage1_top_k, num_cols)
    _, top_indices = torch.topk(jacobian_scores, k=stage1_top_k)

    letter_token_ids = _get_letter_token_ids(tokenizer).to(next(model.parameters()).device)
    stage1_samples = min(cfg.stage1_samples, len(adv_pairs))
    stage2_samples = min(cfg.stage2_samples, len(adv_pairs))
    baseline_stage1 = baseline_margins_all[:stage1_samples]

    top_set = set(int(x) for x in top_indices.tolist())
    all_indices = [i for i in range(num_cols) if i not in top_set]
    rng = random.Random(1000 + layer_idx)
    rand_cols = rng.sample(all_indices, min(cfg.random_baseline_cols, len(all_indices)))

    random_mean_drops: List[float] = []
    for col in rand_cols:
        _, mean_drop = _evaluate_column_ems(
            model=model,
            probe=probe,
            layer_idx=layer_idx,
            col_idx=int(col),
            adv_pairs=adv_pairs[:stage1_samples],
            baseline_margins=baseline_stage1,
            letter_token_ids=letter_token_ids,
            max_samples=stage1_samples,
        )
        random_mean_drops.append(mean_drop)

    rand_tensor = torch.tensor(random_mean_drops, dtype=torch.float32) if random_mean_drops else torch.zeros(1)
    mu_rand = float(rand_tensor.mean().item())
    sigma_rand = float(rand_tensor.std(unbiased=False).item()) if rand_tensor.numel() > 1 else 0.0

    stage1_candidates: List[Tuple[int, float]] = []
    for col in top_indices.tolist():
        _, mean_drop = _evaluate_column_ems(
            model=model,
            probe=probe,
            layer_idx=layer_idx,
            col_idx=int(col),
            adv_pairs=adv_pairs[:stage1_samples],
            baseline_margins=baseline_stage1,
            letter_token_ids=letter_token_ids,
            early_stop_mu=mu_rand,
            early_stop_sigma=sigma_rand,
            max_samples=stage1_samples,
        )
        if mean_drop > 0.0:
            stage1_candidates.append((int(col), mean_drop))

    stage1_candidates.sort(key=lambda x: x[1], reverse=True)
    stage2_columns = [c for c, _ in stage1_candidates[: cfg.stage2_top_k]]
    baseline_stage2 = baseline_margins_all[:stage2_samples]
    adv_pairs_stage2 = adv_pairs[:stage2_samples]

    stage2_results: List[Dict[str, Any]] = []
    validated_anchors: List[Dict[str, Any]] = []
    for col in stage2_columns:
        drops, mean_drop = _evaluate_column_ems(
            model=model,
            probe=probe,
            layer_idx=layer_idx,
            col_idx=int(col),
            adv_pairs=adv_pairs_stage2,
            baseline_margins=baseline_stage2,
            letter_token_ids=letter_token_ids,
            max_samples=stage2_samples,
        )
        z = (mean_drop - mu_rand) / (sigma_rand + 1e-8)
        p_value = 1 - norm.cdf(z)
        result = {
            "layer": layer_idx,
            "column": int(col),
            "mean_drop": float(mean_drop),
            "median_drop": float(torch.median(drops).item()) if drops.numel() > 0 else 0.0,
            "z_score": float(z),
            "p_value": float(p_value),
        }
        stage2_results.append(result)
        if z > cfg.z_threshold:
            validated_anchors.append(result)

    if stage2_results:
        p_values = [r["p_value"] for r in stage2_results]
        fdr_mask = benjamini_hochberg(p_values, q=0.05)
        for r, keep in zip(stage2_results, fdr_mask):
            r["fdr_significant"] = keep

    return {
        "layer": layer_idx,
        "mu_rand": mu_rand,
        "sigma_rand": sigma_rand,
        "stage2_results": stage2_results,
        "validated_anchors": validated_anchors,
    }


def _run_sanity_probe(model, tokenizer, probe: MedNSQProbe, adv_pairs: List[Dict[str, Any]]) -> None:
    n = min(100, len(adv_pairs))
    if n == 0:
        raise RuntimeError("No adversarial pairs available for sanity probe.")
    baseline = probe.compute_per_sample_margins(adv_pairs[:n])
    letter_token_ids = _get_letter_token_ids(tokenizer)
    _, mean_drop = _evaluate_column_ems(
        model=model,
        probe=probe,
        layer_idx=0,
        col_idx=0,
        adv_pairs=adv_pairs[:n],
        baseline_margins=baseline,
        letter_token_ids=letter_token_ids,
        max_samples=n,
    )
    if abs(mean_drop) >= 0.01:
        raise RuntimeError(
            f"Sanity probe failed at layer 0 col 0: |mean_drop|={abs(mean_drop):.6f} >= 0.01"
        )
    print(f"Sanity probe OK: layer=0 col=0 mean_drop={mean_drop:.6f}")


def main() -> None:
    cfg = EMSConfig()
    tokenizer = AutoTokenizer.from_pretrained(
        cfg.model_name,
        trust_remote_code=True,
        use_fast=False,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision("high")

    checkpoint = load_checkpoint()
    if checkpoint:
        completed_seeds = list(checkpoint.get("completed_seeds", []))
        all_seed_results = list(checkpoint.get("all_seed_results", []))
    else:
        completed_seeds = []
        all_seed_results = []

    for seed in cfg.seeds:
        if seed in completed_seeds:
            print(f"Seed {seed} already completed, skipping.")
            continue
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

        model = AutoModelForCausalLM.from_pretrained(
            cfg.model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
        )
        model.eval()
        probe = MedNSQProbe(model)

        total_needed = cfg.calibration_size + cfg.eval_size
        samples = load_mcq_dataset(n_total=total_needed)
        random.shuffle(samples)
        calibration = samples[: cfg.calibration_size]
        evaluation = samples[cfg.calibration_size : cfg.calibration_size + cfg.eval_size]
        adv_pairs = build_adversarial_pairs(model, tokenizer, calibration, n_calib=len(calibration))
        baseline_margins_all = probe.compute_per_sample_margins(adv_pairs)
        baseline_eval = evaluate_model(model, tokenizer, evaluation)
        print(f"Seed={seed} baseline accuracy={baseline_eval['accuracy']:.4f} margin={baseline_eval['mean_margin']:.4f}")

        _run_sanity_probe(model, tokenizer, probe, adv_pairs)

        layer_results = []
        for layer_idx in cfg.layers_to_test:
            print(f"Running layer {layer_idx}")
            layer_summary = _run_ems_for_layer(
                model=model,
                tokenizer=tokenizer,
                probe=probe,
                adv_pairs=adv_pairs,
                layer_idx=layer_idx,
                cfg=cfg,
                baseline_margins_all=baseline_margins_all,
            )
            layer_results.append(layer_summary)
            log_progress(
                f"seed={seed} layer={layer_idx} anchors={len(layer_summary['validated_anchors'])} "
                f"mu_rand={layer_summary['mu_rand']:.6f} sigma_rand={layer_summary['sigma_rand']:.6f}"
            )

        all_seed_results.append({"seed": seed, "layers": layer_results})
        completed_seeds.append(seed)
        save_checkpoint({"completed_seeds": completed_seeds, "all_seed_results": all_seed_results})

        del probe
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

    anchor_to_best: Dict[Tuple[int, int], Dict[str, float]] = {}
    for seed_result in all_seed_results:
        for layer_summary in seed_result["layers"]:
            for a in layer_summary["validated_anchors"]:
                key = (int(a["layer"]), int(a["column"]))
                prev = anchor_to_best.get(key, {"drop": -1e9, "z": -1e9})
                if a["mean_drop"] > prev["drop"]:
                    anchor_to_best[key] = {"drop": float(a["mean_drop"]), "z": float(a["z_score"])}

    sorted_anchors = sorted(anchor_to_best.items(), key=lambda kv: kv[1]["drop"], reverse=True)[:64]
    output_anchors = [
        {"layer": int(layer), "column": int(col), "drop": stats["drop"], "z": stats["z"]}
        for (layer, col), stats in sorted_anchors
    ]

    prompt_template_hash = hashlib.sha256(
        inspect.getsource(format_prompt).encode("utf-8")
    ).hexdigest()[:16]

    output = {
        "metadata": {
            "timestamp": datetime.now().isoformat(),
            "config": asdict(cfg),
            "dtype": "torch.bfloat16",
            "prompt_template_hash": prompt_template_hash,
            "calibration_size": cfg.calibration_size,
            "seed_list": list(cfg.seeds),
            "intervention_type": cfg.intervention_type,
            "selection_rule": f"z>{cfg.z_threshold}",
        },
        "seeds": all_seed_results,
        "anchors": output_anchors,
    }

    with open(cfg.output_file, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)
    print(f"Saved discovery output: {cfg.output_file} (anchors={len(output_anchors)})")


if __name__ == "__main__":
    main()
