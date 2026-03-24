"""
Med42 anchor discovery via EMS only — stripped from run_mednsq_gemma.py.
"""

import random
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import norm
from transformers import AutoModelForCausalLM, AutoTokenizer

from mednsq_data import build_adversarial_pairs, load_mcq_dataset
from mednsq_eval import _get_letter_token_ids, evaluate_model
from mednsq_probe import MedNSQProbe

# EMS configuration constants (same as run_mednsq_gemma.py)
RANDOM_BASELINE_COLS = 90
Z_THRESHOLD = 2.0
BATCH_SIZE = 32


@dataclass
class EMSConfig:
    """Configuration for EMS experiments."""

    layer_idx: int = 2
    calibration_size: int = 800
    stage1_top_k: int = 256
    stage1_samples: int = 120
    stage2_top_k: int = 128
    stage2_samples: int = 800


def _batched_margins_and_predictions(
    model,
    adv_pairs: List[Dict[str, Any]],
    letter_token_ids: torch.Tensor,
    batch_size: int = BATCH_SIZE,
    pad_token_id: int = 0,
) -> Tuple[torch.Tensor, List[int]]:
    """Compute per-sample margins and A/B/C/D predictions in mini-batches."""
    if not adv_pairs:
        return torch.zeros(0, dtype=torch.float32), []

    device = next(model.parameters()).device

    margins: List[float] = []
    preds: List[int] = []

    for start in range(0, len(adv_pairs), batch_size):
        batch = adv_pairs[start : start + batch_size]
        seq_lens = [pair["input_ids"].shape[1] for pair in batch]
        max_len = max(seq_lens)

        input_batch = []
        mask_batch = []

        for pair in batch:
            ids = pair["input_ids"].to(device)
            mask = pair["attention_mask"].to(device)
            pad_len = max_len - ids.shape[1]
            if pad_len > 0:
                ids = F.pad(ids, (0, pad_len), value=pad_token_id)
                mask = F.pad(mask, (0, pad_len), value=0)
            input_batch.append(ids)
            mask_batch.append(mask)

        input_ids = torch.cat(input_batch, dim=0)
        attention_mask = torch.cat(mask_batch, dim=0)

        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits

        for i, pair in enumerate(batch):
            last_idx = seq_lens[i] - 1
            token_logits = logits[i, last_idx, :]

            pos_id = pair["pos_id"]
            neg_id = pair["neg_id"]
            margin = (token_logits[pos_id] - token_logits[neg_id]).item()
            margins.append(margin)

            letter_logits = token_logits[letter_token_ids]
            pred_idx = int(torch.argmax(letter_logits).item())
            preds.append(pred_idx)

    return torch.tensor(margins, dtype=torch.float32), preds


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
    track_flips: bool = False,
    pad_token_id: int = 0,
) -> Tuple[torch.Tensor, float, float, float]:
    if not adv_pairs:
        return torch.zeros(0, dtype=torch.float32), 0.0, 0.0, 0.0

    if max_samples is not None:
        adv_pairs = adv_pairs[:max_samples]
        baseline_margins = baseline_margins[:max_samples]

    original_col = probe.simulate_column_crush(layer_idx, col_idx)

    crushed_margins: List[float] = []
    preds: List[int] = []
    processed = 0
    stop_early = False

    try:
        device = next(model.parameters()).device
        letter_token_ids = letter_token_ids.to(device)

        for start in range(0, len(adv_pairs), BATCH_SIZE):
            batch = adv_pairs[start : start + BATCH_SIZE]
            batch_baseline = baseline_margins[processed : processed + len(batch)]

            margins_batch, preds_batch = _batched_margins_and_predictions(
                model,
                batch,
                letter_token_ids,
                batch_size=BATCH_SIZE,
                pad_token_id=pad_token_id,
            )

            for j, margin in enumerate(margins_batch.tolist()):
                crushed_margins.append(margin)
                if track_flips:
                    preds.append(preds_batch[j])

                processed += 1

                if (
                    early_stop_mu is not None
                    and processed % 10 == 0
                    and processed <= len(baseline_margins)
                ):
                    base_so_far = baseline_margins[:processed]
                    crushed_so_far = torch.tensor(
                        crushed_margins[:processed], dtype=torch.float32
                    )
                    mean_drop_so_far = float(
                        (base_so_far - crushed_so_far).mean().item()
                    )
                    threshold = early_stop_mu
                    if early_stop_sigma is not None:
                        threshold = early_stop_mu + 0.25 * early_stop_sigma
                    if mean_drop_so_far < threshold:
                        stop_early = True
                        break

            if stop_early:
                break

    finally:
        probe.restore_column(layer_idx, col_idx, original_col)

    if not crushed_margins:
        return torch.zeros(0, dtype=torch.float32), 0.0, 0.0, 0.0

    crushed_tensor = torch.tensor(crushed_margins, dtype=torch.float32)
    drops = baseline_margins[: len(crushed_tensor)] - crushed_tensor
    mean_drop = float(drops.mean().item())

    flip_rate = 0.0
    lethal_flip_rate = 0.0

    if track_flips and preds:
        total = len(preds)
        flips = 0
        lethal_flips = 0

        letter_token_ids_cpu = letter_token_ids.detach().cpu()

        for i, pred_idx in enumerate(preds):
            pair = adv_pairs[i]
            correct_token = int(pair["pos_id"])
            neg_token = int(pair["neg_id"])
            pred_token = int(letter_token_ids_cpu[pred_idx].item())

            if pred_token != correct_token:
                flips += 1
                if pred_token == neg_token:
                    lethal_flips += 1

        flip_rate = flips / total if total > 0 else 0.0
        lethal_flip_rate = lethal_flips / total if total > 0 else 0.0

    return drops, mean_drop, flip_rate, lethal_flip_rate


def _run_ems_for_layer(
    model,
    tokenizer,
    probe: MedNSQProbe,
    adv_pairs: List[Dict[str, Any]],
    evaluation_samples: List[Dict[str, Any]],
    layer_idx: int,
    cfg: EMSConfig,
    baseline_margins_all: torch.Tensor,
    baseline_eval: Dict[str, float],
) -> Dict[str, Any]:
    """Run the full EMS pipeline (Taylor → Stage1 → Stage2 → Z) for one layer."""
    jacobian_scores = probe.compute_contrastive_jacobian(layer_idx, adv_pairs)
    num_cols = jacobian_scores.numel()

    stage1_top_k = min(cfg.stage1_top_k, num_cols)
    _top_vals, top_indices = torch.topk(jacobian_scores, k=stage1_top_k)

    stage1_samples = min(cfg.stage1_samples, len(adv_pairs))
    stage2_samples = min(cfg.stage2_samples, len(adv_pairs))

    baseline_stage1 = baseline_margins_all[:stage1_samples]

    letter_token_ids = _get_letter_token_ids(tokenizer)
    pad_token_id = getattr(tokenizer, "pad_token_id", None)
    if pad_token_id is None:
        pad_token_id = getattr(tokenizer, "eos_token_id", 0)

    all_indices = torch.arange(num_cols, device=jacobian_scores.device)
    rand_cols = all_indices[torch.randperm(num_cols)[:RANDOM_BASELINE_COLS]].tolist()

    random_mean_drops: List[float] = []
    for col in rand_cols:
        drops, mean_drop, _, _ = _evaluate_column_ems(
            model,
            probe,
            layer_idx,
            int(col),
            adv_pairs[:stage1_samples],
            baseline_stage1,
            letter_token_ids,
            early_stop_mu=None,
            max_samples=stage1_samples,
            track_flips=False,
            pad_token_id=pad_token_id,
        )
        random_mean_drops.append(mean_drop)

    if random_mean_drops:
        rand_tensor = torch.tensor(random_mean_drops, dtype=torch.float32)
        mu_rand = float(rand_tensor.mean().item())
        sigma_rand = float(rand_tensor.std(unbiased=False).item())
    else:
        mu_rand = 0.0
        sigma_rand = 0.0

    stage1_candidates: List[Tuple[int, float]] = []
    for col in top_indices.tolist():
        drops, mean_drop, _, _ = _evaluate_column_ems(
            model,
            probe,
            layer_idx,
            int(col),
            adv_pairs[:stage1_samples],
            baseline_stage1,
            letter_token_ids,
            early_stop_mu=mu_rand,
            early_stop_sigma=sigma_rand,
            max_samples=stage1_samples,
            track_flips=False,
            pad_token_id=pad_token_id,
        )
        if mean_drop > 0.0:
            stage1_candidates.append((int(col), mean_drop))

    stage1_candidates.sort(key=lambda x: x[1], reverse=True)
    stage2_columns = [c for c, _ in stage1_candidates[: cfg.stage2_top_k]]

    validated_anchors: List[Dict[str, Any]] = []
    stage2_results: List[Dict[str, Any]] = []

    adv_pairs_stage2 = adv_pairs[:stage2_samples]
    baseline_stage2 = baseline_margins_all[:stage2_samples]

    for col in stage2_columns:
        drops, mean_drop, flip_rate, lethal_flip_rate = _evaluate_column_ems(
            model,
            probe,
            layer_idx,
            int(col),
            adv_pairs_stage2,
            baseline_stage2,
            letter_token_ids,
            early_stop_mu=None,
            max_samples=stage2_samples,
            track_flips=True,
            pad_token_id=pad_token_id,
        )

        z = (mean_drop - mu_rand) / (sigma_rand + 1e-8)
        p_value = 1 - norm.cdf(z)

        std_drop = float(drops.std(unbiased=False).item())
        std_drop = max(std_drop, 1e-3)
        cohens_d = mean_drop / std_drop

        median_drop = float(torch.median(drops).item())

        result = {
            "layer": layer_idx,
            "column": int(col),
            "mean_drop": mean_drop,
            "median_drop": median_drop,
            "z_score": z,
            "p_value": p_value,
            "cohens_d": cohens_d,
            "flip_rate": flip_rate,
            "lethal_flip_rate": lethal_flip_rate,
        }
        stage2_results.append(result)

        if z > Z_THRESHOLD:
            validated_anchors.append(result)

    if stage2_results:
        p_values = [r["p_value"] for r in stage2_results]
        fdr_mask = benjamini_hochberg(p_values, q=0.05)

        for r, keep in zip(stage2_results, fdr_mask):
            r["fdr_significant"] = keep

    if validated_anchors:
        max_drop = max(a["mean_drop"] for a in validated_anchors)
        mean_drop_validated = float(
            sum(a["mean_drop"] for a in validated_anchors) / len(validated_anchors)
        )
        max_z = max(a["z_score"] for a in validated_anchors)
    else:
        max_drop = 0.0
        mean_drop_validated = 0.0
        max_z = 0.0

    return {
        "layer": layer_idx,
        "mu_rand": mu_rand,
        "sigma_rand": sigma_rand,
        "validated_anchors": validated_anchors,
        "stage2_results": stage2_results,
        "max_drop": max_drop,
        "mean_drop": mean_drop_validated,
        "max_z": max_z,
        "baseline_eval": baseline_eval,
    }


def benjamini_hochberg(p_values, q=0.05):
    m = len(p_values)
    sorted_indices = sorted(range(m), key=lambda i: p_values[i])
    sorted_p = [p_values[i] for i in sorted_indices]

    thresholds = [(i + 1) / m * q for i in range(m)]

    passed = [False] * m
    max_i = -1

    for i in range(m):
        if sorted_p[i] <= thresholds[i]:
            max_i = i

    if max_i >= 0:
        for i in range(max_i + 1):
            passed[sorted_indices[i]] = True

    return passed


def main() -> None:
    seed = 1
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    model_name = "m42-health/Llama3-Med42-8B"
    layers_to_test = list(range(11, 22))
    cfg = EMSConfig()

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision("high")

    eval_size = 60
    total_needed = cfg.calibration_size + eval_size
    samples = load_mcq_dataset(n_total=total_needed)
    random.shuffle(samples)
    calibration = samples[: cfg.calibration_size]
    evaluation = samples[cfg.calibration_size : cfg.calibration_size + eval_size]

    adv_pairs = build_adversarial_pairs(
        model, tokenizer, calibration, n_calib=len(calibration)
    )

    probe = MedNSQProbe(model)
    baseline_margins_all = probe.compute_per_sample_margins(adv_pairs)
    baseline_eval = evaluate_model(model, tokenizer, evaluation)

    for layer_idx in layers_to_test:
        summary = _run_ems_for_layer(
            model,
            tokenizer,
            probe,
            adv_pairs,
            evaluation,
            layer_idx,
            cfg,
            baseline_margins_all,
            baseline_eval,
        )
        n_anchors = len(summary["validated_anchors"])
        max_z = float(summary["max_z"])
        print(f"layer={layer_idx} anchors={n_anchors} max_Z={max_z:.3f}")


if __name__ == "__main__":
    main()
