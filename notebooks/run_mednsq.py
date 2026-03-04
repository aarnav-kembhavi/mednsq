import math
from typing import List, Dict, Any, Tuple

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM

from mednsq_data import load_mcq_dataset, build_adversarial_pairs
from mednsq_eval import evaluate_model, _get_letter_token_ids
from mednsq_probe import MedNSQProbe


# EMS configuration constants
STAGE1_SAMPLES = 30
STAGE2_TOP_K = 20
RANDOM_BASELINE_COLS = 20
Z_THRESHOLD = 2.0
BATCH_SIZE = 8


def _batched_margins_and_predictions(
    model,
    adv_pairs: List[Dict[str, Any]],
    letter_token_ids: torch.Tensor,
    batch_size: int = BATCH_SIZE,
) -> Tuple[torch.Tensor, List[int]]:
    """Compute per-sample margins and A/B/C/D predictions in mini-batches.

    This function operates on the CURRENT model weights, so it can be used for
    both baseline and crushed evaluations.
    """
    if not adv_pairs:
        return torch.zeros(0, dtype=torch.float32), []

    device = next(model.parameters()).device
    letter_token_ids = letter_token_ids.to(device)

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
                ids = F.pad(ids, (0, pad_len), value=0)
                mask = F.pad(mask, (0, pad_len), value=0)
            input_batch.append(ids)
            mask_batch.append(mask)

        input_ids = torch.cat(input_batch, dim=0)
        attention_mask = torch.cat(mask_batch, dim=0)

        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits  # [B, max_len, vocab]

        for i, pair in enumerate(batch):
            last_idx = seq_lens[i] - 1
            token_logits = logits[i, last_idx, :]

            pos_id = pair["pos_id"]
            neg_id = pair["neg_id"]
            margin = (token_logits[pos_id] - token_logits[neg_id]).item()
            margins.append(margin)

            # Prediction over A/B/C/D.
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
    max_samples: int | None = None,
    track_flips: bool = False,
) -> Tuple[torch.Tensor, float, float, float]:
    """Evaluate EMS statistics for a single column.

    Returns:
        drops: tensor of per-sample margin drops (baseline - crushed)
        mean_drop: mean(drops)
        flip_rate: fraction of samples where prediction != correct (if tracked)
        lethal_flip_rate: fraction where prediction == neg target (if tracked)
    """
    if not adv_pairs:
        return torch.zeros(0, dtype=torch.float32), 0.0, 0.0, 0.0

    if max_samples is not None:
        adv_pairs = adv_pairs[:max_samples]
        baseline_margins = baseline_margins[:max_samples]

    original_col = probe.simulate_column_crush(layer_idx, col_idx)

    crushed_margins: List[float] = []
    preds: List[int] = []
    processed = 0

    try:
        device = next(model.parameters()).device
        letter_token_ids = letter_token_ids.to(device)

        for start in range(0, len(adv_pairs), BATCH_SIZE):
            batch = adv_pairs[start : start + BATCH_SIZE]
            batch_baseline = baseline_margins[processed : processed + len(batch)]

            margins_batch, preds_batch = _batched_margins_and_predictions(
                model, batch, letter_token_ids, batch_size=BATCH_SIZE
            )

            for j, margin in enumerate(margins_batch.tolist()):
                crushed_margins.append(margin)
                if track_flips:
                    preds.append(preds_batch[j])

                processed += 1

                # Early stopping during Stage 1 EMS.
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
                    if mean_drop_so_far < early_stop_mu:
                        # Stop evaluating this column early.
                        break

            if early_stop_mu is not None and processed % 10 == 0:
                # We broke out of the inner loop; exit the outer batch loop too.
                base_so_far = baseline_margins[:processed]
                crushed_so_far = torch.tensor(
                    crushed_margins[:processed], dtype=torch.float32
                )
                mean_drop_so_far = float((base_so_far - crushed_so_far).mean().item())
                if mean_drop_so_far < early_stop_mu:
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
        for i, pred_idx in enumerate(preds):
            correct_idx = adv_pairs[i]["correct_letter_index"]
            neg_idx = adv_pairs[i]["neg_letter_index"]
            if pred_idx != correct_idx:
                flips += 1
                if pred_idx == neg_idx:
                    lethal_flips += 1
        flip_rate = flips / total if total > 0 else 0.0
        lethal_flip_rate = lethal_flips / total if total > 0 else 0.0

    return drops, mean_drop, flip_rate, lethal_flip_rate


def main():

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_name = "google/medgemma-1.5-4b-it"

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
    ).to(device)

    samples = load_mcq_dataset(n_total=80)
    calibration = samples[:40]
    evaluation = samples[40:80]

    adv_pairs = build_adversarial_pairs(
        model, tokenizer, calibration, n_calib=len(calibration)
    )

    probe = MedNSQProbe(model)

    layer_idx = 2

    print("Processing layer", layer_idx)

    # Stage 0: directional Taylor scores for all columns.
    jacobian_scores = probe.compute_contrastive_jacobian(layer_idx, adv_pairs)
    num_cols = jacobian_scores.numel()
    top_k = min(200, num_cols)
    top_vals, top_indices = torch.topk(jacobian_scores, k=top_k)

    # Baseline per-sample margins cached once.
    baseline_margins_all = probe.compute_per_sample_margins(adv_pairs)
    baseline_stage1 = baseline_margins_all[:STAGE1_SAMPLES]

    # Letter token ids for A/B/C/D predictions.
    letter_token_ids = _get_letter_token_ids(tokenizer)

    # Random baseline columns for Z-score calibration.
    all_indices = torch.arange(num_cols, device=jacobian_scores.device)
    rand_cols = all_indices[torch.randperm(num_cols)[:RANDOM_BASELINE_COLS]].tolist()

    random_mean_drops: List[float] = []
    for col in rand_cols:
        drops, mean_drop, _, _ = _evaluate_column_ems(
            model,
            probe,
            layer_idx,
            int(col),
            adv_pairs[:STAGE1_SAMPLES],
            baseline_stage1,
            letter_token_ids,
            early_stop_mu=None,
            max_samples=STAGE1_SAMPLES,
            track_flips=False,
        )
        random_mean_drops.append(mean_drop)

    if random_mean_drops:
        rand_tensor = torch.tensor(random_mean_drops, dtype=torch.float32)
        mu_rand = float(rand_tensor.mean().item())
        sigma_rand = float(rand_tensor.std(unbiased=False).item())
    else:
        mu_rand = 0.0
        sigma_rand = 0.0

    print(f"Random baseline mu_rand: {mu_rand:.6f}")
    print(f"Random baseline sigma_rand: {sigma_rand:.6f}")

    # Stage 1 EMS on top-k columns with early stopping.
    stage1_candidates: List[Tuple[int, float]] = []
    for col in top_indices.tolist():
        drops, mean_drop, _, _ = _evaluate_column_ems(
            model,
            probe,
            layer_idx,
            int(col),
            adv_pairs[:STAGE1_SAMPLES],
            baseline_stage1,
            letter_token_ids,
            early_stop_mu=mu_rand,
            max_samples=STAGE1_SAMPLES,
            track_flips=False,
        )
        if mean_drop > 0.0:
            stage1_candidates.append((int(col), mean_drop))

    # Keep top 20 by absolute drop magnitude.
    stage1_candidates.sort(key=lambda x: abs(x[1]), reverse=True)
    stage2_columns = [c for c, _ in stage1_candidates[:STAGE2_TOP_K]]

    print(f"Stage 1 retained {len(stage2_columns)} columns for Stage 2 EMS.")

    # Stage 2 EMS on full calibration set, with lethal flip tracking.
    validated_anchors = []
    for col in stage2_columns:
        drops, mean_drop, flip_rate, lethal_flip_rate = _evaluate_column_ems(
            model,
            probe,
            layer_idx,
            int(col),
            adv_pairs,
            baseline_margins_all,
            letter_token_ids,
            early_stop_mu=None,
            max_samples=None,
            track_flips=True,
        )

        z = (mean_drop - mu_rand) / (sigma_rand + 1e-8)
        print(
            f"Column {col}: mean_drop={mean_drop:.6f}, Z={z:.3f}, "
            f"flip_rate={flip_rate:.4f}, lethal_flip_rate={lethal_flip_rate:.4f}"
        )

        if z > Z_THRESHOLD:
            validated_anchors.append(
                {
                    "layer": layer_idx,
                    "column": int(col),
                    "mean_drop": mean_drop,
                    "z_score": z,
                    "flip_rate": flip_rate,
                    "lethal_flip_rate": lethal_flip_rate,
                }
            )

    print("\n=== Final EMS Summary ===")
    print(f"Random baseline mu_rand: {mu_rand:.6f}")
    print(f"Random baseline sigma_rand: {sigma_rand:.6f}")
    print(f"Validated Safety Anchors (Z > {Z_THRESHOLD}): {len(validated_anchors)}")

    for anchor in validated_anchors:
        print(
            f"[Safety Anchor] layer={anchor['layer']} "
            f"column={anchor['column']} "
            f"mean_drop={anchor['mean_drop']:.6f} "
            f"Z={anchor['z_score']:.3f} "
            f"flip_rate={anchor['flip_rate']:.4f} "
            f"lethal_flip_rate={anchor['lethal_flip_rate']:.4f}"
        )

    # Baseline evaluation on held-out samples.
    baseline_eval = evaluate_model(model, tokenizer, evaluation)

    print("\n=== Baseline Held-out Evaluation ===")
    print("Baseline accuracy:", baseline_eval["accuracy"])
    print("Baseline mean margin:", baseline_eval["mean_margin"])


if __name__ == "__main__":
    main()