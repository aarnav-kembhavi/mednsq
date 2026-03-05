import math
import random
from collections import Counter
from dataclasses import dataclass
from datetime import datetime
from typing import List, Dict, Any, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM

from mednsq_data import load_mcq_dataset, build_adversarial_pairs
from mednsq_eval import evaluate_model, _get_letter_token_ids
from mednsq_probe import MedNSQProbe
import json


# EMS configuration constants (defaults for next runs)
RANDOM_BASELINE_COLS = 20
Z_THRESHOLD = 2.0
BATCH_SIZE = 8


@dataclass
class EMSConfig:
    """Configuration for EMS experiments."""

    # Per the user-specified default experiment
    layer_idx: int = 2
    calibration_size: int = 120
    stage1_top_k: int = 40
    stage1_samples: int = 20
    stage2_top_k: int = 8
    stage2_samples: int = 200


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
    early_stop_sigma: float | None = None,
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

    # Crush a single column and keep a copy of its original weights.
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
                    threshold = early_stop_mu
                    if early_stop_sigma is not None:
                        threshold = early_stop_mu + 0.5 * early_stop_sigma
                    if mean_drop_so_far < threshold:
                        stop_early = True
                        break

            if stop_early:
                break

    finally:
        # Restore the crushed column and sanity-check restoration.
        probe.restore_column(layer_idx, col_idx, original_col)
        restored_col = probe.get_layer_weight(layer_idx)[:, col_idx].detach().to(
            original_col.dtype
        )
        max_diff = (restored_col - original_col).abs().max().item()
        if max_diff > 1e-3:
            print(
                f"[Warning] Column restoration check failed for "
                f"layer={layer_idx}, column={col_idx}, max_diff={max_diff:.6e}"
            )

    if not crushed_margins:
        return torch.zeros(0, dtype=torch.float32), 0.0, 0.0, 0.0

    crushed_tensor = torch.tensor(crushed_margins, dtype=torch.float32)
    drops = baseline_margins[: len(crushed_tensor)] - crushed_tensor
    mean_drop = float(drops.mean().item())

    flip_rate = 0.0
    lethal_flip_rate = 0.0

    if track_flips and preds:
        # Flip tracking is computed per column with fresh counters.
        total = len(preds)
        flips = 0
        lethal_flips = 0

        # We compare predictions against the positive / negative tokens.
        # This matches the desired per-sample EMS flip semantics.
        letter_token_ids_cpu = letter_token_ids.detach().cpu()

        debug_limit = 0  # Disable per-sample debug printing.
        printed_debug = 0

        for i, pred_idx in enumerate(preds):
            pair = adv_pairs[i]
            correct_token = int(pair["pos_id"])
            neg_token = int(pair["neg_id"])
            pred_token = int(letter_token_ids_cpu[pred_idx].item())

            if pred_token != correct_token:
                flips += 1
                if pred_token == neg_token:
                    lethal_flips += 1

            # Optional lightweight debug trace for the first few samples.
            if printed_debug < debug_limit:
                print(
                    f"[FlipDebug] layer={layer_idx} column={col_idx} "
                    f"sample={i} pred_token={pred_token} "
                    f"correct_token={correct_token} neg_token={neg_token}"
                )
                printed_debug += 1

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
) -> Dict[str, Any]:
    """Run the full EMS pipeline (Taylor → Stage1 → Stage2 → Z) for one layer."""

    print(f"\n=== Processing layer {layer_idx} ===")

    # Stage 0: directional Taylor scores for all columns.
    jacobian_scores = probe.compute_contrastive_jacobian(layer_idx, adv_pairs)
    num_cols = jacobian_scores.numel()
    print(f"[ProbeCheck] layer={layer_idx} num_cols={num_cols}")

    stage1_top_k = min(cfg.stage1_top_k, num_cols)
    top_vals, top_indices = torch.topk(jacobian_scores, k=stage1_top_k)
    print("[Taylor Top20]", top_vals[:20].detach().cpu().tolist())

    # Baseline per-sample margins cached once.
    baseline_margins_all = probe.compute_per_sample_margins(adv_pairs)

    stage1_samples = min(cfg.stage1_samples, len(adv_pairs))
    stage2_samples = min(cfg.stage2_samples, len(adv_pairs))

    baseline_stage1 = baseline_margins_all[:stage1_samples]

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
            adv_pairs[:stage1_samples],
            baseline_stage1,
            letter_token_ids,
            early_stop_mu=None,
            max_samples=stage1_samples,
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
    if sigma_rand > 0.05:
        print(
            f"[Warning] sigma_rand={sigma_rand:.6f} is unusually large; "
            f"Z-scores may be unstable."
        )

    # Stage 1 EMS on top-k columns with early stopping.
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
        )
        if mean_drop > 0.0:
            stage1_candidates.append((int(col), mean_drop))

    # Keep top-k Stage 2 candidates by mean drop (positive = safety-relevant).
    stage1_candidates.sort(key=lambda x: x[1], reverse=True)
    stage2_columns = [c for c, _ in stage1_candidates[: cfg.stage2_top_k]]

    print(
        f"Stage 1 retained {len(stage2_columns)} columns for Stage 2 EMS "
        f"(top_k={cfg.stage2_top_k})."
    )

    # Stage 2 EMS on (possibly truncated) calibration set, with lethal flip tracking.
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
        )

        z = (mean_drop - mu_rand) / (sigma_rand + 1e-8)
        median_drop = float(torch.median(drops).item())

        result = {
            "layer": layer_idx,
            "column": int(col),
            "mean_drop": mean_drop,
            "median_drop": median_drop,
            "z_score": z,
            "flip_rate": flip_rate,
            "lethal_flip_rate": lethal_flip_rate,
        }
        stage2_results.append(result)

        print(
            "[Stage2] "
            f"layer={layer_idx} column={col} "
            f"drop={mean_drop:.6f} median_drop={median_drop:.6f} Z={z:.3f} "
            f"flip={flip_rate:.4f} lethal={lethal_flip_rate:.4f}"
        )

        if z > Z_THRESHOLD:
            validated_anchors.append(result)

    # Sanity checks over Stage 2 flip statistics.
    if stage2_results:
        # Identical flip_rate across many columns.
        flip_values = [round(r["flip_rate"], 6) for r in stage2_results]
        counts = Counter(flip_values)
        for val, count in counts.items():
            if count > 5:
                print(
                    f"[Warning] flip_rate={val:.4f} is identical across "
                    f"{count} columns in layer {layer_idx}; "
                    f"check flip tracking."
                )
                break

        # Lethal flip rate equal to flip rate for most columns.
        same_count = sum(
            abs(r["flip_rate"] - r["lethal_flip_rate"]) < 1e-6
            for r in stage2_results
        )
        if same_count > 0.8 * len(stage2_results):
            print(
                f"[Warning] lethal_flip_rate equals flip_rate for "
                f"{same_count}/{len(stage2_results)} columns in layer {layer_idx}."
            )

    # Per-layer summary for validated anchors.
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

    print("\n=== Layer Summary ===")
    print(f"layer={layer_idx}")
    print(f"validated_anchors={len(validated_anchors)}")
    print(f"max_drop={max_drop:.6f}")
    print(f"mean_drop={mean_drop_validated:.6f}")
    print(f"max_Z={max_z:.3f}")

    # Baseline evaluation on held-out samples (per layer for convenience).
    baseline_eval = evaluate_model(model, tokenizer, evaluation_samples)
    print("\n=== Baseline Held-out Evaluation ===")
    print("Baseline accuracy:", baseline_eval["accuracy"])
    print("Baseline mean margin:", baseline_eval["mean_margin"])

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


def run_multi_anchor_ablation_sweep(
    model,
    tokenizer,
    anchors: List[Tuple[int, int]],
    seeds: List[int],
    calibration_size: int,
    eval_size: int,
) -> None:
    """Run multi-anchor ablations across multiple dataset seeds and average results."""

    print("\n=== Multi Anchor Ablation Sweep ===")

    ablation_counts = [1, 4, 8, 12, 16]
    # Store per-ablation metrics across seeds.
    sweep_results: Dict[int, Dict[str, List[float]]] = {
        k: {"accuracies": [], "margins": []} for k in ablation_counts
    }

    for seed in seeds:
        torch.manual_seed(seed)
        random.seed(seed)
        np.random.seed(seed)

        total_needed = calibration_size + eval_size
        samples = load_mcq_dataset(n_total=total_needed)
        random.shuffle(samples)

        calibration = samples[:calibration_size]
        evaluation = samples[calibration_size : calibration_size + eval_size]

        adv_pairs = build_adversarial_pairs(
            model, tokenizer, calibration, n_calib=len(calibration)
        )

        probe = MedNSQProbe(model)

        for count in ablation_counts:
            if count > len(anchors):
                continue

            subset = anchors[:count]
            saved_columns: List[Tuple[int, int, torch.Tensor]] = []

            try:
                # Crush each selected anchor.
                for layer_idx, col_idx in subset:
                    original_col = probe.simulate_column_crush(layer_idx, col_idx)
                    saved_columns.append((layer_idx, col_idx, original_col))

                # Evaluate accuracy under the joint crush.
                eval_metrics = evaluate_model(model, tokenizer, evaluation)
                accuracy = float(eval_metrics.get("accuracy", 0.0))

                # Compute mean margin on calibration adversarial pairs.
                margins = probe.compute_per_sample_margins(adv_pairs)
                mean_margin = (
                    float(margins.mean().item()) if margins.numel() > 0 else 0.0
                )

                sweep_results[count]["accuracies"].append(accuracy)
                sweep_results[count]["margins"].append(mean_margin)

            finally:
                # Restore all crushed columns to their original values.
                for layer_idx, col_idx, original_col in saved_columns:
                    probe.restore_column(layer_idx, col_idx, original_col)

        del probe
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Aggregate and print averaged results across seeds.
    for count in ablation_counts:
        accs = sweep_results[count]["accuracies"]
        margins = sweep_results[count]["margins"]
        if not accs:
            continue

        mean_acc = sum(accs) / len(accs)
        mean_margin = sum(margins) / len(margins) if margins else 0.0

        print(f"\nAnchors removed: {count}")
        print(f"Mean accuracy: {mean_acc}")
        print(f"Mean margin: {mean_margin}")


def run_random_neuron_ablation_baseline(
    model,
    tokenizer,
    layers: List[int],
    seeds: List[int],
    calibration_size: int,
    eval_size: int,
    k_values: List[int],
) -> None:
    """Run multi-neuron ablation with randomly sampled neurons; average over seeds."""

    N_COLS = 10240  # columns per layer (~10240 for MedGemma MLP down_proj)

    print("\n=== Random Neuron Ablation Baseline ===")

    sweep_results: Dict[int, Dict[str, List[float]]] = {
        k: {"accuracies": [], "margins": []} for k in k_values
    }

    for seed in seeds:
        torch.manual_seed(seed)
        random.seed(seed)
        np.random.seed(seed)

        total_needed = calibration_size + eval_size
        samples = load_mcq_dataset(n_total=total_needed)
        random.shuffle(samples)

        calibration = samples[:calibration_size]
        evaluation = samples[calibration_size : calibration_size + eval_size]

        adv_pairs = build_adversarial_pairs(
            model, tokenizer, calibration, n_calib=len(calibration)
        )

        probe = MedNSQProbe(model)

        for k in k_values:
            # Sample k unique random (layer, column) neurons.
            random_neurons_set: set = set()
            while len(random_neurons_set) < k:
                random_neurons_set.add(
                    (random.choice(layers), random.randint(0, N_COLS - 1))
                )
            random_neurons = list(random_neurons_set)
            saved_columns: List[Tuple[int, int, torch.Tensor]] = []

            try:
                for layer_idx, col_idx in random_neurons:
                    original_col = probe.simulate_column_crush(layer_idx, col_idx)
                    saved_columns.append((layer_idx, col_idx, original_col))

                eval_metrics = evaluate_model(model, tokenizer, evaluation)
                accuracy = float(eval_metrics.get("accuracy", 0.0))

                margins = probe.compute_per_sample_margins(adv_pairs)
                mean_margin = (
                    float(margins.mean().item()) if margins.numel() > 0 else 0.0
                )

                sweep_results[k]["accuracies"].append(accuracy)
                sweep_results[k]["margins"].append(mean_margin)

            finally:
                for layer_idx, col_idx, original_col in saved_columns:
                    probe.restore_column(layer_idx, col_idx, original_col)

        del probe
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    for k in k_values:
        accs = sweep_results[k]["accuracies"]
        margins = sweep_results[k]["margins"]
        if not accs:
            continue
        mean_acc = sum(accs) / len(accs)
        mean_margin = sum(margins) / len(margins) if margins else 0.0
        print(f"\nRandom neurons removed: {k}")
        print(f"Mean accuracy: {mean_acc}")
        print(f"Mean margin: {mean_margin}")


def _run_forward_no_hooks(model, input_ids, attention_mask):
    """Single forward pass with no hooks; returns logits at last position."""
    with torch.no_grad():
        out = model(input_ids=input_ids, attention_mask=attention_mask)
    return out.logits[:, -1, :]


def run_anchor_activation_patching_experiment(
    model,
    tokenizer,
    anchors: List[Tuple[int, int]],
    calibration_samples: List[Dict[str, Any]],
    n_samples: int = 120,
    per_anchor_report_only: bool = False,
) -> None:
    """Counterfactual activation patching on anchor neurons: patch safe-run activations
    into adversarial runs and measure margin/accuracy change. Includes random-neuron control.
    When per_anchor_report_only=True, only runs anchor patching and prints a single-line report
    (for per-anchor causal strength ranking).
    """
    model.eval()
    device = next(model.parameters()).device
    letter_token_ids = _get_letter_token_ids(tokenizer).to(device)

    backbone = getattr(model, "model", model)
    if hasattr(backbone, "language_model"):
        backbone = backbone.language_model
    layer_stack = backbone.layers
    # down_proj input h has shape [batch, seq, in_features]; EMS anchors are columns in in_features
    intermediate_size = layer_stack[0].mlp.down_proj.weight.shape[1]
    pad_token_id = getattr(tokenizer, "pad_token_id", None) or getattr(tokenizer, "eos_token_id", 0)

    adv_pairs = build_adversarial_pairs(
        model, tokenizer, calibration_samples, n_calib=len(calibration_samples)
    )
    n_samples = min(n_samples, len(adv_pairs))
    if n_samples == 0:
        print("No samples for anchor activation patching.")
        return

    # Group anchors by layer for hooks
    anchors_by_layer: Dict[int, List[int]] = {}
    for (layer_idx, col_idx) in anchors:
        if layer_idx not in anchors_by_layer:
            anchors_by_layer[layer_idx] = []
        anchors_by_layer[layer_idx].append(col_idx)

    def _left_pad_to_length(ids: torch.Tensor, mask: torch.Tensor, target_len: int):
        """Left-pad so last position is the last real token; same length for alignment."""
        seq_len = ids.shape[1]
        if seq_len >= target_len:
            return ids.to(device), mask.to(device)
        batch = ids.shape[0]
        new_ids = torch.full((batch, target_len), pad_token_id, dtype=ids.dtype, device=device)
        new_mask = torch.zeros(batch, target_len, dtype=mask.dtype, device=device)
        new_ids[:, -seq_len:] = ids.to(device)
        new_mask[:, -seq_len:] = mask.to(device)
        return new_ids, new_mask

    def _run_with_save_hooks(stored: Dict[Tuple[int, int], torch.Tensor], input_ids, attention_mask):
        handles = []
        for layer_idx, cols in anchors_by_layer.items():
            down_proj = layer_stack[layer_idx].mlp.down_proj

            def _save_hook(module, input, layer=layer_idx, cols_list=cols):
                h = input[0]
                for col in cols_list:
                    stored[(layer, col)] = h[0, -1, col].detach().clone()

            handles.append(down_proj.register_forward_pre_hook(_save_hook))

        with torch.no_grad():
            out = model(input_ids=input_ids, attention_mask=attention_mask)
        for h in handles:
            h.remove()
        return out.logits[:, -1, :]

    def _run_with_patch_hooks(stored: Dict[Tuple[int, int], torch.Tensor], input_ids, attention_mask):
        handles = []
        for layer_idx, cols in anchors_by_layer.items():
            down_proj = layer_stack[layer_idx].mlp.down_proj

            def _patch_hook(module, input, layer=layer_idx, cols_list=cols):
                h = input[0]
                out = h.clone()
                for col in cols_list:
                    if (layer, col) in stored:
                        stored_val = stored[(layer, col)].to(out.dtype).to(out.device)
                        out[0, -1, col] = stored_val
                return (out,)

            handles.append(down_proj.register_forward_pre_hook(_patch_hook))

        with torch.no_grad():
            out = model(input_ids=input_ids, attention_mask=attention_mask)
        for h in handles:
            h.remove()
        return out.logits[:, -1, :]

    def _margin_and_correct(logits, pair):
        pos_id = pair["pos_id"]
        neg_id = pair["neg_id"]
        margin = (logits[0, pos_id] - logits[0, neg_id]).item()
        letter_logits = logits[0, letter_token_ids]
        pred_idx = torch.argmax(letter_logits).item()
        pred_token = letter_token_ids[pred_idx].item()
        correct = 1 if pred_token == pos_id else 0
        return margin, correct

    # ---- Anchor patching ----
    margin_shifts_anchor = []
    correct_baseline_anchor = 0
    correct_patched_anchor = 0

    with torch.no_grad():
        for i in range(n_samples):
            pair = adv_pairs[i]
            safe_ids = pair["safe_input_ids"].to(device)
            safe_mask = pair["safe_attention_mask"].to(device)
            corrupt_ids = pair["input_ids"].to(device)
            corrupt_mask = pair["attention_mask"].to(device)
            target_len = max(safe_ids.shape[1], corrupt_ids.shape[1])
            safe_ids_pad, safe_mask_pad = _left_pad_to_length(safe_ids, safe_mask, target_len)
            corrupt_ids_pad, corrupt_mask_pad = _left_pad_to_length(corrupt_ids, corrupt_mask, target_len)

            stored_activations: Dict[Tuple[int, int], torch.Tensor] = {}
            _run_with_save_hooks(stored_activations, safe_ids_pad, safe_mask_pad)

            logits_baseline = _run_forward_no_hooks(model, corrupt_ids_pad, corrupt_mask_pad)
            baseline_margin, correct_base = _margin_and_correct(logits_baseline, pair)

            logits_patched = _run_with_patch_hooks(stored_activations, corrupt_ids_pad, corrupt_mask_pad)
            patched_margin, correct_patch = _margin_and_correct(logits_patched, pair)

            margin_shifts_anchor.append(patched_margin - baseline_margin)
            correct_baseline_anchor += correct_base
            correct_patched_anchor += correct_patch

    mean_margin_shift_anchor = float(np.mean(margin_shifts_anchor))
    acc_baseline_anchor = correct_baseline_anchor / n_samples
    acc_patched_anchor = correct_patched_anchor / n_samples
    accuracy_change_anchor = acc_patched_anchor - acc_baseline_anchor

    if per_anchor_report_only:
        layer_idx, col_idx = anchors[0]
        print(f"Anchor (layer={layer_idx} col={col_idx})")
        print(f"Mean margin shift: {mean_margin_shift_anchor}")
        return

    print("\n=== Anchor Activation Patching ===")
    print(f"Mean margin shift: {mean_margin_shift_anchor}")
    print(f"Accuracy change: {accuracy_change_anchor}")

    # ---- Random neuron control ----
    anchor_layers = list(anchors_by_layer.keys())
    n_random = len(anchors)
    random_neurons_set: set = set()
    while len(random_neurons_set) < n_random:
        random_neurons_set.add((random.choice(anchor_layers), random.randint(0, intermediate_size - 1)))
    random_neurons = list(random_neurons_set)
    random_by_layer: Dict[int, List[int]] = {}
    for (layer_idx, col_idx) in random_neurons:
        if layer_idx not in random_by_layer:
            random_by_layer[layer_idx] = []
        random_by_layer[layer_idx].append(col_idx)

    def _run_with_save_hooks_random(stored, input_ids, attention_mask):
        handles = []
        for layer_idx, cols in random_by_layer.items():
            down_proj = layer_stack[layer_idx].mlp.down_proj

            def _save_hook(module, input, layer=layer_idx, cols_list=cols):
                h = input[0]
                for col in cols_list:
                    stored[(layer, col)] = h[0, -1, col].detach().clone()

            handles.append(down_proj.register_forward_pre_hook(_save_hook))
        with torch.no_grad():
            model(input_ids=input_ids, attention_mask=attention_mask)
        for h in handles:
            h.remove()

    def _run_with_patch_hooks_random(stored, input_ids, attention_mask):
        handles = []
        for layer_idx, cols in random_by_layer.items():
            down_proj = layer_stack[layer_idx].mlp.down_proj

            def _patch_hook(module, input, layer=layer_idx, cols_list=cols):
                h = input[0]
                out = h.clone()
                for col in cols_list:
                    if (layer, col) in stored:
                        stored_val = stored[(layer, col)].to(out.dtype).to(out.device)
                        out[0, -1, col] = stored_val
                return (out,)

            handles.append(down_proj.register_forward_pre_hook(_patch_hook))
        with torch.no_grad():
            out = model(input_ids=input_ids, attention_mask=attention_mask)
        for h in handles:
            h.remove()
        return out.logits[:, -1, :]

    margin_shifts_random = []
    correct_baseline_random = 0
    correct_patched_random = 0

    with torch.no_grad():
        for i in range(n_samples):
            pair = adv_pairs[i]
            safe_ids = pair["safe_input_ids"].to(device)
            safe_mask = pair["safe_attention_mask"].to(device)
            corrupt_ids = pair["input_ids"].to(device)
            corrupt_mask = pair["attention_mask"].to(device)
            target_len = max(safe_ids.shape[1], corrupt_ids.shape[1])
            safe_ids_pad, safe_mask_pad = _left_pad_to_length(safe_ids, safe_mask, target_len)
            corrupt_ids_pad, corrupt_mask_pad = _left_pad_to_length(corrupt_ids, corrupt_mask, target_len)

            stored_random: Dict[Tuple[int, int], torch.Tensor] = {}
            _run_with_save_hooks_random(stored_random, safe_ids_pad, safe_mask_pad)

            logits_baseline = _run_forward_no_hooks(model, corrupt_ids_pad, corrupt_mask_pad)
            baseline_margin, correct_base = _margin_and_correct(logits_baseline, pair)

            logits_patched = _run_with_patch_hooks_random(stored_random, corrupt_ids_pad, corrupt_mask_pad)
            patched_margin, correct_patch = _margin_and_correct(logits_patched, pair)

            margin_shifts_random.append(patched_margin - baseline_margin)
            correct_baseline_random += correct_base
            correct_patched_random += correct_patch

    mean_margin_shift_random = float(np.mean(margin_shifts_random))
    acc_baseline_random = correct_baseline_random / n_samples
    acc_patched_random = correct_patched_random / n_samples
    accuracy_change_random = acc_patched_random - acc_baseline_random

    print("\n=== Random Neuron Patching ===")
    print(f"Mean margin shift: {mean_margin_shift_random}")
    print(f"Accuracy change: {accuracy_change_random}")


def run_attention_head_ablation_sweep(
    model,
    tokenizer,
    layers: List[int],
    seeds: List[int],
    calibration_size: int,
    eval_size: int,
    k_values: List[int],
) -> None:
    """Run attention head ablation sweep; average results over seeds."""

    print("\n=== Attention Head Ablation Sweep ===")

    backbone = getattr(model, "model", model)
    if hasattr(backbone, "language_model"):
        backbone = backbone.language_model
    layer_stack = backbone.layers

    sample_layer = layer_stack[layers[0]]

    o_proj_weight = sample_layer.self_attn.o_proj.weight
    hidden_size = o_proj_weight.shape[0]

    # assume standard head size of 128 for Gemma
    head_dim = 128
    num_heads = hidden_size // head_dim

    sweep_results: Dict[int, Dict[str, List[float]]] = {
        k: {"accuracies": [], "margins": []} for k in k_values
    }

    for seed in seeds:
        torch.manual_seed(seed)
        random.seed(seed)
        np.random.seed(seed)

        total_needed = calibration_size + eval_size
        samples = load_mcq_dataset(n_total=total_needed)
        random.shuffle(samples)

        calibration = samples[:calibration_size]
        evaluation = samples[calibration_size : calibration_size + eval_size]

        adv_pairs = build_adversarial_pairs(
            model, tokenizer, calibration, n_calib=len(calibration)
        )

        probe = MedNSQProbe(model)

        all_heads = [
            (layer, head) for layer in layers for head in range(num_heads)
        ]

        for k in k_values:
            if k > len(all_heads):
                continue
            random_heads = random.sample(all_heads, k)
            saved_slices: List[Tuple[int, int, torch.Tensor]] = []

            try:
                with torch.no_grad():
                    for layer_idx, head_idx in random_heads:
                        layer = layer_stack[layer_idx]
                        weight = layer.self_attn.o_proj.weight
                        start = head_idx * head_dim
                        end = (head_idx + 1) * head_dim
                        saved_slice = weight[:, start:end].clone()
                        saved_slices.append((layer_idx, head_idx, saved_slice))
                        weight[:, start:end] = 0

                eval_metrics = evaluate_model(model, tokenizer, evaluation)
                accuracy = float(eval_metrics.get("accuracy", 0.0))

                margins_ablated = probe.compute_per_sample_margins(adv_pairs)
                mean_margin = (
                    float(margins_ablated.mean().item())
                    if margins_ablated.numel() > 0
                    else 0.0
                )

                sweep_results[k]["accuracies"].append(accuracy)
                sweep_results[k]["margins"].append(mean_margin)

            finally:
                with torch.no_grad():
                    for layer_idx, head_idx, saved_slice in saved_slices:
                        layer = layer_stack[layer_idx]
                        weight = layer.self_attn.o_proj.weight
                        start = head_idx * head_dim
                        end = (head_idx + 1) * head_dim
                        weight[:, start:end] = saved_slice.to(
                            weight.dtype
                        ).to(weight.device)

        del probe
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    for k in k_values:
        accs = sweep_results[k]["accuracies"]
        margins = sweep_results[k]["margins"]
        if not accs:
            continue
        mean_acc = sum(accs) / len(accs)
        mean_margin = sum(margins) / len(margins) if margins else 0.0
        print(f"\nHeads removed: {k}")
        print(f"Mean accuracy: {mean_acc}")
        print(f"Mean margin: {mean_margin}")


def main():
    # Use device map auto-loading; seeds are handled per-run below.
    _ = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    RUN_HEAD_ONLY = True

    model_name = "google/medgemma-1.5-4b-it"
    layers_to_test = [8, 16]

    # Default experiment configuration (can be adjusted as needed).
    cfg = EMSConfig()

    print("=== Experiment Configuration ===")
    print("Calibration size:", cfg.calibration_size)
    print("Stage1 top_k:", cfg.stage1_top_k)
    print("Stage1 samples:", cfg.stage1_samples)
    print("Stage2 top_k:", cfg.stage2_top_k)
    print("Stage2 samples:", cfg.stage2_samples)
    print("Random baseline cols:", RANDOM_BASELINE_COLS)
    print("===============================")

    # Load tokenizer once.
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    if RUN_HEAD_ONLY:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )
        print("=== Running Anchor Activation Patching Experiment ===")
        calibration_samples = load_mcq_dataset(n_total=120)
        anchors = [
            (16, 1761), (16, 2056), (16, 3433), (16, 3442), (16, 1471),
            (16, 3647), (16, 7918), (16, 6047), (16, 4273), (16, 6845),
            (8, 3977), (8, 2990), (8, 5386), (8, 3347), (8, 2069),
            (8, 1577),
        ]
        run_anchor_activation_patching_experiment(
            model=model,
            tokenizer=tokenizer,
            anchors=anchors,
            calibration_samples=calibration_samples,
            n_samples=120,
        )
        print("\n=== Per-Anchor Causal Strength ===")
        for anchor in anchors:
            run_anchor_activation_patching_experiment(
                model=model,
                tokenizer=tokenizer,
                anchors=[anchor],
                calibration_samples=calibration_samples,
                n_samples=120,
                per_anchor_report_only=True,
            )
    else:
        # Sweep over multiple random seeds to test anchor stability.
        seeds = [1, 2, 3]
        all_seed_results: List[Dict[str, Any]] = []

        for seed in seeds:
            print(f"\n===== RUNNING SEED {seed} =====")
            print(f"Calibration size: {cfg.calibration_size}")

            # Reproducibility: set experiment seeds for this run.
            torch.manual_seed(seed)
            random.seed(seed)
            np.random.seed(seed)

            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.bfloat16,
                device_map="auto",
            )

            # Dataset split: configurable calibration size, fixed held-out evaluation size.
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

            all_layer_summaries: List[Dict[str, Any]] = []
            for layer_idx in layers_to_test:
                summary = _run_ems_for_layer(
                    model,
                    tokenizer,
                    probe,
                    adv_pairs,
                    evaluation,
                    layer_idx,
                    cfg,
                )
                all_layer_summaries.append(summary)

            # Compact per-seed summary.
            print("\n=== Seed Summary ===")
            print(f"seed={seed}")
            for layer_idx in layers_to_test:
                layer_summary = next(
                    s for s in all_layer_summaries if s["layer"] == layer_idx
                )
                num_anchors = len(layer_summary["validated_anchors"])
                print(f"layer{layer_idx} anchors={num_anchors}")

            # Store per-seed results.
            seed_summary = {
                "seed": seed,
                "layers": all_layer_summaries,
            }
            all_seed_results.append(seed_summary)

            # Clear per-seed probe and model; next seed gets a fresh model.
            del probe
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        # Load model once for ablation experiments (no weight modifications during ablation per seed).
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )

        # Final EMS summary across all seeds is stored in all_seed_results.
        print("\n=== Anchor Stability Summary ===")
        for layer_idx in layers_to_test:
            counts: List[int] = []
            for seed_result in all_seed_results:
                layer_summary = next(
                    s for s in seed_result["layers"] if s["layer"] == layer_idx
                )
                counts.append(len(layer_summary["validated_anchors"]))
            mean_anchors = sum(counts) / len(counts) if counts else 0.0
            print(f"layer{layer_idx} mean anchors = {mean_anchors:.3f}")

        # Save experiment output to JSON for reproducibility.
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        with open(f"ems_seed_sweep_{timestamp}.json", "w") as f:
            json.dump(all_seed_results, f, indent=2)

        print("\n=== Using Manually Selected Anchors For Ablation ===")

        anchors = [
            (16, 1761), (16, 2056), (16, 3433), (16, 3442), (16, 1471),
            (16, 3647), (16, 7918), (16, 6047), (16, 4273), (16, 6845),
            (8, 3977), (8, 2990), (8, 5386), (8, 3347), (8, 2069),
            (8, 1577),
        ]

        for (layer, column) in anchors:
            print(f"(layer={layer}, column={column})")

        print("=== Running Anchor Activation Patching Experiment ===")
        calibration_samples = load_mcq_dataset(n_total=120)
        run_anchor_activation_patching_experiment(
            model=model,
            tokenizer=tokenizer,
            anchors=anchors,
            calibration_samples=calibration_samples,
            n_samples=120,
        )
        print("\n=== Per-Anchor Causal Strength ===")
        for anchor in anchors:
            run_anchor_activation_patching_experiment(
                model=model,
                tokenizer=tokenizer,
                anchors=[anchor],
                calibration_samples=calibration_samples,
                n_samples=120,
                per_anchor_report_only=True,
            )

        run_multi_anchor_ablation_sweep(
            model=model,
            tokenizer=tokenizer,
            anchors=anchors,
            seeds=[1, 2, 3, 4, 5],
            calibration_size=cfg.calibration_size,
            eval_size=60,
        )

        run_random_neuron_ablation_baseline(
            model=model,
            tokenizer=tokenizer,
            layers=[2, 8, 16, 24],
            seeds=[1, 2, 3, 4, 5],
            calibration_size=cfg.calibration_size,
            eval_size=60,
            k_values=[1, 4, 8, 12, 16],
        )

        run_attention_head_ablation_sweep(
            model=model,
            tokenizer=tokenizer,
            layers=[8, 16],
            seeds=[1, 2, 3, 4, 5],
            calibration_size=cfg.calibration_size,
            eval_size=60,
            k_values=[1, 2, 4, 8],
        )


if __name__ == "__main__":
    main()