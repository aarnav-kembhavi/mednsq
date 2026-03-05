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
RANDOM_BASELINE_COLS = 40
Z_THRESHOLD = 2.0
BATCH_SIZE = 8


@dataclass
class EMSConfig:
    """Configuration for EMS experiments."""

    # Per the user-specified default experiment
    layer_idx: int = 2
    calibration_size: int = 200
    stage1_top_k: int = 80
    stage1_samples: int = 30
    stage2_top_k: int = 12
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

    ablation_counts = [1, 3, 5]
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


def main():
    # Use device map auto-loading; seeds are handled per-run below.
    _ = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_name = "google/medgemma-1.5-4b-it"
    layers_to_test = [8, 16]

    # Default experiment configuration (can be adjusted as needed).
    cfg = EMSConfig()

    # Sweep over multiple random seeds to test anchor stability.
    seeds = [1, 2, 3]
    all_seed_results: List[Dict[str, Any]] = []

    # Load tokenizer once; model is loaded fresh per seed for EMS discovery.
    tokenizer = AutoTokenizer.from_pretrained(model_name)

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
        eval_size = 40
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

    # Anchor frequency: (layer, column) -> count across seeds.
    anchor_freq: Dict[Tuple[int, int], int] = {}
    num_seeds = len(all_seed_results)
    for seed_result in all_seed_results:
        for layer_summary in seed_result["layers"]:
            for anchor in layer_summary["validated_anchors"]:
                key = (anchor["layer"], anchor["column"])
                anchor_freq[key] = anchor_freq.get(key, 0) + 1
    print("\n=== Anchor Frequency ===")
    for (layer, column), count in sorted(
        anchor_freq.items(), key=lambda x: (-x[1], x[0][0], x[0][1])
    ):
        print(f"(layer={layer}, column={column}) appeared in {count}/{num_seeds} seeds")

    # Select anchors by majority rule: appeared in more than half of seeds.
    min_count = len(seeds) // 2 + 1
    anchors = [
        (layer, column)
        for (layer, column), count in anchor_freq.items()
        if count >= min_count
    ]
    anchors = sorted(anchors, key=lambda x: anchor_freq[x], reverse=True)

    print("\n=== Anchors Selected For Ablation ===")
    for (layer, column) in anchors:
        print(f"(layer={layer}, column={column}) freq={anchor_freq[(layer, column)]}")

    run_multi_anchor_ablation_sweep(
        model=model,
        tokenizer=tokenizer,
        anchors=anchors,
        seeds=[1, 2, 3, 4, 5],
        calibration_size=cfg.calibration_size,
        eval_size=40,
    )

    run_random_neuron_ablation_baseline(
        model=model,
        tokenizer=tokenizer,
        layers=[2, 8, 16, 24],
        seeds=[1, 2, 3, 4, 5],
        calibration_size=cfg.calibration_size,
        eval_size=40,
        k_values=[1, 3, 5],
    )


if __name__ == "__main__":
    main()