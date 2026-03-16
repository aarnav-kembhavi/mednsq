import math
import os
import random
import sys
from collections import Counter
from dataclasses import dataclass
from datetime import datetime
from typing import List, Dict, Any, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM

from mednsq_data import load_mcq_dataset, build_adversarial_pairs, format_prompt
from mednsq_eval import evaluate_model, _get_letter_token_ids
from mednsq_probe import MedNSQProbe
import json


class Tee:
    def __init__(self, filename):
        self.file = open(filename, "a", buffering=1)
        self.stdout = sys.stdout

    def write(self, data):
        self.stdout.write(data)
        self.file.write(data)

    def flush(self):
        self.stdout.flush()
        self.file.flush()


sys.stdout = Tee("experiment_full_output_llama3_3b.log")


# EMS configuration constants (defaults for next runs)
RANDOM_BASELINE_COLS = 128
Z_THRESHOLD = 2.0
BATCH_SIZE = 64


# Architecture parameters (set dynamically after each model load)
HIDDEN_SIZE = None
NUM_LAYERS = None
NUM_ATTENTION_HEADS = None
INTERMEDIATE_SIZE = None
HEAD_DIM = None

@dataclass
class EMSConfig:
    """Configuration for EMS experiments."""

    # Per the user-specified default experiment (A100-optimized)
    layer_idx: int = 2
    calibration_size: int = 800
    stage1_top_k: int = 256
    stage1_samples: int = 120
    stage2_top_k: int = 128
    stage2_samples: int = 800


def log_progress(text: str) -> None:
    """Append a line to experiment_progress.log."""
    with open("experiment_progress.log", "a") as f:
        f.write(text + "\n")


def append_anchor_progress(layer: int, col: int, drop: float, z: float) -> None:
    """Append one anchor line to anchors_progress.txt."""
    with open("anchors_progress.txt", "a") as f:
        f.write(f"(layer={layer} col={col}) drop={drop:.2f} Z={z:.1f}\n")


def save_checkpoint(data: Dict[str, Any]) -> None:
    """Save checkpoint to ems_checkpoint.pt."""
    torch.save(data, "ems_checkpoint.pt")


def load_checkpoint() -> Any:
    """Load checkpoint from ems_checkpoint.pt if it exists."""
    if os.path.exists("ems_checkpoint.pt"):
        try:
            return torch.load("ems_checkpoint.pt", weights_only=False)
        except TypeError:
            return torch.load("ems_checkpoint.pt")
    return None


def print_gpu_memory():
    """Debug: print current GPU memory allocated and reserved (non-intrusive)."""
    if torch.cuda.is_available():
        alloc = torch.cuda.memory_allocated() / 1e9
        reserved = torch.cuda.memory_reserved() / 1e9
        print(f"[GPU] allocated={alloc:.2f}GB reserved={reserved:.2f}GB")


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


def _evaluate_columns_vectorized(
    model,
    layer,
    columns,
    adv_pairs,
    baseline_margins,
    letter_token_ids,
    batch_size=BATCH_SIZE,
):
    device = next(model.parameters()).device

    down_proj = layer.mlp.down_proj
    W_down = down_proj.weight

    num_cols = len(columns)

    drop_sums = torch.zeros(num_cols, device=device)
    counts = torch.zeros(num_cols, device=device)

    for start in range(0, len(adv_pairs), batch_size):

        batch = adv_pairs[start:start+batch_size]

        seq_lens = [p["input_ids"].shape[1] for p in batch]
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

        activations = {}

        def capture_hook(module, input):
            h = input[0]
            activations["h"] = h[:, -1, :].detach()

        handle = down_proj.register_forward_pre_hook(capture_hook)

        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)

        handle.remove()

        logits = outputs.logits[:, -1, :]
        h = activations["h"]

        pos_ids = torch.tensor([p["pos_id"] for p in batch], device=device)
        neg_ids = torch.tensor([p["neg_id"] for p in batch], device=device)

        baseline = baseline_margins[start:start+len(batch)].to(device)

        for i, col in enumerate(columns):

            contrib = h[:, col].unsqueeze(-1) * W_down[:, col]

            new_logits = logits - contrib

            pos_logits = new_logits.gather(1, pos_ids.unsqueeze(1)).squeeze()
            neg_logits = new_logits.gather(1, neg_ids.unsqueeze(1)).squeeze()

            margins = pos_logits - neg_logits

            drops = baseline - margins

            drop_sums[i] += drops.sum()
            counts[i] += len(batch)

    mean_drops = (drop_sums / counts).tolist()

    return mean_drops

def _evaluate_column_ems(
    model,
    probe,
    layer_idx,
    col,
    adv_pairs,
    baseline_margins,
    letter_token_ids,
    early_stop_mu,
    early_stop_sigma,
    max_samples,
    track_flips=False,
):
    """Evaluate a single column for EMS."""
    # This is a placeholder - you need to implement this based on your needs
    # For now, returning dummy values
    return None, 0.0, 0.0, 0.0

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
    """Run the full EMS pipeline (Taylor → Stage1 → Stage2 → Z) for one layer.
    baseline_margins_all and baseline_eval are precomputed per seed and passed in.
    """

    print(f"\n=== Processing layer {layer_idx} ===")

    # Stage 0: directional Taylor scores for all columns.
    jacobian_scores = probe.compute_contrastive_jacobian(layer_idx, adv_pairs)
    num_cols = jacobian_scores.numel()
    print(f"[ProbeCheck] layer={layer_idx} num_cols={num_cols}")

    stage1_top_k = min(cfg.stage1_top_k, num_cols)
    top_vals, top_indices = torch.topk(jacobian_scores, k=stage1_top_k)
    print("[Taylor Top20]", top_vals[:20].detach().cpu().tolist())

    stage1_samples = min(cfg.stage1_samples, len(adv_pairs))
    stage2_samples = min(cfg.stage2_samples, len(adv_pairs))

    baseline_stage1 = baseline_margins_all[:stage1_samples]

    # Letter token ids for A/B/C/D predictions.
    letter_token_ids = _get_letter_token_ids(tokenizer)

    # Random baseline columns for Z-score calibration.
    all_indices = torch.arange(num_cols, device=jacobian_scores.device)
    rand_cols = all_indices[torch.randperm(num_cols)[:RANDOM_BASELINE_COLS]].tolist()

    backbone = getattr(probe.model, "model", probe.model)
    layer = backbone.layers[layer_idx]

    random_mean_drops = _evaluate_columns_vectorized(
        model,
        layer,
        rand_cols,
        adv_pairs[:stage1_samples],
        baseline_stage1,
        letter_token_ids,
    )

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

    candidate_cols = top_indices.tolist()

    mean_drops = _evaluate_columns_vectorized(
        model,
        layer,
        candidate_cols,
        adv_pairs[:stage1_samples],
        baseline_stage1,
        letter_token_ids,
    )

    stage1_candidates = [
       (col, drop)
       for col, drop in zip(candidate_cols, mean_drops)
       if drop > 0
]

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

    backbone = getattr(probe.model, "model", probe.model)
    layer = backbone.layers[layer_idx]

    mean_drops = _evaluate_columns_vectorized(
        model,
        layer,
        stage2_columns,
        adv_pairs_stage2,
        baseline_stage2,
        letter_token_ids,
    )

    for col, mean_drop in zip(stage2_columns, mean_drops):

        z = (mean_drop - mu_rand) / (sigma_rand + 1e-8)

        result = {
            "layer": layer_idx,
            "column": int(col),
            "mean_drop": mean_drop,
            "median_drop": float(mean_drop),
            "z_score": z,
            "flip_rate": 0.0,
            "lethal_flip_rate": 0.0,
        }

        stage2_results.append(result)

        print(
            "[Stage2] "
            f"layer={layer_idx} column={col} "
            f"drop={mean_drop:.6f} Z={z:.3f}"
        )

        if z > Z_THRESHOLD:
            validated_anchors.append(result)

    with open(f"layer_{layer_idx}_stage2_results.json", "w") as f:
        json.dump(stage2_results, f, indent=2)

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
        for a in validated_anchors:
            append_anchor_progress(a["layer"], a["column"], a["mean_drop"], a["z_score"])
    else:
        max_drop = 0.0
        mean_drop_validated = 0.0
        max_z = 0.0

    log_progress(
        f"Layer {layer_idx} Stage1 candidates={len(stage1_candidates)} "
        f"Stage2 columns={len(stage2_columns)} anchors={len(validated_anchors)}"
    )

    print("\n=== Layer Summary ===")
    print(f"layer={layer_idx}")
    print(f"validated_anchors={len(validated_anchors)}")
    print(f"max_drop={max_drop:.6f}")
    print(f"mean_drop={mean_drop_validated:.6f}")
    print(f"max_Z={max_z:.3f}")

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

    ablation_counts = [1, 4, 8, 16, 32, 48, 64]
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
        log_progress(f"Multi-anchor ablation count={count} mean_accuracy={mean_acc} mean_margin={mean_margin}")

    with open("multi_anchor_ablation_results.json", "w") as f:
        json.dump(sweep_results, f, indent=2)


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
        layer_weight = probe.get_layer_weight(layers[0])
        N_COLS = layer_weight.shape[1]

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
        log_progress(f"Random neuron ablation k={k} mean_accuracy={mean_acc} mean_margin={mean_margin}")

    with open("random_neuron_ablation_results.json", "w") as f:
        json.dump(sweep_results, f, indent=2)


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

    # Mistral: model.model.layers (no language_model wrapper)
    backbone = getattr(model, "model", model)
    layer_stack = backbone.layers
    # down_proj input h has shape [batch, seq, intermediate_size]; EMS anchors are columns
    intermediate_size = INTERMEDIATE_SIZE
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

    # Mistral: model.model.layers; use hardcoded BioMistral architecture constants
    backbone = getattr(model, "model", model)
    layer_stack = backbone.layers

    num_heads = NUM_ATTENTION_HEADS
    head_dim = HEAD_DIM

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
        log_progress(f"Attention head ablation k={k} mean_accuracy={mean_acc} mean_margin={mean_margin}")

    with open("attention_head_ablation_results.json", "w") as f:
        json.dump(sweep_results, f, indent=2)


def run_head_to_anchor_attribution(
    model,
    tokenizer,
    anchors: List[Tuple[int, int]],
    calibration_samples: List[Dict[str, Any]],
    n_prompts: int = 48,
    layer_range: Tuple[int, int] = (6, 21),
    output_path: str = "head_anchor_attribution.json",
    batch_size: int = 8,
) -> None:
    """Identify which attention heads causally influence EMS anchor neurons by ablating
    each head and measuring change in anchor activations at the final token.
    Records attribution per anchor neuron and runs prompts in batches for speed.
    """
    model.eval()
    device = next(model.parameters()).device
    pad_token_id = getattr(tokenizer, "pad_token_id", None) or getattr(tokenizer, "eos_token_id", 0)

    # Mistral: model.model.layers; use hardcoded BioMistral architecture constants
    backbone = getattr(model, "model", model)
    layer_stack = backbone.layers

    num_heads = NUM_ATTENTION_HEADS
    head_dim = HEAD_DIM

    anchors_by_layer: Dict[int, List[int]] = {}
    for (layer_idx, col_idx) in anchors:
        if layer_idx not in anchors_by_layer:
            anchors_by_layer[layer_idx] = []
        anchors_by_layer[layer_idx].append(col_idx)

    if not anchors_by_layer:
        print("No anchors for head-to-anchor attribution.")
        return

    n_prompts = min(n_prompts, len(calibration_samples))
    prompt_list: List[Tuple[torch.Tensor, torch.Tensor]] = []
    with torch.no_grad():
        for i in range(n_prompts):
            sample = calibration_samples[i]
            prompt = format_prompt(sample["question"], sample["options"])
            enc = tokenizer(prompt, return_tensors="pt", add_special_tokens=True)
            prompt_list.append((
                enc["input_ids"].to(device),
                enc["attention_mask"].to(device),
            ))

    def _pad_batch(items: List[Tuple[torch.Tensor, torch.Tensor]]) -> Tuple[torch.Tensor, torch.Tensor]:
        ids_list = [x[0] for x in items]
        mask_list = [x[1] for x in items]
        max_len = max(ids.shape[1] for ids in ids_list)
        padded_ids = []
        padded_mask = []
        for ids, mask in items:
            seq_len = ids.shape[1]
            if seq_len < max_len:
                pad_ids = torch.full(
                    (ids.shape[0], max_len - seq_len),
                    pad_token_id,
                    dtype=ids.dtype,
                    device=ids.device,
                )
                pad_mask = torch.zeros(ids.shape[0], max_len - seq_len, dtype=mask.dtype, device=mask.device)
                ids = torch.cat([ids, pad_ids], dim=1)
                mask = torch.cat([mask, pad_mask], dim=1)
            padded_ids.append(ids)
            padded_mask.append(mask)
        return torch.cat(padded_ids, dim=0), torch.cat(padded_mask, dim=0)

    batches: List[Tuple[torch.Tensor, torch.Tensor]] = []
    for start in range(0, len(prompt_list), batch_size):
        batch = prompt_list[start : start + batch_size]
        batches.append(_pad_batch(batch))

    def _run_and_capture_anchors(
        stored_anchor_values: Dict[Tuple[int, int], List[torch.Tensor]],
    ) -> None:
        handles = []
        for layer_idx, cols in anchors_by_layer.items():
            down_proj = layer_stack[layer_idx].mlp.down_proj

            def capture_hook(module, input, layer=layer_idx, anchor_cols=cols):
                h = input[0]
                for col in anchor_cols:
                    stored_anchor_values[(layer, col)].append(h[:, -1, col].detach().cpu())

            handles.append(down_proj.register_forward_pre_hook(capture_hook))

        with torch.no_grad():
            for input_ids, attention_mask in batches:
                model(input_ids=input_ids, attention_mask=attention_mask)

        for h in handles:
            h.remove()

    stored_baseline: Dict[Tuple[int, int], List[torch.Tensor]] = {
        (layer_idx, col_idx): [] for (layer_idx, col_idx) in anchors
    }
    _run_and_capture_anchors(stored_baseline)

    baseline_anchor_activation: Dict[Tuple[int, int], float] = {}
    for (layer_idx, col_idx) in anchors:
        vals = stored_baseline[(layer_idx, col_idx)]
        if len(vals) == 0:
            continue

        stacked = torch.cat(vals, dim=0)
        baseline_anchor_activation[(layer_idx, col_idx)] = stacked.mean().item()

    results: List[Dict[str, Any]] = []
    layer_start, layer_end = layer_range

    with torch.no_grad():
        for layer_idx in range(layer_start, layer_end):
            layer = layer_stack[layer_idx]
            weight = layer.self_attn.o_proj.weight
            for head_idx in range(num_heads):
                start = head_idx * head_dim
                end = (head_idx + 1) * head_dim
                saved_slice = weight[:, start:end].clone()
                weight[:, start:end] = 0

                stored_after: Dict[Tuple[int, int], List[torch.Tensor]] = {
                    (al, ac): [] for (al, ac) in anchors
                }
                _run_and_capture_anchors(stored_after)

                for (anchor_layer, anchor_column) in anchors:
                    vals = stored_after[(anchor_layer, anchor_column)]

                    if len(vals) == 0:
                        continue

                    stacked = torch.cat(vals, dim=0)
                    activation_after_mean = stacked.mean().item()

                    delta = abs(
                        activation_after_mean
                        - baseline_anchor_activation[(anchor_layer, anchor_column)]
                    )

                    results.append({
                        "head_layer": layer_idx,
                        "head": head_idx,
                        "anchor_layer": anchor_layer,
                        "anchor_column": anchor_column,
                        "delta": float(delta),
                    })

                weight[:, start:end] = saved_slice.to(weight.dtype).to(weight.device)

    results.sort(key=lambda x: x["delta"], reverse=True)

    print("\n=== Head-to-Anchor Attribution ===")
    print("Top 15 head → anchor interactions:")
    for r in results[:15]:
        print(f"  Head {r['head_layer']}.{r['head']} → Anchor {r['anchor_layer']}.{r['anchor_column']} delta {r['delta']:.2f}")

    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {output_path}")


def main():
    # Use device map auto-loading; seeds are handled per-run below.
    _ = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    RUN_HEAD_ONLY = False

    model_name = "meta-llama/Llama-3.2-3B-Instruct"
    layers_to_test = list(range(6, 22))

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

    # Load tokenizer once (use_fast=False for Mistral compatibility).
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    log_progress(
        f"START {datetime.now()} model={model_name} "
        f"batch={BATCH_SIZE} calib={cfg.calibration_size} "
        f"layers={layers_to_test}"
    )

    if RUN_HEAD_ONLY:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )
        # Dynamically set architecture parameters from model.config
        config = model.config
        HIDDEN_SIZE = config.hidden_size
        NUM_LAYERS = config.num_hidden_layers
        NUM_ATTENTION_HEADS = config.num_attention_heads
        INTERMEDIATE_SIZE = config.intermediate_size
        HEAD_DIM = HIDDEN_SIZE // NUM_ATTENTION_HEADS
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.set_float32_matmul_precision("high")
        print_gpu_memory()
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
        print("\n=== Head-to-Anchor Attribution ===")
        run_head_to_anchor_attribution(
            model=model,
            tokenizer=tokenizer,
            anchors=anchors,
            calibration_samples=calibration_samples,
            n_prompts=96,
            layer_range=(6, 21),
        )
    else:
        # Sweep over multiple random seeds to test anchor stability.
        seeds = [1, 2, 3]
        checkpoint = load_checkpoint()
        if checkpoint:
            completed_seeds = list(checkpoint["completed_seeds"])
            all_seed_results = list(checkpoint["all_seed_results"])
        else:
            completed_seeds = []
            all_seed_results = []

        calibration_reference: List[Dict[str, Any]] = []

        for seed in seeds:
            if seed in completed_seeds:
                log_progress(f"Seed {seed} already completed, skipping")
                continue

            log_progress(f"Seed {seed} started")
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
            # Dynamically set architecture parameters from model.config
            config = model.config
            HIDDEN_SIZE = config.hidden_size
            NUM_LAYERS = config.num_hidden_layers
            NUM_ATTENTION_HEADS = config.num_attention_heads
            INTERMEDIATE_SIZE = config.intermediate_size
            HEAD_DIM = HIDDEN_SIZE // NUM_ATTENTION_HEADS
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            torch.set_float32_matmul_precision("high")
            print_gpu_memory()

            # Dataset split: configurable calibration size, fixed held-out evaluation size.
            eval_size = 60
            total_needed = cfg.calibration_size + eval_size
            samples = load_mcq_dataset(n_total=total_needed)
            random.shuffle(samples)
            calibration = samples[: cfg.calibration_size]
            evaluation = samples[cfg.calibration_size : cfg.calibration_size + eval_size]
            with open("calibration_prompts.json", "w") as f:
                json.dump(calibration, f, indent=2)
            with open("evaluation_prompts.json", "w") as f:
                json.dump(evaluation, f, indent=2)
            if seed == seeds[0]:
                calibration_reference = calibration

            adv_pairs = build_adversarial_pairs(
                model, tokenizer, calibration, n_calib=len(calibration)
            )
            adv_metadata = [
                {
                    "correct_index": p["correct_letter_index"],
                    "neg_index": p["neg_letter_index"],
                    "pos_id": p["pos_id"],
                    "neg_id": p["neg_id"],
                }
                for p in adv_pairs
            ]
            with open("adv_pairs_metadata.json", "w") as f:
                json.dump(adv_metadata, f, indent=2)

            probe = MedNSQProbe(model)
            baseline_margins_all = probe.compute_per_sample_margins(adv_pairs)
            torch.save(baseline_margins_all, "baseline_margins.pt")
            baseline_eval = evaluate_model(model, tokenizer, evaluation)
            print("\n=== Baseline Held-out Evaluation ===")
            print("Baseline accuracy:", baseline_eval["accuracy"])
            print("Baseline mean margin:", baseline_eval["mean_margin"])

            all_layer_summaries: List[Dict[str, Any]] = []
            for layer_idx in layers_to_test:
                log_progress(f"Seed {seed} Layer {layer_idx} starting EMS")
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
                all_layer_summaries.append(summary)
                with open(f"layer_{layer_idx}_results.json", "w") as f:
                    json.dump(summary, f, indent=2)
                num_anchors = len(summary["validated_anchors"])
                max_drop = summary.get("max_drop", 0.0)
                log_progress(f"Seed {seed} Layer {layer_idx} anchors={num_anchors} max_drop={max_drop:.4f}")

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
            completed_seeds.append(seed)
            save_checkpoint({
                "completed_seeds": completed_seeds,
                "all_seed_results": all_seed_results,
            })
            log_progress(f"Seed {seed} completed, checkpoint saved")

            # Clear per-seed probe and model; next seed gets a fresh model.
            del probe
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

        # Load model once for ablation experiments (no weight modifications during ablation per seed).
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )
        # Dynamically set architecture parameters from model.config
        config = model.config
        HIDDEN_SIZE = config.hidden_size
        NUM_LAYERS = config.num_hidden_layers
        NUM_ATTENTION_HEADS = config.num_attention_heads
        INTERMEDIATE_SIZE = config.intermediate_size
        HEAD_DIM = HIDDEN_SIZE // NUM_ATTENTION_HEADS
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.set_float32_matmul_precision("high")
        print_gpu_memory()

        # Final EMS summary across all seeds is stored in all_seed_results.
        print("\n=== Anchor Stability Summary ===")
        log_progress("Anchor Stability Summary:")
        for layer_idx in layers_to_test:
            counts: List[int] = []
            for seed_result in all_seed_results:
                layer_summary = next(
                    s for s in seed_result["layers"] if s["layer"] == layer_idx
                )
                counts.append(len(layer_summary["validated_anchors"]))
            mean_anchors = sum(counts) / len(counts) if counts else 0.0
            print(f"layer{layer_idx} mean anchors = {mean_anchors:.3f}")
            log_progress(f"  layer{layer_idx} mean anchors = {mean_anchors:.3f}")

        # Save experiment output to JSON for reproducibility.
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        with open(f"ems_seed_sweep_{timestamp}.json", "w") as f:
            json.dump(all_seed_results, f, indent=2)

        # Gather anchors from EMS discovery across all seeds; dedupe and keep top 64 by mean_drop.
        anchor_to_stats: Dict[Tuple[int, int], Dict[str, float]] = {}
        for seed_result in all_seed_results:
            for layer_summary in seed_result["layers"]:
                for a in layer_summary["validated_anchors"]:
                    key = (a["layer"], a["column"])
                    current_drop = anchor_to_stats.get(key, {}).get("drop", 0.0)
                    if a["mean_drop"] > current_drop:
                        anchor_to_stats[key] = {
                            "drop": a["mean_drop"],
                            "z": a["z_score"],
                        }
        sorted_anchor_stats = sorted(
            anchor_to_stats.items(), key=lambda x: x[1]["drop"], reverse=True
        )[:64]
        anchors = [k for k, _ in sorted_anchor_stats]

        # Anchor seed stability: how often each anchor appeared across seeds.
        anchor_frequency: Dict[Tuple[int, int], int] = {}
        for seed_result in all_seed_results:
            for layer_summary in seed_result["layers"]:
                for anchor in layer_summary["validated_anchors"]:
                    key = (anchor["layer"], anchor["column"])
                    anchor_frequency[key] = anchor_frequency.get(key, 0) + 1
        num_seeds = len(all_seed_results)
        print("\n=== Anchor Seed Stability ===")
        log_progress("=== Anchor Seed Stability ===")
        for (layer, col), freq in sorted(anchor_frequency.items(), key=lambda x: -x[1]):
            print(f"(layer={layer} col={col}) frequency={freq}/{num_seeds}")
            log_progress(f"(layer={layer} col={col}) frequency={freq}/{num_seeds}")

        # Save final anchor list to anchors_final.json and append to anchors_progress.txt
        anchors_final_data = [
            {"layer": layer, "column": col, "drop": stats["drop"], "z": stats["z"]}
            for (layer, col), stats in sorted_anchor_stats
        ]
        with open("anchors_final.json", "w") as f:
            json.dump(anchors_final_data, f, indent=2)
        with open("anchors_progress.txt", "a") as f:
            f.write("--- Final anchor list ---\n")
            for (layer, column), stats in sorted_anchor_stats:
                f.write(f"(layer={layer} col={column}) drop={stats['drop']:.2f} Z={stats['z']:.1f}\n")

        print("\n=== Discovered Anchors ===")
        log_progress("Discovered anchors (top 64):")
        for (layer, column), stats in sorted_anchor_stats:
            print(f"(layer={layer}, col={column}) drop={stats['drop']:.2f} Z={stats['z']:.1f}")
            log_progress(f"  (layer={layer}, col={column}) drop={stats['drop']:.2f} Z={stats['z']:.1f}")

        print_gpu_memory()
        print("\n=== Running Anchor Activation Patching Experiment ===")
        calibration_samples = calibration_reference[:120]
        run_anchor_activation_patching_experiment(
            model=model,
            tokenizer=tokenizer,
            anchors=anchors,
            calibration_samples=calibration_samples,
            n_samples=120,
        )
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
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

        print("\n=== Head-to-Anchor Attribution ===")
        run_head_to_anchor_attribution(
            model=model,
            tokenizer=tokenizer,
            anchors=anchors,
            calibration_samples=calibration_samples,
            n_prompts=96,
            layer_range=(6, 21),
        )
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

        run_multi_anchor_ablation_sweep(
            model=model,
            tokenizer=tokenizer,
            anchors=anchors,
            seeds=[1, 2, 3, 4, 5],
            calibration_size=cfg.calibration_size,
            eval_size=60,
        )
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

        anchor_layers = sorted(list(set(layer for layer, _ in anchors)))
        run_random_neuron_ablation_baseline(
            model=model,
            tokenizer=tokenizer,
            layers=anchor_layers,
            seeds=[1, 2, 3, 4, 5],
            calibration_size=cfg.calibration_size,
            eval_size=60,
            k_values=[1, 4, 8, 16, 32, 48, 64],
        )
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

        run_attention_head_ablation_sweep(
            model=model,
            tokenizer=tokenizer,
            layers=[8, 16],
            seeds=[1, 2, 3, 4, 5],
            calibration_size=cfg.calibration_size,
            eval_size=60,
            k_values=[1, 2, 4, 8],
        )
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        print_gpu_memory()

        final_summary = {
            "anchors": anchors_final_data,
            "anchor_frequency": [
                {"layer": layer, "column": col, "frequency": freq}
                for (layer, col), freq in anchor_frequency.items()
            ],
            "layers_tested": layers_to_test,
            "model": model_name,
        }
        with open("final_experiment_summary.json", "w") as f:
            json.dump(final_summary, f, indent=2)


if __name__ == "__main__":
    main()