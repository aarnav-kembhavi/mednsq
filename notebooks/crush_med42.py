"""
Multi-anchor down_proj column crush evaluation for Med42 (Llama3-Med42-8B).

Same evaluation flow, batching, and margin logic as run_phi3_anchor_ablation.py;
only model loading, down_proj access, and anchor list differ.
"""

from __future__ import annotations

import os
import random
import sys
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer


class Tee:
    def __init__(self, file_path):
        self.file = open(file_path, "w", encoding="utf-8")
        self.stdout = sys.stdout

    def write(self, data):
        self.stdout.write(data)
        self.file.write(data)

    def flush(self):
        self.stdout.flush()
        self.file.flush()


sys.stdout = Tee("crushing_med42_full.log")
sys.stderr = sys.stdout


# ----------------------------
# SECTION 1: Config
# ----------------------------

MODEL_NAME = "m42-health/Llama3-Med42-8B"

# Batching constant (preserve default)
BATCH_SIZE = 32

# Tokenizer padding id is set once after tokenizer load.
PAD_TOKEN_ID: int = 0

N_RANDOM_TRIALS = 50

ANCHOR_RECORDS_RAW: List[Tuple[int, int, float]] = [
    (20, 4299, 0.10),
    (19, 7232, 0.11),
    (14, 8, 0.10),
    (17, 7522, 0.06),
    (18, 7417, 0.05),
    (13, 2590, 0.05),
    (20, 3476, 0.03),
    (15, 2808, 0.03),
    (13, 4542, 0.03),
    (16, 9215, 0.03),
    (18, 10857, 0.03),
    (18, 2694, 0.03),
    (14, 318, 0.03),
    (15, 1706, 0.03),
    (17, 13385, 0.03),
    (20, 10638, 0.02),
    (12, 1493, 0.08),
    (12, 2976, 0.02),
    (12, 1780, 0.02),
    (13, 4216, 0.02),
    (13, 1724, 0.02),
    (14, 12639, 0.02),
    (15, 11853, 0.02),
    (15, 1283, 0.02),
    (15, 10813, 0.02),
    (16, 8902, 0.02),
    (16, 13791, 0.02),
    (16, 10142, 0.01),
    (16, 6737, 0.01),
    (17, 10284, 0.02),
    (17, 8887, 0.01),
    (17, 863, 0.01),
    (19, 9770, 0.02),
    (19, 1498, 0.02),
    (20, 5270, 0.02),
    (20, 2786, 0.01),
    (15, 6589, 0.01),
    (15, 2295, 0.01),
]


def _repo_root() -> str:
    """Return repo root; when script lives under notebooks/, return parent dir."""
    path = os.path.abspath(__file__)
    dirname = os.path.dirname(path)
    if os.path.basename(dirname) == "notebooks":
        return os.path.dirname(dirname)
    return dirname


def _ensure_import_paths() -> None:
    root = _repo_root()
    notebooks = os.path.join(root, "notebooks")
    if notebooks not in sys.path:
        sys.path.insert(0, notebooks)
    if root not in sys.path:
        sys.path.insert(0, root)


def log_progress(text: str) -> None:
    with open("experiment_progress.log", "a", encoding="utf-8") as f:
        f.write(text + "\n")


def _set_reproducible(seed: int) -> None:
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)


def _get_med42_layers(model) -> List[Any]:
    if not hasattr(model, "model") or not hasattr(model.model, "layers"):
        raise RuntimeError(
            "Med42 architecture check failed: expected model.model.layers"
        )
    return list(model.model.layers)


def _med42_down_proj_weight(model, layer_idx: int) -> torch.Tensor:
    """Llama-style: layer_stack[layer].mlp.down_proj.weight"""
    layers = _get_med42_layers(model)
    if layer_idx < 0 or layer_idx >= len(layers):
        raise IndexError(f"layer_idx out of range: {layer_idx}")
    layer = layers[layer_idx]
    if not hasattr(layer, "mlp") or not hasattr(layer.mlp, "down_proj"):
        raise RuntimeError(
            f"Med42 architecture check failed at layer {layer_idx}: "
            "expected layer.mlp.down_proj"
        )
    return layer.mlp.down_proj.weight


# ----------------------------
# SECTION 2: Data + batching
# ----------------------------

def _assert_adv_pairs_shape(adv_pairs: List[Dict[str, Any]]) -> None:
    if not adv_pairs:
        raise RuntimeError("adv_pairs is empty. Cannot run EMS or experiments.")
    required = {
        "input_ids",
        "attention_mask",
        "pos_id",
        "neg_id",
        "safe_input_ids",
        "safe_attention_mask",
    }
    missing_any = required - set(adv_pairs[0].keys())
    if missing_any:
        raise RuntimeError(
            f"adv_pairs[0] missing required keys: {sorted(list(missing_any))}"
        )


def _assert_letter_token_ids(letter_token_ids: torch.Tensor) -> None:
    if not isinstance(letter_token_ids, torch.Tensor):
        raise TypeError("letter_token_ids must be a torch.Tensor")
    if letter_token_ids.ndim != 1 or letter_token_ids.numel() != 4:
        raise RuntimeError(
            f"letter_token_ids must be shape [4], got {tuple(letter_token_ids.shape)}"
        )
    if letter_token_ids.dtype not in (torch.int64, torch.long):
        raise RuntimeError(f"letter_token_ids must be int64, got {letter_token_ids.dtype}")
    if (letter_token_ids < 0).any().item():
        raise RuntimeError("letter_token_ids contains negative ids")


def _answer_positions_batch(
    input_ids: torch.Tensor,
    seq_lens: List[int],
    letter_token_ids: torch.Tensor,
) -> List[Optional[int]]:
    """
    For each sample, find the last position where input_ids equals any letter token (A/B/C/D).
    Returns list of answer_idx per sample; None for samples with no answer token.
    """
    letter_set = set(int(x) for x in letter_token_ids.cpu().tolist())
    result: List[Optional[int]] = []
    for i in range(input_ids.shape[0]):
        L = int(seq_lens[i])
        ids_slice = input_ids[i, :L].cpu().tolist()
        positions = [j for j in range(L) if ids_slice[j] in letter_set]
        if not positions:
            result.append(None)
            continue
        answer_idx = max(positions)
        assert answer_idx >= 0, f"answer_idx {answer_idx} < 0"
        assert answer_idx < L, f"answer_idx {answer_idx} >= seq_len {L}"
        result.append(answer_idx)
    return result


def batched_forward_pass(
    model,
    adv_pairs: List[Dict[str, Any]],
    letter_token_ids: torch.Tensor,
    tokenizer: Any = None,
) -> Tuple[torch.Tensor, List[int]]:
    """
    Compute per-sample margins (pos - neg) and A/B/C/D predictions in mini-batches.

    Returns:
      margins: float32 tensor of shape [N]
      predictions: list[int] where each entry is in {0,1,2,3} (index into A/B/C/D)
    """
    if not adv_pairs:
        raise RuntimeError("batched_forward_pass: adv_pairs is empty")
    _assert_letter_token_ids(letter_token_ids)

    device = next(model.parameters()).device
    letter_token_ids = letter_token_ids.to(device)

    margins: List[float] = []
    preds: List[int] = []

    vocab_size = getattr(getattr(model, "config", None), "vocab_size", None)
    if vocab_size is not None:
        if int(letter_token_ids.max().item()) >= int(vocab_size):
            raise RuntimeError(
                f"Invalid letter_token_ids: max={int(letter_token_ids.max().item())} "
                f">= vocab_size={int(vocab_size)}"
            )

    for start in range(0, len(adv_pairs), BATCH_SIZE):
        batch = adv_pairs[start : start + BATCH_SIZE]
        if not batch:
            continue

        seq_lens = [int(pair["input_ids"].shape[1]) for pair in batch]
        if any(l <= 0 for l in seq_lens):
            raise RuntimeError(f"Invalid sequence lengths in batch: {seq_lens}")
        max_len = max(seq_lens)

        ids_list: List[torch.Tensor] = []
        mask_list: List[torch.Tensor] = []
        for pair in batch:
            ids = pair["input_ids"]
            mask = pair["attention_mask"]
            if ids.ndim != 2 or mask.ndim != 2:
                raise RuntimeError(
                    "Expected input_ids/attention_mask to be rank-2 tensors "
                    f"got ids.ndim={ids.ndim} mask.ndim={mask.ndim}"
                )
            if ids.shape != mask.shape:
                raise RuntimeError(
                    f"input_ids and attention_mask shape mismatch: {ids.shape} vs {mask.shape}"
                )
            ids = ids.to(device)
            mask = mask.to(device)
            pad_len = max_len - ids.shape[1]
            if pad_len > 0:
                ids = F.pad(ids, (0, pad_len), value=int(PAD_TOKEN_ID))
                mask = F.pad(mask, (0, pad_len), value=0)
            ids_list.append(ids)
            mask_list.append(mask)

        input_ids = torch.cat(ids_list, dim=0)
        attention_mask = torch.cat(mask_list, dim=0)

        answer_indices = _answer_positions_batch(input_ids, seq_lens, letter_token_ids)

        with torch.no_grad():
            out = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = out.logits  # [B, max_len, vocab]

        for i, pair in enumerate(batch):
            answer_idx = answer_indices[i]
            if answer_idx is None:
                margins.append(float("nan"))
                preds.append(-1)
                continue
            seq_len_i = int(seq_lens[i])
            assert answer_idx >= 0, f"answer_idx {answer_idx} < 0"
            assert answer_idx < seq_len_i, f"answer_idx {answer_idx} >= seq_len {seq_len_i}"
            token_logits = logits[i, answer_idx, :]
            pos_id = int(pair["pos_id"])
            neg_id = int(pair["neg_id"])
            if vocab_size is not None and (pos_id >= vocab_size or neg_id >= vocab_size):
                raise RuntimeError(
                    f"Invalid pos/neg token ids for vocab: pos_id={pos_id} neg_id={neg_id} vocab_size={vocab_size}"
                )
            if pos_id < 0 or neg_id < 0:
                raise RuntimeError(f"Invalid pos/neg token ids: pos_id={pos_id} neg_id={neg_id}")

            margin = (token_logits[pos_id] - token_logits[neg_id]).item()
            margins.append(float(margin))

            letter_logits = token_logits.index_select(0, letter_token_ids)
            pred_idx = int(torch.argmax(letter_logits).item())
            preds.append(pred_idx)

    return torch.tensor(margins, dtype=torch.float32), preds


def simulate_column_crush_med42(model, layer_idx: int, col_idx: int) -> torch.Tensor:
    weight = _med42_down_proj_weight(model, layer_idx)
    if col_idx < 0 or col_idx >= weight.shape[1]:
        raise IndexError(f"col_idx out of range: {col_idx} for weight.shape={tuple(weight.shape)}")
    original_col = weight[:, col_idx].detach().clone()
    scale = original_col.abs().mean()
    crushed = original_col.sign() * scale
    with torch.no_grad():
        weight[:, col_idx] = crushed.to(dtype=weight.dtype, device=weight.device)
    return original_col


def restore_column_med42(model, layer_idx: int, col_idx: int, original_col: torch.Tensor) -> float:
    weight = _med42_down_proj_weight(model, layer_idx)
    with torch.no_grad():
        weight[:, col_idx] = original_col.to(dtype=weight.dtype, device=weight.device)
    restored = weight[:, col_idx].detach().to(original_col.dtype)
    max_diff = float((restored - original_col).abs().max().item())
    return max_diff


def compute_per_sample_margins_full(
    model,
    adv_pairs: List[Dict[str, Any]],
    letter_token_ids: torch.Tensor,
    tokenizer: Any = None,
) -> torch.Tensor:
    """Full margin vector (same as former Phi3Probe.compute_per_sample_margins)."""
    if not adv_pairs:
        return torch.zeros(0, dtype=torch.float32)
    margins, _ = batched_forward_pass(model, adv_pairs, letter_token_ids, tokenizer=tokenizer)
    return margins


def mean_drop_for_set(
    model,
    adv_pairs: List[Dict[str, Any]],
    letter_token_ids: torch.Tensor,
    tokenizer: Any,
    neuron_set: List[Tuple[int, int]],
    baseline_margins: torch.Tensor,
) -> Tuple[float, float, torch.Tensor]:
    saved: List[Tuple[int, int, torch.Tensor]] = []
    try:
        for l, c in neuron_set:
            orig = simulate_column_crush_med42(model, l, c)
            saved.append((l, c, orig))
        crushed_margins, _ = batched_forward_pass(
            model, adv_pairs, letter_token_ids, tokenizer=tokenizer
        )
    finally:
        for l, c, orig in saved:
            restore_column_med42(model, l, c, orig)

    baseline_margins = baseline_margins.float()
    crushed_margins = crushed_margins.float()
    if crushed_margins.device != baseline_margins.device:
        crushed_margins = crushed_margins.to(baseline_margins.device)
    valid = ~torch.isnan(baseline_margins) & ~torch.isnan(crushed_margins)
    drops = baseline_margins[valid] - crushed_margins[valid]
    if drops.numel() == 0:
        z = torch.zeros(0, dtype=torch.float32, device=baseline_margins.device)
        return 0.0, 0.0, z
    mean_drop = float(drops.mean().item())
    std_drop = float(drops.std(unbiased=False).item())
    return mean_drop, std_drop, drops


def enumerate_non_anchor_neurons(
    model, anchor_set: set[Tuple[int, int]]
) -> List[Tuple[int, int]]:
    out: List[Tuple[int, int]] = []
    n_layers = len(_get_med42_layers(model))
    for l in range(n_layers):
        w = _med42_down_proj_weight(model, l)
        n_cols = int(w.shape[1])
        for c in range(n_cols):
            if (l, c) not in anchor_set:
                out.append((l, c))
    return out


def run_dataset_evaluation(
    dataset_name: str,
    model: Any,
    tokenizer: Any,
    letter_token_ids: torch.Tensor,
    calibration_size: int,
    eval_size: int,
    seed: int,
) -> None:
    _set_reproducible(seed)

    from mednsq_data import load_mcq_dataset, build_adversarial_pairs

    samples = load_mcq_dataset(n_total=calibration_size + eval_size)
    random.shuffle(samples)
    calibration = samples[:calibration_size]
    _evaluation = samples[calibration_size : calibration_size + eval_size]

    adv_pairs = build_adversarial_pairs(model, tokenizer, calibration, n_calib=len(calibration))
    _assert_adv_pairs_shape(adv_pairs)

    baseline = compute_per_sample_margins_full(model, adv_pairs, letter_token_ids, tokenizer=tokenizer)

    anchors_sorted = sorted(ANCHOR_RECORDS_RAW, key=lambda x: float(x[2]), reverse=True)
    anchor_neurons_full = [(int(l), int(c)) for (l, c, _) in anchors_sorted]
    anchor_set = set(anchor_neurons_full)

    candidate_neurons = enumerate_non_anchor_neurons(model, anchor_set)

    max_k = len(anchor_neurons_full)
    k_values: List[int] = []
    for k in [4, 8, 16, 32, max_k]:
        if k <= max_k and k not in k_values:
            k_values.append(k)

    print(f"Dataset: {dataset_name}")

    for k in k_values:
        if k > len(candidate_neurons):
            raise RuntimeError(f"Not enough candidate neurons for k={k}")

        anchor_subset = anchor_neurons_full[:k]

        anchor_mean, anchor_std, _ = mean_drop_for_set(
            model,
            adv_pairs,
            letter_token_ids,
            tokenizer,
            anchor_subset,
            baseline,
        )

        random_drops_list = []
        for _ in range(N_RANDOM_TRIALS):
            subset = random.sample(candidate_neurons, k)

            rand_mean, _, _ = mean_drop_for_set(
                model,
                adv_pairs,
                letter_token_ids,
                tokenizer,
                subset,
                baseline,
            )
            random_drops_list.append(rand_mean)

        random_mean = float(np.mean(random_drops_list))
        random_std = float(np.std(random_drops_list))

        print(f"\n=== K = {k} ===")
        print(f"Anchor mean drop: {anchor_mean}")
        print(f"Random mean drop: {random_mean} ± {random_std}")
        print(f"Diff: {anchor_mean - random_mean}")

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def main() -> None:
    _ensure_import_paths()

    from mednsq_eval import _get_letter_token_ids

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    global PAD_TOKEN_ID
    PAD_TOKEN_ID = int(getattr(tokenizer, "pad_token_id", None) or getattr(tokenizer, "eos_token_id", 0) or 0)

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()

    letter_token_ids = _get_letter_token_ids(tokenizer)
    _assert_letter_token_ids(letter_token_ids)
    letter_token_ids = letter_token_ids.to(next(model.parameters()).device)

    for dataset_name in ("mcq",):
        run_dataset_evaluation(
            dataset_name=dataset_name,
            model=model,
            tokenizer=tokenizer,
            letter_token_ids=letter_token_ids,
            calibration_size=400,
            eval_size=60,
            seed=1,
        )


if __name__ == "__main__":
    main()
