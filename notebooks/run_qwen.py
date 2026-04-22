"""
MedNSQ anchor discovery for Meditron3 (Qwen2.5 backbone).

CRITICAL: This script intentionally adapts to the existing modules WITHOUT modifying them:
  - notebooks/mednsq_data.py  (imported as mednsq_data)
  - notebooks/mednsq_eval.py  (imported as mednsq_eval)
  - notebooks/mednsq_probe.py (not imported; EMS core re-implemented here)

Target model:
  OpenMeditron/Meditron3-Qwen2.5-7B

Architecture: generic transformer blocks via _get_layers / mlp.down_proj / self_attn.o_proj.
"""

from __future__ import annotations

import builtins
import json
import os
import random
import sys
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

PRINTED_ANSWER_STATS = False
VERBOSE = False
_ORIG_PRINT = builtins.print


def print(*args, **kwargs):
    if VERBOSE:
        _ORIG_PRINT(*args, **kwargs)


# ----------------------------
# SECTION 1: Config
# ----------------------------

MODEL_NAME = "OpenMeditron/Meditron3-Qwen2.5-7B"

# Experiments to preserve (from notebooks/run_mednsq.py)
SEEDS = [1]
LAYERS_TO_TEST = list(range(10, 23))

# EMS constants (preserve)
RANDOM_BASELINE_COLS = 128
Z_THRESHOLD = 2.0

# Batching constant (preserve default)
BATCH_SIZE = 32

# Baseline quality: margin warning threshold
MARGIN_WARNING_THRESHOLD = 0.01

# Tokenizer padding id is set once after tokenizer load.
PAD_TOKEN_ID: int = 0


@dataclass
class EMSConfig:
    # Per current pipeline defaults (notebooks/run_mednsq.py)
    layer_idx: int = 2
    calibration_size: int = 400
    stage1_top_k: int = 256
    stage1_samples: int = 60
    stage2_top_k: int = 128
    stage2_samples: int = 400


def _repo_root() -> str:
    """Return repo root; when script lives under notebooks/, return parent dir."""
    path = os.path.abspath(__file__)
    dirname = os.path.dirname(path)
    if os.path.basename(dirname) == "notebooks":
        return os.path.dirname(dirname)
    return dirname


def _ensure_import_paths() -> None:
    # Existing modules live in notebooks/ but are imported as top-level modules.
    root = _repo_root()
    notebooks = os.path.join(root, "notebooks")
    if notebooks not in sys.path:
        sys.path.insert(0, notebooks)
    if root not in sys.path:
        sys.path.insert(0, root)


def log_progress(text: str) -> None:
    with open("experiment_progress.log", "a", encoding="utf-8") as f:
        f.write(text + "\n")


def append_anchor_progress(layer: int, col: int, drop: float, z: float) -> None:
    with open("anchors_progress.txt", "a", encoding="utf-8") as f:
        f.write(f"(layer={layer} col={col}) drop={drop:.2f} Z={z:.1f}\n")


def save_checkpoint(data: Dict[str, Any]) -> None:
    torch.save(data, "ems_checkpoint.pt")


def load_checkpoint() -> Any:
    if os.path.exists("ems_checkpoint.pt"):
        try:
            return torch.load("ems_checkpoint.pt", weights_only=False)
        except TypeError:
            return torch.load("ems_checkpoint.pt")
    return None


def _set_reproducible(seed: int) -> None:
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)


def _get_layers(model):
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return list(model.model.layers)
    elif hasattr(model, "layers"):
        return list(model.layers)
    else:
        raise RuntimeError("Cannot find transformer layers")


def _down_proj_weight(model, layer_idx):
    layers = _get_layers(model)
    layer = layers[layer_idx]
    if not hasattr(layer, "mlp") or not hasattr(layer.mlp, "down_proj"):
        raise RuntimeError(f"Missing mlp.down_proj at layer {layer_idx}")
    return layer.mlp.down_proj.weight


def _o_proj_weight(model, layer_idx):
    layers = _get_layers(model)
    layer = layers[layer_idx]
    if not hasattr(layer, "self_attn") or not hasattr(layer.self_attn, "o_proj"):
        raise RuntimeError(f"Missing self_attn.o_proj at layer {layer_idx}")
    return layer.self_attn.o_proj.weight


def _print_dimension_summary_once(model) -> Tuple[int, int, int, int, int]:
    hidden_size = int(model.config.hidden_size)
    num_layers = int(model.config.num_hidden_layers)
    num_heads = int(model.config.num_attention_heads)
    intermediate_size = int(model.config.intermediate_size)
    if num_heads <= 0 or hidden_size % num_heads != 0:
        raise RuntimeError(
            f"Invalid head config: hidden_size={hidden_size} num_heads={num_heads}"
        )
    head_dim = hidden_size // num_heads
    print("=== Model Dimensions ===")
    print("hidden_size:", hidden_size)
    print("num_layers:", num_layers)
    print("num_heads:", num_heads)
    print("intermediate_size:", intermediate_size)
    print("head_dim:", head_dim)
    return hidden_size, num_layers, num_heads, intermediate_size, head_dim


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


def _compute_answer_idx_from_ids(input_ids: torch.Tensor, letter_token_ids: torch.Tensor) -> Optional[int]:
    """For patching: last position where token is A/B/C/D. input_ids shape (1, L). Returns None if none found."""
    seq = input_ids[0].tolist()
    letter_set = set(int(x) for x in letter_token_ids.cpu().tolist())
    positions = [i for i, tok in enumerate(seq) if tok in letter_set]
    if not positions:
        print("WARNING: skipping sample, no answer token found")
        return None
    return max(positions)


def _answer_position_single(input_ids: torch.Tensor, letter_token_ids: torch.Tensor) -> Optional[int]:
    """Find the last position in the sequence where token is one of A/B/C/D. Returns None if none found."""
    if input_ids.dim() == 2:
        seq = input_ids[0].cpu().tolist()
    else:
        seq = input_ids.cpu().tolist()
    letter_set = set(int(x) for x in letter_token_ids.cpu().tolist())
    positions = [j for j in range(len(seq)) if seq[j] in letter_set]
    if not positions:
        print("WARNING: skipping sample, no answer token found")
        return None
    answer_idx = max(positions)
    assert answer_idx >= 0, f"answer_idx {answer_idx} < 0"
    assert answer_idx < len(seq), f"answer_idx {answer_idx} >= seq_len {len(seq)}"
    return answer_idx


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
            print("WARNING: skipping sample, no answer token found")
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
    all_answer_indices: List[int] = []
    all_distances: List[int] = []

    # Validate token ids with vocab size when available.
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

        for idx in answer_indices:
            if idx is not None:
                all_answer_indices.append(idx)
        for i in range(len(answer_indices)):
            if answer_indices[i] is not None:
                all_distances.append(seq_lens[i] - answer_indices[i])

        # STRICT DEBUG CHECK (run once): keep computations; decoded/token-list prints removed.
        if start == 0 and tokenizer is not None:
            i0 = 0
            L0 = int(seq_lens[i0])
            ids0 = input_ids[i0, :L0].cpu().tolist()
            token_list = [(j, ids0[j], tokenizer.decode([ids0[j]]) if ids0[j] != PAD_TOKEN_ID else "<pad>") for j in range(L0)]
            letter_set = set(int(x) for x in letter_token_ids.cpu().tolist())
            letter_positions = [j for j in range(L0) if ids0[j] in letter_set]

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

    global PRINTED_ANSWER_STATS
    if not PRINTED_ANSWER_STATS:
        if len(all_answer_indices) > 0:
            arr = torch.tensor(all_answer_indices)
            print("=== Answer Index Stats ===")
            print("min:", int(arr.min().item()))
            print("max:", int(arr.max().item()))
            print("mean:", float(arr.float().mean().item()))
            hist = torch.histc(arr.float(), bins=10)
            print("histogram:", hist.tolist())
            if arr.max().item() - arr.min().item() > 50:
                print("WARNING: Answer positions are highly scattered")
        if len(all_distances) > 0:
            dist_tensor = torch.tensor(all_distances)
            print("=== Distance to End Stats ===")
            print("mean distance:", float(dist_tensor.float().mean().item()))
            print("max distance:", int(dist_tensor.max().item()))
        PRINTED_ANSWER_STATS = True

    return torch.tensor(margins, dtype=torch.float32), preds


# ----------------------------
# SECTION 3: EMS core
# ----------------------------

def compute_contrastive_jacobian_phi3(
    model,
    layer_idx: int,
    adv_pairs: List[Dict[str, Any]],
    letter_token_ids: torch.Tensor,
) -> torch.Tensor:
    """
    Directional Taylor scoring over a simulated 1-bit column crush.
    Matches the existing EMS math from notebooks/mednsq_probe.py.
    Uses dynamically computed answer position (last A/B/C/D token) per sample.
    """
    if not adv_pairs:
        raise RuntimeError("compute_contrastive_jacobian_phi3: adv_pairs is empty")

    device = next(model.parameters()).device
    weight = _down_proj_weight(model, layer_idx)

    # Freeze all params, enable grad for target weight only.
    original_requires_grad: Dict[str, bool] = {}
    for name, p in model.named_parameters():
        original_requires_grad[name] = bool(p.requires_grad)
        p.requires_grad = False
    weight.requires_grad = True

    grad_accum = torch.zeros_like(weight, dtype=torch.float32, device=weight.device)

    # NOTE: This loop is intentionally unbatched for correctness.
    # This is slow but preserves exact gradient behavior.
    model.zero_grad(set_to_none=True)
    try:
        for pair in adv_pairs:
            model.zero_grad(set_to_none=True)
            input_ids = pair["input_ids"].to(device)
            attention_mask = pair["attention_mask"].to(device)
            out = model(input_ids=input_ids, attention_mask=attention_mask)
            full_logits = out.logits
            answer_idx = _answer_position_single(input_ids, letter_token_ids)
            if answer_idx is None:
                continue
            seq_len = int(input_ids.shape[1])
            assert answer_idx >= 0, f"answer_idx {answer_idx} < 0"
            assert answer_idx < seq_len, f"answer_idx {answer_idx} >= seq_len {seq_len}"
            logits = full_logits[:, answer_idx, :]

            pos_id = int(pair["pos_id"])
            neg_id = int(pair["neg_id"])
            margin = logits[0, pos_id] - logits[0, neg_id]
            margin.backward()

            if weight.grad is not None:
                grad_accum += weight.grad.detach().to(torch.float32)
    finally:
        for name, p in model.named_parameters():
            p.requires_grad = original_requires_grad.get(name, False)

    with torch.no_grad():
        w32 = weight.detach().to(torch.float32)
        scale_per_col = w32.abs().mean(dim=0, keepdim=True)
        crushed = w32.sign() * scale_per_col
        delta = crushed - w32
        col_scores = torch.abs((grad_accum * delta).sum(dim=0))

    return col_scores


def simulate_column_crush_phi3(model, layer_idx: int, col_idx: int) -> torch.Tensor:
    weight = _down_proj_weight(model, layer_idx)
    if col_idx < 0 or col_idx >= weight.shape[1]:
        raise IndexError(f"col_idx out of range: {col_idx} for weight.shape={tuple(weight.shape)}")
    original_col = weight[:, col_idx].detach().clone()
    scale = original_col.abs().mean()
    crushed = original_col.sign() * scale
    with torch.no_grad():
        weight[:, col_idx] = crushed.to(dtype=weight.dtype, device=weight.device)
    return original_col


def restore_column_phi3(model, layer_idx: int, col_idx: int, original_col: torch.Tensor) -> float:
    weight = _down_proj_weight(model, layer_idx)
    with torch.no_grad():
        weight[:, col_idx] = original_col.to(dtype=weight.dtype, device=weight.device)
    restored = weight[:, col_idx].detach().to(original_col.dtype)
    max_diff = float((restored - original_col).abs().max().item())
    return max_diff


def compute_per_sample_margins_phi3(
    model,
    adv_pairs: List[Dict[str, Any]],
    letter_token_ids: torch.Tensor,
    tokenizer: Any = None,
) -> torch.Tensor:
    if not adv_pairs:
        return torch.zeros(0, dtype=torch.float32)
    margins, _ = batched_forward_pass(model, adv_pairs, letter_token_ids, tokenizer=tokenizer)
    margins = margins[~torch.isnan(margins)]
    return margins


def evaluate_column_ems_phi3(
    model,
    layer_idx: int,
    col_idx: int,
    adv_pairs: List[Dict[str, Any]],
    baseline_margins: torch.Tensor,
    letter_token_ids: torch.Tensor,
    early_stop_mu: float | None = None,
    early_stop_sigma: float | None = None,
    max_samples: int | None = None,
    track_flips: bool = False,
    tokenizer: Any = None,
) -> Tuple[torch.Tensor, float, float, float]:
    if not adv_pairs:
        raise RuntimeError("evaluate_column_ems_phi3: adv_pairs is empty")

    if max_samples is not None:
        adv_pairs = adv_pairs[:max_samples]
        baseline_margins = baseline_margins[:max_samples]

    if baseline_margins.ndim != 1:
        raise RuntimeError(f"baseline_margins must be rank-1, got {baseline_margins.shape}")
    if baseline_margins.numel() < len(adv_pairs):
        raise RuntimeError(
            f"baseline_margins shorter than adv_pairs: {baseline_margins.numel()} vs {len(adv_pairs)}"
        )

    original_col = simulate_column_crush_phi3(model, layer_idx, col_idx)
    crushed_margins: List[float] = []
    preds: List[int] = []
    processed = 0
    stop_early = False

    try:
        for start in range(0, len(adv_pairs), BATCH_SIZE):
            batch = adv_pairs[start : start + BATCH_SIZE]
            if not batch:
                continue
            margins_batch, preds_batch = batched_forward_pass(model, batch, letter_token_ids, tokenizer=tokenizer)

            for j, m in enumerate(margins_batch.tolist()):
                crushed_margins.append(float(m))
                if track_flips:
                    preds.append(int(preds_batch[j]))
                processed += 1

                if (
                    early_stop_mu is not None
                    and processed % 10 == 0
                    and processed <= int(baseline_margins.numel())
                ):
                    base_so_far = baseline_margins[:processed]
                    crushed_so_far = torch.tensor(crushed_margins[:processed], dtype=torch.float32)
                    mean_drop_so_far = float((base_so_far - crushed_so_far).mean().item())
                    threshold = float(early_stop_mu)
                    if early_stop_sigma is not None:
                        threshold = float(early_stop_mu) + 0.25 * float(early_stop_sigma)
                    if mean_drop_so_far < threshold:
                        stop_early = True
                        break

            if stop_early:
                break
    finally:
        max_diff = restore_column_phi3(model, layer_idx, col_idx, original_col)
        print(
            f"[RestoreCheck] layer={layer_idx} col={col_idx} max_diff={max_diff:.6e}"
        )
        if max_diff > 1e-3:
            raise RuntimeError(
                f"Column restoration failed: layer={layer_idx} col={col_idx} max_diff={max_diff:.6e}"
            )

    if not crushed_margins:
        return torch.zeros(0, dtype=torch.float32), 0.0, 0.0, 0.0

    crushed_tensor = torch.tensor(crushed_margins, dtype=torch.float32)
    valid_mask = ~torch.isnan(crushed_tensor)
    baseline_slice = baseline_margins[: crushed_tensor.numel()]
    crushed_tensor = crushed_tensor[valid_mask]
    baseline_valid = baseline_slice[valid_mask]
    if crushed_tensor.numel() == 0:
        return torch.zeros(0), 0.0, 0.0, 0.0
    drops = baseline_valid - crushed_tensor
    mean_drop = float(drops.mean().item())

    flip_rate = 0.0
    lethal_flip_rate = 0.0

    if track_flips and preds:
        # Compare predicted A/B/C/D token against pos/neg target tokens.
        letter_token_ids_cpu = letter_token_ids.detach().cpu()
        total = len(preds)
        flips = 0
        lethal = 0
        for i, pred_idx in enumerate(preds):
            pair = adv_pairs[i]
            correct_token = int(pair["pos_id"])
            neg_token = int(pair["neg_id"])
            pred_token = int(letter_token_ids_cpu[pred_idx].item())
            if pred_token != correct_token:
                flips += 1
                if pred_token == neg_token:
                    lethal += 1
        flip_rate = flips / total if total > 0 else 0.0
        lethal_flip_rate = lethal / total if total > 0 else 0.0

    return drops, mean_drop, flip_rate, lethal_flip_rate


def sanity_check_top_vs_random(
    model,
    layer_idx: int,
    adv_pairs: List[Dict[str, Any]],
    baseline_margins_all: torch.Tensor,
    letter_token_ids: torch.Tensor,
    tokenizer: Any,
    top_indices: torch.Tensor,
    num_cols: int,
    n_samples: int = 50,
) -> None:
    """Compare mean_drop of top-20 Taylor columns vs 20 random columns on a small subset."""
    n_samples = min(n_samples, len(adv_pairs))
    if n_samples == 0:
        return
    subset_pairs = adv_pairs[:n_samples]
    baseline_subset = baseline_margins_all[:n_samples]

    top20_cols = top_indices[:20].tolist()
    random_cols = torch.randperm(num_cols)[:20].tolist()

    top_drops: List[float] = []
    for col in top20_cols:
        _, mean_drop, _, _ = evaluate_column_ems_phi3(
            model=model,
            layer_idx=layer_idx,
            col_idx=int(col),
            adv_pairs=subset_pairs,
            baseline_margins=baseline_subset,
            letter_token_ids=letter_token_ids,
            early_stop_mu=None,
            max_samples=n_samples,
            track_flips=False,
            tokenizer=tokenizer,
        )
        top_drops.append(float(mean_drop))

    random_drops: List[float] = []
    for col in random_cols:
        _, mean_drop, _, _ = evaluate_column_ems_phi3(
            model=model,
            layer_idx=layer_idx,
            col_idx=int(col),
            adv_pairs=subset_pairs,
            baseline_margins=baseline_subset,
            letter_token_ids=letter_token_ids,
            early_stop_mu=None,
            max_samples=n_samples,
            track_flips=False,
            tokenizer=tokenizer,
        )
        random_drops.append(float(mean_drop))

    mean_drop_top = float(sum(top_drops) / len(top_drops)) if top_drops else 0.0
    mean_drop_random = float(sum(random_drops) / len(random_drops)) if random_drops else 0.0
    import numpy as np
    print("SANITY CHECK:")
    print("Top mean drop:", mean_drop_top)
    print("Random mean drop:", mean_drop_random)
    print("Top std:", float(np.std(top_drops)))
    print("Random std:", float(np.std(random_drops)))
    if mean_drop_top <= mean_drop_random:
        print("WARNING: EMS signal may be invalid")


def run_ems_for_layer_phi3(
    model,
    tokenizer,
    adv_pairs: List[Dict[str, Any]],
    evaluation_samples: List[Dict[str, Any]],
    layer_idx: int,
    cfg: EMSConfig,
    baseline_margins_all: torch.Tensor,
    baseline_eval: Dict[str, float],
    letter_token_ids: torch.Tensor,
) -> Dict[str, Any]:
    print(f"\n=== Processing layer {layer_idx} ===")

    assert len(adv_pairs) > 0, "adv_pairs is empty before EMS"
    assert baseline_margins_all.numel() > 0, "baseline_margins_all is empty before EMS"

    # Sanity check: print layer weight shape before EMS
    weight = _down_proj_weight(model, layer_idx)
    print("[WeightCheck] layer", layer_idx, "down_proj.weight shape:", tuple(weight.shape))

    # Stage 0: directional Taylor scores for all columns.
    jacobian_scores = compute_contrastive_jacobian_phi3(
        model, layer_idx, adv_pairs, letter_token_ids
    )
    num_cols = int(jacobian_scores.numel())
    print(f"[ProbeCheck] layer={layer_idx} num_cols={num_cols}")

    stage1_top_k = min(cfg.stage1_top_k, num_cols)
    top_vals, top_indices = torch.topk(jacobian_scores, k=stage1_top_k)
    print("[Taylor Top20]", top_vals[:20].detach().cpu().tolist())

    sanity_check_top_vs_random(
        model=model,
        layer_idx=layer_idx,
        adv_pairs=adv_pairs,
        baseline_margins_all=baseline_margins_all,
        letter_token_ids=letter_token_ids,
        tokenizer=tokenizer,
        top_indices=top_indices,
        num_cols=num_cols,
        n_samples=50,
    )

    stage1_samples = min(cfg.stage1_samples, len(adv_pairs))
    stage2_samples = min(cfg.stage2_samples, len(adv_pairs))
    baseline_stage1 = baseline_margins_all[:stage1_samples]

    # Random baseline columns for Z-score calibration.
    all_indices = torch.arange(num_cols, device=jacobian_scores.device)
    rand_cols = all_indices[torch.randperm(num_cols)[:RANDOM_BASELINE_COLS]].tolist()

    random_mean_drops: List[float] = []
    for col in rand_cols:
        _, mean_drop, _, _ = evaluate_column_ems_phi3(
            model=model,
            layer_idx=layer_idx,
            col_idx=int(col),
            adv_pairs=adv_pairs[:stage1_samples],
            baseline_margins=baseline_stage1,
            letter_token_ids=letter_token_ids,
            early_stop_mu=None,
            max_samples=stage1_samples,
            track_flips=False,
            tokenizer=tokenizer,
        )
        random_mean_drops.append(float(mean_drop))

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
            f"[Warning] sigma_rand={sigma_rand:.6f} is unusually large; Z-scores may be unstable."
        )

    # Stage 1 EMS on top-k columns with early stopping.
    stage1_candidates: List[Tuple[int, float]] = []
    for col in top_indices.tolist():
        _, mean_drop, _, _ = evaluate_column_ems_phi3(
            model=model,
            layer_idx=layer_idx,
            col_idx=int(col),
            adv_pairs=adv_pairs[:stage1_samples],
            baseline_margins=baseline_stage1,
            letter_token_ids=letter_token_ids,
            early_stop_mu=mu_rand,
            early_stop_sigma=sigma_rand,
            max_samples=stage1_samples,
            track_flips=False,
            tokenizer=tokenizer,
        )
        if mean_drop > 0.0:
            stage1_candidates.append((int(col), float(mean_drop)))

    stage1_candidates.sort(key=lambda x: x[1], reverse=True)
    stage2_columns = [c for c, _ in stage1_candidates[: cfg.stage2_top_k]]

    print(
        f"Stage 1 retained {len(stage2_columns)} columns for Stage 2 EMS (top_k={cfg.stage2_top_k})."
    )

    # Stage 2 EMS with lethal flip tracking.
    validated_anchors: List[Dict[str, Any]] = []
    stage2_results: List[Dict[str, Any]] = []
    adv_pairs_stage2 = adv_pairs[:stage2_samples]
    baseline_stage2 = baseline_margins_all[:stage2_samples]

    for col in stage2_columns:
        drops, mean_drop, flip_rate, lethal_flip_rate = evaluate_column_ems_phi3(
            model=model,
            layer_idx=layer_idx,
            col_idx=int(col),
            adv_pairs=adv_pairs_stage2,
            baseline_margins=baseline_stage2,
            letter_token_ids=letter_token_ids,
            early_stop_mu=None,
            max_samples=stage2_samples,
            track_flips=True,
            tokenizer=tokenizer,
        )

        z = (float(mean_drop) - mu_rand) / (sigma_rand + 1e-8)
        median_drop = float(torch.median(drops).item()) if drops.numel() else 0.0

        result = {
            "layer": int(layer_idx),
            "column": int(col),
            "mean_drop": float(mean_drop),
            "median_drop": float(median_drop),
            "z_score": float(z),
            "flip_rate": float(flip_rate),
            "lethal_flip_rate": float(lethal_flip_rate),
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

    if validated_anchors:
        max_drop = max(a["mean_drop"] for a in validated_anchors)
        mean_drop_validated = float(sum(a["mean_drop"] for a in validated_anchors) / len(validated_anchors))
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
        "layer": int(layer_idx),
        "mu_rand": float(mu_rand),
        "sigma_rand": float(sigma_rand),
        "validated_anchors": validated_anchors,
        "stage2_results": stage2_results,
        "max_drop": float(max_drop),
        "mean_drop": float(mean_drop_validated),
        "max_z": float(max_z),
        "baseline_eval": baseline_eval,
    }


# ----------------------------
# SECTION 4: Experiments
# ----------------------------

def run_multi_anchor_ablation_sweep_phi3(
    model,
    tokenizer,
    anchors: List[Tuple[int, int]],
    seeds: List[int],
    calibration_size: int,
    eval_size: int,
    letter_token_ids: torch.Tensor,
) -> None:
    print("\n=== Multi Anchor Ablation Sweep ===")
    ablation_counts = [1, 4, 8, 16, 32, 48, 64]
    sweep_results: Dict[int, Dict[str, List[float]]] = {k: {"accuracies": [], "margins": []} for k in ablation_counts}

    from mednsq_data import load_mcq_dataset, build_adversarial_pairs
    from mednsq_eval import evaluate_model

    for seed in seeds:
        _set_reproducible(seed)
        total_needed = calibration_size + eval_size
        samples = load_mcq_dataset(n_total=total_needed)
        random.shuffle(samples)
        calibration = samples[:calibration_size]
        evaluation = samples[calibration_size : calibration_size + eval_size]

        adv_pairs = build_adversarial_pairs(model, tokenizer, calibration, n_calib=len(calibration))
        _assert_adv_pairs_shape(adv_pairs)

        for count in ablation_counts:
            if count > len(anchors):
                continue
            subset = anchors[:count]
            saved: List[Tuple[int, int, torch.Tensor]] = []
            try:
                for layer_idx, col_idx in subset:
                    original_col = simulate_column_crush_phi3(model, int(layer_idx), int(col_idx))
                    saved.append((int(layer_idx), int(col_idx), original_col))

                eval_metrics = evaluate_model(model, tokenizer, evaluation)
                accuracy = float(eval_metrics.get("accuracy", 0.0))

                margins = compute_per_sample_margins_phi3(
                    model, adv_pairs, letter_token_ids, tokenizer=tokenizer
                )
                mean_margin = float(margins.mean().item()) if margins.numel() else 0.0

                sweep_results[count]["accuracies"].append(float(accuracy))
                sweep_results[count]["margins"].append(float(mean_margin))
            finally:
                for layer_idx, col_idx, original_col in saved:
                    max_diff = restore_column_phi3(model, layer_idx, col_idx, original_col)
                    print(f"[RestoreCheck] layer={layer_idx} col={col_idx} max_diff={max_diff:.6e}")
                    if max_diff > 1e-3:
                        raise RuntimeError(
                            f"Column restoration failed: layer={layer_idx} col={col_idx} max_diff={max_diff:.6e}"
                        )

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

    for count in ablation_counts:
        accs = sweep_results[count]["accuracies"]
        margins = sweep_results[count]["margins"]
        if not accs:
            continue
        mean_acc = float(sum(accs) / len(accs))
        mean_margin = float(sum(margins) / len(margins)) if margins else 0.0
        print(f"\nAnchors removed: {count}")
        print(f"Mean accuracy: {mean_acc}")
        print(f"Mean margin: {mean_margin}")
        log_progress(f"Multi-anchor ablation count={count} mean_accuracy={mean_acc} mean_margin={mean_margin}")


def run_random_neuron_ablation_baseline_phi3(
    model,
    tokenizer,
    layers: List[int],
    seeds: List[int],
    calibration_size: int,
    eval_size: int,
    k_values: List[int],
    letter_token_ids: torch.Tensor,
) -> None:
    print("\n=== Random Neuron Ablation Baseline ===")
    sweep_results: Dict[int, Dict[str, List[float]]] = {k: {"accuracies": [], "margins": []} for k in k_values}

    from mednsq_data import load_mcq_dataset, build_adversarial_pairs
    from mednsq_eval import evaluate_model

    # Use weight shape from the first layer in the list for column range.
    w0 = _down_proj_weight(model, int(layers[0]))
    n_cols = int(w0.shape[1])

    for seed in seeds:
        _set_reproducible(seed)
        total_needed = calibration_size + eval_size
        samples = load_mcq_dataset(n_total=total_needed)
        random.shuffle(samples)
        calibration = samples[:calibration_size]
        evaluation = samples[calibration_size : calibration_size + eval_size]

        adv_pairs = build_adversarial_pairs(model, tokenizer, calibration, n_calib=len(calibration))
        _assert_adv_pairs_shape(adv_pairs)

        for k in k_values:
            random_neurons_set: set[Tuple[int, int]] = set()
            while len(random_neurons_set) < int(k):
                random_neurons_set.add((int(random.choice(layers)), int(random.randint(0, n_cols - 1))))
            random_neurons = list(random_neurons_set)
            saved: List[Tuple[int, int, torch.Tensor]] = []
            try:
                for layer_idx, col_idx in random_neurons:
                    original_col = simulate_column_crush_phi3(model, layer_idx, col_idx)
                    saved.append((layer_idx, col_idx, original_col))

                eval_metrics = evaluate_model(model, tokenizer, evaluation)
                accuracy = float(eval_metrics.get("accuracy", 0.0))

                margins = compute_per_sample_margins_phi3(
                    model, adv_pairs, letter_token_ids, tokenizer=tokenizer
                )
                mean_margin = float(margins.mean().item()) if margins.numel() else 0.0

                sweep_results[k]["accuracies"].append(float(accuracy))
                sweep_results[k]["margins"].append(float(mean_margin))
            finally:
                for layer_idx, col_idx, original_col in saved:
                    max_diff = restore_column_phi3(model, layer_idx, col_idx, original_col)
                    print(f"[RestoreCheck] layer={layer_idx} col={col_idx} max_diff={max_diff:.6e}")
                    if max_diff > 1e-3:
                        raise RuntimeError(
                            f"Column restoration failed: layer={layer_idx} col={col_idx} max_diff={max_diff:.6e}"
                        )

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

    for k in k_values:
        accs = sweep_results[k]["accuracies"]
        margins = sweep_results[k]["margins"]
        if not accs:
            continue
        mean_acc = float(sum(accs) / len(accs))
        mean_margin = float(sum(margins) / len(margins)) if margins else 0.0
        print(f"\nRandom neurons removed: {k}")
        print(f"Mean accuracy: {mean_acc}")
        print(f"Mean margin: {mean_margin}")
        log_progress(f"Random neuron ablation k={k} mean_accuracy={mean_acc} mean_margin={mean_margin}")


def _run_forward_no_hooks(
    model,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    letter_token_ids: torch.Tensor,
) -> Optional[torch.Tensor]:
    with torch.no_grad():
        out = model(input_ids=input_ids, attention_mask=attention_mask)
    answer_idx = _compute_answer_idx_from_ids(input_ids, letter_token_ids)
    if answer_idx is None:
        return None
    return out.logits[:, answer_idx, :]


def run_anchor_activation_patching_experiment_phi3(
    model,
    tokenizer,
    anchors: List[Tuple[int, int]],
    calibration_samples: List[Dict[str, Any]],
    letter_token_ids: torch.Tensor,
    n_samples: int = 120,
    per_anchor_report_only: bool = False,
) -> None:
    """
    Counterfactual activation patching on anchor neurons:
      - run safe prompt, capture anchor activations at down_proj *input* (pre-hook)
      - run adversarial prompt baseline
      - run adversarial prompt with captured activations patched in

    # NOTE:
    # This patches only final-token activations.
    # If results are weak, this does NOT invalidate anchors.
    # Behavior may be distributed across tokens.
    """
    from mednsq_data import build_adversarial_pairs

    model.eval()
    device = next(model.parameters()).device

    adv_pairs = build_adversarial_pairs(model, tokenizer, calibration_samples, n_calib=len(calibration_samples))
    _assert_adv_pairs_shape(adv_pairs)
    n_samples = min(int(n_samples), len(adv_pairs))
    if n_samples == 0:
        print("No samples for anchor activation patching.")
        return

    layers = _get_layers(model)
    intermediate_size = int(model.config.intermediate_size)

    anchors_by_layer: Dict[int, List[int]] = {}
    for layer_idx, col_idx in anchors:
        anchors_by_layer.setdefault(int(layer_idx), []).append(int(col_idx))

    def _left_pad_to_length(ids: torch.Tensor, mask: torch.Tensor, target_len: int) -> Tuple[torch.Tensor, torch.Tensor]:
        seq_len = int(ids.shape[1])
        if seq_len >= target_len:
            return ids.to(device), mask.to(device)
        batch = int(ids.shape[0])
        new_ids = torch.full((batch, target_len), int(PAD_TOKEN_ID), dtype=ids.dtype, device=device)
        new_mask = torch.zeros(batch, target_len, dtype=mask.dtype, device=device)
        new_ids[:, -seq_len:] = ids.to(device)
        new_mask[:, -seq_len:] = mask.to(device)
        return new_ids, new_mask

    def _run_with_save_hooks(stored: Dict[Tuple[int, int], torch.Tensor], input_ids, attention_mask) -> Optional[torch.Tensor]:
        answer_idx = _compute_answer_idx_from_ids(input_ids, letter_token_ids)
        if answer_idx is None:
            return None
        handles = []
        for layer_idx, cols in anchors_by_layer.items():
            down_proj = layers[layer_idx].mlp.down_proj

            def _save_hook(module, inputs, layer=layer_idx, cols_list=cols, ans_idx=answer_idx):
                h = inputs[0]
                for col in cols_list:
                    stored[(layer, col)] = h[0, ans_idx, col].detach().clone()

            handles.append(down_proj.register_forward_pre_hook(_save_hook))
        try:
            with torch.no_grad():
                out = model(input_ids=input_ids, attention_mask=attention_mask)
            return out.logits[:, answer_idx, :]
        finally:
            for h in handles:
                h.remove()

    def _run_with_patch_hooks(stored: Dict[Tuple[int, int], torch.Tensor], input_ids, attention_mask) -> Optional[torch.Tensor]:
        answer_idx = _compute_answer_idx_from_ids(input_ids, letter_token_ids)
        if answer_idx is None:
            return None
        handles = []
        for layer_idx, cols in anchors_by_layer.items():
            down_proj = layers[layer_idx].mlp.down_proj

            def _patch_hook(module, inputs, layer=layer_idx, cols_list=cols, ans_idx=answer_idx):
                h = inputs[0]
                out = h.clone()
                for col in cols_list:
                    if (layer, col) in stored:
                        out[0, ans_idx, col] = stored[(layer, col)].to(dtype=out.dtype, device=out.device)
                return (out,)

            handles.append(down_proj.register_forward_pre_hook(_patch_hook))
        try:
            with torch.no_grad():
                out = model(input_ids=input_ids, attention_mask=attention_mask)
            return out.logits[:, answer_idx, :]
        finally:
            for h in handles:
                h.remove()

    def _margin_and_correct(logits: torch.Tensor, pair: Dict[str, Any]) -> Tuple[float, int]:
        pos_id = int(pair["pos_id"])
        neg_id = int(pair["neg_id"])
        margin = float((logits[0, pos_id] - logits[0, neg_id]).item())
        letter_logits = logits[0].index_select(0, letter_token_ids.to(device))
        pred_idx = int(torch.argmax(letter_logits).item())
        pred_token = int(letter_token_ids[pred_idx].item())
        correct = 1 if pred_token == pos_id else 0
        return margin, correct

    # ---- Anchor patching ----
    margin_shifts_anchor: List[float] = []
    correct_baseline_anchor = 0
    correct_patched_anchor = 0

    with torch.no_grad():
        for i in range(n_samples):
            pair = adv_pairs[i]
            safe_ids = pair["safe_input_ids"].to(device)
            safe_mask = pair["safe_attention_mask"].to(device)
            corrupt_ids = pair["input_ids"].to(device)
            corrupt_mask = pair["attention_mask"].to(device)

            target_len = max(int(safe_ids.shape[1]), int(corrupt_ids.shape[1]))
            safe_ids_pad, safe_mask_pad = _left_pad_to_length(safe_ids, safe_mask, target_len)
            corrupt_ids_pad, corrupt_mask_pad = _left_pad_to_length(corrupt_ids, corrupt_mask, target_len)

            stored: Dict[Tuple[int, int], torch.Tensor] = {}
            if _run_with_save_hooks(stored, safe_ids_pad, safe_mask_pad) is None:
                continue
            logits_base = _run_forward_no_hooks(model, corrupt_ids_pad, corrupt_mask_pad, letter_token_ids)
            if logits_base is None:
                continue
            logits_patched = _run_with_patch_hooks(stored, corrupt_ids_pad, corrupt_mask_pad)
            if logits_patched is None:
                continue
            base_margin, base_correct = _margin_and_correct(logits_base, pair)
            patched_margin, patched_correct = _margin_and_correct(logits_patched, pair)

            margin_shifts_anchor.append(float(patched_margin - base_margin))
            correct_baseline_anchor += int(base_correct)
            correct_patched_anchor += int(patched_correct)

    mean_margin_shift_anchor = float(np.mean(margin_shifts_anchor)) if margin_shifts_anchor else 0.0
    acc_base = correct_baseline_anchor / n_samples
    acc_patch = correct_patched_anchor / n_samples
    accuracy_change_anchor = float(acc_patch - acc_base)

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
    random_neurons_set: set[Tuple[int, int]] = set()
    while len(random_neurons_set) < n_random:
        random_neurons_set.add((int(random.choice(anchor_layers)), int(random.randint(0, intermediate_size - 1))))
    random_neurons = list(random_neurons_set)

    random_by_layer: Dict[int, List[int]] = {}
    for layer_idx, col_idx in random_neurons:
        random_by_layer.setdefault(int(layer_idx), []).append(int(col_idx))

    def _run_with_save_hooks_random(stored: Dict[Tuple[int, int], torch.Tensor], input_ids, attention_mask) -> None:
        answer_idx = _compute_answer_idx_from_ids(input_ids, letter_token_ids)
        if answer_idx is None:
            return
        handles = []
        for layer_idx, cols in random_by_layer.items():
            down_proj = layers[layer_idx].mlp.down_proj

            def _save_hook(module, inputs, layer=layer_idx, cols_list=cols, ans_idx=answer_idx):
                h = inputs[0]
                for col in cols_list:
                    stored[(layer, col)] = h[0, ans_idx, col].detach().clone()

            handles.append(down_proj.register_forward_pre_hook(_save_hook))
        try:
            with torch.no_grad():
                model(input_ids=input_ids, attention_mask=attention_mask)
        finally:
            for h in handles:
                h.remove()

    def _run_with_patch_hooks_random(stored: Dict[Tuple[int, int], torch.Tensor], input_ids, attention_mask) -> Optional[torch.Tensor]:
        answer_idx = _compute_answer_idx_from_ids(input_ids, letter_token_ids)
        if answer_idx is None:
            return None
        handles = []
        for layer_idx, cols in random_by_layer.items():
            down_proj = layers[layer_idx].mlp.down_proj

            def _patch_hook(module, inputs, layer=layer_idx, cols_list=cols, ans_idx=answer_idx):
                h = inputs[0]
                out = h.clone()
                for col in cols_list:
                    if (layer, col) in stored:
                        out[0, ans_idx, col] = stored[(layer, col)].to(dtype=out.dtype, device=out.device)
                return (out,)

            handles.append(down_proj.register_forward_pre_hook(_patch_hook))
        try:
            with torch.no_grad():
                out = model(input_ids=input_ids, attention_mask=attention_mask)
            return out.logits[:, answer_idx, :]
        finally:
            for h in handles:
                h.remove()

    margin_shifts_random: List[float] = []
    correct_baseline_random = 0
    correct_patched_random = 0

    with torch.no_grad():
        for i in range(n_samples):
            pair = adv_pairs[i]
            safe_ids = pair["safe_input_ids"].to(device)
            safe_mask = pair["safe_attention_mask"].to(device)
            corrupt_ids = pair["input_ids"].to(device)
            corrupt_mask = pair["attention_mask"].to(device)

            target_len = max(int(safe_ids.shape[1]), int(corrupt_ids.shape[1]))
            safe_ids_pad, safe_mask_pad = _left_pad_to_length(safe_ids, safe_mask, target_len)
            corrupt_ids_pad, corrupt_mask_pad = _left_pad_to_length(corrupt_ids, corrupt_mask, target_len)

            stored_r: Dict[Tuple[int, int], torch.Tensor] = {}
            _run_with_save_hooks_random(stored_r, safe_ids_pad, safe_mask_pad)
            if not stored_r:
                continue
            logits_base = _run_forward_no_hooks(model, corrupt_ids_pad, corrupt_mask_pad, letter_token_ids)
            if logits_base is None:
                continue
            logits_patched = _run_with_patch_hooks_random(stored_r, corrupt_ids_pad, corrupt_mask_pad)
            if logits_patched is None:
                continue
            base_margin, base_correct = _margin_and_correct(logits_base, pair)
            patched_margin, patched_correct = _margin_and_correct(logits_patched, pair)

            margin_shifts_random.append(float(patched_margin - base_margin))
            correct_baseline_random += int(base_correct)
            correct_patched_random += int(patched_correct)

    mean_margin_shift_random = float(np.mean(margin_shifts_random)) if margin_shifts_random else 0.0
    acc_base_r = correct_baseline_random / n_samples
    acc_patch_r = correct_patched_random / n_samples
    accuracy_change_random = float(acc_patch_r - acc_base_r)

    print("\n=== Random Neuron Patching ===")
    print(f"Mean margin shift: {mean_margin_shift_random}")
    print(f"Accuracy change: {accuracy_change_random}")


def run_attention_head_ablation_sweep_phi3(
    model,
    tokenizer,
    layers_to_ablate: List[int],
    seeds: List[int],
    calibration_size: int,
    eval_size: int,
    k_values: List[int],
    num_heads: int,
    head_dim: int,
    letter_token_ids: torch.Tensor,
) -> None:
    print("\n=== Attention Head Ablation Sweep ===")
    from mednsq_data import load_mcq_dataset, build_adversarial_pairs
    from mednsq_eval import evaluate_model

    sweep_results: Dict[int, Dict[str, List[float]]] = {k: {"accuracies": [], "margins": []} for k in k_values}
    layer_stack = _get_layers(model)

    for seed in seeds:
        _set_reproducible(seed)
        total_needed = calibration_size + eval_size
        samples = load_mcq_dataset(n_total=total_needed)
        random.shuffle(samples)
        calibration = samples[:calibration_size]
        evaluation = samples[calibration_size : calibration_size + eval_size]

        adv_pairs = build_adversarial_pairs(model, tokenizer, calibration, n_calib=len(calibration))
        _assert_adv_pairs_shape(adv_pairs)

        all_heads = [(int(layer), int(head)) for layer in layers_to_ablate for head in range(int(num_heads))]

        for k in k_values:
            if int(k) > len(all_heads):
                continue
            random_heads = random.sample(all_heads, int(k))
            saved_slices: List[Tuple[int, int, torch.Tensor]] = []
            try:
                with torch.no_grad():
                    for layer_idx, head_idx in random_heads:
                        o_w = layer_stack[layer_idx].self_attn.o_proj.weight
                        start = int(head_idx) * int(head_dim)
                        end = (int(head_idx) + 1) * int(head_dim)
                        saved = o_w[:, start:end].clone()
                        saved_slices.append((layer_idx, head_idx, saved))
                        o_w[:, start:end] = 0

                eval_metrics = evaluate_model(model, tokenizer, evaluation)
                accuracy = float(eval_metrics.get("accuracy", 0.0))

                margins = compute_per_sample_margins_phi3(
                    model, adv_pairs, letter_token_ids, tokenizer=tokenizer
                )
                mean_margin = float(margins.mean().item()) if margins.numel() else 0.0

                sweep_results[int(k)]["accuracies"].append(float(accuracy))
                sweep_results[int(k)]["margins"].append(float(mean_margin))
            finally:
                with torch.no_grad():
                    for layer_idx, head_idx, saved in saved_slices:
                        o_w = layer_stack[layer_idx].self_attn.o_proj.weight
                        start = int(head_idx) * int(head_dim)
                        end = (int(head_idx) + 1) * int(head_dim)
                        o_w[:, start:end] = saved.to(dtype=o_w.dtype, device=o_w.device)

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

    for k in k_values:
        accs = sweep_results[int(k)]["accuracies"]
        margins = sweep_results[int(k)]["margins"]
        if not accs:
            continue
        mean_acc = float(sum(accs) / len(accs))
        mean_margin = float(sum(margins) / len(margins)) if margins else 0.0
        print(f"\nHeads removed: {k}")
        print(f"Mean accuracy: {mean_acc}")
        print(f"Mean margin: {mean_margin}")
        log_progress(f"Attention head ablation k={k} mean_accuracy={mean_acc} mean_margin={mean_margin}")


def run_head_to_anchor_attribution_phi3(
    model,
    tokenizer,
    anchors: List[Tuple[int, int]],
    calibration_samples: List[Dict[str, Any]],
    letter_token_ids: torch.Tensor,
    n_prompts: int,
    layer_range: Tuple[int, int],
    num_heads: int,
    head_dim: int,
    output_path: str = "head_anchor_attribution.json",
    batch_size: int = 8,
) -> None:
    from mednsq_data import format_prompt

    print("\n=== Head-to-Anchor Attribution ===")
    model.eval()
    device = next(model.parameters()).device
    layers = _get_layers(model)

    anchors_by_layer: Dict[int, List[int]] = {}
    for layer_idx, col_idx in anchors:
        anchors_by_layer.setdefault(int(layer_idx), []).append(int(col_idx))
    if not anchors_by_layer:
        print("No anchors for head-to-anchor attribution.")
        return

    n_prompts = min(int(n_prompts), len(calibration_samples))
    prompts: List[Tuple[torch.Tensor, torch.Tensor]] = []
    with torch.no_grad():
        for i in range(n_prompts):
            sample = calibration_samples[i]
            prompt = format_prompt(sample["question"], sample["options"])
            enc = tokenizer(prompt, return_tensors="pt", add_special_tokens=True)
            prompts.append((enc["input_ids"].to(device), enc["attention_mask"].to(device)))

    def _pad_batch(items: List[Tuple[torch.Tensor, torch.Tensor]]) -> Tuple[torch.Tensor, torch.Tensor]:
        max_len = max(int(ids.shape[1]) for ids, _ in items)
        ids_out = []
        mask_out = []
        for ids, mask in items:
            seq_len = int(ids.shape[1])
            if seq_len < max_len:
                pad_ids = torch.full((ids.shape[0], max_len - seq_len), int(PAD_TOKEN_ID), dtype=ids.dtype, device=ids.device)
                pad_mask = torch.zeros(ids.shape[0], max_len - seq_len, dtype=mask.dtype, device=mask.device)
                ids = torch.cat([ids, pad_ids], dim=1)
                mask = torch.cat([mask, pad_mask], dim=1)
            ids_out.append(ids)
            mask_out.append(mask)
        return torch.cat(ids_out, dim=0), torch.cat(mask_out, dim=0)

    batches: List[Tuple[torch.Tensor, torch.Tensor]] = []
    for start in range(0, len(prompts), int(batch_size)):
        batches.append(_pad_batch(prompts[start : start + int(batch_size)]))

    def _run_and_capture_anchors(store: Dict[Tuple[int, int], List[torch.Tensor]]) -> None:
        letter_set = set(int(x) for x in letter_token_ids.cpu().tolist())
        with torch.no_grad():
            for input_ids, attention_mask in batches:
                B = input_ids.shape[0]
                answer_idxs: List[Optional[int]] = []
                for i in range(B):
                    seq = input_ids[i].tolist()
                    positions = [j for j in range(len(seq)) if seq[j] in letter_set]
                    if not positions:
                        print("WARNING: skipping sample, no answer token found")
                        answer_idxs.append(None)
                        continue
                    answer_idxs.append(max(positions))

                def _make_capture(ans_idxs: List[Optional[int]], layer_idx: int, anchor_cols: List[int]):
                    def _capture(module, inputs, layer=layer_idx, cols=anchor_cols):
                        h = inputs[0]
                        for b in range(h.shape[0]):
                            if ans_idxs[b] is None:
                                continue
                            ans_idx = ans_idxs[b]
                            for col in cols:
                                store[(layer, col)].append(h[b, ans_idx, col].detach().cpu())
                    return _capture

                handles = []
                for layer_idx, cols in anchors_by_layer.items():
                    down_proj = layers[layer_idx].mlp.down_proj
                    hook = _make_capture(answer_idxs, layer_idx, cols)
                    handles.append(down_proj.register_forward_pre_hook(hook))
                try:
                    model(input_ids=input_ids, attention_mask=attention_mask)
                finally:
                    for h in handles:
                        h.remove()

    stored_baseline: Dict[Tuple[int, int], List[torch.Tensor]] = {(l, c): [] for (l, c) in anchors}
    _run_and_capture_anchors(stored_baseline)
    baseline_anchor_mean: Dict[Tuple[int, int], float] = {}
    for (layer_idx, col_idx) in anchors:
        stacked = torch.cat(stored_baseline[(int(layer_idx), int(col_idx))], dim=0)
        baseline_anchor_mean[(int(layer_idx), int(col_idx))] = float(stacked.mean().item())

    results: List[Dict[str, Any]] = []
    layer_start, layer_end = int(layer_range[0]), int(layer_range[1])

    with torch.no_grad():
        for layer_idx in range(layer_start, layer_end):
            o_w = layers[layer_idx].self_attn.o_proj.weight
            for head_idx in range(int(num_heads)):
                start = int(head_idx) * int(head_dim)
                end = (int(head_idx) + 1) * int(head_dim)
                saved = o_w[:, start:end].clone()
                o_w[:, start:end] = 0

                stored_after: Dict[Tuple[int, int], List[torch.Tensor]] = {(l, c): [] for (l, c) in anchors}
                _run_and_capture_anchors(stored_after)

                for (al, ac) in anchors:
                    stacked = torch.cat(stored_after[(int(al), int(ac))], dim=0)
                    after_mean = float(stacked.mean().item())
                    delta = abs(after_mean - baseline_anchor_mean[(int(al), int(ac))])
                    results.append(
                        {
                            "head_layer": int(layer_idx),
                            "head": int(head_idx),
                            "anchor_layer": int(al),
                            "anchor_column": int(ac),
                            "delta": float(delta),
                        }
                    )

                o_w[:, start:end] = saved.to(dtype=o_w.dtype, device=o_w.device)

    results.sort(key=lambda x: x["delta"], reverse=True)
    print("Top 15 head → anchor interactions:")
    for r in results[:15]:
        print(
            f"  Head {r['head_layer']}.{r['head']} → Anchor {r['anchor_layer']}.{r['anchor_column']} delta {r['delta']:.2f}"
        )
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {output_path}")


# ----------------------------
# SECTION 5: Main
# ----------------------------

def main() -> None:
    _ensure_import_paths()

    # Import unchanged external modules with expected interfaces.
    from mednsq_data import load_mcq_dataset, build_adversarial_pairs
    from mednsq_eval import evaluate_model, _get_letter_token_ids

    # Load tokenizer once.
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME, use_fast=False, trust_remote_code=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    global PAD_TOKEN_ID
    PAD_TOKEN_ID = int(getattr(tokenizer, "pad_token_id", None) or getattr(tokenizer, "eos_token_id", 0) or 0)

    cfg = EMSConfig()

    print("=== Experiment Configuration ===")
    print("Model:", MODEL_NAME)
    print("Calibration size:", cfg.calibration_size)
    print("Stage1 top_k:", cfg.stage1_top_k)
    print("Stage1 samples:", cfg.stage1_samples)
    print("Stage2 top_k:", cfg.stage2_top_k)
    print("Stage2 samples:", cfg.stage2_samples)
    print("Random baseline cols:", RANDOM_BASELINE_COLS)
    print("Batch size:", BATCH_SIZE)
    print("Pad token id:", PAD_TOKEN_ID)
    print("Layers to test:", LAYERS_TO_TEST)
    print("===============================")

    log_progress(
        f"START {datetime.now()} model={MODEL_NAME} batch={BATCH_SIZE} calib={cfg.calibration_size} layers={LAYERS_TO_TEST}"
    )

    # Model/device strategy: single-device (required by explicit model.device tensor moves).
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch_dtype = torch.bfloat16 if device.type == "cuda" else torch.float32

    # Load model ONCE and reuse across seeds/experiments (dimension constants set once).
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch_dtype,
        trust_remote_code=True,
    )
    model.to(device)
    model.eval()

    # Device safety: ensure ALL model parameters are on the same device.
    device = next(model.parameters()).device
    print("Model device:", device)
    for name, p in model.named_parameters():
        if p.device != device:
            raise RuntimeError(f"Parameter {name} is on {p.device}, expected {device}")

    # Mandatory sanity check after model load: dimensions printed once, and layer weight shape.
    hidden_size, num_layers, num_heads, intermediate_size, head_dim = _print_dimension_summary_once(model)
    _ = hidden_size, num_layers, num_heads, intermediate_size, head_dim  # dims set once; do not recompute later.

    layers = _get_layers(model)
    print("Num layers:", len(layers))
    print("Down proj shape:", layers[0].mlp.down_proj.weight.shape)

    letter_token_ids = _get_letter_token_ids(tokenizer)
    _assert_letter_token_ids(letter_token_ids)
    letter_token_ids = letter_token_ids.to(device)

    # Checkpoint support for seed sweep.
    checkpoint = load_checkpoint()
    if checkpoint:
        completed_seeds = list(checkpoint.get("completed_seeds", []))
        all_seed_results = list(checkpoint.get("all_seed_results", []))
    else:
        completed_seeds = []
        all_seed_results = []

    for seed in SEEDS:
        if seed in completed_seeds:
            log_progress(f"Seed {seed} already completed, skipping")
            continue

        log_progress(f"Seed {seed} started")
        print(f"\n===== RUNNING SEED {seed} =====")

        _set_reproducible(seed)

        # Load data.
        eval_size = 60
        total_needed = cfg.calibration_size + eval_size
        samples = load_mcq_dataset(n_total=total_needed)
        print("[Data] dataset size:", len(samples))
        random.shuffle(samples)
        calibration = samples[: cfg.calibration_size]
        evaluation = samples[cfg.calibration_size : cfg.calibration_size + eval_size]

        adv_pairs = build_adversarial_pairs(model, tokenizer, calibration, n_calib=len(calibration))
        _assert_adv_pairs_shape(adv_pairs)

        print("Filtering invalid samples...")
        valid_pairs = []
        for pair in adv_pairs:
            idx = _answer_position_single(pair["input_ids"], letter_token_ids)
            if idx is not None:
                valid_pairs.append(pair)
        print(f"Kept {len(valid_pairs)} / {len(adv_pairs)} samples")
        adv_pairs = valid_pairs

        # Baselines.
        baseline_margins_all = compute_per_sample_margins_phi3(
            model, adv_pairs, letter_token_ids, tokenizer=tokenizer
        )
        baseline_margins_all = baseline_margins_all[~torch.isnan(baseline_margins_all)]
        print("Valid baseline samples:", baseline_margins_all.numel())

        baseline_eval = evaluate_model(model, tokenizer, evaluation)
        print("\n=== Baseline Held-out Evaluation ===")
        print("Baseline accuracy:", baseline_eval.get("accuracy", 0.0))
        print("Baseline mean margin:", baseline_eval.get("mean_margin", 0.0))
        accuracy = float(baseline_eval.get("accuracy", 0.0))
        mean_margin = float(baseline_eval.get("mean_margin", 0.0))
        if accuracy > 0.9:
            print("WARNING: dataset may be too easy")
        if accuracy < 0.25:
            print("WARNING: dataset may be too noisy")
        if abs(mean_margin) < MARGIN_WARNING_THRESHOLD:
            print("WARNING: margins are too small")

        # EMS discovery per layer.
        all_layer_summaries: List[Dict[str, Any]] = []
        for layer_idx in LAYERS_TO_TEST:
            log_progress(f"Seed {seed} Layer {layer_idx} starting EMS")
            summary = run_ems_for_layer_phi3(
                model=model,
                tokenizer=tokenizer,
                adv_pairs=adv_pairs,
                evaluation_samples=evaluation,
                layer_idx=int(layer_idx),
                cfg=cfg,
                baseline_margins_all=baseline_margins_all,
                baseline_eval=baseline_eval,
                letter_token_ids=letter_token_ids,
            )
            all_layer_summaries.append(summary)
            num_anchors = len(summary["validated_anchors"])
            max_drop = float(summary.get("max_drop", 0.0))
            log_progress(f"Seed {seed} Layer {layer_idx} anchors={num_anchors} max_drop={max_drop:.4f}")

        print("\n=== Seed Summary ===")
        print(f"seed={seed}")
        for layer_idx in LAYERS_TO_TEST:
            layer_summary = next(s for s in all_layer_summaries if s["layer"] == int(layer_idx))
            print(f"layer{layer_idx} anchors={len(layer_summary['validated_anchors'])}")

        seed_summary = {"seed": int(seed), "layers": all_layer_summaries}
        all_seed_results.append(seed_summary)
        completed_seeds.append(seed)
        save_checkpoint({"completed_seeds": completed_seeds, "all_seed_results": all_seed_results})
        log_progress(f"Seed {seed} completed, checkpoint saved")

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

    # Anchor stability summary.
    print("\n=== Anchor Stability Summary ===")
    log_progress("Anchor Stability Summary:")
    for layer_idx in LAYERS_TO_TEST:
        counts: List[int] = []
        for seed_result in all_seed_results:
            layer_summary = next(s for s in seed_result["layers"] if s["layer"] == int(layer_idx))
            counts.append(len(layer_summary["validated_anchors"]))
        mean_anchors = float(sum(counts) / len(counts)) if counts else 0.0
        print(f"layer{layer_idx} mean anchors = {mean_anchors:.3f}")
        log_progress(f"  layer{layer_idx} mean anchors = {mean_anchors:.3f}")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    with open(f"ems_seed_sweep_{timestamp}.json", "w", encoding="utf-8") as f:
        json.dump(all_seed_results, f, indent=2)

    # Gather anchors across seeds; dedupe and keep top 64 by score = 0.5*mean_drop + 0.5*median_drop.
    anchor_to_stats: Dict[Tuple[int, int], Dict[str, float]] = {}
    for seed_result in all_seed_results:
        for layer_summary in seed_result["layers"]:
            for a in layer_summary["validated_anchors"]:
                key = (int(a["layer"]), int(a["column"]))
                mean_drop = float(a["mean_drop"])
                median_drop = float(a.get("median_drop", mean_drop))
                score = 0.5 * mean_drop + 0.5 * median_drop
                current = anchor_to_stats.get(key, {})
                current_score = 0.5 * current.get("drop", 0.0) + 0.5 * current.get("median_drop", current.get("drop", 0.0))
                if score > current_score:
                    anchor_to_stats[key] = {
                        "drop": mean_drop,
                        "median_drop": median_drop,
                        "z": float(a["z_score"]),
                    }

    def _anchor_score(item: Tuple[Tuple[int, int], Dict[str, float]]) -> float:
        d = item[1]
        return 0.5 * d["drop"] + 0.5 * d.get("median_drop", d["drop"])

    sorted_anchor_stats = sorted(anchor_to_stats.items(), key=_anchor_score, reverse=True)
    print("Top 10 anchor drops:", [x[1]["drop"] for x in sorted_anchor_stats[:10]])
    sorted_anchor_stats = sorted_anchor_stats[:64]
    anchors: List[Tuple[int, int]] = [k for k, _ in sorted_anchor_stats]

    # Anchor frequency across seeds.
    anchor_frequency: Dict[Tuple[int, int], int] = {}
    for seed_result in all_seed_results:
        for layer_summary in seed_result["layers"]:
            for a in layer_summary["validated_anchors"]:
                key = (int(a["layer"]), int(a["column"]))
                anchor_frequency[key] = int(anchor_frequency.get(key, 0) + 1)

    print("\n=== Anchor Seed Stability ===")
    log_progress("=== Anchor Seed Stability ===")
    num_seeds = len(all_seed_results)
    for (layer, col), freq in sorted(anchor_frequency.items(), key=lambda x: -x[1]):
        print(f"(layer={layer} col={col}) frequency={freq}/{num_seeds}")
        log_progress(f"(layer={layer} col={col}) frequency={freq}/{num_seeds}")

    anchors_final_data = [
        {
            "layer": int(layer),
            "column": int(col),
            "drop": float(stats["drop"]),
            "median_drop": float(stats.get("median_drop", stats["drop"])),
            "z": float(stats["z"]),
        }
        for (layer, col), stats in sorted_anchor_stats
    ]
    with open("anchors_final.json", "w", encoding="utf-8") as f:
        json.dump(anchors_final_data, f, indent=2)
    with open("anchors_progress.txt", "a", encoding="utf-8") as f:
        f.write("--- Final anchor list ---\n")
        for (layer, col), stats in sorted_anchor_stats:
            f.write(f"(layer={layer} col={col}) drop={stats['drop']:.2f} Z={stats['z']:.1f}\n")

    log_progress("Discovered anchors (top 64):")
    for (layer, col), stats in sorted_anchor_stats:
        _ORIG_PRINT(
            f"(layer={int(layer)}, col={int(col)}, drop={float(stats['drop']):.6f}, z={float(stats['z']):.6f})"
        )
        log_progress(f"  (layer={layer}, col={col}) drop={stats['drop']:.2f} Z={stats['z']:.1f}")

    if False:
        # Activation patching (use fresh calibration to avoid data leak; independence from EMS calibration).
        print("\n=== Running Anchor Activation Patching Experiment ===")
        calibration_samples = load_mcq_dataset(n_total=120)
        run_anchor_activation_patching_experiment_phi3(
            model=model,
            tokenizer=tokenizer,
            anchors=anchors,
            calibration_samples=calibration_samples,
            letter_token_ids=letter_token_ids,
            n_samples=120,
            per_anchor_report_only=False,
        )
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

        # Per-anchor causal strength.
        print("\n=== Per-Anchor Causal Strength ===")
        for anchor in anchors:
            run_anchor_activation_patching_experiment_phi3(
                model=model,
                tokenizer=tokenizer,
                anchors=[anchor],
                calibration_samples=calibration_samples,
                letter_token_ids=letter_token_ids,
                n_samples=120,
                per_anchor_report_only=True,
            )

        # Head-to-anchor attribution.
        run_head_to_anchor_attribution_phi3(
            model=model,
            tokenizer=tokenizer,
            anchors=anchors,
            calibration_samples=calibration_samples,
            letter_token_ids=letter_token_ids,
            n_prompts=96,
            layer_range=(6, 21),
            num_heads=num_heads,
            head_dim=head_dim,
            output_path="head_anchor_attribution.json",
            batch_size=8,
        )
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

        # Multi-anchor ablation sweep.
        run_multi_anchor_ablation_sweep_phi3(
            model=model,
            tokenizer=tokenizer,
            anchors=anchors,
            seeds=SEEDS,
            calibration_size=cfg.calibration_size,
            eval_size=60,
            letter_token_ids=letter_token_ids,
        )
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

        # Random neuron ablation baseline.
        anchor_layers = sorted(list(set(int(layer) for layer, _ in anchors))) or [
            int(LAYERS_TO_TEST[0])
        ]
        run_random_neuron_ablation_baseline_phi3(
            model=model,
            tokenizer=tokenizer,
            layers=anchor_layers,
            seeds=SEEDS,
            calibration_size=cfg.calibration_size,
            eval_size=60,
            k_values=[1, 4, 8, 16, 32, 48, 64],
            letter_token_ids=letter_token_ids,
        )
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

        # Attention head ablation.
        run_attention_head_ablation_sweep_phi3(
            model=model,
            tokenizer=tokenizer,
            layers_to_ablate=[8, 16],
            seeds=SEEDS,
            calibration_size=cfg.calibration_size,
            eval_size=60,
            k_values=[1, 2, 4, 8],
            num_heads=num_heads,
            head_dim=head_dim,
            letter_token_ids=letter_token_ids,
        )
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()


if __name__ == "__main__":
    main()

