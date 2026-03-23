"""
Fixed multi-anchor ablation for Phi-3 (down_proj column crush only).

Reuses crushing and sweep logic from run_mednsq_phi3.py without EMS, probing, or Jacobian.
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


sys.stdout = Tee("crushing_phi3.txt")
sys.stderr = sys.stdout


# ----------------------------
# SECTION 1: Config
# ----------------------------

MODEL_NAME = "microsoft/Phi-3-mini-128k-instruct"

# Batching constant (preserve default)
BATCH_SIZE = 32

# Tokenizer padding id is set once after tokenizer load.
PAD_TOKEN_ID: int = 0


ANCHORS: List[Tuple[int, int]] = [
    (9, 1384),
    (8, 7649),
    (21, 5671),
    (8, 4360),
    (11, 411),
    (8, 3352),
    (22, 1970),
    (16, 1605),
    (21, 7260),
    (19, 6536),
    (20, 4208),
    (19, 7628),
    (12, 4157),
    (21, 6041),
    (15, 147),
    (22, 5),
    (19, 7279),
    (21, 6367),
    (17, 7006),
    (17, 7050),
    (21, 3557),
    (21, 1294),
    (17, 514),
    (8, 221),
    (10, 7665),
    (19, 109),
    (20, 2564),
    (15, 2385),
    (21, 2452),
    (19, 4048),
    (8, 3562),
    (11, 6445),
    (19, 3938),
    (22, 3223),
    (18, 3198),
    (19, 7519),
    (18, 2960),
    (20, 3500),
    (19, 7207),
    (16, 5736),
    (19, 4873),
    (19, 1228),
    (21, 5354),
    (15, 4696),
    (14, 5222),
    (20, 4196),
    (22, 3734),
    (19, 3896),
    (19, 6221),
    (22, 5805),
    (22, 2028),
    (19, 8156),
    (17, 1957),
    (22, 4562),
    (14, 6621),
    (13, 7321),
    (20, 907),
    (19, 6665),
    (19, 5406),
    (21, 2321),
    (22, 2692),
    (18, 2322),
    (11, 4321),
    (22, 4813),
]


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


def _set_reproducible(seed: int) -> None:
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)


def _get_phi3_layers(model) -> List[Any]:
    if not hasattr(model, "model") or not hasattr(model.model, "layers"):
        raise RuntimeError(
            "Phi-3 architecture check failed: expected model.model.layers"
        )
    return list(model.model.layers)


def _phi3_down_proj_weight(model, layer_idx: int) -> torch.Tensor:
    layers = _get_phi3_layers(model)
    if layer_idx < 0 or layer_idx >= len(layers):
        raise IndexError(f"layer_idx out of range: {layer_idx}")
    layer = layers[layer_idx]
    if not hasattr(layer, "mlp") or not hasattr(layer.mlp, "down_proj"):
        raise RuntimeError(
            f"Phi-3 architecture check failed at layer {layer_idx}: "
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

        # STRICT DEBUG CHECK (run once): token list with indices, A/B/C/D positions, chosen answer_idx.
        if start == 0 and tokenizer is not None:
            print("=== DEBUG TOKEN CHECK ===")
            i0 = 0
            L0 = int(seq_lens[i0])
            ids0 = input_ids[i0, :L0].cpu().tolist()
            token_list = [(j, ids0[j], tokenizer.decode([ids0[j]]) if ids0[j] != PAD_TOKEN_ID else "<pad>") for j in range(L0)]
            print("Token list (idx, id, decode):", token_list)
            letter_set = set(int(x) for x in letter_token_ids.cpu().tolist())
            letter_positions = [j for j in range(L0) if ids0[j] in letter_set]
            print("Positions of A/B/C/D tokens:", letter_positions)
            print("Chosen answer_idx:", answer_indices[i0])
            print("Answer idx distribution sample:", all_answer_indices[:10])
            print(tokenizer.decode(input_ids[0, :L0]))

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

    return torch.tensor(margins, dtype=torch.float32), preds


def simulate_column_crush_phi3(model, layer_idx: int, col_idx: int) -> torch.Tensor:
    weight = _phi3_down_proj_weight(model, layer_idx)
    if col_idx < 0 or col_idx >= weight.shape[1]:
        raise IndexError(f"col_idx out of range: {col_idx} for weight.shape={tuple(weight.shape)}")
    original_col = weight[:, col_idx].detach().clone()
    scale = original_col.abs().mean()
    crushed = original_col.sign() * scale
    with torch.no_grad():
        weight[:, col_idx] = crushed.to(dtype=weight.dtype, device=weight.device)
    return original_col


def restore_column_phi3(model, layer_idx: int, col_idx: int, original_col: torch.Tensor) -> float:
    weight = _phi3_down_proj_weight(model, layer_idx)
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


def run_anchor_vs_random_ablation(
    model,
    tokenizer,
    anchors,
    seeds,
    calibration_size,
    eval_size,
    letter_token_ids,
):
    print("\n=== ANCHOR vs RANDOM ABLATION ===")

    ablation_counts = [1, 4, 8, 16, 32, 48, 64]

    anchor_results = {k: {"acc": [], "margin": []} for k in ablation_counts}
    random_results = {k: {"acc": [], "margin": []} for k in ablation_counts}

    from mednsq_data import load_mcq_dataset, build_adversarial_pairs
    from mednsq_eval import evaluate_model

    anchor_layers = list(set(l for l, _ in anchors))

    for seed in seeds:
        print(f"\n===== SEED {seed} =====")
        _set_reproducible(seed)

        samples = load_mcq_dataset(n_total=calibration_size + eval_size)
        random.shuffle(samples)

        calibration = samples[:calibration_size]
        evaluation = samples[calibration_size : calibration_size + eval_size]

        adv_pairs = build_adversarial_pairs(model, tokenizer, calibration, n_calib=len(calibration))
        _assert_adv_pairs_shape(adv_pairs)

        baseline_eval = evaluate_model(model, tokenizer, evaluation)
        print(f"[Seed {seed}] BASELINE acc={baseline_eval.get('accuracy', 0.0):.6f}")

        for k in ablation_counts:
            if k > len(anchors):
                continue

            subset = anchors[:k]

            saved = []
            try:
                for l, c in subset:
                    orig = simulate_column_crush_phi3(model, l, c)
                    saved.append((l, c, orig))

                eval_metrics = evaluate_model(model, tokenizer, evaluation)
                acc = float(eval_metrics.get("accuracy", 0.0))

                margins = compute_per_sample_margins_phi3(
                    model, adv_pairs, letter_token_ids, tokenizer=tokenizer
                )
                mean_margin = float(margins.mean().item()) if margins.numel() else 0.0

                anchor_results[k]["acc"].append(acc)
                anchor_results[k]["margin"].append(mean_margin)

                print(f"[ANCHOR][Seed {seed}] k={k} acc={acc:.6f} margin={mean_margin:.6f}")

            finally:
                for l, c, orig in saved:
                    restore_column_phi3(model, l, c, orig)

            random_neurons = set()
            while len(random_neurons) < k:
                l = random.choice(anchor_layers)
                max_col = _phi3_down_proj_weight(model, l).shape[1] - 1
                c = random.randint(0, max_col)
                random_neurons.add((l, c))

            subset_random = list(random_neurons)

            saved = []
            try:
                for l, c in subset_random:
                    orig = simulate_column_crush_phi3(model, l, c)
                    saved.append((l, c, orig))

                eval_metrics = evaluate_model(model, tokenizer, evaluation)
                acc = float(eval_metrics.get("accuracy", 0.0))

                margins = compute_per_sample_margins_phi3(
                    model, adv_pairs, letter_token_ids, tokenizer=tokenizer
                )
                mean_margin = float(margins.mean().item()) if margins.numel() else 0.0

                random_results[k]["acc"].append(acc)
                random_results[k]["margin"].append(mean_margin)

                print(f"[RANDOM][Seed {seed}] k={k} acc={acc:.6f} margin={mean_margin:.6f}")

            finally:
                for l, c, orig in saved:
                    restore_column_phi3(model, l, c, orig)

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

    print("\n=== FINAL COMPARISON ===")

    for k in ablation_counts:
        if not anchor_results[k]["acc"]:
            continue

        a_acc = sum(anchor_results[k]["acc"]) / len(anchor_results[k]["acc"])
        r_acc = sum(random_results[k]["acc"]) / len(random_results[k]["acc"])

        print(f"\nk={k}")
        print(f"Anchor acc: {a_acc}")
        print(f"Random acc: {r_acc}")


def main() -> None:
    _ensure_import_paths()

    from mednsq_eval import _get_letter_token_ids

    print("=== RUNNING FIXED ANCHOR ABLATION ===")
    print("Total anchors:", len(ANCHORS))

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    global PAD_TOKEN_ID
    PAD_TOKEN_ID = int(getattr(tokenizer, "pad_token_id", None) or getattr(tokenizer, "eos_token_id", 0) or 0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch_dtype = torch.bfloat16 if device.type == "cuda" else torch.float32

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch_dtype,
        trust_remote_code=True,
    )
    model.to(device)
    model.eval()

    letter_token_ids = _get_letter_token_ids(tokenizer)
    _assert_letter_token_ids(letter_token_ids)
    letter_token_ids = letter_token_ids.to(device)

    run_anchor_vs_random_ablation(
        model=model,
        tokenizer=tokenizer,
        anchors=ANCHORS,
        seeds=[1, 2, 3],
        calibration_size=400,
        eval_size=60,
        letter_token_ids=letter_token_ids,
    )


if __name__ == "__main__":
    main()
