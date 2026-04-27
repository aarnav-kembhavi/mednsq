"""
MedQA data loading + adversarial pair construction.
Used by both discovery and ablation. Single source of truth for prompt format.
"""

from typing import Any, Dict, List

import torch
from datasets import load_dataset


def load_mcq_dataset(n_total: int = 100, split: str = "train") -> List[Dict[str, Any]]:
    """
    Load N samples from MedQA (openlifescienceai/medqa).
    `split` can be "train", "test", or "validation".
    """
    ds = load_dataset("openlifescienceai/medqa")
    if split not in ds:
        # Fallback for HF schema variants
        split = "train"
    raw = ds[split]
    n_total = min(n_total, len(raw))

    samples: List[Dict[str, Any]] = []
    for i in range(n_total):
        row = raw[i]
        q = row["data"]["Question"]
        opts = row["data"]["Options"]
        options = [opts["A"], opts["B"], opts["C"], opts["D"]]
        correct_letter = row["data"]["Correct Option"]
        idx = "ABCD".index(correct_letter)
        samples.append({
            "question": q,
            "options": options,
            "correct_index": idx,
        })
    return samples


def format_prompt(question: str, options: List[str]) -> str:
    """Single canonical MedQA prompt. Do not change without bumping the hash."""
    letters = ["A", "B", "C", "D"]
    option_lines = [f"{L}. {t}" for L, t in zip(letters, options)]
    return (
        f"Question: {question}\n"
        f"Options:\n" + "\n".join(option_lines) + "\n"
        f"Answer:"
    )


def _get_letter_token_ids(tokenizer) -> torch.Tensor:
    """Token id for each of A, B, C, D as standalone tokens."""
    ids = []
    for letter in ["A", "B", "C", "D"]:
        enc = tokenizer(letter, add_special_tokens=False, return_tensors=None)
        token_id = enc["input_ids"]
        if isinstance(token_id, list) and isinstance(token_id[0], list):
            token_id = token_id[0][0]
        elif isinstance(token_id, list):
            token_id = token_id[0]
        ids.append(int(token_id))
    return torch.tensor(ids, dtype=torch.long)


def build_adversarial_pairs(
    model,
    tokenizer,
    dataset: List[Dict[str, Any]],
    n_calib: int = 400,
    batch_size: int = 16,
) -> List[Dict[str, Any]]:
    """
    For each sample, identify the model's strongest distractor letter under baseline
    and build a contrastive (pos, neg) pair on the same prompt.

    Returns a list of dicts with input_ids, attention_mask, pos_id, neg_id, etc.
    Tokenization is done one prompt at a time so each pair retains its own length;
    batched forward passes are used during evaluation, not here.
    """
    model.eval()
    device = next(model.parameters()).device
    letter_token_ids = _get_letter_token_ids(tokenizer).to(device)

    n_calib = min(n_calib, len(dataset))
    adv_pairs: List[Dict[str, Any]] = []

    # Tokenize all prompts first
    prompts: List[str] = []
    for i in range(n_calib):
        s = dataset[i]
        prompts.append(format_prompt(s["question"], s["options"]))

    encs: List[Dict[str, torch.Tensor]] = []
    for p in prompts:
        e = tokenizer(p, return_tensors="pt", add_special_tokens=True)
        encs.append({
            "input_ids": e["input_ids"].to(device),
            "attention_mask": e["attention_mask"].to(device),
        })

    # Batched forward to find strongest distractor
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id

    with torch.no_grad():
        for batch_start in range(0, n_calib, batch_size):
            batch = encs[batch_start: batch_start + batch_size]
            seq_lens = [e["input_ids"].shape[1] for e in batch]
            max_len = max(seq_lens)
            # Right-pad inside the batch; track real last index per row
            ids = torch.full((len(batch), max_len), pad_id, dtype=torch.long, device=device)
            mask = torch.zeros((len(batch), max_len), dtype=torch.long, device=device)
            for j, e in enumerate(batch):
                L = seq_lens[j]
                ids[j, :L] = e["input_ids"][0]
                mask[j, :L] = e["attention_mask"][0]

            out = model(input_ids=ids, attention_mask=mask)
            logits = out.logits  # [B, T, V]

            for j, e in enumerate(batch):
                    L = seq_lens[j]
                    last_logits = logits[j, L - 1, :]
                    letter_logits = last_logits[letter_token_ids].float()
                    probs = torch.softmax(letter_logits, dim=-1)

                    sample = dataset[batch_start + j]
                    correct_index = int(sample["correct_index"])

                    # Only keep pairs where model is already correct at baseline.
                    # frac_neg > 0.3 means discovery finds "fix wrong" neurons, not
                    # "protect right" neurons — that's not what we want.
                    pred_idx = int(torch.argmax(letter_logits).item())
                    if pred_idx != correct_index:
                        continue

                    masked = probs.clone()
                    masked[correct_index] = -float("inf")
                    neg_local_idx = int(torch.argmax(masked).item())

                    pos_id = int(letter_token_ids[correct_index].item())
                    neg_id = int(letter_token_ids[neg_local_idx].item())

                    adv_pairs.append({
                        "input_ids": e["input_ids"],
                        "attention_mask": e["attention_mask"],
                        "pos_id": pos_id,
                        "neg_id": neg_id,
                        "correct_index": correct_index,
                        "neg_local_idx": neg_local_idx,
                        "seq_len": L,
                    })

    return adv_pairs