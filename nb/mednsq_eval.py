"""
4-way accuracy + margin evaluation for held-out sets.
Uses the same format_prompt as discovery.
"""

from typing import Any, Dict, List

import torch

from mednsq_data import _get_letter_token_ids, format_prompt


@torch.no_grad()
def evaluate_model(
    model,
    tokenizer,
    samples: List[Dict[str, Any]],
    batch_size: int = 16,
) -> Dict[str, float]:
    """
    4-way MedQA accuracy + mean margin (logit gap between correct and best distractor).
    Margin here is computed against the model's CURRENT best distractor at eval time
    (not the baseline-time distractor) — this is the natural eval-time margin.
    """
    if not samples:
        return {"accuracy": 0.0, "mean_margin": 0.0, "n": 0}

    model.eval()
    device = next(model.parameters()).device
    letter_token_ids = _get_letter_token_ids(tokenizer).to(device)
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id

    correct = 0
    margins: List[float] = []

    # Pre-tokenize
    encs = []
    for s in samples:
        p = format_prompt(s["question"], s["options"])
        e = tokenizer(p, return_tensors="pt", add_special_tokens=True)
        encs.append({
            "input_ids": e["input_ids"].to(device),
            "attention_mask": e["attention_mask"].to(device),
            "correct_index": int(s["correct_index"]),
        })

    for batch_start in range(0, len(encs), batch_size):
        batch = encs[batch_start: batch_start + batch_size]
        seq_lens = [e["input_ids"].shape[1] for e in batch]
        max_len = max(seq_lens)
        ids = torch.full((len(batch), max_len), pad_id, dtype=torch.long, device=device)
        mask = torch.zeros((len(batch), max_len), dtype=torch.long, device=device)
        for j, e in enumerate(batch):
            L = seq_lens[j]
            ids[j, :L] = e["input_ids"][0]
            mask[j, :L] = e["attention_mask"][0]

        out = model(input_ids=ids, attention_mask=mask)
        logits = out.logits

        for j, e in enumerate(batch):
            L = seq_lens[j]
            last_logits = logits[j, L - 1, :]
            letter_logits = last_logits[letter_token_ids].float()
            pred_idx = int(torch.argmax(letter_logits).item())
            ci = e["correct_index"]
            if pred_idx == ci:
                correct += 1
            correct_logit = letter_logits[ci].item()
            other = letter_logits.clone()
            other[ci] = -float("inf")
            best_other = other.max().item()
            margins.append(correct_logit - best_other)

    return {
        "accuracy": correct / len(samples),
        "mean_margin": sum(margins) / len(margins),
        "n": len(samples),
    }