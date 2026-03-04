from typing import List, Dict, Any

import torch
from mednsq_data import format_prompt


def _get_letter_token_ids(tokenizer) -> torch.Tensor:
    ids = []

    for letter in ["A", "B", "C", "D"]:
        encoded = tokenizer(
            letter,
            add_special_tokens=False,
            return_tensors=None,
        )

        input_ids = encoded["input_ids"]

        if isinstance(input_ids[0], list):
            token_id = input_ids[0][0]
        else:
            token_id = input_ids[0]

        ids.append(token_id)

    return torch.tensor(ids, dtype=torch.long)


def evaluate_model(
    model,
    tokenizer,
    dataset_subset: List[Dict[str, Any]],
) -> Dict[str, float]:

    model.eval()
    device = next(model.parameters()).device
    letter_token_ids = _get_letter_token_ids(tokenizer).to(device)

    correct = 0
    margins = []

    if not dataset_subset:
        return {"accuracy": 0.0, "mean_margin": 0.0}

    with torch.no_grad():
        for sample in dataset_subset:
            prompt = format_prompt(sample["question"], sample["options"])

            enc = tokenizer(
                prompt,
                return_tensors="pt",
                add_special_tokens=True,
            )

            input_ids = enc["input_ids"].to(device)
            attention_mask = enc["attention_mask"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits[:, -1, :]

            probs = torch.softmax(logits, dim=-1)
            p_letters = probs[:, letter_token_ids]

            pred_idx = int(torch.argmax(p_letters, dim=-1).item())
            correct_index = int(sample["correct_index"])

            if pred_idx == correct_index:
                correct += 1

            pos_id = int(letter_token_ids[correct_index].item())

            mask = torch.ones_like(p_letters, dtype=torch.bool)
            mask[0, correct_index] = False
            masked_probs = p_letters.masked_fill(~mask, -float("inf"))
            neg_local_idx = int(torch.argmax(masked_probs, dim=-1).item())
            neg_id = int(letter_token_ids[neg_local_idx].item())

            margin = (logits[0, pos_id] - logits[0, neg_id]).item()
            margins.append(margin)

    accuracy = correct / len(dataset_subset)
    mean_margin = float(sum(margins) / len(margins))

    return {
        "accuracy": float(accuracy),
        "mean_margin": mean_margin,
    }