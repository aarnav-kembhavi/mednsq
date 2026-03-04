import math
from typing import List, Dict, Any

import torch
from datasets import load_dataset


def load_mcq_dataset(n_total: int = 100) -> List[Dict[str, Any]]:
    ds = load_dataset("openlifescienceai/medqa")
    train_split = ds["train"]
    n_total = min(n_total, len(train_split))

    samples: List[Dict[str, Any]] = []

    for i in range(n_total):
        row = train_split[i]
        q = row["data"]["Question"]
        opts = row["data"]["Options"]

        options = [opts["A"], opts["B"], opts["C"], opts["D"]]
        correct_letter = row["data"]["Correct Option"]
        idx = "ABCD".index(correct_letter)

        samples.append(
            {
                "question": q,
                "options": options,
                "correct_index": idx,
            }
        )

    return samples


def format_prompt(question: str, options: List[str]) -> str:
    letters = ["A", "B", "C", "D"]
    option_lines = []
    for letter, text in zip(letters, options):
        option_lines.append(f"{letter}. {text}")

    options_block = "\n".join(option_lines)

    prompt = (
        f"Question: {question}\n"
        f"Options:\n{options_block}\n"
        f"Answer:"
    )
    return prompt


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


def build_adversarial_pairs(
    model,
    tokenizer,
    dataset,
    n_calib: int = 80,
):
    model.eval()

    device = next(model.parameters()).device
    letter_token_ids = _get_letter_token_ids(tokenizer).to(device)

    n_calib = min(n_calib, len(dataset))
    adv_pairs = []

    with torch.no_grad():
        for i in range(n_calib):
            sample = dataset[i]
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

            correct_index = int(sample["correct_index"])
            pos_id = int(letter_token_ids[correct_index].item())

            mask = torch.ones_like(p_letters, dtype=torch.bool)
            mask[0, correct_index] = False
            masked_probs = p_letters.masked_fill(~mask, -float("inf"))
            neg_local_idx = int(torch.argmax(masked_probs, dim=-1).item())
            neg_id = int(letter_token_ids[neg_local_idx].item())

            adv_pairs.append(
                {
                    "correct_letter_index": correct_index,
                    "neg_letter_index": neg_local_idx,
                    "input_ids": input_ids,
                    "attention_mask": attention_mask,
                    "pos_id": pos_id,
                    "neg_id": neg_id,
                }
            )

    return adv_pairs