"""
Compute Pearson correlations between activations of EMS-discovered neurons
at the final token position on MedNSQ adversarial pairs.
"""

import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from mednsq_data import load_mcq_dataset, build_adversarial_pairs

# -----------------------------------------------------------------------------
# Neurons to probe: (layer, column)
# -----------------------------------------------------------------------------
NEURONS = [
    (16, 1761),
    (16, 2056),
    (16, 3433),
    (8, 3977),
    (8, 3347),
]
N_SAMPLES = 200


def main():
    model_name = "google/medgemma-1.5-4b-it"

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    model.eval()

    samples = load_mcq_dataset(n_total=N_SAMPLES)
    adv_pairs = build_adversarial_pairs(
        model, tokenizer, samples, n_calib=len(samples)
    )

    # activations[i] = vector of activations for neuron i across samples
    activations = [[] for _ in NEURONS]

    with torch.no_grad():
        for pair in adv_pairs:
            input_ids = pair["input_ids"].to(model.device)
            attention_mask = pair["attention_mask"].to(model.device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
            )

            hidden_states = outputs.hidden_states
            for i, (layer, col) in enumerate(NEURONS):
                act = hidden_states[layer + 1][0, -1, col].float().item()
                activations[i].append(act)

    # (n_neurons, n_samples)
    activations_matrix = np.array(activations, dtype=np.float64)
    # (n_neurons, n_neurons) Pearson correlation between neurons
    corr_matrix = np.corrcoef(activations_matrix)

    # Print header and matrix
    labels = [f"({layer},{col})" for layer, col in NEURONS]
    header = " " * 12 + " ".join(f"{lab:>10}" for lab in labels)
    print("Neuron correlations")
    print(header)
    for i, label in enumerate(labels):
        row_str = " ".join(f"{corr_matrix[i, j]:10.4f}" for j in range(len(labels)))
        print(f"{label:>12} {row_str}")


if __name__ == "__main__":
    main()
