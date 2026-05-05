import random
import numpy as np
import torch

from transformers import AutoModelForCausalLM, AutoTokenizer

from mednsq_data import load_mcq_dataset, build_adversarial_pairs
from mednsq_eval import evaluate_model
from mednsq_probe import MedNSQProbe


# =========================================================
# CONFIG
# =========================================================
MODEL_PATH = "/workspace/AlphaMed-7B-instruct-rl"
SEED = 42

K_VALUES = [1, 2, 4, 8, 16, 32]
N_RANDOM = 5
EVAL_PAIRS = 300
TEST_SIZE = 300


# =========================================================
# YOUR ANCHORS (FIRST 32 ONLY)
# =========================================================
ALL_ANCHORS = [
    (12, 11452), (20, 2163), (19, 13744), (17, 16475),
    (12, 606), (20, 13290), (14, 903), (20, 9558), (12, 4149),
    (12, 16751), (15, 7484), (22, 9786), (15, 640), (15, 3642),
    (14, 15227), (13, 8480), (23, 5298), (17, 6201), (16, 16736),
    (19, 15268), (20, 11735), (14, 10204), (12, 8109), (16, 4570),
    (12, 12574), (15, 10782), (16, 3147), (17, 14360), (15, 10943),
    (14, 298), (15, 12746), (15, 13037),
]

ANCHORS = ALL_ANCHORS[:32]


# =========================================================
# UTILS
# =========================================================
def crush_many(probe, neurons):
    saved = []
    for l, c in neurons:
        orig = probe.simulate_column_crush(l, c)
        saved.append((l, c, orig))
    return saved


def restore_many(probe, saved):
    for l, c, orig in saved:
        probe.restore_column(l, c, orig)


def sample_random_neurons(k, anchors_subset, intermediate_size, rng):
    from collections import defaultdict

    layer_counts = defaultdict(int)
    for l, _ in anchors_subset:
        layer_counts[l] += 1

    anchor_set = set(anchors_subset)
    chosen = []
    chosen_set = set()

    for layer, count in layer_counts.items():
        added = 0
        while added < count:
            c = rng.randrange(intermediate_size)
            t = (layer, c)
            if t not in anchor_set and t not in chosen_set:
                chosen.append(t)
                chosen_set.add(t)
                added += 1
    return chosen


# =========================================================
# MAIN
# =========================================================
def main():
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    pad_id = tokenizer.pad_token_id

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )

    probe = MedNSQProbe(model)

    # --------------------------
    # DATA
    # --------------------------
    samples = load_mcq_dataset(n_total=EVAL_PAIRS + 100)
    pairs = build_adversarial_pairs(model, tokenizer, samples, n_calib=EVAL_PAIRS)
    test_samples = load_mcq_dataset(n_total=TEST_SIZE, split="test")

    base_margins = probe.compute_per_sample_margins(pairs, pad_id=pad_id)
    base_margin = float(base_margins.mean().item())

    base_acc = evaluate_model(model, tokenizer, test_samples)["accuracy"]

    print(f"\nBASE margin={base_margin:.4f} acc={base_acc:.4f}\n")

    rng = random.Random(SEED + 99)

    # --------------------------
    # K SWEEP
    # --------------------------
    for K in K_VALUES:
        subset = ANCHORS[:K]

        # ---- anchor ablation ----
        saved = crush_many(probe, subset)
        try:
            m = probe.compute_per_sample_margins(pairs, pad_id=pad_id)
            margin = float(m.mean().item())

            acc = evaluate_model(model, tokenizer, test_samples)["accuracy"]
        finally:
            restore_many(probe, saved)

        margin_drop = base_margin - margin
        acc_drop = base_acc - acc

        # ---- random baseline ----
        rand_drops = []
        for _ in range(N_RANDOM):
            rand_neurons = sample_random_neurons(K, subset, probe.intermediate_size, rng)

            saved_r = crush_many(probe, rand_neurons)
            try:
                rm = probe.compute_per_sample_margins(pairs, pad_id=pad_id)
                rand_margin = float(rm.mean().item())
            finally:
                restore_many(probe, saved_r)

            rand_drops.append(base_margin - rand_margin)

        rand_mean = np.mean(rand_drops)
        rand_std = np.std(rand_drops)

        z = (margin_drop - rand_mean) / (rand_std + 1e-8)

        print(f"K={K:2d} | "
              f"ΔMargin={margin_drop:+.4f} | "
              f"Rand={rand_mean:+.4f}±{rand_std:.4f} | "
              f"z={z:+.2f} | "
              f"ΔAcc={acc_drop:+.4f}")


if __name__ == "__main__":
    main()