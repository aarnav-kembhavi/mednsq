"""
Neuron Ablation Sweep on Meditron-2.5-8B
=========================================
Progressively ablates anchor neurons vs random neurons on MedQA.
Compares accuracy and logit margin drops.
"""

import os
import gc
import random
import numpy as np
import torch
import torch.nn.functional as F
from collections import defaultdict
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

# =====================================================================
# CONFIG
# =====================================================================
MODEL_NAME = "epfl-llm/meditron-2.5-8b"
N_SAMPLES = 120
BATCH_SIZE = 8
SEED = 42
K_VALUES = [1, 2, 4, 8, 16, 32, 64]
N_RANDOM_TRIALS = 20
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float16 if DEVICE == "cuda" else torch.float32

# Anchor list: (layer, column), sorted by descending mean drop
ANCHORS = [
    (22, 13440), (21, 18647), (22, 2813),  (20, 14718), (11, 7786),
    (22, 18477), (22, 13345), (20, 16538), (22, 15807), (17, 13477),
    (22, 3921),  (13, 5628),  (20, 15056), (22, 18656), (18, 3615),
    (21, 16175), (21, 3117),  (20, 15034), (20, 9838),  (22, 16052),
    (17, 10493), (21, 12337), (21, 3857),  (22, 16333), (20, 7417),
    (21, 1990),  (20, 5919),  (10, 17567), (21, 18543), (21, 7165),
    (18, 4427),  (19, 3161),  (14, 11647), (10, 17885), (19, 6451),
    (21, 15751), (18, 716),   (20, 2164),  (21, 18059), (18, 14644),
    (21, 12791), (14, 12236), (12, 11755), (22, 11465), (14, 12171),
    (20, 5874),  (13, 12995), (17, 18595), (20, 11094), (20, 12080),
    (19, 8987),  (16, 7137),  (13, 3958),  (22, 12259), (22, 8152),
    (13, 1297),  (13, 13238), (22, 13310), (22, 2148),  (22, 4647),
    (22, 9736),  (13, 5445),  (11, 8887),  (13, 12737),
]

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# =====================================================================
# MODEL LOADING
# =====================================================================
print(f"Loading {MODEL_NAME} on {DEVICE} ...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "left"

load_kwargs = {
    "torch_dtype": DTYPE,
    "low_cpu_mem_usage": True,
}
if DEVICE == "cuda":
    # Let HF place modules on available GPU(s).
    load_kwargs["device_map"] = "auto"

model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, **load_kwargs)
if DEVICE == "cpu":
    model.to("cpu")
model.eval()


# =====================================================================
# LOCATE MLP MODULES (no hardcoding)
# =====================================================================
def discover_mlp_modules(model):
    """
    Walk the model and find per-layer MLP blocks.
    For LLaMA-style: layer.mlp has gate_proj, up_proj, down_proj.
    We hook the INPUT to down_proj — that is the post-activation,
    pre-projection neuron vector of size = intermediate_size.
    """
    layer_modules = {}
    # Try common roots
    roots = []
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        roots.append(("model.model.layers", model.model.layers))
    elif hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        roots.append(("model.transformer.h", model.transformer.h))
    else:
        # Fall back: scan for ModuleList of decoder-like blocks
        for name, mod in model.named_modules():
            if isinstance(mod, torch.nn.ModuleList) and len(mod) > 8:
                roots.append((name, mod))
                break

    if not roots:
        raise RuntimeError("Could not locate decoder layers in model.")

    root_name, layers = roots[0]
    print(f"Found decoder stack at: {root_name} (n_layers = {len(layers)})")

    for i, layer in enumerate(layers):
        if hasattr(layer, "mlp") and hasattr(layer.mlp, "down_proj"):
            layer_modules[i] = layer.mlp.down_proj
        elif hasattr(layer, "mlp") and hasattr(layer.mlp, "c_proj"):
            layer_modules[i] = layer.mlp.c_proj
        else:
            raise RuntimeError(f"Layer {i} MLP has no down_proj/c_proj")
    return layer_modules


layer_to_module = discover_mlp_modules(model)
N_LAYERS = len(layer_to_module)

# Confirm one example
sample_idx = list(layer_to_module.keys())[0]
sample_mod = layer_to_module[sample_idx]
INTERMEDIATE_SIZE = sample_mod.in_features
HIDDEN_SIZE = sample_mod.out_features
print(f"MLP down_proj: in={INTERMEDIATE_SIZE} (neurons), out={HIDDEN_SIZE} (hidden)")
print(f"Hooking PRE-FORWARD on down_proj => zeroing post-activation neuron columns.\n")


# =====================================================================
# CRUSHING MECHANISM (forward pre-hooks on down_proj input)
# =====================================================================
class NeuronAblator:
    """
    Manages forward_pre_hooks on MLP down_proj modules.
    Input to down_proj is [batch, seq, intermediate_size] — we zero
    the specified columns (neurons) in-place per layer.
    """
    def __init__(self, layer_to_module):
        self.layer_to_module = layer_to_module
        self.handles = []

    def _make_hook(self, neuron_idx_tensor):
        def pre_hook(module, args):
            x = args[0]
            # x: [batch, seq, intermediate_size]
            x = x.clone()
            x[..., neuron_idx_tensor] = 0.0
            return (x,) + args[1:]
        return pre_hook

    def attach(self, neurons):
        """neurons: list of (layer, column)"""
        by_layer = defaultdict(list)
        for (layer, col) in neurons:
            by_layer[layer].append(col)
        for layer, cols in by_layer.items():
            mod = self.layer_to_module[layer]
            idx = torch.tensor(cols, dtype=torch.long, device=DEVICE)
            h = mod.register_forward_pre_hook(self._make_hook(idx))
            self.handles.append(h)

    def remove(self):
        for h in self.handles:
            h.remove()
        self.handles = []

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.remove()


# =====================================================================
# DATA: MedQA
# =====================================================================
def load_medqa(n_samples=N_SAMPLES, seed=SEED):
    """
    Load MedQA (USMLE, 4-option) and take a fixed-seed sample of n_samples.
    """
    print("Loading MedQA ...")
    candidates = [
        ("bigbio/med_qa", "med_qa_en_4options_source"),
        ("bigbio/med_qa", "med_qa_en_4options_bigbio_qa"),
        ("GBaker/MedQA-USMLE-4-options", None),
    ]
    ds = None
    for name, config in candidates:
        try:
            ds = load_dataset(name, config) if config else load_dataset(name)
            print(f"  Loaded: {name} ({config})")
            break
        except Exception as e:
            print(f"  Failed {name}/{config}: {e}")
    if ds is None:
        raise RuntimeError("Could not load any MedQA variant.")

    split = "test" if "test" in ds else ("validation" if "validation" in ds else "train")
    raw = ds[split]

    rng = random.Random(seed)
    indices = list(range(len(raw)))
    rng.shuffle(indices)
    indices = indices[:n_samples]

    samples = []
    for i in indices:
        ex = raw[i]
        # Normalize different schemas
        if "question" in ex and "options" in ex:
            q = ex["question"]
            opts = ex["options"]
            if isinstance(opts, dict):
                # e.g. {"A": "...", "B": "..."}
                keys = sorted(opts.keys())
                choices = [opts[k] for k in keys]
                letters = keys
            elif isinstance(opts, list) and len(opts) > 0 and isinstance(opts[0], dict):
                # e.g. [{"key":"A","value":"..."}]
                choices = [o.get("value", o.get("text", "")) for o in opts]
                letters = [o.get("key", chr(65 + j)) for j, o in enumerate(opts)]
            else:
                choices = list(opts)
                letters = [chr(65 + j) for j in range(len(choices))]
            if ex.get("answer_idx") is not None:
                ans = ex.get("answer_idx")
            elif ex.get("answer") is not None:
                ans = ex.get("answer")
            else:
                ans = ex.get("correct_answer")
            if isinstance(ans, str) and ans in letters:
                correct = letters.index(ans)
            elif isinstance(ans, int):
                correct = ans
            elif isinstance(ans, str) and ans in choices:
                correct = choices.index(ans)
            else:
                # Try "answer" as the text content
                txt = ex.get("answer", "")
                correct = choices.index(txt) if txt in choices else 0
        elif "question" in ex and "choices" in ex:
            q = ex["question"]
            choices = ex["choices"]
            letters = [chr(65 + j) for j in range(len(choices))]
            correct = ex.get("answer", 0)
            if isinstance(correct, str):
                correct = letters.index(correct) if correct in letters else 0
        else:
            continue

        if len(choices) < 2:
            continue
        if not (0 <= int(correct) < len(choices)):
            continue
        # Keep the script's 4-choice behavior, but only keep examples where
        # the gold answer remains inside those first four choices.
        if correct >= 4:
            continue
        samples.append({
            "question": q,
            "choices": choices[:4],
            "letters": letters[:4],
            "correct": correct,
        })
    print(f"  Selected {len(samples)} samples")
    return samples


def format_prompt(sample):
    letters = sample["letters"]
    body = sample["question"].strip() + "\n\n"
    for L, c in zip(letters, sample["choices"]):
        body += f"{L}. {c}\n"
    body += "\nAnswer:"
    return body


# =====================================================================
# EVALUATION
# =====================================================================
def get_choice_token_ids(letters):
    """
    For each choice letter, get the token id corresponding to the
    space-prefixed letter (LLaMA tokenizers usually need ' A').
    """
    ids = []
    for L in letters:
        # Try " A" first, fall back to "A"
        toks = tokenizer.encode(" " + L, add_special_tokens=False)
        if len(toks) == 0:
            toks = tokenizer.encode(L, add_special_tokens=False)
        ids.append(toks[-1])
    return ids


@torch.no_grad()
def evaluate(samples, batch_size=BATCH_SIZE):
    """
    Returns (accuracy, mean_margin) over samples.
    Margin = logit(correct_letter_token) - max(logit(other_letter_tokens))
    """
    n_correct = 0
    margins = []

    for start in range(0, len(samples), batch_size):
        batch = samples[start:start + batch_size]
        prompts = [format_prompt(s) for s in batch]

        enc = tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=2048,
        ).to(DEVICE)

        out = model(**enc)
        logits = out.logits  # [B, T, V]
        # Last non-pad token: with left-padding, that's position -1
        last_logits = logits[:, -1, :]  # [B, V]

        for i, s in enumerate(batch):
            choice_ids = get_choice_token_ids(s["letters"])
            choice_logits = last_logits[i, choice_ids].float()  # [n_choices]
            pred = int(torch.argmax(choice_logits).item())
            correct_idx = s["correct"]

            if pred == correct_idx:
                n_correct += 1

            correct_logit = choice_logits[correct_idx].item()
            mask = torch.ones_like(choice_logits, dtype=torch.bool)
            mask[correct_idx] = False
            other_max = choice_logits[mask].max().item()
            margins.append(correct_logit - other_max)

    acc = n_correct / len(samples)
    mean_margin = float(np.mean(margins))
    return acc, mean_margin


# =====================================================================
# RANDOM NEURON SAMPLING (matched layer distribution)
# =====================================================================
def sample_random_neurons(k, anchors_subset, rng):
    """
    Sample k random (layer, column) neurons matching layer distribution
    of anchors_subset, avoiding collisions with the anchor set.
    """
    anchor_set = set(anchors_subset)
    layer_counts = defaultdict(int)
    for (l, _) in anchors_subset:
        layer_counts[l] += 1

    chosen = []
    for layer, count in layer_counts.items():
        attempts = 0
        added = 0
        while added < count and attempts < count * 100:
            col = rng.randrange(INTERMEDIATE_SIZE)
            if (layer, col) not in anchor_set and (layer, col) not in chosen:
                chosen.append((layer, col))
                added += 1
            attempts += 1
    return chosen


# =====================================================================
# MAIN SWEEP
# =====================================================================
def main():
    samples = load_medqa(N_SAMPLES, SEED)

    # Baseline
    print("\n=== Baseline (no ablation) ===")
    base_acc, base_margin = evaluate(samples)
    print(f"Baseline accuracy: {base_acc:.4f}")
    print(f"Baseline mean margin: {base_margin:.4f}")

    rng = random.Random(SEED)
    results = {}

    for K in K_VALUES:
        if K > len(ANCHORS):
            print(f"\nSkipping K={K} (only {len(ANCHORS)} anchors available)")
            continue

        print(f"\n=== K = {K} ===")
        anchor_subset = ANCHORS[:K]

        # ---- Anchor run ----
        with NeuronAblator(layer_to_module) as ab:
            ab.attach(anchor_subset)
            anchor_acc, anchor_margin = evaluate(samples)

        # ---- Random trials ----
        rand_accs = []
        rand_margins = []
        for trial in range(N_RANDOM_TRIALS):
            rand_neurons = sample_random_neurons(K, anchor_subset, rng)
            with NeuronAblator(layer_to_module) as ab:
                ab.attach(rand_neurons)
                r_acc, r_margin = evaluate(samples)
            rand_accs.append(r_acc)
            rand_margins.append(r_margin)

        rand_acc_mean = float(np.mean(rand_accs))
        rand_acc_std = float(np.std(rand_accs))
        rand_margin_mean = float(np.mean(rand_margins))
        rand_margin_std = float(np.std(rand_margins))

        anchor_acc_drop = base_acc - anchor_acc
        rand_acc_drop = base_acc - rand_acc_mean
        anchor_margin_drop = base_margin - anchor_margin
        rand_margin_drop = base_margin - rand_margin_mean

        print(f"Anchor accuracy: {anchor_acc:.4f}")
        print(f"Random accuracy: {rand_acc_mean:.4f} ± {rand_acc_std:.4f}")
        print(f"Accuracy drop (anchor): {anchor_acc_drop:+.4f}")
        print(f"Accuracy drop (random): {rand_acc_drop:+.4f}")
        print(f"Anchor mean margin drop: {anchor_margin_drop:+.4f}")
        print(f"Random mean margin drop: {rand_margin_drop:+.4f} "
              f"(margin std={rand_margin_std:.4f})")
        print(f"Diff (anchor - random) accuracy drop: "
              f"{anchor_acc_drop - rand_acc_drop:+.4f}")
        print(f"Diff (anchor - random) margin drop:   "
              f"{anchor_margin_drop - rand_margin_drop:+.4f}")

        results[K] = {
            "anchor_acc": anchor_acc,
            "anchor_margin": anchor_margin,
            "rand_acc_mean": rand_acc_mean,
            "rand_acc_std": rand_acc_std,
            "rand_margin_mean": rand_margin_mean,
            "rand_margin_std": rand_margin_std,
        }

        torch.cuda.empty_cache()
        gc.collect()

    # ---- Summary ----
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Baseline acc={base_acc:.4f}  margin={base_margin:.4f}")
    print(f"{'K':>4}  {'AnchAcc':>8}  {'RandAcc':>8}  {'ΔAccA':>7}  "
          f"{'ΔAccR':>7}  {'ΔMarA':>7}  {'ΔMarR':>7}")
    for K, r in results.items():
        print(f"{K:>4}  "
              f"{r['anchor_acc']:>8.4f}  "
              f"{r['rand_acc_mean']:>8.4f}  "
              f"{base_acc - r['anchor_acc']:>+7.4f}  "
              f"{base_acc - r['rand_acc_mean']:>+7.4f}  "
              f"{base_margin - r['anchor_margin']:>+7.4f}  "
              f"{base_margin - r['rand_margin_mean']:>+7.4f}")


if __name__ == "__main__":
    main()