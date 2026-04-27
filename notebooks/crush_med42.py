"""
Neuron Ablation Sweep on Med42-v2 (8B)
=======================================
Progressively ablates anchor neurons vs random neurons on MedQA.
Compares accuracy and logit margin drops.

Base architecture: Llama-3-8B
  - num_hidden_layers = 32
  - hidden_size       = 4096
  - intermediate_size = 14336
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
MODEL_NAME = "m42-health/Llama3-Med42-8B"
N_SAMPLES = 120
BATCH_SIZE = 8
SEED = 42
K_VALUES = [1, 2, 4, 8, 16, 32]   # max 32 since len(ANCHORS) = 38
N_RANDOM_TRIALS = 20
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float16 if DEVICE == "cuda" else torch.float32

# Med42 anchor list: (layer, column), sorted by descending mean drop
ANCHORS = [
    (19, 7232),   (20, 4299),   (14, 8),      (12, 1493),   (17, 7522),
    (11, 11902),  (13, 2590),   (18, 7417),   (13, 4542),   (14, 318),
    (15, 1706),   (15, 2808),   (16, 9215),   (17, 13385),  (18, 10857),
    (18, 2694),   (20, 3476),   (12, 2976),   (12, 1780),   (13, 4216),
    (13, 1724),   (14, 12639),  (15, 11853),  (15, 1283),   (16, 8902),
    (16, 13791),  (17, 10284),  (19, 9770),   (19, 1498),   (20, 5270),
    (20, 10638),  (15, 6589),   (15, 2295),   (16, 10142),  (16, 6737),
    (17, 8887),   (17, 863),    (20, 2786),
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
    For Llama-3 / Qwen2.5: layer.mlp has gate_proj, up_proj, down_proj.
    We hook the INPUT to down_proj — that is the post-SiLU, post-gating
    neuron vector of size = intermediate_size.
    """
    layer_modules = {}
    roots = []
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        roots.append(("model.model.layers", model.model.layers))
    elif hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        roots.append(("model.transformer.h", model.transformer.h))
    else:
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

sample_idx = list(layer_to_module.keys())[0]
sample_mod = layer_to_module[sample_idx]
INTERMEDIATE_SIZE = sample_mod.in_features
HIDDEN_SIZE = sample_mod.out_features
print(f"MLP down_proj: in={INTERMEDIATE_SIZE} (neurons), out={HIDDEN_SIZE} (hidden)")
print(f"Hooking PRE-FORWARD on down_proj => zeroing post-activation neuron columns.\n")

# ---- Sanity check anchors against the actual architecture ----
max_layer = max(l for l, _ in ANCHORS)
max_col = max(c for _, c in ANCHORS)
assert max_layer < N_LAYERS, (
    f"Anchor layer {max_layer} >= N_LAYERS {N_LAYERS}"
)
assert max_col < INTERMEDIATE_SIZE, (
    f"Anchor column {max_col} >= INTERMEDIATE_SIZE {INTERMEDIATE_SIZE}. "
    f"Anchors do not match this model's architecture."
)
print(f"Anchor sanity: max_layer={max_layer}/{N_LAYERS-1}, "
      f"max_col={max_col}/{INTERMEDIATE_SIZE-1}  OK")
print(f"Anchor count: {len(ANCHORS)}\n")


# =====================================================================
# CRUSHING MECHANISM (forward pre-hooks on down_proj input)
# =====================================================================
class NeuronAblator:
    """
    Manages forward_pre_hooks on MLP down_proj modules.
    Input to down_proj is [batch, seq, intermediate_size] — we zero
    the specified columns (neurons) per layer.
    """
    def __init__(self, layer_to_module):
        self.layer_to_module = layer_to_module
        self.handles = []

    def _make_hook(self, neuron_idx_tensor):
        def pre_hook(module, args):
            x = args[0]
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
        if "question" in ex and "options" in ex:
            q = ex["question"]
            opts = ex["options"]
            if isinstance(opts, dict):
                keys = sorted(opts.keys())
                choices = [opts[k] for k in keys]
                letters = keys
            elif isinstance(opts, list) and len(opts) > 0 and isinstance(opts[0], dict):
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
_choice_token_cache = {}

def get_choice_token_ids(letters):
    key = tuple(letters)
    if key in _choice_token_cache:
        return _choice_token_cache[key]
    ids = []
    for L in letters:
        toks = tokenizer.encode(" " + L, add_special_tokens=False)
        if len(toks) == 0:
            toks = tokenizer.encode(L, add_special_tokens=False)
        ids.append(toks[-1])
    _choice_token_cache[key] = ids
    return ids


@torch.no_grad()
def evaluate(samples, batch_size=BATCH_SIZE):
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
        last_logits = logits[:, -1, :]

        for i, s in enumerate(batch):
            choice_ids = get_choice_token_ids(s["letters"])
            choice_logits = last_logits[i, choice_ids].float()
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
    anchor_set = set(anchors_subset)
    layer_counts = defaultdict(int)
    for (l, _) in anchors_subset:
        layer_counts[l] += 1

    chosen = []
    chosen_set = set()
    for layer, count in layer_counts.items():
        attempts = 0
        added = 0
        while added < count and attempts < count * 100:
            col = rng.randrange(INTERMEDIATE_SIZE)
            cand = (layer, col)
            if cand not in anchor_set and cand not in chosen_set:
                chosen.append(cand)
                chosen_set.add(cand)
                added += 1
            attempts += 1
    return chosen


# =====================================================================
# MAIN SWEEP
# =====================================================================
def main():
    samples = load_medqa(N_SAMPLES, SEED)

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