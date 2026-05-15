"""
K-sweep ablation only for Bio-Medical-Llama-3-8B using fixed anchors from
anchors_bio_medical_llama_8b.json (no EMS discovery).

Loads model from /workspace/biomedical_llama3 (local_files_only).
Pads 31 discovered anchors to 32 by duplicating the last entry.
"""

import hashlib
import inspect
import json
import os
import random
from dataclasses import asdict, dataclass
from datetime import datetime
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from mednsq_data import build_adversarial_pairs, format_prompt, load_mcq_dataset
from mednsq_eval import evaluate_model
from mednsq_probe import MedNSQProbe

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_SCRIPT_DIR)

# Hardcoded anchor source (31 anchors; padded to 32 at runtime).
ANCHOR_JSON = os.path.join(_REPO_ROOT, "anchors_bio_medical_llama_8b.json")
TARGET_ANCHOR_COUNT = 32


@dataclass
class Config:
    model_name: str = "/workspace/biomedical_llama3"
    seed: int = 42

    calib_size: int = 400
    test_size: int = 300

    k_values: Tuple[int, ...] = (1, 2, 4, 8, 16, 32, 64)
    n_random_trials: int = 30
    ablation_eval_pairs: int = 200
    ablation_test_size: int = 300

    ablation_file: str = "ablation_bio_medical_llama_8b.json"
    log_file: str = "ksweep_bio_medical_llama_8b.log"
    intervention_type: str = "column_crush_1bit"


CFG = Config()


def log(msg: str) -> None:
    line = f"[{datetime.now().strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(CFG.log_file, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def resolve_local_model_path(configured: str) -> str:
    path = os.path.abspath(os.path.expanduser(os.environ.get("MEDNSQ_MODEL_PATH", configured)))
    if not os.path.isdir(path):
        raise FileNotFoundError(
            f"Local model directory not found: {path}\n"
            f"Expected weights at {configured!r} or set MEDNSQ_MODEL_PATH."
        )
    if not os.path.isfile(os.path.join(path, "config.json")):
        raise FileNotFoundError(f"Missing config.json under: {path}")
    return path


def load_hardcoded_anchors() -> List[Dict[str, Any]]:
    """Load anchors_bio_medical_llama_8b.json; pad to TARGET_ANCHOR_COUNT."""
    candidates = [
        ANCHOR_JSON,
        os.path.join(_SCRIPT_DIR, "anchors_bio_medical_llama_8b.json"),
        os.path.abspath("anchors_bio_medical_llama_8b.json"),
    ]
    path = next((p for p in candidates if os.path.isfile(p)), None)
    if path is None:
        raise FileNotFoundError(
            "Anchor JSON not found. Tried:\n  " + "\n  ".join(candidates)
        )

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    anchors = list(data["anchors"])
    n = len(anchors)
    log(f"Loaded {n} anchors from {path}")

    if n == 0:
        raise RuntimeError("Anchor list is empty.")

    if n < TARGET_ANCHOR_COUNT:
        pad = dict(anchors[-1])
        pad["padded_duplicate"] = True
        anchors.append(pad)
        log(
            f"Padded {n} -> {TARGET_ANCHOR_COUNT} by duplicating last anchor "
            f"L{pad['layer']} C{pad['column']}"
        )
    elif n > TARGET_ANCHOR_COUNT:
        log(f"Truncating {n} anchors to first {TARGET_ANCHOR_COUNT}")
        anchors = anchors[:TARGET_ANCHOR_COUNT]

    return anchors


def setup_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def setup_perf() -> None:
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision("high")


def crush_many(probe: MedNSQProbe, neurons: List[Tuple[int, int]]) -> List[Tuple[int, int, torch.Tensor]]:
    saved = []
    for layer, col in neurons:
        orig = probe.simulate_column_crush(layer, col)
        saved.append((layer, col, orig))
    return saved


def restore_many(probe: MedNSQProbe, saved: List[Tuple[int, int, torch.Tensor]]) -> None:
    for layer, col, orig in saved:
        probe.restore_column(layer, col, orig)


def sample_random_neurons(
    k: int,
    anchors_subset: List[Tuple[int, int]],
    intermediate_size: int,
    rng: random.Random,
) -> List[Tuple[int, int]]:
    from collections import defaultdict

    layer_counts: Dict[int, int] = defaultdict(int)
    for l, _ in anchors_subset:
        layer_counts[l] += 1
    anchor_set = set(anchors_subset)
    chosen: List[Tuple[int, int]] = []
    chosen_set = set()
    for layer, count in layer_counts.items():
        added = 0
        attempts = 0
        while added < count and attempts < count * 200:
            c = rng.randrange(intermediate_size)
            t = (layer, c)
            if t not in anchor_set and t not in chosen_set:
                chosen.append(t)
                chosen_set.add(t)
                added += 1
            attempts += 1
    return chosen


def run_ablation(
    model,
    tokenizer,
    probe: MedNSQProbe,
    anchors: List[Dict[str, Any]],
    calib_pairs: List[Dict[str, Any]],
    test_samples: List[Dict[str, Any]],
    pad_id: int,
) -> Dict[str, Any]:
    log("=== K-sweep ablation ===")
    anchor_tuples = [(a["layer"], a["column"]) for a in anchors]

    eval_pairs = calib_pairs[: CFG.ablation_eval_pairs]
    base_margins = probe.compute_per_sample_margins(eval_pairs, batch_size=32, pad_id=pad_id)
    base_margin = float(base_margins.mean().item())
    base_test = evaluate_model(model, tokenizer, test_samples)
    base_acc = base_test["accuracy"]
    log(f"Baseline margin={base_margin:.4f} accuracy={base_acc:.4f}")

    rng = random.Random(CFG.seed + 99)
    results: Dict[int, Dict[str, Any]] = {}

    for K in CFG.k_values:
        if K > len(anchor_tuples):
            log(f"Skipping K={K} (only {len(anchor_tuples)} anchors)")
            continue
        log(f"--- K={K} ---")
        subset = anchor_tuples[:K]

        saved = crush_many(probe, subset)
        try:
            anchor_margins = probe.compute_per_sample_margins(eval_pairs, batch_size=32, pad_id=pad_id)
            anchor_margin = float(anchor_margins.mean().item())
            anchor_test = evaluate_model(model, tokenizer, test_samples)
            anchor_acc = anchor_test["accuracy"]
        finally:
            restore_many(probe, saved)

        anchor_margin_drop = base_margin - anchor_margin
        anchor_acc_drop = base_acc - anchor_acc

        rand_margin_drops: List[float] = []
        rand_acc_drops: List[float] = []
        for trial in range(CFG.n_random_trials):
            rand_neurons = sample_random_neurons(K, subset, probe.intermediate_size, rng)
            saved_r = crush_many(probe, rand_neurons)
            try:
                rm = probe.compute_per_sample_margins(eval_pairs, batch_size=32, pad_id=pad_id)
                rand_margin = float(rm.mean().item())
                if trial < min(10, CFG.n_random_trials):
                    rt = evaluate_model(model, tokenizer, test_samples)
                    rand_acc = rt["accuracy"]
                    rand_acc_drops.append(base_acc - rand_acc)
            finally:
                restore_many(probe, saved_r)
            rand_margin_drops.append(base_margin - rand_margin)

        rmd = np.array(rand_margin_drops)
        rad = np.array(rand_acc_drops) if rand_acc_drops else np.array([0.0])
        z = (anchor_margin_drop - rmd.mean()) / (rmd.std() + 1e-8)
        translation = (anchor_acc_drop / anchor_margin_drop) if abs(anchor_margin_drop) > 1e-6 else 0.0

        log(f"  Anchor margin drop: {anchor_margin_drop:+.4f}")
        log(f"  Random margin drop: {rmd.mean():+.4f} ± {rmd.std():.4f}  (z = {z:+.2f})")
        log(f"  Anchor accuracy drop: {anchor_acc_drop:+.4f}")
        log(f"  Random accuracy drop: {rad.mean():+.4f} ± {rad.std():.4f}  (n={len(rad)})")
        log(f"  Translation rate (acc_drop / margin_drop): {translation:+.3f}")

        results[K] = {
            "K": K,
            "anchor_margin_drop": anchor_margin_drop,
            "anchor_acc_drop": anchor_acc_drop,
            "rand_margin_drop_mean": float(rmd.mean()),
            "rand_margin_drop_std": float(rmd.std()),
            "rand_acc_drop_mean": float(rad.mean()),
            "rand_acc_drop_std": float(rad.std()),
            "z_margin": float(z),
            "translation_rate": float(translation),
            "n_random_trials": CFG.n_random_trials,
        }

    return {
        "baseline_margin": base_margin,
        "baseline_accuracy": base_acc,
        "per_k": results,
    }


def main():
    setup_perf()
    setup_seeds(CFG.seed)
    log(f"Config: {asdict(CFG)}")

    anchors = load_hardcoded_anchors()
    log(f"Using {len(anchors)} anchors for K-sweep")

    model_path = resolve_local_model_path(CFG.model_name)
    log(f"Loading tokenizer + model from {model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True,
        use_fast=True,
        local_files_only=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    pad_id = tokenizer.pad_token_id

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
        local_files_only=True,
    )
    model.eval()
    probe = MedNSQProbe(model)

    log("Loading MedQA train (calib) + test splits...")
    train_pool = load_mcq_dataset(n_total=CFG.calib_size + 100, split="train")
    random.Random(CFG.seed).shuffle(train_pool)
    calib_samples = train_pool[: CFG.calib_size]
    test_samples = load_mcq_dataset(n_total=CFG.test_size, split="test")
    log(f"Calib={len(calib_samples)} Test={len(test_samples)}")

    log("Building adversarial pairs (calib)...")
    calib_pairs = build_adversarial_pairs(
        model, tokenizer, calib_samples, n_calib=len(calib_samples)
    )
    assert all(p["pos_id"] != p["neg_id"] for p in calib_pairs), "Bad adversarial pair (pos==neg)"
    log(f"Built {len(calib_pairs)} calib pairs")

    ablation = run_ablation(
        model,
        tokenizer,
        probe,
        anchors,
        calib_pairs,
        test_samples[: CFG.ablation_test_size],
        pad_id,
    )

    prompt_hash = hashlib.sha256(inspect.getsource(format_prompt).encode("utf-8")).hexdigest()[:16]
    ablation_output = {
        "metadata": {
            "timestamp": datetime.now().isoformat(),
            "model_name": CFG.model_name,
            "anchors_source": ANCHOR_JSON,
            "anchor_count": len(anchors),
            "anchors_used": [(a["layer"], a["column"]) for a in anchors],
            "config": asdict(CFG),
            "prompt_template_hash": prompt_hash,
            "intervention_type": CFG.intervention_type,
        },
        "results": ablation,
    }
    with open(CFG.ablation_file, "w", encoding="utf-8") as f:
        json.dump(ablation_output, f, indent=2)
    log(f"Saved ablation to {CFG.ablation_file}")

    log("\n=== FINAL SUMMARY ===")
    log(f"Baseline: margin={ablation['baseline_margin']:.4f}, accuracy={ablation['baseline_accuracy']:.4f}")
    log(f"{'K':>4}  {'AnchMarΔ':>10}  {'RandMarΔ':>10}  {'zMar':>6}  {'AnchAccΔ':>10}  {'TransRate':>10}")
    for K in CFG.k_values:
        r = ablation["per_k"].get(K)
        if r is None:
            continue
        log(
            f"{K:>4}  "
            f"{r['anchor_margin_drop']:>+10.4f}  "
            f"{r['rand_margin_drop_mean']:>+10.4f}  "
            f"{r['z_margin']:>+6.2f}  "
            f"{r['anchor_acc_drop']:>+10.4f}  "
            f"{r['translation_rate']:>+10.3f}"
        )


if __name__ == "__main__":
    main()
