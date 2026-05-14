"""
Cross-dataset anchor transfer for AlphaMed-8B-instruct-rl (che111/AlphaMed-8B-instruct-rl).

Weights load from CFG.model_path only (local snapshot; see
https://huggingface.co/che111/AlphaMed-8B-instruct-rl). Fixed anchors from JSON;
per-anchor column crush on MedQA / MedMCQA / PubMedQA with margin drops vs baselines.
"""

import os
import json
import random
import hashlib
from typing import Any, Dict, List, Optional, Tuple

from dataclasses import dataclass, asdict
from datetime import datetime

import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM

_mednsq_lib_dir = os.getenv("MEDNSQ_LIB_DIR")
if _mednsq_lib_dir and _mednsq_lib_dir not in os.sys.path:
    os.sys.path.insert(0, _mednsq_lib_dir)

from mednsq_data import build_adversarial_pairs, load_mcq_dataset
from mednsq_probe import MedNSQProbe


# ============================================================================
# CONFIGURATION
# ============================================================================

@dataclass
class Config:
    """Immutable configuration for reproducibility."""
    model_key: str = "alphamed_8b_instruct_rl"
    model_path: str = "/workspace/alphamed"
    anchor_file: str = "anchors_alphamed_8b_instruct_rl.json"

    n_medqa: int = 400
    n_medmcqa: int = 400
    n_pubmedqa: int = 400

    anchor_limit: Optional[int] = None

    random_seed: int = 42

    medqa_cache: str = "alphamed_medqa_pairs.txt"
    medmcqa_cache: str = "alphamed_medmcqa_pairs.txt"
    pubmedqa_cache: str = "alphamed_pubmedqa_pairs.txt"

    output_file: str = "crossdataset_alphamed_8b_instruct_rl.json"

    max_contexts: int = 3
    max_context_chars: int = 2200

    def __post_init__(self):
        if self.anchor_limit is not None:
            assert self.anchor_limit > 0, "anchor_limit must be positive when set"

    @property
    def config_dict(self) -> Dict:
        return asdict(self)

    @property
    def config_hash(self) -> str:
        return hashlib.sha256(
            json.dumps(self.config_dict, sort_keys=True, default=str).encode()
        ).hexdigest()[:8]


CONFIG = Config()


# ============================================================================
# TOKEN UTILITIES
# ============================================================================

def get_single_token_id(tokenizer, text: str) -> int:
    """Robust token ID resolution for single tokens."""
    for candidate in [f" {text}", text]:
        ids = tokenizer.encode(candidate, add_special_tokens=False)
        if len(ids) == 1:
            return int(ids[0])

    ids = tokenizer.encode(f" {text}", add_special_tokens=False)
    if not ids:
        raise RuntimeError(f"Cannot resolve token id for: '{text}'")

    print(f"Warning: '{text}' split into {len(ids)} tokens, using first")
    return int(ids[0])


def get_letter_token_ids(tokenizer) -> Dict[str, int]:
    """Get token IDs for A, B, C, D."""
    return {l: get_single_token_id(tokenizer, l) for l in ["A", "B", "C", "D"]}


# ============================================================================
# DATA CACHING
# ============================================================================

def save_pairs(path: str, pairs: List[Dict[str, Any]]) -> None:
    """Save adversarial pairs to cache with metadata."""
    if not pairs:
        return
    with open(path, "w", encoding="utf-8") as f:
        metadata = {
            "version": "1.0",
            "n_pairs": len(pairs),
            "created": datetime.now().isoformat(),
            "config_hash": CONFIG.config_hash,
        }
        f.write("#" + json.dumps(metadata) + "\n")

        for p in pairs:
            safe_ids = p.get("safe_input_ids", p["input_ids"])
            safe_mask = p.get("safe_attention_mask", p["attention_mask"])
            row = {
                "input_ids": p["input_ids"][0].tolist(),
                "attention_mask": p["attention_mask"][0].tolist(),
                "safe_input_ids": safe_ids[0].tolist(),
                "safe_attention_mask": safe_mask[0].tolist(),
                "pos_id": int(p["pos_id"]),
                "neg_id": int(p["neg_id"]),
            }
            f.write(json.dumps(row) + "\n")


def load_pairs(path: str) -> List[Dict[str, Any]]:
    """Load adversarial pairs from cache, skipping metadata."""
    pairs: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            row = json.loads(line)
            sid = row.get("safe_input_ids", row["input_ids"])
            sm = row.get("safe_attention_mask", row["attention_mask"])
            pairs.append({
                "input_ids": torch.tensor([row["input_ids"]], dtype=torch.long),
                "attention_mask": torch.tensor([row["attention_mask"]], dtype=torch.long),
                "safe_input_ids": torch.tensor([sid], dtype=torch.long),
                "safe_attention_mask": torch.tensor([sm], dtype=torch.long),
                "pos_id": int(row["pos_id"]),
                "neg_id": int(row["neg_id"]),
            })
    return pairs


# ============================================================================
# DATASET LOADERS
# ============================================================================

def _build_medqa_pairs_with_pooling(
    model,
    tokenizer,
    n_target: int,
) -> List[Dict[str, Any]]:
    """MedQA pairs require baseline-correct MCQ answers; pool more items if needed."""
    splits_order = ("train", "test", "validation")
    multipliers = (1, 2, 3, 4, 6, 8, 12, 16, 24, 32)
    best: List[Dict[str, Any]] = []
    for split in splits_order:
        for m in multipliers:
            n_pool = min(n_target * m, 20000)
            try:
                ds = load_mcq_dataset(n_total=n_pool, split=split)
            except Exception as exc:
                print(f"  MedQA: skip split={split!r} n_pool={n_pool}: {exc}")
                continue
            if not ds:
                continue
            pairs = build_adversarial_pairs(
                model=model,
                tokenizer=tokenizer,
                dataset=ds,
                n_calib=len(ds),
            )
            if len(pairs) > len(best):
                best = list(pairs)
            if len(pairs) >= n_target:
                return pairs[:n_target]
    if len(best) >= n_target:
        return best[:n_target]
    return best


def get_medqa_pairs(model, tokenizer, n_total: int) -> List[Dict[str, Any]]:
    """Load or build MedQA adversarial pairs (invalid/short cache is rebuilt)."""
    cache_path = CONFIG.medqa_cache
    if os.path.exists(cache_path):
        cached = load_pairs(cache_path)
        if len(cached) >= n_total:
            print(f"Loading MedQA from cache: {cache_path}")
            return cached[:n_total]
        print(
            f"MedQA cache ignored ({len(cached)} pairs, need {n_total}); rebuilding..."
        )
        try:
            os.remove(cache_path)
        except OSError:
            pass

    print(f"Building MedQA pairs (target n={n_total})...")
    pairs = _build_medqa_pairs_with_pooling(model, tokenizer, n_total)
    if not pairs:
        raise RuntimeError(
            "MedQA: no adversarial pairs passed the baseline-correct filter. "
            "Check the model checkpoint and tokenizer alignment with MedQA."
        )
    if len(pairs) < n_total:
        print(
            f"  MedQA: only {len(pairs)} pairs available (wanted {n_total}); "
            "proceeding with fewer."
        )
    else:
        pairs = pairs[:n_total]
    save_pairs(cache_path, pairs)
    return pairs


def make_pubmed_prompt(question: str, contexts: List[str]) -> str:
    """Format PubMedQA prompt with context truncation."""
    ctx = "\n".join(f"- {c}" for c in contexts[:CONFIG.max_contexts] if c)
    if len(ctx) > CONFIG.max_context_chars:
        ctx = ctx[:CONFIG.max_context_chars]
    return (
        "You are answering a biomedical yes/no question.\n"
        f"Question: {question}\n"
        f"Context:\n{ctx}\n\n"
        "Answer with one word (yes or no).\n"
        "Answer:"
    )


def get_pubmedqa_pairs(tokenizer, n_total: int) -> List[Dict[str, Any]]:
    """Load or build PubMedQA adversarial pairs."""
    if os.path.exists(CONFIG.pubmedqa_cache):
        cached = load_pairs(CONFIG.pubmedqa_cache)
        if len(cached) >= n_total:
            print(f"Loading PubMedQA from cache: {CONFIG.pubmedqa_cache}")
            return cached[:n_total]
        print(
            f"PubMedQA cache ignored ({len(cached)} pairs, need {n_total}); rebuilding..."
        )
        try:
            os.remove(CONFIG.pubmedqa_cache)
        except OSError:
            pass

    print(f"Building PubMedQA pairs (n={n_total})...")
    yes_id = get_single_token_id(tokenizer, "yes")
    no_id = get_single_token_id(tokenizer, "no")
    ds = load_dataset("pubmed_qa", "pqa_labeled", split="train")

    pairs = []
    for row in ds:
        gold = str(row.get("final_decision", "")).strip().lower()
        if gold not in {"yes", "no"}:
            continue

        question = str(row.get("question", "")).strip()
        ctx_obj = row.get("context", {})
        contexts = ctx_obj.get("contexts", []) if isinstance(ctx_obj, dict) else []

        if not question:
            continue

        enc = tokenizer(
            make_pubmed_prompt(question, contexts),
            return_tensors="pt",
        )
        pos_id = yes_id if gold == "yes" else no_id
        neg_id = no_id if gold == "yes" else yes_id

        pairs.append({
            "input_ids": enc["input_ids"],
            "attention_mask": enc["attention_mask"],
            "safe_input_ids": enc["input_ids"].clone(),
            "safe_attention_mask": enc["attention_mask"].clone(),
            "pos_id": int(pos_id),
            "neg_id": int(neg_id),
        })

        if len(pairs) >= n_total:
            break

    if not pairs:
        raise RuntimeError("No PubMedQA pairs built.")

    save_pairs(CONFIG.pubmedqa_cache, pairs)
    return pairs


def make_medmcqa_prompt(row: Dict[str, Any]) -> str:
    """Format MedMCQA prompt."""
    return (
        f"Question: {str(row.get('question', '')).strip()}\n"
        f"Options: (A) {str(row.get('opa', '')).strip()} "
        f"(B) {str(row.get('opb', '')).strip()} "
        f"(C) {str(row.get('opc', '')).strip()} "
        f"(D) {str(row.get('opd', '')).strip()}\n"
        "Answer: ("
    )


def get_medmcqa_pairs(model, tokenizer, n_total: int) -> List[Dict[str, Any]]:
    """Load or build MedMCQA adversarial pairs."""
    if os.path.exists(CONFIG.medmcqa_cache):
        cached = load_pairs(CONFIG.medmcqa_cache)
        if len(cached) >= n_total:
            print(f"Loading MedMCQA from cache: {CONFIG.medmcqa_cache}")
            return cached[:n_total]
        print(
            f"MedMCQA cache ignored ({len(cached)} pairs, need {n_total}); rebuilding..."
        )
        try:
            os.remove(CONFIG.medmcqa_cache)
        except OSError:
            pass

    print(f"Building MedMCQA pairs (n={n_total})...")
    ds = load_dataset("openlifescienceai/medmcqa", split="train")
    letter_ids = get_letter_token_ids(tokenizer)
    idx_to_letter = {0: "A", 1: "B", 2: "C", 3: "D"}
    device = next(model.parameters()).device

    pairs = []
    for row in ds:
        try:
            cop = int(row.get("cop", -1))
        except (TypeError, ValueError):
            continue

        if cop not in idx_to_letter:
            continue

        enc = tokenizer(make_medmcqa_prompt(row), return_tensors="pt")

        with torch.no_grad():
            logits = model(
                input_ids=enc["input_ids"].to(device),
                attention_mask=enc["attention_mask"].to(device),
            ).logits[0, -1, :].float()

        correct = idx_to_letter[cop]
        pos_id = int(letter_ids[correct])
        wrong_ids = [int(letter_ids[l]) for l in ["A", "B", "C", "D"] if l != correct]
        wrong_logits = logits[torch.tensor(wrong_ids, device=logits.device)]
        neg_id = int(wrong_ids[int(torch.argmax(wrong_logits).item())])

        pairs.append({
            "input_ids": enc["input_ids"],
            "attention_mask": enc["attention_mask"],
            "safe_input_ids": enc["input_ids"].clone(),
            "safe_attention_mask": enc["attention_mask"].clone(),
            "pos_id": pos_id,
            "neg_id": neg_id,
        })

        if len(pairs) >= n_total:
            break

    if not pairs:
        raise RuntimeError("No MedMCQA pairs built.")

    save_pairs(CONFIG.medmcqa_cache, pairs)
    return pairs


# ============================================================================
# CORE METRICS
# ============================================================================

def mean_drop_for_neuron(
    probe: MedNSQProbe,
    pairs: List[Dict[str, Any]],
    baseline: torch.Tensor,
    layer: int,
    col: int,
) -> float:
    """Mean margin drop for a single neuron (column crush)."""
    orig = probe.simulate_column_crush(layer, col)

    try:
        ablated = probe.compute_per_sample_margins(pairs)
    finally:
        probe.restore_column(layer, col, orig)

    if baseline.numel() == 0:
        return 0.0

    drops = baseline - ablated
    return float(drops.mean().item())


def margin_tensor_stats(margins: torch.Tensor) -> Dict[str, float]:
    """Baseline margin statistics for one dataset."""
    if margins.numel() == 0:
        return {"mean": 0.0, "std": 0.0, "frac_neg": 0.0, "n": 0}
    m = margins.float()
    n = int(m.numel())
    mean_v = float(m.mean().item())
    std_v = float(m.std().item()) if n > 1 else 0.0
    frac_neg = float((m < 0).float().mean().item())
    return {"mean": mean_v, "std": std_v, "frac_neg": frac_neg, "n": n}


def rel_drop(drop: float, baseline_mean: float) -> float:
    if abs(baseline_mean) < 1e-12:
        return 0.0
    return drop / baseline_mean


# ============================================================================
# ANCHORS
# ============================================================================

def load_anchors_from_json(path: str) -> Tuple[List[List[int]], List[Tuple[int, int]]]:
    """
    Returns anchors_input (all [layer, column] from JSON) and anchor_neurons
    (possibly truncated by CONFIG.anchor_limit) as (layer, column) tuples.
    """
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    raw = [(int(a["layer"]), int(a["column"])) for a in data["anchors"]]
    anchors_input = [[l, c] for l, c in raw]
    tuples = list(raw)
    if CONFIG.anchor_limit is not None:
        tuples = tuples[: CONFIG.anchor_limit]
    return anchors_input, tuples


# ============================================================================
# MAIN
# ============================================================================

def main():
    random.seed(CONFIG.random_seed)
    torch.manual_seed(CONFIG.random_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(CONFIG.random_seed)

    model_path = CONFIG.model_path

    print("Loading model...")
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True,
        use_fast=True,
        local_files_only=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
            local_files_only=True,
        )
    except TypeError:
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
            local_files_only=True,
        )
    model.eval()
    probe = MedNSQProbe(model)

    print("Preparing dataset pairs...")
    medqa = get_medqa_pairs(model, tokenizer, CONFIG.n_medqa)
    medmcqa = get_medmcqa_pairs(model, tokenizer, CONFIG.n_medmcqa)
    pubmedqa = get_pubmedqa_pairs(tokenizer, CONFIG.n_pubmedqa)

    datasets = {
        "medqa": medqa,
        "medmcqa": medmcqa,
        "pubmedqa": pubmedqa,
    }

    print("Loading anchors...")
    if not os.path.isfile(CONFIG.anchor_file):
        raise FileNotFoundError(
            f"Anchor file not found: {CONFIG.anchor_file} "
            "(expected keys: anchors[].layer, anchors[].column)"
        )
    anchors_input, anchor_neurons = load_anchors_from_json(CONFIG.anchor_file)
    if not anchor_neurons:
        raise RuntimeError("No anchors loaded from JSON.")
    anchors_used = [[layer, col] for layer, col in anchor_neurons]

    print("Computing dataset baselines...")
    baseline_tensors = {
        name: probe.compute_per_sample_margins(pairs)
        for name, pairs in datasets.items()
    }
    dataset_baselines = {
        name: margin_tensor_stats(baseline_tensors[name])
        for name in ("medqa", "medmcqa", "pubmedqa")
    }

    bm = {k: dataset_baselines[k]["mean"] for k in dataset_baselines}

    print("Per-anchor cross-dataset margin drops...")
    rows: List[Dict[str, Any]] = []
    n_anchors = len(anchor_neurons)
    for i, (l, c) in enumerate(anchor_neurons, start=1):
        d1 = mean_drop_for_neuron(probe, medqa, baseline_tensors["medqa"], l, c)
        d2 = mean_drop_for_neuron(probe, medmcqa, baseline_tensors["medmcqa"], l, c)
        d3 = mean_drop_for_neuron(probe, pubmedqa, baseline_tensors["pubmedqa"], l, c)
        row_data = {
            "layer": l,
            "column": c,
            "drop_medqa": d1,
            "drop_medmcqa": d2,
            "drop_pubmedqa": d3,
            "rel_drop_medqa": rel_drop(d1, bm["medqa"]),
            "rel_drop_medmcqa": rel_drop(d2, bm["medmcqa"]),
            "rel_drop_pubmedqa": rel_drop(d3, bm["pubmedqa"]),
        }
        rows.append(row_data)
        print(
            f"[{i:03d}/{n_anchors:03d}] L{l} C{c}: "
            f"mqa={d1:+.4f}, mmcqa={d2:+.4f}, pmqa={d3:+.4f}"
        )

    metadata = {
        "model_key": CONFIG.model_key,
        "model_path": CONFIG.model_path,
        "timestamp": datetime.now().isoformat(),
        "torch_version": torch.__version__,
        "ablation_method": "column_crush_1bit",
        "anchor_count": len(anchor_neurons),
        "anchors_input": anchors_input,
        "anchors_used": anchors_used,
    }

    out = {
        "metadata": metadata,
        "dataset_baselines": dataset_baselines,
        "anchors": rows,
    }

    print(f"Saving results to {CONFIG.output_file}...")
    with open(CONFIG.output_file, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)

    print(f"Done. Results saved to {CONFIG.output_file}")


if __name__ == "__main__":
    main()
