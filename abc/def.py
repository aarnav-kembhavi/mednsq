"""
BRUTE FORCE DISCOVERY ENGINE
=============================
Runs 50+ statistical comparisons across all cross_dataset_results.json files.
Finds patterns nobody looked for. Prints everything significant.

Usage:
    python bruteforce_discovery.py

Put this in the same folder as your *_cross_dataset_results.json files.
"""

import json
import glob
import os
import numpy as np
from scipy import stats
from scipy.stats import (
    mannwhitneyu, kruskal, spearmanr, pearsonr,
    chi2_contingency, ttest_ind, ks_2samp, entropy
)
from itertools import combinations, product
import warnings

def safe_kruskal(groups):
    # remove small groups
    groups = [g for g in groups if len(g) >= 2]
    if len(groups) < 2:
        return None, None

    # check variance inside each group
    if any(len(set(g)) <= 1 for g in groups):
        return None, None

    # check global variance
    all_vals = [x for g in groups for x in g]
    if len(set(all_vals)) <= 1:
        return None, None

    return kruskal(*groups)
warnings.filterwarnings("ignore")

# ── ANSI colors for terminal ──────────────────────────────────────────────
GREEN  = "\033[92m"
RED    = "\033[91m"
YELLOW = "\033[93m"
CYAN   = "\033[96m"
BOLD   = "\033[1m"
RESET  = "\033[0m"

def sig(p):
    if p < 0.001: return f"{RED}*** p={p:.2e}{RESET}"
    if p < 0.01:  return f"{YELLOW}**  p={p:.4f}{RESET}"
    if p < 0.05:  return f"{GREEN}*   p={p:.4f}{RESET}"
    return f"    p={p:.4f} ns"

def banner(title):
    print(f"\n{BOLD}{CYAN}{'='*65}{RESET}")
    print(f"{BOLD}{CYAN}  {title}{RESET}")
    print(f"{BOLD}{CYAN}{'='*65}{RESET}")

# ── Load all result files ─────────────────────────────────────────────────
files = glob.glob("*_cross_dataset_results.json")
if not files:
    files = glob.glob("**/*_cross_dataset_results.json", recursive=True)

print(f"Found {len(files)} result files: {[os.path.basename(f) for f in files]}")

models = {}
for f in files:
    with open(f) as fp:
        data = json.load(fp)
    if "anchors" not in data:
        continue
    key = data.get("metadata", {}).get("model_key",
          os.path.basename(f).split("_cross")[0])
    models[key] = data

print(f"Loaded models: {list(models.keys())}\n")

# ── Build unified anchor dataframe ────────────────────────────────────────
all_anchors = []
for model_key, data in models.items():
    training = data.get("metadata", {}).get("config", {}).get("training",
               "unknown")
    # Try to infer training regime from model name
    mk = model_key.lower()
    if "rl" in mk or "openmed" in mk or "afm" in mk and "base" not in mk:
        regime = "RL"
    elif "meditron" in mk:
        regime = "CPT+SFT"
    elif "med42" in mk or "medgemma" in mk:
        regime = "SFT"
    elif "base" in mk or "llama" in mk or "qwen" in mk or "gemma" in mk:
        regime = "BASE"
    else:
        regime = "UNKNOWN"

    for a in data["anchors"]:
        all_anchors.append({
            "model":        model_key,
            "regime":       regime,
            "layer":        int(a["layer"]),
            "column":       int(a["column"]),
            "drop_medqa":   float(a.get("drop_medqa",   a.get("drop_medqa",   0))),
            "drop_medmcqa": float(a.get("drop_medmcqa", a.get("drop_medmcqa", 0))),
            "drop_pubmedqa":float(a.get("drop_pubmedqa",a.get("drop_pubmedqa",0))),
            "cluster":      int(a.get("cluster", -1)),
            # Derived
            "is_sabotage":  float(a.get("drop_pubmedqa", 0)) < 0,
            "is_helper":    float(a.get("drop_pubmedqa", 0)) > 0,
            "total_abs_ems":abs(float(a.get("drop_medqa",0))) +
                            abs(float(a.get("drop_medmcqa",0))) +
                            abs(float(a.get("drop_pubmedqa",0))),
            "cross_task_sign_consistency":
                # +1 if all three same sign, -1 if mixed
                1 if (
                    float(a.get("drop_medqa",0)) > 0 and
                    float(a.get("drop_medmcqa",0)) > 0 and
                    float(a.get("drop_pubmedqa",0)) > 0
                ) or (
                    float(a.get("drop_medqa",0)) < 0 and
                    float(a.get("drop_medmcqa",0)) < 0 and
                    float(a.get("drop_pubmedqa",0)) < 0
                ) else -1,
            "mcq_vs_binary_divergence":
                abs(float(a.get("drop_medqa",0))) -
                abs(float(a.get("drop_pubmedqa",0))),
        })

print(f"Total anchors across all models: {len(all_anchors)}\n")

# ── Helper: get arrays ────────────────────────────────────────────────────
def get(field, model=None, regime=None, sabotage=None):
    filtered = all_anchors
    if model:   filtered = [a for a in filtered if a["model"] == model]
    if regime:  filtered = [a for a in filtered if a["regime"] == regime]
    if sabotage is not None:
        filtered = [a for a in filtered if a["is_sabotage"] == sabotage]
    return np.array([a[field] for a in filtered])

discoveries = []
comparison_count = 0

def record(title, finding, p=None, effect=None):
    global comparison_count
    comparison_count += 1
    is_significant = p is not None and p < 0.05
    discoveries.append({
        "n": comparison_count,
        "title": title,
        "finding": finding,
        "p": p,
        "effect": effect,
        "significant": is_significant,
    })
    marker = f"{RED}🔥 SIGNIFICANT{RESET}" if is_significant else "   "
    print(f"  [{comparison_count:>3}] {marker} {title}")
    print(f"       {finding}")
    if p is not None:
        print(f"       {sig(p)}" + (f"  effect={effect:.4f}" if effect else ""))
    print()

# =============================================================================
# BLOCK 1: SABOTAGE NEURON PROPERTIES
# =============================================================================
banner("BLOCK 1: SABOTAGE NEURON PROPERTIES")

# 1. Do sabotage neurons live in deeper layers? (within each model)
for model_key in models:
    sab_layers  = get("layer", model=model_key, sabotage=True)
    norm_layers = get("layer", model=model_key, sabotage=False)
    if len(sab_layers) < 3 or len(norm_layers) < 3:
        continue
    stat, p = mannwhitneyu(sab_layers, norm_layers, alternative="greater")
    effect = sab_layers.mean() - norm_layers.mean()
    record(
        f"[{model_key}] Sabotage neurons in deeper layers?",
        f"sab_mean={sab_layers.mean():.2f}  norm_mean={norm_layers.mean():.2f}  shift={effect:+.2f}",
        p=p, effect=effect
    )

# 2. Do sabotage neurons have HIGHER absolute EMS on MedQA?
for model_key in models:
    sab_ems  = np.abs(get("drop_medqa", model=model_key, sabotage=True))
    norm_ems = np.abs(get("drop_medqa", model=model_key, sabotage=False))
    if len(sab_ems) < 3 or len(norm_ems) < 3:
        continue
    stat, p = mannwhitneyu(sab_ems, norm_ems, alternative="greater")
    record(
        f"[{model_key}] Sabotage neurons have higher MedQA EMS?",
        f"sab_abs_ems={sab_ems.mean():.5f}  norm_abs_ems={norm_ems.mean():.5f}",
        p=p, effect=sab_ems.mean()-norm_ems.mean()
    )

# 3. Do sabotage neurons have higher TOTAL absolute EMS?
for model_key in models:
    sab  = get("total_abs_ems", model=model_key, sabotage=True)
    norm = get("total_abs_ems", model=model_key, sabotage=False)
    if len(sab) < 3 or len(norm) < 3:
        continue
    stat, p = mannwhitneyu(sab, norm, alternative="two-sided")
    record(
        f"[{model_key}] Sabotage neurons vs normal: total |EMS|?",
        f"sab={sab.mean():.5f}  norm={norm.mean():.5f}",
        p=p, effect=sab.mean()-norm.mean()
    )

# 4. Is sabotage% correlated with depth shift ACROSS models?
model_stats = []
for model_key, data in models.items():
    anchors = data["anchors"]
    sab = [a for a in anchors if a.get("drop_pubmedqa", 0) < 0]
    non = [a for a in anchors if a.get("drop_pubmedqa", 0) >= 0]
    if not sab or not non:
        continue
    sab_pct   = len(sab) / len(anchors)
    depth_shift = np.mean([a["layer"] for a in sab]) - np.mean([a["layer"] for a in non])
    model_stats.append((sab_pct, depth_shift, model_key))

if len(model_stats) >= 3:
    xs = [m[0] for m in model_stats]
    ys = [m[1] for m in model_stats]
    r, p = spearmanr(xs, ys)
    record(
        "CROSS-MODEL: Sabotage% vs depth shift correlation",
        f"r={r:.3f}  models={[m[2] for m in model_stats]}",
        p=p, effect=r
    )

# =============================================================================
# BLOCK 2: LAYER POSITION PATTERNS
# =============================================================================
banner("BLOCK 2: LAYER POSITION PATTERNS")

# 5. Do RL anchors have LOWER layer variance than SFT?
for regime_a, regime_b in [("RL","SFT"), ("RL","CPT+SFT"), ("SFT","BASE")]:
    a_layers = get("layer", regime=regime_a)
    b_layers = get("layer", regime=regime_b)
    if len(a_layers) < 3 or len(b_layers) < 3:
        continue
    # Levene's test for equal variance
    stat, p = stats.levene(a_layers, b_layers)
    effect = a_layers.std() - b_layers.std()
    record(
        f"Layer variance: {regime_a} vs {regime_b}",
        f"{regime_a}_std={a_layers.std():.3f}  {regime_b}_std={b_layers.std():.3f}",
        p=p, effect=effect
    )

# 6. Do anchors concentrate in specific layer bands?
for model_key, data in models.items():
    layers = [a["layer"] for a in data["anchors"]]
    if len(layers) < 5:
        continue
    total_layers = data.get("metadata", {}).get("config", {}).get("n_layers", 36)
    # Test if distribution is uniform across thirds
    third = total_layers // 3
    early  = sum(1 for l in layers if l < third)
    mid    = sum(1 for l in layers if third <= l < 2*third)
    late   = sum(1 for l in layers if l >= 2*third)
    expected = len(layers) / 3
    chi2 = ((early-expected)**2 + (mid-expected)**2 + (late-expected)**2) / expected
    p = 1 - stats.chi2.cdf(chi2, df=2)
    record(
        f"[{model_key}] Anchors non-uniform across depth?",
        f"early={early} mid={mid} late={late} (expected {expected:.0f} each)",
        p=p, effect=chi2
    )

# 7. Is there a preferred "anchor column range"?
for model_key, data in models.items():
    cols = [a["column"] for a in data["anchors"]]
    n_cols = data.get("metadata", {}).get("anchor_count", len(cols))
    if len(cols) < 5:
        continue
    # Test if columns cluster (high variance = random; low = clustered)
    col_cv = np.std(cols) / (np.mean(cols) + 1e-8)
    # Compare to uniform distribution on [0, max_col]
    max_col = max(cols)
    uniform_cv = (max_col / np.sqrt(12)) / (max_col / 2)
    record(
        f"[{model_key}] Column indices clustered vs uniform?",
        f"actual_CV={col_cv:.3f}  uniform_CV={uniform_cv:.3f}  "
        f"col_mean={np.mean(cols):.0f}  col_std={np.std(cols):.0f}",
        p=None
    )

# =============================================================================
# BLOCK 3: EMS MAGNITUDE AND DISTRIBUTION
# =============================================================================
banner("BLOCK 3: EMS MAGNITUDE AND DISTRIBUTION")

# 8. Is EMS distribution heavy-tailed (Pareto-like)?
for model_key, data in models.items():
    ems_vals = np.abs([a.get("drop_medqa", 0) for a in data["anchors"]])
    if len(ems_vals) < 5:
        continue
    # Gini coefficient
    ems_sorted = np.sort(ems_vals)
    n = len(ems_sorted)
    gini = (2 * np.sum(np.arange(1, n+1) * ems_sorted) /
            (n * ems_sorted.sum()) - (n+1)/n) if ems_sorted.sum() > 0 else 0
    # Top-5 concentration
    top5_conc = ems_sorted[-5:].sum() / ems_sorted.sum() if ems_sorted.sum() > 0 else 0
    record(
        f"[{model_key}] EMS concentration (Gini/top5)",
        f"Gini={gini:.3f}  top5_frac={top5_conc:.3f}  "
        f"max_ems={ems_sorted.max():.5f}  mean_ems={ems_sorted.mean():.5f}",
        p=None
    )

# 9. KS test: are EMS distributions different across training regimes?
regime_ems = {}
for model_key, data in models.items():
    mk = model_key.lower()
    if "rl" in mk and "base" not in mk:   regime = "RL"
    elif "meditron" in mk:                regime = "CPT+SFT"
    elif "med42" in mk or "medgemma" in mk: regime = "SFT"
    else:                                 regime = "BASE"
    vals = [abs(a.get("drop_medqa", 0)) for a in data["anchors"]]
    regime_ems.setdefault(regime, []).extend(vals)

for r1, r2 in combinations(regime_ems.keys(), 2):
    if len(regime_ems[r1]) < 3 or len(regime_ems[r2]) < 3:
        continue
    stat, p = ks_2samp(regime_ems[r1], regime_ems[r2])
    record(
        f"EMS distribution: {r1} vs {r2} (KS test)",
        f"KS_stat={stat:.3f}  {r1}_mean={np.mean(regime_ems[r1]):.5f}  "
        f"{r2}_mean={np.mean(regime_ems[r2]):.5f}",
        p=p, effect=stat
    )

# 10. Cross-dataset EMS correlation per neuron
for model_key, data in models.items():
    anchors = data["anchors"]
    if len(anchors) < 5:
        continue
    mqa   = [a.get("drop_medqa",   0) for a in anchors]
    mmcqa = [a.get("drop_medmcqa", 0) for a in anchors]
    pmqa  = [a.get("drop_pubmedqa",0) for a in anchors]

    r1, p1 = spearmanr(mqa, mmcqa)
    r2, p2 = spearmanr(mqa, pmqa)
    r3, p3 = spearmanr(mmcqa, pmqa)

    record(
        f"[{model_key}] MedQA↔MedMCQA per-neuron correlation",
        f"r={r1:.3f}", p=p1, effect=r1
    )
    record(
        f"[{model_key}] MedQA↔PubMedQA per-neuron correlation",
        f"r={r2:.3f}", p=p2, effect=r2
    )
    record(
        f"[{model_key}] MedMCQA↔PubMedQA per-neuron correlation",
        f"r={r3:.3f}", p=p3, effect=r3
    )

# =============================================================================
# BLOCK 4: FORMAT SPECIFICITY SIGNATURES
# =============================================================================
banner("BLOCK 4: FORMAT SPECIFICITY SIGNATURES")

# 11. MCQ-format neurons vs binary-format neurons: layer difference?
for model_key, data in models.items():
    anchors = data["anchors"]
    mcq_specialist = [a for a in anchors
                      if a.get("drop_medqa",0) > 0 and a.get("drop_pubmedqa",0) < 0]
    binary_helper  = [a for a in anchors
                      if a.get("drop_pubmedqa",0) > 0 and a.get("drop_medqa",0) < 0]
    neutral        = [a for a in anchors
                      if abs(a.get("drop_pubmedqa",0)) < 0.002 and
                         abs(a.get("drop_medqa",0)) < 0.002]

    record(
        f"[{model_key}] Functional breakdown",
        f"MCQ-specialist(help_mqa,hurt_pmqa)={len(mcq_specialist)}  "
        f"Binary-helper(help_pmqa,hurt_mqa)={len(binary_helper)}  "
        f"Neutral={len(neutral)}  Total={len(anchors)}",
        p=None
    )

    if len(mcq_specialist) >= 2 and len(binary_helper) >= 2:
        mcq_layers    = [a["layer"] for a in mcq_specialist]
        binary_layers = [a["layer"] for a in binary_helper]
        stat, p = mannwhitneyu(mcq_layers, binary_layers, alternative="two-sided")
        record(
            f"[{model_key}] MCQ-specialist vs binary-helper: layer difference?",
            f"MCQ_layer={np.mean(mcq_layers):.1f}  binary_layer={np.mean(binary_layers):.1f}",
            p=p, effect=np.mean(mcq_layers)-np.mean(binary_layers)
        )

# 12. Sign consistency across datasets: does RL have more consistent neurons?
for model_key, data in models.items():
    anchors = data["anchors"]
    if not anchors:
        continue
    consistent = sum(1 for a in anchors
                     if (a.get("drop_medqa",0) > 0) ==
                        (a.get("drop_medmcqa",0) > 0) ==
                        (a.get("drop_pubmedqa",0) > 0))
    pct_consistent = consistent / len(anchors)
    record(
        f"[{model_key}] Cross-dataset sign consistency",
        f"consistent={consistent}/{len(anchors)} = {pct_consistent:.1%}",
        p=None
    )

# =============================================================================
# BLOCK 5: LAYER × EMS INTERACTIONS
# =============================================================================
banner("BLOCK 5: LAYER × EMS INTERACTIONS")

# 13. Does layer depth predict EMS magnitude within each model?
for model_key, data in models.items():
    anchors = data["anchors"]
    if len(anchors) < 5:
        continue
    layers  = [a["layer"] for a in anchors]
    ems_abs = [abs(a.get("drop_medqa", 0)) for a in anchors]
    r, p = spearmanr(layers, ems_abs)
    record(
        f"[{model_key}] Layer depth → |EMS| correlation",
        f"r={r:.3f}  (positive = later layers have bigger effects)",
        p=p, effect=r
    )

# 14. Does layer predict SABOTAGE SIGN?
for model_key, data in models.items():
    anchors = data["anchors"]
    if len(anchors) < 5:
        continue
    layers   = [a["layer"] for a in anchors]
    is_sabotage = [1 if a.get("drop_pubmedqa",0) < 0 else 0 for a in anchors]
    if sum(is_sabotage) < 2 or sum(1-s for s in is_sabotage) < 2:
        continue
    r, p = spearmanr(layers, is_sabotage)
    record(
        f"[{model_key}] Layer depth predicts sabotage sign?",
        f"r={r:.3f}  (positive = deeper → more sabotage)",
        p=p, effect=r
    )

# 15. Does layer predict MCQ↔binary DIVERGENCE?
for model_key, data in models.items():
    anchors = data["anchors"]
    if len(anchors) < 5:
        continue
    layers = [a["layer"] for a in anchors]
    diverg = [abs(a.get("drop_medqa",0)) - abs(a.get("drop_pubmedqa",0))
              for a in anchors]
    r, p = spearmanr(layers, diverg)
    record(
        f"[{model_key}] Layer depth → MCQ/binary EMS divergence",
        f"r={r:.3f}  (positive = deeper layers more MCQ-specific)",
        p=p, effect=r
    )

# =============================================================================
# BLOCK 6: CROSS-MODEL COMPARISONS
# =============================================================================
banner("BLOCK 6: CROSS-MODEL COMPARISONS")

# 16. Kruskal-Wallis: do all models differ in layer distribution?
all_model_layers = {
    k: [a["layer"] for a in v["anchors"]]
    for k, v in models.items() if v["anchors"]
}
if len(all_model_layers) >= 3:
    groups = [v for v in all_model_layers.values() if len(v) >= 3]
    if len(groups) >= 3:
        stat, p = safe_kruskal(groups)
        if stat is None:
            record(
                "your title here",
                "Skipped (degenerate data)",
                p=None
            )
        else:
            record(
                "KRUSKAL-WALLIS: Do all models differ in anchor layer distribution?",
                f"H={stat:.3f}  k={len(groups)} groups",
                p=p, effect=stat
            )

# 17. SFT vs RL: layer distribution (pooled)
sft_layers = []
rl_layers  = []
for model_key, data in models.items():
    mk = model_key.lower()
    layers = [a["layer"] for a in data["anchors"]]
    if "rl" in mk and "base" not in mk:
        rl_layers.extend(layers)
    elif "med42" in mk or "medgemma" in mk:
        sft_layers.extend(layers)

if sft_layers and rl_layers:
    stat, p = mannwhitneyu(sft_layers, rl_layers, alternative="two-sided")
    record(
        "SFT vs RL (pooled): layer distribution",
        f"SFT_mean={np.mean(sft_layers):.2f}  RL_mean={np.mean(rl_layers):.2f}",
        p=p, effect=np.mean(sft_layers)-np.mean(rl_layers)
    )

# 18. Does model anchor count correlate with sabotage rate?
anchor_counts  = []
sabotage_rates = []
model_names_for_corr = []
for model_key, data in models.items():
    anchors = data["anchors"]
    if len(anchors) < 3:
        continue
    sab_rate = sum(1 for a in anchors if a.get("drop_pubmedqa",0) < 0) / len(anchors)
    anchor_counts.append(len(anchors))
    sabotage_rates.append(sab_rate)
    model_names_for_corr.append(model_key)

if len(anchor_counts) >= 3:
    r, p = spearmanr(anchor_counts, sabotage_rates)
    record(
        "CROSS-MODEL: Anchor count vs sabotage rate",
        f"r={r:.3f}  models: " +
        ", ".join(f"{m}({n},{s:.0%})"
                  for m, n, s in zip(model_names_for_corr,
                                      anchor_counts, sabotage_rates)),
        p=p, effect=r
    )

# 19. Does mean EMS differ significantly across all models?
model_ems_groups = []
model_ems_labels = []
for model_key, data in models.items():
    vals = [abs(a.get("drop_medqa", 0)) for a in data["anchors"]]
    if len(vals) >= 3:
        model_ems_groups.append(vals)
        model_ems_labels.append(model_key)

if len(model_ems_groups) >= 3:
    stat, p = safe_kruskal(model_ems_groups)
    if stat is None:
        record(
            "your title here",
            "Skipped (degenerate data)",
            p=None
        )
    else:
        record(
            "KRUSKAL-WALLIS: |EMS| magnitude differs across models?",
            f"H={stat:.3f}  means: " +
            ", ".join(f"{l}={np.mean(v):.5f}"
                      for l, v in zip(model_ems_labels, model_ems_groups)),
            p=p, effect=stat
        )

# =============================================================================
# BLOCK 7: CLUSTER ANALYSIS
# =============================================================================
banner("BLOCK 7: CLUSTER ANALYSIS")

# 20. Do cluster assignments predict layer position?
for model_key, data in models.items():
    anchors = [a for a in data["anchors"] if a.get("cluster", -1) >= 0]
    if len(anchors) < 5:
        continue
    clusters = list(set(a["cluster"] for a in anchors))
    if len(clusters) < 2:
        continue
    cluster_groups = [[a["layer"] for a in anchors if a["cluster"] == c]
                      for c in clusters if sum(1 for a in anchors if a["cluster"]==c) >= 2]
    if len(cluster_groups) >= 2:
        # skip if any group has zero variance
        if any(len(set(g)) <= 1 for g in cluster_groups):
            record(
                f"[{model_key}] Cluster → layer position?",
                "Skipped (no variance in one or more groups)",
                p=None
            )
        else:
            stat, p = safe_kruskal(cluster_groups)
            if stat is None:
                record(
                    "your title here",
                    "Skipped (degenerate data)",
                    p=None
                )
            else:
                means = [np.mean(g) for g in cluster_groups]
                record(
                    f"[{model_key}] Cluster → layer position?",
                    f"cluster_layer_means={[f'{m:.1f}' for m in means]}",
                    p=p, effect=stat
                )

# 21. Do clusters differ in |EMS|?
for model_key, data in models.items():
    anchors = [a for a in data["anchors"] if a.get("cluster", -1) >= 0]
    if len(anchors) < 5:
        continue
    clusters = list(set(a["cluster"] for a in anchors))
    cluster_groups = [[abs(a.get("drop_medqa",0)) for a in anchors
                       if a["cluster"] == c]
                      for c in clusters
                      if sum(1 for a in anchors if a["cluster"]==c) >= 2]
    if len(cluster_groups) >= 2:
        stat, p = safe_kruskal(cluster_groups)
        if stat is None:
            record(
                f"[{model_key}] Cluster → |EMS| magnitude?",
                "Skipped (degenerate data)",
                p=None
            )
        else:
            record(
                f"[{model_key}] Cluster → |EMS| magnitude?",
                f"cluster_ems_means={[f'{np.mean(g):.5f}' for g in cluster_groups]}",
                p=p, effect=stat
            )

# =============================================================================
# BLOCK 8: INFORMATION-THEORETIC
# =============================================================================
banner("BLOCK 8: INFORMATION-THEORETIC PATTERNS")

# 22. Shannon entropy of layer distribution per model
for model_key, data in models.items():
    layers = [a["layer"] for a in data["anchors"]]
    if len(layers) < 3:
        continue
    # Histogram across 10 bins
    hist, _ = np.histogram(layers, bins=10)
    hist = hist + 1e-10  # avoid log(0)
    H = entropy(hist / hist.sum())
    H_max = np.log(10)
    record(
        f"[{model_key}] Layer distribution entropy",
        f"H={H:.3f}  H_max={H_max:.3f}  normalized={H/H_max:.3f}  "
        f"(1=uniform, 0=concentrated)",
        p=None
    )

# 23. Mutual information: does knowing it's a sabotage neuron tell you the layer?
for model_key, data in models.items():
    anchors = data["anchors"]
    if len(anchors) < 5:
        continue
    layers   = np.array([a["layer"] for a in anchors])
    is_sab   = np.array([1 if a.get("drop_pubmedqa",0) < 0 else 0
                          for a in anchors])
    if is_sab.sum() < 2 or (1-is_sab).sum() < 2:
        continue
    # Point-biserial correlation as proxy for MI
    r, p = stats.pointbiserialr(is_sab, layers)
    record(
        f"[{model_key}] Sabotage label ↔ layer (point-biserial r)",
        f"r={r:.3f}  (positive = sabotage neurons deeper)",
        p=p, effect=r
    )

# =============================================================================
# BLOCK 9: NOVEL INTERACTION TERMS
# =============================================================================
banner("BLOCK 9: NOVEL INTERACTION TERMS")

# 24. EMS asymmetry: |MedQA_EMS| vs |PubMedQA_EMS| — which dominates?
for model_key, data in models.items():
    anchors = data["anchors"]
    if not anchors:
        continue
    mcq_dom = sum(1 for a in anchors
                  if abs(a.get("drop_medqa",0)) > abs(a.get("drop_pubmedqa",0)))
    bin_dom = len(anchors) - mcq_dom
    # Binomial test: is it significantly MCQ-dominant?
    
    try:
        from scipy.stats import binomtest
        p = binomtest(mcq_dom, len(anchors), 0.5).pvalue
    except ImportError:
        from scipy.stats import binom_test
        p = binom_test(mcq_dom, len(anchors), 0.5)

# 25. Is there a "bimodal" column distribution? (anchors clustered in specific neuron ranges)
for model_key, data in models.items():
    cols = [a["column"] for a in data["anchors"]]
    if len(cols) < 10:
        continue
    # Hartigan's dip test approximation: compare to normal
    stat, p = stats.normaltest(cols)
    record(
        f"[{model_key}] Column distribution non-normal (bimodal/clustered)?",
        f"D={stat:.3f}  col_mean={np.mean(cols):.0f}  col_std={np.std(cols):.0f}",
        p=p, effect=stat
    )

# 26. Cross-model: do sabotage neurons across models share column ranges?
all_sab_cols  = [a["column"] for a in all_anchors if a["is_sabotage"]]
all_norm_cols = [a["column"] for a in all_anchors if not a["is_sabotage"]]
if len(all_sab_cols) >= 5 and len(all_norm_cols) >= 5:
    stat, p = mannwhitneyu(all_sab_cols, all_norm_cols, alternative="two-sided")
    record(
        "GLOBAL: Sabotage neurons in different column ranges?",
        f"sab_col_mean={np.mean(all_sab_cols):.0f}  "
        f"norm_col_mean={np.mean(all_norm_cols):.0f}",
        p=p, effect=np.mean(all_sab_cols)-np.mean(all_norm_cols)
    )

# 27. Does depth shift (sab_layer - norm_layer) predict cross-dataset d?
if len(model_stats) >= 3:
    depth_shifts  = [m[1] for m in model_stats]
    # Use sabotage rate as proxy for "cross-dataset interference"
    sab_rates     = [m[0] for m in model_stats]
    r, p = spearmanr(depth_shifts, sab_rates)
    record(
        "CROSS-MODEL: Depth shift predicts sabotage rate?",
        f"r={r:.3f}  pairs={list(zip([m[2] for m in model_stats], depth_shifts, sab_rates))}",
        p=p, effect=r
    )

# 28. Within-model: does PubMedQA EMS predict MedMCQA EMS sign?
for model_key, data in models.items():
    anchors = data["anchors"]
    if len(anchors) < 5:
        continue
    pmqa_pos = [a for a in anchors if a.get("drop_pubmedqa",0) > 0]
    pmqa_neg = [a for a in anchors if a.get("drop_pubmedqa",0) < 0]
    if len(pmqa_pos) < 2 or len(pmqa_neg) < 2:
        continue
    mmcqa_if_pmqa_pos = [a.get("drop_medmcqa",0) for a in pmqa_pos]
    mmcqa_if_pmqa_neg = [a.get("drop_medmcqa",0) for a in pmqa_neg]
    stat, p = mannwhitneyu(mmcqa_if_pmqa_pos, mmcqa_if_pmqa_neg,
                            alternative="two-sided")
    record(
        f"[{model_key}] PubMedQA sign predicts MedMCQA EMS?",
        f"if_pmqa_pos→mmcqa={np.mean(mmcqa_if_pmqa_pos):+.5f}  "
        f"if_pmqa_neg→mmcqa={np.mean(mmcqa_if_pmqa_neg):+.5f}",
        p=p, effect=np.mean(mmcqa_if_pmqa_pos)-np.mean(mmcqa_if_pmqa_neg)
    )

# =============================================================================
# BLOCK 10: EMERGENT PATTERNS
# =============================================================================
banner("BLOCK 10: EMERGENT PATTERNS — THE UNEXPECTED")

# 29. Are there "universal" neurons — same column appears across models?
from collections import Counter
col_layer_pairs = Counter()
for a in all_anchors:
    col_layer_pairs[(a["layer"], a["column"])] += 1

shared = [(k, v) for k, v in col_layer_pairs.items() if v > 1]
record(
    "Universal neurons: same (layer,col) across multiple models?",
    f"Shared pairs={len(shared)}  "
    f"top_shared={sorted(shared, key=lambda x:-x[1])[:5]}",
    p=None
)

# 30. Do sabotage neurons have lower column index than helpers?
sab_cols  = [a["column"] for a in all_anchors if a["is_sabotage"]]
help_cols = [a["column"] for a in all_anchors if a["is_helper"]]
if sab_cols and help_cols:
    stat, p = mannwhitneyu(sab_cols, help_cols, alternative="two-sided")
    record(
        "GLOBAL: Sabotage vs helper column index distribution",
        f"sab_col_mean={np.mean(sab_cols):.0f}  help_col_mean={np.mean(help_cols):.0f}",
        p=p, effect=np.mean(sab_cols)-np.mean(help_cols)
    )

# 31. Does layer × regime interaction predict sabotage?
# Build a 2-way contingency: early/late × SFT/RL → sabotage yes/no
for threshold_name, layer_threshold in [("median", None)]:
    all_a_regime = []
    for a in all_anchors:
        mk = a["model"].lower()
        if "rl" in mk and "base" not in mk:
            regime = "RL"
        elif "med42" in mk or "medgemma" in mk:
            regime = "SFT"
        else:
            continue
        all_a_regime.append(a)

    if len(all_a_regime) < 10:
        continue
    median_layer = np.median([a["layer"] for a in all_a_regime])
    ct = np.zeros((2, 2))  # [early/late] × [sab/not_sab]
    for a in all_a_regime:
        late = 1 if a["layer"] >= median_layer else 0
        sab  = 1 if a["is_sabotage"] else 0
        ct[late, sab] += 1

    chi2, p, dof, expected = chi2_contingency(ct)
    record(
        "CHI2: Layer(early/late) × Sabotage — interaction with SFT/RL?",
        f"contingency=\n{ct}\nchi2={chi2:.3f}",
        p=p, effect=chi2
    )

# 32. Is there a "quiet zone" (layer range with NO anchors)?
for model_key, data in models.items():
    layers = sorted(set(a["layer"] for a in data["anchors"]))
    if len(layers) < 3:
        continue
    # Find max gap between consecutive anchor layers
    gaps = [(layers[i+1]-layers[i], layers[i], layers[i+1])
            for i in range(len(layers)-1)]
    if gaps:
        max_gap, gap_start, gap_end = max(gaps, key=lambda x: x[0])
        record(
            f"[{model_key}] Largest anchor-free zone",
            f"gap={max_gap} layers  between L{gap_start} and L{gap_end}  "
            f"anchor_layers={layers}",
            p=None
        )

# =============================================================================
# FINAL SUMMARY
# =============================================================================
banner("DISCOVERY SUMMARY")

significant = [d for d in discoveries if d["significant"]]
print(f"\nTotal comparisons run: {comparison_count}")
print(f"Significant findings (p<0.05): {len(significant)}")
print(f"Highly significant (p<0.001): "
      f"{sum(1 for d in discoveries if d['p'] and d['p'] < 0.001)}")

print(f"\n{BOLD}{'='*65}")
print("TOP SIGNIFICANT FINDINGS (ranked by p-value)")
print(f"{'='*65}{RESET}")

sig_sorted = sorted(
    [d for d in discoveries if d["p"] is not None and d["p"] < 0.05],
    key=lambda x: x["p"]
)

for i, d in enumerate(sig_sorted[:20], 1):
    stars = "***" if d["p"] < 0.001 else "** " if d["p"] < 0.01 else "*  "
    effect_str = f"  effect={d['effect']:.4f}" if d["effect"] else ""
    print(f"\n{i:>3}. {RED if d['p']<0.001 else YELLOW}{stars}{RESET} "
          f"p={d['p']:.2e}{effect_str}")
    print(f"     {BOLD}{d['title']}{RESET}")
    print(f"     {d['finding']}")

print(f"\n\n{BOLD}POTENTIAL NEW DISCOVERIES TO INVESTIGATE:{RESET}")
print("  1. Any finding where effect direction is opposite to expectation")
print("  2. Any model that breaks the SFT>RL depth-shift pattern")
print("  3. Column clustering — do sabotage neurons share feature families?")
print("  4. Universal neurons appearing in 2+ models")
print("  5. Quiet zones — layer ranges consistently avoided by anchors")
print()

# Save all results
with open("bruteforce_discovery_results.json", "w") as f:
    json.dump([{k: v for k, v in d.items() if k != "p" or v is None or not np.isnan(v)}
               for d in discoveries], f, indent=2, default=str)
print("✅ Full results saved to bruteforce_discovery_results.json")