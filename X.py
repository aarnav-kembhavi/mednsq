import numpy as np
from scipy import stats

# 🔴 HARD-CODED VALUES

ANCHOR_STATS = {
    "MedQA":    {"mean": 0.0837, "std": 0.0140},
    "MedMCQA":  {"mean": 0.0415, "std": 0.0072},
    "PubMedQA": {"mean": 0.0105, "std": 0.0012},
}

RANDOM_STATS = {
    "MedQA":    {"mean": 0.0091, "std": 0.0012},
    "MedMCQA":  {"mean": 0.0051, "std": 0.0086},
    "PubMedQA": {"mean": 0.0047, "std": 0.0003},
}

N = 200  # sample size per group (keep consistent)

def compute_stats(a_mean, a_std, r_mean, r_std, n):
    # pooled std
    pooled_std = np.sqrt((a_std**2 + r_std**2) / 2)

    # Cohen's d
    d = (a_mean - r_mean) / pooled_std if pooled_std > 0 else 0.0

    # Welch t-test approximation using stats
    # simulate distributions (since we only have summary stats)
    anchor_samples = np.random.normal(a_mean, a_std, n)
    random_samples = np.random.normal(r_mean, r_std, n)

    _, p = stats.ttest_ind(anchor_samples, random_samples, equal_var=False)

    return d, p


print("\nDataset     Anchor (mean±std)     Random (mean±std)     Cohen's d     p-value")

for dataset in ANCHOR_STATS:
    a = ANCHOR_STATS[dataset]
    r = RANDOM_STATS[dataset]

    d, p = compute_stats(a["mean"], a["std"], r["mean"], r["std"], N)

    print(f"{dataset:10s}  "
          f"{a['mean']:+.4f}±{a['std']:.4f}   "
          f"{r['mean']:+.4f}±{r['std']:.4f}   "
          f"{d:+.4f}     {p:.3e}")