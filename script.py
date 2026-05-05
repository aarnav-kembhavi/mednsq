import json
import numpy as np
from scipy import stats


def load_json(path):
    with open(path) as f:
        return json.load(f)


def get_anchor_array(data, key):
    return np.array([a[key] for a in data["anchors"] if key in a], dtype=float)


# 🔴 Hardcoded FINAL random stats (mean ± std)
RANDOM_STATS = {
    "MedQA": {"mean": 0.0091, "std": 0.0135},      # adjust std if needed
    "MedMCQA": {"mean": 0.0051, "std": 0.0068},
    "PubMedQA": {"mean": 0.0047, "std": 0.0013},
}


def compute_stats(anchor, r_mean, r_std):
    mean = anchor.mean()
    std = anchor.std(ddof=1)

    pooled_std = np.sqrt((std**2 + r_std**2) / 2)
    cohen_d = (mean - r_mean) / pooled_std if pooled_std > 0 else 0.0

    # approximate p-value using normal assumption
    t_stat = (mean - r_mean) / (pooled_std / np.sqrt(len(anchor)))
    p_val = 2 * (1 - stats.norm.cdf(abs(t_stat)))

    return {
        "mean": mean,
        "std": std,
        "random_mean": r_mean,
        "random_std": r_std,
        "cohen_d": cohen_d,
        "p_value": p_val,
    }


def print_row(name, s):
    print(
        f"{name:10s}  "
        f"{s['mean']:+.4f} ± {s['std']:.4f}   "
        f"{s['random_mean']:+.4f} ± {s['random_std']:.4f}   "
        f"{s['cohen_d']:+.4f}   "
        f"{s['p_value']:.3g}"
    )


def main(json_path):
    data = load_json(json_path)

    datasets = {
        "MedQA": "drop_medqa",
        "MedMCQA": "drop_medmcqa",
        "PubMedQA": "drop_pubmedqa",
    }

    print("\nDataset     Anchor Drop           Random Drop           Cohen's d   p-value")

    for name, drop_key in datasets.items():

        if drop_key not in data["anchors"][0]:
            print(f"{name:10s}  ERROR: missing {drop_key}")
            continue

        anchor = get_anchor_array(data, drop_key)

        if len(anchor) == 0:
            print(f"{name:10s}  ERROR: empty anchor array")
            continue

        r_mean = RANDOM_STATS[name]["mean"]
        r_std = RANDOM_STATS[name]["std"]

        stats_dict = compute_stats(anchor, r_mean, r_std)
        print_row(name, stats_dict)


if __name__ == "__main__":
    import sys
    main(sys.argv[1])