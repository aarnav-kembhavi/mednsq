import json
import numpy as np
from scipy import stats


# HARDCODED (MedGemma SFT)
RANDOM_BASELINES = {
    "medqa": (-0.0012, 0.0048),
    "medmcqa": (0.00007, 0.0027),
    "pubmedqa": (-0.00022, 0.0020),
}


def load_json(path):
    with open(path) as f:
        return json.load(f)


def get_anchor_array(data, key):
    return np.array([a[key] for a in data["anchors"] if key in a], dtype=float)


def compute_stats(anchor, dataset_name):
    mean = anchor.mean()
    std = anchor.std(ddof=1)

    r_mean, r_std = RANDOM_BASELINES[dataset_name]

    # analytical pooled std (no sampling)
    pooled_std = np.sqrt((std**2 + r_std**2) / 2)

    cohen_d = (mean - r_mean) / pooled_std

    # synthetic normal only for p-value
    random = np.random.normal(r_mean, r_std, size=len(anchor))
    t_stat, p_val = stats.ttest_ind(anchor, random, equal_var=False)

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
    np.random.seed(42)
    data = load_json(json_path)

    datasets = {
        "MedQA": ("drop_medqa", "medqa"),
        "MedMCQA": ("drop_medmcqa", "medmcqa"),
        "PubMedQA": ("drop_pubmedqa", "pubmedqa"),
    }

    print("\nDataset     Anchor Drop           Random Drop           Cohen's d   p-value")

    for name, (key, dname) in datasets.items():
        arr = get_anchor_array(data, key)

        if len(arr) == 0:
            print(f"{name:10s}  ERROR: missing {key}")
            continue

        stats_dict = compute_stats(arr, dname)
        print_row(name, stats_dict)


if __name__ == "__main__":
    import sys
    main(sys.argv[1])