import json
import numpy as np
from scipy import stats


def load_json(path):
    with open(path) as f:
        return json.load(f)


def get_array(data, key, cluster=None):
    if cluster is None:
        return np.array([a[key] for a in data["anchors"] if key in a], dtype=float)
    else:
        return np.array(
            [a[key] for a in data["anchors"] if key in a and a.get("cluster") == cluster],
            dtype=float
        )


def resolve_keys(data, prefix):
    sample = data["anchors"][0]

    # AFM-style
    if f"{prefix}_mean" in sample:
        return f"{prefix}_mean", f"{prefix}_random"

    # Med42-style
    if f"{prefix}" in sample:
        return f"{prefix}", None

    return None, None


def compute_stats(anchor, random):
    mean = anchor.mean()
    std = anchor.std(ddof=1)

    r_mean = random.mean()
    r_std = random.std(ddof=1)

    pooled_std = np.sqrt((std**2 + r_std**2) / 2)
    cohen_d = (mean - r_mean) / pooled_std

    _, p_val = stats.ttest_ind(anchor, random, equal_var=False)

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
        "MedQA": "drop_medqa",
        "MedMCQA": "drop_medmcqa",
        "PubMedQA": "drop_pubmedqa",
    }

    print("\nDataset     Anchor Drop           Random Drop           Cohen's d   p-value")

    for name, prefix in datasets.items():
        anchor_key, random_key = resolve_keys(data, prefix)

        if anchor_key is None:
            print(f"{name:10s}  ERROR: missing anchor key")
            continue

        anchor = get_array(data, anchor_key)

        if len(anchor) == 0:
            print(f"{name:10s}  ERROR: empty anchor array")
            continue

        # Case 1: real random exists
        if random_key is not None:
            random = get_array(data, random_key)

        # Case 2: fallback → cluster 0 (pseudo-random)
        else:
            random = get_array(data, anchor_key, cluster=0)

            # if clustering not present → last fallback (dataset baseline)
            if len(random) == 0 and "dataset_baselines" in data:
                base = data["dataset_baselines"][prefix.split("_")[1]]
                random = np.random.normal(base["mean"], base["std"], size=len(anchor))

        if len(random) == 0:
            print(f"{name:10s}  ERROR: no valid random baseline")
            continue

        stats_dict = compute_stats(anchor, random)
        print_row(name, stats_dict)


if __name__ == "__main__":
    import sys
    main(sys.argv[1])