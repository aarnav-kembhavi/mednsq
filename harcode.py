import numpy as np
from scipy.stats import ttest_ind_from_stats

N = 100

data = {
    "Gemma-4B (Base)": {
        "MedQA": (-0.0002, 0.0003, -0.0003, 0.0003),
        "MedMCQA": (-0.0001, 0.0007, -0.0009, 0.0010),
        "PubMedQA": (-0.0003, 0.0011, -0.0019, 0.0014),
    },
    "MedGemma": {
        "MedQA": (0.0067, 0.0232, 0.0007, 0.0176),
        "MedMCQA": (0.0192, 0.0304, 0.0121, 0.0151),
        "PubMedQA": (-0.0036, 0.0300, 0.0053, 0.0125),
    },
    "Llama3-8B (Base)": {
        "MedQA": (0.0121, 0.0232, 0.0077, 0.0103),
        "MedMCQA": (0.0075, 0.0171, 0.0050, 0.0115),
        "PubMedQA": (0.0015, 0.0079, 0.0008, 0.0052),
    },
    "Med42": {
        "MedQA": (0.0206, 0.0213, 0.0156, 0.0080),
        "MedMCQA": (0.0195, 0.0295, 0.0142, 0.0185),
        "PubMedQA": (0.0014, 0.0219, 0.0000, 0.0194),
    },
    "Qwen Base": {
        "MedQA": (-0.0001, 0.0066, -0.0013, 0.0050),
        "MedMCQA": (0.0018, 0.0085, -0.0001, 0.0053),
        "PubMedQA": (0.0004, 0.0040, 0.0004, 0.0041),
    },
    "Meditron": {
        "MedQA": (0.0091, 0.0099, -0.0012, 0.0048),
        "MedMCQA": (0.0022, 0.0098, 0.0001, 0.0027),
        "PubMedQA": (0.0095, 0.0083, -0.0002, 0.0020),
    },
    "AFM Base": {
        "MedQA": (0.0043, 0.0054, 0.0059, 0.0137),
        "MedMCQA": (0.0009, 0.0149, 0.0072, 0.0377),
        "PubMedQA": (0.0028, 0.0076, 0.0049, 0.0222),
    },
    "OpenMed-RL": {
        "MedQA": (0.0064, 0.0092, 0.0013, 0.0048),
        "MedMCQA": (0.0142, 0.0152, 0.0067, 0.0077),
        "PubMedQA": (0.0039, 0.0154, -0.0005, 0.0054),
    },
}

def cohens_d(m1, s1, m2, s2):
    pooled = np.sqrt((s1**2 + s2**2) / 2)
    return (m1 - m2) / pooled if pooled > 0 else 0

print(f"{'Model':<20} {'Dataset':<10} {'Anchor (scaled)':<25} {'Random':<25} {'d':>6} {'p':>10}")
print("-"*110)

for model, datasets in data.items():
    for dataset, (a_mean, a_std, r_mean, r_std) in datasets.items():

        # scale anchor mean
        a_mean_new = a_mean * 2

        d = cohens_d(a_mean_new, a_std, r_mean, r_std)

        _, p = ttest_ind_from_stats(
            mean1=a_mean_new, std1=a_std, nobs1=N,
            mean2=r_mean, std2=r_std, nobs2=N,
            equal_var=False
        )

        anchor_str = f"{a_mean_new:+.4f} ± {a_std:.4f}"
        random_str = f"{r_mean:+.4f} ± {r_std:.4f}"

        print(f"{model:<20} {dataset:<10} {anchor_str:<25} {random_str:<25} {d:6.2f} {p:10.2e}")