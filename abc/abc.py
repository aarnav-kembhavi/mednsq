import json
import glob
import os
import numpy as np
import matplotlib.pyplot as plt


def get_pubmedqa_key(sample):
    if 'drop_pubmedqa_mean' in sample:
        return 'drop_pubmedqa_mean'
    elif 'drop_pubmedqa' in sample:
        return 'drop_pubmedqa'
    else:
        return None


def analyze_sabotage_depth():
    print(f"{'Model':<20} | {'Total':<5} | {'Sabotage%':<10} | {'Sab. Layer':<10} | {'Norm. Layer':<10} | {'Shift'}")
    print("-" * 75)

    all_stats = []

    for f in glob.glob('*_cross_dataset_results.json'):
        with open(f) as fp:
            data = json.load(fp)

        if 'anchors' not in data or len(data['anchors']) == 0:
            continue

        model_name = os.path.basename(f).split('_cross')[0].upper()
        anchors = data['anchors']

        # 🔑 Detect correct key dynamically
        key = get_pubmedqa_key(anchors[0])
        if key is None:
            print(f"[SKIP] {model_name} → no valid pubmedqa key")
            continue

        # normalize (optional but clean)
        for a in anchors:
            a['drop'] = a.get(key, 0)

        sab_anchors = [a for a in anchors if a['drop'] < 0]
        norm_anchors = [a for a in anchors if a['drop'] >= 0]

        total = len(anchors)
        perc_sab = (len(sab_anchors) / total) * 100 if total > 0 else 0

        sab_layers = [a['layer'] for a in sab_anchors]
        norm_layers = [a['layer'] for a in norm_anchors]

        avg_sab = np.mean(sab_layers) if len(sab_layers) > 0 else np.nan
        avg_norm = np.mean(norm_layers) if len(norm_layers) > 0 else np.nan
        shift = avg_sab - avg_norm if not (np.isnan(avg_sab) or np.isnan(avg_norm)) else 0

        print(f"{model_name:<20} | {total:<5} | {perc_sab:>8.1f}% | {avg_sab:>10.1f} | {avg_norm:>10.1f} | {shift:>+5.1f}")

        if len(sab_layers) == 0 or len(norm_layers) == 0:
            continue

        all_stats.append({
            'model': model_name,
            'sab_layers': sab_layers,
            'norm_layers': norm_layers
        })

    # -------- Plot --------
    if all_stats:
        order = [
            "GEMMA4B",
            "MEDGEMMA",
            "LLAMA3_8B",
            "MED42",
            "QWEN25_8B",
            "MEDITRON",
            "OPENMED",
            "BASE_AFM"
        ]

        stats_dict = {s['model']: s for s in all_stats}
        ordered_stats = [stats_dict[m] for m in order if m in stats_dict]

        plt.style.use('seaborn-v0_8-white')
        fig, ax = plt.subplots(figsize=(12, 6))

        for i, stat in enumerate(ordered_stats):
            sab = stat['sab_layers']
            norm = stat['norm_layers']

            m_sab, m_norm = np.mean(sab), np.mean(norm)
            shift = m_sab - m_norm

            ax.boxplot(sab, positions=[i - 0.2], widths=0.15, patch_artist=True,
                       showfliers=False,
                       boxprops=dict(facecolor='#ffcdd2', color='#b71c1c'),
                       medianprops=dict(color='#b71c1c', lw=2))

            ax.boxplot(norm, positions=[i + 0.2], widths=0.15, patch_artist=True,
                       showfliers=False,
                       boxprops=dict(facecolor='#bbdefb', color='#0d47a1'),
                       medianprops=dict(color='#0d47a1', lw=2))

            ax.scatter(np.random.normal(i - 0.2, 0.03, len(sab)), sab,
                       color='#b71c1c', alpha=0.3, s=15)

            ax.scatter(np.random.normal(i + 0.2, 0.03, len(norm)), norm,
                       color='#0d47a1', alpha=0.3, s=15)

            if abs(shift) > 0.5:
                ax.annotate('',
                            xy=(i - 0.2, m_sab),
                            xytext=(i + 0.2, m_norm),
                            arrowprops=dict(arrowstyle="->", lw=1.5,
                                            connectionstyle="arc3,rad=-0.3"))

        ax.set_xticks(range(len(ordered_stats)))
        ax.set_xticklabels([s['model'] for s in ordered_stats], fontsize=10)
        ax.set_ylabel("Transformer Layer (Depth)")
        ax.set_title("Causal Migration: Sabotage vs Normal Neurons")

        for spine in ['top', 'right']:
            ax.spines[spine].set_visible(False)

        from matplotlib.lines import Line2D
        ax.legend([
            Line2D([0], [0], color='#b71c1c', lw=4),
            Line2D([0], [0], color='#0d47a1', lw=4)
        ], ['Sabotage', 'Normal'], frameon=False)

        plt.grid(axis='y', linestyle=':', alpha=0.4)
        plt.tight_layout()
        plt.savefig("sabotage_depth_analysis.png", dpi=500)

        print("\n[SUCCESS] Figure saved as sabotage_depth_analysis.png")


if __name__ == "__main__":
    analyze_sabotage_depth()