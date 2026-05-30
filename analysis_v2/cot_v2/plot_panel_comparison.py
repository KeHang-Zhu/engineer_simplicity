"""
Generate the three-panel comparison figure:
v1 standard / v2 strict / v2 CoT, side-by-side truthful-rate marginals
for the c-axis, plus a small audit-bar figure for dominance invocation.

Outputs:
  writeup_v2/reports/digital_mice_v2/fig9_panel_comparison.pdf
"""

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
MARGINALS_IN = os.path.join(REPO_ROOT, "analysis_v2/cot_v2/panel_comparison_axis_marginals.csv")
AUDIT_IN = os.path.join(REPO_ROOT, "analysis_v2/cot_v2/panel_comparison_audit.csv")
OUT_DIR = os.path.join(REPO_ROOT, "writeup_v2/reports/digital_mice_v2")


PANEL_ORDER = [
    ("v1_standard", "v1 standard\n(instruction only)"),
    ("strict_v2", "v2 strict\n(no example)"),
    ("cot_v2", "v2 CoT\n(worked example)"),
]
COLORS = {"v1_standard": "#999999", "strict_v2": "#c08040", "cot_v2": "#3b6db5"}


def fig9():
    mg = pd.read_csv(MARGINALS_IN)
    audit = pd.read_csv(AUDIT_IN)
    fig, axes = plt.subplots(1, 4, figsize=(14, 3.6))

    # Three axes (c, f, b), each as a grouped bar: levels 0/1/2, panels side by side
    for ax, axis_name, title in zip(
        axes[:3],
        ["contingent", "forward", "beliefs"],
        ["Contingent reasoning", "Forward planning", "Belief reasoning"],
    ):
        sub = mg[(mg["axis"] == axis_name) & (mg["metric"] == "truthful_rate")]
        levels = [0, 1, 2]
        w = 0.27
        x = np.arange(len(levels))
        for i, (panel_id, panel_label) in enumerate(PANEL_ORDER):
            row = sub[sub["panel"] == panel_id]
            if row.empty:
                continue
            vals = [row[f"level{l}"].values[0] for l in levels]
            ax.bar(x + (i - 1) * w, vals, w, color=COLORS[panel_id],
                   edgecolor="black", label=panel_label if axis_name == "contingent" else None)
        ax.set_xticks(x); ax.set_xticklabels(["L0", "L1", "L2"])
        ax.set_ylim(0, 1.05); ax.set_title(title)
        if axis_name == "contingent":
            ax.set_ylabel("truthful-bid rate")
    axes[0].legend(loc="upper left", fontsize=8)

    # Right: dominance invocation by c-level (CoT vs strict; v1 numbers reported in caption)
    cot = audit[audit["panel"] == "cot_v2"].sort_values("c_level")
    strict = audit[audit["panel"] == "strict_v2"].sort_values("c_level")
    x = np.array([0, 1, 2])
    w = 0.35
    ax = axes[3]
    ax.bar(x - w/2, strict["dominance_invocation_rate"].values, w,
           color=COLORS["strict_v2"], edgecolor="black", label="strict")
    ax.bar(x + w/2, cot["dominance_invocation_rate"].values, w,
           color=COLORS["cot_v2"], edgecolor="black", label="CoT")
    # Annotate v1 numbers
    v1_vals = [0.41, 0.77, 0.66]
    ax.scatter(x, v1_vals, color=COLORS["v1_standard"], edgecolor="black",
               s=70, zorder=3, label="v1 (point)")
    ax.set_xticks(x); ax.set_xticklabels(["c=0", "c=1", "c=2"])
    ax.set_ylim(0, 1.0)
    ax.set_title("Dominance-invocation rate")
    ax.legend(loc="upper left", fontsize=8)

    plt.tight_layout()
    out = os.path.join(OUT_DIR, "fig9_panel_comparison.pdf")
    plt.savefig(out)
    plt.close()
    print(f"wrote {out}")


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    fig9()


if __name__ == "__main__":
    main()
