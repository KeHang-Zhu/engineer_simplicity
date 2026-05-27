"""
Generate the three figures for the Digital Mice empirical report.

No LLM calls. Reads existing construct-validation and calibration artifacts and
writes fig1.pdf, fig2.pdf, fig3.pdf next to this script.

Run from repo root:
    ./venv/bin/python writeup_v2/reports/digital_mice/make_figures.py
"""

import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
HERE = os.path.dirname(os.path.abspath(__file__))

# Reuse the calibration script's supplemental-mice loader for raw bid points.
sys.path.insert(0, os.path.join(REPO_ROOT, "analysis_v2", "calibration"))
from spsb_calibration_scores import load_supplemental_results  # noqa: E402

AXIS = os.path.join(REPO_ROOT, "analysis_v2/construct_validation/spsb_axis_diagnostics.csv")
GRID = os.path.join(REPO_ROOT, "data_v2/results/spsb_basis_grid_summary.csv")
COVERAGE = os.path.join(REPO_ROOT, "analysis_v2/calibration/coverage_scores.csv")
INTENSITY = os.path.join(REPO_ROOT, "analysis_v2/calibration/intensity_scores.csv")
PORTABILITY = os.path.join(
    REPO_ROOT,
    "analysis_v2/evolution/model_portability/"
    "run_20260526_231759_p3g1s20260529_gpt54mini_vs_gpt54.csv",
)

# Print-safe, restrained style.
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 11,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi": 150,
    "savefig.bbox": "tight",
})

INK = "#1f2a44"
ACCENT = "#b3402a"
GREY = "#8a8f9a"
GREEN = "#2e6e4e"


def fig1_construct_marginals():
    """1x3: truthful rate at axis levels 0/1/2 for each cognitive axis."""
    ax_diag = pd.read_csv(AXIS)
    axes_order = [
        ("contingent", "Contingent reasoning"),
        ("forward", "Forward planning"),
        ("beliefs", "Belief reasoning"),
    ]
    level_labels = {
        "contingent": ["off", "enumerate", "worst-case"],
        "forward": ["myopic", "one-step", "tree"],
        "beliefs": ["none", "first-order", "second-order"],
    }

    fig, axs = plt.subplots(1, 3, figsize=(9.6, 3.2), sharey=True)
    for k, (axis_name, pretty) in enumerate(axes_order):
        row = ax_diag[(ax_diag["axis"] == axis_name)
                      & (ax_diag["metric"] == "truthful_rate")].iloc[0]
        ys = [row["level0_mean"], row["level1_mean"], row["level2_mean"]]
        delta = row["level2_minus_level0"]
        xs = [0, 1, 2]
        ax = axs[k]
        ax.plot(xs, ys, "-o", color=INK, lw=2, markersize=7)
        ax.set_xticks(xs)
        ax.set_xticklabels(level_labels[axis_name], rotation=15, ha="right")
        ax.set_title(pretty)
        ax.set_ylim(0.30, 1.0)
        ax.grid(axis="y", color=GREY, alpha=0.25, lw=0.6)
        ax.annotate(
            rf"$\Delta = {delta:+.2f}$",
            xy=(2, ys[2]), xytext=(0.05, 0.86), textcoords="axes fraction",
            color=ACCENT, fontsize=11, fontweight="bold",
        )
    axs[0].set_ylabel("Truthful-bid rate")
    fig.suptitle("Repairing each cognitive axis raises truthful bidding (level 0 $\\rightarrow$ 2)",
                 y=1.04, fontsize=12)
    out = os.path.join(HERE, "fig1.pdf")
    fig.savefig(out)
    plt.close(fig)
    return out


def fig2_behavioral_signatures():
    """1x3 bid-vs-value: rational strain, contingent-off strain, supplemental error mice."""
    grid = pd.read_csv(GRID)
    rational = grid[grid["basis_id"] == "c1_f0_b0"]   # most truthful basis strain
    deficient = grid[grid["basis_id"] == "c0_f0_b0"]   # least truthful (underbids)
    supp = load_supplemental_results()

    fig, axs = plt.subplots(1, 3, figsize=(10.0, 3.4), sharex=True, sharey=True)
    vmax = 50

    def scatter(ax, df, color, label):
        ax.scatter(df["value"], df["bid"], s=22, color=color, alpha=0.75,
                   edgecolor="white", linewidth=0.4, label=label)

    # Panel A: rational
    axs[0].plot([0, vmax], [0, vmax], "--", color=GREY, lw=1)
    scatter(axs[0], rational, GREEN, "c1_f0_b0")
    axs[0].set_title("Rational mouse\n(contingent=enumerate)")

    # Panel B: contingent-off (underbids)
    axs[1].plot([0, vmax], [0, vmax], "--", color=GREY, lw=1)
    scatter(axs[1], deficient, INK, "c0_f0_b0")
    axs[1].set_title("Deficit mouse\n(all axes off)")

    # Panel C: supplemental error mice
    axs[2].plot([0, vmax], [0, vmax], "--", color=GREY, lw=1)
    over = supp[supp["target_class"] == "spsb_overbid"]
    nearzero = supp[supp["target_class"] == "spsb_near_zero_bid"]
    scatter(axs[2], over, ACCENT, "overbidder")
    scatter(axs[2], nearzero, "#c79a00", "near-zero")
    axs[2].set_title("Supplemental error mice\n(overbid + near-zero)")
    axs[2].legend(loc="upper left", fontsize=8, frameon=False)

    for ax in axs:
        ax.set_xlim(0, vmax)
        ax.set_ylim(0, vmax + 8)
        ax.set_xlabel("Private value")
        ax.grid(color=GREY, alpha=0.2, lw=0.6)
    axs[0].set_ylabel("Submitted bid")
    fig.suptitle("Separable behavioral signatures relative to the 45$^\\circ$ truthful line",
                 y=1.05, fontsize=12)
    out = os.path.join(HERE, "fig2.pdf")
    fig.savefig(out)
    plt.close(fig)
    return out


def fig3_calibration_coverage():
    """Horizontal bars: attained max rate vs required threshold, per mistake class."""
    cov = pd.read_csv(COVERAGE)
    inten = pd.read_csv(INTENSITY)[["class_id", "intensity_band"]]
    cov = cov.merge(inten, on="class_id", how="left")

    pretty = {
        "spsb_truthful": "Truthful",
        "spsb_underbid": "Moderate underbid",
        "spsb_overbid": "Overbid",
        "spsb_extreme_overbid": "Extreme overbid",
        "spsb_near_zero_bid": "Near-zero bid",
    }
    cov["label"] = cov["class_id"].map(pretty)
    cov = cov.iloc[::-1].reset_index(drop=True)

    band_color = {"very_high": GREEN, "plausible": INK, "too_low": "#c79a00", "missing": ACCENT}

    fig, ax = plt.subplots(figsize=(8.2, 3.4))
    y = np.arange(len(cov))
    ax.barh(y, cov["max_rate"], color=[band_color.get(b, GREY) for b in cov["intensity_band"]],
            height=0.55, alpha=0.9, zorder=3)
    # threshold markers
    ax.scatter(cov["threshold"], y, marker="|", s=420, color=ACCENT, lw=2.0, zorder=4,
               label="required threshold")
    for yi, (_, r) in zip(y, cov.iterrows()):
        ax.text(r["max_rate"] + 0.015, yi, f"{r['max_rate']:.2f}", va="center",
                fontsize=9, color=INK)
    ax.set_yticks(y)
    ax.set_yticklabels(cov["label"])
    ax.set_xlim(0, 1.42)
    ax.set_xticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_xlabel("Attained max rate across the panel")
    ax.grid(axis="x", color=GREY, alpha=0.25, lw=0.6)
    ax.set_title("All five human SPSB mistake classes are covered (rate $\\geq$ threshold)")
    # Legend in the reserved white margin to the right of every bar.
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    handles = [
        Line2D([0], [0], marker="|", color=ACCENT, lw=0, markersize=12,
               markeredgewidth=2, label="required threshold"),
        Patch(color=GREEN, label="intensity: very high"),
        Patch(color=INK, label="intensity: plausible"),
    ]
    ax.legend(handles=handles, loc="center left", bbox_to_anchor=(0.74, 0.5),
              frameon=False, fontsize=8.5)
    out = os.path.join(HERE, "fig3.pdf")
    fig.savefig(out)
    plt.close(fig)
    return out


def fig4_model_portability():
    """Slope chart: robust fitness and intended-action rate, search model -> held-out model."""
    d = pd.read_csv(PORTABILITY)
    short = {
        "mech_000_all_pay_refundable_deposit_no_reveal": "all-pay: deposit, no reveal",
        "mech_002_all_pay_refundable_deposit_post_reveal": "all-pay: deposit, post-reveal",
        "mech_001_all_pay_reserve_ranked_prebid_signal": "all-pay: reserve, pre-bid signal",
        "mech_000_second_price_refundable_deposit_binary_accept_shared_signal":
            "second-price: deposit, binary",
    }

    def label(m):
        for key, lab in short.items():
            if m.startswith(key):
                return lab
        return m[:28]

    d = d.copy()
    d["label"] = d["mechanism"].map(label)
    d["is_allpay"] = d["mechanism"].str.contains("all_pay")

    fig, axs = plt.subplots(1, 2, figsize=(10.0, 3.8))
    x = [0, 1]
    xlabels = ["gpt-5.4-mini\n(search model)", "gpt-5.4\n(held-out)"]

    # Panel A: robust fitness
    for _, r in d.iterrows():
        c = ACCENT if r["is_allpay"] else INK
        axs[0].plot(x, [r["robust_fitness_gpt54mini"], r["robust_fitness_gpt54"]],
                    "-o", color=c, lw=1.8, markersize=6, alpha=0.9)
        axs[0].annotate(r["label"], xy=(1.02, r["robust_fitness_gpt54"]),
                        fontsize=7.5, va="center", color=c)
    axs[0].axhline(0, color=GREY, lw=0.7, ls=":")
    axs[0].set_xticks(x); axs[0].set_xticklabels(xlabels)
    axs[0].set_xlim(-0.15, 2.05)
    axs[0].set_ylabel("Robust fitness on the panel")
    axs[0].set_title("Robust fitness does not fully port")
    axs[0].grid(axis="y", color=GREY, alpha=0.25, lw=0.6)

    # Panel B: intended-action rate (only the scoreable targets)
    scoreable = d[d["intended_action_rate_gpt54mini"] + d["intended_action_rate_gpt54"] > 0]
    for _, r in scoreable.iterrows():
        c = ACCENT if r["is_allpay"] else INK
        axs[1].plot(x, [r["intended_action_rate_gpt54mini"], r["intended_action_rate_gpt54"]],
                    "-o", color=c, lw=1.8, markersize=6, alpha=0.9)
        axs[1].annotate(r["label"], xy=(1.02, r["intended_action_rate_gpt54"]),
                        fontsize=7.5, va="center", color=c)
    axs[1].set_xticks(x); axs[1].set_xticklabels(xlabels)
    axs[1].set_xlim(-0.15, 2.05); axs[1].set_ylim(0, 1.0)
    axs[1].set_ylabel("Intended-action rate")
    axs[1].set_title("``Bid aggressively'' collapses on the held-out model")
    axs[1].grid(axis="y", color=GREY, alpha=0.25, lw=0.6)

    from matplotlib.lines import Line2D
    handles = [Line2D([0], [0], color=ACCENT, lw=2, marker="o", label="all-pay variant"),
               Line2D([0], [0], color=INK, lw=2, marker="o", label="second-price")]
    axs[0].legend(handles=handles, loc="upper right", frameon=False, fontsize=8)
    fig.suptitle("Cross-model portability of candidate mechanisms on the digital-mouse panel",
                 y=1.04, fontsize=12)
    out = os.path.join(HERE, "fig4.pdf")
    fig.savefig(out)
    plt.close(fig)
    return out


def main():
    for fn in (fig1_construct_marginals, fig2_behavioral_signatures, fig3_calibration_coverage,
               fig4_model_portability):
        path = fn()
        size = os.path.getsize(path)
        print(f"wrote {os.path.relpath(path, REPO_ROOT)}  ({size} bytes)")


if __name__ == "__main__":
    main()
