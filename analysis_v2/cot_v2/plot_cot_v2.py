"""
Plots for the v2 CoT run.

Generates four figures used by writeup_v2/reports/digital_mice_v2:
    fig1_cot.pdf  Axis marginals (truthful rate at levels 0/1/2 by axis)
    fig2_cot.pdf  Bid-vs-value scatter for 4 representative strains + outcome strains
    fig3_cot.pdf  Coverage scorecard against detection floors
    fig7_cot.pdf  Reasoning-chain audit (dominance-invocation rate by c-level)
"""

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
PANEL_IN = os.path.join(REPO_ROOT, "data_v2/results/spsb_basis_cot_v2_summary.csv")
OUTCOME_IN = os.path.join(REPO_ROOT, "data_v2/results/spsb_outcome_v2_summary.csv")
STRAIN_SUMMARY = os.path.join(REPO_ROOT, "analysis_v2/cot_v2/spsb_cot_strain_summary.csv")
AUDIT_SUMMARY = os.path.join(REPO_ROOT, "analysis_v2/cot_v2/spsb_cot_audit.csv")
COVERAGE_IN = os.path.join(REPO_ROOT, "analysis_v2/cot_v2/spsb_cot_coverage_scorecard.csv")
OUT_DIR = os.path.join(REPO_ROOT, "writeup_v2/reports/digital_mice_v2")


def fig1_axis_marginals():
    strain = pd.read_csv(STRAIN_SUMMARY)
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.6), sharey=True)
    titles = [
        ("c", "contingent", "Contingent reasoning"),
        ("f", "forward", "Forward planning"),
        ("b", "beliefs", "Belief reasoning"),
    ]
    for ax, (col, _, title) in zip(axes, titles):
        by_lvl = strain.groupby(col)["truthful_rate"].mean()
        levels = [0, 1, 2]
        vals = [by_lvl.get(l, np.nan) for l in levels]
        bars = ax.bar(levels, vals, color=["#d96666", "#d9b566", "#5fa84f"],
                      edgecolor="black", width=0.7)
        ax.set_xticks(levels)
        ax.set_xticklabels(["L0 (ablated)", "L1", "L2 (engaged)"])
        ax.set_ylim(0, 1.0)
        ax.set_title(title)
        delta = (vals[2] - vals[0]) if (vals[0] is not np.nan and vals[2] is not np.nan) else None
        if delta is not None:
            ax.text(
                0.5, 0.96, fr"$\Delta=+{delta:.2f}$" if delta >= 0 else fr"$\Delta={delta:.2f}$",
                transform=ax.transAxes, ha="center", va="top",
                fontsize=11, fontweight="bold",
            )
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2, v + 0.02,
                    f"{v:.2f}", ha="center", va="bottom", fontsize=9)
    axes[0].set_ylabel("Truthful-bid rate")
    plt.tight_layout()
    out = os.path.join(OUT_DIR, "fig1_cot.pdf")
    plt.savefig(out)
    plt.close()
    print(f"wrote {out}")


def fig2_bid_value_scatter():
    df = pd.read_csv(PANEL_IN)
    outcome = pd.read_csv(OUTCOME_IN)

    cells = [
        ("c2_f2_b2", "c2_f2_b2 (all-engaged)"),
        ("c0_f0_b0", "c0_f0_b0 (all-ablated)"),
    ]
    outcome_cells = [
        ("spsb_outcome_overbid", "outcome: overbid"),
        ("spsb_outcome_underbid", "outcome: underbid"),
    ]
    fig, axes = plt.subplots(1, 4, figsize=(13, 3.6), sharey=True, sharex=True)

    for ax, (bid, label) in zip(axes[:2], cells):
        sub = df[df["basis_id"] == bid]
        ax.scatter(sub["value"], sub["bid"], alpha=0.5, s=24, color="#3b6db5")
        x = np.linspace(0, 50, 100)
        ax.plot(x, x, "--", color="grey", linewidth=1)
        ax.set_xlim(0, 50); ax.set_ylim(0, 55)
        ax.set_title(label, fontsize=10)
        ax.set_xlabel("private value $v$")
    for ax, (oid, label) in zip(axes[2:], outcome_cells):
        sub = outcome[outcome["outcome_id"] == oid]
        ax.scatter(sub["value"], sub["bid"], alpha=0.5, s=24, color="#c75450")
        x = np.linspace(0, 50, 100)
        ax.plot(x, x, "--", color="grey", linewidth=1)
        ax.set_xlim(0, 50); ax.set_ylim(0, 55)
        ax.set_title(label, fontsize=10)
        ax.set_xlabel("private value $v$")
    axes[0].set_ylabel("submitted bid $b$")
    plt.tight_layout()
    out = os.path.join(OUT_DIR, "fig2_cot.pdf")
    plt.savefig(out)
    plt.close()
    print(f"wrote {out}")


def fig3_coverage():
    cov = pd.read_csv(COVERAGE_IN)
    fig, ax = plt.subplots(figsize=(9.5, 4.2))
    labels = [c.replace("_", " ") for c in cov["mistake_class"]]
    panel_rates = cov["max_rate_cot_panel"]
    floors = cov["floor"]
    outcome_rates = []
    for row in cov.itertuples(index=False):
        if isinstance(row.outcome_alt, str) and "(" in row.outcome_alt:
            r = float(row.outcome_alt.split("(")[1].rstrip(")"))
            outcome_rates.append(r)
        else:
            outcome_rates.append(np.nan)

    x = np.arange(len(labels))
    w = 0.35
    panel_bars = ax.bar(x - w/2, panel_rates, w, label="CoT panel max", color="#3b6db5", edgecolor="black")
    outcome_bars = ax.bar(x + w/2, outcome_rates, w, label="outcome-instruction strain", color="#c75450", edgecolor="black")
    for xi, fl in zip(x, floors):
        ax.hlines(fl, xi - w, xi + w, colors="red", linestyles="--", linewidth=1.2)
    ax.set_xticks(x); ax.set_xticklabels(labels, rotation=15, ha="right")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("max rate attained")
    ax.legend(loc="upper right", fontsize=9)
    ax.set_title("Coverage: CoT panel vs outcome-instruction strains (red dashes = detection floor)")
    plt.tight_layout()
    out = os.path.join(OUT_DIR, "fig3_cot.pdf")
    plt.savefig(out)
    plt.close()
    print(f"wrote {out}")


def fig7_reasoning_audit():
    audit = pd.read_csv(AUDIT_SUMMARY)
    panel = pd.read_csv(PANEL_IN)

    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))

    # LEFT: bar chart of dominance invocation by c-level (CoT vs v1)
    by_c = audit.groupby("c")["dominance_invocation_rate"].mean()
    v1_by_c = {0: 0.41, 1: 0.77, 2: 0.66}  # from v1 reported numbers
    levels = [0, 1, 2]
    cot_vals = [by_c.get(l, np.nan) for l in levels]
    v1_vals = [v1_by_c[l] for l in levels]
    w = 0.35
    x = np.array(levels)
    axes[0].bar(x - w/2, v1_vals, w, label="v1 standard prompt", color="#bbbbbb", edgecolor="black")
    axes[0].bar(x + w/2, cot_vals, w, label="CoT v2", color="#3b6db5", edgecolor="black")
    axes[0].set_xticks(levels); axes[0].set_xticklabels(["c=0 (off)", "c=1 (enum)", "c=2 (worst-case)"])
    axes[0].set_ylim(0, 1.0)
    axes[0].set_ylabel("dominance-argument invocation rate")
    axes[0].set_title("Reasoning audit by contingent level")
    axes[0].legend(loc="upper left", fontsize=9)
    for xi, vc in zip(x - w/2, v1_vals):
        axes[0].text(xi, vc + 0.02, f"{vc:.2f}", ha="center", fontsize=8)
    for xi, vc in zip(x + w/2, cot_vals):
        axes[0].text(xi, vc + 0.02, f"{vc:.2f}", ha="center", fontsize=8)

    # RIGHT: scatter of dominance invocation vs truthful rate per strain
    audit2 = audit.merge(
        panel.groupby("basis_id").agg(
            truthful=("bid", lambda s: ((panel.loc[s.index, "bid"] - panel.loc[s.index, "value"]).abs() <= 0.05).mean())
        ).reset_index(), on="basis_id"
    )
    colors = ["#d96666", "#d9b566", "#5fa84f"]
    for c, color in enumerate(colors):
        sub = audit2[audit2["c"] == c]
        axes[1].scatter(sub["dominance_invocation_rate"], sub["truthful"],
                        color=color, edgecolor="black", s=80, label=f"c={c}", alpha=0.9)
    axes[1].set_xlabel("dominance-invocation rate")
    axes[1].set_ylabel("truthful-bid rate")
    axes[1].set_xlim(-0.02, 1.05); axes[1].set_ylim(-0.02, 1.05)
    axes[1].legend(loc="lower right", fontsize=9)
    axes[1].set_title("Per strain: reasoning vs behavior")

    plt.tight_layout()
    out = os.path.join(OUT_DIR, "fig7_cot.pdf")
    plt.savefig(out)
    plt.close()
    print(f"wrote {out}")


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    fig1_axis_marginals()
    fig2_bid_value_scatter()
    fig3_coverage()
    fig7_reasoning_audit()


if __name__ == "__main__":
    main()
