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
FPSB_FITNESS = os.path.join(
    REPO_ROOT,
    "config_v2/configs_auction/stress_test_first_price/fpsb_stress.fitness.json",
)
RESULTS_DB = os.path.join(REPO_ROOT, "data_v2/results.sqlite")
EA_RUN_ID = 9
SMAD_CSV = os.path.join(REPO_ROOT, "analysis_v2/evolution/ea_smad_ranking.csv")
AUDIT_CSV = os.path.join(REPO_ROOT, "analysis_v2/reasoning_audit/reasoning_chain_audit.csv")

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


def fig4_first_price():
    """First-price bid vs value: panel strains against the 45-degree value line and the BNE line."""
    import json
    d = json.load(open(FPSB_FITNESS))
    reasoning = {"c0_f0_b0", "c2_f0_b2", "c2_f2_b2"}
    pts = {"reasoning": [], "overgen": [], "panic": []}
    for r in d["per_run"]:
        p = r["persona"]
        cat = ("reasoning" if p in reasoning
               else "overgen" if "overgeneralizer" in p
               else "panic" if "payment_panic" in p else None)
        if cat is None:
            continue
        for v, b in zip(r["values"], r["bids"]):
            pts[cat].append((v, b))

    fig, ax = plt.subplots(figsize=(7.6, 4.6))
    vmax = 50
    xs = [0, vmax]
    ax.plot(xs, xs, "--", color=GREY, lw=1.2, label="truthful ($b=v$)")
    ax.plot(xs, [0, (2/3) * vmax], "-", color="#2e6e4e", lw=1.4,
            label="risk-neutral BNE ($b=\\frac{2}{3}v$)")

    def scat(cat, color, lab, marker="o"):
        if not pts[cat]:
            return
        vv = [p[0] for p in pts[cat]]
        bb = [p[1] for p in pts[cat]]
        ax.scatter(vv, bb, s=26, color=color, alpha=0.8, marker=marker,
                   edgecolor="white", linewidth=0.4, label=lab)

    scat("reasoning", INK, "reasoning mice (3 strains)")
    scat("overgen", ACCENT, "second-price overgeneralizer", marker="^")
    scat("panic", "#c79a00", "payment-panic mouse", marker="s")

    ax.set_xlim(0, vmax); ax.set_ylim(0, vmax + 10)
    ax.set_xlabel("Private value"); ax.set_ylabel("Submitted bid")
    ax.grid(color=GREY, alpha=0.2, lw=0.6)
    ax.legend(loc="upper left", frameon=False, fontsize=8.5)
    ax.set_title("First-price stress test: reasoning mice track the BNE;\n"
                 "the second-price-overgeneralizer mouse overbids above value")
    out = os.path.join(HERE, "fig4.pdf")
    fig.savefig(out)
    plt.close(fig)
    return out


def fig5_ea_dynamics():
    """EA-over-genotypes: fitness by (generation, op) + top-5 mechanisms by fitness."""
    import json
    import sqlite3
    conn = sqlite3.connect(RESULTS_DB)
    rows = list(conn.execute(
        "SELECT generation, op, fitness, rule_hash, genotype_json "
        "FROM individuals WHERE run_id = ? ORDER BY generation, fitness DESC",
        (EA_RUN_ID,),
    ))

    op_color = {
        "init": GREY,
        "elite": GREEN,
        "genotype_mutation": INK,
        "genotype_crossover": ACCENT,
    }
    op_label = {
        "init": "init",
        "elite": "elite",
        "genotype_mutation": "mutation",
        "genotype_crossover": "crossover",
    }

    fig, axs = plt.subplots(1, 2, figsize=(10.4, 4.0),
                            gridspec_kw={"width_ratios": [1.2, 1.0]})

    # ---- Panel A: strip by (gen, op) ----
    rng = np.random.default_rng(0)
    by_gen_max = {}
    seen = set()
    for gen, op, fit, h, _g in rows:
        if (gen, op) not in seen:
            seen.add((gen, op))
        by_gen_max[gen] = max(by_gen_max.get(gen, -1e9), fit)
        op_offset = {"init": -0.20, "elite": -0.07, "genotype_mutation": 0.06,
                     "genotype_crossover": 0.20}.get(op, 0)
        x = gen + op_offset + rng.uniform(-0.025, 0.025)
        axs[0].scatter([x], [fit], s=26, color=op_color.get(op, GREY),
                       alpha=0.85, edgecolor="white", lw=0.4)
    gens = sorted(by_gen_max)
    axs[0].plot(gens, [by_gen_max[g] for g in gens], "-", color=GREEN, lw=1.6,
                marker="o", markersize=7, markerfacecolor="white",
                markeredgewidth=1.6, label="best-of-generation")
    axs[0].axhline(0, color=GREY, lw=0.6, ls=":")
    axs[0].set_xticks(gens)
    axs[0].set_xlabel("Generation")
    axs[0].set_ylabel("Robust fitness")
    axs[0].set_title("EA dynamics on the strict 5-mouse panel")
    axs[0].grid(axis="y", color=GREY, alpha=0.25, lw=0.6)

    from matplotlib.lines import Line2D
    handles = [Line2D([0], [0], marker="o", color="w",
                      markerfacecolor=op_color[k], markersize=7, label=op_label[k])
               for k in ("init", "elite", "genotype_mutation", "genotype_crossover")]
    handles.append(Line2D([0], [0], color=GREEN, lw=1.6, marker="o",
                          markerfacecolor="white", markersize=7,
                          label="best-of-generation"))
    axs[0].legend(handles=handles, loc="lower left", frameon=False, fontsize=8)

    # ---- Panel B: top-5 unique mechanisms by fitness ----
    seen_h = set()
    top = []
    for r in sorted(rows, key=lambda x: -x[2]):
        if r[3] in seen_h:
            continue
        seen_h.add(r[3])
        top.append(r)
        if len(top) >= 5:
            break
    pay_color = {"first_price": ACCENT, "second_price": INK,
                 "posted_price": "#c79a00", "all_pay": GREEN}
    labels = []; fits = []; cols = []
    pay_rules = []
    for gen, op, fit, h, gjson in reversed(top):
        g = json.loads(gjson) if gjson else {}
        pay = (g.get("mechanism") or {}).get("payment_rule", "?")
        lang = (g.get("mechanism") or {}).get("bid_language", "?")
        short = f"{pay}, {lang}".replace("_", "-")
        labels.append(short)
        fits.append(fit)
        cols.append(pay_color.get(pay, GREY))
        pay_rules.append(pay)
    y = np.arange(len(labels))
    axs[1].barh(y, fits, color=cols, alpha=0.9, height=0.6, zorder=3)
    for yi, f in zip(y, fits):
        axs[1].text(f + 0.04, yi, f"{f:.2f}", va="center", fontsize=9, color=INK)
    axs[1].set_yticks(y)
    axs[1].set_yticklabels(labels, fontsize=9)
    axs[1].set_xlim(0, max(fits) * 1.22)
    axs[1].set_xlabel("Robust fitness")
    axs[1].set_title("Top-5 evolved mechanisms")
    axs[1].grid(axis="x", color=GREY, alpha=0.25, lw=0.6)
    # No legend: payment-rule is already in each y-tick label; the bar colors
    # are a redundant visual cue (red=first-price, ink=second-price, gold=posted).

    fig.suptitle("Evolutionary search over auction genotypes "
                 "(24 individuals $\\times$ 2 generations $\\times$ 5-mouse panel)",
                 y=1.04, fontsize=12)
    out = os.path.join(HERE, "fig5.pdf")
    fig.savefig(out)
    plt.close(fig)
    return out


def fig6_smad_vs_fitness():
    """SMAD (closeness to each mechanism's optimal bid) vs composite robust fitness."""
    df = pd.read_csv(SMAD_CSV)
    pay_color = {"first_price": ACCENT, "second_price": INK,
                 "posted_price": "#c79a00", "all_pay": GREEN, "third_price": "#6a4c93"}
    valid = df[df["smad"].notna()].copy()

    fig, axs = plt.subplots(1, 2, figsize=(10.2, 4.0),
                            gridspec_kw={"width_ratios": [1.25, 1.0]})

    # Panel A: scatter SMAD (x, lower=better) vs robust fitness (y, higher=better)
    for pay, sub in valid.groupby("payment_rule"):
        axs[0].scatter(sub["smad"], sub["robust_fitness"], s=34,
                       color=pay_color.get(pay, GREY), alpha=0.8,
                       edgecolor="white", linewidth=0.4,
                       label=pay.replace("_", "-"))
    # mark the two different winners
    best_fit = valid.sort_values("robust_fitness", ascending=False).iloc[0]
    best_smad = valid.sort_values("smad").iloc[0]
    axs[0].annotate("best fitness\n(first-price)",
                    xy=(best_fit["smad"], best_fit["robust_fitness"]),
                    xytext=(best_fit["smad"] + 1.5, best_fit["robust_fitness"] - 0.9),
                    fontsize=8.5, color=ACCENT,
                    arrowprops=dict(arrowstyle="->", color=ACCENT, lw=1))
    axs[0].annotate("lowest SMAD\n(second-price)",
                    xy=(best_smad["smad"], best_smad["robust_fitness"]),
                    xytext=(best_smad["smad"] - 3.0, best_smad["robust_fitness"] + 0.7),
                    fontsize=8.5, color=INK,
                    arrowprops=dict(arrowstyle="->", color=INK, lw=1))
    axs[0].set_xlabel("SMAD: % deviation from optimal bid (lower better)")
    axs[0].set_ylabel("Robust fitness (higher better)")
    axs[0].grid(color=GREY, alpha=0.2, lw=0.6)
    axs[0].legend(loc="lower left", frameon=False, fontsize=8)
    axs[0].set_title("Two objectives, two winners")

    # Panel B: mean SMAD by payment rule
    msm = (valid.groupby("payment_rule")["smad"].mean()
           .sort_values(ascending=True))
    y = np.arange(len(msm))
    axs[1].barh(y, msm.values, color=[pay_color.get(p, GREY) for p in msm.index],
                height=0.6, alpha=0.9, zorder=3)
    for yi, v in zip(y, msm.values):
        axs[1].text(v + 0.6, yi, f"{v:.0f}", va="center", fontsize=9, color=INK)
    axs[1].set_yticks(y)
    axs[1].set_yticklabels([p.replace("_", "-") for p in msm.index], fontsize=9)
    axs[1].set_xlabel("Mean SMAD")
    axs[1].set_xlim(0, msm.max() * 1.2)
    axs[1].grid(axis="x", color=GREY, alpha=0.25, lw=0.6)
    axs[1].set_title("Mice bid closest to optimum\nunder second-price")

    fig.suptitle("SMAD broadly tracks robust fitness, but ranks second-price first",
                 y=1.04, fontsize=12)
    out = os.path.join(HERE, "fig6.pdf")
    fig.savefig(out)
    plt.close(fig)
    return out


def fig7_reasoning_audit():
    """Do the mice reason as instructed? Dominance-argument invocation in the PLAN logs."""
    df = pd.read_csv(AUDIT_CSV)
    by_c = df.groupby("contingent")["dominance_invocation_rate"].mean()

    fig, axs = plt.subplots(1, 2, figsize=(10.2, 4.0),
                            gridspec_kw={"width_ratios": [1.0, 1.2]})

    # Panel A: dominance invocation by contingent level
    xs = [0, 1, 2]
    ys = [by_c.get(c, np.nan) for c in xs]
    axs[0].bar(xs, ys, color=[ACCENT, GREY, GREEN], width=0.6, alpha=0.9, zorder=3)
    for x, yv in zip(xs, ys):
        axs[0].text(x, yv + 0.02, f"{yv:.2f}", ha="center", fontsize=10, color=INK)
    axs[0].set_xticks(xs)
    axs[0].set_xticklabels(["0 (off)", "1", "2 (intact)"])
    axs[0].set_ylim(0, 1.0)
    axs[0].set_xlabel("Contingent-reasoning level")
    axs[0].set_ylabel("Share of plans invoking\nthe dominant-strategy argument")
    axs[0].grid(axis="y", color=GREY, alpha=0.25, lw=0.6)
    axs[0].axhspan(0, by_c.get(0, 0), color=ACCENT, alpha=0.06)
    axs[0].set_title("Reasoning reflects the instructed level\n(but leaks at level 0)")

    # Panel B: per-strain dominance invocation vs truthful behavior
    cmap = {0: ACCENT, 1: GREY, 2: GREEN}
    for c, sub in df.groupby("contingent"):
        axs[1].scatter(sub["dominance_invocation_rate"], sub["truthful_rate"],
                       s=46, color=cmap[c], alpha=0.85, edgecolor="white",
                       linewidth=0.5, label=f"contingent = {c}")
    axs[1].set_xlabel("Share of plans invoking the dominant-strategy argument")
    axs[1].set_ylabel("Truthful-bid rate")
    axs[1].grid(color=GREY, alpha=0.2, lw=0.6)
    axs[1].legend(loc="lower right", frameon=False, fontsize=8.5)
    axs[1].set_title("Stated reasoning predicts\nrealized truthful bidding")

    fig.suptitle("Auditing the reasoning chains: do the digital mice reason as instructed?",
                 y=1.04, fontsize=12)
    out = os.path.join(HERE, "fig7.pdf")
    fig.savefig(out)
    plt.close(fig)
    return out


def main():
    # Regenerate the SMAD and reasoning-audit CSVs first (no LLM calls).
    import make_extra_analyses as mx
    mx.ea_smad_ranking()
    mx.reasoning_audit()
    for fn in (fig1_construct_marginals, fig2_behavioral_signatures, fig3_calibration_coverage,
               fig4_first_price, fig5_ea_dynamics, fig6_smad_vs_fitness, fig7_reasoning_audit):
        path = fn()
        size = os.path.getsize(path)
        print(f"wrote {os.path.relpath(path, REPO_ROOT)}  ({size} bytes)")


if __name__ == "__main__":
    main()
