#!/usr/bin/env python3
"""
Figure F2 (auction-v4 §5): execution fidelity vs. plan selection.

Panel (a): stated intended bid (parsed from the plan with the parser of
record, build_trace_mediation.parse_stated_bid) vs. realized bid -- pooled
gray hexbin underlay + per-model binned means in the repo's Okabe-Ito model
colors + 45-degree line. Message: models submit the number they state.

Panel (b): realized deviation (bid - value) distributions conditional on the
stated-intent class (shade / truthful / overbid / unanchored), horizontal
violins with mean diamonds and the +-$0.50 truthful band. Message: the bids
follow whichever plan was selected; the biggest errors live in the plans
that never anchor on value at all.

Scope: legacy sealed second-price traces (the lever-analysis scope).
Supersedes results/traces/mediation/consistency_calibration.pdf.
Output: plots/v4_fidelity_selection.pdf
"""

import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(REPO, "analysis"))

from build_trace_mediation import parse_stated_bid  # noqa: E402

CSV = os.path.join(REPO, "results", "traces", "trace_features.csv")
OUT = os.path.join(REPO, "plots", "v4_fidelity_selection.pdf")

C = {"gpt-4o": "#0072B2", "claude-3-5-haiku-20241022": "#D55E00",
     "gemini-2.0-flash": "#009E73", "google/gemma-3-27b-it": "#CC79A7"}
LAB = {"gpt-4o": "GPT-4o", "claude-3-5-haiku-20241022": "Claude 3.5 Haiku",
       "gemini-2.0-flash": "Gemini 2.0 Flash",
       "google/gemma-3-27b-it": "Gemma 3 27B"}


def main():
    df = pd.read_csv(CSV, low_memory=False)
    sp = df[df["mechanism"] == "spsb_sealed"].copy()
    sp["stated_bid"] = sp["plan"].apply(parse_stated_bid)
    print(f"consumed {len(sp)} sealed-SPSB rows from {CSV}")

    plt.rcParams.update({
        "font.size": 10, "axes.titlesize": 11.5, "axes.titleweight": "bold",
        "axes.labelsize": 10.5, "legend.frameon": False,
        "legend.fontsize": 8.5, "figure.facecolor": "white",
        "axes.facecolor": "white", "savefig.facecolor": "white",
        "savefig.dpi": 200, "axes.grid": False,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.linewidth": 0.6,
    })
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.6, 4.1),
                                   gridspec_kw={"width_ratios": [1, 1.25],
                                                "wspace": 0.24})

    # ---- (a) stated vs realized ----
    cal = sp[sp["stated_bid"].notna()].copy()
    cal["sb_err"] = cal["stated_bid"] - cal["bid"]
    hb = ax1.hexbin(cal["stated_bid"], cal["bid"], gridsize=35, cmap="Greys",
                    mincnt=1, linewidths=0.0, vmax=None, alpha=0.55,
                    extent=(0, 50, 0, 50))
    ax1.plot([0, 50], [0, 50], ls="--", lw=0.9, color="#999999", zorder=2)
    bins = np.arange(0, 51, 5)
    centers = (bins[:-1] + bins[1:]) / 2
    stats = []
    for model in C:
        g = cal[cal.model == model].copy()
        g["bin"] = pd.cut(g["stated_bid"], bins=bins, labels=centers,
                          include_lowest=True)
        m = g.groupby("bin", observed=True)["bid"].mean()
        ax1.plot(m.index.astype(float), m.values, marker="o", ms=3.6, lw=1.6,
                 color=C[model], label=LAB[model], zorder=4)
        r = g["stated_bid"].corr(g["bid"])
        mae = g["sb_err"].abs().mean()
        stats.append(f"{LAB[model]}: r={r:.2f}, |Δ|=${mae:.2f}")
    ax1.set_xlabel("Stated intended bid parsed from the plan ($)")
    ax1.set_ylabel("Realized bid ($)")
    ax1.set_title("(a) Models submit the number they state", loc="left")
    ax1.set_xlim(0, 50)
    ax1.set_ylim(0, 50)
    ax1.legend(loc="upper left", handlelength=1.4)
    ax1.text(0.985, 0.03, "\n".join(stats), transform=ax1.transAxes,
             ha="right", va="bottom", fontsize=7.4, color="#555555")

    # ---- (b) deviation by stated-intent class ----
    # class definitions match the paper's §J.3 (single-feature, inclusive),
    # so the printed shares reproduce the published 97.1 / 87.6 / 77.6.
    shade = sp[sp.shading_intent == 1]
    overb = sp[sp.overbid_intent == 1]
    truth = sp[sp.truthful_intent == 1]
    unanc = sp[(sp.shading_intent == 0) & (sp.overbid_intent == 0)
               & (sp.truthful_intent == 0)]
    groups = [
        ("states overbid intent", overb,
         f"{(overb['deviation'] > 0).mean():.1%} above value · "
         f"mean {overb['deviation'].mean():+.2f}$"),
        ("states truthful intent", truth,
         f"{(truth['abs_dev'] <= 0.5).mean():.1%} within ±$0.50"),
        ("states shading intent", shade,
         f"{(shade['deviation'] < 0).mean():.1%} below value · "
         f"mean {shade['deviation'].mean():+.2f}$"),
        ("no value-anchored intent", unanc,
         f"n={len(unanc):,} · mean {unanc['deviation'].mean():+.2f}$ · "
         "largest errors"),
    ]
    ax2.axvspan(-0.5, 0.5, color="#EDEDEA", zorder=0)
    ax2.axvline(0, ls="--", lw=0.8, color="#B8B8B8", zorder=1)
    lo, hi = -16, 8
    ypos = np.arange(len(groups))[::-1]
    for y, (name, g, note) in zip(ypos, groups):
        data = g["deviation"].clip(lo, hi).values
        vp = ax2.violinplot([data], positions=[y], vert=False, widths=0.86,
                            showextrema=False)
        for body in vp["bodies"]:
            body.set_facecolor("#0072B2")
            body.set_alpha(0.32)
            body.set_edgecolor("#0072B2")
            body.set_linewidth(0.8)
        ax2.plot(g["deviation"].mean(), y, "D", ms=6.5, color="#0072B2",
                 mec="white", mew=0.8, zorder=5)
        ax2.text(hi + 0.4, y + 0.16, name, fontsize=9, fontweight="bold",
                 va="center", ha="left", color="#222222")
        ax2.text(hi + 0.4, y - 0.22, note, fontsize=7.8, va="center",
                 ha="left", color="#555555")
    ax2.set_yticks([])
    ax2.set_xlim(lo, hi + 13)
    ax2.set_xticks(np.arange(-15, 9, 5))
    ax2.set_xlabel("Realized deviation, bid − value ($; clipped at "
                   f"[{lo}, {hi}] for display)")
    ax2.set_title("(b) …so the failure is which plan gets selected",
                  loc="left")
    ax2.annotate("±$0.50", xy=(0, ypos.max() + 0.62), fontsize=7.2,
                 color="#888888", ha="center")

    fig.tight_layout()
    fig.savefig(OUT, bbox_inches="tight")
    print(f"wrote {OUT}")
    for name, g, note in groups:
        print(f"  {name}: n={len(g)}, {note}")


if __name__ == "__main__":
    main()
