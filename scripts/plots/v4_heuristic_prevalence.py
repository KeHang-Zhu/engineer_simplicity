#!/usr/bin/env python3
"""
Figure F3 (auction-v4 §5): which script does each family retrieve?

Heatmap of primary heuristic-taxonomy shares (analysis/classify_heuristics.py)
by column blocks: legacy models x SPSB | frontier models x SPSB | ascending
clock (pooled by corpus). Rows = the paper's taxonomy in three visual groups:
sealed-bid heuristics (H1-H5), the clock rule (H6), normative modes (H8, H7,
T), residual U. Structurally-empty cells (labels the cascade cannot assign in
that column) are hatched, not zero.

Judge-validation caveat carried in the caption (coarse decision-mode kappa
0.55; 10-label kappa 0.36; appendix confusion matrix) -- shares here are
descriptive, never inputs to the causal battery.

Data: results/traces/heuristic_portfolio.csv
Output: plots/v4_heuristic_prevalence.pdf
"""

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
CSV = os.path.join(REPO, "results", "traces", "heuristic_portfolio.csv")
OUT = os.path.join(REPO, "plots", "v4_heuristic_prevalence.pdf")

ROWS = [("H1", "H1 profit-margin shading"),
        ("H2", "H2 avoid overpayment"),
        ("H3", "H3 opponent anchoring"),
        ("H4", "H4 salient-number anchor"),
        ("H5", "H5 win-prob. / aggressive"),
        ("H6", "H6 price-threshold exit"),
        ("H8", "H8 worst-case proof"),
        ("H7", "H7 formal derivation"),
        ("T", "T truthful assertion"),
        ("U", "U unclassified")]
GROUP_BREAKS = [5, 6, 9]  # after H5, after H6, after T

LEGACY = [("gpt-4o", "GPT-4o"),
          ("claude-3-5-haiku-20241022", "Claude 3.5 H."),
          ("gemini-2.0-flash", "Gemini 2.0 F."),
          ("google/gemma-3-27b-it", "Gemma 3 27B")]
FRONTIER = [("openai/gpt-5", "GPT-5"),
            ("openai/gpt-5-mini", "GPT-5 mini"),
            ("anthropic/claude-sonnet-5", "Claude Son. 5"),
            ("google/gemini-2.5-flash", "Gemini 2.5 F.")]


def share_map(port, corpus, model, mech):
    sub = port[(port.corpus == corpus) & (port.model == model)
               & (port.mechanism == mech)]
    return dict(zip(sub.heuristic, sub.share))


def pooled_clock(port, corpus):
    sub = port[(port.corpus == corpus) & (port.mechanism == "ascending_clock")]
    tot = sub.groupby("heuristic")["n"].sum()
    return (tot / tot.sum()).to_dict()


def main():
    port = pd.read_csv(CSV)
    print(f"consumed {len(port)} portfolio rows from {CSV}")

    cols, blocks = [], []
    for m, lab in LEGACY:
        cols.append((lab, share_map(port, "legacy", m, "spsb_sealed"),
                     "sealed"))
    blocks.append(("Legacy models — sealed SPSB", 0, 4))
    for m, lab in FRONTIER:
        cols.append((lab, share_map(port, "frontier", m, "spsb_sealed"),
                     "sealed"))
    blocks.append(("Frontier models — sealed SPSB", 4, 8))
    cols.append(("legacy", pooled_clock(port, "legacy"), "clock"))
    cols.append(("frontier", pooled_clock(port, "frontier"), "clock"))
    blocks.append(("Clock", 8, 10))

    M = np.full((len(ROWS), len(cols)), np.nan)
    for j, (_lab, smap, kind) in enumerate(cols):
        for i, (h, _name) in enumerate(ROWS):
            structural = ((kind == "sealed" and h == "H6")
                          or (kind == "clock"
                              and h in {"H1", "H2", "H3", "H4", "H5", "H8",
                                        "T"}))
            if structural:
                continue
            M[i, j] = smap.get(h, 0.0)

    plt.rcParams.update({
        "font.size": 9.5, "axes.titlesize": 11.5, "axes.titleweight": "bold",
        "figure.facecolor": "white", "savefig.facecolor": "white",
        "savefig.dpi": 200,
    })
    fig, ax = plt.subplots(figsize=(8.6, 5.0))
    cmap = mcolors.LinearSegmentedColormap.from_list(
        "blues1", ["#FFFFFF", "#B7D3E8", "#0072B2", "#03426B"])
    masked = np.ma.masked_invalid(M)
    ax.imshow(masked, cmap=cmap, vmin=0, vmax=1.0, aspect="auto",
              interpolation="nearest")

    # hatch structural gaps
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            if np.isnan(M[i, j]):
                ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1,
                                           facecolor="#F4F4F1",
                                           edgecolor="#DDDDDA", hatch="///",
                                           linewidth=0.4, zorder=2))
            elif M[i, j] >= 0.05:
                lum = M[i, j]
                ax.text(j, i, f"{M[i, j]*100:.0f}", ha="center", va="center",
                        fontsize=8.6,
                        color=("white" if lum > 0.55 else "#1A1A1A"),
                        zorder=3)

    # white separators between column blocks and row groups
    for _name, _s, e in blocks[:-1]:
        ax.axvline(e - 0.5, color="white", lw=3.5)
    for b in GROUP_BREAKS[:-1]:
        ax.axhline(b - 0.5, color="white", lw=3.5)

    ax.set_xticks(range(len(cols)))
    ax.set_xticklabels([c[0] for c in cols], rotation=28, ha="right",
                       fontsize=8.6)
    ax.set_yticks(range(len(ROWS)))
    ax.set_yticklabels([r[1] for r in ROWS], fontsize=9)
    for name, s, e in blocks:
        ax.text((s + e - 1) / 2, -0.95, name, ha="center", va="bottom",
                fontsize=9, fontweight="bold", color="#333333")
    ax.set_title("Primary decision script per trace (% of traces), "
                 "by model and mechanism", loc="left", pad=30)
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)

    fig.tight_layout()
    fig.savefig(OUT, bbox_inches="tight")
    print(f"wrote {OUT}")
    print(pd.DataFrame(M, index=[r[0] for r in ROWS],
                       columns=[c[0] for c in cols]).round(3).to_string())


if __name__ == "__main__":
    main()
