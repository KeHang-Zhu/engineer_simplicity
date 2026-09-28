#!/usr/bin/env python3
"""Plot six headline interventions using the original Figure 4 estimates.

The CSV is an unchanged local copy of the original analysis export; see
docs/FIGURE4.md for provenance.
Language estimates have no stored confidence intervals. Bid-error intervals
are the original percentile run-cluster bootstrap intervals, not inversions
of the wild-cluster-bootstrap tests reported in the manuscript.

Run from any directory:
    python3 scripts/plot_intervention_language_bids.py
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
HEADLINE = (
    "Payoff Safety",
    "Payoff Tree",
    "Worst-case scaffold",
    "Menu restatement",
    "First-order beliefs",
    "Risk-averse persona",
)
Y_POSITIONS = (6.0, 5.0, 4.0, 3.0, 1.2, 0.2)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path,
                        default=ROOT / "data/intervention_language_bids.csv")
    parser.add_argument("--output", type=Path,
                        default=ROOT / "plots/intervention_language_bids.pdf")
    args = parser.parse_args()
    with args.data.open(newline="") as file:
        source_rows = list(csv.DictReader(file))
    by_name = {row["lever"]: row for row in source_rows}
    if len(by_name) != len(source_rows):
        raise ValueError("Duplicate intervention names in input data")
    rows = [by_name[name] for name in HEADLINE]
    expected_tiers = ("designed",) * 4 + ("exploratory",) * 2
    for row, expected in zip(rows, expected_tiers):
        if row["tier"] != expected:
            raise ValueError(f"Unexpected comparison tier: {row['lever']}")

    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "axes.titlesize": 10.5,
        "axes.labelsize": 9.5,
        "xtick.labelsize": 9,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.spines.left": False,
        "axes.linewidth": 0.65,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "savefig.facecolor": "white",
        "savefig.dpi": 220,
    })
    fig = plt.figure(figsize=(7.0, 3.15))
    grid = fig.add_gridspec(
        1, 2, left=0.292, right=0.985, bottom=0.18, top=0.83,
        width_ratios=(1.0, 1.23), wspace=0.28,
    )
    language = fig.add_subplot(grid[0, 0])
    bids = fig.add_subplot(grid[0, 1], sharey=language)
    ink = "#214D66"
    muted = "#666666"

    for ax in (language, bids):
        ax.set_ylim(-0.45, 7.0)
        ax.axvline(0, color="#999999", linewidth=0.7,
                   linestyle=(0, (3, 3)), zorder=1)
        ax.tick_params(axis="y", which="both", left=False, labelleft=False)
        ax.tick_params(axis="x", length=3, color="#777777")
        ax.spines["bottom"].set_color("#777777")
        ax.axhline(2.35, color="#DDDDDD", linewidth=0.7, zorder=0)

    for row, y in zip(rows, Y_POSITIONS):
        language.text(-0.06, y, row["lever"],
                      transform=language.get_yaxis_transform(),
                      ha="right", va="center", fontsize=9.5,
                      color="#252525", clip_on=False)
        if row["echo_class"] == "by-construction":
            language.text(5, y, "Removed by prompt", va="center",
                          fontsize=8.5, color=muted)
        else:
            language.plot(float(row["prev_diff"]) * 100, y,
                          marker="o", markersize=5.4, color=ink, zorder=3)
        effect = float(row["absdev_diff"])
        lower, upper = float(row["absdev_ci_lo"]), float(row["absdev_ci_hi"])
        if not lower <= effect <= upper:
            raise ValueError(f"Invalid bid-error interval: {row['lever']}")
        bids.errorbar(effect, y, xerr=[[effect - lower], [upper - effect]],
                      fmt="o", color=ink, markersize=5.4, elinewidth=1.05,
                      capsize=2.6, capthick=1.05, zorder=3)

    for label, y in (("Primary contrasts", 6.8),
                     ("Exploratory comparisons", 2.0)):
        language.text(-0.06, y, label,
                      transform=language.get_yaxis_transform(),
                      ha="right", va="center", fontsize=8.4,
                      fontweight="bold", color=muted, clip_on=False)

    language.set_title("Targeted language\n(change in percentage points)", pad=10)
    bids.set_title("Mean absolute bid error\n(change in dollars)", pad=10)
    language.set_xlim(-7, 85)
    language.set_xticks((0, 20, 40, 60, 80))
    bids.set_xlim(-4.25, 4.75)
    bids.set_xticks((-4, -2, 0, 2, 4))
    language.set_xlabel("More target language →", labelpad=6)
    bids.set_xlabel("← Smaller error      Larger error →", labelpad=6)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output)
    png_path = args.output.with_suffix(".png")
    fig.savefig(png_path)
    plt.close(fig)
    print(f"Read {len(source_rows)} interventions; plotted {len(rows)}")
    print(f"Wrote {args.output}")
    print(f"Wrote {png_path}")


if __name__ == "__main__":
    main()
