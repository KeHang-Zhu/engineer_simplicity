"""
Summarize and analyze the FPSB CoT stress-test run.

Inputs:
    data_v2/experiment_logs/gpt5mini/fpsb_cot_*/result_1_*.json

Outputs:
    data_v2/results/spsb_basis_fpsb_cot_summary.csv
    analysis_v2/cot_v2/fpsb_cot_strain_summary.csv
    writeup_v2/reports/digital_mice_v2/fig4_cot.pdf

Run from repo root:
    ./venv/bin/python analysis_v2/cot_v2/fpsb_cot_analysis.py
"""

import glob
import json
import os
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
LOG_GLOB = os.path.join(REPO_ROOT, "data_v2/experiment_logs/gpt5mini/fpsb_cot_*")
RESULTS_OUT = os.path.join(REPO_ROOT, "data_v2/results/spsb_basis_fpsb_cot_summary.csv")
STRAIN_OUT = os.path.join(REPO_ROOT, "analysis_v2/cot_v2/fpsb_cot_strain_summary.csv")
FIG_OUT = os.path.join(REPO_ROOT, "writeup_v2/reports/digital_mice_v2/fig4_cot.pdf")

BNE_RATIO = 2.0 / 3.0  # n=3 risk-neutral BNE for U[0, V_max]

COT_RE = re.compile(r"^fpsb_cot_(c\d_f\d_b\d)$")


def _iter_results(strain_dir):
    for path in sorted(glob.glob(os.path.join(strain_dir, "result_1_*.json"))):
        with open(path) as fp:
            yield path, json.load(fp)


def _summarize_one(payload, run_idx):
    rows = []
    for _, rd in payload.items():
        values = rd.get("value", [])
        bids = [b["bid"] for b in rd.get("history", {}).get("bidding history", [])]
        plans = rd.get("plan", [])
        for agent_idx, (v, b) in enumerate(zip(values, bids)):
            rows.append(
                {
                    "run": run_idx,
                    "agent": agent_idx,
                    "value": v,
                    "bid": b,
                    "deviation_from_value": (b - v) if v is not None else None,
                    "ratio": (b / v) if v else None,
                    "deviation_from_bne": (b - BNE_RATIO * v) if v is not None else None,
                    "plan": plans[agent_idx] if agent_idx < len(plans) else "",
                }
            )
    return rows


def summarize():
    out = []
    for strain_dir in sorted(glob.glob(LOG_GLOB)):
        name = os.path.basename(strain_dir)
        slug = name[len("fpsb_cot_"):]
        m = COT_RE.match(name)
        if m:
            strain_id = m.group(1)
            kind = "cot_basis"
            c, f, b = int(strain_id[1]), int(strain_id[4]), int(strain_id[7])
        else:
            strain_id = slug
            c, f, b = (None, None, None)
            kind = "outcome" if slug.startswith("spsb_outcome") else "error_mouse"
        for run_idx, (_, payload) in enumerate(_iter_results(strain_dir)):
            for row in _summarize_one(payload, run_idx):
                row.update(
                    {
                        "strain_id": strain_id,
                        "kind": kind,
                        "c": c,
                        "f": f,
                        "b": b,
                    }
                )
                out.append(row)
    return pd.DataFrame(out)


def per_strain_summary(df):
    df = df.copy()
    df["above_bne"] = (df["bid"] > BNE_RATIO * df["value"]).astype(float)
    df["above_value"] = (df["bid"] > df["value"]).astype(float)
    df["near_zero"] = (df["bid"] <= 0.1).astype(float)
    grouped = df.groupby(["strain_id", "kind", "c", "f", "b"], dropna=False)
    out = grouped.agg(
        n=("bid", "size"),
        mean_value=("value", "mean"),
        mean_bid=("bid", "mean"),
        mean_ratio=("ratio", "mean"),
        median_ratio=("ratio", "median"),
        above_bne_rate=("above_bne", "mean"),
        above_value_rate=("above_value", "mean"),
        near_zero_rate=("near_zero", "mean"),
        mean_dev_from_bne=("deviation_from_bne", "mean"),
        mean_abs_dev_from_bne=("deviation_from_bne", lambda s: s.abs().mean()),
    ).reset_index()
    return out


def plot_fpsb(df):
    fig, axes = plt.subplots(1, 4, figsize=(14, 3.8), sharex=True, sharey=True)

    panels = [
        ("c0_f0_b0", "CoT c0_f0_b0\n(margin-of-safety)", "#d96666"),
        ("c1_f1_b1", "CoT c1_f1_b1\n(intermediate)", "#d9b566"),
        ("c2_f2_b2", "CoT c2_f2_b2\n(rational target)", "#5fa84f"),
    ]
    for ax, (sid, title, color) in zip(axes[:3], panels):
        sub = df[df["strain_id"] == sid]
        ax.scatter(sub["value"], sub["bid"], color=color, alpha=0.6, s=28, edgecolor="black")
        x = np.linspace(0, 50, 100)
        ax.plot(x, x, "--", color="grey", linewidth=1, label="45° (value)")
        ax.plot(x, BNE_RATIO * x, "-", color="black", linewidth=1, label="BNE = (2/3)v")
        ax.set_xlim(0, 50); ax.set_ylim(0, 55)
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("private value $v$")
    axes[0].set_ylabel("bid $b$")
    axes[0].legend(loc="upper left", fontsize=7)

    # Right panel: error mice + outcome strains overlaid
    overlay_strains = [
        ("spsb_error_win_seeker", "win_seeker (error)", "#c75450"),
        ("spsb_error_loss_averse", "loss_averse (error)", "#666666"),
        ("spsb_outcome_overbid", "outcome: overbid", "#a02050"),
        ("spsb_outcome_underbid", "outcome: underbid", "#4070a0"),
        ("spsb_outcome_truthful", "outcome: truthful", "#208050"),
    ]
    ax = axes[3]
    for sid, label, color in overlay_strains:
        sub = df[df["strain_id"] == sid]
        if sub.empty:
            continue
        ax.scatter(sub["value"], sub["bid"], color=color, alpha=0.7, s=28,
                   edgecolor="black", label=label)
    x = np.linspace(0, 50, 100)
    ax.plot(x, x, "--", color="grey", linewidth=1)
    ax.plot(x, BNE_RATIO * x, "-", color="black", linewidth=1)
    ax.set_xlim(0, 50); ax.set_ylim(0, 55)
    ax.set_title("Error mice + outcome strains", fontsize=10)
    ax.set_xlabel("private value $v$")
    ax.legend(loc="upper left", fontsize=6)

    plt.tight_layout()
    plt.savefig(FIG_OUT)
    plt.close()
    print(f"wrote {FIG_OUT}")


def main():
    df = summarize()
    df.to_csv(RESULTS_OUT, index=False)
    print(f"wrote {RESULTS_OUT}: rows={len(df)}")
    strain = per_strain_summary(df)
    strain.to_csv(STRAIN_OUT, index=False)
    print(f"wrote {STRAIN_OUT}")

    print("\nStrain summary (first 12 rows):")
    print(strain.head(12).to_string(index=False))
    print("\nKey strains:")
    key = ["c0_f0_b0", "c1_f1_b1", "c2_f2_b2",
           "spsb_error_second_price_overgeneralizer",
           "spsb_error_payment_panic_near_zero",
           "spsb_outcome_overbid", "spsb_outcome_underbid",
           "spsb_outcome_truthful"]
    cols = ["strain_id", "kind", "n", "mean_ratio", "median_ratio",
            "above_bne_rate", "above_value_rate"]
    print(strain[strain["strain_id"].isin(key)][cols].to_string(index=False))

    plot_fpsb(df)


if __name__ == "__main__":
    main()
