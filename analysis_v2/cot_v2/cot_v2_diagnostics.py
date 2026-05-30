"""
Construct-validity + coverage diagnostics for the v2 CoT run.

Inputs:
    data_v2/results/spsb_basis_cot_v2_summary.csv  (27 CoT strains)
    data_v2/results/spsb_outcome_v2_summary.csv    (3 outcome strains)
    data_v2/results/spsb_basis_grid_summary.csv    (v1 standard-prompt baseline)

Writes:
    analysis_v2/cot_v2/spsb_cot_strain_summary.csv
    analysis_v2/cot_v2/spsb_cot_axis_diagnostics.csv
    analysis_v2/cot_v2/spsb_cot_construct_table.csv  (Δ vs bars, with v1 comparison)
    analysis_v2/cot_v2/spsb_cot_coverage_scorecard.csv

Run from repo root:
    ./venv/bin/python analysis_v2/cot_v2/cot_v2_diagnostics.py
"""

import os

import numpy as np
import pandas as pd

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
RESULTS_DIR = os.path.join(REPO_ROOT, "data_v2/results")
OUT_DIR = os.path.join(REPO_ROOT, "analysis_v2/cot_v2")

PANEL_IN = os.path.join(RESULTS_DIR, "spsb_basis_cot_v2_summary.csv")
OUTCOME_IN = os.path.join(RESULTS_DIR, "spsb_outcome_v2_summary.csv")
V1_IN = os.path.join(RESULTS_DIR, "spsb_basis_grid_summary.csv")

STRAIN_OUT = os.path.join(OUT_DIR, "spsb_cot_strain_summary.csv")
AXIS_OUT = os.path.join(OUT_DIR, "spsb_cot_axis_diagnostics.csv")
TABLE_OUT = os.path.join(OUT_DIR, "spsb_cot_construct_table.csv")
COVERAGE_OUT = os.path.join(OUT_DIR, "spsb_cot_coverage_scorecard.csv")


def classify(row):
    v, b = row["value"], row["bid"]
    if pd.isna(v) or pd.isna(b):
        return "missing"
    tol = 0.05
    if v == 0:
        return "truthful" if abs(b) <= tol else "overbid"
    if b > v + tol:
        return "overbid"
    if b < v - tol:
        return "underbid"
    return "truthful"


def add_classes(df):
    df = df.copy()
    df["mistake_class"] = df.apply(classify, axis=1)
    df["truthful"] = (df["mistake_class"] == "truthful").astype(float)
    df["overbid"] = (df["mistake_class"] == "overbid").astype(float)
    df["underbid"] = (df["mistake_class"] == "underbid").astype(float)
    df["abs_deviation"] = df["deviation"].abs()
    # Extra: extreme overbid, near-zero
    df["extreme_overbid"] = (df["deviation"] >= 5).astype(float)
    near_zero = (df["bid"] <= 0.1) & (df["value"] > 5)
    df["near_zero"] = near_zero.astype(float)
    df["moderate_underbid"] = (
        (df["mistake_class"] == "underbid") & (~near_zero)
    ).astype(float)
    return df


def summarize_strains(df, group_cols):
    grouped = df.groupby(group_cols, dropna=False)
    out = grouped.agg(
        n=("bid", "size"),
        mean_value=("value", "mean"),
        mean_bid=("bid", "mean"),
        mean_ratio=("ratio", "mean"),
        mean_deviation=("deviation", "mean"),
        mean_abs_deviation=("abs_deviation", "mean"),
        truthful_rate=("truthful", "mean"),
        overbid_rate=("overbid", "mean"),
        underbid_rate=("underbid", "mean"),
        extreme_overbid_rate=("extreme_overbid", "mean"),
        near_zero_rate=("near_zero", "mean"),
        moderate_underbid_rate=("moderate_underbid", "mean"),
    ).reset_index()
    return out


def axis_diagnostics(strain_summary):
    rows = []
    axes = [("c", "contingent"), ("f", "forward"), ("b", "beliefs")]
    metrics = [
        "truthful_rate",
        "overbid_rate",
        "underbid_rate",
        "mean_abs_deviation",
        "mean_ratio",
    ]
    for axis_code, axis_name in axes:
        for metric in metrics:
            by_level = strain_summary.groupby(axis_code)[metric].mean().to_dict()
            low, mid, high = by_level.get(0), by_level.get(1), by_level.get(2)
            rows.append(
                {
                    "axis": axis_name,
                    "metric": metric,
                    "level0_mean": low,
                    "level1_mean": mid,
                    "level2_mean": high,
                    "level2_minus_level0": None
                    if low is None or high is None
                    else high - low,
                }
            )
    return pd.DataFrame(rows)


def construct_table(cot_axis, v1_axis):
    """Five pre-registered directional tests, CoT vs v1 standard."""
    bars = {
        ("contingent", "truthful_rate"): (0.05, "positive"),
        ("contingent", "underbid_rate"): (0.05, "negative"),
        ("forward", "truthful_rate"): (0.05, "positive"),
        ("forward", "underbid_rate"): (0.05, "negative"),
        ("beliefs", "truthful_rate"): (0.03, "positive"),
    }
    rows = []
    for (axis, metric), (bar, expected) in bars.items():
        cot_row = cot_axis[
            (cot_axis["axis"] == axis) & (cot_axis["metric"] == metric)
        ].iloc[0]
        v1_row = v1_axis[
            (v1_axis["axis"] == axis) & (v1_axis["metric"] == metric)
        ].iloc[0]
        delta_cot = cot_row["level2_minus_level0"]
        delta_v1 = v1_row["level2_minus_level0"]
        sign_ok = (
            (delta_cot > 0 and expected == "positive")
            or (delta_cot < 0 and expected == "negative")
        )
        bar_ok = abs(delta_cot) >= bar
        rows.append(
            {
                "axis": axis,
                "metric": metric,
                "delta_v1": round(delta_v1, 3),
                "delta_cot": round(delta_cot, 3),
                "bar": bar,
                "expected_sign": expected,
                "sign_ok": sign_ok,
                "bar_ok": bar_ok,
                "pass": sign_ok and bar_ok,
                "level0_cot": round(cot_row["level0_mean"], 3),
                "level1_cot": round(cot_row["level1_mean"], 3),
                "level2_cot": round(cot_row["level2_mean"], 3),
            }
        )
    return pd.DataFrame(rows)


def coverage_scorecard(strain_summary, outcome_summary):
    """Five canonical SPSB mistake classes with pre-registered detection floors."""
    classes = [
        ("truthful", "truthful_rate", 0.50),
        ("moderate_underbid", "moderate_underbid_rate", 0.10),
        ("overbid", "overbid_rate", 0.10),
        ("extreme_overbid", "extreme_overbid_rate", 0.05),
        ("near_zero", "near_zero_rate", 0.05),
    ]
    rows = []
    for cls, metric, floor in classes:
        max_strain = strain_summary.loc[strain_summary[metric].idxmax()]
        # outcome strains as alternative source
        if not outcome_summary.empty:
            o_max = outcome_summary.loc[outcome_summary[metric].idxmax()]
            o_label = f"{o_max['outcome_id']} ({o_max[metric]:.3f})"
        else:
            o_label = ""
        rows.append(
            {
                "mistake_class": cls,
                "metric": metric,
                "floor": floor,
                "max_rate_cot_panel": round(max_strain[metric], 3),
                "best_strain": max_strain["basis_id"],
                "covered_by_cot": max_strain[metric] >= floor,
                "outcome_alt": o_label,
            }
        )
    return pd.DataFrame(rows)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    panel = add_classes(pd.read_csv(PANEL_IN))
    outcome = add_classes(pd.read_csv(OUTCOME_IN))
    v1 = add_classes(pd.read_csv(V1_IN))

    # Strain-level summaries
    cot_strain = summarize_strains(
        panel,
        ["basis_id", "c", "f", "b", "contingent", "forward", "beliefs"],
    )
    v1_strain = summarize_strains(
        v1,
        ["basis_id", "c", "f", "b", "contingent", "forward", "beliefs"],
    )
    outcome_strain = summarize_strains(outcome, ["outcome_id", "target_class"])

    cot_strain.to_csv(STRAIN_OUT, index=False)
    print(f"wrote {STRAIN_OUT}: rows={len(cot_strain)}")

    # Axis marginals (CoT and v1)
    cot_axis = axis_diagnostics(cot_strain)
    v1_axis = axis_diagnostics(v1_strain)
    cot_axis["panel"] = "cot_v2"
    v1_axis["panel"] = "v1_standard"
    pd.concat([cot_axis, v1_axis]).to_csv(AXIS_OUT, index=False)
    print(f"wrote {AXIS_OUT}")

    # 5 pre-reg directional tests with v1 comparison
    table = construct_table(cot_axis, v1_axis)
    table.to_csv(TABLE_OUT, index=False)
    print(f"wrote {TABLE_OUT}")
    print("\nConstruct-validity table (CoT vs v1):")
    print(table.to_string(index=False))

    # Coverage scorecard
    coverage = coverage_scorecard(cot_strain, outcome_strain)
    coverage.to_csv(COVERAGE_OUT, index=False)
    print(f"\nwrote {COVERAGE_OUT}")
    print("\nCoverage scorecard:")
    print(coverage.to_string(index=False))

    print("\nOutcome-instruction strains:")
    print(outcome_strain.to_string(index=False))


if __name__ == "__main__":
    main()
