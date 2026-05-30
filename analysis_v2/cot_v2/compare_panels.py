"""
Side-by-side comparison of three SPSB digital-mouse panels:

  * v1 standard prompt     (data_v2/results/spsb_basis_grid_summary.csv)
  * v2 CoT (demonstration) (data_v2/results/spsb_basis_cot_v2_summary.csv)
  * v2 strict (no example)  (data_v2/results/spsb_basis_strict_v2_summary.csv)

Outputs:
  analysis_v2/cot_v2/panel_comparison_axis_marginals.csv
  analysis_v2/cot_v2/panel_comparison_construct_table.csv
  analysis_v2/cot_v2/panel_comparison_audit.csv

Run from repo root:
    ./venv/bin/python analysis_v2/cot_v2/compare_panels.py
"""

import os
import re

import pandas as pd

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
OUT_DIR = os.path.join(REPO_ROOT, "analysis_v2/cot_v2")

INPUTS = {
    "v1_standard": "data_v2/results/spsb_basis_grid_summary.csv",
    "cot_v2": "data_v2/results/spsb_basis_cot_v2_summary.csv",
    "strict_v2": "data_v2/results/spsb_basis_strict_v2_summary.csv",
}

# Dominance-invocation regex (same as cot_v2_reasoning_audit.py)
DOMINANCE_PATTERNS = [
    r"\bdominant\b",
    r"\bweakly dominant\b",
    r"\bdominance\b",
    r"\bvickrey\b",
    r"\btruthful(ly)? bid",
    r"\bbid(ding)? (my|the) value\b",
    r"\bbid(ding)? equal to my value\b",
    r"\bsecond[- ]highest bid\b.*(?:not (?:my own|the bidder.?s own) bid)",
    r"\bonly (?:changes|affects) whether i win",
    r"\bdoes? not (?:change|affect) (?:how much|what) i pay",
    r"\bbid my value\b",
]
DOMINANCE_RE = re.compile("|".join(DOMINANCE_PATTERNS), re.IGNORECASE)


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


def load_panel(path):
    df = pd.read_csv(path)
    df["mistake_class"] = df.apply(classify, axis=1)
    df["truthful"] = (df["mistake_class"] == "truthful").astype(float)
    df["overbid"] = (df["mistake_class"] == "overbid").astype(float)
    df["underbid"] = (df["mistake_class"] == "underbid").astype(float)
    df["abs_deviation"] = df["deviation"].abs()
    if "plan" in df.columns:
        df["plan"] = df["plan"].fillna("").astype(str)
        df["dominance_invocation"] = df["plan"].str.contains(DOMINANCE_RE).astype(int)
    else:
        df["dominance_invocation"] = float("nan")
    return df


def axis_marginals(df):
    strain = df.groupby(
        ["basis_id", "c", "f", "b"], dropna=False
    ).agg(
        truthful_rate=("truthful", "mean"),
        overbid_rate=("overbid", "mean"),
        underbid_rate=("underbid", "mean"),
        mean_ratio=("ratio", "mean"),
    ).reset_index()
    rows = []
    for axis_code, axis_name in [("c", "contingent"), ("f", "forward"), ("b", "beliefs")]:
        for metric in ["truthful_rate", "overbid_rate", "underbid_rate", "mean_ratio"]:
            by_lvl = strain.groupby(axis_code)[metric].mean().to_dict()
            rows.append({
                "axis": axis_name,
                "metric": metric,
                "level0": by_lvl.get(0),
                "level1": by_lvl.get(1),
                "level2": by_lvl.get(2),
                "delta_l2_l0": by_lvl.get(2) - by_lvl.get(0) if 0 in by_lvl and 2 in by_lvl else None,
            })
    return pd.DataFrame(rows)


def audit_by_c(df):
    rows = []
    for c in [0, 1, 2]:
        sub = df[df["c"] == c]
        rows.append({
            "c_level": c,
            "n": len(sub),
            "dominance_invocation_rate": sub["dominance_invocation"].mean() if "dominance_invocation" in df.columns else None,
            "truthful_rate": sub["truthful"].mean(),
            "underbid_rate": sub["underbid"].mean(),
        })
    return pd.DataFrame(rows)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    panels = {}
    for name, rel in INPUTS.items():
        path = os.path.join(REPO_ROOT, rel)
        if not os.path.isfile(path):
            print(f"[warn] skipping {name}: {rel} not found")
            continue
        panels[name] = load_panel(path)
        print(f"[loaded] {name}: rows={len(panels[name])}")

    # Stacked axis marginals
    parts = []
    for name, df in panels.items():
        m = axis_marginals(df)
        m["panel"] = name
        parts.append(m)
    marginals = pd.concat(parts)
    marginals_out = os.path.join(OUT_DIR, "panel_comparison_axis_marginals.csv")
    marginals.to_csv(marginals_out, index=False)
    print(f"\nwrote {marginals_out}")

    # Pretty: truthful-rate Δ across panels
    pivot = (
        marginals[marginals["metric"] == "truthful_rate"]
        .pivot_table(index="axis", columns="panel", values="delta_l2_l0")
        .reset_index()
    )
    table_out = os.path.join(OUT_DIR, "panel_comparison_construct_table.csv")
    pivot.to_csv(table_out, index=False)
    print(f"wrote {table_out}")
    print("\nTruthful-rate Delta (L2 - L0) by axis and panel:")
    print(pivot.to_string(index=False))

    # Audit comparison
    audit_parts = []
    for name, df in panels.items():
        a = audit_by_c(df)
        a["panel"] = name
        audit_parts.append(a)
    audit = pd.concat(audit_parts)
    audit_out = os.path.join(OUT_DIR, "panel_comparison_audit.csv")
    audit.to_csv(audit_out, index=False)
    print(f"\nwrote {audit_out}")
    print("\nDominance-invocation by c-level and panel:")
    pivot2 = audit.pivot_table(
        index="c_level", columns="panel", values="dominance_invocation_rate"
    )
    print(pivot2.to_string())
    print("\nTruthful-rate by c-level and panel:")
    pivot3 = audit.pivot_table(
        index="c_level", columns="panel", values="truthful_rate"
    )
    print(pivot3.to_string())


if __name__ == "__main__":
    main()
