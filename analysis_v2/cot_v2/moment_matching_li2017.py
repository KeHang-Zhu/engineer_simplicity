"""
Fit mixture weights over the 5-strain cluster panel to match the human
SPSB moment vector estimated from Li (2017)'s 2P treatment.

Inputs:
    data_v2/results/spsb_basis_cot_v2_summary.csv     (CoT strains)
    data_v2/results/spsb_outcome_v2_summary.csv       (outcome strains)
    data_v2/results/spsb_basis_error_v2r2_summary.csv (error mice, this round)

Target h:
    Estimated from Li (2017) Figure 2, SP treatment standard auctions,
    rescaled from auction-level to per-bidder moments. See notes in the
    file for the exact conversion.

Outputs:
    analysis_v2/cot_v2/li2017_target_moments.csv
    analysis_v2/cot_v2/panel_moments_matrix.csv
    analysis_v2/cot_v2/mixture_weights_li2017.csv

Run from repo root:
    ./venv/bin/python analysis_v2/cot_v2/moment_matching_li2017.py
"""

import os

import numpy as np
import pandas as pd
from scipy.optimize import minimize

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
SPSB_COT_IN = os.path.join(REPO_ROOT, "data_v2/results/spsb_basis_cot_v2_summary.csv")
SPSB_OUTCOME_IN = os.path.join(REPO_ROOT, "data_v2/results/spsb_outcome_v2_summary.csv")
SPSB_ERROR_IN = os.path.join(REPO_ROOT, "data_v2/results/spsb_basis_error_v2r2_summary.csv")

OUT_DIR = os.path.join(REPO_ROOT, "analysis_v2/cot_v2")
TARGET_OUT = os.path.join(OUT_DIR, "li2017_target_moments.csv")
MATRIX_OUT = os.path.join(OUT_DIR, "panel_moments_matrix.csv")
WEIGHTS_OUT = os.path.join(OUT_DIR, "mixture_weights_li2017.csv")


# -- Target moments from Li (2017) -------------------------------------------
#
# Source: Li (2017, AER), Figure 2 standard auctions, SP treatment.
# Histogram of (2nd-highest bid - 2nd-highest value) bin frequencies:
#   <= -$6: 0.13
#   (-$6, -$2]: 0.08
#   (-$2, $2]: 0.39  (near truthful)
#   ($2, $6]: 0.17
#   > $6: 0.22
#
# Conversion to per-bidder moments: the second-highest bid is a robust
# statistic of the bidder population, so the per-bidder distribution has
# roughly the same shape with slightly heavier tails. We adopt the
# auction-level shares as a first-cut per-bidder target, then add a small
# near-zero estimate (Li's data do not directly resolve bids below $1, so
# the near-zero rate is taken from comparable IPV 2P data; we use a
# conservative 0.03).
#
# The moment vector is:
#   m = (p_truthful, p_underbid, p_overbid, p_extreme_overbid, p_near_zero)

LI2017_MOMENTS = {
    "p_truthful":       0.39,
    "p_underbid":       0.21,    # 0.13 + 0.08, combining ≤-$6 and (-$6,-$2]
    "p_overbid":        0.39,    # 0.17 + 0.22, combining ($2,$6] and >$6
    "p_extreme_overbid": 0.22,   # > $6 above value
    "p_near_zero":      0.03,    # conservative estimate; not resolved by Li's figure
}

MOMENT_NAMES = list(LI2017_MOMENTS.keys())


# -- Cluster panel strain definitions ---------------------------------------

CLUSTER_PANEL = [
    {
        "persona_id": "c0_f0_b0_cot",
        "source": "spsb_basis_cot_v2",
        "filter": ("basis_id", "c0_f0_b0"),
    },
    {
        "persona_id": "c2_f2_b2_cot",
        "source": "spsb_basis_cot_v2",
        "filter": ("basis_id", "c2_f2_b2"),
    },
    {
        "persona_id": "spsb_error_overgeneralizer",
        "source": "spsb_error_v2r2",
        "filter": ("outcome_id", "spsb_error_second_price_overgeneralizer"),
    },
    {
        "persona_id": "spsb_error_payment_panic",
        "source": "spsb_error_v2r2",
        "filter": ("outcome_id", "spsb_error_payment_panic_near_zero"),
    },
    {
        "persona_id": "spsb_outcome_truthful",
        "source": "spsb_outcome_v2",
        "filter": ("outcome_id", "spsb_outcome_truthful"),
    },
]


def classify(row):
    v, b = row["value"], row["bid"]
    if pd.isna(v) or pd.isna(b) or v <= 0:
        return "missing"
    tol = 0.05
    if abs(b - v) <= tol:
        return "truthful"
    if b > v + tol:
        return "overbid"
    return "underbid"


def moments_for_pairs(df):
    df = df.copy()
    df["class"] = df.apply(classify, axis=1)
    df = df[df["class"] != "missing"]
    n = len(df)
    if n == 0:
        return None
    truthful = (df["class"] == "truthful").mean()
    underbid = (df["class"] == "underbid").mean()
    overbid = (df["class"] == "overbid").mean()
    extreme_overbid = ((df["bid"] - df["value"]) >= 5).mean()
    near_zero = ((df["bid"] <= 0.1) & (df["value"] > 5)).mean()
    return {
        "p_truthful": truthful,
        "p_underbid": underbid,
        "p_overbid": overbid,
        "p_extreme_overbid": extreme_overbid,
        "p_near_zero": near_zero,
    }


def load_strain(slot):
    src = slot["source"]
    if src == "spsb_basis_cot_v2":
        df = pd.read_csv(SPSB_COT_IN)
        col, val = slot["filter"]
        return df[df[col] == val]
    if src == "spsb_outcome_v2":
        df = pd.read_csv(SPSB_OUTCOME_IN)
        col, val = slot["filter"]
        return df[df[col] == val]
    if src == "spsb_error_v2r2":
        df = pd.read_csv(SPSB_ERROR_IN)
        col, val = slot["filter"]
        return df[df[col] == val]
    raise ValueError(f"unknown source {src}")


def build_panel_matrix():
    rows = []
    for slot in CLUSTER_PANEL:
        df = load_strain(slot)
        m = moments_for_pairs(df)
        if m is None:
            raise SystemExit(f"empty data for {slot['persona_id']}")
        row = {"persona_id": slot["persona_id"], "n": len(df)}
        row.update(m)
        rows.append(row)
    panel = pd.DataFrame(rows)
    return panel


def fit_mixture(panel_df, target):
    """Constrained least squares: w in simplex, minimize ||M^T w - h||^2."""
    M = panel_df[MOMENT_NAMES].values  # 5 x 5
    h = np.array([target[m] for m in MOMENT_NAMES])  # length 5
    J = len(panel_df)

    def objective(w):
        return np.sum((M.T @ w - h) ** 2)

    w0 = np.full(J, 1.0 / J)
    bounds = [(0.0, 1.0)] * J
    constraints = [{"type": "eq", "fun": lambda w: w.sum() - 1.0}]
    result = minimize(objective, w0, method="SLSQP", bounds=bounds,
                      constraints=constraints,
                      options={"ftol": 1e-12, "disp": False})
    if not result.success:
        print(f"[warn] optimizer did not fully converge: {result.message}")
    w_hat = result.x
    fitted = M.T @ w_hat
    residual = float(np.sqrt(np.sum((fitted - h) ** 2)))
    return w_hat, fitted, residual


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    pd.DataFrame([
        {"moment": k, "human_value": v, "source": "Li (2017) Figure 2 SP, adapted"}
        for k, v in LI2017_MOMENTS.items()
    ]).to_csv(TARGET_OUT, index=False)
    print(f"wrote {TARGET_OUT}")

    panel = build_panel_matrix()
    panel.to_csv(MATRIX_OUT, index=False)
    print(f"wrote {MATRIX_OUT}")
    print("\nPanel moments (rows = strains, cols = moments):")
    print(panel.to_string(index=False))

    w_hat, fitted, residual = fit_mixture(panel, LI2017_MOMENTS)

    weights_df = pd.DataFrame({
        "persona_id": panel["persona_id"],
        "weight": w_hat,
    })
    weights_df.to_csv(WEIGHTS_OUT, index=False)
    print(f"\nwrote {WEIGHTS_OUT}")
    print("\nFitted mixture weights (human-calibrated):")
    print(weights_df.to_string(index=False))

    print("\nFitted moments vs Li (2017) target:")
    cmp_df = pd.DataFrame({
        "moment": MOMENT_NAMES,
        "fitted": fitted,
        "target": [LI2017_MOMENTS[m] for m in MOMENT_NAMES],
    })
    cmp_df["residual"] = cmp_df["fitted"] - cmp_df["target"]
    print(cmp_df.to_string(index=False))
    print(f"\nTotal L2 residual: {residual:.4f}")


if __name__ == "__main__":
    main()
