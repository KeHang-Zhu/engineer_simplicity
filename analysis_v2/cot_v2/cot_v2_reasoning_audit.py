"""
Audit the logged reasoning chains in the v2 CoT run: does each strain's
<PLAN> actually use the assigned reasoning style, or does it leak into
the dominance argument that the contingent ablation forbids?

We flag PLAN text for two patterns:
  * dominance_invocation: phrases indicating the model invoked the
    second-price truthful-dominance argument (the move c=0 forbids).
  * own_value_anchor: phrases indicating the model anchored to its
    own value via a margin-of-safety rule (the move the deficits use).

Outputs:
  analysis_v2/cot_v2/spsb_cot_audit.csv
  analysis_v2/cot_v2/spsb_cot_audit_examples.csv
"""

import os
import re

import pandas as pd

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
PANEL_IN = os.path.join(REPO_ROOT, "data_v2/results/spsb_basis_cot_v2_summary.csv")
OUT_DIR = os.path.join(REPO_ROOT, "analysis_v2/cot_v2")
AUDIT_OUT = os.path.join(OUT_DIR, "spsb_cot_audit.csv")
EXAMPLES_OUT = os.path.join(OUT_DIR, "spsb_cot_audit_examples.csv")

# Patterns that indicate the dominance argument has been invoked.
DOMINANCE_PATTERNS = [
    r"\bdominant\b",
    r"\bweakly dominant\b",
    r"\bdominance\b",
    r"\bvickrey\b",
    r"\btruthful(ly)? bid",
    r"\bbid(ding)? (my|the) value\b",
    r"\bbid(ding)? equal to my value\b",
    r"\bbid(ding)? \$?\d+ ?\(?my value\)?\b",
    r"\bsecond[- ]highest bid\b.*(?:not (?:my own|the bidder.?s own) bid|i (?:don.?t|do not) pay (?:my own|the bidder.?s) bid)",
    r"\bonly (?:changes|affects) whether i win",
    r"\bdoes? not (?:change|affect) (?:how much|what) i pay",
    r"\bbid my value\b",
    r"\bvalue is my (?:best|optimal) bid\b",
]

DOMINANCE_RE = re.compile("|".join(DOMINANCE_PATTERNS), re.IGNORECASE)

OWN_VALUE_PATTERNS = [
    r"\bmargin[- ]of[- ]safety\b",
    r"\bconservative anchor\b",
    r"\bsafer (?:lower )?bid\b",
    r"\b\d+\s*%\s*of\s+(?:my\s+)?value\b",
    r"\bsimple cue\b",
    r"\bown value rule\b",
]
OWN_VALUE_RE = re.compile("|".join(OWN_VALUE_PATTERNS), re.IGNORECASE)


def audit(df):
    df = df.copy()
    df["plan"] = df["plan"].fillna("").astype(str)
    df["dominance_invocation"] = df["plan"].str.contains(DOMINANCE_RE).astype(int)
    df["own_value_anchor"] = df["plan"].str.contains(OWN_VALUE_RE).astype(int)
    return df


def per_strain(df):
    grouped = df.groupby(
        ["basis_id", "c", "f", "b", "contingent", "forward", "beliefs"], dropna=False
    )
    out = grouped.agg(
        n=("plan", "size"),
        dominance_invocation_rate=("dominance_invocation", "mean"),
        own_value_anchor_rate=("own_value_anchor", "mean"),
        truthful_rate=("bid", lambda s: ((df.loc[s.index, "bid"] - df.loc[s.index, "value"]).abs() <= 0.05).mean()),
    ).reset_index()
    return out


def per_c_level(audit_df):
    rows = []
    for c in [0, 1, 2]:
        sub = audit_df[audit_df["c"] == c]
        rows.append(
            {
                "contingent_level": c,
                "n_plans": len(sub),
                "dominance_invocation_rate": sub["dominance_invocation"].mean(),
                "own_value_anchor_rate": sub["own_value_anchor"].mean(),
            }
        )
    return pd.DataFrame(rows)


def representative_examples(audit_df, k_per_cell=2):
    rows = []
    # Leaky: c=0 strains that invoke dominance
    leak = audit_df[(audit_df["c"] == 0) & (audit_df["dominance_invocation"] == 1)]
    for _, r in leak.head(k_per_cell).iterrows():
        rows.append(
            {
                "kind": "leak_c0_dominance",
                "basis_id": r["basis_id"],
                "plan": r["plan"][:400],
            }
        )
    # Held: c=0 strains that anchor to own value
    held = audit_df[(audit_df["c"] == 0) & (audit_df["own_value_anchor"] == 1)]
    for _, r in held.head(k_per_cell).iterrows():
        rows.append(
            {
                "kind": "held_c0_anchor",
                "basis_id": r["basis_id"],
                "plan": r["plan"][:400],
            }
        )
    # Engaged: c=2 strains invoking dominance
    eng = audit_df[(audit_df["c"] == 2) & (audit_df["dominance_invocation"] == 1)]
    for _, r in eng.head(k_per_cell).iterrows():
        rows.append(
            {
                "kind": "engaged_c2_dominance",
                "basis_id": r["basis_id"],
                "plan": r["plan"][:400],
            }
        )
    return pd.DataFrame(rows)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    df = pd.read_csv(PANEL_IN)
    audit_df = audit(df)
    strain = per_strain(audit_df)
    c_marginal = per_c_level(audit_df)
    examples = representative_examples(audit_df)

    strain.to_csv(AUDIT_OUT, index=False)
    examples.to_csv(EXAMPLES_OUT, index=False)
    print(f"wrote {AUDIT_OUT}")
    print(f"wrote {EXAMPLES_OUT}")
    print("\nAudit by contingent level:")
    print(c_marginal.to_string(index=False))
    print("\nPer-strain audit (top 5 most-leaking c=0 strains):")
    print(
        strain[strain["c"] == 0]
        .sort_values("dominance_invocation_rate", ascending=False)
        .head(5)
        .to_string(index=False)
    )
    print("\nPer-strain audit (top 5 c=2 strains):")
    print(
        strain[strain["c"] == 2]
        .sort_values("dominance_invocation_rate", ascending=False)
        .head(5)
        .to_string(index=False)
    )


if __name__ == "__main__":
    main()
