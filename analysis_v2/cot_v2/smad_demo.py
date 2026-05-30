"""
Demonstrate the SMAD-only EA fitness on the data we already have.

For each of two mechanisms (SPSB IPV, FPSB IPV), compute SMAD per strain
in the 5-strain cluster panel, then aggregate to worst-strain SMAD
(the EA's new scalar fitness). Each mechanism gets a per-mechanism
.py emitted by write_smad_module.

Outputs:
    data_v2/results/smad_scoring_demo/<mechanism>.py     (the auto-emitted modules)
    analysis_v2/cot_v2/smad_demo_panel.csv               (SMAD per strain per mechanism)
    analysis_v2/cot_v2/smad_demo_worst.csv               (worst-strain SMAD per mechanism)

Run from repo root:
    ./venv/bin/python analysis_v2/cot_v2/smad_demo.py
"""

import json
import os
import sys

import pandas as pd

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(REPO_ROOT, "simulation_v2", "src", "evolutionary"))

from smad_scoring import write_smad_module, _load_module  # type: ignore  # noqa: E402

OUT_MODULES = os.path.join(REPO_ROOT, "data_v2/results/smad_scoring_demo")
OUT_PANEL = os.path.join(REPO_ROOT, "analysis_v2/cot_v2/smad_demo_panel.csv")
OUT_WORST = os.path.join(REPO_ROOT, "analysis_v2/cot_v2/smad_demo_worst.csv")

SPSB_DATA = os.path.join(REPO_ROOT, "data_v2/results/spsb_basis_cot_v2_summary.csv")
SPSB_OUTCOME = os.path.join(REPO_ROOT, "data_v2/results/spsb_outcome_v2_summary.csv")
FPSB_DATA = os.path.join(REPO_ROOT, "data_v2/results/spsb_basis_fpsb_cot_summary.csv")

# v1 error-mouse SPSB summary -- pull from the data_v2/experiment_logs raw
# files; for the demo, we use the FPSB CoT data which already included the
# error mice. For SPSB error-mouse data we re-derive from data_v2/results
# if available.

CLUSTER_PANEL = [
    {
        "persona_id": "c0_f0_b0_cot",
        "label": "c=0 underbidder (CoT)",
        "strain_match": lambda df, mech: df["basis_id"] == "c0_f0_b0" if "basis_id" in df.columns else df["strain_id"] == "c0_f0_b0",
    },
    {
        "persona_id": "c2_f2_b2_cot",
        "label": "c=2 rational (CoT)",
        "strain_match": lambda df, mech: df["basis_id"] == "c2_f2_b2" if "basis_id" in df.columns else df["strain_id"] == "c2_f2_b2",
    },
    {
        "persona_id": "spsb_error_overgeneralizer",
        "label": "overgeneralizer",
        "strain_match": lambda df, mech: df["strain_id"] == "spsb_error_second_price_overgeneralizer" if "strain_id" in df.columns else False,
    },
    {
        "persona_id": "spsb_error_payment_panic",
        "label": "payment panic",
        "strain_match": lambda df, mech: df["strain_id"] == "spsb_error_payment_panic_near_zero" if "strain_id" in df.columns else False,
    },
    {
        "persona_id": "spsb_outcome_truthful",
        "label": "outcome: truthful",
        "strain_match": lambda df, mech: df.get("outcome_id", df.get("strain_id", pd.Series())) == "spsb_outcome_truthful",
    },
]


def _smad_for_pairs(pairs, mod):
    return mod.smad(pairs)


def _load_fpsb_strain(strain_id):
    df = pd.read_csv(FPSB_DATA)
    sub = df[df["strain_id"] == strain_id]
    return list(zip(sub["value"], sub["bid"]))


def _load_spsb_basis(basis_id):
    df = pd.read_csv(SPSB_DATA)
    sub = df[df["basis_id"] == basis_id]
    return list(zip(sub["value"], sub["bid"]))


def _load_spsb_outcome(oid):
    df = pd.read_csv(SPSB_OUTCOME)
    sub = df[df["outcome_id"] == oid]
    return list(zip(sub["value"], sub["bid"]))


def _load_strain_for_mechanism(mechanism, persona_id):
    """Map persona_id to the right data source given the mechanism."""
    if mechanism == "spsb":
        if persona_id == "c0_f0_b0_cot":
            return _load_spsb_basis("c0_f0_b0")
        if persona_id == "c2_f2_b2_cot":
            return _load_spsb_basis("c2_f2_b2")
        if persona_id == "spsb_outcome_truthful":
            return _load_spsb_outcome("spsb_outcome_truthful")
        # error mice not in our SPSB CoT run; mark as missing
        return []
    if mechanism == "fpsb":
        if persona_id == "c0_f0_b0_cot":
            return _load_fpsb_strain("c0_f0_b0")
        if persona_id == "c2_f2_b2_cot":
            return _load_fpsb_strain("c2_f2_b2")
        if persona_id == "spsb_error_overgeneralizer":
            return _load_fpsb_strain("spsb_error_second_price_overgeneralizer")
        if persona_id == "spsb_error_payment_panic":
            return _load_fpsb_strain("spsb_error_payment_panic_near_zero")
        if persona_id == "spsb_outcome_truthful":
            return _load_fpsb_strain("spsb_outcome_truthful")
    return []


def main():
    os.makedirs(OUT_MODULES, exist_ok=True)
    os.makedirs(os.path.dirname(OUT_PANEL), exist_ok=True)

    # Two example mechanism genotypes (minimal) -----------------------------
    mechanisms = {
        "spsb_ipv_continuous": {
            "experiment": {"name": "spsb_ipv_continuous"},
            "mechanism": {"payment_rule": "second_price"},
            "target_policy": {"benchmark": "value"},
        },
        "fpsb_ipv_continuous": {
            "experiment": {"name": "fpsb_ipv_continuous"},
            "mechanism": {"payment_rule": "first_price"},
            "target_policy": {"benchmark": "((n-1)/n)*value"},
        },
    }

    rows = []
    summary_rows = []
    for mech_id, genotype in mechanisms.items():
        module_path = write_smad_module(genotype, OUT_MODULES, mechanism_id=mech_id)
        mod = _load_module(module_path)
        print(f"\n=== {mech_id} -> {module_path} ===")
        mechanism_key = "fpsb" if "fpsb" in mech_id else "spsb"
        per_strain = {}
        for slot in CLUSTER_PANEL:
            pairs = _load_strain_for_mechanism(mechanism_key, slot["persona_id"])
            smad_val = _smad_for_pairs(pairs, mod) if pairs else None
            per_strain[slot["persona_id"]] = smad_val
            rows.append({
                "mechanism": mech_id,
                "persona_id": slot["persona_id"],
                "label": slot["label"],
                "n_decisions": len(pairs),
                "smad": smad_val,
            })
            print(f"  {slot['label']:35s}  n={len(pairs):4d}  SMAD={smad_val}")
        valid = [v for v in per_strain.values() if v is not None]
        worst = max(valid) if valid else None
        mean_smad = (sum(valid) / len(valid)) if valid else None
        fitness = -worst if worst is not None else None
        summary_rows.append({
            "mechanism": mech_id,
            "worst_strain_smad": worst,
            "mean_smad": mean_smad,
            "fitness": fitness,
            "module_path": os.path.relpath(module_path, REPO_ROOT),
        })

    pd.DataFrame(rows).to_csv(OUT_PANEL, index=False)
    pd.DataFrame(summary_rows).to_csv(OUT_WORST, index=False)
    print(f"\nwrote {OUT_PANEL}")
    print(f"wrote {OUT_WORST}")
    print("\nWorst-strain SMAD per mechanism (lower is better):")
    print(pd.DataFrame(summary_rows).to_string(index=False))


if __name__ == "__main__":
    main()
