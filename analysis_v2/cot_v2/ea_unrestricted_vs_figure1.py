"""
Compute SMAD ranking from the unrestricted-scope EA run and compare to the
human SMAD ranking in Figure 1 of llm-auction (Zhu et al.):
    FP-CV (~48) >> FP-IPV (~25) > SP-CV (~18) > SP-APV (~9) > SP-IPV (~5) > AC (~3)

For each EA-evaluated mechanism:
  - Read its yaml to recover (value_model, payment_rule).
  - Auto-emit a per-mechanism SMAD module with the closest closed-form
    benchmark for that (value_model, payment_rule). The CV mechanisms use
    b*(v) = v as a proxy (no clean closed-form), so their SMAD numbers are
    interpretable as deviation from truthful bidding, not from the BNE.
  - Compute SMAD per panel strain and aggregate (worst, mean, Li-weighted).

Outputs:
    analysis_v2/cot_v2/ea_unrestricted_smad_per_strain.csv
    analysis_v2/cot_v2/ea_unrestricted_smad_ranking.csv
    analysis_v2/cot_v2/ea_unrestricted_vs_figure1.csv  (grouped by cell)
"""

import glob
import json
import os
import sys

import pandas as pd
import yaml

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(REPO_ROOT, "simulation_v2", "src", "evolutionary"))
from smad_scoring import write_smad_module, _load_module  # type: ignore  # noqa: E402

EA_ROOT = os.path.join(REPO_ROOT, "data_v2/results/ea_unrestricted_r2")
WEIGHTS_FILE = os.path.join(REPO_ROOT, "analysis_v2/cot_v2/mixture_weights_li2017.csv")
MODULES_DIR = os.path.join(REPO_ROOT, "data_v2/results/smad_scoring_ea_unrestricted")

PER_STRAIN_OUT = os.path.join(REPO_ROOT, "analysis_v2/cot_v2/ea_unrestricted_smad_per_strain.csv")
RANKING_OUT = os.path.join(REPO_ROOT, "analysis_v2/cot_v2/ea_unrestricted_smad_ranking.csv")
GROUPED_OUT = os.path.join(REPO_ROOT, "analysis_v2/cot_v2/ea_unrestricted_vs_figure1.csv")


# Human SMAD reference from llm-auction Figure 1.
HUMAN_SMAD = {
    ("CV", "first_price"):   48.0,
    ("IPV", "first_price"):  25.0,
    ("CV", "second_price"):  18.0,
    ("APV", "second_price"):  9.0,
    ("APV", "first_price"):  None,  # not in Figure 1 -- treat as N/A
    ("IPV", "second_price"):  5.0,
    ("IPV", "third_price"):  None,  # not in Figure 1
}


# Benchmark expressions per (value_model, payment_rule).
def benchmark_expr(value_model, payment_rule):
    if payment_rule == "first_price":
        return "((3 - 1) / 3) * v"           # (n-1)/n * v
    if payment_rule == "second_price":
        return "v"
    if payment_rule == "third_price":
        return "((3 - 1) / (3 - 2)) * v"     # (n-1)/(n-2) * v
    if payment_rule in ("all_pay", "allpay"):
        return "v ** 3 / 49 ** 2"
    return "v"


def find_mechanisms():
    out = []
    for yaml_path in sorted(glob.glob(os.path.join(EA_ROOT, "**/*.yaml"), recursive=True)):
        fit_path = yaml_path.replace(".yaml", ".fitness.json")
        if not os.path.isfile(fit_path):
            continue
        out.append((yaml_path, fit_path))
    return out


def main():
    os.makedirs(MODULES_DIR, exist_ok=True)
    weights = pd.read_csv(WEIGHTS_FILE).set_index("persona_id")["weight"].to_dict()
    mechs = find_mechanisms()
    print(f"Found {len(mechs)} evaluated mechanisms")

    per_strain = []
    summary = []
    for yaml_path, fit_path in mechs:
        with open(yaml_path) as fp:
            cfg = yaml.safe_load(fp)
        with open(fit_path) as fp:
            fitness = json.load(fp)

        # value_model and payment_rule from rule.private_value + rule.price_order
        # (these are the runner-side overrides set by the genotype phenotype).
        rule = cfg.get("rule", {})
        price = rule.get("price_order", "first")
        pv = rule.get("private_value", "private")
        value_model = {"private": "IPV", "affiliated": "APV", "common": "CV"}.get(pv, "IPV")
        payment_rule = {"first": "first_price", "second": "second_price",
                        "third": "third_price", "allpay": "all_pay"}.get(price, "first_price")

        mech_id = cfg.get("experiment", {}).get("name", os.path.basename(yaml_path))
        fake = {
            "experiment": {"name": mech_id},
            "mechanism": {"payment_rule": payment_rule},
            "target_policy": {"benchmark": benchmark_expr(value_model, payment_rule)},
        }
        mod_path = write_smad_module(fake, MODULES_DIR, mechanism_id=mech_id)
        mod = _load_module(mod_path)

        by_persona = {}
        for r in fitness.get("per_run", []):
            pid = r.get("persona")
            for v, b in zip(r.get("values", []), r.get("bids", [])):
                by_persona.setdefault(pid, []).append((v, b))

        smads = []
        for persona_id, pairs in by_persona.items():
            smad = mod.smad(pairs)
            per_strain.append({
                "mech_id": mech_id,
                "value_model": value_model,
                "payment_rule": payment_rule,
                "persona_id": persona_id,
                "n_pairs": len(pairs),
                "smad": smad,
                "weight": weights.get(persona_id),
            })
            if smad is not None and weights.get(persona_id) is not None:
                smads.append((persona_id, smad, weights[persona_id]))
        if not smads:
            continue
        ws = pd.Series([w for _, _, w in smads])
        ss = pd.Series([s for _, s, _ in smads])
        ws_norm = ws / ws.sum() if ws.sum() > 0 else ws
        weighted = float((ws_norm * ss).sum())
        worst = float(ss.max())
        mean = float(ss.mean())
        gen = os.path.basename(os.path.dirname(yaml_path))
        summary.append({
            "mech_id": mech_id,
            "value_model": value_model,
            "payment_rule": payment_rule,
            "generation": gen,
            "weighted_smad_Li2017": weighted,
            "worst_strain_smad": worst,
            "mean_smad": mean,
            "n_strains": len(smads),
        })

    per_strain_df = pd.DataFrame(per_strain)
    ranking_df = pd.DataFrame(summary).sort_values("weighted_smad_Li2017")
    per_strain_df.to_csv(PER_STRAIN_OUT, index=False)
    ranking_df.to_csv(RANKING_OUT, index=False)

    print(f"\nwrote {PER_STRAIN_OUT}")
    print(f"wrote {RANKING_OUT}")

    # Group by (value_model, payment_rule); compute mean Li-weighted SMAD.
    grouped = ranking_df.groupby(["value_model", "payment_rule"]).agg(
        n_mechs=("mech_id", "count"),
        llm_weighted_smad_mean=("weighted_smad_Li2017", "mean"),
        llm_weighted_smad_min=("weighted_smad_Li2017", "min"),
        llm_weighted_smad_max=("weighted_smad_Li2017", "max"),
    ).reset_index()
    grouped["human_smad_figure1"] = grouped.apply(
        lambda r: HUMAN_SMAD.get((r["value_model"], r["payment_rule"])), axis=1
    )
    grouped = grouped.sort_values("llm_weighted_smad_mean", ascending=False)
    grouped.to_csv(GROUPED_OUT, index=False)
    print(f"wrote {GROUPED_OUT}")

    print("\nLLM ranking (grouped by cell, mean Li-weighted SMAD):")
    print(grouped.to_string(index=False))

    print("\nHuman ranking (from Figure 1):")
    rows = [(k[0], k[1], v) for k, v in HUMAN_SMAD.items() if v is not None]
    rows.sort(key=lambda r: -r[2])
    for vm, pr, smad in rows:
        print(f"  {vm:3s} {pr:14s}  human SMAD = {smad:.1f}")


if __name__ == "__main__":
    main()
