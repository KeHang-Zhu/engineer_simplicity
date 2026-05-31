"""
Combine the EA unrestricted IPV results with the hand-built CV/APV
mechanism data to produce a single LLM-vs-human ranking table aligned
with llm-auction Figure 1.
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

WEIGHTS_FILE = os.path.join(REPO_ROOT, "analysis_v2/cot_v2/mixture_weights_li2017.csv")
MODULES_DIR = os.path.join(REPO_ROOT, "data_v2/results/smad_scoring_cv_apv")

CV_APV_LOG = os.path.join(REPO_ROOT, "data_v2/experiment_logs/gpt5mini")
EA_RANKING = os.path.join(REPO_ROOT, "analysis_v2/cot_v2/ea_unrestricted_smad_ranking.csv")
MANIFEST = os.path.join(REPO_ROOT, "config_v2/configs_auction/cv_apv_panel/_manifest.csv")

OUT_PER_STRAIN = os.path.join(REPO_ROOT, "analysis_v2/cot_v2/figure1_recovery_per_strain.csv")
OUT_GROUPED = os.path.join(REPO_ROOT, "analysis_v2/cot_v2/figure1_recovery_grouped.csv")


HUMAN_SMAD = {
    ("CV", "first_price"):   48.0,
    ("IPV", "first_price"):  25.0,
    ("CV", "second_price"):  18.0,
    ("APV", "second_price"):  9.0,
    ("IPV", "second_price"):  5.0,
}


CV_APV_TO_CELL = {
    "fpsb_apv": ("APV", "first_price"),
    "spsb_apv": ("APV", "second_price"),
    "fpsb_cv":  ("CV",  "first_price"),
    "spsb_cv":  ("CV",  "second_price"),
}


def _collect_pairs(dirname):
    pairs = []
    paths = sorted(glob.glob(os.path.join(CV_APV_LOG, dirname, "result_1_*.json")))
    for p in paths:
        with open(p) as fp:
            data = json.load(fp)
        for _, rd in data.items():
            values = rd.get("value", [])
            bids = [b["bid"] for b in rd.get("history", {}).get("bidding history", [])]
            pairs.extend(zip(values, bids))
    return pairs


def main():
    os.makedirs(MODULES_DIR, exist_ok=True)
    weights = pd.read_csv(WEIGHTS_FILE).set_index("persona_id")["weight"].to_dict()
    manifest = pd.read_csv(MANIFEST)

    rows = []
    for _, r in manifest.iterrows():
        mech_id = r["mechanism_id"]
        value_model, payment_rule = CV_APV_TO_CELL[mech_id]
        persona_id = r["persona_id"]
        dirname = f"cv_apv_{mech_id}__{persona_id}"
        pairs = _collect_pairs(dirname)
        # Write SMAD module
        fake = {
            "experiment": {"name": mech_id},
            "mechanism": {"payment_rule": payment_rule},
            "target_policy": {"benchmark": r["benchmark"]},
        }
        mod_path = write_smad_module(fake, MODULES_DIR, mechanism_id=mech_id)
        mod = _load_module(mod_path)
        smad = mod.smad(pairs)
        rows.append({
            "mech_id": mech_id,
            "value_model": value_model,
            "payment_rule": payment_rule,
            "persona_id": persona_id,
            "n_pairs": len(pairs),
            "smad": smad,
            "weight": weights.get(persona_id),
        })

    per_strain = pd.DataFrame(rows)
    per_strain.to_csv(OUT_PER_STRAIN, index=False)
    print(f"wrote {OUT_PER_STRAIN}")
    print(per_strain.to_string(index=False))

    # Aggregate per mechanism
    cv_apv_summary = []
    for mech_id, g in per_strain.groupby("mech_id"):
        valid = g.dropna(subset=["smad", "weight"])
        ws = valid["weight"].astype(float)
        ss = valid["smad"].astype(float)
        ws_norm = ws / ws.sum() if ws.sum() > 0 else ws
        weighted = float((ws_norm * ss).sum())
        cv_apv_summary.append({
            "value_model": g["value_model"].iloc[0],
            "payment_rule": g["payment_rule"].iloc[0],
            "source": "hand-built",
            "n_mechs": 1,
            "llm_weighted_smad_mean": weighted,
            "llm_weighted_smad_min": weighted,
            "llm_weighted_smad_max": weighted,
        })

    # Load EA IPV results
    ea_ranking = pd.read_csv(EA_RANKING)
    ea_summary = (
        ea_ranking.groupby(["value_model", "payment_rule"])
        .agg(
            n_mechs=("mech_id", "count"),
            llm_weighted_smad_mean=("weighted_smad_Li2017", "mean"),
            llm_weighted_smad_min=("weighted_smad_Li2017", "min"),
            llm_weighted_smad_max=("weighted_smad_Li2017", "max"),
        )
        .reset_index()
    )
    ea_summary["source"] = "EA unrestricted"

    combined = pd.concat([
        pd.DataFrame(cv_apv_summary),
        ea_summary
    ], ignore_index=True)
    combined["human_smad_figure1"] = combined.apply(
        lambda r: HUMAN_SMAD.get((r["value_model"], r["payment_rule"])), axis=1
    )
    combined = combined.sort_values("llm_weighted_smad_mean", ascending=False)
    combined.to_csv(OUT_GROUPED, index=False)
    print(f"\nwrote {OUT_GROUPED}")
    print("\nCombined LLM ranking vs Figure 1:")
    print(combined.to_string(index=False))

    # Rank correlation diagnostic where both LLM and human exist
    sub = combined.dropna(subset=["human_smad_figure1"]).copy()
    if len(sub) >= 2:
        from scipy.stats import spearmanr
        rho, p = spearmanr(sub["llm_weighted_smad_mean"], sub["human_smad_figure1"])
        print(f"\nSpearman rank correlation (LLM SMAD vs human SMAD) = {rho:.3f}  p={p:.3f}")


if __name__ == "__main__":
    main()
