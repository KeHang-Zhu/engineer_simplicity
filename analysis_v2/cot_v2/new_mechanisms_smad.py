"""
Score the 4 new candidate mechanisms by weighted-SMAD against the 5-strain
cluster panel, using mixture weights fitted to Li (2017)'s 2P moments.

Pipeline:
  1. Read raw decisions from data_v2/experiment_logs/gpt5mini/newmech_*
  2. For each mechanism, write a per-mechanism SMAD module that encodes the
     mechanism's b*(v) formula.
  3. Compute SMAD per (mechanism, strain).
  4. Aggregate to weighted SMAD per mechanism using the Li-fitted weights.
  5. Also report the worst-strain SMAD and the strain-average SMAD as
     transparent alternatives.

Outputs:
  data_v2/results/smad_scoring_new_mechs/<mech>.py        (auto-emitted modules)
  analysis_v2/cot_v2/newmech_smad_per_strain.csv
  analysis_v2/cot_v2/newmech_smad_ranking.csv
"""

import glob
import json
import os
import sys

import pandas as pd

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(REPO_ROOT, "simulation_v2", "src", "evolutionary"))
from smad_scoring import write_smad_module, _load_module  # type: ignore  # noqa: E402

LOG_BASE = os.path.join(REPO_ROOT, "data_v2/experiment_logs/gpt5mini")
MANIFEST = os.path.join(
    REPO_ROOT,
    "config_v2/configs_auction/new_mechanisms_cluster/_manifest.csv",
)
WEIGHTS_FILE = os.path.join(
    REPO_ROOT, "analysis_v2/cot_v2/mixture_weights_li2017.csv"
)

MODULES_DIR = os.path.join(REPO_ROOT, "data_v2/results/smad_scoring_new_mechs")
PER_STRAIN_OUT = os.path.join(REPO_ROOT, "analysis_v2/cot_v2/newmech_smad_per_strain.csv")
RANKING_OUT = os.path.join(REPO_ROOT, "analysis_v2/cot_v2/newmech_smad_ranking.csv")


def _iter(strain_dir):
    for path in sorted(glob.glob(os.path.join(strain_dir, "result_1_*.json"))):
        with open(path) as fp:
            yield json.load(fp)


def collect_pairs(name):
    """Pull (value, bid) pairs from newmech_<name>/result_1_*.json."""
    pairs = []
    for payload in _iter(os.path.join(LOG_BASE, f"newmech_{name}")):
        for _, rd in payload.items():
            values = rd.get("value", [])
            bids = [b["bid"] for b in rd.get("history", {}).get("bidding history", [])]
            for v, b in zip(values, bids):
                pairs.append((v, b))
    return pairs


def main():
    os.makedirs(MODULES_DIR, exist_ok=True)
    os.makedirs(os.path.dirname(PER_STRAIN_OUT), exist_ok=True)

    manifest = pd.read_csv(MANIFEST)
    weights = pd.read_csv(WEIGHTS_FILE).set_index("persona_id")["weight"].to_dict()

    # One SMAD module per mechanism.
    mech_modules = {}
    for mech_id, group in manifest.groupby("mechanism_id"):
        benchmark = group["benchmark"].iloc[0]
        fake_genotype = {
            "experiment": {"name": mech_id},
            "mechanism": {"payment_rule": "first_price"},  # unused since benchmark provided
            "target_policy": {"benchmark": benchmark},
        }
        path = write_smad_module(fake_genotype, MODULES_DIR, mechanism_id=mech_id)
        mech_modules[mech_id] = path
        print(f"[smad-module] {mech_id} -> {os.path.relpath(path, REPO_ROOT)}")

    rows = []
    for _, r in manifest.iterrows():
        run_name = f"{r['mechanism_id']}__{r['persona_id']}"
        pairs = collect_pairs(run_name)
        mod = _load_module(mech_modules[r["mechanism_id"]])
        smad = mod.smad(pairs)
        rows.append({
            "mechanism_id": r["mechanism_id"],
            "mechanism_label": r["mechanism_label"],
            "category": r["category"],
            "persona_id": r["persona_id"],
            "n_decisions": len(pairs),
            "smad": smad,
            "weight": weights.get(r["persona_id"]),
        })
    per_strain = pd.DataFrame(rows)
    per_strain.to_csv(PER_STRAIN_OUT, index=False)
    print(f"\nwrote {PER_STRAIN_OUT}")
    print("\nSMAD per (mechanism, strain):")
    print(per_strain.to_string(index=False))

    # Aggregate
    ranking = []
    for (mid, mlabel, cat), g in per_strain.groupby(["mechanism_id", "mechanism_label", "category"]):
        valid = g.dropna(subset=["smad"])
        if valid.empty:
            continue
        w = valid["weight"].astype(float).fillna(0)
        smads = valid["smad"].astype(float)
        # Weights may not sum to 1 if some strain SMAD is None; renormalize
        w_norm = w / w.sum() if w.sum() > 0 else w
        weighted = float((w_norm * smads).sum())
        worst = float(smads.max())
        mean = float(smads.mean())
        ranking.append({
            "mechanism_id": mid,
            "mechanism_label": mlabel,
            "category": cat,
            "weighted_smad_Li2017": weighted,
            "worst_strain_smad": worst,
            "mean_smad": mean,
            "n_strains_scored": len(valid),
        })
    ranking_df = pd.DataFrame(ranking).sort_values("weighted_smad_Li2017")
    ranking_df.to_csv(RANKING_OUT, index=False)
    print(f"\nwrote {RANKING_OUT}")
    print("\nRanking (lower SMAD = bidder closer to mechanism's own optimum):")
    print(ranking_df.to_string(index=False))


if __name__ == "__main__":
    main()
