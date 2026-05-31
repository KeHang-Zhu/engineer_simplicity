"""
Combine the EA's 3-persona data with the supplementary 2-persona data to
produce a full 5-strain weighted-SMAD for the 4 selected EA mechanisms.

Inputs:
  data_v2/results/ea_second_price_r1/run_*/gen_*/<mech>.fitness.json  (3 strains)
  data_v2/experiment_logs/gpt5mini/ea_sp_<label>__<persona>/          (2 strains)

Outputs:
  analysis_v2/cot_v2/ea_top4_full_per_strain.csv
  analysis_v2/cot_v2/ea_top4_full_ranking.csv
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

EA_ROOT = os.path.join(REPO_ROOT, "data_v2/results/ea_second_price_r1")
SUPP_LOG = os.path.join(REPO_ROOT, "data_v2/experiment_logs/gpt5mini")
WEIGHTS_FILE = os.path.join(REPO_ROOT, "analysis_v2/cot_v2/mixture_weights_li2017.csv")
MODULES_DIR = os.path.join(REPO_ROOT, "data_v2/results/smad_scoring_ea_top4")

PER_STRAIN_OUT = os.path.join(REPO_ROOT, "analysis_v2/cot_v2/ea_top4_full_per_strain.csv")
RANKING_OUT = os.path.join(REPO_ROOT, "analysis_v2/cot_v2/ea_top4_full_ranking.csv")


# Manual mapping from EA hash to a friendlier label + category.
SELECTED = [
    ("mech_002_second_price_sealed_bid_1eaa7719aca5",
     "E1_easy_value_anchor",
     "easy"),
    ("mech_007_sealed_bid_second_price_truthful_b748a8d9d403",
     "E2_easy_close_to_value",
     "easy"),
    ("mech_001_sealed_bid_second_price_private_values_f54eacba774c",
     "H1_hard_no_nudge_explicit_example",
     "hard"),
    ("mech_000_sealed_second_price_continuous_a6ba0bfb5b52",
     "H2_hard_no_nudge_losing_example",
     "hard"),
]


def _load_ea_pairs(hash_id):
    """From EA fitness JSON, return {persona_id: [(v,b), ...]}."""
    matches = glob.glob(os.path.join(EA_ROOT, f"run_*/gen_*/{hash_id}.fitness.json"))
    if not matches:
        return {}
    with open(matches[0]) as fp:
        data = json.load(fp)
    by_persona = {}
    for r in data.get("per_run", []):
        pid = r.get("persona")
        if pid is None:
            continue
        for v, b in zip(r.get("values", []), r.get("bids", [])):
            by_persona.setdefault(pid, []).append((v, b))
    return by_persona


def _load_supp_pairs(label, persona_id):
    """From supplementary run, return [(v,b), ...]."""
    dirname = f"ea_sp_{label}__{persona_id}"
    paths = sorted(glob.glob(os.path.join(SUPP_LOG, dirname, "result_1_*.json")))
    pairs = []
    for path in paths:
        with open(path) as fp:
            data = json.load(fp)
        for _, rd in data.items():
            values = rd.get("value", [])
            bids = [b["bid"] for b in rd.get("history", {}).get("bidding history", [])]
            pairs.extend(zip(values, bids))
    return pairs


def main():
    os.makedirs(MODULES_DIR, exist_ok=True)
    weights = pd.read_csv(WEIGHTS_FILE).set_index("persona_id")["weight"].to_dict()

    rows = []
    summary = []
    for hash_id, label, category in SELECTED:
        # Write per-mechanism SMAD module (b*(v) = v for second-price).
        fake_genotype = {
            "experiment": {"name": label},
            "mechanism": {"payment_rule": "second_price"},
            "target_policy": {"benchmark": "value"},
        }
        mod_path = write_smad_module(fake_genotype, MODULES_DIR, mechanism_id=label)
        mod = _load_module(mod_path)

        # Collect pairs from EA (3 strains) and supplement (2 strains).
        by_persona = _load_ea_pairs(hash_id)
        for persona_id in ["spsb_outcome_truthful", "spsb_error_loss_averse"]:
            pairs = _load_supp_pairs(label, persona_id)
            if pairs:
                by_persona[persona_id] = pairs

        per_strain_smads = []
        for persona_id, pairs in sorted(by_persona.items()):
            smad = mod.smad(pairs)
            rows.append({
                "mechanism_label": label,
                "category": category,
                "mechanism_hash": hash_id,
                "persona_id": persona_id,
                "n_pairs": len(pairs),
                "smad": smad,
                "weight": weights.get(persona_id),
            })
            if smad is not None and weights.get(persona_id) is not None:
                per_strain_smads.append((persona_id, smad, weights[persona_id]))

        ws = pd.Series([w for _, _, w in per_strain_smads])
        ss = pd.Series([s for _, s, _ in per_strain_smads])
        ws_norm = ws / ws.sum() if ws.sum() > 0 else ws
        weighted = float((ws_norm * ss).sum())
        worst = float(ss.max())
        mean = float(ss.mean())
        summary.append({
            "mechanism_label": label,
            "mechanism_hash": hash_id,
            "category": category,
            "weighted_smad_Li2017": weighted,
            "worst_strain_smad": worst,
            "mean_smad": mean,
            "n_strains": len(per_strain_smads),
        })

    per_strain = pd.DataFrame(rows)
    ranking = pd.DataFrame(summary).sort_values("weighted_smad_Li2017")
    per_strain.to_csv(PER_STRAIN_OUT, index=False)
    ranking.to_csv(RANKING_OUT, index=False)
    print(f"wrote {PER_STRAIN_OUT}")
    print(f"wrote {RANKING_OUT}")
    print("\nPer-strain SMAD:")
    print(per_strain.to_string(index=False))
    print("\nFinal ranking (4 EA-selected mechanisms, full 5-strain panel):")
    print(ranking.to_string(index=False))


if __name__ == "__main__":
    main()
