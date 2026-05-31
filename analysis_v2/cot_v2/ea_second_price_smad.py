"""
Post-hoc SMAD scoring for the second-price EA run.

Reads the EA output directory, finds every evaluated mechanism (its .yaml +
.fitness.json), uses the auto-emitted SMAD scoring module for each, computes
SMAD per (mechanism, panel strain), and aggregates to weighted SMAD with the
Li-2017 mixture weights.

Outputs:
    analysis_v2/cot_v2/ea_second_price_smad_per_strain.csv
    analysis_v2/cot_v2/ea_second_price_smad_ranking.csv
"""

import argparse
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
MODULES_DIR = os.path.join(REPO_ROOT, "data_v2/results/smad_scoring_ea_second_price")


def find_mechanisms(ea_root):
    """Return list of (yaml_path, fitness_json_path) tuples."""
    out = []
    # The EA writes to <ea_root>/run_*/gen_<n>/. Find all .yaml + .fitness.json
    # pairs underneath ea_root recursively.
    for yaml_path in sorted(glob.glob(os.path.join(ea_root, "**/*.yaml"), recursive=True)):
        fitness_path = yaml_path.replace(".yaml", ".fitness.json")
        if not os.path.isfile(fitness_path):
            continue
        out.append((yaml_path, fitness_path))
    return out


def collect_per_persona_pairs(fitness_json):
    """From a fitness JSON, return {persona_id: [(value, bid), ...]}."""
    with open(fitness_json) as fp:
        data = json.load(fp)
    by_persona = {}
    for r in data.get("per_run", []):
        pid = r.get("persona")
        if pid is None:
            continue
        values = r.get("values", [])
        bids = r.get("bids", [])
        by_persona.setdefault(pid, []).extend(zip(values, bids))
    return by_persona


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ea_root",
                    default="data_v2/results/ea_second_price_r1",
                    help="EA output root containing ea_gen*/ subdirs")
    ap.add_argument("--out_prefix",
                    default="analysis_v2/cot_v2/ea_second_price")
    args = ap.parse_args()

    ea_root = os.path.join(REPO_ROOT, args.ea_root) if not os.path.isabs(args.ea_root) else args.ea_root
    out_prefix = os.path.join(REPO_ROOT, args.out_prefix) if not os.path.isabs(args.out_prefix) else args.out_prefix
    os.makedirs(MODULES_DIR, exist_ok=True)

    weights = pd.read_csv(WEIGHTS_FILE).set_index("persona_id")["weight"].to_dict()
    print(f"Li-2017 weights ({len(weights)} strains): {weights}")

    mechs = find_mechanisms(ea_root)
    print(f"Found {len(mechs)} evaluated mechanisms under {ea_root}")

    rows = []
    summary = []
    for yaml_path, fitness_path in mechs:
        with open(yaml_path) as fp:
            cfg = yaml.safe_load(fp)
        # Construct a "genotype-like" dict for write_smad_module. For second-
        # price the formula is always b*(v) = v.
        mech_id = cfg.get("experiment", {}).get("name", os.path.basename(yaml_path))
        fake_genotype = {
            "experiment": {"name": mech_id},
            "mechanism": {"payment_rule": "second_price"},
            "target_policy": {"benchmark": "value"},
        }
        mod_path = write_smad_module(fake_genotype, MODULES_DIR, mechanism_id=mech_id)
        mod = _load_module(mod_path)

        by_persona = collect_per_persona_pairs(fitness_path)
        smads = {}
        for persona_id, pairs in by_persona.items():
            smad = mod.smad(pairs)
            smads[persona_id] = smad
            rows.append({
                "mechanism_id": mech_id,
                "yaml": os.path.relpath(yaml_path, REPO_ROOT),
                "persona_id": persona_id,
                "n_pairs": len(pairs),
                "smad": smad,
                "weight": weights.get(persona_id),
            })

        # Aggregate weighted / mean / worst for this mechanism.
        valid = [(persona_id, smads[persona_id], weights.get(persona_id))
                 for persona_id in smads if smads[persona_id] is not None and weights.get(persona_id) is not None]
        if not valid:
            continue
        ws = pd.Series([w for _, _, w in valid])
        ss = pd.Series([s for _, s, _ in valid])
        ws_norm = ws / ws.sum() if ws.sum() > 0 else ws
        weighted = float((ws_norm * ss).sum())
        worst = float(ss.max())
        mean = float(ss.mean())
        # Read coordinates for selection
        gen = os.path.basename(os.path.dirname(yaml_path))
        summary.append({
            "mechanism_id": mech_id,
            "generation": gen,
            "yaml": os.path.relpath(yaml_path, REPO_ROOT),
            "weighted_smad_Li2017": weighted,
            "worst_strain_smad": worst,
            "mean_smad": mean,
            "n_strains": len(valid),
            "price_order": cfg.get("rule", {}).get("price_order"),
            "bid_language": cfg.get("mechanism_metadata", {})
                .get("yaml_overrides", {})
                .get("bid_language", "?"),
            "special_name": cfg.get("rule", {}).get("special_name"),
        })

    per_strain_df = pd.DataFrame(rows)
    ranking_df = pd.DataFrame(summary).sort_values("weighted_smad_Li2017")
    per_strain_path = f"{out_prefix}_smad_per_strain.csv"
    ranking_path = f"{out_prefix}_smad_ranking.csv"
    per_strain_df.to_csv(per_strain_path, index=False)
    ranking_df.to_csv(ranking_path, index=False)
    print(f"\nwrote {per_strain_path} ({len(per_strain_df)} rows)")
    print(f"wrote {ranking_path} ({len(ranking_df)} rows)")

    print("\nEA second-price ranking (lower SMAD = closer to b*(v)=v):")
    cols = ["mechanism_id", "generation", "weighted_smad_Li2017",
            "worst_strain_smad", "mean_smad"]
    print(ranking_df[cols].to_string(index=False))


if __name__ == "__main__":
    main()
