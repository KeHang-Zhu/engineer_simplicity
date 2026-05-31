"""
Build YAMLs that pair the 4 selected EA second-price mechanisms with the
2 panel strains the original EA loop did not run (loss_averse, outcome_truthful).
The EA used --n_personas 3 so only 3/5 strains have data; this fills in the
gap so weighted-SMAD can be computed over the full Li-2017 cluster panel.

Inputs:
    data_v2/results/ea_second_price_r1/run_*/gen_*/mech_*.yaml (EA outputs)

Outputs:
    config_v2/configs_auction/ea_second_price_supplement/<mech>__<strain>.yaml

Run from repo root:
    ./venv/bin/python simulation_v2/src/build_ea_top_mechs_missing_strains.py
"""

import csv
import glob
import os
import textwrap

import yaml

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
EA_ROOT = os.path.join(REPO_ROOT, "data_v2/results/ea_second_price_r1")
OUT_DIR = os.path.join(REPO_ROOT, "config_v2/configs_auction/ea_second_price_supplement")


# The 4 selected mechanisms from the EA ranking.
SELECTED = [
    ("mech_002_second_price_sealed_bid_1eaa7719aca5",          "E1_easy_value_anchor"),
    ("mech_007_sealed_bid_second_price_truthful_b748a8d9d403", "E2_easy_close_to_value"),
    ("mech_001_sealed_bid_second_price_private_values_f54eacba774c", "H1_hard_no_nudge_explicit_example"),
    ("mech_000_sealed_second_price_continuous_a6ba0bfb5b52",   "H2_hard_no_nudge_losing_example"),
]

# Missing strains (the EA only ran the first 3 of 5 cluster-panel personas).
MISSING = [
    ("spsb_outcome_truthful",
     "rule_prompt_v2/personas/spsb_outcome_instructions/spsb_outcome_truthful.txt"),
    ("spsb_error_loss_averse",
     "rule_prompt_v2/personas/spsb_error_modes/spsb_error_loss_averse.txt"),
]


def find_mech_yaml(hash_id):
    matches = glob.glob(os.path.join(EA_ROOT, f"run_*/gen_*/{hash_id}.yaml"))
    if not matches:
        raise SystemExit(f"could not find {hash_id}.yaml under {EA_ROOT}")
    return matches[0]


def make_yaml(mech_yaml, mech_label, persona_id, persona_rel, repetitions=15):
    with open(mech_yaml) as fp:
        base = yaml.safe_load(fp)
    base["experiment"]["name"] = f"ea_sp_{mech_label}__{persona_id}"
    base["experiment"]["version"] = "v2-ea-second-price-supplement"
    base["execution"]["repetitions"] = repetitions
    base["execution"]["parallel"] = True
    base["execution"]["max_workers"] = 4
    base["execution"]["output_dir"] = (
        f"data_v2/experiment_logs/gpt5mini/ea_sp_{mech_label}__{persona_id}"
    )
    base.setdefault("prompt", {})
    base["prompt"]["persona_file"] = persona_rel
    return base


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    rows = []
    for hash_id, label in SELECTED:
        mech_yaml = find_mech_yaml(hash_id)
        for persona_id, persona_rel in MISSING:
            cfg = make_yaml(mech_yaml, label, persona_id, persona_rel)
            out_name = f"{label}__{persona_id}.yaml"
            out_path = os.path.join(OUT_DIR, out_name)
            with open(out_path, "w") as fp:
                yaml.safe_dump(cfg, fp, sort_keys=False)
            rows.append({
                "mechanism_hash": hash_id,
                "mechanism_label": label,
                "persona_id": persona_id,
                "config_file": (
                    f"config_v2/configs_auction/ea_second_price_supplement/{out_name}"
                ),
            })
    manifest = os.path.join(OUT_DIR, "_manifest.csv")
    with open(manifest, "w", newline="") as fp:
        w = csv.DictWriter(fp, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {len(rows)} configs to {OUT_DIR}")
    print(f"manifest: {manifest}")


if __name__ == "__main__":
    main()
