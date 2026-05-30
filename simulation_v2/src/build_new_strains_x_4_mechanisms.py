"""
For each of the 4 new candidate mechanisms (E1, E2, H1, H2), build configs
that pair the mechanism with each of the 2 replacement error strains
(loss_averse, win_seeker). 4 x 2 = 8 configs at 15 reps each.

This is the supplement to build_new_mechanisms_cluster_panel.py: we already
have data for {c0, c2, outcome_truthful} x {E1, E2, H1, H2}; this adds the
two replacement error mice so the new cluster panel covers all 5 strains
x 4 mechanisms.

Outputs:
    config_v2/configs_auction/new_mechanisms_cluster/<mech>__<strain>.yaml
        for strain in {spsb_error_loss_averse, spsb_error_win_seeker}.

Run from repo root:
    ./venv/bin/python simulation_v2/src/build_new_strains_x_4_mechanisms.py
"""

import csv
import os
import sys
import textwrap

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
CONFIGS_DIR = os.path.join(
    REPO_ROOT, "config_v2/configs_auction/new_mechanisms_cluster"
)
MANIFEST = os.path.join(CONFIGS_DIR, "_manifest.csv")

# Pull mechanism specs from the existing manifest (which already covers the
# 4 mechanisms x 5 panel strains).
EXISTING_MANIFEST = pd_read_csv = None  # noqa: F841 - placeholder

import pandas as pd  # noqa: E402


def load_mechanisms_from_manifest(path):
    df = pd.read_csv(path)
    # Reduce to per-mechanism (drop persona-specific columns)
    keep = df.drop_duplicates(subset=["mechanism_id"])[
        ["mechanism_id", "mechanism_label", "category", "benchmark"]
    ].to_dict(orient="records")
    return keep


# Per-mechanism payment-rule / template (must match the existing builder).
MECH_PARAMS = {
    "vickrey_baseline":        ("second", 0, "vickrey_baseline.txt", "truthful_bid"),
    "vickrey_osp_hint":        ("second", 0, "vickrey_osp_hint.txt", "truthful_bid"),
    "allpay_first_price":      ("allpay", 0, "allpay_first_price.txt", "target_bid_value_ratio"),
    "firstprice_with_reserve": ("first", 15, "firstprice_with_reserve.txt", "shade_below_value"),
}


NEW_STRAINS = [
    {
        "persona_id": "spsb_error_loss_averse",
        "persona_file": "rule_prompt_v2/personas/spsb_error_modes/spsb_error_loss_averse.txt",
    },
    {
        "persona_id": "spsb_error_win_seeker",
        "persona_file": "rule_prompt_v2/personas/spsb_error_modes/spsb_error_win_seeker.txt",
    },
]


def make_yaml(mech, strain, repetitions=15):
    out_name = f"{mech['mechanism_id']}__{strain['persona_id']}"
    price_order, reserve, template, action = MECH_PARAMS[mech["mechanism_id"]]
    return textwrap.dedent(f"""\
        # New-mechanisms cluster-panel run -- replacement error strain
        # mechanism: {mech['mechanism_label']}
        # persona:   {strain['persona_id']}

        experiment:
          name: "{out_name}"
          version: "v2-newmech-cluster-v2"
          description: "{mech['mechanism_label']} -- panel strain {strain['persona_id']}"

        mechanism_metadata:
          incentive_target:
            action: "{action}"
            description: "{mech['mechanism_label']}"
            parameters:
              benchmark: "{mech['benchmark']}"

        auction:
          number_agents: 3
          rounds: 1

        rule:
          seal_clock: "seal"
          ascend_descend: "ascend"
          price_order: "{price_order}"
          private_value: "private"
          open_blind: "open"
          closing: false
          reserve_price: {reserve}
          special_name: "{template}"

        value:
          common_range: [0, 29]
          private_range: 49
          increment: 0.1
          seed_base: 9601

        llm:
          model: "gpt-5.4-mini"
          temperature: 0.5

        prompt:
          strategy_type: "plan_reflection"
          prompt_dir: "Prompt/"
          rule_template_dir: "rule_template/auctions/"
          persona_file: "{strain['persona_file']}"
          include_payment_example: true

        execution:
          repetitions: {repetitions}
          parallel: true
          max_workers: 4
          output_dir: "data_v2/experiment_logs/gpt5mini/newmech_{out_name}"
        """)


def main(repetitions=15):
    mechanisms = load_mechanisms_from_manifest(MANIFEST)
    os.makedirs(CONFIGS_DIR, exist_ok=True)
    added = []
    for mech in mechanisms:
        for strain in NEW_STRAINS:
            out_name = f"{mech['mechanism_id']}__{strain['persona_id']}"
            path = os.path.join(CONFIGS_DIR, f"{out_name}.yaml")
            with open(path, "w") as fp:
                fp.write(make_yaml(mech, strain, repetitions))
            added.append({
                "mechanism_id": mech["mechanism_id"],
                "mechanism_label": mech["mechanism_label"],
                "category": mech["category"],
                "persona_id": strain["persona_id"],
                "persona_file": strain["persona_file"],
                "benchmark": mech["benchmark"],
                "config_file": (
                    f"config_v2/configs_auction/new_mechanisms_cluster/{out_name}.yaml"
                ),
            })

    # Append to existing manifest
    existing = pd.read_csv(MANIFEST)
    new_rows = pd.DataFrame(added)
    combined = pd.concat([existing, new_rows], ignore_index=True)
    combined.to_csv(MANIFEST, index=False)
    print(f"appended {len(added)} configs to {CONFIGS_DIR}")
    print(f"manifest now has {len(combined)} rows")


if __name__ == "__main__":
    main()
