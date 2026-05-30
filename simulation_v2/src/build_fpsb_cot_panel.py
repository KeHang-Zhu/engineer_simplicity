"""
Build first-price sealed-bid (FPSB) stress-test configs that point to the
v2 CoT panel, the v2 outcome-instruction strains, and the v1 SPSB error mice.

This is the external-validity check for the CoT panel: do strains whose
worked example is in SPSB transfer to a different mechanism (FPSB) in
mechanistically sensible ways?

Inputs:
  rule_prompt_v2/personas/spsb_basis_cot/c*_f*_b*.txt              (27 CoT)
  rule_prompt_v2/personas/spsb_error_modes/spsb_error_*.txt         (2 error mice)
  rule_prompt_v2/personas/spsb_outcome_instructions/spsb_outcome_*.txt (3 outcome)

Outputs:
  config_v2/configs_auction/fpsb_stress_cot/fpsb_cot_<strain>.yaml
  config_v2/configs_auction/fpsb_stress_cot/_manifest.csv

Run from repo root:
    ./venv/bin/python simulation_v2/src/build_fpsb_cot_panel.py
"""

import csv
import os
import textwrap

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
CONFIGS_DIR = os.path.join(
    REPO_ROOT, "config_v2/configs_auction/fpsb_stress_cot"
)
MANIFEST = os.path.join(CONFIGS_DIR, "_fpsb_cot_manifest.csv")


# Strains to evaluate in FPSB.
COT_STRAINS = [f"c{c}_f{f}_b{b}" for c in range(3) for f in range(3) for b in range(3)]
ERROR_MICE = [
    "spsb_error_second_price_overgeneralizer",
    "spsb_error_payment_panic_near_zero",
]
OUTCOME_STRAINS = [
    "spsb_outcome_overbid",
    "spsb_outcome_underbid",
    "spsb_outcome_truthful",
]


def make_yaml(name, persona_rel, repetitions):
    return textwrap.dedent(f"""\
        # FPSB IPV stress test for {name}
        # n=3 risk-neutral BNE bid = (n-1)/n * v = (2/3)v
        # Human regularity: overbidding relative to RNNE

        experiment:
          name: "fpsb_cot_{name}"
          version: "v2-fpsb-cot-stress"
          description: "FPSB IPV stress test on the v2 CoT/error/outcome panel ({name})"

        mechanism_metadata:
          incentive_target:
            action: "shade_below_value"
            description: "Risk-neutral bidders shade toward the (n-1)/n = 2/3 BNE."
            parameters:
              min_bid_value_ratio: 0.4

        auction:
          number_agents: 3
          rounds: 1

        rule:
          seal_clock: "seal"
          ascend_descend: "ascend"
          price_order: "first"
          private_value: "private"
          open_blind: "open"
          closing: false
          reserve_price: 0
          special_name: "private_first_price.txt"

        value:
          common_range: [0, 29]
          private_range: 49
          increment: 0.1
          seed_base: 9201

        llm:
          model: "gpt-5.4-mini"
          temperature: 0.5

        prompt:
          strategy_type: "plan_reflection"
          prompt_dir: "Prompt/"
          rule_template_dir: "rule_template/auctions/"
          persona_file: "{persona_rel}"
          include_payment_example: true

        execution:
          repetitions: {repetitions}
          parallel: true
          max_workers: 4
          output_dir: "data_v2/experiment_logs/gpt5mini/fpsb_cot_{name}"
        """)


def main(repetitions=10):
    os.makedirs(CONFIGS_DIR, exist_ok=True)
    rows = []
    for strain in COT_STRAINS:
        persona_rel = f"rule_prompt_v2/personas/spsb_basis_cot/{strain}.txt"
        with open(os.path.join(CONFIGS_DIR, f"fpsb_cot_{strain}.yaml"), "w") as fp:
            fp.write(make_yaml(strain, persona_rel, repetitions))
        rows.append(
            {"name": strain, "kind": "cot_basis", "persona_file": persona_rel}
        )
    for strain in ERROR_MICE:
        persona_rel = f"rule_prompt_v2/personas/spsb_error_modes/{strain}.txt"
        with open(os.path.join(CONFIGS_DIR, f"fpsb_cot_{strain}.yaml"), "w") as fp:
            fp.write(make_yaml(strain, persona_rel, repetitions))
        rows.append(
            {"name": strain, "kind": "error_mouse", "persona_file": persona_rel}
        )
    for strain in OUTCOME_STRAINS:
        persona_rel = (
            f"rule_prompt_v2/personas/spsb_outcome_instructions/{strain}.txt"
        )
        with open(os.path.join(CONFIGS_DIR, f"fpsb_cot_{strain}.yaml"), "w") as fp:
            fp.write(make_yaml(strain, persona_rel, repetitions))
        rows.append(
            {"name": strain, "kind": "outcome", "persona_file": persona_rel}
        )
    with open(MANIFEST, "w", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {len(rows)} configs to {CONFIGS_DIR}")
    print(f"manifest: {MANIFEST}")


if __name__ == "__main__":
    main()
