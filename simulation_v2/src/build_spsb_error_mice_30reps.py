"""
Emit fresh SPSB configs for the 2 error mice at 30 reps (matching the
v2 CoT round-2 scale), so the moment vector for moment-matching against
Li (2017) is computed on the same scale across all 5 cluster-panel strains.

Outputs:
    config_v2/configs_auction/interventions_gpt5mini_v2round2/
        spsb_error_second_price_overgeneralizer.yaml
        spsb_error_payment_panic_near_zero.yaml

Run from repo root:
    ./venv/bin/python simulation_v2/src/build_spsb_error_mice_30reps.py
"""

import os
import textwrap

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
OUT_DIR = os.path.join(
    REPO_ROOT, "config_v2/configs_auction/interventions_gpt5mini_v2round2"
)

MICE = [
    "spsb_error_second_price_overgeneralizer",
    "spsb_error_payment_panic_near_zero",
]


def make_yaml(mouse_id, repetitions=30):
    persona_rel = f"rule_prompt_v2/personas/spsb_error_modes/{mouse_id}.txt"
    return textwrap.dedent(f"""\
        # SPSB IPV -- error-mouse {mouse_id} (v2 round-2 scale: 30 reps)
        experiment:
          name: "{mouse_id}_v2r2"
          version: "v2-round2-error-mice"
          description: "SPSB IPV with error mouse {mouse_id} at the v2 round-2 scale"

        auction:
          number_agents: 3
          rounds: 1

        rule:
          seal_clock: "seal"
          ascend_descend: "ascend"
          price_order: "second"
          private_value: "private"
          open_blind: "open"
          closing: false
          reserve_price: 0
          special_name: "private_second_price.txt"

        value:
          common_range: [0, 29]
          private_range: 49
          increment: 0.1
          seed_base: 9301

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
          output_dir: "data_v2/experiment_logs/gpt5mini/{mouse_id}_v2r2"
        """)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    for mouse in MICE:
        path = os.path.join(OUT_DIR, f"{mouse}.yaml")
        with open(path, "w") as fp:
            fp.write(make_yaml(mouse))
        print(f"wrote {path}")


if __name__ == "__main__":
    main()
