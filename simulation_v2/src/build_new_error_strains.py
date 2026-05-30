"""
Build SPSB IPV YAML configs for the two replacement error mice
(loss_averse, win_seeker) at the v2 round-2 scale (30 reps).

Outputs:
    config_v2/configs_auction/interventions_gpt5mini_v2round2/
        spsb_error_loss_averse.yaml
        spsb_error_win_seeker.yaml
    Logs land in data_v2/experiment_logs/gpt5mini/spsb_error_<id>_v2r2/

Run from repo root:
    ./venv/bin/python simulation_v2/src/build_new_error_strains.py
"""

import os
import textwrap

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
OUT_DIR = os.path.join(
    REPO_ROOT, "config_v2/configs_auction/interventions_gpt5mini_v2round2"
)

STRAINS = ["spsb_error_loss_averse", "spsb_error_win_seeker"]


def make_yaml(strain_id, repetitions=30):
    persona_rel = f"rule_prompt_v2/personas/spsb_error_modes/{strain_id}.txt"
    return textwrap.dedent(f"""\
        # SPSB IPV -- replacement error mouse {strain_id} (v2 round-2 scale)
        experiment:
          name: "{strain_id}_v2r2"
          version: "v2-round2-error-mice-v2"
          description: "SPSB IPV with replacement error mouse {strain_id}"

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
          seed_base: 9501

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
          output_dir: "data_v2/experiment_logs/gpt5mini/{strain_id}_v2r2"
        """)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    for strain in STRAINS:
        path = os.path.join(OUT_DIR, f"{strain}.yaml")
        with open(path, "w") as fp:
            fp.write(make_yaml(strain))
        print(f"wrote {path}")


if __name__ == "__main__":
    main()
