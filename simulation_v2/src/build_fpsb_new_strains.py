"""Build FPSB IPV configs for win_seeker + loss_averse (the 2 replacement
error mice) so the FPSB stress-test table can be updated with the new
panel."""

import os
import textwrap

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
OUT_DIR = os.path.join(REPO_ROOT, "config_v2/configs_auction/fpsb_stress_cot")


STRAINS = [
    ("spsb_error_win_seeker",
     "rule_prompt_v2/personas/spsb_error_modes/spsb_error_win_seeker.txt"),
    ("spsb_error_loss_averse",
     "rule_prompt_v2/personas/spsb_error_modes/spsb_error_loss_averse.txt"),
]


def make_yaml(name, persona_rel, reps=10):
    return textwrap.dedent(f"""\
        experiment:
          name: "fpsb_cot_{name}"
          version: "v2-fpsb-cot-stress-v2"
          description: "FPSB IPV stress test for {name}"

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
          seed_base: 9701

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
          repetitions: {reps}
          parallel: true
          max_workers: 4
          output_dir: "data_v2/experiment_logs/gpt5mini/fpsb_cot_{name}"
        """)


def main(reps=10):
    os.makedirs(OUT_DIR, exist_ok=True)
    for name, persona in STRAINS:
        path = os.path.join(OUT_DIR, f"fpsb_cot_{name}.yaml")
        with open(path, "w") as fp:
            fp.write(make_yaml(name, persona, reps))
        print(f"wrote {path}")


if __name__ == "__main__":
    main()
