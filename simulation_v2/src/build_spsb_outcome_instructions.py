"""
Build outcome-instruction alternative prompts for the SPSB panel (appendix
robustness check).

Each persona simply names the outcome -- "you bid above your value" / "below" /
"equal to your value" -- without prescribing a reasoning procedure. The purpose
is to compare these outcome-instructed strains to the procedure-instructed
CoT strains: a deficit induced procedurally should differ from a deficit
induced by labeling the bid behavior directly, especially on off-target
metrics and on the reasoning chains.

Personas land in rule_prompt_v2/personas/spsb_outcome_instructions/.
Configs land in config_v2/configs_auction/interventions_gpt5mini_outcome/.
Logs land in data_v2/experiment_logs/gpt5mini/spsb_outcome_<id>/.

Run from repo root:
    ./venv/bin/python simulation_v2/src/build_spsb_outcome_instructions.py
"""

import csv
import os
import textwrap

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
PERSONAS_DIR = os.path.join(
    REPO_ROOT, "rule_prompt_v2/personas/spsb_outcome_instructions"
)
CONFIGS_DIR = os.path.join(
    REPO_ROOT, "config_v2/configs_auction/interventions_gpt5mini_outcome"
)
MANIFEST = os.path.join(CONFIGS_DIR, "_outcome_instructions_manifest.csv")

OUTCOMES = [
    {
        "id": "spsb_outcome_overbid",
        "label": "you bid above your value",
        "target_class": "spsb_overbid",
        "instruction": "You bid above your value.",
        "example_30": ("My behavior is to bid above my value.", "33"),
        "example_8": ("My behavior is to bid above my value.", "10"),
    },
    {
        "id": "spsb_outcome_underbid",
        "label": "you bid below your value",
        "target_class": "spsb_underbid",
        "instruction": "You bid below your value.",
        "example_30": ("My behavior is to bid below my value.", "21"),
        "example_8": ("My behavior is to bid below my value.", "5"),
    },
    {
        "id": "spsb_outcome_truthful",
        "label": "you bid equal to your value",
        "target_class": "spsb_truthful",
        "instruction": "You bid equal to your value.",
        "example_30": ("My behavior is to bid equal to my value.", "30"),
        "example_8": ("My behavior is to bid equal to my value.", "8"),
    },
]


PERSONA_HEADER = (
    "You are a Digital Mouse bidder in a single-item auction experiment. "
    "Your job is not to be a perfect auction theorist. Your job is to use "
    "exactly the behavior assigned below, so researchers can test how this "
    "type of bounded behavior compares to a procedure-instructed mouse."
)

GENERAL_RESPONSE_RULE = (
    "General response rule:\n"
    "- Always produce a short public <PLAN> followed by one numeric <ACTION> bid.\n"
    "- The <PLAN> may simply restate that you follow the assigned behavior; do "
    "not silently upgrade into a fuller reasoning style.\n"
    "- If the auction has an allowed bid grid, your <ACTION> must be one of the "
    "allowed grid bids.\n"
    "- If the prompt says this is a second-price sealed-bid auction, the winner "
    "pays the second-highest bid. You still follow only your assigned behavior."
)


def make_persona(outcome):
    plan30, action30 = outcome["example_30"]
    plan8, action8 = outcome["example_8"]
    return (
        f"{PERSONA_HEADER}\n\n"
        f"Behavior profile: {outcome['instruction']}\n\n"
        f"{GENERAL_RESPONSE_RULE}\n\n"
        "Example 1: second-price sealed-bid auction, private value = 30.\n"
        f"<PLAN>{plan30}</PLAN>\n"
        f"<ACTION>{action30}</ACTION>\n\n"
        "Example 2: second-price sealed-bid auction, private value = 8.\n"
        f"<PLAN>{plan8}</PLAN>\n"
        f"<ACTION>{action8}</ACTION>\n\n"
        "At runtime, adapt the examples to the actual private value and the "
        "allowed action set. Keep the same behavior throughout the decision.\n"
    )


def make_yaml(outcome, repetitions):
    persona_rel = (
        f"rule_prompt_v2/personas/spsb_outcome_instructions/{outcome['id']}.txt"
    )
    return textwrap.dedent(f"""\
        # SPSB IPV — outcome-instruction alternative {outcome['id']}
        # target_class={outcome['target_class']}
        # outcome label: "{outcome['label']}"
        # appendix-only: tests whether outcome instruction reproduces procedural strains

        experiment:
          name: "{outcome['id']}"
          version: "v2-outcome-instructions"
          description: "SPSB IPV with outcome-instruction prompt: {outcome['label']}"

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
          seed_base: 7001

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
          output_dir: "data_v2/experiment_logs/gpt5mini/{outcome['id']}"
        """)


def main(repetitions=30):
    os.makedirs(PERSONAS_DIR, exist_ok=True)
    os.makedirs(CONFIGS_DIR, exist_ok=True)
    rows = []
    for outcome in OUTCOMES:
        ppath = os.path.join(PERSONAS_DIR, f"{outcome['id']}.txt")
        with open(ppath, "w") as fp:
            fp.write(make_persona(outcome))
        ypath = os.path.join(CONFIGS_DIR, f"{outcome['id']}.yaml")
        with open(ypath, "w") as fp:
            fp.write(make_yaml(outcome, repetitions))
        rows.append(
            {
                "outcome_id": outcome["id"],
                "label": outcome["label"],
                "target_class": outcome["target_class"],
                "persona_file": (
                    f"rule_prompt_v2/personas/spsb_outcome_instructions/"
                    f"{outcome['id']}.txt"
                ),
                "config_file": (
                    f"config_v2/configs_auction/interventions_gpt5mini_outcome/"
                    f"{outcome['id']}.yaml"
                ),
            }
        )
    with open(MANIFEST, "w", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {len(rows)} personas to {PERSONAS_DIR}")
    print(f"wrote {len(rows)} configs to {CONFIGS_DIR}")
    print(f"manifest: {MANIFEST}")


if __name__ == "__main__":
    main()
