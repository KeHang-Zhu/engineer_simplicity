"""
Build the full 27-cell CoT panel for v2 round 2.

Differences from build_basis_grid.py:
  * Short base prompt: "Your goal is to place bids which maximize your profit."
  * Each persona ends with a worked chain-of-thought example whose reasoning
    is consistent with that strain's (c, f, b) cognitive style.
  * Personas land in rule_prompt_v2/personas/spsb_basis_cot/ (overwriting the
    two-strain pilot).
  * Configs land in config_v2/configs_auction/interventions_gpt5mini_cot/ with
    names spsb_basis_cot_<id>.yaml.
  * Experiment logs land in data_v2/experiment_logs/gpt5mini/spsb_basis_cot_<id>/
    so the previous run remains untouched as a comparison set.

Run from repo root:
    ./venv/bin/python simulation_v2/src/build_spsb_basis_cot_full.py
"""

import csv
import os
import textwrap

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
SNIPPETS_DIR = os.path.join(REPO_ROOT, "rule_prompt_v2/personas/_snippets")
PERSONAS_DIR = os.path.join(REPO_ROOT, "rule_prompt_v2/personas/spsb_basis_cot")
CONFIGS_DIR = os.path.join(
    REPO_ROOT, "config_v2/configs_auction/interventions_gpt5mini_cot"
)
MANIFEST = os.path.join(CONFIGS_DIR, "_basis_cot_grid_manifest.csv")

AXIS_LEVELS = {
    "c": ["off", "enumerate", "worstcase"],
    "f": ["off", "onestep", "tree"],
    "b": ["off", "firstorder", "secondorder"],
}

SNIPPET_FILES = {
    ("c", 0): "c0_contingent_off.txt",
    ("c", 1): "c1_contingent_enumerate.txt",
    ("c", 2): "c2_contingent_worstcase.txt",
    ("f", 0): "f0_forward_off.txt",
    ("f", 1): "f1_forward_onestep.txt",
    ("f", 2): "f2_forward_tree.txt",
    ("b", 0): "b0_beliefs_off.txt",
    ("b", 1): "b1_beliefs_firstorder.txt",
    ("b", 2): "b2_beliefs_secondorder.txt",
}

# The new short base prompt.
PERSONA_HEADER = (
    "Your goal is to place bids which maximize your profit.\n\n"
    "You are a bidder with the following cognitive style:"
)

# Worked example value.
EXAMPLE_VALUE = 30


def load_snippet(axis, level):
    with open(os.path.join(SNIPPETS_DIR, SNIPPET_FILES[(axis, level)])) as fp:
        return fp.read().strip()


def _c_plan(c):
    if c == 0:
        return (
            "I do not enumerate what opponents might bid or compare candidate "
            "bids by opponent response; I apply a simple own-value rule."
        )
    if c == 1:
        return (
            "Enumerate a few possible opponent bid levels. For each, ask "
            "whether my best response shifts. Because the winner pays the "
            "second-highest bid, my own bid only changes whether I win, not "
            "how much I pay, so one bid performs well across all opponent "
            "levels."
        )
    return (
        "Compare candidate bids on their worst-case outcome over opponent "
        "bids. The winner pays the second-highest bid, so my bid changes "
        "WHETHER I win, never HOW MUCH I pay; one bid dominates in the worst "
        "case."
    )


def _f_plan(f):
    if f == 0:
        return "I do not trace expected profit case by case; I rely on the rule above."
    if f == 1:
        return (
            "Trace one step for the candidate I have in mind: probability of "
            "winning at this bid and profit if I win."
        )
    return (
        "Build a small tree across candidates: for each, trace win/lose "
        "outcomes and compute expected profit, then pick the best."
    )


def _b_plan(b):
    if b == 0:
        return "I do not model the other bidders' minds; the environment is fixed background."
    if b == 1:
        return (
            "I assume other bidders are rational and similarly minded, but "
            "this only informs, not overrides, the rule above."
        )
    return (
        "Other bidders anticipate my reasoning and I anticipate theirs; in "
        "equilibrium this still leads back to the same answer."
    )


def _conclusion(c):
    if c == 0:
        return (
            f"Apply a margin-of-safety underbid: roughly 70% of ${EXAMPLE_VALUE} "
            f"is about $21.",
            "21",
        )
    return (
        f"The bid that is best across cases is exactly my value, ${EXAMPLE_VALUE}.",
        f"{EXAMPLE_VALUE}",
    )


def make_worked_example(c, f, b):
    plan_body = " ".join([_c_plan(c), _f_plan(f), _b_plan(b)])
    conclusion_text, action = _conclusion(c)
    return textwrap.dedent(f"""\

        Worked example (reason in this style; this is a second-price auction):
        Value = ${EXAMPLE_VALUE}.
        PLAN: {plan_body} {conclusion_text}
        ACTION: {action}
        """)


def make_persona(c, f, b):
    parts = [
        PERSONA_HEADER,
        f"(1) {load_snippet('c', c)}",
        f"(2) {load_snippet('f', f)}",
        f"(3) {load_snippet('b', b)}",
    ]
    body = "\n\n".join(parts)
    return body + "\n" + make_worked_example(c, f, b)


def basis_id(c, f, b):
    return f"c{c}_f{f}_b{b}"


def basis_label(c, f, b):
    return (
        f"contingent={AXIS_LEVELS['c'][c]}, forward={AXIS_LEVELS['f'][f]}, "
        f"beliefs={AXIS_LEVELS['b'][b]}"
    )


def make_yaml(c, f, b, repetitions):
    bid = basis_id(c, f, b)
    label = basis_label(c, f, b)
    persona_rel = f"rule_prompt_v2/personas/spsb_basis_cot/{bid}.txt"
    return textwrap.dedent(f"""\
        # SPSB IPV — CoT basis grid cell {bid}
        # {label}
        # base prompt: short ("Your goal is to place bids which maximize your profit.")
        # each persona ends with a worked example consistent with the cognitive style

        experiment:
          name: "spsb_basis_cot_{bid}"
          version: "v2-cot-grid"
          description: "SPSB IPV CoT-induced cognitive persona ({label})"

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
          seed_base: 6001

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
          output_dir: "data_v2/experiment_logs/gpt5mini/spsb_basis_cot_{bid}"
        """)


def main(repetitions=30):
    os.makedirs(PERSONAS_DIR, exist_ok=True)
    os.makedirs(CONFIGS_DIR, exist_ok=True)
    rows = []
    n_persona = 0
    n_yaml = 0
    for c in range(3):
        for f in range(3):
            for b in range(3):
                bid = basis_id(c, f, b)
                ppath = os.path.join(PERSONAS_DIR, f"{bid}.txt")
                with open(ppath, "w") as fp:
                    fp.write(make_persona(c, f, b))
                n_persona += 1
                ypath = os.path.join(CONFIGS_DIR, f"spsb_basis_cot_{bid}.yaml")
                with open(ypath, "w") as fp:
                    fp.write(make_yaml(c, f, b, repetitions))
                n_yaml += 1
                rows.append(
                    {
                        "basis_id": bid,
                        "contingent": AXIS_LEVELS["c"][c],
                        "forward": AXIS_LEVELS["f"][f],
                        "beliefs": AXIS_LEVELS["b"][b],
                        "persona_file": f"rule_prompt_v2/personas/spsb_basis_cot/{bid}.txt",
                        "config_file": (
                            "config_v2/configs_auction/interventions_gpt5mini_cot/"
                            f"spsb_basis_cot_{bid}.yaml"
                        ),
                    }
                )
    with open(MANIFEST, "w", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {n_persona} personas to {PERSONAS_DIR}")
    print(f"wrote {n_yaml} configs to {CONFIGS_DIR}")
    print(f"manifest: {MANIFEST}")


if __name__ == "__main__":
    main()
