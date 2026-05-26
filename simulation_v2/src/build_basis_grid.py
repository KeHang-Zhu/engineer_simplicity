"""
Build the 3x3x3 cognitive-basis grid for Stage A v2.

Composes 27 persona files from axis snippets and writes 27 matching YAML configs
plus a manifest CSV of the full grid. Idempotent — safe to re-run.

Run from repo root:
    ./venv/bin/python simulation_v2/src/build_basis_grid.py
"""

import csv
import os
import textwrap

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
SNIPPETS_DIR = os.path.join(REPO_ROOT, "rule_prompt_v2/personas/_snippets")
PERSONAS_DIR = os.path.join(REPO_ROOT, "rule_prompt_v2/personas/spsb_basis")
CONFIGS_DIR = os.path.join(REPO_ROOT, "config_v2/configs_auction/interventions_gpt5mini")
MANIFEST = os.path.join(REPO_ROOT, "config_v2/configs_auction/interventions_gpt5mini/_basis_grid_manifest.csv")

AXIS_LEVELS = {
    "c": ["off", "enumerate", "worstcase"],     # axis 1 — contingent thinking
    "f": ["off", "onestep", "tree"],            # axis 2 — forward planning
    "b": ["off", "firstorder", "secondorder"],  # axis 3 — beliefs
}

# Map (axis_code, level_idx) -> snippet filename in _snippets/
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


def load_snippet(axis_code, level_idx):
    path = os.path.join(SNIPPETS_DIR, SNIPPET_FILES[(axis_code, level_idx)])
    with open(path) as f:
        return f.read().strip()


PERSONA_HEADER = (
    "Your TOP PRIORITY is to place bids which maximize your profit in the long run. "
    "Learn from the history of previous rounds in order to maximize your total profit. "
    "Don't forget the values are redrawn independently each round.\n\n"
    "You are a bidder with the following cognitive style:"
)


def make_persona(c_idx, f_idx, b_idx):
    """Compose one persona from the three axis snippets."""
    parts = [
        PERSONA_HEADER,
        f"(1) {load_snippet('c', c_idx)}",
        f"(2) {load_snippet('f', f_idx)}",
        f"(3) {load_snippet('b', b_idx)}",
    ]
    return "\n\n".join(parts) + "\n"


def basis_id(c_idx, f_idx, b_idx):
    return f"c{c_idx}_f{f_idx}_b{b_idx}"


def basis_label(c_idx, f_idx, b_idx):
    return f"contingent={AXIS_LEVELS['c'][c_idx]}, forward={AXIS_LEVELS['f'][f_idx]}, beliefs={AXIS_LEVELS['b'][b_idx]}"


def make_yaml(c_idx, f_idx, b_idx):
    bid = basis_id(c_idx, f_idx, b_idx)
    label = basis_label(c_idx, f_idx, b_idx)
    persona_rel = f"rule_prompt_v2/personas/spsb_basis/{bid}.txt"
    return textwrap.dedent(f"""\
        # SPSB IPV — basis grid cell {bid}
        # {label}

        experiment:
          name: "spsb_basis_{bid}"
          version: "v2-grid"
          description: "SPSB IPV with cognitive persona ({label})"

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
          seed_base: 5001

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
          repetitions: 10
          parallel: true
          max_workers: 4
          output_dir: "data_v2/experiment_logs/gpt5mini/spsb_basis_{bid}"
        """)


def main():
    os.makedirs(PERSONAS_DIR, exist_ok=True)
    os.makedirs(CONFIGS_DIR, exist_ok=True)
    rows = []
    n_persona = 0
    n_yaml = 0
    for c in range(3):
        for f in range(3):
            for b in range(3):
                bid = basis_id(c, f, b)
                # persona
                ppath = os.path.join(PERSONAS_DIR, f"{bid}.txt")
                with open(ppath, "w") as fp:
                    fp.write(make_persona(c, f, b))
                n_persona += 1
                # yaml
                ypath = os.path.join(CONFIGS_DIR, f"spsb_basis_{bid}.yaml")
                with open(ypath, "w") as fp:
                    fp.write(make_yaml(c, f, b))
                n_yaml += 1
                rows.append({
                    "basis_id": bid,
                    "contingent": AXIS_LEVELS["c"][c],
                    "forward": AXIS_LEVELS["f"][f],
                    "beliefs": AXIS_LEVELS["b"][b],
                    "persona_file": f"rule_prompt_v2/personas/spsb_basis/{bid}.txt",
                    "config_file": f"config_v2/configs_auction/interventions_gpt5mini/spsb_basis_{bid}.yaml",
                })
    with open(MANIFEST, "w", newline="") as fp:
        w = csv.DictWriter(fp, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {n_persona} personas to {PERSONAS_DIR}")
    print(f"wrote {n_yaml} configs to {CONFIGS_DIR}")
    print(f"manifest: {MANIFEST}")


if __name__ == "__main__":
    main()
