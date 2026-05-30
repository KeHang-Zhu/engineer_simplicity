"""
Build the full 27-cell strict-prompt panel for v2 round 2.

The strict scaffold (existing in rule_prompt_v2/personas/spsb_basis_strict/)
makes the reasoning-procedure constraint the top priority and explicitly
forbids dominance shortcuts. Unlike the CoT panel it includes no worked
example -- the deficit is held by the strict instruction alone.

This builder writes only the 27 YAML configs that point to the existing
strict persona files. Personas themselves were generated previously by
build_spsb_strict_basis_grid.py.

Configs land in config_v2/configs_auction/interventions_gpt5mini_strict/
Logs land in data_v2/experiment_logs/gpt5mini/spsb_basis_strict_<id>/.

Run from repo root:
    ./venv/bin/python simulation_v2/src/build_spsb_basis_strict_full.py
"""

import csv
import os
import textwrap

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
PERSONAS_DIR = os.path.join(REPO_ROOT, "rule_prompt_v2/personas/spsb_basis_strict")
CONFIGS_DIR = os.path.join(
    REPO_ROOT, "config_v2/configs_auction/interventions_gpt5mini_strict"
)
MANIFEST = os.path.join(CONFIGS_DIR, "_basis_strict_grid_manifest.csv")

AXIS_LEVELS = {
    "c": ["off", "enumerate", "worstcase"],
    "f": ["off", "onestep", "tree"],
    "b": ["off", "firstorder", "secondorder"],
}


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
    persona_rel = f"rule_prompt_v2/personas/spsb_basis_strict/{bid}.txt"
    return textwrap.dedent(f"""\
        # SPSB IPV -- strict basis grid cell {bid}
        # {label}
        # strict scaffold: cognitive procedure is the top priority; no CoT example

        experiment:
          name: "spsb_basis_strict_{bid}"
          version: "v2-strict-grid"
          description: "SPSB IPV strict-instruction cognitive persona ({label})"

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
          seed_base: 8001

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
          output_dir: "data_v2/experiment_logs/gpt5mini/spsb_basis_strict_{bid}"
        """)


def main(repetitions=30):
    if not os.path.isdir(PERSONAS_DIR):
        raise SystemExit(f"strict personas dir missing: {PERSONAS_DIR}")
    os.makedirs(CONFIGS_DIR, exist_ok=True)
    rows = []
    for c in range(3):
        for f in range(3):
            for b in range(3):
                bid = basis_id(c, f, b)
                if not os.path.isfile(os.path.join(PERSONAS_DIR, f"{bid}.txt")):
                    raise SystemExit(f"missing strict persona: {bid}.txt")
                ypath = os.path.join(CONFIGS_DIR, f"spsb_basis_strict_{bid}.yaml")
                with open(ypath, "w") as fp:
                    fp.write(make_yaml(c, f, b, repetitions))
                rows.append(
                    {
                        "basis_id": bid,
                        "contingent": AXIS_LEVELS["c"][c],
                        "forward": AXIS_LEVELS["f"][f],
                        "beliefs": AXIS_LEVELS["b"][b],
                        "persona_file": (
                            f"rule_prompt_v2/personas/spsb_basis_strict/{bid}.txt"
                        ),
                        "config_file": (
                            "config_v2/configs_auction/interventions_gpt5mini_strict/"
                            f"spsb_basis_strict_{bid}.yaml"
                        ),
                    }
                )
    with open(MANIFEST, "w", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {len(rows)} configs to {CONFIGS_DIR}")
    print(f"manifest: {MANIFEST}")


if __name__ == "__main__":
    main()
