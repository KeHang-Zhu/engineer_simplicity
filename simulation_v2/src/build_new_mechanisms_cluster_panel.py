"""
Build YAML configs for 4 new candidate mechanisms x 5 cluster-panel strains.

Mechanisms:
  E1 vickrey_baseline       -- 2P, b*(v) = v
  E2 vickrey_osp_hint       -- 2P + explicit OSP-style sentence, b*(v) = v
  H1 allpay_first_price     -- all bidders pay their bid, highest wins
                                b*(v) = v^n / V_max^(n-1) for n=3
  H2 firstprice_with_reserve -- 1P with $15 reserve, b*(v) = max(15, (n-1)/n v) if v >= 15

Panel:
  c0_f0_b0 (CoT), c2_f2_b2 (CoT), overgeneralizer, payment_panic, outcome_truthful

Outputs:
  config_v2/configs_auction/new_mechanisms_cluster/<mech>_<persona>.yaml
  config_v2/configs_auction/new_mechanisms_cluster/_manifest.csv

Run from repo root:
    ./venv/bin/python simulation_v2/src/build_new_mechanisms_cluster_panel.py
"""

import csv
import os
import textwrap

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
CONFIGS_DIR = os.path.join(
    REPO_ROOT, "config_v2/configs_auction/new_mechanisms_cluster"
)
MANIFEST = os.path.join(CONFIGS_DIR, "_manifest.csv")


MECHANISMS = [
    {
        "id": "vickrey_baseline",
        "label": "E1: Vickrey 2P (baseline easy)",
        "category": "easy",
        "price_order": "second",
        "reserve_price": 0,
        "special_name": "vickrey_baseline.txt",
        "benchmark": "value",
        "action": "truthful_bid",
    },
    {
        "id": "vickrey_osp_hint",
        "label": "E2: 2P + OSP-style framing",
        "category": "easy",
        "price_order": "second",
        "reserve_price": 0,
        "special_name": "vickrey_osp_hint.txt",
        "benchmark": "value",
        "action": "truthful_bid",
    },
    {
        "id": "allpay_first_price",
        "label": "H1: All-pay first-price",
        "category": "hard",
        "price_order": "allpay",
        "reserve_price": 0,
        "special_name": "allpay_first_price.txt",
        # b*(v) = v^n / V_max^(n-1) for n=3, V_max=49
        "benchmark": "value**3 / 49**2",
        "action": "target_bid_value_ratio",
    },
    {
        "id": "firstprice_with_reserve",
        "label": "H2: First-price with $15 reserve",
        "category": "hard",
        "price_order": "first",
        "reserve_price": 15,
        "special_name": "firstprice_with_reserve.txt",
        # b*(v) = max(15, (n-1)/n * v) if v >= 15 else 0
        "benchmark": "max(15, ((3-1)/3) * value) if value >= 15 else 0",
        "action": "shade_below_value",
    },
]


PANEL = [
    {
        "persona_id": "c0_f0_b0_cot",
        "persona_file": "rule_prompt_v2/personas/spsb_basis_cot/c0_f0_b0.txt",
    },
    {
        "persona_id": "c2_f2_b2_cot",
        "persona_file": "rule_prompt_v2/personas/spsb_basis_cot/c2_f2_b2.txt",
    },
    {
        "persona_id": "spsb_error_overgeneralizer",
        "persona_file": "rule_prompt_v2/personas/spsb_error_modes/spsb_error_second_price_overgeneralizer.txt",
    },
    {
        "persona_id": "spsb_error_payment_panic",
        "persona_file": "rule_prompt_v2/personas/spsb_error_modes/spsb_error_payment_panic_near_zero.txt",
    },
    {
        "persona_id": "spsb_outcome_truthful",
        "persona_file": "rule_prompt_v2/personas/spsb_outcome_instructions/spsb_outcome_truthful.txt",
    },
]


def make_yaml(mech, persona, repetitions=15):
    out_name = f"{mech['id']}__{persona['persona_id']}"
    return textwrap.dedent(f"""\
        # New-mechanisms cluster-panel run
        # mechanism: {mech['label']}
        # persona:   {persona['persona_id']}

        experiment:
          name: "{out_name}"
          version: "v2-newmech-cluster"
          description: "{mech['label']} -- panel strain {persona['persona_id']}"

        mechanism_metadata:
          incentive_target:
            action: "{mech['action']}"
            description: "{mech['label']}"
            parameters:
              benchmark: "{mech['benchmark']}"

        auction:
          number_agents: 3
          rounds: 1

        rule:
          seal_clock: "seal"
          ascend_descend: "ascend"
          price_order: "{mech['price_order']}"
          private_value: "private"
          open_blind: "open"
          closing: false
          reserve_price: {mech['reserve_price']}
          special_name: "{mech['special_name']}"

        value:
          common_range: [0, 29]
          private_range: 49
          increment: 0.1
          seed_base: 9401

        llm:
          model: "gpt-5.4-mini"
          temperature: 0.5

        prompt:
          strategy_type: "plan_reflection"
          prompt_dir: "Prompt/"
          rule_template_dir: "rule_template/auctions/"
          persona_file: "{persona['persona_file']}"
          include_payment_example: true

        execution:
          repetitions: {repetitions}
          parallel: true
          max_workers: 4
          output_dir: "data_v2/experiment_logs/gpt5mini/newmech_{out_name}"
        """)


def main(repetitions=15):
    os.makedirs(CONFIGS_DIR, exist_ok=True)
    rows = []
    for mech in MECHANISMS:
        for persona in PANEL:
            out_name = f"{mech['id']}__{persona['persona_id']}"
            path = os.path.join(CONFIGS_DIR, f"{out_name}.yaml")
            with open(path, "w") as fp:
                fp.write(make_yaml(mech, persona, repetitions))
            rows.append({
                "mechanism_id": mech["id"],
                "mechanism_label": mech["label"],
                "category": mech["category"],
                "persona_id": persona["persona_id"],
                "persona_file": persona["persona_file"],
                "benchmark": mech["benchmark"],
                "config_file": (
                    f"config_v2/configs_auction/new_mechanisms_cluster/{out_name}.yaml"
                ),
            })
    with open(MANIFEST, "w", newline="") as fp:
        w = csv.DictWriter(fp, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {len(rows)} configs to {CONFIGS_DIR}")
    print(f"manifest: {MANIFEST}")


if __name__ == "__main__":
    main()
