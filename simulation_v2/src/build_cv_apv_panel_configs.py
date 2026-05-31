"""
Hand-build 4 mechanisms (FP-APV, SP-APV, FP-CV, SP-CV) that the v1
sealed-bid runner actually executes with the correct value model, and pair
each with the 5-strain Li-2017 cluster panel.

These supplement the unrestricted EA run, which drifted everything back to
IPV at the LLM-mediated phenotype step. The combined IPV (from the EA) +
CV/APV (from this builder) ranking is what we compare to Figure 1.

Outputs:
    config_v2/configs_auction/cv_apv_panel/<mech>__<strain>.yaml
"""

import csv
import os
import textwrap

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
OUT_DIR = os.path.join(REPO_ROOT, "config_v2/configs_auction/cv_apv_panel")
MANIFEST = os.path.join(OUT_DIR, "_manifest.csv")


MECHANISMS = [
    {
        "id": "fpsb_apv",
        "label": "FPSB APV",
        "price_order": "first",
        "private_value": "affiliated",
        "special_name": "affiliated_fpsb.txt",
        # APV with affiliated values: optimal still ~ (n-1)/n*v as proxy.
        "benchmark": "((3 - 1) / 3) * v",
    },
    {
        "id": "spsb_apv",
        "label": "SPSB APV",
        "price_order": "second",
        "private_value": "affiliated",
        "special_name": "affiliated_spsb.txt",
        # APV with affiliated values in 2P: truthful is still a good benchmark.
        "benchmark": "v",
    },
    {
        "id": "fpsb_cv",
        "label": "FPSB CV",
        "price_order": "first",
        "private_value": "common",
        "special_name": "common_first_price.txt",
        # CV with own signal as v; (n-1)/n*v is a rough proxy (no closed form
        # for the winner's-curse correction here).
        "benchmark": "((3 - 1) / 3) * v",
    },
    {
        "id": "spsb_cv",
        "label": "SPSB CV",
        "price_order": "second",
        "private_value": "common",
        "special_name": "common_second_price.txt",
        # CV 2P: truthful-on-signal is a reasonable benchmark for the proxy.
        "benchmark": "v",
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
        "persona_id": "spsb_error_win_seeker",
        "persona_file": "rule_prompt_v2/personas/spsb_error_modes/spsb_error_win_seeker.txt",
    },
    {
        "persona_id": "spsb_error_loss_averse",
        "persona_file": "rule_prompt_v2/personas/spsb_error_modes/spsb_error_loss_averse.txt",
    },
    {
        "persona_id": "spsb_outcome_truthful",
        "persona_file": "rule_prompt_v2/personas/spsb_outcome_instructions/spsb_outcome_truthful.txt",
    },
]


def make_yaml(mech, persona, repetitions=10):
    out_name = f"{mech['id']}__{persona['persona_id']}"
    common_low = 0 if mech["private_value"] == "private" else 20
    common_high = 0 if mech["private_value"] == "private" else 29
    return textwrap.dedent(f"""\
        # CV/APV supplement to the unrestricted EA run
        # mechanism: {mech['label']}
        # persona:   {persona['persona_id']}

        experiment:
          name: "{out_name}"
          version: "v2-cv-apv-supplement"
          description: "{mech['label']} -- panel strain {persona['persona_id']}"

        mechanism_metadata:
          incentive_target:
            action: "shade_below_value"
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
          private_value: "{mech['private_value']}"
          open_blind: "open"
          closing: false
          reserve_price: 0
          special_name: "{mech['special_name']}"

        value:
          common_range: [{common_low}, {common_high}]
          private_range: 49
          increment: 0.1
          seed_base: 9801

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
          output_dir: "data_v2/experiment_logs/gpt5mini/cv_apv_{out_name}"
        """)


def main(repetitions=10):
    os.makedirs(OUT_DIR, exist_ok=True)
    rows = []
    for mech in MECHANISMS:
        for persona in PANEL:
            out_name = f"{mech['id']}__{persona['persona_id']}"
            with open(os.path.join(OUT_DIR, f"{out_name}.yaml"), "w") as fp:
                fp.write(make_yaml(mech, persona, repetitions))
            rows.append({
                "mechanism_id": mech["id"],
                "mechanism_label": mech["label"],
                "persona_id": persona["persona_id"],
                "benchmark": mech["benchmark"],
                "config_file": f"config_v2/configs_auction/cv_apv_panel/{out_name}.yaml",
            })
    with open(MANIFEST, "w", newline="") as fp:
        w = csv.DictWriter(fp, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {len(rows)} configs to {OUT_DIR}")


if __name__ == "__main__":
    main()
