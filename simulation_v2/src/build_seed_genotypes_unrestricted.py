"""
Build seed genotypes for the unrestricted EA round 2. Spans value_model
in {IPV, APV, CV} x payment_rule in {first_price, second_price, third_price,
all_pay} x framing variations, to test whether the EA recovers the human
SMAD ranking from llm-auction Figure 1 (FP-CV >> FP-IPV > SP-CV > SP-APV >
SP-IPV > AC APV).

Note: The v1 sealed-bid runner supports private/affiliated/common value
models. Ascending-clock mechanisms are not currently runnable through
parallel_mice and are excluded.

Outputs:
    config_v2/configs_auction/ea_unrestricted_r2/seed_genotypes.yaml

Run from repo root:
    ./venv/bin/python simulation_v2/src/build_seed_genotypes_unrestricted.py \\
        --n 18 --seed 20260601
"""

import argparse
import itertools
import os
import sys

import yaml

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(REPO_ROOT, "simulation_v2/src/evolutionary"))

from genotype import (  # noqa: E402
    default_target_for_payment,
    validate_genotype,
)


# (value_model, payment_rule) cells that approximately mirror Figure 1
# of the llm-auction paper. Each cell will be paired with 1-2 framing
# variations.
TARGET_CELLS = [
    ("IPV", "first_price"),
    ("IPV", "second_price"),
    ("IPV", "third_price"),
    ("IPV", "all_pay"),
    ("APV", "first_price"),
    ("APV", "second_price"),
    ("CV",  "first_price"),
    ("CV",  "second_price"),
]


FRAMING_CYCLE = [
    ("traditional", "none"),
    ("menu", "short_hint"),
    ("traditional", "worked_example"),
]


def make_genotype(value_model, payment_rule, description_style, scaffold):
    bid_language = "continuous_bid"
    grid = None
    target = default_target_for_payment(payment_rule, 3)
    if value_model == "CV":
        target = {
            "type": "common_value_shading",
            "benchmark_formula": "custom",
            "target_bid_value_ratio": None,
            "tolerance": 0.15,
        }
    raw = {
        "environment": {
            "value_model": value_model,
            "value_support": {
                "private_range": [0, 49],
                "common_range": [20, 29] if value_model in {"APV", "CV"} else None,
                "signal_noise": [-20, 20] if value_model == "CV" else None,
            },
        },
        "mechanism": {
            "interaction_form": "sealed_bid",
            "allocation_rule": "highest_bid_wins",
            "payment_rule": payment_rule,
            "reserve_rule": "none",
            "bid_language": bid_language,
            "grid": grid,
        },
        "framing": {
            "description_style": description_style,
            "mechanism_scaffold": scaffold,
            "has_chain_of_thought_demo": scaffold == "worked_example",
        },
        "target_policy": target,
    }
    return validate_genotype(raw)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=18)
    ap.add_argument("--seed", type=int, default=20260601)
    ap.add_argument("--out",
                    default="config_v2/configs_auction/ea_unrestricted_r2/seed_genotypes.yaml")
    args = ap.parse_args()

    seeds = []
    framing_iter = itertools.cycle(FRAMING_CYCLE)
    for value_model, payment_rule in TARGET_CELLS:
        if len(seeds) >= args.n:
            break
        for _ in range(min(2, args.n - len(seeds))):
            ds, sc = next(framing_iter)
            try:
                g = make_genotype(value_model, payment_rule, ds, sc)
                seeds.append(g)
            except Exception as exc:
                print(f"[skip] {value_model}/{payment_rule}: {exc}")

    out_abs = os.path.join(REPO_ROOT, args.out) if not os.path.isabs(args.out) else args.out
    os.makedirs(os.path.dirname(out_abs), exist_ok=True)
    with open(out_abs, "w") as fp:
        yaml.safe_dump({"genotypes": seeds}, fp, sort_keys=False)

    print(f"wrote {len(seeds)} seeds to {out_abs}")
    for g in seeds:
        print(f"  - {g['environment']['value_model']:3s}  "
              f"{g['mechanism']['payment_rule']:14s}  "
              f"{g['framing']['description_style']:11s}  "
              f"{g['framing']['mechanism_scaffold']}")


if __name__ == "__main__":
    main()
