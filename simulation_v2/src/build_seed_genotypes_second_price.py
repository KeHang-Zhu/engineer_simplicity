"""
Build a YAML of second-price IPV seed genotypes for the EA loop's --genotypes
input. Samples broadly across bid_language / description_style / scaffold and
restricts to second_price + IPV.

Outputs:
    config_v2/configs_auction/ea_second_price/seed_genotypes.yaml

Run from repo root:
    ./venv/bin/python simulation_v2/src/build_seed_genotypes_second_price.py \\
        --n 12 --seed 20260530
"""

import argparse
import json
import os
import sys

import yaml

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(REPO_ROOT, "simulation_v2/src/evolutionary"))

from genotype import (  # noqa: E402
    apply_genotype_scope,
    default_target_for_payment,
    sample_genotypes,
    validate_genotype_scope,
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=12,
                    help="number of second-price seed genotypes to keep")
    ap.add_argument("--seed", type=int, default=20260530)
    ap.add_argument("--out",
                    default="config_v2/configs_auction/ea_second_price/seed_genotypes.yaml")
    args = ap.parse_args()

    # Sample a large pool; we'll filter to second-price and IPV.
    pool = sample_genotypes(60, seed=args.seed, include_benchmarks=False)

    kept = []
    for g in pool:
        if g["mechanism"]["payment_rule"] != "second_price":
            continue
        if g["environment"]["value_model"] != "IPV":
            continue
        # Apply scope to force consistent target_policy and grid defaults.
        try:
            g2 = apply_genotype_scope(g, scope="second_price_ipv")
            validate_genotype_scope(g2, scope="second_price_ipv")
        except Exception as exc:
            print(f"[skip] {exc}")
            continue
        kept.append(g2)
        if len(kept) >= args.n:
            break

    if len(kept) < args.n:
        # Hand-rotate a couple of synthetic seeds to fill any gap.
        defaults = [
            ("continuous_bid", "traditional", "none"),
            ("discrete_grid", "menu", "short_hint"),
            ("ranked_price_menu", "traditional", "decision_table"),
            ("continuous_bid", "menu", "worked_example"),
        ]
        rotation = 0
        while len(kept) < args.n:
            bl, ds, sc = defaults[rotation % len(defaults)]
            rotation += 1
            raw = {
                "environment": {
                    "value_model": "IPV",
                    "n_bidders": 3,
                    "value_support": {
                        "private_range": [0, 49],
                        "common_range": None,
                        "signal_noise": None,
                    },
                },
                "mechanism": {
                    "interaction_form": "sealed_bid",
                    "allocation_rule": "highest_bid_wins",
                    "payment_rule": "second_price",
                    "reserve_rule": "none",
                    "bid_language": bl,
                    "grid": (
                        [0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 49]
                        if bl in {"discrete_grid", "ranked_price_menu"}
                        else None
                    ),
                },
                "framing": {
                    "description_style": ds,
                    "mechanism_scaffold": sc,
                    "has_chain_of_thought_demo": sc == "worked_example",
                },
                "target_policy": default_target_for_payment("second_price", 3),
            }
            g2 = apply_genotype_scope(raw, scope="second_price_ipv")
            kept.append(g2)

    out_abs = os.path.join(REPO_ROOT, args.out) if not os.path.isabs(args.out) else args.out
    os.makedirs(os.path.dirname(out_abs), exist_ok=True)
    with open(out_abs, "w") as fp:
        yaml.safe_dump({"genotypes": kept}, fp, sort_keys=False)
    print(f"wrote {len(kept)} second-price IPV seeds to {out_abs}")
    print("Bid languages:")
    for g in kept:
        print(f"  - {g['mechanism']['bid_language']:18s}  "
              f"framing={g['framing']['description_style']:14s}  "
              f"scaffold={g['framing']['mechanism_scaffold']}")


if __name__ == "__main__":
    main()
