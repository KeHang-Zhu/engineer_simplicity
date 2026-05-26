"""
Component 2 of Stage E: fitness evaluator for a single auction mechanism.

Takes a mechanism YAML + Jinja2 template (the output of `mechanism_generator.py`)
and scores it by running K simulations against a uniform mixture over the
27-cell cognitive basis (Stage A persona files at
rule_prompt_v2/personas/spsb_basis/c{0,1,2}_f{0,1,2}_b{0,1,2}.txt).

This is the placeholder evaluator described in the README: Stage C calibration
of mixture weights is not yet done, so we use uniform weights over the 27 cells.

Per-mechanism we report four sub-metrics aggregated across personas:
  - revenue:           winner's payment (mean over runs)
  - efficiency:        winner_value / max_value, in [0,1] (allocative efficiency)
  - simplicity_proxy:  1 - std(bid/value across bidders within a run);
                       high when bidders converge on a strategy (mechanism is
                       "obvious" to reason about); low when bids scatter.
  - regret_var:        variance of realized profits across runs (mechanism
                       volatility — high = bidders are gambling).

Scalar fitness = w_rev * revenue_norm + w_eff * efficiency - w_simp * (1-simplicity_proxy) - w_reg * regret_var_norm.
Weights default to (1, 1, 0.5, 0.25) — overridable via CLI.

Reuses simulation_v1's Auction_plan.run() the same way simulation_v2/run_basis_batch.py
does, so it picks up any cache hits from prior basis-grid runs.

Run from repo root:
    ./venv/bin/python simulation_v2/src/evolutionary/fitness.py \
        data_v2/evo/gen0/mech_000_*.yaml \
        --reps_per_persona 2 --n_personas 3 --max_workers 4
"""

import argparse
import concurrent.futures
import glob as _glob
import itertools
import json
import os
import statistics
import sys

import pandas as pd
import yaml
from edsl import Cache

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
sys.path.insert(0, os.path.join(REPO_ROOT, "simulation_v1", "src"))
from util_plan import Auction_plan, Rule_plan  # type: ignore  # noqa: E402

PERSONA_DIR = os.path.join(REPO_ROOT, "rule_prompt_v2", "personas", "spsb_basis")
PERSONA_CELLS = [
    f"c{c}_f{f}_b{b}" for c, f, b in itertools.product(range(3), repeat=3)
]


def _load_yaml(path):
    with open(path) as f:
        return yaml.safe_load(f)


def _load_persona_text(cell):
    path = os.path.join(PERSONA_DIR, f"{cell}.txt")
    with open(path) as f:
        return f.read()


def _build_rule(cfg):
    a, r, v, p = cfg["auction"], cfg["rule"], cfg["value"], cfg["prompt"]
    return Rule_plan(
        seal_clock=r["seal_clock"],
        price_order=r["price_order"],
        private_value=r["private_value"],
        open_blind=r["open_blind"],
        rounds=a["rounds"],
        turns=20,
        common_range=v["common_range"],
        private_range=v["private_range"],
        increment=v["increment"],
        number_agents=a["number_agents"],
        special_name=r.get("special_name", ""),
        closing=r.get("closing", False),
        reserve_price=r.get("reserve_price", 0),
        templates_dir=p.get("rule_template_dir", "rule_template/auctions/"),
        include_payment_example=p.get("include_payment_example", False),
    )


def _run_one(cfg, cache, persona_text, rep_idx, seed_offset):
    """Run one rep of (mechanism, persona) and return a per-run record."""
    rule = _build_rule(cfg)
    rule.persona = persona_text  # uniform-mixture cell override

    a, v, l = cfg["auction"], cfg["value"], cfg["llm"]
    timestring = pd.Timestamp.now().strftime("%Y-%m-%d_%H-%M-%S-%f")
    out_dir = cfg["execution"]["output_dir"]
    os.makedirs(out_dir, exist_ok=True)

    auction = Auction_plan(
        number_agents=a["number_agents"],
        rule=rule,
        output_dir=out_dir,
        timestring=timestring,
        cache=cache,
        model=l["model"],
        temperature=l["temperature"],
        service_name=l.get("service_name"),
    )
    auction.draw_value(seed=v["seed_base"] + seed_offset + rep_idx)
    auction.run_repeated()

    # The mechanism wrote round_0 only (rounds=1 by construction).
    round_0 = auction.data_to_save["round_0"]
    values = list(round_0["value"])
    winner_block = round_0["history"]["winner"]
    bids = [b["bid"] for b in round_0["history"]["bidding history"]]
    profits = list(round_0["profit"])

    # Map winner name → index
    winner_name = winner_block["winner"]
    bid_records = round_0["history"]["bidding history"]
    bidder_names = [b["agent"] for b in bid_records]
    try:
        w_idx = bidder_names.index(winner_name)
        winner_value = values[w_idx]
    except ValueError:
        winner_value = None  # no-sale or unexpected winner string

    price = float(winner_block["price"]) if winner_block.get("price") is not None else 0.0
    max_val = max(values) if values else 0
    efficiency = (winner_value / max_val) if (winner_value is not None and max_val > 0) else 0.0

    bid_value_ratios = [
        (float(bids[i]) / values[i]) for i in range(len(values)) if values[i] > 0
    ]
    if len(bid_value_ratios) >= 2:
        # std of bid/value across bidders in this run; lower std = more "obvious" strategy
        bv_std = statistics.pstdev(bid_value_ratios)
    else:
        bv_std = 0.0

    return {
        "values": values,
        "bids": [float(b) for b in bids],
        "winner_name": winner_name,
        "winner_value": winner_value,
        "price": price,
        "profits": [float(x) for x in profits],
        "max_val": max_val,
        "efficiency": efficiency,
        "bid_value_std": bv_std,
    }


def evaluate_mechanism(yaml_path, n_personas=27, reps_per_persona=2,
                       max_workers=4, seed_offset=0,
                       weights=(1.0, 1.0, 0.5, 0.25), cache=None,
                       verbose=True):
    """Run a uniform-mixture fitness eval; return per-metric dict + scalar."""
    cfg = _load_yaml(yaml_path)
    cells = PERSONA_CELLS[:n_personas]
    if cache is None:
        cache = Cache()

    # Each task: (cell, rep). Tasks are independent — run in a thread pool.
    tasks = []
    for cell in cells:
        ptxt = _load_persona_text(cell)
        for r in range(reps_per_persona):
            tasks.append((cell, ptxt, r))

    runs = []  # list of dicts: per-cell, per-rep records
    if verbose:
        print(f"[fitness] {os.path.basename(yaml_path)}: "
              f"{len(cells)} personas × {reps_per_persona} reps "
              f"= {len(tasks)} runs (workers={max_workers})")

    def _job(cell, ptxt, r):
        rec = _run_one(cfg, cache, ptxt, r, seed_offset)
        rec["persona"] = cell
        rec["rep"] = r
        return rec

    if max_workers > 1:
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as ex:
            futs = [ex.submit(_job, c, p, r) for (c, p, r) in tasks]
            for fut in concurrent.futures.as_completed(futs):
                try:
                    runs.append(fut.result())
                except Exception as exc:
                    import traceback
                    print(f"[fitness] run failed: {exc}")
                    traceback.print_exc()
    else:
        for (c, p, r) in tasks:
            try:
                runs.append(_job(c, p, r))
            except Exception as exc:
                import traceback
                print(f"[fitness] run failed: {exc}")
                traceback.print_exc()

    if not runs:
        raise RuntimeError("no successful runs — cannot compute fitness")

    revenues = [r["price"] for r in runs]
    effs = [r["efficiency"] for r in runs]
    # simplicity proxy: 1 - mean(bid/value std). Truth-telling / consistent
    # shading → low std → high simplicity.
    bv_stds = [r["bid_value_std"] for r in runs]
    simplicity_proxy = 1.0 - (sum(bv_stds) / len(bv_stds))
    # regret_var: variance of realized winner-profit (= max_val * eff - price) across runs.
    winner_profits = [
        (r["winner_value"] - r["price"]) if r["winner_value"] is not None else 0.0
        for r in runs
    ]
    regret_var = statistics.pvariance(winner_profits) if len(winner_profits) > 1 else 0.0

    # Normalization: revenue and regret_var scaled by max possible value
    # (private_range + common_range upper). For SPSB IPV with private_range=49,
    # max ~ 49. Use observed max_val for a slightly more realistic scale.
    max_val_obs = max((r["max_val"] for r in runs), default=1) or 1
    revenue_mean = sum(revenues) / len(revenues)
    rev_norm = revenue_mean / max_val_obs
    eff_mean = sum(effs) / len(effs)
    reg_norm = regret_var / (max_val_obs ** 2)

    w_rev, w_eff, w_simp, w_reg = weights
    # Penalty form for simplicity: incentivize HIGHER simplicity_proxy.
    scalar = (
        w_rev * rev_norm
        + w_eff * eff_mean
        + w_simp * simplicity_proxy
        - w_reg * reg_norm
    )

    result = {
        "yaml": os.path.relpath(yaml_path, REPO_ROOT),
        "n_runs": len(runs),
        "n_personas": len(cells),
        "reps_per_persona": reps_per_persona,
        "metrics": {
            "revenue_mean": revenue_mean,
            "revenue_normalized": rev_norm,
            "efficiency_mean": eff_mean,
            "simplicity_proxy": simplicity_proxy,
            "regret_var": regret_var,
            "regret_var_normalized": reg_norm,
            "max_val_observed": max_val_obs,
        },
        "weights": {"revenue": w_rev, "efficiency": w_eff,
                    "simplicity": w_simp, "regret": w_reg},
        "fitness": scalar,
        "per_run": runs,
    }
    return result


def _save(result, yaml_path):
    out_path = yaml_path.replace(".yaml", ".fitness.json")
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2, default=str)
    return out_path


def _expand(argv):
    out = []
    for a in argv:
        if any(ch in a for ch in "*?["):
            out.extend(sorted(_glob.glob(a)))
        else:
            out.append(a)
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("yaml_paths", nargs="+", help="mechanism YAML(s) to score")
    p.add_argument("--n_personas", type=int, default=27,
                   help="number of cognitive-basis cells to mix uniformly (max 27)")
    p.add_argument("--reps_per_persona", type=int, default=2)
    p.add_argument("--max_workers", type=int, default=4)
    p.add_argument("--seed_offset", type=int, default=0)
    p.add_argument("--w_rev", type=float, default=1.0)
    p.add_argument("--w_eff", type=float, default=1.0)
    p.add_argument("--w_simp", type=float, default=0.5)
    p.add_argument("--w_reg", type=float, default=0.25)
    args = p.parse_args()

    paths = _expand(args.yaml_paths)
    cache = Cache()
    summary = []
    for path in paths:
        if not os.path.exists(path):
            print(f"missing: {path}")
            continue
        result = evaluate_mechanism(
            path,
            n_personas=args.n_personas,
            reps_per_persona=args.reps_per_persona,
            max_workers=args.max_workers,
            seed_offset=args.seed_offset,
            weights=(args.w_rev, args.w_eff, args.w_simp, args.w_reg),
            cache=cache,
        )
        out = _save(result, path)
        print(f"[fitness] wrote {out}")
        m = result["metrics"]
        print(f"  fitness={result['fitness']:.4f} "
              f"rev={m['revenue_mean']:.2f} eff={m['efficiency_mean']:.3f} "
              f"simp={m['simplicity_proxy']:.3f} regvar={m['regret_var']:.3f}")
        summary.append({"yaml": os.path.basename(path),
                        "fitness": result["fitness"], **m})

    if len(summary) > 1:
        print("\n=== Summary ===")
        for s in sorted(summary, key=lambda x: -x["fitness"]):
            print(f"  {s['yaml']:60s}  f={s['fitness']:+.4f}")


if __name__ == "__main__":
    main()
