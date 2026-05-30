"""
SMAD-only fitness for the digital-mice EA.

The v1 fitness was a multi-objective weighted sum (revenue, efficiency,
intended action, simplicity, regret, parser failure, action noncompliance,
tail penalties). The v2 round-2 evidence (Section 6, CoT panel) showed that
the contingent-reasoning axis is the only construct that moves behavior in
the panel, and that the panel collapses into two clusters along that axis.
The right EA objective in this regime is much simpler: find mechanisms
whose realized bids stay close to the mechanism's own declared optimum
across a small, behaviorally distinct screening panel -- one stressor per
documented mistake class, with c0 as the primary stressor.

The single scalar is SMAD (scaled mean absolute deviation), aggregated
across the panel with a worst-strain (max) operator so a mechanism wins
only if it works for every stressor:

    SMAD_jk = 100 * mean_a |b_a - b*(v_a)| / mean_a |b*(v_a)|
    SMAD_k  = max_j SMAD_jk             (worst-strain robustness)
    fitness_k = -SMAD_k                  (lower is better, so we negate)

The optimal-bid function b*(v) is mechanism-specific. The EA auto-writes
a per-mechanism Python module (one .py per candidate) that defines
``optimal_bid(v, n)`` so the SMAD computation can be inspected and audited
mechanism by mechanism, instead of buried inside a generic scorer.

Usage:
    from simulation_v2.src.evolutionary.smad_scoring import (
        write_smad_module, smad_for_runs,
    )

    module_path = write_smad_module(genotype, out_dir)
    smad = smad_for_runs(runs, module_path)
"""

import importlib.util
import os
import re
import textwrap

import yaml


# -- Optimal-bid formulas per payment rule -----------------------------------
# Closed-form risk-neutral best response for the canonical sealed-bid
# mechanisms with i.i.d. values on a bounded support. n is the bidder count.
PAYMENT_RULE_FORMULAS = {
    "first_price":   "((n - 1) / n) * v",
    "second_price":  "v",
    "third_price":   "v + (v - v) * 0",  # third-price has no closed-form best
    "all_pay":       "v ** 2 / (2 * v if v else 1)",  # placeholder
    "average_price": "v",  # rough; depends on opponent distribution
    "posted_price":  "v",  # binary accept/reject, accept iff v >= price
}


def _payment_rule(genotype):
    return (
        genotype.get("mechanism", {}).get("payment_rule")
        or genotype.get("payment_rule")
        or "first_price"
    )


def _benchmark_expr(genotype):
    """Read target_policy.benchmark if present; otherwise look up by payment rule."""
    target_policy = genotype.get("target_policy") or {}
    benchmark = target_policy.get("benchmark")
    if benchmark:
        # The genotype string uses 'value' as the variable; rename to v.
        return benchmark.replace("value", "v")
    rule = _payment_rule(genotype)
    return PAYMENT_RULE_FORMULAS.get(rule, "v")


def _safe_id(name):
    return re.sub(r"[^A-Za-z0-9_]", "_", str(name))


def write_smad_module(genotype, out_dir, n_bidders=3, mechanism_id=None):
    """Emit a per-mechanism .py with an ``optimal_bid(v, n)`` function.

    Returns the path to the written module.
    """
    os.makedirs(out_dir, exist_ok=True)
    if mechanism_id is None:
        mechanism_id = (
            genotype.get("experiment", {}).get("name")
            or genotype.get("mechanism_id")
            or "candidate"
        )
    mechanism_id = _safe_id(mechanism_id)

    rule = _payment_rule(genotype)
    expr = _benchmark_expr(genotype)

    source = textwrap.dedent(f'''\
        """Auto-generated SMAD scoring module for mechanism: {mechanism_id}

        Payment rule: {rule}
        Optimal-bid expression: b*(v) = {expr}
        Default bidder count: n = {n_bidders}

        SMAD = 100 * mean_a |b_a - b*(v_a)| / mean_a |b*(v_a)|.
        Lower SMAD is better; SMAD = 0 means realized bids equal the
        mechanism's declared optimum.
        """

        N = {n_bidders}


        def optimal_bid(v, n=N):
            """Risk-neutral best response for this mechanism at value v."""
            return {expr}


        def smad(value_bid_pairs):
            """Compute SMAD given a list of (value, bid) pairs."""
            errors = []
            benchmarks = []
            for v, b in value_bid_pairs:
                if v is None or b is None or v <= 0:
                    continue
                b_star = optimal_bid(v)
                errors.append(abs(b - b_star))
                benchmarks.append(abs(b_star))
            if not benchmarks:
                return None
            mean_b = sum(benchmarks) / len(benchmarks)
            if mean_b == 0:
                return None
            return 100.0 * (sum(errors) / len(errors)) / mean_b
        ''')

    out_path = os.path.join(out_dir, f"smad_{mechanism_id}.py")
    with open(out_path, "w") as fp:
        fp.write(source)
    return out_path


def _load_module(path):
    spec = importlib.util.spec_from_file_location(
        f"_smad_{_safe_id(os.path.basename(path))}", path
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def smad_for_runs(runs, module_path, group_by="persona"):
    """Compute SMAD per group (default: per persona) and aggregate.

    runs: list of dicts, each with keys 'persona', 'values', 'bids' (lists).
    module_path: path to a per-mechanism module produced by write_smad_module.

    Returns: dict with per-strain SMAD and aggregate worst-strain SMAD.
    """
    mod = _load_module(module_path)
    by_group = {}
    for r in runs:
        key = r.get(group_by, "_pooled")
        values = r.get("values", []) or r.get("value", [])
        bids = r.get("bids", []) or [b["bid"] for b in r.get("history", {}).get("bidding history", [])]
        for v, b in zip(values, bids):
            by_group.setdefault(key, []).append((v, b))
    per_strain = {k: mod.smad(pairs) for k, pairs in by_group.items()}
    # Strip None entries before aggregating.
    valid = [v for v in per_strain.values() if v is not None]
    if not valid:
        return {"per_strain_smad": per_strain, "worst_strain_smad": None,
                "mean_smad": None, "fitness": None}
    return {
        "per_strain_smad": per_strain,
        "worst_strain_smad": max(valid),
        "mean_smad": sum(valid) / len(valid),
        "fitness": -max(valid),  # higher is better
    }


def evaluate_candidate_smad(candidate_yaml_path, runs, out_dir):
    """High-level entry point: load genotype, write SMAD module, compute SMAD."""
    with open(candidate_yaml_path) as fp:
        genotype = yaml.safe_load(fp)
    module_path = write_smad_module(genotype, out_dir)
    return module_path, smad_for_runs(runs, module_path)


if __name__ == "__main__":
    # Quick smoke test using the existing fpsb_stress YAML.
    import json
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
    sample_yaml = os.path.join(
        repo_root, "config_v2/configs_auction/stress_test_first_price/fpsb_stress.yaml"
    )
    out_dir = os.path.join(repo_root, "data_v2/results/smad_scoring_demo")
    with open(sample_yaml) as fp:
        g = yaml.safe_load(fp)
    p = write_smad_module(g, out_dir, mechanism_id="fpsb_stress")
    print(f"wrote {p}")
    with open(p) as fp:
        print(fp.read())
