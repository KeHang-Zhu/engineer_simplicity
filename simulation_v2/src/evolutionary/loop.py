"""
Component 4c of Stage E: generational EA driver.

Pipeline per run:
  gen 0: sample N coordinates in the design-axis space, instantiate via LLM
         (mechanism_generator.instantiate_mechanism), write to disk, evaluate
         fitness against the 27-cell cognitive-basis persona mixture.
  gen i (i = 1..N): keep best-`elite_size` unchanged, fill the rest with
         children produced via tournament-selected mutation (rate p) /
         crossover (rate 1-p). Each child is schema-validated upstream by
         the operator, written to disk, evaluated.
  At each step, individuals are logged to a SQLite provenance DB
  (`data_v2/results.sqlite` by default).

Run from repo root (debug scale):
    ./venv/bin/python simulation_v2/src/evolutionary/loop.py \
        --pop_size 5 --n_generations 2 --n_personas 3 --reps_per_persona 1 \
        --max_workers 4

The driver is deterministic in its non-LLM choices (selection, gene mix) given
`--seed`. LLM outputs are stochastic — set `--temperature 0` for closer to
deterministic LLM behaviour, but variety in mutation/crossover is desirable.
"""

import argparse
import os
import random
import sys
import time

import yaml
from edsl import Cache, Model

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
sys.path.insert(0, os.path.dirname(__file__))

from axes import DESIGN_AXES  # noqa: E402
from crossover import crossover  # noqa: E402
from fitness import evaluate_mechanism, load_stage_e_weights, save_result  # noqa: E402
from parallel_mice import evaluate_population  # noqa: E402
from mechanism_generator import (  # noqa: E402
    instantiate_genotype,
    instantiate_mechanism,
    sample_coordinates,
    write_mechanism,
)
from genotype import (  # noqa: E402
    crossover_genotypes,
    mutate_genotype,
    sample_genotypes,
    validate_genotype,
    validate_genotype_scope,
)
from mutation import mutate  # noqa: E402
import provenance  # noqa: E402
import selection  # noqa: E402


def _evaluate(yaml_path, n_personas, reps_per_persona, max_workers, weights, cache,
              persona_manifest):
    return evaluate_mechanism(
        yaml_path,
        n_personas=n_personas,
        reps_per_persona=reps_per_persona,
        max_workers=max_workers,
        weights=weights,
        cache=cache,
        persona_manifest=persona_manifest,
        verbose=False,
    )


def _score_entries(yaml_paths, *, n_personas, reps_per_persona, max_workers,
                   weights, cache, persona_manifest, batch_scoring,
                   edsl_max_concurrent_tasks=None):
    """Score a generation's mechanisms and return {yaml_path: result_dict}.

    When `batch_scoring` is set, all mechanisms are scored in ONE batched async
    EDSL pass (parallel_mice.evaluate_population) for first-price single-round
    auctions; otherwise each mechanism is scored serially (original behavior).
    Result dicts mirror fitness.evaluate_mechanism; a mechanism that produced no
    scoreable runs carries an `error` key instead of `fitness`.
    """
    if not yaml_paths:
        return {}
    if batch_scoring:
        try:
            return evaluate_population(
                yaml_paths,
                n_personas=n_personas,
                reps_per_persona=reps_per_persona,
                weights=weights,
                cache=cache,
                persona_manifest=persona_manifest,
                max_concurrent=edsl_max_concurrent_tasks,
                verbose=False,
            )
        except Exception as e:  # noqa: BLE001
            print(f"[score] batch scoring failed ({type(e).__name__}: {e}); "
                  f"falling back to serial")
    results = {}
    for path in yaml_paths:
        try:
            results[path] = _evaluate(path, n_personas, reps_per_persona,
                                      max_workers, weights, cache, persona_manifest)
        except Exception as e:  # noqa: BLE001
            results[path] = {"error": f"{type(e).__name__}: {e}"}
    return results


def _seed_gen_zero(*, pop_size, model, cache, rng, gen0_dir, gen_tag,
                   n_personas, reps_per_persona, max_workers, weights,
                   persona_manifest, conn, run_id, genotype_rows=None,
                   batch_scoring=False, edsl_max_concurrent_tasks=None):
    if genotype_rows is not None:
        rows = genotype_rows[:pop_size]
        genotype_mode = True
        print(f"[gen 0] loaded {len(rows)} structured genotypes")
    else:
        rows = sample_coordinates(pop_size, DESIGN_AXES, seed=rng.randint(0, 2**31 - 1))
        genotype_mode = False
        print(f"[gen 0] sampled {len(rows)} design-axis coordinates")

    # Phase A: instantiate and write every viable mechanism (no scoring yet).
    candidates = []
    for i, row in enumerate(rows):
        mech = None
        last_error = None
        for attempt in range(1, 4):
            try:
                mech = (
                    instantiate_genotype(row, model)
                    if genotype_mode
                    else instantiate_mechanism(row, model)
                )
                break
            except Exception as e:
                last_error = e
                print(
                    f"[gen 0] instantiate failed at idx {i} "
                    f"(attempt {attempt}/3): {type(e).__name__}: {e}"
                )
        if mech is None:
            print(
                f"[gen 0] giving up at idx {i}: "
                f"{type(last_error).__name__}: {last_error}"
            )
            continue
        mech.setdefault("op", "init")
        entry = write_mechanism(mech, gen0_dir, i, gen_tag=gen_tag)
        candidates.append({"mech": mech, "entry": entry, "idx": i})

    # Phase B: score the whole generation (one batched pass when enabled).
    results = _score_entries(
        [c["entry"]["yaml"] for c in candidates],
        n_personas=n_personas, reps_per_persona=reps_per_persona,
        max_workers=max_workers, weights=weights, cache=cache,
        persona_manifest=persona_manifest, batch_scoring=batch_scoring,
        edsl_max_concurrent_tasks=edsl_max_concurrent_tasks,
    )

    # Phase C: build the population and record provenance.
    pop = []
    for c in candidates:
        entry, mech = c["entry"], c["mech"]
        result = results.get(entry["yaml"])
        if not result or "fitness" not in result:
            err = (result or {}).get("error", "no result")
            print(f"[gen 0] fitness failed for {entry['base']}: {err}")
            continue
        save_result(result, entry["yaml"])
        ind = {
            "mech": mech,
            "rule_hash": mech["rule_hash"],
            "yaml_path": entry["yaml"],
            "tpl_path": entry["tpl"],
            "fitness": result["fitness"],
            "metrics": result["metrics"],
            "op": "init",
            "generation": 0,
        }
        provenance.record_individual(
            conn, run_id=run_id, generation=0, mech=mech,
            fitness=result["fitness"], metrics=result["metrics"],
            yaml_path=entry["yaml"], tpl_path=entry["tpl"],
        )
        pop.append(ind)
        print(f"[gen 0] [{c['idx']}] {mech['name']}  fit={result['fitness']:+.4f}  rev={result['metrics']['revenue_mean']:.2f}")
    return pop


def _make_child(pop, *, model, rng, mutation_rate, genotype_mode=False,
                genotype_scope="unrestricted"):
    """Return a child mech dict via tournament-selected mutation OR crossover."""
    use_mutation = (rng.random() < mutation_rate) or (len({ind['rule_hash'] for ind in pop}) < 2)
    if genotype_mode:
        if use_mutation:
            parent = selection.tournament(pop, k=3, rng=rng)
            child_genotype = mutate_genotype(
                parent["mech"]["genotype"], rng=rng, scope=genotype_scope
            )
            child = instantiate_genotype(child_genotype, model)
            child["op"] = "genotype_mutation"
            child["parent_hashes"] = [parent.get("rule_hash", "")]
            return child
        pa, pb = selection.select_parents(pop, n=2, k=3, rng=rng)
        tries = 0
        while pa["rule_hash"] == pb["rule_hash"] and tries < 5:
            pb = selection.tournament(pop, k=3, rng=rng)
            tries += 1
        child_genotype = crossover_genotypes(
            pa["mech"]["genotype"], pb["mech"]["genotype"], rng=rng,
            scope=genotype_scope,
        )
        child = instantiate_genotype(child_genotype, model)
        child["op"] = "genotype_crossover"
        child["parent_hashes"] = [pa.get("rule_hash", ""), pb.get("rule_hash", "")]
        return child

    if use_mutation:
        parent = selection.tournament(pop, k=3, rng=rng)
        child = mutate(parent["mech"], model)
        return child
    pa, pb = selection.select_parents(pop, n=2, k=3, rng=rng)
    tries = 0
    while pa["rule_hash"] == pb["rule_hash"] and tries < 5:
        pb = selection.tournament(pop, k=3, rng=rng)
        tries += 1
    child = crossover(pa["mech"], pb["mech"], model, rng=rng)
    return child


def _evolve_one_generation(prev_pop, *, gen, pop_size, elite_size, mutation_rate,
                           model, cache, rng, gen_dir, gen_tag,
                           n_personas, reps_per_persona, max_workers, weights,
                           persona_manifest, conn, run_id, genotype_mode=False,
                           genotype_scope="unrestricted", batch_scoring=False,
                           edsl_max_concurrent_tasks=None):
    elites = selection.elitism(prev_pop, n=elite_size)
    print(f"[gen {gen}] elites: " + ", ".join(
        f"{e['mech']['name']}({e['fitness']:+.3f})" for e in elites
    ))

    next_pop = []
    # Carry elites unchanged; log to provenance with op="elite" for this gen.
    seen_hashes = set()
    for e in elites:
        elite_record = dict(e["mech"])
        elite_record["op"] = "elite"
        elite_record["parent_hashes"] = [e["rule_hash"]]
        provenance.record_individual(
            conn, run_id=run_id, generation=gen, mech=elite_record,
            fitness=e["fitness"], metrics=e["metrics"],
            yaml_path=e["yaml_path"], tpl_path=e["tpl_path"],
        )
        next_pop.append({**e, "generation": gen})
        seen_hashes.add(e["rule_hash"])

    # Phase A: produce and write all offspring for this generation (no scoring).
    n_offspring = pop_size - elite_size
    candidates = []
    next_idx = 0
    attempts = 0
    max_attempts = n_offspring * 4
    while len(candidates) < n_offspring and attempts < max_attempts:
        attempts += 1
        try:
            child = _make_child(prev_pop, model=model, rng=rng,
                                mutation_rate=mutation_rate,
                                genotype_mode=genotype_mode,
                                genotype_scope=genotype_scope)
        except Exception as e:
            print(f"[gen {gen}] operator failed: {type(e).__name__}: {e}")
            continue
        if child["rule_hash"] in seen_hashes:
            print(f"[gen {gen}] dedup skip: {child['rule_hash']}")
            continue
        seen_hashes.add(child["rule_hash"])
        entry = write_mechanism(child, gen_dir, next_idx, gen_tag=gen_tag)
        candidates.append({"mech": child, "entry": entry, "idx": next_idx})
        next_idx += 1

    # Phase B: score the whole generation (one batched pass when enabled).
    results = _score_entries(
        [c["entry"]["yaml"] for c in candidates],
        n_personas=n_personas, reps_per_persona=reps_per_persona,
        max_workers=max_workers, weights=weights, cache=cache,
        persona_manifest=persona_manifest, batch_scoring=batch_scoring,
        edsl_max_concurrent_tasks=edsl_max_concurrent_tasks,
    )

    # Phase C: build the population and record provenance.
    for c in candidates:
        entry, child = c["entry"], c["mech"]
        result = results.get(entry["yaml"])
        if not result or "fitness" not in result:
            err = (result or {}).get("error", "no result")
            print(f"[gen {gen}] fitness failed for {entry['base']}: {err}")
            continue
        save_result(result, entry["yaml"])
        ind = {
            "mech": child,
            "rule_hash": child["rule_hash"],
            "yaml_path": entry["yaml"],
            "tpl_path": entry["tpl"],
            "fitness": result["fitness"],
            "metrics": result["metrics"],
            "op": child.get("op", "child"),
            "generation": gen,
        }
        provenance.record_individual(
            conn, run_id=run_id, generation=gen, mech=child,
            fitness=result["fitness"], metrics=result["metrics"],
            yaml_path=entry["yaml"], tpl_path=entry["tpl"],
        )
        next_pop.append(ind)
        print(f"[gen {gen}] [{c['idx']}] {child.get('op')}/{child['name']}  fit={result['fitness']:+.4f}")

    return next_pop


def run(*, pop_size, n_generations, n_personas, reps_per_persona,
        max_workers, mutation_rate, elite_size, tournament_k,
        weights, model_name, temperature, seed, out_root, db_path,
        persona_manifest=None, genotype_rows=None, genotype_scope="unrestricted",
        batch_scoring=False, edsl_max_concurrent_tasks=None):
    rng = random.Random(seed)
    model = Model(model_name, temperature=temperature)
    cache = Cache()

    run_tag = time.strftime("run_%Y%m%d_%H%M%S") + f"_p{pop_size}g{n_generations}s{seed}"
    out_dir = os.path.join(out_root, run_tag)
    os.makedirs(out_dir, exist_ok=True)
    print(f"=== EA run: {run_tag} ===")
    print(f"    out_dir = {out_dir}")
    print(f"    db_path = {db_path}")

    conn = provenance.connect(db_path)
    run_id = provenance.start_run(
        conn,
        pop_size=pop_size, n_generations=n_generations,
        n_personas=n_personas, reps_per_persona=reps_per_persona,
        model=model_name, seed=seed,
        notes=(
            f"out_dir={os.path.relpath(out_dir, REPO_ROOT)}; "
            f"persona_manifest={persona_manifest or 'default_27_cell_basis'}; "
            f"genotype_mode={genotype_rows is not None}; "
            f"genotype_scope={genotype_scope}"
        ),
    )
    print(f"    run_id = {run_id}")

    gen0_dir = os.path.join(out_dir, "gen_0")
    os.makedirs(gen0_dir, exist_ok=True)
    pop = _seed_gen_zero(
        pop_size=pop_size, model=model, cache=cache, rng=rng,
        gen0_dir=gen0_dir, gen_tag=f"{run_tag}/gen_0",
        n_personas=n_personas, reps_per_persona=reps_per_persona,
        max_workers=max_workers, weights=weights,
        persona_manifest=persona_manifest,
        conn=conn, run_id=run_id,
        genotype_rows=genotype_rows,
        batch_scoring=batch_scoring,
        edsl_max_concurrent_tasks=edsl_max_concurrent_tasks,
    )
    if len(pop) < 2:
        print(f"!!! gen 0 returned only {len(pop)} viable individuals — aborting")
        return run_id

    for gen in range(1, n_generations + 1):
        gen_dir = os.path.join(out_dir, f"gen_{gen}")
        os.makedirs(gen_dir, exist_ok=True)
        pop = _evolve_one_generation(
            pop, gen=gen, pop_size=pop_size,
            elite_size=elite_size, mutation_rate=mutation_rate,
            model=model, cache=cache, rng=rng,
            gen_dir=gen_dir, gen_tag=f"{run_tag}/gen_{gen}",
            n_personas=n_personas, reps_per_persona=reps_per_persona,
            max_workers=max_workers, weights=weights,
            persona_manifest=persona_manifest,
            conn=conn, run_id=run_id,
            genotype_mode=genotype_rows is not None,
            genotype_scope=genotype_scope,
            batch_scoring=batch_scoring,
            edsl_max_concurrent_tasks=edsl_max_concurrent_tasks,
        )

    print(f"\n=== top-{min(5, pop_size)} across all generations (run_id={run_id}) ===")
    for row in provenance.top_k(conn, run_id, k=min(5, pop_size * (n_generations + 1))):
        print(f"  gen{row['generation']:>2}  {row['op']:>9}  f={row['fitness']:+.4f}  "
              f"{row['rule_hash']}  {row['name']}")
    return run_id


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--pop_size", type=int, default=5)
    p.add_argument("--n_generations", type=int, default=2)
    p.add_argument("--n_personas", type=int, default=3)
    p.add_argument("--reps_per_persona", type=int, default=1)
    p.add_argument("--max_workers", type=int, default=4)
    p.add_argument("--mutation_rate", type=float, default=0.7,
                   help="fraction of children produced via mutation; rest via crossover")
    p.add_argument("--elite_size", type=int, default=2)
    p.add_argument("--tournament_k", type=int, default=3)
    p.add_argument("--w_rev", type=float, default=1.0)
    p.add_argument("--w_eff", type=float, default=1.0)
    p.add_argument("--w_intended", type=float, default=0.0)
    p.add_argument("--w_action_compliance", type=float, default=1.0)
    p.add_argument("--w_simp", type=float, default=0.5)
    p.add_argument("--w_reg", type=float, default=0.25)
    p.add_argument("--w_fail", type=float, default=1.0)
    p.add_argument("--w_worst_eff", type=float, default=0.0)
    p.add_argument("--w_p20_eff", type=float, default=0.0)
    p.add_argument("--w_min_simp", type=float, default=0.0)
    p.add_argument("--w_persona_var", type=float, default=0.0)
    p.add_argument("--weights_yaml",
                   help="optional YAML with stage_e_weights mapping")
    p.add_argument("--model", default="gpt-5.4-mini")
    p.add_argument("--persona_manifest",
                   help="optional CSV with persona_id and persona_file columns")
    p.add_argument("--genotypes",
                   help="optional YAML file with a `genotypes` list for structured genotype EA")
    p.add_argument("--sample_genotypes", action="store_true",
                   help="use deterministic single-item auction genotype samples for gen 0")
    p.add_argument("--genotype_scope", default="unrestricted",
                   choices=["unrestricted", "first_price_ipv", "second_price_ipv"],
                   help="lock structural genotype axes during mutation/crossover")
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--batch_scoring", action="store_true",
                   help="score each generation in one batched EDSL async pass "
                        "(first_price_ipv single-round scope; cache-identical to serial)")
    p.add_argument("--edsl_max_concurrent_tasks", type=int,
                   help="temporary EDSL async concurrency cap when --batch_scoring is used")
    p.add_argument("--seed", type=int, default=20260525)
    p.add_argument("--out_root", default=os.path.join(REPO_ROOT, "data_v2", "evo"))
    p.add_argument("--db_path", default=os.path.join(REPO_ROOT, "data_v2", "results.sqlite"))
    args = p.parse_args()

    weights = (
        load_stage_e_weights(args.weights_yaml)
        if args.weights_yaml
        else {
            "revenue": args.w_rev,
            "efficiency": args.w_eff,
            "intended_action": args.w_intended,
            "action_compliance": args.w_action_compliance,
            "simplicity": args.w_simp,
            "regret": args.w_reg,
            "failure": args.w_fail,
            "worst_persona_efficiency": args.w_worst_eff,
            "p20_persona_efficiency": args.w_p20_eff,
            "min_persona_simplicity": args.w_min_simp,
            "persona_profit_variance": args.w_persona_var,
        }
    )
    genotype_rows = None
    if args.genotypes:
        with open(args.genotypes) as f:
            data = yaml.safe_load(f)
        rows = data.get("genotypes", data) if isinstance(data, dict) else data
        if not isinstance(rows, list):
            raise ValueError("--genotypes must point to a list or a mapping with `genotypes`")
        genotype_rows = [validate_genotype_scope(row, args.genotype_scope) for row in rows]
    elif args.sample_genotypes:
        genotype_rows = sample_genotypes(args.pop_size, seed=args.seed)

    run(
        pop_size=args.pop_size,
        n_generations=args.n_generations,
        n_personas=args.n_personas,
        reps_per_persona=args.reps_per_persona,
        max_workers=args.max_workers,
        mutation_rate=args.mutation_rate,
        elite_size=args.elite_size,
        tournament_k=args.tournament_k,
        weights=weights,
        model_name=args.model,
        temperature=args.temperature,
        seed=args.seed,
        out_root=args.out_root,
        db_path=args.db_path,
        persona_manifest=args.persona_manifest,
        genotype_rows=genotype_rows,
        genotype_scope=args.genotype_scope,
        batch_scoring=args.batch_scoring,
        edsl_max_concurrent_tasks=args.edsl_max_concurrent_tasks,
    )


if __name__ == "__main__":
    main()
