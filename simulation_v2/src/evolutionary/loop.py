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

from edsl import Cache, Model

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
sys.path.insert(0, os.path.dirname(__file__))

from axes import DESIGN_AXES  # noqa: E402
from crossover import crossover  # noqa: E402
from fitness import evaluate_mechanism  # noqa: E402
from mechanism_generator import (  # noqa: E402
    instantiate_mechanism,
    sample_coordinates,
    write_mechanism,
)
from mutation import mutate  # noqa: E402
import provenance  # noqa: E402
import selection  # noqa: E402


def _evaluate(yaml_path, n_personas, reps_per_persona, max_workers, weights, cache):
    return evaluate_mechanism(
        yaml_path,
        n_personas=n_personas,
        reps_per_persona=reps_per_persona,
        max_workers=max_workers,
        weights=weights,
        cache=cache,
        verbose=False,
    )


def _seed_gen_zero(*, pop_size, model, cache, rng, gen0_dir, gen_tag,
                   n_personas, reps_per_persona, max_workers, weights,
                   conn, run_id):
    coords = sample_coordinates(pop_size, DESIGN_AXES, seed=rng.randint(0, 2**31 - 1))
    print(f"[gen 0] sampled {len(coords)} design-axis coordinates")
    pop = []
    for i, c in enumerate(coords):
        try:
            mech = instantiate_mechanism(c, model)
        except Exception as e:
            print(f"[gen 0] instantiate failed at idx {i}: {type(e).__name__}: {e}")
            continue
        mech.setdefault("op", "init")
        entry = write_mechanism(mech, gen0_dir, i, gen_tag=gen_tag)
        try:
            result = _evaluate(entry["yaml"], n_personas, reps_per_persona,
                               max_workers, weights, cache)
        except Exception as e:
            print(f"[gen 0] fitness failed for {entry['base']}: {type(e).__name__}: {e}")
            continue
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
        print(f"[gen 0] [{i}] {mech['name']}  fit={result['fitness']:+.4f}  rev={result['metrics']['revenue_mean']:.2f}")
    return pop


def _make_child(pop, *, model, rng, mutation_rate):
    """Return a child mech dict via tournament-selected mutation OR crossover."""
    use_mutation = (rng.random() < mutation_rate) or (len({ind['rule_hash'] for ind in pop}) < 2)
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
                           conn, run_id):
    elites = selection.elitism(prev_pop, n=elite_size)
    print(f"[gen {gen}] elites: " + ", ".join(
        f"{e['mech']['name']}({e['fitness']:+.3f})" for e in elites
    ))

    next_pop = []
    # Carry elites unchanged; log to provenance with op="elite" for this gen.
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

    n_offspring = pop_size - elite_size
    next_idx = 0
    attempts = 0
    max_attempts = n_offspring * 4
    while len(next_pop) < pop_size and attempts < max_attempts:
        attempts += 1
        try:
            child = _make_child(prev_pop, model=model, rng=rng,
                                mutation_rate=mutation_rate)
        except Exception as e:
            print(f"[gen {gen}] operator failed: {type(e).__name__}: {e}")
            continue
        if any(child["rule_hash"] == ind["rule_hash"] for ind in next_pop):
            print(f"[gen {gen}] dedup skip: {child['rule_hash']}")
            continue
        entry = write_mechanism(child, gen_dir, next_idx, gen_tag=gen_tag)
        next_idx += 1
        try:
            result = _evaluate(entry["yaml"], n_personas, reps_per_persona,
                               max_workers, weights, cache)
        except Exception as e:
            print(f"[gen {gen}] fitness failed for {entry['base']}: {type(e).__name__}: {e}")
            continue
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
        print(f"[gen {gen}] [{next_idx-1}] {child.get('op')}/{child['name']}  fit={result['fitness']:+.4f}")

    return next_pop


def run(*, pop_size, n_generations, n_personas, reps_per_persona,
        max_workers, mutation_rate, elite_size, tournament_k,
        weights, model_name, temperature, seed, out_root, db_path):
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
        notes=f"out_dir={os.path.relpath(out_dir, REPO_ROOT)}",
    )
    print(f"    run_id = {run_id}")

    gen0_dir = os.path.join(out_dir, "gen_0")
    os.makedirs(gen0_dir, exist_ok=True)
    pop = _seed_gen_zero(
        pop_size=pop_size, model=model, cache=cache, rng=rng,
        gen0_dir=gen0_dir, gen_tag=f"{run_tag}/gen_0",
        n_personas=n_personas, reps_per_persona=reps_per_persona,
        max_workers=max_workers, weights=weights,
        conn=conn, run_id=run_id,
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
            conn=conn, run_id=run_id,
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
    p.add_argument("--w_simp", type=float, default=0.5)
    p.add_argument("--w_reg", type=float, default=0.25)
    p.add_argument("--model", default="gpt-5.4-mini")
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--seed", type=int, default=20260525)
    p.add_argument("--out_root", default=os.path.join(REPO_ROOT, "data_v2", "evo"))
    p.add_argument("--db_path", default=os.path.join(REPO_ROOT, "data_v2", "results.sqlite"))
    args = p.parse_args()

    run(
        pop_size=args.pop_size,
        n_generations=args.n_generations,
        n_personas=args.n_personas,
        reps_per_persona=args.reps_per_persona,
        max_workers=args.max_workers,
        mutation_rate=args.mutation_rate,
        elite_size=args.elite_size,
        tournament_k=args.tournament_k,
        weights=(args.w_rev, args.w_eff, args.w_simp, args.w_reg),
        model_name=args.model,
        temperature=args.temperature,
        seed=args.seed,
        out_root=args.out_root,
        db_path=args.db_path,
    )


if __name__ == "__main__":
    main()
