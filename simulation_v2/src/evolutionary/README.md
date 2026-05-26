# Evolutionary auction-rule generation

Stage D + Stage E of `writeup_v2/plan.md`. Adapted from Google DeepMind's Concordia
persona-generation pattern ([github](https://github.com/google-deepmind/concordia/tree/main/concordia/contrib/persona_generators),
[arXiv:2602.03545](https://arxiv.org/abs/2602.03545)) — but the *artifact* being
generated and evolved is an **auction mechanism** (Jinja2 rule template + YAML
parameters), not a persona. Personas are the *evaluators* in our setting (Stage C
calibrated mixture).

## Conceptual map: Concordia persona → our auction rule

| Concordia (persona) | Ours (auction rule) |
|---|---|
| `initial_context` = social setting | "single-item sealed-bid auction" |
| `diversity_axes` = personality traits | design-space knobs (pricing, payment, info, simplicity) |
| Stage 1: pre-sample trait coordinates | pre-sample coordinates in design space (Latin-hypercube-ish) |
| Stage 1.5: archetype priming on each axis | optional: describe "extreme" mechanism at each axis end |
| Stage 2: instantiate one persona per coord | instantiate one mechanism YAML + template per coord |
| Generator variants `alphaevolve_1…5` | future: evolved generator prompts found by Stage E loop |

## Component breakdown

```
┌─────────────────────────────────────────────────────────────────────────┐
│  Component 1: Diverse Initial-Population Generator  (Stage D)          │
│  - design-space axes (from writeup_v2/design_space.md, currently stub) │
│  - sample N coordinates in axis space                                  │
│  - per coordinate: LLM instantiates rule template + YAML               │
│  - schema-validates output                                             │
│  - returns Population of candidate mechanisms                          │
└─────────────────────────────────────────────────────────────────────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  Component 2: Fitness Evaluator   (Stage E inner loop)                 │
│  - input: mechanism (template + YAML)                                  │
│  - run K simulations with calibrated-persona mixture from Stage C      │
│  - compute (revenue, efficiency, simplicity_proxy, regret_var)         │
│  - aggregate into a scalar (Pareto rank, or weighted sum)              │
└─────────────────────────────────────────────────────────────────────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  Component 3: Variation Operators   (Stage E outer loop)               │
│  - mutation:  LLM-driven small edit on Jinja2 rule template            │
│               + parameter mutation (price rule, reserve, etc.)         │
│  - crossover: LLM-driven blend of two parents' rule templates          │
│  - new individuals are schema-validated; failures rejected             │
└─────────────────────────────────────────────────────────────────────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  Component 4: Selection + Generational Loop   (Stage E)                │
│  - tournament selection (k=3) → top half survives                      │
│  - elitism: keep best-2 unchanged                                      │
│  - dedup via rule_text hash                                            │
│  - run N generations (proof-of-concept: 5)                             │
│  - persist population + lineage in `data_v2/results.sqlite`            │
└─────────────────────────────────────────────────────────────────────────┘
```

## What's implemented in this folder right now

- `axes.py` — design-space axis spec (preliminary; awaiting Stage B literature pass).
- `mechanism_generator.py` — **Component 1** (Stage D). Diverse initial-population
  generator: stratified-random axis sampling, then per-coord LLM instantiation of a
  Jinja2 rule template + YAML config compatible with the v1 batch runner.
- `fitness.py` — **Component 2** (Stage E inner loop). Uniform mixture over the 27-cell
  cognitive basis (Stage C calibration pending). Computes (revenue, efficiency,
  simplicity_proxy, regret_var); scalar = weighted sum.
- `mutation.py` — **Component 3a**. LLM-driven small-edit operator on a single parent
  mechanism: rewords 1-2 sentences of rule_text, flips one design knob, or perturbs
  scalar params. Schema-validated.
- `crossover.py` — **Component 3b**. Gene-wise mix of two parents' yaml_overrides, then
  LLM rewrites rule_text to be consistent with the mixed param vector.
- `selection.py` — **Component 4a**. Tournament-k (default 3), elitism (default top-2),
  rule_hash dedup. Pure-functional.
- `provenance.py` — **Component 4b**. SQLite at `data_v2/results.sqlite`: `runs` and
  `individuals` tables with full lineage (parent_hashes per child).
- `loop.py` — **Component 4c**. Generational driver. Seeds gen 0 → evaluates fitness →
  tournament + mutation/crossover → repeats. Per-individual provenance logged.

## What's still a gap

- **Stage C calibration**: persona-mixture weights are still uniform across the 27 cells.
  Real moment-matching against e.g. eBay data is the next big thing.
- **Stage B literature pass**: `axes.py` is the preliminary axis set. A grounded
  `writeup_v2/design_space_schema.yaml` will replace it.
- **Scale**: the full plan.md spec is pop=50 × gens=10 × personas=27 × reps≥3 ≈ 300k LLM
  calls. We're currently exercising the loop at debug scale (5×2×3×1 ≈ 110 calls).

## Quick start

```bash
# Stage D only — generate diverse initial population:
./venv/bin/python simulation_v2/src/evolutionary/mechanism_generator.py --n 5
# → data_v2/evo/gen0/mech_*.yaml + mech_*.txt + manifest.json

# Score one (or many) mechanisms against the 27-cell persona mixture:
./venv/bin/python simulation_v2/src/evolutionary/fitness.py \
    'data_v2/evo/gen0/mech_*.yaml' --n_personas 3 --reps_per_persona 1
# → mech_*.fitness.json next to each YAML

# Full Stage E loop, debug scale (~110 LLM calls, a few minutes):
./venv/bin/python simulation_v2/src/evolutionary/loop.py \
    --pop_size 5 --n_generations 2 --n_personas 3 --reps_per_persona 1 \
    --max_workers 4 --seed 20260525
# → data_v2/evo/run_<timestamp>_p5g2s.../gen_{0,1,2}/mech_*.{yaml,txt}
# → data_v2/results.sqlite  (queryable lineage: provenance.top_k / provenance.lineage)
```

After a run, inspect the DB:

```bash
./venv/bin/python -c "
import sys; sys.path.insert(0, 'simulation_v2/src/evolutionary')
import provenance
conn = provenance.connect('data_v2/results.sqlite')
for row in provenance.top_k(conn, run_id=1, k=10):
    print(row)
"
```

## References

- Concordia persona generators: <https://github.com/google-deepmind/concordia/blob/main/concordia/contrib/persona_generators/generate_personas.py>
- Persona Generators paper (2026): <https://arxiv.org/abs/2602.03545>
- AlphaEvolve (the prompt-evolution method that produced the `alphaevolve_*` variants):
  Concordia's repo links to the underlying technique
- This codebase's Stage B design-space stub: `writeup_v2/design_space.md`
