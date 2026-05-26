# simulation_v2/

v2 simulation code. Imports v1 framework (`simulation_v1.src.util_plan`, `util_da`) and adds new modules per the [plan](../writeup_v2/plan.md):

- `src/english_auction.py` — Stage A new mechanism
- `src/calibration/` — Stage C moment-matching package
- `src/design_proposer/` — Stage D LLM-as-mechanism-designer
- `src/evolutionary/` — Stage E AlphaProof-style loop
- `src/async_batch.py`, `src/persona_schema.py` — cross-cutting infra

**v1 is frozen.** Do not modify files in `simulation_v1/`; import from it instead.
