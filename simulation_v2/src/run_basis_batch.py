"""
v2 batch runner for the cognitive-basis grid (Stage A v2).

Wraps simulation_v1.src.run_auction_batch but overrides `rule.persona` with the
per-config persona file (config['prompt']['persona_file'], path relative to repo root).
The v1 SPSB rule template is reused unchanged — only persona varies across the grid.

Run from repo root, e.g.:
    ./venv/bin/python simulation_v2/src/run_basis_batch.py \
        config_v2/configs_auction/interventions_gpt5mini/spsb_basis_c0_f0_b0.yaml

    # or glob:
    ./venv/bin/python simulation_v2/src/run_basis_batch.py \
        config_v2/configs_auction/interventions_gpt5mini/spsb_basis_c*.yaml
"""

import concurrent.futures
import glob as _glob
import os
import sys
import pandas as pd
import yaml

from edsl import Cache

# Make v1 source importable
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(REPO_ROOT, "simulation_v1", "src"))

from util_plan import Auction_plan, Rule_plan  # type: ignore  # noqa: E402


def load_config(config_path):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def load_persona(persona_file):
    """Read a persona file, resolving paths relative to repo root if not absolute."""
    if os.path.isabs(persona_file):
        path = persona_file
    else:
        path = os.path.join(REPO_ROOT, persona_file)
    with open(path) as f:
        return f.read()


def run_single_experiment(i, config, cache):
    a = config["auction"]
    r = config["rule"]
    v = config["value"]
    l = config["llm"]
    p = config["prompt"]
    e = config["execution"]

    rule = Rule_plan(
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

    # v2 override: replace the default persona.txt content with the per-basis persona.
    persona_file = p.get("persona_file")
    if persona_file:
        rule.persona = load_persona(persona_file)

    timestring = pd.Timestamp.now().strftime("%Y-%m-%d_%H-%M-%S-%f")
    auction = Auction_plan(
        number_agents=a["number_agents"],
        rule=rule,
        output_dir=e["output_dir"],
        timestring=timestring,
        cache=cache,
        model=l["model"],
        temperature=l["temperature"],
        service_name=l.get("service_name"),
    )
    auction.draw_value(seed=v["seed_base"] + i)
    auction.run_repeated()
    cache.write_jsonl(os.path.join(e["output_dir"], f"raw_output__{timestring}.jsonl"))
    print(f"[done] {config['experiment']['name']} rep {i+1}")


def run_config(config_path):
    print(f"\n{'='*70}\nLoading: {config_path}\n{'='*70}")
    cfg = load_config(config_path)
    print(f"name={cfg['experiment']['name']}  model={cfg['llm']['model']}  "
          f"persona={cfg['prompt'].get('persona_file', '(v1 default)')}  "
          f"reps={cfg['execution']['repetitions']}")
    os.makedirs(cfg["execution"]["output_dir"], exist_ok=True)
    cache = Cache()
    reps = cfg["execution"]["repetitions"]
    parallel = cfg["execution"].get("parallel", False)
    max_workers = cfg["execution"].get("max_workers", 4)

    if parallel:
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as ex:
            futs = [ex.submit(run_single_experiment, i, cfg, cache) for i in range(reps)]
            for fut in concurrent.futures.as_completed(futs):
                try:
                    fut.result()
                except Exception as exc:
                    import traceback
                    print(f"error: {exc}")
                    traceback.print_exc()
    else:
        for i in range(reps):
            try:
                run_single_experiment(i, cfg, cache)
            except Exception as exc:
                import traceback
                print(f"error rep {i}: {exc}")
                traceback.print_exc()
    print(f"{'='*70}\ndone: {cfg['experiment']['name']}\n{'='*70}")


def expand_args(argv):
    out = []
    for a in argv:
        if any(ch in a for ch in "*?["):
            out.extend(sorted(_glob.glob(a)))
        else:
            out.append(a)
    return out


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    paths = expand_args(sys.argv[1:])
    for p in paths:
        if not os.path.exists(p):
            print(f"missing: {p}")
            continue
        try:
            run_config(p)
        except Exception as exc:
            import traceback
            print(f"error in {p}: {exc}")
            traceback.print_exc()
