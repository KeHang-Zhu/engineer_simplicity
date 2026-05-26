"""
Diverse auction-mechanism generator (Stage D prototype).

Concordia-inspired two-stage pattern, adapted for auction rules:
  Stage 1: pre-sample N points in the design-axis space (stratified random)
  Stage 2: per point, ask an LLM to instantiate a full auction:
           - Jinja2 rule template (bidder-facing prompt)
           - YAML config compatible with simulation_v1's run_auction_batch

Each generated mechanism is written to data_v2/evo/gen0/mechanism_{i}.{txt,yaml}
together with a manifest. The output YAMLs are immediately runnable via the
existing v2 batch runner.

Run from repo root:
    ./venv/bin/python simulation_v2/src/evolutionary/mechanism_generator.py --n 5
"""

import argparse
import hashlib
import json
import os
import random
import re
import sys

from edsl import Model
from edsl.questions import QuestionFreeText

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
sys.path.insert(0, os.path.dirname(__file__))
from axes import DESIGN_AXES, axis_dict, CategoricalAxis, OrdinalAxis  # noqa: E402


# --- Sampling: stratified-random coordinates in axis space ---

def sample_coordinates(n, axes, seed=0):
    """Stratified random sample of N points in the design-axis space."""
    rng = random.Random(seed)
    coords = []
    for _ in range(n):
        coord = {}
        for ax in axes:
            choices = ax.values if isinstance(ax, CategoricalAxis) else ax.levels
            coord[ax.name] = rng.choice(choices)
        coords.append(coord)
    return coords


# --- Stage 2 prompt: instantiate a full mechanism from a coordinate ---

INSTANTIATION_TEMPLATE = """\
You are an auction designer. Below is a specification for a single-item sealed-bid auction in terms of its design knobs. Your job is to produce a runnable mechanism.

Context: 3 bidders draw independent private values uniformly from $0 to $50. Each bidder sees only their own value, then submits one sealed bid in $0.1 increments.

Design choices (one row per axis):
{coord_lines}

Produce a JSON object with EXACTLY these top-level keys, and nothing else:
  - "name":          short snake_case identifier (e.g. "second_price_discrete_high_simplicity")
  - "one_line":      one-sentence description in plain English
  - "rule_text":     The bidder-facing rule prompt. A Jinja2 template (you may use {{{{num_bidders}}}} and {{{{private}}}} as placeholders for $-amount upper bound and number of opponents). Should describe the auction format end-to-end in 3-7 sentences, consistent with ALL design choices above. If "simplicity_emphasis" is medium/high, add a sentence that explicitly hints at the dominant action (where one exists).
  - "yaml_overrides": A dict of YAML overrides for the simulation runner. MUST include at minimum:
        {{
          "rule.price_order":   one of {{"first", "second", "third"}}   (or "first" for all-pay/average-price; we'll handle special pricing in rule_text),
          "rule.special_name":  basename of the rule template file (we'll generate this filename, so leave as ""),
          "value.private_range": 49,
          "value.increment":     0.1
        }}
  - "rationale":     one-paragraph explanation of why this mechanism implements the spec above and what its predicted behavior is.

CRITICAL: output ONLY the JSON object — no markdown fences, no preamble, no trailing text. The JSON must parse with Python's json.loads.
"""


def _coord_lines(coord):
    return "\n".join(f"  - {k}: {v}" for k, v in coord.items())


# --- Extract JSON from LLM free-text robustly ---

def _extract_json(text):
    """Pull the first {...} block out of model output and json.loads it."""
    # Strip ```json ... ``` fences if present
    text = re.sub(r"^\s*```(?:json)?\s*", "", text, flags=re.MULTILINE)
    text = re.sub(r"\s*```\s*$", "", text, flags=re.MULTILINE)
    # Find first balanced { ... }
    depth = 0
    start = None
    for i, ch in enumerate(text):
        if ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0 and start is not None:
                return json.loads(text[start:i + 1])
    raise ValueError("no JSON object found in model output")


# --- Stage 2: instantiate one mechanism ---

def instantiate_mechanism(coord, model):
    """Ask the model to produce a mechanism for one design-axis coordinate."""
    q = QuestionFreeText(
        question_name="mechanism",
        question_text=INSTANTIATION_TEMPLATE.format(coord_lines=_coord_lines(coord)),
    )
    res = q.by(model).run(verbose=False, progress_bar=False)
    raw = res.select("answer.mechanism").to_list()[0]
    mech = _extract_json(raw)
    # Required-field sanity check
    for k in ("name", "one_line", "rule_text", "yaml_overrides", "rationale"):
        if k not in mech:
            raise ValueError(f"missing required key {k!r} in generated mechanism")
    mech["coord"] = coord
    mech["rule_hash"] = hashlib.sha256(mech["rule_text"].encode()).hexdigest()[:12]
    return mech


# --- Write a generated mechanism to disk as a runnable YAML + template ---

def write_mechanism(mech, out_dir, idx, gen_tag="evo_gen0"):
    safe_name = re.sub(r"[^a-z0-9_]", "_", mech["name"].lower())
    base = f"mech_{idx:03d}_{safe_name}_{mech['rule_hash']}"
    tpl_path = os.path.join(out_dir, f"{base}.txt")
    yaml_path = os.path.join(out_dir, f"{base}.yaml")

    with open(tpl_path, "w") as f:
        f.write(mech["rule_text"].rstrip() + "\n")

    ov = mech["yaml_overrides"]
    # Build a config that points at our generated template (saved in out_dir)
    yaml_text = f"""\
# Generated mechanism: {mech['name']}
# {mech['one_line']}
# Design coordinates: {json.dumps(mech['coord'])}
# Rationale: {mech['rationale']}

experiment:
  name: "{base}"
  version: "v2-evo-gen0"
  description: "{mech['one_line']}"

auction:
  number_agents: 3
  rounds: 1

rule:
  seal_clock: "seal"
  ascend_descend: "ascend"
  price_order: "{ov.get('rule.price_order', 'second')}"
  private_value: "private"
  open_blind: "open"
  closing: false
  reserve_price: 0
  special_name: "{os.path.basename(tpl_path)}"

value:
  common_range: [0, 29]
  private_range: {ov.get('value.private_range', 49)}
  increment: {ov.get('value.increment', 0.1)}
  seed_base: 7001

llm:
  model: "gpt-5.4-mini"
  temperature: 0.5

prompt:
  strategy_type: "plan_reflection"
  prompt_dir: "Prompt/"
  rule_template_dir: "{os.path.relpath(out_dir, os.path.join(REPO_ROOT, 'simulation_v1'))}/"
  include_payment_example: false

execution:
  repetitions: 5
  parallel: true
  max_workers: 4
  output_dir: "data_v2/experiment_logs/{gen_tag}/{base}"
"""
    with open(yaml_path, "w") as f:
        f.write(yaml_text)
    return {"base": base, "yaml": yaml_path, "tpl": tpl_path,
            "hash": mech["rule_hash"], "name": mech["name"], "coord": mech["coord"]}


# --- CLI ---

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=5, help="number of mechanisms to generate")
    p.add_argument("--seed", type=int, default=20260525)
    p.add_argument("--model", default="gpt-5.4-mini")
    p.add_argument("--out_dir", default=os.path.join(REPO_ROOT, "data_v2/evo/gen0"))
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    coords = sample_coordinates(args.n, DESIGN_AXES, seed=args.seed)
    print(f"=== Sampled {len(coords)} coordinates ===")
    for i, c in enumerate(coords):
        print(f"  [{i}] {c}")

    model = Model(args.model, temperature=0.7)
    manifest = []
    for i, c in enumerate(coords):
        print(f"\n=== Instantiating mechanism {i+1}/{len(coords)} ===")
        try:
            mech = instantiate_mechanism(c, model)
        except Exception as e:
            print(f"  FAILED: {type(e).__name__}: {e}")
            continue
        entry = write_mechanism(mech, args.out_dir, i)
        manifest.append({**entry, "one_line": mech["one_line"]})
        print(f"  -> {entry['base']}")
        print(f"     one_line: {mech['one_line']}")

    manifest_path = os.path.join(args.out_dir, "manifest.json")
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"\nwrote manifest: {manifest_path}")
    print(f"{len(manifest)} of {len(coords)} generations succeeded")


if __name__ == "__main__":
    main()
