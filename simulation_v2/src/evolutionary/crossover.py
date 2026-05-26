"""
Component 3b of Stage E: crossover operator.

LLM-driven blend of two parent mechanisms. The child should pick a coherent
subset of design choices from each parent (not a literal text concatenation)
and produce a runnable mechanism. Parameter dicts are mixed independently:
for each key in `yaml_overrides`, the child takes parent A's value with prob
0.5, else parent B's; the LLM then sees the resulting param vector and is
asked to rewrite `rule_text` to be consistent with it.

Public API:
    crossover(parent_a, parent_b, model, *, rng=None) -> child_mech
"""

import hashlib
import json
import os
import random
import sys

from edsl.questions import QuestionFreeText

sys.path.insert(0, os.path.dirname(__file__))
from mechanism_generator import _extract_json  # noqa: E402


CROSSOVER_TEMPLATE = """\
You are recombining two parent auction mechanisms to produce a coherent CHILD.
The child should inherit *some* design choices from each parent — not
literally concatenate text. The result must be a runnable single-item
sealed-bid auction with 3 bidders, private values uniform in $0-$50, sealed
bids in $0.1 increments.

The yaml_overrides dict for the child has already been mixed gene-wise from
the two parents and is provided below — your `rule_text` MUST be consistent
with it.

Parent A:
{parent_a_json}

Parent B:
{parent_b_json}

Pre-mixed yaml_overrides for the child:
{mixed_overrides_json}

Produce a JSON object with EXACTLY these top-level keys, and nothing else:
  - "name":          short snake_case identifier for the child
  - "one_line":      one-sentence description in plain English
  - "rule_text":     Jinja2 template (placeholders {{num_bidders}}, {{private}} ok), 3-7 sentences,
                     fully consistent with the pre-mixed yaml_overrides above.
  - "yaml_overrides": echo back the pre-mixed dict above verbatim (do not modify keys);
                     you may add up to 2 extra string keys that name your design choices.
  - "rationale":     one paragraph: which choices came from which parent and why.
  - "crossover_summary": one sentence describing the blend.

CRITICAL: output ONLY the JSON object — no markdown fences, no preamble, no trailing text.
"""


def _mix_overrides(a_overrides, b_overrides, rng):
    """Gene-wise mix of two yaml_overrides dicts; each key independently from A or B."""
    keys = set(a_overrides) | set(b_overrides)
    mixed = {}
    for k in sorted(keys):
        pick_a = rng.random() < 0.5
        if pick_a and k in a_overrides:
            mixed[k] = a_overrides[k]
        elif k in b_overrides:
            mixed[k] = b_overrides[k]
        else:
            mixed[k] = a_overrides.get(k, b_overrides.get(k))
    return mixed


def _serialize_parent(parent_mech):
    keep = ("name", "one_line", "rule_text", "yaml_overrides", "rationale", "coord")
    return {k: parent_mech[k] for k in keep if k in parent_mech}


def crossover(parent_a, parent_b, model, rng=None):
    """Produce one child by LLM blend of two parents."""
    rng = rng or random.Random()
    mixed = _mix_overrides(
        parent_a.get("yaml_overrides", {}),
        parent_b.get("yaml_overrides", {}),
        rng,
    )

    prompt = CROSSOVER_TEMPLATE.format(
        parent_a_json=json.dumps(_serialize_parent(parent_a), indent=2),
        parent_b_json=json.dumps(_serialize_parent(parent_b), indent=2),
        mixed_overrides_json=json.dumps(mixed, indent=2),
    )
    q = QuestionFreeText(question_name="crossover", question_text=prompt)
    res = q.by(model).run(verbose=False, progress_bar=False)
    raw = res.select("answer.crossover").to_list()[0]
    child = _extract_json(raw)

    for k in ("name", "one_line", "rule_text", "yaml_overrides", "rationale"):
        if k not in child:
            raise ValueError(f"crossover produced child missing key {k!r}")

    child["op"] = "crossover"
    child["parent_hashes"] = [
        parent_a.get("rule_hash", ""),
        parent_b.get("rule_hash", ""),
    ]
    child["coord"] = {
        "op": "crossover",
        "parent_hashes": child["parent_hashes"],
        "mixed_overrides": mixed,
        "crossover_summary": child.get("crossover_summary", ""),
    }
    child["rule_hash"] = hashlib.sha256(child["rule_text"].encode()).hexdigest()[:12]
    return child


if __name__ == "__main__":
    import argparse

    from edsl import Model

    p = argparse.ArgumentParser()
    p.add_argument("parent_a_json")
    p.add_argument("parent_b_json")
    p.add_argument("--model", default="gpt-5.4-mini")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    with open(args.parent_a_json) as f:
        a = json.load(f)
    with open(args.parent_b_json) as f:
        b = json.load(f)

    model = Model(args.model, temperature=0.7)
    child = crossover(a, b, model, rng=random.Random(args.seed))
    print(json.dumps(child, indent=2))
