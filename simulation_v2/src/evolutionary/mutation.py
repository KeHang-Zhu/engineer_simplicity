"""
Component 3a of Stage E: mutation operator.

Takes a parent mechanism dict (the same shape produced by
`mechanism_generator.instantiate_mechanism`) and asks an LLM to apply a
*small, local edit* — typically rewording one or two sentences of the Jinja2
rule template, optionally flipping one design-axis knob, and/or perturbing
one or two scalar parameters in `yaml_overrides`. The child must remain
schema-valid and runnable by the v1 framework.

This is intentionally a *narrow* edit, not a fresh resample: tournament
selection drives exploration; mutation drives exploitation around survivors.

Public API:
    mutate(parent_mech, model, *, edit_strength="small") -> child_mech

`parent_mech` is the dict shape produced upstream:
    {
      "name": str, "one_line": str, "rule_text": str,
      "yaml_overrides": dict, "rationale": str, "coord": dict, "rule_hash": str
    }
The returned `child_mech` has the same keys plus `parent_hashes`, `op="mutation"`,
and a fresh `rule_hash` derived from the new `rule_text`.
"""

import hashlib
import json
import os
import sys

from edsl.questions import QuestionFreeText

sys.path.insert(0, os.path.dirname(__file__))
from mechanism_generator import _extract_json  # noqa: E402


MUTATION_TEMPLATE = """\
You are evolving a population of single-item sealed-bid auction mechanisms.
Below is one parent mechanism. Produce a CHILD mechanism that is a *small,
local* edit of the parent — not a fresh design. The child should differ from
the parent in roughly one of:
  - rewording 1-2 sentences of the bidder-facing rule_text (clarity, framing),
  - flipping ONE design-axis knob (e.g. pricing_rule, payment_modifier, info_disclosure),
  - perturbing ONE or TWO scalar parameters in yaml_overrides.

Keep every other field consistent. The child must remain a coherent, runnable
auction in the same context: 3 bidders draw independent private values from $0
to $50, sealed bids in $0.1 increments.

Parent:
{parent_json}

Produce a JSON object with EXACTLY these top-level keys, and nothing else:
  - "name":          short snake_case identifier for the child
  - "one_line":      one-sentence description in plain English
  - "rule_text":     Jinja2 template (placeholders {{num_bidders}}, {{private}} ok), 3-7 sentences
  - "yaml_overrides": dict including at least:
        {{
          "rule.price_order":   one of {{"first", "second", "third"}},
          "rule.special_name":  leave as empty string "",
          "value.private_range": 49,
          "value.increment":     0.1
        }}
  - "rationale":     one paragraph describing the edit and predicted effect
  - "edit_summary":  one sentence describing exactly what changed vs the parent

CRITICAL: output ONLY the JSON object — no markdown fences, no preamble, no trailing text.
"""


def _serialize_parent(parent_mech):
    """Trim a parent mech to the fields the LLM needs to see."""
    keep = ("name", "one_line", "rule_text", "yaml_overrides", "rationale", "coord")
    return {k: parent_mech[k] for k in keep if k in parent_mech}


def mutate(parent_mech, model, edit_strength="small"):
    """Produce one mutated child of `parent_mech` via LLM small-edit."""
    parent_view = _serialize_parent(parent_mech)
    prompt = MUTATION_TEMPLATE.format(parent_json=json.dumps(parent_view, indent=2))
    q = QuestionFreeText(question_name="mutation", question_text=prompt)
    res = q.by(model).run(verbose=False, progress_bar=False)
    raw = res.select("answer.mutation").to_list()[0]
    child = _extract_json(raw)

    for k in ("name", "one_line", "rule_text", "yaml_overrides", "rationale"):
        if k not in child:
            raise ValueError(f"mutation produced child missing key {k!r}")

    child["op"] = "mutation"
    child["parent_hashes"] = [parent_mech.get("rule_hash", "")]
    child["coord"] = {
        "op": "mutation",
        "parent_hash": parent_mech.get("rule_hash", ""),
        "edit_strength": edit_strength,
        "edit_summary": child.get("edit_summary", ""),
    }
    child["rule_hash"] = hashlib.sha256(child["rule_text"].encode()).hexdigest()[:12]
    return child


if __name__ == "__main__":
    import argparse

    from edsl import Model

    p = argparse.ArgumentParser()
    p.add_argument("parent_json", help="path to a parent mechanism JSON (manifest entry or full mech dict)")
    p.add_argument("--model", default="gpt-5.4-mini")
    args = p.parse_args()

    with open(args.parent_json) as f:
        parent = json.load(f)

    model = Model(args.model, temperature=0.7)
    child = mutate(parent, model)
    print(json.dumps(child, indent=2))
