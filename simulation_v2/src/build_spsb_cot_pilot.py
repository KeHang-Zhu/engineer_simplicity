"""
Build chain-of-thought (few-shot exemplar) variants of SPSB cognitive-basis
personas, for a small instruction-only vs CoT pilot.

Motivation: the standard panel induces a deficit purely by *instruction*. An
alternative is to also *demonstrate* the bounded reasoning procedure with a
worked chain-of-thought example, in the spirit of few-shot CoT prompting. This
script composes a CoT persona = standard basis persona text + one worked example
block whose reasoning matches the strain's reasoning level.

We build two pilot strains:
  c0_f0_b0  (all-deficit)  -> demonstrate a margin-of-safety underbid heuristic
  c2_f2_b2  (all-repaired) -> demonstrate the contingent/forward/belief reasoning
                              that yields truthful (Vickrey) bidding in SPSB.

Run from repo root:
    ./venv/bin/python simulation_v2/src/build_spsb_cot_pilot.py
"""

import csv
import os

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
BASIS_DIR = os.path.join(REPO_ROOT, "rule_prompt_v2/personas/spsb_basis")
COT_DIR = os.path.join(REPO_ROOT, "rule_prompt_v2/personas/spsb_basis_cot")
PANEL_CSV = os.path.join(REPO_ROOT, "analysis_v2/calibration/cot_pilot_panel.csv")

# Worked chain-of-thought exemplars. Each demonstrates, on a concrete value, the
# reasoning operations the strain is allowed to use, ending in a bid. The format
# mirrors the model's own <PLAN>/<ACTION> output so it reads as a demonstration.

DEFICIT_EXAMPLE = """\

Worked example (reason in this style; this is a second-price auction):
Value = $30.
PLAN: I will not enumerate what opponents might bid, will not trace the payoff
of different candidate bids, and will not model other bidders' reasoning. I
simply apply a margin-of-safety rule: bid somewhat below my value so that I keep
surplus if I happen to win. Roughly 70% of $30 is about $21.
ACTION: 21
"""

REPAIRED_EXAMPLE = """\

Worked example (reason in this style; this is a second-price auction):
Value = $30.
PLAN: Compare candidate bids. The winner pays the second-highest bid, so my own
bid only changes WHETHER I win, never HOW MUCH I pay. Trace the cases: if I bid
below $30 I might lose an auction that would have been profitable; if I bid above
$30 I can win cases where the second price exceeds my value and lose money. The
bid that is best in the worst case, for any opponent behavior, is exactly my
value. Rational opponents reason the same way, which does not change my answer.
ACTION: 30
"""

PILOT = [
    ("c0_f0_b0", DEFICIT_EXAMPLE, "spsb_underbid"),
    ("c2_f2_b2", REPAIRED_EXAMPLE, "spsb_truthful"),
]


def main():
    os.makedirs(COT_DIR, exist_ok=True)
    rows = []
    for strain_id, example, target in PILOT:
        with open(os.path.join(BASIS_DIR, f"{strain_id}.txt")) as fp:
            base = fp.read().rstrip()
        cot_text = base + "\n" + example
        out_file = f"rule_prompt_v2/personas/spsb_basis_cot/{strain_id}.txt"
        with open(os.path.join(REPO_ROOT, out_file), "w") as fp:
            fp.write(cot_text)
        rows.append({
            "persona_id": f"{strain_id}_cot",
            "persona_file": out_file,
            "source": "basis_cot",
            "target_class": target,
        })
    with open(PANEL_CSV, "w", newline="") as fp:
        w = csv.DictWriter(fp, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {len(rows)} CoT personas to {COT_DIR}")
    print(f"pilot panel: {PANEL_CSV}")


if __name__ == "__main__":
    main()
