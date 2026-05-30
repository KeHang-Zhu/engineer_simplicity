"""
Summarize the v2 CoT-induced SPSB panel run (27 CoT strains + 3 outcome
instructions) into long-form per-bidder-decision CSVs.

Reads result_1_*.json files from data_v2/experiment_logs/gpt5mini/spsb_basis_cot_*/
and data_v2/experiment_logs/gpt5mini/spsb_outcome_*/, one row per (strain, run,
agent_index).

Writes:
    data_v2/results/spsb_basis_cot_v2_summary.csv  (27 cells)
    data_v2/results/spsb_outcome_v2_summary.csv    (3 outcome strains)

Run from repo root:
    ./venv/bin/python analysis_v2/cot_v2/summarize_cot_v2.py
"""

import glob
import json
import os
import re

import pandas as pd

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
LOG_BASE = os.path.join(REPO_ROOT, "data_v2/experiment_logs/gpt5mini")
OUT_DIR = os.path.join(REPO_ROOT, "data_v2/results")

COT_GLOB = os.path.join(LOG_BASE, "spsb_basis_cot_c*_f*_b*")
OUTCOME_GLOB = os.path.join(LOG_BASE, "spsb_outcome_*")

BASIS_ID_RE = re.compile(r"spsb_basis_cot_(c\d_f\d_b\d)$")


def _iter_results(strain_dir):
    for path in sorted(glob.glob(os.path.join(strain_dir, "result_1_*.json"))):
        with open(path) as fp:
            yield path, json.load(fp)


def _summarize_one_run(payload, run_idx):
    rows = []
    for key, rd in payload.items():
        values = rd.get("value", [])
        bids = [b["bid"] for b in rd.get("history", {}).get("bidding history", [])]
        plans = rd.get("plan", [])
        for agent_idx, (v, b) in enumerate(zip(values, bids)):
            rows.append(
                {
                    "run": run_idx,
                    "agent": agent_idx,
                    "value": v,
                    "bid": b,
                    "deviation": b - v if v is not None and b is not None else None,
                    "ratio": (b / v) if v else None,
                    "plan": plans[agent_idx] if agent_idx < len(plans) else "",
                }
            )
    return rows


def summarize_cot_panel():
    out = []
    for strain_dir in sorted(glob.glob(COT_GLOB)):
        name = os.path.basename(strain_dir)
        match = BASIS_ID_RE.match(name)
        if not match:
            continue
        basis_id = match.group(1)
        c, f, b = (int(basis_id[1]), int(basis_id[4]), int(basis_id[7]))
        for run_idx, (_, payload) in enumerate(_iter_results(strain_dir)):
            rows = _summarize_one_run(payload, run_idx)
            for row in rows:
                row.update(
                    {
                        "basis_id": basis_id,
                        "c": c,
                        "f": f,
                        "b": b,
                        "contingent": ["off", "enumerate", "worstcase"][c],
                        "forward": ["off", "onestep", "tree"][f],
                        "beliefs": ["off", "firstorder", "secondorder"][b],
                    }
                )
            out.extend(rows)
    return pd.DataFrame(out)


def summarize_outcome():
    out = []
    for strain_dir in sorted(glob.glob(OUTCOME_GLOB)):
        name = os.path.basename(strain_dir)
        for run_idx, (_, payload) in enumerate(_iter_results(strain_dir)):
            rows = _summarize_one_run(payload, run_idx)
            for row in rows:
                row["outcome_id"] = name
                row["target_class"] = {
                    "spsb_outcome_overbid": "spsb_overbid",
                    "spsb_outcome_underbid": "spsb_underbid",
                    "spsb_outcome_truthful": "spsb_truthful",
                }.get(name, "unknown")
            out.extend(rows)
    return pd.DataFrame(out)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    panel = summarize_cot_panel()
    outcome = summarize_outcome()
    panel_out = os.path.join(OUT_DIR, "spsb_basis_cot_v2_summary.csv")
    outcome_out = os.path.join(OUT_DIR, "spsb_outcome_v2_summary.csv")
    panel.to_csv(panel_out, index=False)
    outcome.to_csv(outcome_out, index=False)
    print(f"wrote {panel_out}: rows={len(panel)}, strains={panel['basis_id'].nunique()}")
    print(f"wrote {outcome_out}: rows={len(outcome)}, outcome_ids={outcome['outcome_id'].nunique()}")


if __name__ == "__main__":
    main()
