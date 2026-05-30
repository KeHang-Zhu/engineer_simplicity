"""
Summarize a SPSB digital-mouse panel run (CoT or strict or any other variant)
into a long-form per-bidder-decision CSV.

Usage:
    ./venv/bin/python analysis_v2/cot_v2/summarize_panel.py --prefix spsb_basis_cot \\
        --out data_v2/results/spsb_basis_cot_v2_summary.csv

    ./venv/bin/python analysis_v2/cot_v2/summarize_panel.py --prefix spsb_basis_strict \\
        --out data_v2/results/spsb_basis_strict_v2_summary.csv
"""

import argparse
import glob
import json
import os
import re

import pandas as pd

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
LOG_BASE = os.path.join(REPO_ROOT, "data_v2/experiment_logs/gpt5mini")


def _iter_results(strain_dir):
    for path in sorted(glob.glob(os.path.join(strain_dir, "result_1_*.json"))):
        with open(path) as fp:
            yield path, json.load(fp)


def _summarize_one(payload, run_idx):
    rows = []
    for _, rd in payload.items():
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prefix", required=True,
                    help="experiment_logs subdir prefix, e.g. spsb_basis_cot or spsb_basis_strict")
    ap.add_argument("--out", required=True, help="output CSV path (relative to repo root or absolute)")
    args = ap.parse_args()

    pattern = os.path.join(LOG_BASE, f"{args.prefix}_c*_f*_b*")
    basis_re = re.compile(rf"{re.escape(args.prefix)}_(c\d_f\d_b\d)$")

    out = []
    for strain_dir in sorted(glob.glob(pattern)):
        name = os.path.basename(strain_dir)
        m = basis_re.match(name)
        if not m:
            continue
        basis_id = m.group(1)
        c, f, b = int(basis_id[1]), int(basis_id[4]), int(basis_id[7])
        for run_idx, (_, payload) in enumerate(_iter_results(strain_dir)):
            for row in _summarize_one(payload, run_idx):
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
                out.append(row)
    df = pd.DataFrame(out)
    out_path = args.out if os.path.isabs(args.out) else os.path.join(REPO_ROOT, args.out)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"wrote {out_path}: rows={len(df)}, strains={df['basis_id'].nunique() if len(df) else 0}")


if __name__ == "__main__":
    main()
