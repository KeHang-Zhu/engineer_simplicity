"""Summarize the v2 round-2 SPSB error-mice run into a long-form CSV."""

import glob
import json
import os

import pandas as pd

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
LOG_BASE = os.path.join(REPO_ROOT, "data_v2/experiment_logs/gpt5mini")
OUT = os.path.join(REPO_ROOT, "data_v2/results/spsb_basis_error_v2r2_summary.csv")


def _iter(strain_dir):
    for path in sorted(glob.glob(os.path.join(strain_dir, "result_1_*.json"))):
        with open(path) as fp:
            yield json.load(fp)


def main():
    rows = []
    for strain_dir in sorted(glob.glob(os.path.join(LOG_BASE, "*_v2r2"))):
        name = os.path.basename(strain_dir).replace("_v2r2", "")
        for run_idx, payload in enumerate(_iter(strain_dir)):
            for _, rd in payload.items():
                values = rd.get("value", [])
                bids = [b["bid"] for b in rd.get("history", {}).get("bidding history", [])]
                plans = rd.get("plan", [])
                for ai, (v, b) in enumerate(zip(values, bids)):
                    rows.append({
                        "outcome_id": name,
                        "run": run_idx,
                        "agent": ai,
                        "value": v,
                        "bid": b,
                        "deviation": b - v if v is not None else None,
                        "ratio": (b / v) if v else None,
                        "plan": plans[ai] if ai < len(plans) else "",
                    })
    df = pd.DataFrame(rows)
    df.to_csv(OUT, index=False)
    print(f"wrote {OUT}: rows={len(df)}, strains={df['outcome_id'].nunique()}")


if __name__ == "__main__":
    main()
