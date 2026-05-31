"""Extract per-generation evolution-path statistics from an EA run.

For each generation, report:
  - n mechanisms scored
  - operator breakdown (init / mutation / crossover / elite)
  - min / median / max Li-weighted SMAD (or fitness)
  - best-of-generation trajectory

Uses post-hoc SMAD scoring already done by ea_second_price_smad.py /
ea_unrestricted_vs_figure1.py.
"""

import argparse
import glob
import json
import os
import re

import pandas as pd
import yaml

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


def find_run_dir(ea_root):
    matches = sorted(glob.glob(os.path.join(ea_root, "run_*")))
    if not matches:
        raise SystemExit(f"no run_* under {ea_root}")
    return matches[0]


def collect(ea_root, smad_csv):
    """Read fitness JSONs and merge with the post-hoc SMAD scoring."""
    smads = pd.read_csv(smad_csv).set_index("mech_id" if "mech_id" in pd.read_csv(smad_csv).columns else "mechanism_id")
    rows = []
    for yaml_path in sorted(glob.glob(os.path.join(ea_root, "**/*.yaml"), recursive=True)):
        fit = yaml_path.replace(".yaml", ".fitness.json")
        if not os.path.isfile(fit):
            continue
        gen = os.path.basename(os.path.dirname(yaml_path))  # gen_0, gen_1, gen_2
        with open(yaml_path) as fp:
            cfg = yaml.safe_load(fp)
        name = cfg.get("experiment", {}).get("name") or os.path.basename(yaml_path)
        with open(fit) as fp:
            data = json.load(fp)
        # operator is in the genotype's op field or the rationale
        op = data.get("op") or "?"
        if op == "?":
            # fall back: parse from yaml comments
            with open(yaml_path) as fp:
                head = fp.read(800)
            if "operator: init" in head or "v2-newmech" in head or "evo_gen0" in head:
                op = "init"
            elif "operator: mutation" in head:
                op = "mutation"
            elif "operator: crossover" in head:
                op = "crossover"
        weighted_smad = None
        if name in smads.index:
            row = smads.loc[name]
            if hasattr(row, "iloc"):
                row = row.iloc[0] if hasattr(row, "iloc") else row
            try:
                weighted_smad = float(row.get("weighted_smad_Li2017"))
            except Exception:
                weighted_smad = None
        rows.append({
            "generation": gen,
            "mech_id": name,
            "operator": op,
            "weighted_smad": weighted_smad,
        })
    return pd.DataFrame(rows)


def summarize(df):
    by_gen = df.groupby("generation").agg(
        n=("mech_id", "count"),
        min_smad=("weighted_smad", "min"),
        median_smad=("weighted_smad", "median"),
        max_smad=("weighted_smad", "max"),
    ).reset_index()
    # best-so-far at each generation (cumulative min, smaller SMAD = better)
    by_gen["best_so_far"] = by_gen["min_smad"].cummin()
    return by_gen


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ea_root", required=True)
    ap.add_argument("--smad_csv", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    ea_root = os.path.join(REPO_ROOT, args.ea_root) if not os.path.isabs(args.ea_root) else args.ea_root
    smad_csv = os.path.join(REPO_ROOT, args.smad_csv) if not os.path.isabs(args.smad_csv) else args.smad_csv

    df = collect(ea_root, smad_csv)
    by_gen = summarize(df)
    out_path = os.path.join(REPO_ROOT, args.out) if not os.path.isabs(args.out) else args.out
    by_gen.to_csv(out_path, index=False)
    print(f"wrote {out_path}")
    print("\nPer-generation summary:")
    print(by_gen.to_string(index=False))
    print("\nBy generation x operator:")
    cross = df.groupby(["generation", "operator"]).size().unstack(fill_value=0)
    print(cross.to_string())


if __name__ == "__main__":
    main()
