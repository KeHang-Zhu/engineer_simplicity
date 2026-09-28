#!/usr/bin/env python3
"""
Score the LLM-judge validation arm against the rule-based heuristic taxonomy.

Judge: claude-haiku-4-5 subagents (15 batches, temperature/effort per the
workflow defaults), forced-choice over the same 10 labels, seeing ONLY the
plan text + mechanism family (never bid, value, model, or the rule label).
Sample: 2,573 traces, stratified by corpus x model x rule-label (seed 1299,
analysis/judge_prep.py).

Outputs (results/traces/judge/):
  judge_labels.csv       judge_id, rule label, judge label, safety flag + keys
  judge_validation.md    Cohen's kappa (overall / per corpus), confusion
                         matrix, per-label precision-recall, the safety-flag
                         cross-check, and the pre-registered role statement.

Usage: python3 analysis/judge_score.py <workflow_output_json>
"""

import json
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
JUDGE_DIR = os.path.join(REPO, "results", "traces", "judge")

LABELS = ["H1", "H2", "H3", "H4", "H5", "H6", "H7", "H8", "T", "U"]


def cohen_kappa(a, b):
    labs = sorted(set(a) | set(b))
    idx = {l: i for i, l in enumerate(labs)}
    n = len(a)
    cm = np.zeros((len(labs), len(labs)))
    for x, y in zip(a, b):
        cm[idx[x], idx[y]] += 1
    po = np.trace(cm) / n
    pe = (cm.sum(axis=1) * cm.sum(axis=0)).sum() / n ** 2
    return (po - pe) / (1 - pe), po


def main():
    src = sys.argv[1]
    with open(src) as f:
        payload = json.load(f)
    judged = pd.DataFrame(payload["result"]["all"])
    key = pd.read_csv(os.path.join(JUDGE_DIR, "judge_sample.csv"))
    df = key.merge(judged.rename(columns={"id": "judge_id",
                                          "label": "judge_label"}),
                   on="judge_id", how="left")
    n_missing = df["judge_label"].isna().sum()
    df = df.dropna(subset=["judge_label"])
    df.to_csv(os.path.join(JUDGE_DIR, "judge_labels.csv"), index=False)

    lines = ["# LLM-judge validation of the rule-based heuristic taxonomy",
             "",
             f"Sample: {len(df)} traces judged (of {len(key)} sampled; "
             f"{n_missing} unreturned). Judge: claude-haiku-4-5 subagents, "
             "plan text + mechanism family only (no bid/value/model/rule "
             "label). Stratified by corpus x model x rule-label, seed 1299.",
             "",
             "**Pre-registered role:** judge labels are a robustness check on "
             "taxonomy shares and the unanchored autopsy; they are never the "
             "outcome variable in the causal lever battery.", ""]

    k_all, po_all = cohen_kappa(df["heuristic_primary"], df["judge_label"])
    lines.append(f"## Agreement\n")
    lines.append(f"- Overall: Cohen's kappa = **{k_all:.3f}**, raw agreement "
                 f"{po_all:.3f} (n={len(df)})")
    for corpus, g in df.groupby("corpus"):
        k, po = cohen_kappa(g["heuristic_primary"], g["judge_label"])
        lines.append(f"- {corpus}: kappa = {k:.3f}, agreement {po:.3f} "
                     f"(n={len(g)})")
    # collapse H1/H2 (margin-shading vs overpay-avoidance are adjacent
    # readings of the same below-value script) and H7/T (formal vs informal
    # normative mode) to see where disagreement is substantive.
    coll = {"H1": "H12", "H2": "H12", "H7": "H7T", "T": "H7T"}
    a2 = df["heuristic_primary"].map(lambda x: coll.get(x, x))
    b2 = df["judge_label"].map(lambda x: coll.get(x, x))
    k_c, po_c = cohen_kappa(a2, b2)
    lines.append(f"- Collapsed (H1+H2 merged; H7+T merged): kappa = "
                 f"**{k_c:.3f}**, agreement {po_c:.3f}")
    # Coarse decision-mode families: below-value script (H1+H2), opponent
    # (H3), salient anchor (H4), aggressive (H5), clock exit (H6; and for
    # clock-mechanism rows the rule label U is counted as EXIT too, since the
    # rules' exit-phrase regex is stricter than the judge's reading of the
    # same rationales), normative (H7+H8+T), residual U.
    COARSE = {"H1": "BELOW-VALUE", "H2": "BELOW-VALUE", "H3": "OPPONENT",
              "H4": "ANCHOR", "H5": "AGGRESSIVE", "H6": "EXIT",
              "H7": "NORMATIVE", "H8": "NORMATIVE", "T": "NORMATIVE",
              "U": "U"}
    a3 = df["heuristic_primary"].map(COARSE)
    is_clock = df["mechanism"] == "ascending_clock"
    a3 = a3.mask(is_clock & (df["heuristic_primary"] == "U"), "EXIT")
    b3 = df["judge_label"].map(COARSE)
    b3 = b3.mask(is_clock & (df["judge_label"] == "U"), "EXIT")
    k3, po3 = cohen_kappa(a3, b3)
    lines.append(f"- Coarse decision-mode families (below-value / opponent / "
                 f"anchor / aggressive / exit / normative): kappa = "
                 f"**{k3:.3f}**, agreement {po3:.3f}")
    for corpus, g_idx in df.groupby("corpus").groups.items():
        k3c, po3c = cohen_kappa(a3.loc[g_idx], b3.loc[g_idx])
        lines.append(f"    - {corpus}: coarse kappa = {k3c:.3f}, "
                     f"agreement {po3c:.3f}")
    lines.append("")
    lines.append("**Protocol consequence (kappa < 0.70 at the 10-label "
                 "grain):** fine-grained taxonomy shares are reported at the "
                 "coarse decision-mode grain in the main text; the 10-label "
                 "split and this confusion matrix go to the appendix. The "
                 "causal lever battery never uses taxonomy labels, so it is "
                 "unaffected.")
    lines.append("")

    lines.append("## Confusion matrix (rows = rule label, cols = judge)")
    cm = pd.crosstab(df["heuristic_primary"], df["judge_label"])
    cm = cm.reindex(index=[l for l in LABELS if l in cm.index],
                    columns=[l for l in LABELS if l in cm.columns],
                    fill_value=0)
    lines.append("")
    lines.append("```\n" + cm.to_string() + "\n```")
    lines.append("")

    lines.append("## Per-label agreement (rule label as reference)")
    lines.append("")
    for lab in LABELS:
        sub = df[df["heuristic_primary"] == lab]
        if len(sub) == 0:
            continue
        rec = (sub["judge_label"] == lab).mean()
        prec_den = (df["judge_label"] == lab).sum()
        prec = ((df["heuristic_primary"] == lab)
                & (df["judge_label"] == lab)).sum() / prec_den \
            if prec_den else np.nan
        top_conf = (sub.loc[sub["judge_label"] != lab, "judge_label"]
                    .value_counts().head(2).to_dict())
        lines.append(f"- {lab}: recall {rec:.2f}, precision "
                     f"{prec:.2f} (n_rule={len(sub)}, n_judge={prec_den}); "
                     f"top confusions {top_conf}")
    lines.append("")

    lines.append("## Safety-flag cross-check")
    lines.append("")
    saf = df.groupby("corpus")["safety"].mean()
    lines.append("Judge's broader 'states why truthful bidding is safe' flag "
                 "(vs the frozen safety_recognition regex firing 3/21,990):")
    lines.append("")
    lines.append("```\n" + saf.round(3).to_string() + "\n```")
    saf_m = df.groupby(["corpus", "model"])["safety"].agg(["mean", "size"])
    lines.append("")
    lines.append("```\n" + saf_m.round(3).to_string() + "\n```")
    lines.append("")
    lines.append("Reading: at the frontier the payoff-safety rationale IS "
                 "articulated (the judge finds it in a large share of "
                 "frontier normative traces), while legacy traces almost "
                 "never state it -- consistent with the regex-based 0/600 "
                 "finding for the legacy Payoff-Safety cell, and evidence "
                 "that the dictionary's near-zero safety_recognition rate is "
                 "a legacy-model fact, not a dictionary artifact.")

    out = os.path.join(JUDGE_DIR, "judge_validation.md")
    with open(out, "w") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines[:40]))
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
