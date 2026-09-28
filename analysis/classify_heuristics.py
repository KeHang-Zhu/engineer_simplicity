#!/usr/bin/env python3
"""
Rule-based heuristic taxonomy for the traces re-analysis (auction-v4 Section 5,
"What the Traces Say": the eight named heuristics).

Assigns every trace in results/traces/trace_features_v2.csv a PRIMARY heuristic
label via a documented decision list over (i) the frozen 18-feature dictionary,
(ii) the additive FEATURES_V2 formal-derivation group, (iii) the stated-bid
parser of record (build_trace_mediation.parse_stated_bid), and (iv) three
taxonomy-local helper regexes declared here (taxonomy layer, NOT part of the
frozen dictionary).

Labels (v4 §5 numbering + two housekeeping buckets):
  H1 profit-margin shading          H5 win-probability / aggressive
  H2 avoid overpayment              H6 price-threshold exit (clock)
  H3 opponent anchoring             H7 formal dominance derivation
  H4 salient-number anchoring       H8 worst-case safety proof
  T  truthful / dominance assertion without formal derivation
     (not one of the 8; the informal counterpart of H7 -- kept separate so the
      retrieval-vs-derivation boundary is measurable)
  U  unclassified (residual)

Priority order (first match wins; specificity descending). Non-exclusive
binary flags h1..h8, t, and an overlap matrix are emitted alongside so every
conflict resolution is auditable.

  1. H7  >=2 of the 5 formal_derivation features, OR equilibrium_formula
         alone, OR (dominance_proof AND dominance_language).
         Sub-flag wrong_mechanism_derivation: equilibrium_formula AND
         first_price_mention in a second-/third-price cell.
  2. H8  worst_case AND (safety_recognition OR payment_rule_correct OR
         dominance_proof).
  3. H6  clock rows with exit-threshold phrasing (clock rows failing the
         phrase test stay clock-unclassified).
  4. T   (truthful_intent OR dominance_language) AND no shading/overbid intent.
  5. H1s shading_intent AND (margin_language OR zero_profit_fallacy)   [strict]
  6. H2  overpay_concern AND (shading_intent OR stated_bid < value).
  7. H5  overbid_intent OR (aggressive_language AND probability_reasoning AND
         stated_bid > value).
  8. H3  opponent_modeling AND (anchor phrasing OR (no value-anchored intent
         AND stated_bid present)).
  9. H1l bare shading_intent (below-value intent with no articulated reason;
         the implicit margin script)                                   [loose]
 10. H4  no value-anchored intent AND stated_bid present AND (salient-fraction
         phrasing OR round-number rule: stated_bid % 5 == 0 and
         |stated_bid - value| > 2).
 11. U   residual.

Outputs (results/traces/):
  heuristic_labels.csv     keyed by corpus,row index; primary + flags
  heuristic_portfolio.csv  heuristic x mechanism x model (+corpus): share,
                           conditional bid distribution (v4 reporting template)
  heuristic_overlap.csv    co-fire matrix of the non-exclusive flags
  heuristic_examples.md    3 shortest verbatim exemplars per label
Deterministic; no resampling. Run: python3 analysis/classify_heuristics.py
"""

import os
import re
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from build_trace_mediation import parse_stated_bid  # parser of record

REPO = os.path.dirname(HERE)
TRACES = os.path.join(REPO, "results", "traces")
V2_CSV = os.path.join(TRACES, "trace_features_v2.csv")

V2_COLS = ["equilibrium_formula", "dominance_proof", "algebraic_derivation",
           "case_enumeration", "env_model_recital"]

# --- taxonomy-local helper regexes (taxonomy layer, not the frozen dict) ----
EXIT_THRESHOLD_RE = re.compile(
    r"(?:drop|exit|stop|withdraw|leave|stay)[^.]{0,60}"
    r"(?:price|value|when|once|if|until|below|above|reach|exceed)"
    r"|until the price reaches|price (?:is |gets |goes )?(?:above|over|beyond|"
    r"exceeds|reaches) my value|remain (?:in|active)", re.I)
ANCHOR_PHRASE_RE = re.compile(
    r"(?:just|slightly|a (?:bit|little)) (?:above|below|over|under|higher|"
    r"lower)[^.]{0,50}(?:their|others|other bidders|competitors|rivals|"
    r"expected|likely|anticipated|average|typical)"
    r"|(?:outbid|beat|top) (?:the )?(?:others|other bidders|competitors|rivals)"
    r"|stay competitive|remain competitive", re.I)
SALIENT_NUMBER_RE = re.compile(
    r"half (?:of )?(?:my|the) value|midpoint|middle of the (?:range|"
    r"distribution)|round(?:ed)? (?:number|figure)|average (?:value|of the "
    r"range)|expected value of the (?:item|distribution|range)"
    r"|mid[- ]range", re.I)

LABELS = ["H1", "H2", "H3", "H4", "H5", "H6", "H7", "H8", "T", "U"]
LABEL_NAMES = {
    "H1": "profit-margin shading", "H2": "avoid overpayment",
    "H3": "opponent anchoring", "H4": "salient-number anchoring",
    "H5": "win-probability / aggressive", "H6": "price-threshold exit",
    "H7": "formal dominance derivation", "H8": "worst-case safety proof",
    "T": "truthful/dominance assertion (informal)", "U": "unclassified",
}


def classify(df: pd.DataFrame) -> pd.DataFrame:
    plan = df["plan"].astype(str)
    low = plan.str.lower()
    stated = df["stated_bid"]
    has_stated = stated.notna()
    no_intent = ((df["shading_intent"] == 0) & (df["overbid_intent"] == 0)
                 & (df["truthful_intent"] == 0))
    is_clock = df["mechanism"] == "ascending_clock"

    # non-exclusive condition flags (auditable)
    f = pd.DataFrame(index=df.index)
    f["h7"] = ((df[V2_COLS].sum(axis=1) >= 2)
               | (df["equilibrium_formula"] == 1)
               | ((df["dominance_proof"] == 1)
                  & (df["dominance_language"] == 1))).astype(int)
    # Primary-label variant: a bare dominance recital followed by a stated
    # intent to shade is NOT derivation-as-decision-mode (the model recites
    # the invariant and disobeys it); such traces cascade to H1/H2. The
    # formula / >=2-feature branches keep everything (a wrong formula
    # faithfully executed is still formal mode).
    f["h7_primary"] = ((df[V2_COLS].sum(axis=1) >= 2)
                       | (df["equilibrium_formula"] == 1)
                       | ((df["dominance_proof"] == 1)
                          & (df["dominance_language"] == 1)
                          & (df["shading_intent"] == 0))).astype(int)
    f["wrong_mechanism_derivation"] = (
        (df["equilibrium_formula"] == 1) & (df["first_price_mention"] == 1)
        & df["mechanism"].isin(["spsb_sealed", "tpsb_sealed"])).astype(int)
    f["h8"] = ((df["worst_case"] == 1)
               & ((df["safety_recognition"] == 1)
                  | (df["payment_rule_correct"] == 1)
                  | (df["dominance_proof"] == 1))).astype(int)
    f["h6"] = (is_clock
               & low.str.contains(EXIT_THRESHOLD_RE, regex=True)).astype(int)
    f["t"] = (((df["truthful_intent"] == 1) | (df["dominance_language"] == 1))
              & (df["shading_intent"] == 0)
              & (df["overbid_intent"] == 0)).astype(int)
    f["h1_strict"] = ((df["shading_intent"] == 1)
                      & ((df["margin_language"] == 1)
                         | (df["zero_profit_fallacy"] == 1))).astype(int)
    f["h2"] = ((df["overpay_concern"] == 1)
               & ((df["shading_intent"] == 1)
                  | (has_stated & (stated < df["value"])))).astype(int)
    f["h5"] = ((df["overbid_intent"] == 1)
               | ((df["aggressive_language"] == 1)
                  & (df["probability_reasoning"] == 1)
                  & has_stated & (stated > df["value"]))).astype(int)
    f["h3"] = ((df["opponent_modeling"] == 1)
               & (low.str.contains(ANCHOR_PHRASE_RE, regex=True)
                  | (no_intent & has_stated))).astype(int)
    f["h1_loose"] = (df["shading_intent"] == 1).astype(int)
    # Strong H4: explicit salient-fraction phrasing ("half my value",
    # "midpoint") is the decision rule even when an overpay justification is
    # appended -- v4's canonical H4 example is exactly this phrasing. Ranked
    # above H2. Weak H4 (round-number rule) stays last before U.
    f["h4_strong"] = ((df["truthful_intent"] == 0)
                      & (df["overbid_intent"] == 0)
                      & low.str.contains(SALIENT_NUMBER_RE, regex=True)
                      ).astype(int)
    f["h4_weak"] = (no_intent & has_stated
                    & ((stated % 5 == 0)
                       & ((stated - df["value"]).abs() > 2))).astype(int)
    f["h4"] = f["h4_strong"] | f["h4_weak"]
    f["h1"] = f["h1_strict"] | f["h1_loose"]

    # priority cascade -> primary label
    order = [("h7_primary", "H7"), ("h8", "H8"), ("h6", "H6"), ("t", "T"),
             ("h1_strict", "H1"), ("h4_strong", "H4"), ("h2", "H2"),
             ("h5", "H5"), ("h3", "H3"), ("h1_loose", "H1"),
             ("h4_weak", "H4")]
    primary = pd.Series("U", index=df.index)
    assigned = pd.Series(False, index=df.index)
    for col, lab in order:
        pick = (~assigned) & (f[col] == 1)
        primary[pick] = lab
        assigned |= pick
    # clock rows never fall through to sealed-bid labels
    clock_unassigned = is_clock & (primary != "H6") & (primary != "H7")
    primary[clock_unassigned & (primary != "U")] = "U"
    f["heuristic_primary"] = primary
    return f


def portfolio(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    dd = df.drop_duplicates(subset=["model", "experiment", "value",
                                    "bid", "plan"])
    for (corpus, mech, model), g in df.groupby(["corpus", "mechanism",
                                                "model"]):
        gd = dd[(dd["corpus"] == corpus) & (dd["mechanism"] == mech)
                & (dd["model"] == model)]
        for lab in LABELS:
            sub = g[g["heuristic_primary"] == lab]
            if len(sub) == 0:
                continue
            subd = gd[gd["heuristic_primary"] == lab]
            dev = sub["deviation"]
            rows.append(dict(
                corpus=corpus, mechanism=mech, model=model,
                heuristic=lab, heuristic_name=LABEL_NAMES[lab],
                n=len(sub), share=round(len(sub) / len(g), 4),
                share_dedup=round(len(subd) / len(gd), 4) if len(gd) else np.nan,
                mean_dev=round(dev.mean(), 3),
                mean_absdev=round(dev.abs().mean(), 3),
                dev_q10=round(dev.quantile(0.10), 2),
                dev_q50=round(dev.quantile(0.50), 2),
                dev_q90=round(dev.quantile(0.90), 2),
                p_dev_below_m5=round((dev < -5).mean(), 4),
                p_dev_above_0=round((dev > 0).mean(), 4),
            ))
    return pd.DataFrame(rows)


def write_examples(df: pd.DataFrame, path: str):
    lines = ["# Exact exemplar traces per heuristic (verbatim)",
             "",
             "Auto-generated by analysis/classify_heuristics.py. Three",
             "shortest plans (>= 120 chars, distinct models where possible)",
             "per primary label, deduplicated. Discharges the auction-v4 §5",
             "`\\required` blocker (exact, short traces).", ""]
    dd = df.drop_duplicates(subset=["model", "experiment", "value",
                                    "bid", "plan"])
    dd = dd[dd["plan"].str.len() >= 120].copy()
    dd["plen"] = dd["plan"].str.len()
    for lab in LABELS:
        sub = dd[dd["heuristic_primary"] == lab].sort_values("plen")
        lines.append(f"## {lab} — {LABEL_NAMES[lab]}  (n={len(sub)})")
        lines.append("")
        seen_models = set()
        picked = []
        for _, r in sub.iterrows():
            if r["model"] in seen_models and len(picked) < 2:
                continue
            picked.append(r)
            seen_models.add(r["model"])
            if len(picked) == 3:
                break
        for r in picked:
            lines.append(
                f"- **{r['model']}** | {r['experiment']} ({r['mechanism']}) | "
                f"value ${r['value']:g}, bid ${r['bid']:g}, "
                f"dev {r['deviation']:+.1f}")
            plan_txt = " ".join(str(r["plan"]).split())
            lines.append(f"  > {plan_txt}")
            lines.append("")
    # the wrong-mechanism derivation exhibit
    wm = dd[dd["wrong_mechanism_derivation"] == 1].sort_values("plen")
    lines.append(f"## H7 sub-flag — wrong-mechanism derivation  (n={len(wm)})")
    lines.append("")
    for _, r in wm.head(3).iterrows():
        lines.append(
            f"- **{r['model']}** | {r['experiment']} ({r['mechanism']}) | "
            f"value ${r['value']:g}, bid ${r['bid']:g}, dev {r['deviation']:+.1f}")
        lines.append(f"  > {' '.join(str(r['plan']).split())}")
        lines.append("")
    with open(path, "w") as fh:
        fh.write("\n".join(lines))


def main():
    df = pd.read_csv(V2_CSV, low_memory=False)
    print(f"corpus v2: {len(df)} rows", file=sys.stderr)
    df["stated_bid"] = df["plan"].apply(parse_stated_bid)

    flags = classify(df)
    df = pd.concat([df, flags], axis=1)

    keep = ["corpus", "source", "model", "model_dir", "experiment",
            "mechanism", "run_id", "auction_id", "value", "bid", "deviation",
            "abs_dev", "family", "stated_bid", "heuristic_primary",
            "wrong_mechanism_derivation",
            "h7", "h8", "h6", "t", "h1", "h1_strict", "h1_loose", "h2", "h5",
            "h3", "h4"]
    df[keep].to_csv(os.path.join(TRACES, "heuristic_labels.csv"), index=False)

    port = portfolio(df)
    port.to_csv(os.path.join(TRACES, "heuristic_portfolio.csv"), index=False)

    flag_cols = ["h1", "h2", "h3", "h4", "h5", "h6", "h7", "h8", "t"]
    overlap = pd.DataFrame(
        np.zeros((len(flag_cols), len(flag_cols)), dtype=int),
        index=flag_cols, columns=flag_cols)
    for a in flag_cols:
        for b in flag_cols:
            overlap.loc[a, b] = int(((df[a] == 1) & (df[b] == 1)).sum())
    overlap.to_csv(os.path.join(TRACES, "heuristic_overlap.csv"))

    write_examples(df, os.path.join(TRACES, "heuristic_examples.md"))

    # ---- console summary ----------------------------------------------------
    print("\nPrimary-label shares by corpus:", file=sys.stderr)
    tab = (df.groupby(["corpus", "heuristic_primary"]).size()
           .unstack(fill_value=0))
    print((tab.div(tab.sum(axis=1), axis=0).round(3)).to_string(),
          file=sys.stderr)
    print("\nSealed SPSB primary shares by model (legacy):", file=sys.stderr)
    leg = df[(df["corpus"] == "legacy") & (df["mechanism"] == "spsb_sealed")]
    tab2 = (leg.groupby(["model", "heuristic_primary"]).size()
            .unstack(fill_value=0))
    print((tab2.div(tab2.sum(axis=1), axis=0).round(3)).to_string(),
          file=sys.stderr)
    print("\nFrontier sealed primary shares by model:", file=sys.stderr)
    fro = df[(df["corpus"] == "frontier") & (df["mechanism"] == "spsb_sealed")]
    tab3 = (fro.groupby(["model", "heuristic_primary"]).size()
            .unstack(fill_value=0))
    print((tab3.div(tab3.sum(axis=1), axis=0).round(3)).to_string(),
          file=sys.stderr)
    wm_counts = (df.loc[df["wrong_mechanism_derivation"] == 1, "model"]
                 .value_counts().to_dict())
    print(f"\nwrong_mechanism_derivation count: "
          f"{int(df['wrong_mechanism_derivation'].sum())} ({wm_counts})",
          file=sys.stderr)
    print(f"\nunclassified share overall: "
          f"{(df['heuristic_primary'] == 'U').mean():.3f}", file=sys.stderr)


if __name__ == "__main__":
    main()
