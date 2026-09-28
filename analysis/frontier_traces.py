#!/usr/bin/env python3
"""
Frontier-model and ascending-clock trace analyses (auction-v4 §5 re-analysis).

Four deliverables, all from results/traces/trace_features_v2.csv +
heuristic_labels.csv (no new model runs):

 1. formal_derivation_prevalence.csv
    Formal-derivation mode (FEATURES_V2 / taxonomy H7) by corpus x model x
    mechanism, with conditional bid error given H7 -- the quantitative basis
    for paper Heuristic 7 ("a distinct decision mode, not a human heuristic")
    and the capability boundary. Includes wrong_mechanism_derivation counts
    (equilibrium formula of the WRONG mechanism recited in a second/third-
    price cell -- the claude-sonnet-5 menu break).

 2. frontier_battery.csv
    The lever mini-battery on the four frontier models (only levers that were
    actually run at the frontier). Auction-level clusters (1-2 runs/cell);
    per-model AND pooled-frontier rows; NEVER pooled with the legacy battery.
    The two pilot_false_safety_* cells are marked exploratory=True (gated on
    co-author ratification per plan/STORY.md) and carry no headline claims.

 3. clock_h6.csv  (resolves TODO E-v3-6)
    Scores every ascending-clock trace (510 legacy + frontier + robustness)
    for the price-threshold-exit script (taxonomy H6): P(H6 stated) by
    corpus x model, exit-price deviation conditional on H6 stated vs not,
    and the stayed-beyond-value share P(dev > +2). Caveat carried in a
    column: the recorded clock 'bid' is the exit/drop price; winners are
    censored at the runner-up's exit, so negative deviations pool winner
    censoring with early exits.

 4. script_invariance.csv
    Cross-mechanism script invariance beyond GPT-4o (the paper's own J.7 #2
    agenda item): shading-script prevalence and bid/value ratio by model x
    mechanism in the robustness corpus (fpsb / spsb / tpsb / all-pay cells),
    testing whether "one shading script for all payment rules" replicates
    across families.

Deterministic, seed 1299. Run: python3 analysis/frontier_traces.py
"""

import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from build_trace_mediation_v2 import wcb_fast, perm_fast, boot_diff_ci  # noqa

REPO = os.path.dirname(HERE)
TRACES = os.path.join(REPO, "results", "traces")

V2_COLS = ["equilibrium_formula", "dominance_proof", "algebraic_derivation",
           "case_enumeration", "env_model_recital"]

FRONTIER_BASELINES = ["axis1_contingent_baseline", "axis3_beliefs_baseline"]
FRONTIER_LEVERS = [
    # label, experiment, target_feature, exploratory
    ("Payoff Safety", "axis2_forward_onestep", "safety_recognition", False),
    ("Payoff Tree", "axis2_forward_tree", "second_price_mention", False),
    ("Menu restatement", "intervention_menu", "second_price_mention", False),
    ("Clock-framing", "intervention_proxy_breitmoser", "clock_exit_language",
     False),
    ("Two-stage clock-exit descr.", "axis2_forward_baseline",
     "clock_exit_language", False),
    ("Second-order beliefs", "axis3_beliefs_secondorder", "opponent_modeling",
     False),
    ("False safety (up)", "pilot_false_safety_up", "safety_recognition", True),
    ("False safety (down)", "pilot_false_safety_down", "safety_recognition",
     True),
]


def formal_derivation_prevalence(df, labels):
    rows = []
    sealed = df[df["mechanism"].str.contains("sealed", na=False)]
    lab = labels.loc[sealed.index]
    for (corpus, model, mech), g in sealed.groupby(["corpus", "model",
                                                    "mechanism"]):
        gl = lab.loc[g.index]
        h7 = gl["h7"] == 1
        rows.append(dict(
            corpus=corpus, model=model, mechanism=mech, n=len(g),
            h7_share=round(h7.mean(), 4),
            h7_primary_share=round((gl["heuristic_primary"] == "H7").mean(), 4),
            t_share=round((gl["heuristic_primary"] == "T").mean(), 4),
            equilibrium_formula=round(g["equilibrium_formula"].mean(), 4),
            dominance_proof=round(g["dominance_proof"].mean(), 4),
            env_model_recital=round(g["env_model_recital"].mean(), 4),
            absdev_given_h7=round(g.loc[h7.values, "abs_dev"].mean(), 3)
            if h7.sum() else np.nan,
            absdev_given_not_h7=round(g.loc[~h7.values, "abs_dev"].mean(), 3)
            if (~h7).sum() else np.nan,
            wrong_mech_derivation_n=int(gl["wrong_mechanism_derivation"].sum()),
            dev_given_wrong_mech=round(
                g.loc[(gl["wrong_mechanism_derivation"] == 1).values,
                      "deviation"].mean(), 3)
            if gl["wrong_mechanism_derivation"].sum() else np.nan,
        ))
    return pd.DataFrame(rows)


def frontier_battery(df):
    fro = df[(df["corpus"] == "frontier")
             & (df["mechanism"] == "spsb_sealed")].copy()
    fro["clock_exit_language"] = fro["clock_exit_language"].astype(int)
    models = sorted(fro["model"].unique())
    rows = []
    for label, exp, target, explor in FRONTIER_LEVERS:
        for model in models + ["POOLED_FRONTIER"]:
            if model == "POOLED_FRONTIER":
                treat = fro[fro.experiment == exp]
                ctrl = fro[fro.experiment.isin(FRONTIER_BASELINES)]
            else:
                treat = fro[(fro.experiment == exp) & (fro.model == model)]
                ctrl = fro[fro.experiment.isin(FRONTIER_BASELINES)
                           & (fro.model == model)]
            if len(treat) == 0 or len(ctrl) == 0:
                continue
            both = pd.concat([treat.assign(treated=1),
                              ctrl.assign(treated=0)])
            d_prev = treat[target].mean() - ctrl[target].mean()
            _, p_lang, _ = wcb_fast(both[target], both["treated"],
                                    both["auction_id"].values)
            d_ad = treat["abs_dev"].mean() - ctrl["abs_dev"].mean()
            _, p_ad, ncl = wcb_fast(both["abs_dev"], both["treated"],
                                    both["auction_id"].values)
            perm_ad = perm_fast(both, "abs_dev", "auction_id")
            ci = boot_diff_ci(both, "abs_dev", "auction_id")
            rows.append(dict(
                lever=label, experiment=exp, model=model,
                exploratory=explor,
                n_treat=len(treat), n_ctrl=len(ctrl),
                n_auction_clusters=ncl,
                prev_baseline=round(ctrl[target].mean(), 4),
                prev_treated=round(treat[target].mean(), 4),
                lang_wcb_p_auction=round(p_lang, 4) if pd.notna(p_lang)
                else np.nan,
                dev_baseline=round(ctrl["deviation"].mean(), 3),
                dev_treated=round(treat["deviation"].mean(), 3),
                absdev_baseline=round(ctrl["abs_dev"].mean(), 3),
                absdev_treated=round(treat["abs_dev"].mean(), 3),
                absdev_diff=round(d_ad, 3),
                absdev_ci_lo=round(ci[0], 3), absdev_ci_hi=round(ci[1], 3),
                absdev_wcb_p_auction=round(p_ad, 4) if pd.notna(p_ad)
                else np.nan,
                absdev_perm_p_auction=round(perm_ad, 4) if pd.notna(perm_ad)
                else np.nan,
            ))
    return pd.DataFrame(rows)


def clock_h6(df, labels, by_experiment=False):
    clock = df[df["mechanism"] == "ascending_clock"].copy()
    lab = labels.loc[clock.index]
    keys = (["corpus", "model", "experiment"] if by_experiment
            else ["corpus", "model"])
    rows = []
    for key, g in clock.groupby(keys):
        corpus, model = key[0], key[1]
        gl = lab.loc[g.index]
        h6 = (gl["heuristic_primary"] == "H6").values
        rows.append(dict(
            corpus=corpus, model=model,
            experiment=(key[2] if by_experiment else "ALL"),
            n=len(g),
            p_h6_stated=round(h6.mean(), 4),
            p_h7=round((gl["heuristic_primary"] == "H7").mean(), 4),
            p_unclassified=round((gl["heuristic_primary"] == "U").mean(), 4),
            mean_dev=round(g["deviation"].mean(), 3),
            mean_absdev=round(g["abs_dev"].mean(), 3),
            mean_dev_h6=round(g.loc[h6, "deviation"].mean(), 3)
            if h6.sum() else np.nan,
            mean_absdev_h6=round(g.loc[h6, "abs_dev"].mean(), 3)
            if h6.sum() else np.nan,
            mean_dev_not_h6=round(g.loc[~h6, "deviation"].mean(), 3)
            if (~h6).sum() else np.nan,
            mean_absdev_not_h6=round(g.loc[~h6, "abs_dev"].mean(), 3)
            if (~h6).sum() else np.nan,
            n_losers=int((g["is_winner"] == 0).sum()),
            mean_absdev_losers=round(
                g.loc[g["is_winner"] == 0, "abs_dev"].mean(), 3)
            if (g["is_winner"] == 0).sum() else np.nan,
            p_stayed_beyond_value=round((g["deviation"] > 2).mean(), 4),
            note=("clock bid = exit/drop price; winners censored at "
                  "runner-up exit"),
        ))
    return pd.DataFrame(rows)


def script_invariance(df, labels):
    """Shading-script prevalence by model x mechanism -- replication of the
    GPT-4o-only 'one script for all payment rules' fact across families.
    Uses the robustness corpus (per-format cells) plus legacy GPT-4o."""
    pool = df[df["corpus"].isin(["robustness", "legacy"])
              & df["mechanism"].isin(["spsb_sealed", "fpsb_sealed",
                                      "tpsb_sealed", "allpay_sealed"])].copy()
    lab = labels.loc[pool.index]
    rows = []
    for (corpus, model, mech), g in pool.groupby(["corpus", "model",
                                                  "mechanism"]):
        gl = lab.loc[g.index]
        pos = g[g["value"] > 0]
        rows.append(dict(
            corpus=corpus, model=model, mechanism=mech, n=len(g),
            shading_intent=round(g["shading_intent"].mean(), 4),
            h1_or_h2_primary=round(
                gl["heuristic_primary"].isin(["H1", "H2"]).mean(), 4),
            overpay_concern=round(g["overpay_concern"].mean(), 4),
            mean_bid_over_value=round((pos["bid"] / pos["value"]).mean(), 4)
            if len(pos) else np.nan,
            mean_dev=round(g["deviation"].mean(), 3),
        ))
    return pd.DataFrame(rows)


def main():
    df = pd.read_csv(os.path.join(TRACES, "trace_features_v2.csv"),
                     low_memory=False)
    labels = pd.read_csv(os.path.join(TRACES, "heuristic_labels.csv"),
                         low_memory=False)
    assert len(df) == len(labels)

    fd = formal_derivation_prevalence(df, labels)
    fd.to_csv(os.path.join(TRACES, "formal_derivation_prevalence.csv"),
              index=False)
    print("=== formal-derivation prevalence (sealed) ===")
    print(fd[fd.mechanism == "spsb_sealed"]
          [["corpus", "model", "n", "h7_primary_share", "t_share",
            "absdev_given_h7", "absdev_given_not_h7",
            "wrong_mech_derivation_n", "dev_given_wrong_mech"]]
          .to_string(index=False))

    fb = frontier_battery(df)
    fb.to_csv(os.path.join(TRACES, "frontier_battery.csv"), index=False)
    print("\n=== frontier mini-battery (auction clusters) ===")
    print(fb[["lever", "model", "exploratory", "prev_treated",
              "dev_baseline", "dev_treated", "absdev_diff",
              "absdev_wcb_p_auction"]].to_string(index=False))

    ch = clock_h6(df, labels)
    ch.to_csv(os.path.join(TRACES, "clock_h6.csv"), index=False)
    print("\n=== clock H6 (price-threshold exit) ===")
    print(ch.drop(columns=["note"]).to_string(index=False))

    che = clock_h6(df, labels, by_experiment=True)
    che.to_csv(os.path.join(TRACES, "clock_h6_by_experiment.csv"),
               index=False)
    print("\n=== clock H6 by experiment (APV vs IPV description) ===")
    print(che[che.corpus == "frontier"].drop(columns=["note"])
          .to_string(index=False))

    si = script_invariance(df, labels)
    si.to_csv(os.path.join(TRACES, "script_invariance.csv"), index=False)
    print("\n=== cross-mechanism script invariance ===")
    print(si.to_string(index=False))


if __name__ == "__main__":
    main()
