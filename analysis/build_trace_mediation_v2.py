#!/usr/bin/env python3
"""
Full-battery dissociation analysis v2 (auction-v4 §5 re-analysis).

Upgrades analysis/build_trace_mediation.py from the 4-lever 2x2 to the FULL
20-lever probe battery, with the statistical discipline a battery requires:

  (a) full_battery      all 20 LEVERS: Delta target-language (pp) and
                        Delta |dev| ($) vs the provenance-corrected baselines,
                        wild-cluster-bootstrap p (run level, reproducing
                        manipulation_checks.csv exactly), cluster-permutation
                        p, and Benjamini-Hochberg FDR q-values ACROSS the
                        battery (language tests and bid tests corrected as two
                        separate families). Joint-response class at q<0.05
                        with `soft` flags (WCB-q significant, permutation not)
                        and two annotation axes: direction (improving /
                        backfiring) and echo_class (none / partial /
                        manipulation / by-construction). Variants: full +
                        dedup. (The legacy-baseline "precorrection" comparison
                        lives in the absdev_*_legacy columns, as before.)
  (b) per_family        every lever x model family: auction-level clustering
                        (auction_id from trace_features_v2.csv; run-level
                        clustering is degenerate at 1 treated run per model x
                        cell), fast vectorized WCB + auction-level permutation,
                        plus the 4-model sign vector that the paper's
                        robustness rule actually needs. Stated assumption:
                        auctions are independent API calls within a fixed
                        prompt; run-level idiosyncrasy is not identified.
  (c) tost              equivalence tests for the null claims. Cluster-
                        bootstrap 90% CI inside pre-committed margins:
                        delta_|dev| = $0.50 (sensitivity $1.00), delta_lang =
                        5pp. B2's safety_recognition is degenerate (0/600):
                        rule-of-three upper bound instead.
  (d) mediation_all     the B2 product-of-coefficients bound generalized to
                        Payoff Tree and the worst-case scaffold.
  (e) unanchored        autopsy of the 4,541 no-intent sealed SPSB traces,
                        per model, with taxonomy shares from
                        heuristic_labels.csv.
  (f) consistency_v2    stated-vs-realized fidelity per corpus x model
                        (adds the frontier models).
  (g) reconciliation    recomputes the Claude Payoff-Safety sign (+0.30) and
                        both prevalence tables from the frozen corpus in one
                        function; lists every writeup/ line carrying the
                        stale numbers.

Inputs:  results/traces/trace_features_v2.csv (legacy rows byte-equal to the
         frozen trace_features.csv -- asserted at build time),
         results/traces/heuristic_labels.csv,
         results/traces/mediation/manipulation_checks{,_dedup}.csv (reproduce-
         and-verify reference).
Outputs: results/traces/mediation/dissociation_full_battery{,_dedup}.csv,
         dissociation_per_family.csv, equivalence_tost.csv,
         mediation_bounds_all.csv, unanchored_autopsy.csv,
         consistency_pooled_v2.csv, reconciliation.md, battery_log.txt.

Reproducibility: numpy seed 1299 (SEED from build_trace_mediation), N_BOOT
2000, N_PERM 5000. Run: python3 analysis/build_trace_mediation_v2.py
"""

import os
import re
import sys

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from statsmodels.stats.multitest import multipletests

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from build_trace_mediation import (  # noqa: E402
    LEVERS, DISSOC_LEVERS, AXIS_BASELINES_CORRECTED, AXIS_BASELINES_LEGACY,
    SEED, N_BOOT, N_PERM, wild_cluster_boot_p, cluster_perm_p,
    cluster_boot_diff_ci, manipulation_checks, parse_stated_bid, sanity,
)

REPO = os.path.dirname(HERE)
TRACES = os.path.join(REPO, "results", "traces")
OUT = os.path.join(TRACES, "mediation")
os.makedirs(OUT, exist_ok=True)

V2_CSV = os.path.join(TRACES, "trace_features_v2.csv")
LABELS_CSV = os.path.join(TRACES, "heuristic_labels.csv")

ALPHA = 0.05
TOST_DELTA_DEV = (0.50, 1.00)   # $ margins on |dev| (primary, sensitivity)
TOST_DELTA_LANG = 0.05          # 5 pp margin on prevalence

# echo_class taxonomy (annotation axis for the battery figure):
#   none            clean language contrast (B2: prompt asserts the invariant
#                   but 0/600 echo it; Tree: shares rule vocab with baselines)
#   by-construction menu (prompt removes the target vocabulary wholesale;
#                   language axis not interpretable)
#   manipulation    clock-framing texts (the echo IS the treatment; the
#                   language cell verifies delivery, not uptake)
#   partial         everything else (prompt seeds the target vocabulary)
ECHO_CLASS = {
    "Payoff Safety": "none",
    "Payoff Tree": "none",
    "Menu restatement": "by-construction",
    "Clock-framing": "manipulation",
    "Two-stage clock-exit descr.": "manipulation",
}

MODELS = ["claude-3-5-haiku-20241022", "gemini-2.0-flash",
          "google/gemma-3-27b-it", "gpt-4o"]

LOG_LINES = []


def log(s=""):
    LOG_LINES.append(str(s))
    print(s)


# ---------------------------------------------------------------------------
# Fast vectorized wild cluster bootstrap (for the per-family battery, where
# auction-level clustering gives 100-250 clusters x 160 tests; the reference
# implementation is kept for the pooled battery to reproduce the published
# p-values bit-for-bit).
# ---------------------------------------------------------------------------
def wcb_fast(y, treated, clusters, n_boot=N_BOOT, seed=SEED):
    """Wild cluster bootstrap p for the treated coefficient in
    y ~ const + treated (no FE; use within one model family), Rademacher
    weights, restricted residuals, CR1. Vectorized over bootstrap draws.
    (np.errstate: the macOS Accelerate BLAS emits spurious divide/overflow
    warnings in matmul on clean float64 inputs; results verified against the
    reference wild_cluster_boot_p implementation.)"""
    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        return _wcb_fast_inner(y, treated, clusters, n_boot, seed)


def _wcb_fast_inner(y, treated, clusters, n_boot, seed):
    y = np.asarray(y, dtype=float)
    t = np.asarray(treated, dtype=float)
    if y.std() == 0 or len(np.unique(t)) < 2:
        return np.nan, np.nan, len(np.unique(clusters))
    X = np.column_stack([np.ones_like(y), t])
    uniq, cl_idx = np.unique(clusters, return_inverse=True)
    G = len(uniq)
    n, k = X.shape

    XtXi = np.linalg.pinv(X.T @ X)

    def cr1_t(yv):
        beta = XtXi @ (X.T @ yv)
        resid = yv - X @ beta
        s = np.zeros((G, k))
        for j in range(k):
            s[:, j] = np.bincount(cl_idx, weights=X[:, j] * resid,
                                  minlength=G)
        meat = s.T @ s
        adj = (G / (G - 1)) * ((n - 1) / (n - k)) if G > 1 and n > k else 1.0
        V = adj * XtXi @ meat @ XtXi
        se = np.sqrt(max(V[1, 1], 1e-30))
        return beta[1], beta[1] / se

    beta1, t_obs = cr1_t(y)
    if not np.isfinite(t_obs):
        return float(beta1), np.nan, G

    # restricted fit (treated coef = 0)
    ybar = y.mean()
    resid_r = y - ybar
    yhat_r = np.full_like(y, ybar)

    rng = np.random.default_rng(seed)
    W = rng.choice([-1.0, 1.0], size=(n_boot, G))
    # precompute per-cluster pieces
    A = np.zeros((G, k))       # X_c' resid_r_c
    B = np.zeros((G, k))       # X_c' yhat_r_c
    Gc = np.zeros((G, k, k))   # X_c' X_c
    for j in range(k):
        A[:, j] = np.bincount(cl_idx, weights=X[:, j] * resid_r, minlength=G)
        B[:, j] = np.bincount(cl_idx, weights=X[:, j] * yhat_r, minlength=G)
        for j2 in range(k):
            Gc[:, j, j2] = np.bincount(cl_idx, weights=X[:, j] * X[:, j2],
                                       minlength=G)
    base = B.sum(axis=0)                       # X'yhat_r
    XtY = base[None, :] + W @ A                # (n_boot, k)
    betas = XtY @ XtXi.T                       # (n_boot, k)
    # score contributions s_cb = B_c + w_bc A_c - G_c beta_b   -> (nb, G, k)
    S = (B[None, :, :] + W[:, :, None] * A[None, :, :]
         - np.einsum("gjk,bk->bgj", Gc, betas))
    meat = np.einsum("bgj,bgl->bjl", S, S)     # (nb, k, k)
    adj = (G / (G - 1)) * ((n - 1) / (n - k)) if G > 1 and n > k else 1.0
    V = adj * np.einsum("jk,bkl,lm->bjm", XtXi, meat, XtXi)
    se = np.sqrt(np.maximum(V[:, 1, 1], 1e-30))
    t_boot = betas[:, 1] / se
    p = float(np.mean(np.abs(t_boot) >= abs(t_obs)))
    p = min(1.0, (p * n_boot + 1) / (n_boot + 1))
    return float(beta1), p, G


def perm_fast(frame, dv, cluster_col, n_perm=N_PERM, seed=SEED):
    """Cluster-level permutation p (difference of cluster means)."""
    cl = frame.groupby(cluster_col).agg(treated=("treated", "first"),
                                        val=(dv, "mean")).reset_index()
    n_t = int(cl["treated"].sum())
    G = len(cl)
    if n_t == 0 or n_t == G or frame[dv].std() == 0:
        return np.nan
    obs = (cl.loc[cl.treated == 1, "val"].mean()
           - cl.loc[cl.treated == 0, "val"].mean())
    vals = cl["val"].values
    rng = np.random.default_rng(seed + 7)
    cnt = 0
    for _ in range(n_perm):
        perm = rng.permutation(G)
        stat = vals[perm[:n_t]].mean() - vals[perm[n_t:]].mean()
        if abs(stat) >= abs(obs):
            cnt += 1
    return (cnt + 1) / (n_perm + 1)


def boot_diff_ci(frame, dv, cluster_col, pct=(2.5, 97.5), n_boot=N_BOOT,
                 seed=SEED):
    """Cluster bootstrap CI for the treated-control mean difference (fast:
    resamples cluster aggregates, exact for difference-of-means)."""
    ag = frame.groupby(cluster_col).agg(treated=("treated", "first"),
                                        s=(dv, "sum"), n=(dv, "size"))
    t_ag = ag[ag.treated == 1][["s", "n"]].values
    c_ag = ag[ag.treated == 0][["s", "n"]].values
    if len(t_ag) == 0 or len(c_ag) == 0:
        return (np.nan, np.nan)
    rng = np.random.default_rng(seed + 3)
    G_t, G_c = len(t_ag), len(c_ag)
    diffs = np.empty(n_boot)
    for b in range(n_boot):
        ti = rng.integers(0, G_t, G_t)
        ci_ = rng.integers(0, G_c, G_c)
        ts, tn = t_ag[ti].sum(axis=0)
        cs, cn = c_ag[ci_].sum(axis=0)
        diffs[b] = ts / tn - cs / cn
    return tuple(np.percentile(diffs, pct))


# ---------------------------------------------------------------------------
# (a) Full battery with FDR + classes
# ---------------------------------------------------------------------------
def full_battery(sp, dedup=False, verify_against=None):
    mc = manipulation_checks(sp, dedup=dedup)
    if verify_against is not None and os.path.exists(verify_against):
        ref = pd.read_csv(verify_against)
        merged = mc.merge(ref, on="lever", suffixes=("", "_ref"))
        num_cols = ["prev_baseline", "prev_treated", "absdev_baseline",
                    "absdev_treated", "absdev_diff", "lang_wcb_p",
                    "absdev_wcb_p"]
        bad = []
        for c in num_cols:
            a = merged[c].astype(float)
            b = merged[f"{c}_ref"].astype(float)
            ok = (a.sub(b).abs() < 1e-6) | (a.isna() & b.isna())
            if not ok.all():
                bad.append((c, merged.loc[~ok, "lever"].tolist()))
        if bad:
            raise AssertionError(f"battery does not reproduce {verify_against}: {bad}")
        log(f"[verify] battery reproduces {os.path.basename(verify_against)} "
            f"exactly ({len(merged)} levers x {len(num_cols)} columns)")

    # FDR across the battery: language tests and bid tests as two families.
    lang_p = mc["lang_wcb_p"].values.astype(float)
    bid_p = mc["absdev_wcb_p"].values.astype(float)
    lang_q = np.full_like(lang_p, np.nan)
    m = ~np.isnan(lang_p)
    if m.sum():
        lang_q[m] = multipletests(lang_p[m], method="fdr_bh")[1]
    bid_q = np.full_like(bid_p, np.nan)
    m = ~np.isnan(bid_p)
    if m.sum():
        bid_q[m] = multipletests(bid_p[m], method="fdr_bh")[1]
    mc["lang_q"] = np.round(lang_q, 4)
    mc["bid_q"] = np.round(bid_q, 4)

    mc["echo_class"] = mc["lever"].map(ECHO_CLASS).fillna("partial")
    mc["direction"] = np.where(mc["absdev_diff"] < 0, "improving",
                               "backfiring")
    # Two-tier classification. The four DESIGNED contrasts (the paper's
    # pre-specified levers with ex-ante directional predictions: Payoff
    # Safety, Worst-case scaffold, Payoff Tree, Menu) are classified at the
    # raw WCB p (confirmatory tier); the 16 exploratory battery rows are
    # classified at the BH-FDR q (screening tier). Both p and q are reported
    # for every row; the permutation soft flags apply to both tiers.
    mc["tier"] = np.where(mc["lever"].isin(DISSOC_LEVERS), "designed",
                          "exploratory")
    lang_interp = (mc["echo_class"] != "by-construction") & mc["lang_q"].notna()
    mc["lang_interpretable"] = lang_interp
    lang_stat = np.where(mc["tier"] == "designed", mc["lang_wcb_p"],
                         mc["lang_q"])
    bid_stat = np.where(mc["tier"] == "designed", mc["absdev_wcb_p"],
                        mc["bid_q"])
    mc["moves_language"] = np.where(
        lang_interp & (lang_stat < ALPHA), "yes",
        np.where(~lang_interp, "n/a", "no"))
    mc["moves_bids"] = np.where(bid_stat < ALPHA, "yes", "no")
    mc["soft_bids"] = ((mc["moves_bids"] == "yes")
                       & (mc["absdev_perm_p"] >= ALPHA))
    mc["soft_lang"] = ((mc["moves_language"] == "yes")
                       & (mc["lang_perm_p"] >= ALPHA))
    mc["cell"] = np.select(
        [(mc.moves_language == "yes") & (mc.moves_bids == "yes"),
         (mc.moves_language == "yes") & (mc.moves_bids == "no"),
         (mc.moves_language != "yes") & (mc.moves_bids == "yes")],
        ["both", "language-only", "bids-only"], default="neither")
    return mc


# ---------------------------------------------------------------------------
# (b) Per-family battery (auction-level clusters + sign table)
# ---------------------------------------------------------------------------
def per_family(sp):
    rows = []
    for (label, fam, exp, target, base_list, echo) in LEVERS:
        for model in MODELS:
            treat = sp[(sp.experiment == exp) & (sp.model == model)]
            ctrl = sp[sp.experiment.isin(base_list) & (sp.model == model)]
            if len(treat) == 0 or len(ctrl) == 0:
                continue
            both = pd.concat([treat.assign(treated=1),
                              ctrl.assign(treated=0)])
            n_cl = both["auction_id"].nunique()
            d_prev = treat[target].mean() - ctrl[target].mean()
            _, p_lang, _ = wcb_fast(both[target], both["treated"],
                                    both["auction_id"].values)
            d_ad = treat["abs_dev"].mean() - ctrl["abs_dev"].mean()
            _, p_ad, _ = wcb_fast(both["abs_dev"], both["treated"],
                                  both["auction_id"].values)
            perm_ad = perm_fast(both, "abs_dev", "auction_id")
            ci = boot_diff_ci(both, "abs_dev", "auction_id")
            rows.append(dict(
                lever=label, family=fam, model=model,
                n_treat=len(treat), n_ctrl=len(ctrl),
                n_auction_clusters=n_cl,
                prev_baseline=round(ctrl[target].mean(), 4),
                prev_treated=round(treat[target].mean(), 4),
                prev_diff=round(d_prev, 4),
                lang_wcb_p_auction=round(p_lang, 4) if pd.notna(p_lang) else np.nan,
                absdev_baseline=round(ctrl["abs_dev"].mean(), 3),
                absdev_treated=round(treat["abs_dev"].mean(), 3),
                absdev_diff=round(d_ad, 3),
                absdev_ci_lo=round(ci[0], 3), absdev_ci_hi=round(ci[1], 3),
                absdev_wcb_p_auction=round(p_ad, 4) if pd.notna(p_ad) else np.nan,
                absdev_perm_p_auction=round(perm_ad, 4) if pd.notna(perm_ad) else np.nan,
                dev_baseline=round(ctrl["deviation"].mean(), 3),
                dev_treated=round(treat["deviation"].mean(), 3),
                sign=np.sign(round(d_ad, 3)),
            ))
    pf = pd.DataFrame(rows)
    # 4-model sign consistency per lever (the paper's robustness rule)
    sign_tab = (pf.groupby("lever")["sign"]
                .agg(lambda s: "".join("+" if x > 0 else "-" if x < 0 else "0"
                                       for x in s)))
    pf = pf.merge(sign_tab.rename("sign_vector_4models"), on="lever")
    return pf


# ---------------------------------------------------------------------------
# (c) TOST equivalence for the null claims
# ---------------------------------------------------------------------------
def tost_table(sp):
    """CI-inclusion TOST: equivalence at margin delta iff the cluster-
    bootstrap 90% CI of the treated-control difference lies inside
    [-delta, +delta]. Run-level clusters (headline convention)."""
    targets = [
        # (lever, experiment, dv, margins, kind)
        ("Worst-case scaffold", "axis1_contingent_worstcase",
         ["axis1_contingent_baseline"], "abs_dev", TOST_DELTA_DEV, "bids"),
        ("Menu restatement", "intervention_menu",
         AXIS_BASELINES_CORRECTED, "abs_dev", TOST_DELTA_DEV, "bids"),
        ("Payoff Tree", "axis2_forward_tree",
         AXIS_BASELINES_CORRECTED, "second_price_mention",
         (TOST_DELTA_LANG,), "language"),
        ("Payoff Safety", "axis2_forward_onestep",
         AXIS_BASELINES_CORRECTED, "payment_rule_correct",
         (TOST_DELTA_LANG,), "language"),
    ]
    rows = []
    for lever, exp, base, dv, margins, kind in targets:
        treat = sp[sp.experiment == exp]
        ctrl = sp[sp.experiment.isin(base)]
        both = pd.concat([treat.assign(treated=1), ctrl.assign(treated=0)])
        diff = treat[dv].mean() - ctrl[dv].mean()
        # Two cluster levels: run (conservative headline convention; 8-12
        # clusters, so CIs are wide and equivalence is hard to certify) and
        # auction (stated assumption: auctions are independent API calls;
        # 150-500 clusters). Both reported.
        for cl_col, cl_name in [("run_id", "run"), ("auction_id", "auction")]:
            lo90, hi90 = boot_diff_ci(both, dv, cl_col, pct=(5, 95))
            for delta in margins:
                rows.append(dict(
                    lever=lever, dv=dv, kind=kind, cluster_level=cl_name,
                    n_clusters=both[cl_col].nunique(),
                    diff=round(diff, 4),
                    ci90_lo=round(lo90, 4), ci90_hi=round(hi90, 4),
                    margin=delta,
                    equivalent=bool(lo90 > -delta and hi90 < delta),
                ))
    # B2 safety_recognition: degenerate zero-variance cell -> rule of three.
    n_treat = int((sp.experiment == "axis2_forward_onestep").sum())
    rows.append(dict(
        lever="Payoff Safety", dv="safety_recognition", kind="language",
        diff=0.0, ci90_lo=np.nan, ci90_hi=np.nan, margin=np.nan,
        equivalent=np.nan,
        note=(f"degenerate: 0/{n_treat} treated traces echo the invariant; "
              f"rule-of-three one-sided 95% upper bound on prevalence = "
              f"{3 / n_treat:.4f}")))
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# (d) Mediation bounds generalized (B2 pattern -> Tree, worst-case)
# ---------------------------------------------------------------------------
def mediation_lever(sp, exp, meds, base_list, label):
    treat = sp[sp.experiment == exp].copy()
    ctrl = sp[sp.experiment.isin(base_list)].copy()
    both = pd.concat([treat.assign(treated=1), ctrl.assign(treated=0)]).copy()
    both["any_med"] = (both[meds].sum(axis=1) > 0).astype(int)
    tot = treat["abs_dev"].mean() - ctrl["abs_dev"].mean()
    rng = np.random.default_rng(SEED + 11)
    clusters = both["run_id"].unique()
    groups = {c: both[both["run_id"] == c] for c in clusters}
    acmes, shares = [], []
    for _ in range(N_BOOT):
        pick = rng.choice(clusters, size=len(clusters), replace=True)
        boot = pd.concat([groups[c] for c in pick])
        bt, bc = boot[boot.treated == 1], boot[boot.treated == 0]
        if len(bt) == 0 or len(bc) == 0:
            continue
        a = bt["any_med"].mean() - bc["any_med"].mean()
        if bt["any_med"].nunique() > 1:
            b = (bt.loc[bt.any_med == 1, "abs_dev"].mean()
                 - bt.loc[bt.any_med == 0, "abs_dev"].mean())
        else:
            b = 0.0
        t = bt["abs_dev"].mean() - bc["abs_dev"].mean()
        acmes.append(a * b)
        if t != 0:
            shares.append(a * b / t)
    acmes, shares = np.array(acmes), np.array(shares)
    a_pt = treat[meds].sum(axis=1).gt(0).mean() - ctrl[meds].sum(axis=1).gt(0).mean()
    within = treat.assign(any_med=(treat[meds].sum(axis=1) > 0).astype(int))
    if within["any_med"].nunique() > 1:
        b_gap = (within.loc[within.any_med == 1, "abs_dev"].mean()
                 - within.loc[within.any_med == 0, "abs_dev"].mean())
    else:
        b_gap = np.nan
    return dict(
        lever=label, experiment=exp, mediators="+".join(meds),
        n_treat=len(treat), n_ctrl=len(ctrl),
        first_stage_a=round(a_pt, 4),
        within_treat_gap_b=round(b_gap, 4) if pd.notna(b_gap) else np.nan,
        total_effect_absdev=round(tot, 4),
        acme_median=round(float(np.median(acmes)), 4),
        acme_ci_lo=round(float(np.percentile(acmes, 2.5)), 4),
        acme_ci_hi=round(float(np.percentile(acmes, 97.5)), 4),
        share_median=round(float(np.median(shares)), 4),
        share_abs_upper95=round(float(np.percentile(np.abs(shares), 95)), 4),
    )


# ---------------------------------------------------------------------------
# (e) Unanchored autopsy
# ---------------------------------------------------------------------------
def unanchored_autopsy(sp, labels):
    no_int = sp[(sp["shading_intent"] == 0) & (sp["overbid_intent"] == 0)
                & (sp["truthful_intent"] == 0)].copy()
    lab = labels.loc[no_int.index]
    rows = []
    for model, g in no_int.groupby("model"):
        gl = lab.loc[g.index]
        stated_cov = g["plan"].apply(parse_stated_bid).notna().mean()
        rows.append(dict(
            model=model, n=len(g),
            share_of_model_traces=round(
                len(g) / (sp.model == model).sum(), 3),
            mean_dev=round(g["deviation"].mean(), 3),
            mean_absdev=round(g["abs_dev"].mean(), 3),
            p_dev_below_m5=round((g["deviation"] < -5).mean(), 3),
            opponent_modeling=round(g["opponent_modeling"].mean(), 3),
            probability_reasoning=round(g["probability_reasoning"].mean(), 3),
            overpay_concern=round(g["overpay_concern"].mean(), 3),
            margin_language=round(g["margin_language"].mean(), 3),
            conservative_language=round(g["conservative_language"].mean(), 3),
            none_of_18=round((g[[c for c in FEATURE_18 if c in g.columns]]
                              .sum(axis=1) == 0).mean(), 3),
            stated_bid_coverage=round(stated_cov, 3),
            share_H3=round((gl["heuristic_primary"] == "H3").mean(), 3),
            share_H4=round((gl["heuristic_primary"] == "H4").mean(), 3),
            share_H2=round((gl["heuristic_primary"] == "H2").mean(), 3),
            share_U=round((gl["heuristic_primary"] == "U").mean(), 3),
        ))
    pooled = dict(
        model="POOLED", n=len(no_int),
        share_of_model_traces=round(len(no_int) / len(sp), 3),
        mean_dev=round(no_int["deviation"].mean(), 3),
        mean_absdev=round(no_int["abs_dev"].mean(), 3),
        p_dev_below_m5=round((no_int["deviation"] < -5).mean(), 3))
    return pd.DataFrame(rows + [pooled])


FEATURE_18 = [
    "dominance_language", "truthful_intent", "payment_rule_correct",
    "second_price_mention", "first_price_mention", "opponent_modeling",
    "probability_reasoning", "expected_value_reasoning", "shading_intent",
    "overbid_intent", "worst_case", "safety_recognition", "overpay_concern",
    "zero_profit_fallacy", "margin_language", "conservative_language",
    "aggressive_language", "risk_language",
]


# ---------------------------------------------------------------------------
# (f) Consistency / execution fidelity, per corpus x model
# ---------------------------------------------------------------------------
def consistency_v2(df_all, labels_all):
    """Per corpus x model fidelity. Caveat measured, not assumed: the frozen
    shading_intent regex has no negation handling, and frontier formal-
    derivation traces routinely DISCUSS shading in order to reject it
    ("shading only increases the chance of losing"), which fires the regex
    and mechanically deflates dir_consistency. dir_consistency_excl_h7
    excludes taxonomy-H7 traces to quantify the artifact."""
    sealed = df_all[df_all["mechanism"].str.contains("sealed", na=False)].copy()
    h7 = labels_all.loc[sealed.index, "h7"] == 1
    sealed["is_h7"] = h7.values
    sealed["stated_bid"] = sealed["plan"].apply(parse_stated_bid)

    def direction_match(r):
        if r["overbid_intent"] == 1 and r["shading_intent"] == 0:
            return int(r["deviation"] > 0)
        if r["shading_intent"] == 1 and r["overbid_intent"] == 0:
            return int(r["deviation"] < 0)
        if (r["truthful_intent"] == 1 and r["shading_intent"] == 0
                and r["overbid_intent"] == 0):
            return int(r["abs_dev"] <= 0.5)
        return np.nan
    sealed["dir_match"] = sealed.apply(direction_match, axis=1)
    rows = []
    for (corpus, model), g in sealed.groupby(["corpus", "model"]):
        stated = g[g["dir_match"].notna()]
        stated_nh7 = stated[~stated["is_h7"]]
        cal = g[g["stated_bid"].notna()].copy()
        cal["sb_err"] = cal["stated_bid"] - cal["bid"]
        rows.append(dict(
            corpus=corpus, model=model, n=len(g),
            n_intent_stated=len(stated),
            dir_consistency=round(stated["dir_match"].mean(), 4)
            if len(stated) else np.nan,
            dir_consistency_excl_h7=round(stated_nh7["dir_match"].mean(), 4)
            if len(stated_nh7) else np.nan,
            n_calib=len(cal),
            calib_coverage=round(len(cal) / len(g), 4),
            exact_match=round((cal["sb_err"].abs() < 0.01).mean(), 4)
            if len(cal) else np.nan,
            corr_stated_realized=round(cal["stated_bid"].corr(cal["bid"]), 4)
            if len(cal) > 2 else np.nan,
            mean_abs_stated_minus_realized=round(cal["sb_err"].abs().mean(), 4)
            if len(cal) else np.nan,
        ))
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# (g) Reconciliation
# ---------------------------------------------------------------------------
def reconciliation(sp):
    lines = ["# Reconciliation report (auto-generated by "
             "build_trace_mediation_v2.py)", ""]
    # (i) Claude Payoff-Safety sign
    cl = sp[(sp.experiment == "axis2_forward_onestep")
            & (sp.model == "claude-3-5-haiku-20241022")]
    lines.append("## Claude Payoff-Safety sign flag")
    lines.append("")
    lines.append(f"Recomputed from the frozen corpus: mean deviation = "
                 f"**{cl['deviation'].mean():+.3f}** (n={len(cl)}), "
                 f"mean |dev| = {cl['abs_dev'].mean():.3f}.")
    lines.append("The ES manuscript lineage reports **−0.30** for this cell; "
                 "the data give **+0.30** (mild overbidding). Same magnitude; "
                 "|dev| quantities unaffected. Manuscript transcription "
                 "error, already logged in mediation_summary.md item 5.")
    lines.append("")
    lines.append("Occurrences of the stale numbers in writeup/ (manual fix "
                 "list for the next prose pass):")
    lines.append("")
    pat = re.compile(r"[-−]0\.30|[-−]0\.49")
    wu = os.path.join(REPO, "writeup")
    for root, _dirs, files in os.walk(wu):
        if any(x in root for x in (".git", "figures")):
            continue
        for fn in sorted(files):
            if not fn.endswith((".tex", ".md")):
                continue
            fp = os.path.join(root, fn)
            try:
                for i, line in enumerate(open(fp, errors="ignore"), 1):
                    if pat.search(line) and ("afety" in line or "onestep" in line
                                             or "laude" in line or "B2" in line
                                             or "0.49" in line):
                        rel = os.path.relpath(fp, REPO)
                        lines.append(f"- `{rel}:{i}` {line.strip()[:160]}")
            except OSError:
                pass
    lines.append("")
    # (ii) the two prevalence tables from one function
    lines.append("## Prevalence tables regenerated from the frozen corpus")
    lines.append("")
    lines.append("Pooled ALL sealed-SPSB cells per model (the text's "
                 "'fingerprint' numbers):")
    pooled = sp.groupby("model")[["shading_intent", "overbid_intent",
                                  "truthful_intent", "opponent_modeling",
                                  "overpay_concern", "first_price_mention",
                                  "payment_rule_correct",
                                  "dominance_language"]].mean().round(3)
    lines.append("")
    lines.append("```\n" + pooled.to_string() + "\n```")
    lines.append("")
    lines.append("Baseline `spsb` cell only (n=150/model; the appendix "
                 "table's granularity):")
    base = sp[sp.experiment == "spsb"]
    base_tab = base.groupby("model")[FEATURE_18].mean().round(3)
    lines.append("")
    lines.append("```\n" + base_tab.to_string() + "\n```")
    lines.append("")
    lines.append("Verdict: both granularities are internally correct; the "
                 "manuscript must label which one each number uses "
                 "(pooled-cells fingerprints vs baseline-cell table). Known "
                 "adjacent-numbers pairs: Gemini shading 0.86 pooled vs "
                 f"{base_tab.loc['gemini-2.0-flash', 'shading_intent']:.2f} "
                 "baseline-only; Gemma 0.106 pooled vs "
                 f"{base_tab.loc['google/gemma-3-27b-it', 'shading_intent']:.2f}.")
    with open(os.path.join(OUT, "reconciliation.md"), "w") as f:
        f.write("\n".join(lines) + "\n")
    log(f"[reconciliation] Claude B2 mean dev = {cl['deviation'].mean():+.3f} "
        f"(manuscript says −0.30)")


# ---------------------------------------------------------------------------
def main():
    df_all = pd.read_csv(V2_CSV, low_memory=False)
    labels = pd.read_csv(LABELS_CSV, low_memory=False)
    assert len(df_all) == len(labels)

    legacy = df_all[df_all["corpus"] == "legacy"].copy()
    sp = legacy[legacy["mechanism"] == "spsb_sealed"].copy()
    sanity(sp, log)

    # (a) full battery, verified against the published manipulation checks
    log("=== (a) full battery (pooled, run-level WCB, BH-FDR) ===")
    mc = full_battery(sp, dedup=False,
                      verify_against=os.path.join(OUT,
                                                  "manipulation_checks.csv"))
    mc.to_csv(os.path.join(OUT, "dissociation_full_battery.csv"), index=False)
    log(mc[["lever", "prev_diff", "lang_q", "absdev_diff", "bid_q",
            "echo_class", "direction", "cell", "soft_bids"]]
        .to_string(index=False))

    log("\n=== (a2) full battery, deduplicated ===")
    mc_dd = full_battery(sp, dedup=True,
                         verify_against=os.path.join(
                             OUT, "manipulation_checks_dedup.csv"))
    mc_dd.to_csv(os.path.join(OUT, "dissociation_full_battery_dedup.csv"),
                 index=False)
    flips = mc.merge(mc_dd[["lever", "cell"]], on="lever",
                     suffixes=("", "_dedup"))
    flips = flips[flips.cell != flips.cell_dedup]
    log("dedup cell flips: " + (flips[["lever", "cell", "cell_dedup"]]
                                .to_string(index=False) if len(flips)
                                else "none"))

    # (b) per-family
    log("\n=== (b) per-family battery (auction-level clusters) ===")
    pf = per_family(sp)
    pf.to_csv(os.path.join(OUT, "dissociation_per_family.csv"), index=False)
    headline = pf[pf.lever.isin(["Payoff Safety", "Payoff Tree",
                                 "Worst-case scaffold", "Menu restatement",
                                 "First-order beliefs",
                                 "Risk-averse persona"])]
    log(headline[["lever", "model", "prev_diff", "lang_wcb_p_auction",
                  "absdev_diff", "absdev_wcb_p_auction",
                  "sign_vector_4models"]].to_string(index=False))

    # (c) TOST
    log("\n=== (c) TOST equivalence ===")
    tt = tost_table(sp)
    tt.to_csv(os.path.join(OUT, "equivalence_tost.csv"), index=False)
    log(tt.to_string(index=False))

    # (d) mediation bounds
    log("\n=== (d) mediation bounds (B2 / Tree / Worst-case) ===")
    med_rows = [
        mediation_lever(sp, "axis2_forward_onestep",
                        ["payment_rule_correct", "safety_recognition",
                         "second_price_mention"],
                        AXIS_BASELINES_CORRECTED, "Payoff Safety"),
        mediation_lever(sp, "axis2_forward_tree",
                        ["second_price_mention", "payment_rule_correct",
                         "dominance_language"],
                        AXIS_BASELINES_CORRECTED, "Payoff Tree"),
        mediation_lever(sp, "axis1_contingent_worstcase", ["worst_case"],
                        ["axis1_contingent_baseline"], "Worst-case scaffold"),
    ]
    med = pd.DataFrame(med_rows)
    med.to_csv(os.path.join(OUT, "mediation_bounds_all.csv"), index=False)
    log(med.to_string(index=False))

    # (e) unanchored autopsy
    log("\n=== (e) unanchored autopsy ===")
    ua = unanchored_autopsy(sp, labels)
    ua.to_csv(os.path.join(OUT, "unanchored_autopsy.csv"), index=False)
    log(ua.to_string(index=False))

    # (f) consistency v2 (adds frontier)
    log("\n=== (f) consistency / execution fidelity v2 ===")
    lf_mask = df_all["corpus"].isin(["legacy", "frontier"])
    cv = consistency_v2(df_all[lf_mask], labels[lf_mask])
    cv.to_csv(os.path.join(OUT, "consistency_pooled_v2.csv"), index=False)
    log(cv.to_string(index=False))

    # (g) reconciliation
    log("\n=== (g) reconciliation ===")
    reconciliation(sp)

    with open(os.path.join(OUT, "battery_log.txt"), "w") as f:
        f.write("\n".join(LOG_LINES) + "\n")
    log(f"\nWrote outputs to {OUT}")


if __name__ == "__main__":
    main()
