"""
Two analyses that feed new sections of the Digital Mice report. No LLM calls.

  1. EA SMAD ranking (Section: Evolutionary search). Re-ranks the evolved
     mechanisms of run 9 by SMAD = 100 * mean|bid - target_bid| / mean|target_bid|,
     the scaled mean absolute deviation from each mechanism's OWN declared
     theoretical-optimum bid (truthful for second-price, (n-1)/n * v for
     first-price). Compares the SMAD ranking to the composite robust-fitness
     ranking. Writes analysis_v2/evolution/ea_smad_ranking.csv.

  2. Reasoning-chain audit (Section: Do the mice reason as instructed?). Parses
     the free-text PLAN reasoning logged for each of the 27 basis strains and
     measures how often a strain's reasoning invokes the second-price dominance
     argument (reconstructing the dominant strategy) and how often it conditions
     on opponents -- both of which the contingent-reasoning deficit forbids.
     Writes analysis_v2/reasoning_audit/reasoning_chain_audit.csv and
     reasoning_audit_examples.csv.

Run from repo root:
    ./venv/bin/python writeup_v2/reports/digital_mice/make_extra_analyses.py
"""

import glob
import json
import os
import re
import sqlite3

import pandas as pd

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
RESULTS_DB = os.path.join(REPO_ROOT, "data_v2/results.sqlite")
EA_RUN_ID = 9
BASIS_LOG_DIR = os.path.join(REPO_ROOT, "data_v2/experiment_logs/gpt5mini")
GRID = os.path.join(REPO_ROOT, "data_v2/results/spsb_basis_grid_summary.csv")

EVO_OUT = os.path.join(REPO_ROOT, "analysis_v2/evolution/ea_smad_ranking.csv")
AUDIT_OUT = os.path.join(REPO_ROOT, "analysis_v2/reasoning_audit/reasoning_chain_audit.csv")
AUDIT_EX_OUT = os.path.join(REPO_ROOT, "analysis_v2/reasoning_audit/reasoning_audit_examples.csv")


# --------------------------------------------------------------------------- #
# 1. EA SMAD ranking
# --------------------------------------------------------------------------- #
def ea_smad_ranking():
    conn = sqlite3.connect(RESULTS_DB)
    rows = conn.execute(
        "SELECT generation, op, name, rule_hash, fitness, metrics_json, "
        "genotype_json, coord_json FROM individuals WHERE run_id=? "
        "ORDER BY fitness DESC",
        (EA_RUN_ID,),
    ).fetchall()

    seen, recs = set(), []
    for gen, op, name, h, fit, mj, gj, cj in rows:
        if h in seen:
            continue
        seen.add(h)
        m = json.loads(mj) if mj else {}
        g = json.loads(gj) if gj else {}
        mech = g.get("mechanism", {}) or {}
        tgt = g.get("target_policy", {}) or {}
        smad = m.get("smad")
        recs.append({
            "name": name,
            "generation": gen,
            "op": op.replace("genotype_", ""),
            "payment_rule": mech.get("payment_rule", "?"),
            "bid_language": mech.get("bid_language", "?"),
            "benchmark_ratio": tgt.get("target_bid_value_ratio"),
            "smad": round(smad, 2) if isinstance(smad, (int, float)) else None,
            "robust_fitness": round(fit, 3),
            "efficiency": round(m.get("efficiency_mean", float("nan")), 3),
            "intended_action_rate": round(m.get("intended_action_rate") or float("nan"), 3),
            "overbid_value_rate": round(m.get("overbid_value_rate") or 0.0, 3),
            "near_zero_rate": round(m.get("near_zero_rate") or 0.0, 3),
        })

    df = pd.DataFrame(recs)
    df["rank_robust"] = df["robust_fitness"].rank(ascending=False, method="min").astype(int)
    # rank_smad assigned in-place (NaN SMAD -> NaN rank); no self-merge (names
    # are not unique across generations, which would fan out the rows).
    df["rank_smad"] = df["smad"].rank(ascending=True, method="min")
    df = df.sort_values("robust_fitness", ascending=False).reset_index(drop=True)

    os.makedirs(os.path.dirname(EVO_OUT), exist_ok=True)
    df.to_csv(EVO_OUT, index=False)

    # Spearman rank correlation between SMAD (lower better -> negate) and fitness.
    v = df.dropna(subset=["smad", "robust_fitness"])
    rho = (-v["smad"]).corr(v["robust_fitness"], method="spearman")
    smad_df = v
    print(f"[smad] {len(df)} unique mechs; {len(smad_df)} with valid SMAD; "
          f"Spearman(-SMAD, F_robust)={rho:.2f}")
    print(f"[smad] best by F_robust: {df.iloc[0]['payment_rule']} "
          f"(SMAD={df.iloc[0]['smad']}); "
          f"best by SMAD: {smad_df.sort_values('smad').iloc[0]['payment_rule']} "
          f"(SMAD={smad_df['smad'].min()})")
    print(f"[smad] wrote {os.path.relpath(EVO_OUT, REPO_ROOT)}")
    return df


# --------------------------------------------------------------------------- #
# 2. Reasoning-chain audit
# --------------------------------------------------------------------------- #

# Markers that a plan *concludes* the second-price dominance / truthful argument
# (the move the contingent-reasoning deficit, c=0, is instructed NOT to make).
# Deliberately excludes pure mechanism description ("the winner pays the
# second-highest bid"), which is boilerplate from the rule text, not leakage.
DOMINANCE_PATTERNS = [
    r"dominant strateg", r"weakly dominant", r"\btruthful",
    r"bid(ding)? (my|the|your)\s*(true |private )?value",
    r"bid equal to (my|the) value", r"exactly (my|the) value", r"vickrey",
]
# Markers that a plan conditions on opponents (also forbidden when c=0).
OPPONENT_PATTERNS = [
    r"other bidder", r"opponent", r"\brivals?\b", r"others (might|will|may|could) bid",
    r"if others", r"what others", r"competitor", r"the other two",
]
DOM_RE = re.compile("|".join(DOMINANCE_PATTERNS), re.I)
OPP_RE = re.compile("|".join(OPPONENT_PATTERNS), re.I)


def _plans_for_strain(strain_id):
    d = os.path.join(BASIS_LOG_DIR, f"spsb_basis_{strain_id}")
    plans = []
    for fp in glob.glob(os.path.join(d, "*.json")):
        try:
            obj = json.load(open(fp))
        except Exception:
            continue
        for _k, r in obj.items():
            for p in (r.get("plan") or []):
                if isinstance(p, str) and p.strip():
                    plans.append(p)
    return plans


def reasoning_audit():
    grid = pd.read_csv(GRID)
    # grid is long format (one row per bidder-decision); truthful = |b-v| <= 0.05.
    grid["_truthful"] = grid["deviation"].abs() <= 0.05
    truth_by_strain = grid.groupby("basis_id")["_truthful"].mean().to_dict()

    recs, examples = [], []
    for c in (0, 1, 2):
        for f in (0, 1, 2):
            for b in (0, 1, 2):
                sid = f"c{c}_f{f}_b{b}"
                plans = _plans_for_strain(sid)
                if not plans:
                    continue
                dom = [bool(DOM_RE.search(p)) for p in plans]
                opp = [bool(OPP_RE.search(p)) for p in plans]
                recs.append({
                    "strain": sid, "contingent": c, "forward": f, "beliefs": b,
                    "n_plans": len(plans),
                    "dominance_invocation_rate": round(sum(dom) / len(plans), 3),
                    "opponent_conditioning_rate": round(sum(opp) / len(plans), 3),
                    "truthful_rate": round(truth_by_strain.get(sid, float("nan")), 3),
                })
                # keep one leakage example for contingent-OFF strains that invoke dominance
                if c == 0:
                    for p, hit in zip(plans, dom):
                        if hit:
                            examples.append({"strain": sid, "quote": p.strip()[:320]})
                            break

    df = pd.DataFrame(recs)
    os.makedirs(os.path.dirname(AUDIT_OUT), exist_ok=True)
    df.to_csv(AUDIT_OUT, index=False)
    pd.DataFrame(examples).to_csv(AUDIT_EX_OUT, index=False)

    by_c = df.groupby("contingent")["dominance_invocation_rate"].mean()
    leak = df[(df["contingent"] == 0)]["dominance_invocation_rate"]
    print(f"[audit] {len(df)} strains; dominance-invocation by contingent level: "
          f"L0={by_c.get(0, float('nan')):.2f} L1={by_c.get(1, float('nan')):.2f} "
          f"L2={by_c.get(2, float('nan')):.2f}")
    print(f"[audit] contingent-OFF leakage rate: mean={leak.mean():.2f}, "
          f"max={leak.max():.2f}; n_examples={len(examples)}")
    print(f"[audit] wrote {os.path.relpath(AUDIT_OUT, REPO_ROOT)}")
    return df


if __name__ == "__main__":
    ea_smad_ranking()
    reasoning_audit()
