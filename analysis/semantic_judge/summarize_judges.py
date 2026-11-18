"""Summarize the two-judge semantic validation used in the ICLR 2027 rebuttal (submission 34553).

Inputs: results/traces/trace_features.csv and the two judge label files in results/traces/semantic_judge/.
Outputs (results/traces/semantic_judge/): judge_agreement.csv, keyword_vs_judge.csv, semantic_effects.csv, summary.md.
Inference: within-model exact permutation over run clusters (each model: treated run vs its baseline runs),
the same test the rebuttal reports for bid error. Run: python analysis/semantic_judge/summarize_judges.py
"""
import hashlib, itertools, json
from pathlib import Path
import numpy as np, pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results/traces/semantic_judge"
JUDGES = {"claude-sonnet-5-5": OUT / "labels_claude-sonnet-5-5.jsonl", "gpt-6.1-sol": OUT / "labels_gpt-6.1-sol.jsonl"}
CONSTRUCTS = ["safety_reason_valid", "says_truthful_best", "payment_rule_correct", "shading_intent",
              "worst_case", "opponent_modeling", "dominance_language"]
KEYWORD_FEATURES = ["payment_rule_correct", "truthful_intent", "shading_intent", "worst_case",
                    "opponent_modeling", "dominance_language"]
BASE = ["axis1_contingent_baseline", "axis3_beliefs_baseline"]
CONTRASTS = [("Payoff Safety", "axis2_forward_onestep", BASE),
             ("Payoff Tree", "axis2_forward_tree", BASE),
             ("Worst-case scaffold", "axis1_contingent_worstcase", ["axis1_contingent_baseline"]),
             ("Dominated scaffold", "axis1_contingent_dominated", ["axis1_contingent_baseline"]),
             ("First-order beliefs", "axis3_beliefs_firstorder", ["axis3_beliefs_baseline"]),
             ("Second-order beliefs", "axis3_beliefs_secondorder", ["axis3_beliefs_baseline"])]

def load(path):
    lab = {}
    for line in open(path):
        r = json.loads(line)
        if "result" in r:
            lab[r["h"]] = r["result"]
    return lab

def kappa(a, b):
    po = (a == b).mean(); pe = a.mean() * b.mean() + (1 - a.mean()) * (1 - b.mean())
    return po, (po - pe) / (1 - pe) if pe < 1 else float("nan")

def within_model_test(df, col, exp, base):
    sub = df[df.experiment.isin(base + [exp])].copy(); sub["t"] = (sub.experiment == exp).astype(int)
    cl = sub.groupby(["model", "run_id"]).agg(t=("t", "first"), v=(col, "mean")).reset_index()
    per_model = {m: g[g.t == 1].v.mean() - g[g.t == 0].v.mean() for m, g in cl.groupby("model")}
    base_mean = np.mean([g[g.t == 0].v.mean() for _, g in cl.groupby("model")])
    obs = np.mean(list(per_model.values())); options = []
    for _, g in cl.groupby("model"):
        vals = g.v.values; nt = int(g.t.sum()); o = []
        for idx in itertools.combinations(range(len(vals)), nt):
            mask = np.zeros(len(vals), bool); mask[list(idx)] = True
            o.append(vals[mask].mean() - vals[~mask].mean())
        options.append(o)
    stats = [np.mean(c) for c in itertools.product(*options)]
    p = np.mean([abs(s) >= abs(obs) - 1e-12 for s in stats])
    return base_mean, base_mean + obs, obs, p, len(stats), per_model

def main():
    df = pd.read_csv(ROOT / "results/traces/trace_features.csv", low_memory=False)
    df = df[df.mechanism == "spsb_sealed"].copy()
    df["h"] = df.plan.map(lambda p: hashlib.sha256(p.encode()).hexdigest())
    labs = {j: load(p) for j, p in JUDGES.items()}
    (ja, la), (jb, lb) = labs.items()
    common = sorted(set(la) & set(lb))
    rows = []
    for k in CONSTRUCTS:
        a = np.array([bool(la[h]["labels"][k]) for h in common]); b = np.array([bool(lb[h]["labels"][k]) for h in common])
        po, kp = kappa(a, b); rows.append(dict(construct=k, n=len(common), agreement=po, kappa=kp, rate_a=a.mean(), rate_b=b.mean()))
    agree = pd.DataFrame(rows)
    direction = np.mean([la[h]["direction"] == lb[h]["direction"] for h in common])
    agree.to_csv(OUT / "judge_agreement.csv", index=False)
    kw = []
    for j, lab in labs.items():
        u = df[df.h.isin(lab)].drop_duplicates("h")
        for k in KEYWORD_FEATURES:
            key = u[k].astype(bool); jud = u.h.map(lambda h: bool(lab[h]["labels"][k]))
            tp = (key & jud).sum()
            kw.append(dict(judge=j, feature=k, n=len(u), keyword_rate=key.mean(), judge_rate=jud.mean(),
                           precision=tp / key.sum() if key.sum() else np.nan, recall=tp / jud.sum() if jud.sum() else np.nan))
    pd.DataFrame(kw).to_csv(OUT / "keyword_vs_judge.csv", index=False)
    eff = []
    for j, lab in labs.items():
        d = df[df.h.isin(lab)].copy()
        for k in CONSTRUCTS:
            d["j_" + k] = d.h.map(lambda h: float(bool(lab[h]["labels"][k])))
        d["j_below"] = d.h.map(lambda h: float(lab[h]["direction"] == "below"))
        for name, exp, base in CONTRASTS:
            for col in ["abs_dev"] + ["j_" + k for k in CONSTRUCTS] + ["j_below"]:
                b0, t1, e, p, n, pm = within_model_test(d, col, exp, base)
                eff.append(dict(judge=j, intervention=name, outcome=col, baseline=b0, treated=t1, effect=e, p=p,
                                assignments=n, **{"effect_" + m.split("/")[-1]: v for m, v in pm.items()}))
    E = pd.DataFrame(eff); E.to_csv(OUT / "semantic_effects.csv", index=False)
    pick = lambda j, i, o: E[(E.judge == j) & (E.intervention == i) & (E.outcome == o)].iloc[0]
    lines = ["# Semantic validation of plan-level text measures (ICLR 2027 rebuttal)", "",
             f"Judges: {ja}, {jb}. Plans labeled by both: {len(common)} of {df[df.experiment.isin(sum([[c[1]]+c[2] for c in CONTRASTS],[]))].h.nunique()} unique plans in the scored cells.",
             f"Mean agreement across {len(CONSTRUCTS)} constructs: {agree.agreement.mean():.1%} (mean kappa {agree.kappa.mean():.2f}); intended bid direction: {direction:.1%}.", "",
             "| Contrast | Outcome | " + " | ".join(labs) + " |", "|---|---|" + "---|" * len(labs)]
    for i, o in [("Payoff Safety", "j_safety_reason_valid"), ("Payoff Safety", "j_says_truthful_best"), ("Payoff Safety", "j_below"),
                 ("Payoff Tree", "j_payment_rule_correct"), ("Payoff Tree", "j_below"),
                 ("Worst-case scaffold", "j_worst_case"), ("Dominated scaffold", "j_dominance_language"),
                 ("First-order beliefs", "j_opponent_modeling")]:
        cells = [f"{100*pick(j,i,o).baseline:.1f}% → {100*pick(j,i,o).treated:.1f}% (p={pick(j,i,o).p:.3f})" for j in labs]
        lines.append(f"| {i} | {o[2:]} | " + " | ".join(cells) + " |")
    lines += ["", "Bid error (abs_dev) contrasts are identical across judge files up to coverage; see semantic_effects.csv. Labels are LLM classifications of plan text, not a human reference."]
    (OUT / "summary.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))

if __name__ == "__main__":
    main()
