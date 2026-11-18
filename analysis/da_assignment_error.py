"""Common-scale outcome for DA cells: match differs from truthful-DA match; utility loss vs truthful.
Usage: python -I da_common_scale.py <experiment_logs/da dir> <out.csv>"""
import json, sys, glob, os, itertools, csv
from statistics import mean
root, out = sys.argv[1], sys.argv[2]

def da(prefs, prio):
    # prefs: {s: [schools...]}, prio: {s: {school: rank}} (1 = highest). student-proposing, unit capacity.
    nxt = {s: 0 for s in prefs}; held = {}; free = [s for s in prefs]
    while free:
        s = free.pop(0)
        if nxt[s] >= len(prefs[s]): continue
        c = prefs[s][nxt[s]]; nxt[s] += 1
        cur = held.get(c)
        if cur is None: held[c] = s
        elif prio[s][c] < prio[cur][c]: held[c] = s; free.append(cur)
        else: free.append(s)
    return {s: c for c, s in held.items()}

def true_pref(vals):  # descending value, ties broken alphabetically (matches the paper's deterministic tie-break assumption)
    return sorted(vals, key=lambda c: (-vals[c], c))

def kendall(rep, tru):
    pos = {c: i for i, c in enumerate(rep)}
    pairs = [(a, b) for a, b in itertools.combinations(tru, 2) if a in pos and b in pos]
    if not pairs: return None
    return sum(pos[a] > pos[b] for a, b in pairs) / len(pairs)

rows = []
for mdir in sorted(glob.glob(os.path.join(root, '*'))):
    model = os.path.basename(mdir)
    for cdir in sorted(glob.glob(os.path.join(mdir, '*'))):
        cell = os.path.basename(cdir)
        files = glob.glob(os.path.join(cdir, 'raw_data', '*.json'))
        if not files: continue
        mism, loss, lossp, kd, regret = [], [], [], [], []
        for f in files:
            try: d = json.load(open(f))
            except Exception: continue
            vals, prio, match = d.get('values'), d.get('priorities'), d.get('matches') or {}
            if not vals or not prio: continue
            tp = {s: true_pref(vals[s]) for s in vals}
            tm = da(tp, prio)
            reps = d.get('rankings')
            for s in vals:
                ut = vals[s][tm[s]] if s in tm else 0
                ur = vals[s][match[s]] if match.get(s) in vals[s] else 0
                mism.append(match.get(s) != tm.get(s)); loss.append(ut - ur); lossp.append((ut - ur) / ut if ut else 0)
                if reps and reps.get(s):
                    k = kendall(reps[s], tp[s]);
                    if k is not None: kd.append(k)
                    # individual regret: swap own report to truthful, others as reported
                    alt = {t: (tp[t] if t == s else (reps.get(t) or tp[t])) for t in vals}
                    am = da(alt, prio)
                    regret.append((vals[s][am[s]] if s in am else 0) - ur)
        n = len(mism)
        rows.append(dict(model=model, cell=cell, markets=len(files), students=n,
                         mismatch_pct=round(100*mean(mism),2) if n else None,
                         util_loss=round(mean(loss),3) if n else None,
                         util_loss_pct=round(100*mean(lossp),3) if n else None,
                         kendall_pct=round(100*mean(kd),2) if kd else None,
                         regret=round(mean(regret),3) if regret else None,
                         regret_pos_pct=round(100*mean([r>0 for r in regret]),2) if regret else None))
with open(out, 'w', newline='') as fh:
    w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
print(f'{len(rows)} rows -> {out}')
