"""Within-model exact permutation test for bid error (rebuttal, ICLR 2027 submission 34553).
Each model contributes one treated run and its baseline runs; treatment is reassigned only within model.
Output: results/paper/within_model_permutation_bids.csv
"""
import sys
from pathlib import Path
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent / "semantic_judge"))
from summarize_judges import within_model_test, CONTRASTS, ROOT

df = pd.read_csv(ROOT / "results/traces/trace_features.csv", low_memory=False)
df = df[df.mechanism == "spsb_sealed"]
rows = []
for name, exp, base in CONTRASTS:
    b0, t1, e, p, n, pm = within_model_test(df, "abs_dev", exp, base)
    rows.append(dict(intervention=name, baseline_abs_dev=b0, treated_abs_dev=t1, effect=e, p_two_sided=p,
                     assignments=n, **{"effect_" + m.split("/")[-1]: v for m, v in pm.items()}))
out = pd.DataFrame(rows); out.to_csv(ROOT / "results/paper/within_model_permutation_bids.csv", index=False)
print(out.round(3).to_string(index=False))
