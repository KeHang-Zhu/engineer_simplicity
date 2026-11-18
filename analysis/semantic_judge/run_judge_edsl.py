"""Same blind judge as run_judge.py, run through EDSL (Expected Parrot remote inference).
Writes records in run_judge.py's cache format so `run_judge.py --analyze --labels <file>` works unchanged.

Key: ~/.expected_parrot_key (one line).
  python run_judge_edsl.py --list-models [filter]
  python run_judge_edsl.py --model <edsl model name> [--service <service>] [--limit N] --out edsl_<name>.jsonl
"""
import argparse, json, os, sys, time
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import run_judge as rj  # rubric (SYS), cells, schema hash, validator

key = Path.home().joinpath('.expected_parrot_key').read_text().strip()
if not key or key.startswith('PASTE_'):
    sys.exit('Put your Expected Parrot key in ~/.expected_parrot_key first.')
os.environ['EXPECTED_PARROT_API_KEY'] = key

from edsl import QuestionFreeText, ScenarioList, Model

ap = argparse.ArgumentParser()
ap.add_argument('--list-models', nargs='?', const='', default=None)
ap.add_argument('--model'); ap.add_argument('--service')
ap.add_argument('--scope', default='checks', choices=['primary', 'checks', 'all'])
ap.add_argument('--limit', type=int, default=0)
ap.add_argument('--out')
a = ap.parse_args()

if a.list_models is not None:
    for m in Model.available():
        s = str(m)
        if a.list_models.lower() in s.lower(): print(s)
    sys.exit()

df = rj.select(rj.read_frame(str(HERE.parents[1] / 'results/traces/trace_features.csv')), a.scope)
plans = list(df.plan.drop_duplicates())
done = set()
out = Path(a.out)
if out.exists():
    for l in out.read_text().splitlines():
        d = json.loads(l)
        if 'result' in d: done.add(d['h'])
todo = [p for p in plans if rj.plan_hash(p) not in done]
if a.limit: todo = todo[:a.limit]
print(f'{len(plans)} unique plans; {len(todo)} to judge with {a.model}', flush=True)
if not todo: sys.exit()

q = QuestionFreeText(question_name='label',
                     question_text=rj.SYS + '\n\n<PLAN>\n{{ scenario.plan }}\n</PLAN>\n\nReturn the JSON object only.')
kw = dict(temperature=0, max_tokens=1600)
m = Model(a.model, service_name=a.service, **kw) if a.service else Model(a.model, **kw)
res = q.by(ScenarioList.from_list('plan', todo)).by(m).run(progress_bar=False, print_exceptions=False)

ok = bad = 0
with out.open('a') as fh:
    for p, txt in res.select('scenario.plan', 'answer.label').to_list():
        txt = txt or ''
        rec = dict(model='edsl/' + a.model, h=rj.plan_hash(p), schema=rj.SCHEMA, timestamp=time.time())
        try:
            j = json.loads(txt[txt.index('{'): txt.rindex('}') + 1])
            rec['result'] = rj.validate(j, p); ok += 1
        except Exception as e:
            rec['error'] = f'{type(e).__name__}: {str(e)[:60]}'; bad += 1
        fh.write(json.dumps(rec) + '\n')
print(f'ok {ok}; failed {bad}  -> {out}')
