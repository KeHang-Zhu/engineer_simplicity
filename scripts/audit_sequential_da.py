"""Read existing DA logs and count value-inconsistent sequential decisions.

This check does not call a model or alter experimental results. Binary responses
and nonforced final picks are scored separately from reconstructed rankings.
"""
from pathlib import Path
import json

ROOT = Path(__file__).resolve().parents[1]
LOGS = ROOT / 'experiment_logs/da'


def audit(model):
    counts = dict(markets=0, informative=0, false_no=0, false_yes=0, bad_pick=0)
    examples = []
    for path in sorted((LOGS / model / 'osp_yesno_fixed/raw_data').glob('*.json')):
        data = json.loads(path.read_text())
        counts['markets'] += 1
        for node in data.get('osp_history', []):
            kind = node.get('type')
            values = data['values'].get(node.get('student'), {})
            error = None
            if kind in ('yes_no_a', 'yes_no_b'):
                candidate = node['candidate']
                available = set(node.get('fallback', [])) | {candidate}
                vals = {s: values[s] for s in available if s in values}
                if not vals:
                    continue
                counts['informative'] += 1
                maximum = max(vals.values())
                best = vals[candidate] == maximum
                tied = best and sum(v == maximum for v in vals.values()) > 1
                if not tied and ((node['answer'] == 'YES') != best):
                    error = 'false_no' if best else 'false_yes'
            elif kind in ('serial_dictatorship', 'final_pick_a', 'final_pick_b'):
                available = node.get('available', node.get('remaining_schools', []))
                vals = {s: values[s] for s in available if s in values}
                choice = node.get('choice')
                if len(vals) < 2 or choice not in vals:
                    continue
                counts['informative'] += 1
                if vals[choice] < max(vals.values()):
                    error = 'bad_pick'
            if error:
                counts[error] += 1
                if len(examples) < 2:
                    examples.append(dict(file=str(path.relative_to(ROOT)), student=node['student'],
                                         type=kind, error=error))
    counts['errors'] = sum(counts[k] for k in ('false_no', 'false_yes', 'bad_pick'))
    counts['examples'] = examples
    return counts


if __name__ == '__main__':
    print(json.dumps({model: audit(model) for model in ('claude', 'gemini', 'gpt4o', 'gemma')}, indent=2))
