#!/usr/bin/env python3
"""Read-only checks of corpus definitions, paper estimates, and DA decisions.

This verifies point estimates and data alignment, not bootstrap p-values.
"""
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'analysis'))
from build_trace_features import FEATURES, FEATURE_COLS
from extend_trace_corpus import FEATURES_V2, assert_legacy_equals_frozen
from build_trace_mediation import LEVERS, CLOCK_EXIT_RE
from audit_sequential_da import audit


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def main():
    trace_dir = ROOT / 'results/traces'
    frozen = pd.read_csv(trace_dir / 'trace_features.csv', low_memory=False)
    expanded = pd.read_csv(trace_dir / 'trace_features_v2.csv', low_memory=False)
    require(len(frozen) == 21990, 'Frozen corpus row count changed')
    require(expanded.groupby('corpus').size().to_dict() ==
            {'legacy': 21990, 'frontier': 5985, 'robustness': 8616},
            'Expanded corpus coverage changed')
    assert_legacy_equals_frozen(expanded[expanded.corpus == 'legacy'].reset_index(drop=True), frozen)
    manifest = json.loads((trace_dir / 'corpus_manifest.json').read_text())
    for key, features in [('features_sha256', FEATURES), ('features_v2_sha256', FEATURES_V2)]:
        actual = hashlib.sha256(repr(sorted(features.items())).encode()).hexdigest()
        require(actual == manifest[key], f'{key} changed')
    require({c: int(frozen[c].sum()) for c in FEATURE_COLS} == manifest['frozen_feature_sums'],
            'Frozen feature counts changed')
    labels = pd.read_csv(trace_dir / 'heuristic_labels.csv', low_memory=False)
    require(len(labels) == len(expanded), 'Label count differs from corpus')
    for column in ('model', 'experiment', 'plan'):
        if column in labels:
            pd.testing.assert_series_equal(labels[column], expanded[column], check_names=False)

    paper_path = ROOT / 'data/intervention_language_bids.csv'
    paper = pd.read_csv(paper_path).set_index('lever')
    require(len(paper) == len(LEVERS) == 20, 'Intervention coverage changed')
    archived = pd.read_csv(trace_dir / 'mediation/dissociation_full_battery.csv').set_index('lever')
    pd.testing.assert_frame_equal(paper, archived, atol=1e-10, rtol=1e-10)
    sp = frozen[frozen.mechanism == 'spsb_sealed'].copy()
    sp['clock_exit_language'] = sp.plan.str.lower().str.contains(CLOCK_EXIT_RE, regex=True).astype(int)
    for name, family, experiment, feature, baselines, note in LEVERS:
        treated = sp[sp.experiment == experiment]
        control = sp[sp.experiment.isin(baselines)]
        row = paper.loc[name]
        require(len(treated) == row.n_treat and len(control) == row.n_ctrl,
                f'{name}: sample count mismatch')
        checks = {'prev_diff': (treated[feature].mean() - control[feature].mean(), 0.000051),
                  'absdev_diff': (treated.abs_dev.mean() - control.abs_dev.mean(), 0.00051)}
        for field, (actual, tolerance) in checks.items():
            require(np.isclose(actual, row[field], atol=tolerance, rtol=0),
                    f'{name}: {field} differs from rounded paper estimate')

    expected = {'claude': (36,194,13,2,1), 'gemini': (37,179,0,0,0),
                'gpt4o': (39,188,0,0,0), 'gemma': (43,210,3,0,0)}
    extract = pd.read_csv(ROOT / 'data/da_interface_error_accounting.csv')
    for model, counts in expected.items():
        result = audit(model)
        actual = tuple(result[k] for k in ('markets','informative','false_no','false_yes','bad_pick'))
        require(actual == counts, f'{model}: sequential decision counts changed')
        row = extract[(extract.model_folder == model) & (extract.condition == 'osp_yesno_fixed')].iloc[0]
        require(row.n_markets == result['markets'] and row.n_decisions_informative == result['informative']
                and row.n_misreports == result['errors'], f'{model}: paper extract differs from raw audit')
    cells = ROOT / 'results/merged_ranking/da_cells.csv'
    if cells.exists():
        all_cells = pd.read_csv(cells)
        keys = ['variant', 'model_folder', 'condition']
        actual = all_cells.set_index(keys).loc[pd.MultiIndex.from_frame(extract[keys])]
        for column in ['n_markets','n_students','n_decisions_informative','n_misreports','mean_tau_pct']:
            np.testing.assert_allclose(actual[column], extract[column], rtol=1e-6, equal_nan=True)
    print('Verified corpus counts/features, label alignment, all 20 paper point estimates, and DA decision counts.')


if __name__ == '__main__':
    main()
