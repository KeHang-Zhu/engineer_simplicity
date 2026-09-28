# Analysis and reproduction

All commands below run from the repository root, using `requirements-analysis.txt`. Exact package versions used for the release checks are recorded in `requirements-analysis-tested.txt`. Paths are resolved relative to the scripts. Archived outputs are included so readers can inspect estimates without rerunning bootstrap inference.

## 1. Verify the released data and reconstruct trace features

```bash
python scripts/verify_data.py
python analysis/build_trace_features.py --corpus-only
python analysis/extend_trace_corpus.py
python analysis/classify_heuristics.py
python scripts/verify_analysis.py
```

The frozen corpus contains 21,990 bidder-round records: 13,416 from the combined experiment CSV, 1,167 from menu/clock-framing JSONs, and 7,407 from recovered first/third-price JSONs. The extension adds 5,985 model-extension records and 8,616 robustness records, for 36,591 total. It asserts that the reloaded legacy records match the frozen features before writing the expanded corpus. The original 18 regex features remain unchanged; five additional formal-language features and a clock-language indicator are additive. Dictionary fingerprints are in `results/traces/corpus_manifest.json`.

`--corpus-only` skips the historical regression report. Running `build_trace_features.py` without it additionally reproduces that early report, whose pooled baseline includes all three axes. **Use the corrected comparison pipeline below for the revised paper.** Rebuilding CSVs can change their byte representation with library versions; use `verify_analysis.py` for numerical and structural checks after rebuilding, and `verify_data.py` for byte-level checks on the released snapshot.

## 2. Reproduce intervention inference

```bash
python analysis/build_trace_mediation.py
python analysis/build_trace_mediation_v2.py
python analysis/frontier_traces.py
```

These commands can take substantially longer than corpus construction. They use seed 1299, 2,000 bootstrap draws, and 5,000 permutation draws where specified. Outputs go to `results/traces/mediation/` and `results/traces/`. The first step produces manipulation checks; the second verifies those checks and computes the full 20-intervention table, deduplication sensitivity, model-specific estimates, equivalence tests, and mediation summaries. Historical output filenames are preserved for compatibility.

The authoritative intervention mapping is `LEVERS` in `build_trace_mediation.py`. Payoff Safety and Payoff Tree use the corrected pooled axis-1/axis-3 baseline. The old axis-2 baseline contains a two-stage clock-exit description and is analyzed as a treatment. Some other interventions use their own axis-specific baselines. An intervention's language feature is its targeted verbal indicator, not a common accuracy score. Supplied prompt vocabulary can drive feature changes.

The main comparison uses run-cluster wild-bootstrap tests. Exploratory results use Benjamini–Hochberg adjustments over all 20 language tests and separately over all 20 bid tests. The four primary comparisons are distinguished from exploratory contrasts; their raw tests do not establish significance after full-family adjustment. Ordinary percentile bootstrap intervals shown in Figure 4 are not inversions of the wild-bootstrap tests. Four-model-cluster permutation checks qualify several bid results. Per-model auction-level analyses assume independent auction calls within a fixed prompt and do not identify run-level variation.

`frontier_traces.py` also writes supplemental contrasts not all used in the paper. Folder/model identifiers are historical metadata, not a claim about currently available models. The heuristic taxonomy is descriptive. Agreement with archived LLM judge labels is about 0.55 at the coarse decision-mode level and 0.36 for the ten-label taxonomy; it is not human ground truth. To rescore the existing responses, without calling a model:

```bash
python analysis/judge_score.py results/traces/judge/workflow_result.json
```

The archived judge sample and responses are included. This command scores the fixed sample; it does not reproduce the original judge API execution.

## 3. Reproduce matching measures

```bash
python scripts/audit_sequential_da.py
python analysis/build_da_cells.py
```

The first is an independent, standard-library-only audit of `osp_yesno_fixed`. It scores YES/NO responses and nonforced final picks against values, treating tied best options as value-consistent. The second produces `results/merged_ranking/da_cells.csv` and a detailed summary for ordinal and cardinal variants, including rank errors, tie-robust errors, retained market counts, parse failures, decision errors, and bootstrap intervals. The eight-row paper extract is `data/da_interface_error_accounting.csv`.

| Model | Retained markets | Informative decisions | False NO | False YES | Inferior final pick | Total errors |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Claude 3.5 Haiku | 36 | 194 | 13 | 2 | 1 | 16 |
| Gemini 2.0 Flash | 37 | 179 | 0 | 0 | 0 | 0 |
| GPT-4o | 39 | 188 | 0 | 0 | 0 | 0 |
| Gemma 3 27B | 43 | 210 | 3 | 0 | 0 | 3 |

These denominators exclude forced final picks and describe retained logs, not every attempted call. Partial sequential records can contain no comparable ranking pairs, producing zero Kendall error despite decision errors. The round-based `osp_baseline`, binary `osp_yesno_fixed`, and derived-pick `osp_yesno` must remain distinct. Historical prompts are preserved as experimental inputs, including wording later identified as too strong.

## 4. Reproduce current trace figures

```bash
python scripts/plot_intervention_language_bids.py
python scripts/plots/v4_heuristic_prevalence.py
python scripts/plots/v4_fidelity_selection.py
```

Outputs are under `plots/`. Figure 4 reads the unchanged paper export in `data/`, so a new inference run does not silently replace the paper figure's inputs. Compare any newly generated estimates to that export before explicitly refreshing it. See [Figure 4 documentation](../docs/FIGURE4.md).

Historical auction and DA figure scripts remain in `plots/`; they predate the current trace analysis and may use earlier comparison conventions. Exact provenance of the paper's historical Figure 1 PDF remains unresolved. Do not infer that running a similarly named script recreates that exact PDF.
