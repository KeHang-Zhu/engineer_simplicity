# Data sources and units

This repository contains recorded outputs of synthetic auction and matching experiments with LLM agents, the prompts/configurations used to generate them, and derived analysis tables. It does not contain newly collected human-subject observations. Provider-generated plan text is an observed response, not access to a model's internal reasoning.

## Source-to-analysis map

| Source | Unit and scope | Consumer / derived output |
| --- | --- | --- |
| `results/all_experiments_combined_20260204_114522.csv` | 13,416 bidder-round records, four primary model families; values, bids, plans, and run identifiers | `analysis/build_trace_features.py` → frozen trace corpus |
| `experiment_logs/{claude,gemini,gemma,gpt4o}/intervention_{menu,proxy_breitmoser}/result_*.json` | 1,167 bidder-round records missing from that combined export | Same frozen corpus; `proxy_breitmoser` is a description intervention, not an implemented ascending clock |
| `recovered_logs/experiment_logs_gpt_4o/V12/*_{first,third}/run_*/raw_data/result_*.json` | 7,407 GPT-4o bidder-round records in first- and third-price mechanisms | Frozen corpus's cross-mechanism comparison; unsuffixed second-price runs are excluded |
| `experiment_logs/{gpt5,gpt5mini,claude_sonnet5,gemini25flash}/` | 5,985 additional bidder-round records, with saved run configurations | `analysis/extend_trace_corpus.py` → expanded corpus (`corpus=frontier`) |
| `robustness_logs/` | 8,616 additional bidder-round records with model, temperature, and format metadata | Expanded corpus (`corpus=robustness`); duplicate `V10/` subtree excluded |
| `experiment_logs/da/` and `experiment_logs/da_cardinal/` | One JSON per retained matching market; school values, rankings or sequential actions | `analysis/build_da_cells.py` → market-cluster summaries; independent binary-query audit |
| `results/traces/judge/judge_sample.csv` and `workflow_result.json` | Fixed sample of 2,573 traces and archived LLM judge responses | `analysis/judge_score.py` → agreement checks; no new API execution |

The primary model IDs are `gpt-4o`, `claude-3-5-haiku-20241022`, `gemini-2.0-flash`, and `google/gemma-3-27b-it`. Directory names are aliases. Extension IDs are retained as recorded in run configurations, including provider prefixes; these are historical metadata rather than independent verification of provider routing.

## Released tables

- `results/traces/trace_features.csv`: frozen 21,990-row corpus, one bidder-round trace per row, with 18 binary language features, value/bid/deviation, model, experiment, mechanism, and run IDs.
- `results/traces/trace_features_v2.csv`: 36,591 rows with corpus membership, auction IDs, configuration metadata, additive formal-language features, and clock-language indicator. In clock logs, bids are aligned by agent name rather than exit order; the winner's recorded price is censored at the runner-up's exit and is flagged.
- `results/traces/heuristic_labels.csv`: descriptive rule-based classifications aligned with the expanded corpus. Labels are not ground-truth comprehension measurements.
- `results/traces/mediation/`: intervention-level estimates and sensitivity analyses. Historical filenames are kept; `*_precorrection.csv` and `*_legacy` columns encode earlier baseline definitions and are not the revised comparison.
- `intervention_language_bids.csv`: unchanged 20-intervention paper export, identical to the archived `results/traces/mediation/dissociation_full_battery.csv`. `prev_diff` is a proportion difference (multiply by 100 for percentage points); `absdev_diff` and its intervals are changes in absolute bid error in experimental dollar units. Figure 4 displays six selected rows. See [its documentation](../docs/FIGURE4.md).
- `da_interface_error_accounting.csv`: eight rows (four models × two sequential interfaces), extracted from the matching analysis. `n_markets`, `n_decisions_informative`, and `n_misreports` have different units; zero Kendall distance does not imply zero misreports.

## Provenance and integrity

The September 2026 update ports the trace-analysis scripts and existing inputs/exports from the authors' local auction-analysis workspace. It retains the repository's pre-existing experimental observations and supplements them with the first/third-price, extension, and robustness inputs needed by those scripts. No new model responses or experimental observations were generated for this update. Existing prompts are historical records and are not silently corrected to match later prose.

`source_manifest.json` lists the released source/derived data files, their sizes, and SHA-256 hashes using repository-relative paths. Verify it with `python scripts/verify_data.py`. It provides byte-level identity for this release, not independent certification of how providers generated each observation. `results/traces/corpus_manifest.json` separately records feature-dictionary fingerprints and frozen feature sums. The [analysis guide](../analysis/README.md) explains reconstruction and numerical validation.

The corpus uses the combined export for its primary grid rather than re-exporting that grid from every raw JSON. Supplemental sources are selected explicitly to avoid double-counting those records. Raw logs can contain missing/failed calls, malformed records, repeated text, and unequal cell sizes; the loaders' retained counts and coverage tables describe the analyzed sample. Deduplicated sensitivity estimates are provided separately. Do not pool datasets solely because their experiment names look alike.

## Interpretation limits

Full-ranking errors, errors over partial revealed rankings, and errors over informative sequential decisions are not interchangeable. The `osp_baseline` round-based interface, `osp_yesno_fixed` binary interface, and older `osp_yesno` derived-pick records are separate conditions. Equal-value ties are treated explicitly in the decision audit. Dropped or unavailable calls are not evidence of correct behavior.

The original Figure 1 PDF's exact input-to-render provenance is not established: the available plotting code differs in layout and condition selection. The current sequential-decision counts have been independently checked from raw logs. The Figure 4 export has exact source correspondence and a standalone renderer.

Prompted terminology can mechanically affect language features. Text indicators and judge agreement do not establish latent understanding, and observational mediation is not identification of a cognitive mechanism. See the analysis guide for baseline corrections, clustering, and multiplicity qualifications.
