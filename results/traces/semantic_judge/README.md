# Semantic judge labels (ICLR 2027 rebuttal)

Two LLM judges from different model families labeled every unique sealed second-price plan in the baseline, Payoff Safety, Payoff Tree, worst-case, dominated, and belief cells (2,939 unique plans). Judges saw only the plan text and a fixed rubric (`analysis/semantic_judge/run_judge.py`, `SYS`), never the cell, model, value, or bid.

| File | Judge | Route | Coverage |
|---|---|---|---|
| `labels_claude-sonnet-5-5.jsonl` | Claude Sonnet 5.5 | EDSL (`run_judge_edsl.py`) | 2,939 / 2,939 |
| `labels_gpt-6.1-sol.jsonl` | GPT-6.1 Sol | EDSL (`run_judge_edsl.py`) | 2,851 / 2,939 (88 empty responses) |
| `robustness/labels_openrouter_gpt-4.1-mini.jsonl` | GPT-4.1 mini | OpenRouter (`run_judge.py`) | complete |
| `robustness/labels_openrouter_claude-haiku-4.5_partial.jsonl` | Claude Haiku 4.5 | OpenRouter | ~96% |
| `robustness/labels_openrouter_kimi-k2_partial.jsonl` | Kimi K2 | OpenRouter | partial (stopped) |

Each line: `model`, `h` (SHA-256 of the plan text), `schema` (rubric hash), and `result` (labels, direction, payment_rule, stated amounts, evidence) or `error`. Join to `results/traces/trace_features.csv` by hashing `plan`.

`python analysis/semantic_judge/summarize_judges.py` regenerates `judge_agreement.csv`, `keyword_vs_judge.csv`, `semantic_effects.csv`, and `summary.md`. Headline: the two primary judges agree on 93.6% of labels on average (mean kappa 0.78) and on 98.6% of intended bid directions.
