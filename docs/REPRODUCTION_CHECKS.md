# Release verification — 2026-09-28

The repository update was checked using the package versions in `requirements-analysis-tested.txt` (Python 3.9.6 on macOS). The setup guide recommends Python 3.10 or later for new installations.

Completed checks:

- Rebuilt the frozen corpus from the combined export, supplementary JSONs, and recovered first/third-price logs: 21,990 records, with source counts 13,416 + 1,167 + 7,407.
- Rebuilt the expanded corpus from raw inputs: 36,591 records, comprising 21,990 legacy, 5,985 extension, and 8,616 robustness records. The loader's legacy identifiers and all 18 frozen features matched.
- Recomputed heuristic classifications. Frozen corpus, expanded corpus, and labels all matched the pre-existing archived tables numerically and structurally. The original CSV bytes were retained for these three released snapshots.
- Independently recalculated all 20 intervention point estimates and sample sizes against their specified comparison groups, matching the paper export to its stored rounding precision.
- Rebuilt the matching summary (254 rows) and checked its paper extract. Independently verified sequential binary-query decisions: Claude 16/194, Gemini 0/179, GPT-4o 0/188, Gemma 3/210.
- Rescored the fixed 2,573-trace LLM-judge sample using archived responses; no judge API calls were made.
- Regenerated Figure 4 and the two trace appendix figures; visually inspected the simplified Figure 4 PNG.
- Compiled all added Python analysis/utility scripts, checked authored documentation links, and scanned added text/configuration files for credential patterns.
- Generated and verified a SHA-256 manifest for the released data, prompt, and configuration files.

Scope: the full bootstrap/permutation inference pipelines are supplied with their archived outputs, but were not rerun end to end for this repository update. Point estimates and data alignment were verified independently; the checks above should not be read as a new validation of every archived p-value or interval. Exact provenance of the historical Figure 1 PDF remains unresolved and is documented in the main README.
