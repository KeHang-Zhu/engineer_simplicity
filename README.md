# Engineering Simplicity: Simple Mechanism Interfaces Steer LLM Agents

[![arXiv](https://img.shields.io/badge/arXiv-2609.36365-b31b1b.svg)](https://arxiv.org/abs/2609.36365)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Code and data for the paper by Kehang Zhu\*, Anand V. Shah\*, and David C. Parkes (\*equal contribution).

We use second-price auctions and student-proposing deferred acceptance (DA) as testbeds for LLM agents, because each has a known optimal strategy (truthful play) against which decisions can be scored. Across four model families, the ascending clock interface lowers bid deviations, and describing payoff contingencies or why truth-telling is safe improves choices, while prompts to plan ahead or to model opponents make play worse. In auctions, the descriptions that improve bids do not raise verbal indicators of strategic understanding in the agents' stated plans.

## Repository layout

```
├── src/                  Experiment code: auction and DA mechanisms, LLM agents, batch runners
│   └── orchestrator/     Earlier config-driven runner with parallel task control
├── configs/
│   ├── auction/          One YAML file per auction run (model × intervention)
│   └── da/               One YAML file per matching run, grouped by model
├── prompts/
│   ├── templates/        Agent prompt templates
│   └── rules/            Mechanism rule descriptions shown to agents (auctions/, DA/)
├── data/
│   ├── raw/              Recorded model responses, one folder per run (read-only)
│   │   ├── experiment_logs/    Main auction and matching experiments, plus the frontier-model extension
│   │   ├── recovered_logs/     Recovered GPT-4o first- and third-price auction runs
│   │   └── robustness_logs/    Model × temperature × format robustness grid
│   ├── README.md         Source-to-analysis map, units, and provenance
│   └── source_manifest.json    SHA-256 hash of every released data file
├── analysis/             Offline analysis of recorded responses (no API calls)
│   └── plotting/         Figure scripts; all write to figures/
├── results/
│   ├── paper/            Tables exported for the paper (Figure 4, DA error accounting)
│   ├── traces/           Stated-plan corpus, features, heuristic labels, intervention estimates
│   └── merged_ranking/   Matching summaries by model and condition
├── figures/              Generated figures
├── scripts/              Data verification, DA decision audit, experiment launchers
├── tests/                Manual checks of the DA implementation (require EDSL and API access)
└── docs/                 Figure 4 notes, intervention taxonomy, matching implementation guide
```

## Reproducing the analysis

The analysis runs offline on the recorded responses and needs Python 3.10 or later. Run all commands from the repository root.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements-analysis.txt
```

`requirements-analysis-tested.txt` pins the exact versions used for the release checks.

Verify the data and the main estimates:

```bash
python scripts/verify_data.py        # checks all 26,592 data files against data/source_manifest.json
python scripts/audit_sequential_da.py   # independent audit of the sequential DA decisions
python scripts/verify_analysis.py    # checks corpus counts and all 20 intervention point estimates
```

Regenerate Figure 4:

```bash
python analysis/plotting/plot_intervention_language_bids.py
```

The figure shows six selected interventions; all 20 estimates are in `results/paper/intervention_language_bids.csv`. See [docs/FIGURE4.md](docs/FIGURE4.md).

[analysis/README.md](analysis/README.md) gives the full pipeline: rebuilding the stated-plan corpus from the raw logs, classifying heuristics, the bootstrap and permutation inference for the intervention table, the matching measures, rescoring the archived LLM-judge labels, and the appendix figures.

## Running new experiments

Running experiments requires API access to the model providers and [EDSL](https://github.com/expectedparrot/edsl), which `requirements-analysis.txt` does not include. Each run is defined by a YAML file in `configs/` and writes its responses, together with a copy of its configuration and prompts, to a new folder under `data/raw/experiment_logs/`.

```bash
python src/run_auction_batch.py configs/auction/interventions_gpt4o/axis1_contingent_tree.yaml
bash scripts/run_auction_experiments.sh claude all
python src/run_da_batch.py configs/da/gpt4o/da_direct_baseline.yaml
```

See [configs/auction/README.md](configs/auction/README.md) for the configuration schema and [docs/matching_experiments.md](docs/matching_experiments.md) for the DA implementation. Some historical model identifiers are no longer served, and new responses from the same model need not reproduce the archived ones.

## Notes on interpreting the results

- **Bids and stated plans are separate outcomes.** The language features and heuristic labels describe the text of each plan. They do not measure the model's internal understanding.
- **Each intervention has its own comparison.** `axis2_forward_baseline` describes a two-stage clock exit and is analyzed as a treatment. Payoff Safety and Payoff Tree are compared with the pooled `axis1_contingent_baseline` and `axis3_beliefs_baseline`. The mapping is `LEVERS` in `analysis/build_trace_mediation.py`.
- **Matching metrics have different denominators.** Kendall distance over a full ranking is not comparable to the error rate over sequential yes/no decisions. In `osp_yesno_fixed`, the retained logs contain 16 inconsistent decisions out of 194 for Claude, 0/179 for Gemini, 0/188 for GPT-4o, and 3/210 for Gemma, although all four have zero Kendall distance. The round-based `osp_baseline` is a different condition.
- **The Gemma model is Gemma 3 27B.** Folder names such as `gemma` are historical; [data/README.md](data/README.md) lists the exact model identifiers.
- **Figure 1's exact provenance is not established.** `analysis/plotting/plot_osp_comparison.py` produces a figure with a different layout from the paper's Figure 1. The DA decision counts are independently reproducible with `scripts/audit_sequential_da.py`.

## Citation

```bibtex
@article{zhu2026engineering,
  title   = {Engineering Simplicity: Simple Mechanism Interfaces Steer {LLM} Agents},
  author  = {Zhu, Kehang and Shah, Anand V. and Parkes, David C.},
  journal = {arXiv preprint arXiv:2609.36365},
  year    = {2026}
}
```

## License

Released under the [MIT License](LICENSE).
