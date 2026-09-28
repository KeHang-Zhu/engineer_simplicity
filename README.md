# Engineering Simplicity

Code and data for **Engineering Simplicity: Simple Mechanism Interfaces Steer LLM Agents**.

Can interfaces, scaffolds, and task descriptions improve LLM decisions? This project uses auctions and school matching as controlled multi-agent testbeds: their rules map agents' choices into allocations and payments, and provide explicit benchmarks for evaluating those choices. Changes to interfaces and descriptions can improve behavior without corresponding improvements in measured verbal indicators of strategic understanding.

## Reproduce the analysis without model API calls

Use Python 3.10 or later and a fresh environment. The historical `venv/` in the repository is not a portable installation.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-analysis.txt
python scripts/verify_data.py
python scripts/audit_sequential_da.py
python scripts/plot_intervention_language_bids.py
```

The last command regenerates the simplified Figure 4 as `plots/intervention_language_bids.pdf` and `.png`. It displays six selected interventions; all 20 estimates remain in `data/intervention_language_bids.csv`. See [Figure 4's source and interpretation](docs/FIGURE4.md).

For raw-log reconstruction, trace classification, matching analysis, inference, and appendix figures, follow [analysis/README.md](analysis/README.md). These are offline analyses of recorded responses; they do not rerun experiments or call an LLM judge.

## Repository guide

| Location | Contents |
| --- | --- |
| [`analysis/`](analysis/) | Trace features, intervention comparisons, heuristic labels, matching measures, and offline judge scoring |
| [`data/`](data/) | Paper-facing exports, source documentation, and SHA-256 manifest |
| [`experiment_logs/`](experiment_logs/) | Recorded auction/matching responses and run metadata; includes the added model extension |
| [`recovered_logs/`](recovered_logs/) | Recovered GPT-4o first- and third-price auction inputs used in trace analysis |
| [`robustness_logs/`](robustness_logs/) | Additional model/temperature/format inputs to the expanded trace corpus |
| [`results/traces/`](results/traces/) | Frozen 21,990-row corpus, expanded 36,591-row corpus, classifications, and archived estimates |
| [`scripts/`](scripts/) | Data verification, independent sequential-decision audit, and current trace figures |
| [`plots/`](plots/) | Figure outputs and historical auction/matching plotting scripts |
| [`Prompt/`](Prompt/), [`rule_template/`](rule_template/), [`configs_auction/`](configs_auction/), [`configs_da/`](configs_da/) | Historical prompts, rules, and experiment configurations |
| [`src/`](src/) | Experiment implementation and runners |

## Reading the results

- **Behavior and language are different outcomes.** Regex features and heuristic labels describe recorded text. They are not direct measurements of latent understanding or faithful reasoning.
- **Use the comparison specified for each intervention.** In the current trace analysis, `axis2_forward_baseline` is a two-stage clock-exit description, not a plain baseline. The corrected pooled baseline uses `axis1_contingent_baseline` and `axis3_beliefs_baseline`. Historical analyses retain their original definitions; see the [analysis guide](analysis/README.md).
- **Matching metrics have different denominators.** Full-ranking Kendall error cannot be directly compared with sequential decision error. In `osp_yesno_fixed`, retained logs contain 16/194 inconsistent decisions for Claude, 0/179 for Gemini, 0/188 for GPT-4o, and 3/210 for Gemma. All four can nevertheless have zero reconstructed Kendall error. The earlier round-based `osp_baseline` is a separate condition.
- **The primary Gemma model is Gemma 3 27B.** Folder names and historical labels are preserved; model metadata and source scopes are documented in [data/README.md](data/README.md).
- **Historical figure provenance is not always complete.** The current DA plotting scripts and the paper's historical Figure 1 differ in layout; exact reproduction of that PDF from the presently identified inputs has not been established. The decision audit above is independently reproducible.

## Running new experiments

The original entry points include `src/run_auction_batch.py`, `src/run_da_batch.py`, and the root shell runners. Consult [`configs_auction/README.md`](configs_auction/README.md) and the [historical matching implementation guide](DA_README.md) before use. Experiment execution requires provider access and additional dependencies (including EDSL); `requirements-analysis.txt` covers offline analysis only. Historical model identifiers may no longer be available. New model responses need not reproduce archived results.

The September 2026 repository update preserves historical prompts and observations, adds portable analysis and its data inputs, and documents the revised measurement interpretation. See [data provenance](data/README.md) for scope and limitations.
