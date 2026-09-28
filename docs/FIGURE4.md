# Figure 4 simplification, 2026-09-25

Output: `plots/intervention_language_bids.pdf` and a PNG preview with the same stem.
Reproduce with `python3 scripts/plot_intervention_language_bids.py`.
The script resolves input and output paths relative to itself, so it also works
when launched from a different directory. Requires Python and Matplotlib.

## Design

Two aligned panels replace the original crowded scatterplot and side strip.
The six displayed rows are the four designed comparisons (Payoff Safety,
Payoff Tree, worst-case scaffold, menu restatement) and two exploratory
comparisons illustrating jointly increased target language and bid error
(first-order beliefs, risk-averse persona). This is a selected headline display,
not a display of all 20 interventions. The manuscript's full appendix table is
retained as the comprehensive result.

The left panel shows changes in the prevalence of each intervention's targeted
language feature. It does not equate more target language with better reasoning.
The menu's language contrast is omitted because its prompt removes the measured
rule vocabulary. The right panel shows treatment-minus-baseline change in mean
absolute bid error. Both panels use the same mark for all estimates; neither
color nor shape assigns statistical significance or comprehension.

The visual is 7.0 by 3.15 inches. Include at `width=\textwidth` in both versions;
the previous ICLR width of 0.74 textwidth would unnecessarily shrink the labels.
No overall title is embedded: the caption supplies context.

## Exact source and reproducibility

The local input `data/intervention_language_bids.csv` is a byte-for-byte copy of:

`results/traces/mediation/dissociation_full_battery.csv`

SHA-256 of both files:
`0550099e0aeb310f6245cd7099945c6be04c3a80e8da1a016be6d6f2d79a81d5`

The prior plotting code was:
the historical scatterplot script; the current renderer is
`scripts/plot_intervention_language_bids.py`

The analysis code documenting the intervals and classification is:
`analysis/build_trace_mediation_v2.py`

All plotted numbers come directly from the same CSV fields used for the original
figure: `prev_diff * 100`, `absdev_diff`, `absdev_ci_lo`, `absdev_ci_hi`.
No estimates were digitized from the PDF, recomputed, or inferred from rounded
manuscript values. The historical estimates remain unchanged.

## Required caption/method qualifications

- Four model families pooled, sealed second-price conditions; effects are
  treatment minus the comparison's appropriate baseline.
- Left: point estimates only. The source export has no language confidence
  intervals. Right: original percentile run-cluster-bootstrap 95% confidence
  intervals.
- These descriptive bootstrap intervals are not confidence intervals obtained
  by inverting the wild-cluster-bootstrap tests used for classification. Do not
  classify significance by whether these whiskers cross zero. For example,
  Safety's original interval crosses zero while its wild-bootstrap p is .0145.
- Designed comparisons use raw wild-cluster-bootstrap p values; exploratory
  comparisons use Benjamini-Hochberg q values, adjusted separately over all
  language tests and all bid tests. The six displayed rows do not define a new
  multiplicity family. Full-set q for each of the two designed improvers is
  .068. The figure deliberately has no significance stars or class colors.
- The gains in worst-case, first-order-belief, and risk-averse language may partly
  repeat wording supplied by the prompt. Menu language is not comparable because
  the prompt removes the measured rule terminology.
- Bid results for Safety, Tree, first-order beliefs, and risk-averse persona are
  not significant under the four-model-cluster permutation check; the original
  data carry `soft_bids=True` for these four rows. This qualification can be
  stated in the surrounding text or cross-referenced to the table/appendix.

Suggested compact caption (expand qualifications in text/table as needed):

> Language and bid responses to six headline interventions (four model families,
> sealed second-price conditions). Effects are treatment minus baseline. Left:
> change in each intervention's targeted language feature; point estimates only.
> Right: change in mean absolute bid error with run-cluster-bootstrap 95%
> intervals; negative values indicate smaller errors. The menu removes the
> measured rule vocabulary, so its language contrast is omitted. Designed raw-p
> and exploratory FDR-adjusted tests are reported in Table [headline], and all
> twenty interventions in Table [appendix].

## Verification

Generated the PDF with the checked-in plotting script, then rendered the actual
PDF with Poppler and visually inspected all labels, marks, axes, and intervals.
The source-copy hash matches the original. The script validates intervention
names, comparison tiers, and interval ordering before plotting.

An unrelated manuscript rounding inconsistency became visible during this
check: first-order beliefs' treated mean absolute error is stored as 3.925.
The old headline table printed 3.93 but the appendix printed 3.92. Harmonize this
in the manuscript; do not infer more precision than the stored analysis export.
