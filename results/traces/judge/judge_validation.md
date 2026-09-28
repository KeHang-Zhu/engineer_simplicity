# LLM-judge validation of the rule-based heuristic taxonomy

Sample: 2573 traces judged (of 2573 sampled; 0 unreturned). Judge: claude-haiku-4-5 subagents, plan text + mechanism family only (no bid/value/model/rule label). Stratified by corpus x model x rule-label, seed 1299.

**Pre-registered role:** judge labels are a robustness check on taxonomy shares and the unanchored autopsy; they are never the outcome variable in the causal lever battery.

## Agreement

- Overall: Cohen's kappa = **0.361**, raw agreement 0.433 (n=2573)
- frontier: kappa = 0.256, agreement 0.356 (n=1189)
- legacy: kappa = 0.437, agreement 0.499 (n=1384)
- Collapsed (H1+H2 merged; H7+T merged): kappa = **0.451**, agreement 0.547
- Coarse decision-mode families (below-value / opponent / anchor / aggressive / exit / normative): kappa = **0.549**, agreement 0.641
    - frontier: coarse kappa = 0.517, agreement 0.661
    - legacy: coarse kappa = 0.547, agreement 0.624

**Protocol consequence (kappa < 0.70 at the 10-label grain):** fine-grained taxonomy shares are reported at the coarse decision-mode grain in the main text; the 10-label split and this confusion matrix go to the appendix. The causal lever battery never uses taxonomy labels, so it is unaffected.

## Confusion matrix (rows = rule label, cols = judge)

```
judge_label         H1  H2  H3   H4   H5   H6  H7  H8    T  U
heuristic_primary                                            
H1                 153   8  32    8    5    0   9   0  110  0
H2                 116  20  33    6    7    0   9   1  142  0
H3                  62   2  64    8   18    0   0   0    6  0
H4                  27   2  23  110   10    0   0   0    0  0
H5                  21   0   6    3  166    0   6   0   53  0
H6                   1   1   0    0    0  302   0  11    0  0
H7                  11   0   1    8    2    0  80   0  148  0
H8                  13   2   1    1    7    0   4   0   66  0
T                   50   5  10   19    5    1  21   2  217  2
U                   38  13  23   28   24  171   7  16   14  2
```

## Per-label agreement (rule label as reference)

- H1: recall 0.47, precision 0.31 (n_rule=325, n_judge=492); top confusions {'T': 110, 'H3': 32}
- H2: recall 0.06, precision 0.38 (n_rule=334, n_judge=53); top confusions {'T': 142, 'H1': 116}
- H3: recall 0.40, precision 0.33 (n_rule=160, n_judge=193); top confusions {'H1': 62, 'H5': 18}
- H4: recall 0.64, precision 0.58 (n_rule=172, n_judge=191); top confusions {'H1': 27, 'H3': 23}
- H5: recall 0.65, precision 0.68 (n_rule=255, n_judge=244); top confusions {'T': 53, 'H1': 21}
- H6: recall 0.96, precision 0.64 (n_rule=315, n_judge=474); top confusions {'H8': 11, 'H2': 1}
- H7: recall 0.32, precision 0.59 (n_rule=250, n_judge=136); top confusions {'T': 148, 'H1': 11}
- H8: recall 0.00, precision 0.00 (n_rule=94, n_judge=30); top confusions {'T': 66, 'H1': 13}
- T: recall 0.65, precision 0.29 (n_rule=332, n_judge=756); top confusions {'H1': 50, 'H7': 21}
- U: recall 0.01, precision 0.50 (n_rule=336, n_judge=4); top confusions {'H6': 171, 'H1': 38}

## Safety-flag cross-check

Judge's broader 'states why truthful bidding is safe' flag (vs the frozen safety_recognition regex firing 3/21,990):

```
corpus
frontier    0.453
legacy      0.027
```

```
                                     mean  size
corpus   model                                 
frontier anthropic/claude-sonnet-5  0.579   318
         google/gemini-2.5-flash    0.260   369
         openai/gpt-5               0.538   249
         openai/gpt-5-mini          0.494   253
legacy   claude-3-5-haiku-20241022  0.054   355
         gemini-2.0-flash           0.038   313
         google/gemma-3-27b-it      0.003   328
         gpt-4o                     0.015   388
```

Reading: at the frontier the payoff-safety rationale IS articulated (the judge finds it in a large share of frontier normative traces), while legacy traces almost never state it -- consistent with the regex-based 0/600 finding for the legacy Payoff-Safety cell, and evidence that the dictionary's near-zero safety_recognition rate is a legacy-model fact, not a dictionary artifact.
