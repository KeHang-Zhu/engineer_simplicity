# Semantic validation of plan-level text measures (ICLR 2027 rebuttal)

Judges: claude-sonnet-5-5, gpt-6.1-sol. Plans labeled by both: 2847 of 2939 unique plans in the scored cells.
Mean agreement across 7 constructs: 93.6% (mean kappa 0.78); intended bid direction: 98.6%.

| Contrast | Outcome | claude-sonnet-5-5 | gpt-6.1-sol |
|---|---|---|---|
| Payoff Safety | safety_reason_valid | 2.5% → 7.5% (p=0.111) | 2.5% → 6.4% (p=0.111) |
| Payoff Safety | says_truthful_best | 4.1% → 8.7% (p=0.037) | 6.4% → 7.5% (p=0.148) |
| Payoff Safety | below | 82.3% → 77.2% (p=0.272) | 82.2% → 77.3% (p=0.296) |
| Payoff Tree | payment_rule_correct | 4.1% → 10.6% (p=0.111) | 5.6% → 12.4% (p=0.074) |
| Payoff Tree | below | 82.3% → 82.8% (p=0.975) | 82.2% → 82.7% (p=0.951) |
| Worst-case scaffold | worst_case | 2.0% → 46.5% (p=0.125) | 27.7% → 60.4% (p=0.125) |
| Dominated scaffold | dominance_language | 7.5% → 59.3% (p=0.125) | 6.8% → 57.6% (p=0.125) |
| First-order beliefs | opponent_modeling | 76.2% → 92.9% (p=0.250) | 82.0% → 94.2% (p=0.250) |

Bid error (abs_dev) contrasts are identical across judge files up to coverage; see semantic_effects.csv. Labels are LLM classifications of plan text, not a human reference.
