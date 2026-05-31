# Human-Experiment Briefing: Four EA-Searched Second-Price Mechanisms

Prepared for human-subject implementation. Four mechanisms produced by a
2-generation evolutionary search over `second_price_ipv`-scoped genotypes,
scored on the Li-2017-calibrated 5-strain cluster panel via auto-emitted
per-mechanism SMAD modules.

**All four mechanisms are second-price IPV, $n=3$ bidders, values $U[0,49]$,
bids in $\$0.1$ increments.** The dominant strategy is constant
($b^\star(v) = v$) across all four. The only differences are in the
**rule-text framing**: whether and how strongly the description nudges
bidders toward truthful bidding. This isolates the marginal effect of
framing on truthful bidding, parallel to Li (2017)'s AC-vs-2P comparison
at the rule-text level.

The EA searched a population of 12 second-price seed genotypes (varying
bid-language $\in\{$continuous, discrete-grid, ranked-price-menu$\}$,
description style, scaffold) across 2 generations, producing 28 scored
mechanisms. We selected 2 easy (top of ranking) + 2 hard (bottom of
ranking, excluding LLM hallucinations like "2 bidders" when $n=3$).

| Code | Predicted difficulty | Weighted SMAD (Li-2017) | Worst-strain SMAD |
|------|----------------------|------------------------|-------------------|
| **E1** | easy (highest framing) | **11.0** | 34.2 |
| **E2** | easy (moderate framing) | 11.8 | 33.1 |
| **H1** | hard (no truthful nudge) | 15.3 | 34.8 |
| **H2** | hard (losing example) | 15.5 | 36.3 |

The Li-2017-calibrated mixture weights are unchanged from the previous
briefing: c0_underbidder 0.106, c2_rational 0.194, win_seeker 0.400,
loss_averse 0.106, outcome_truthful 0.194.

Decisive prediction: within second-price, the easy-cluster mechanisms (E1,
E2) should produce a higher truthful-bid rate than the hard-cluster
mechanisms (H1, H2). Worst-strain (loss-averse) SMAD is similar across all
four ($\approx 33$–$36$): the loss-averse strain shades regardless of the
framing, so a framing nudge moves it less than it moves the contingent-off
underbidder. The discriminating strain is c0_f0_b0 (CoT underbidder), whose
SMAD swings from $\approx 3$ (with truthful nudge) to $\approx 30$–$36$
(without).

---

## Common environment

- $n = 3$ bidders per auction.
- Private values drawn i.i.d. uniformly on $[\$0, \$49]$.
- Bids on a $\$0.1$ grid.
- One-shot, sealed bids.
- Tie-breaking: random.
- All four mechanisms are second-price (winner pays the second-highest bid)
  with $b^\star(v) = v$.

We recommend at least 15 rounds per subject per mechanism. With 4 mechanisms,
that's 60 rounds per subject; a within-subject design with the mechanism
identity revealed before each block is natural.

---

## E1. Highest-framing 2P (predicted easiest; weighted SMAD = 11.0)

**Rule shown to subjects:**

> You are one of 3 bidders in a sealed-bid auction. Your private value for
> the item is _v_, and you should submit one numeric bid as your action. The
> bidder with the highest bid wins the item. Because this is a second-price
> auction, the winner pays the second-highest bid, not their own bid. Bids
> are continuous and may be any number in $0.1 increments between $0 and
> $49. If your value is $20 and you bid $20, you are simply stating your true
> value; in a second-price auction, that is the standard benchmark. Submit
> only your bid number.

**Optimal bid:** $b^\star(v) = v$. Truthful bidding is weakly dominant.

**Panel signature observed.** c2 rational and outcome-truthful bid value
exactly. c0_underbidder SMAD = 3.4 (the value-anchor sentence pulls the
underbidder very close to truthful). win_seeker SMAD = 17.6 (still overbids
~20% above value, framing doesn't reach the win-seeker). loss_averse SMAD =
34.2 (largest residual; the affective shading is robust to rule-text nudges).

---

## E2. Moderate-framing 2P (predicted easy; weighted SMAD = 11.8)

**Rule shown to subjects:**

> You are one of 3 bidders in a sealed-bid auction. Your private value for
> the item is _v_, and you must submit one numeric bid as your action. The
> highest bid wins; if there is a tie, the winner is chosen randomly. The
> winner pays the second-highest bid, not their own bid. Example: if your
> value is $20 and you bid $20, and the other bids are $14 and $9, you win
> and pay $14. Please bid as close as possible to your true value.

**Optimal bid:** $b^\star(v) = v$.

**Panel signature observed.** Almost identical to E1: c0_underbidder SMAD =
2.1 (slightly tighter than E1), c2 = 0, outcome_truthful = 0, win_seeker =
20.1, loss_averse = 33.1. The "bid as close as possible to your true value"
phrasing is a softer nudge than E1's "that is the standard benchmark", which
explains the small E1 < E2 gap.

---

## H1. No-nudge 2P with explicit example (predicted hard; weighted SMAD = 15.3)

**Rule shown to subjects:**

> You are one of 3 bidders in a single sealed-bid auction. Your private value
> is _v_, and you must submit one numeric bid as your action. The highest bid
> wins the item, and the winner pays the second-highest bid; if there is a
> tie for the highest bid, the winner is chosen randomly. Your bid may be any
> continuous number consistent with the auction format, in $0.1 increments
> between $0 and $49. You will not receive feedback during the auction.
> Example: if your value is $20 and the highest other bid is $17, then
> bidding $20 can win and, if you win, you pay $17.

**Optimal bid:** $b^\star(v) = v$. Same as E1 / E2.

**Panel signature observed.** c0_underbidder SMAD jumps to 30.8 (no nudge to
the contingent-off mouse → it stays at margin-of-safety underbid). The
example shows a winning truthful bid but doesn't explicitly endorse it. Other
strains: c2 = 0, outcome_truthful = 0, win_seeker = 20.7, loss_averse = 34.8.

---

## H2. No-nudge 2P with losing example (predicted hardest; weighted SMAD = 15.5)

**Rule shown to subjects:**

> You are participating in a sealed-bid auction with 3 bidders for a single
> item. Your private value for the item is _v_, and you should submit one
> numeric bid as your action. The bidder with the highest bid wins the item.
> The winner pays the second-highest bid, not their own bid. If there is a
> tie for the highest bid, the winner is chosen randomly among the tied
> bidders. Example: if you bid $12, another bidder bids $15, and the third
> bidder bids $9, then the bidder who bid $15 wins and pays $12.

**Optimal bid:** $b^\star(v) = v$. Same as E1 / E2.

**Panel signature observed.** c0_underbidder SMAD = 36.3 (highest of the
four — the losing-bidder example *implicitly anchors* the reader on a
sub-value bid). c2 = 0, outcome_truthful = 0, win_seeker = 19.7, loss_averse
= 35.4. The losing-bidder example moves the contingent-off bidder slightly
further from truthful than H1's winning example does.

---

## What to report back

For each subject and each mechanism:
1. Round-level $(v, b)$ pairs (private value $v$, submitted bid $b$).
2. Order of mechanisms presented (E1 / E2 / H1 / H2) per subject.
3. Demographics and any experience covariates.

We will use those to:
1. Compute the realized per-subject SMAD using $b^\star(v) = v$ for all four
   mechanisms.
2. Compare the human ranking on weighted SMAD against the prediction above.
3. Refit Li-2017 mixture weights against the new human population if needed.
4. Identify subjects whose framing-sensitivity is large (E1 SMAD $<<$ H2 SMAD)
   versus framing-insensitive subjects.

The decisive prediction is **within-subject** rather than between-mechanism:
each subject should have lower SMAD on E1 / E2 than on H1 / H2 if framing
matters in the same way for humans as for the digital-mouse panel.
