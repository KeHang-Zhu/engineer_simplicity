# Human-Experiment Briefing: Four New Auction Mechanisms

Prepared for human-subject implementation. Each mechanism has been pre-screened
on the Li-2017-calibrated 5-strain digital-mouse cluster panel using a SMAD
fitness with an auto-emitted per-mechanism scoring module. The predicted
ranking is **two "easy" mechanisms (low SMAD) and two "hard" mechanisms (high
SMAD)**, with the gap between the two clusters of roughly 4× in weighted SMAD.

| Code | Mechanism | Predicted difficulty | Weighted SMAD (Li-2017) |
|------|-----------|----------------------|------------------------|
| **E1** | Vickrey 2nd-price baseline | easy | 14.1 |
| **E2** | Vickrey + OSP-style hint | easy | 13.4 |
| **H1** | All-pay first-price | hard | 107.6 |
| **H2** | First-price with $15 reserve | hard | 58.5 |

The fitted mixture weights over the cluster panel are:
- c0 underbidder (CoT): 0.174
- c2 rational (CoT): 0.190
- second-price overgeneralizer (error mouse): 0.413
- payment-panic (error mouse): 0.033
- outcome-instructed truthful: 0.190

The weights were fitted by simplex-constrained least squares against the
five-moment human SPSB target vector (p_truthful 0.39, p_underbid 0.21,
p_overbid 0.39, p_extreme_overbid 0.22, p_near_zero 0.03) read off
Li (2017) Figure 2 SP. Total L2 residual was 0.11.

---

## Common environment

- $n = 3$ bidders per auction.
- Private values drawn i.i.d. uniformly on $[\$0, \$49]$.
- Bids on a $\$0.1$ grid (or whatever grid your platform supports).
- One-shot, sealed bids.
- After each auction, all bids are revealed and the winner / price displayed.
- Ties for the highest bid are broken at random.

We recommend at least 15 rounds per subject so that mechanism-level summary
statistics (truthful rate, mean ratio, near-zero rate) are stable.

---

## E1. Vickrey baseline (predicted easy; weighted SMAD = 14.1)

**Rule shown to subjects:**

> In this game, you will participate in an auction for a prize against 2
> other bidders. At the start of each round, you will see your value for the
> prize, randomly drawn between $0 and $49, with all values equally likely.
> After learning your value, you will submit a bid privately at the same time
> as the other bidders. Bids must be in $0.1 increments.
> The highest bidder wins the prize and pays the second-highest bid. If you
> win, your earnings will increase by your value for the prize, and decrease
> by the second-highest bid. If you don't win, your earnings will remain
> unchanged.
> After each auction, we will display all bids. Ties for the highest bid will
> be resolved randomly.

**Optimal bid:** $b^\star(v) = v$. Truthful bidding is weakly dominant.

**Panel signature observed.** c2 rational and outcome-truthful both hit $b^\star$
exactly. c0 underbids by $\approx 30\%$ (SMAD 26.5), as expected for a
margin-of-safety mouse that does not invoke dominance.

---

## E2. Vickrey + OSP-style hint (predicted easiest; weighted SMAD = 13.4)

**Rule shown to subjects** (one extra sentence relative to E1, italicized):

> In this game, you will participate in an auction for a prize against 2
> other bidders. At the start of each round, you will see your value for the
> prize, randomly drawn between $0 and $49, with all values equally likely.
> After learning your value, you will submit a bid privately at the same time
> as the other bidders. Bids must be in $0.1 increments.
> The highest bidder wins the prize and pays the second-highest bid. If you
> win, your earnings will increase by your value for the prize, and decrease
> by the second-highest bid. If you don't win, your earnings will remain
> unchanged.
> _**Important: in this auction, your own bid affects ONLY whether you win the
> prize. The price you pay if you win is set by the second-highest bid, which
> is somebody else's bid, not your own.**_
> After each auction, we will display all bids. Ties for the highest bid will
> be resolved randomly.

**Optimal bid:** $b^\star(v) = v$. Same as E1.

**Panel signature observed.** The added sentence reduces c0's SMAD from $26.5$
(E1) to $21.6$ (E2), a $19\%$ improvement on the c=0 stressor. The other four
strains are essentially unchanged. The hypothesis to test on humans is whether
the same effect holds in the human population, particularly among subjects who
do not spontaneously derive the dominance argument.

---

## H1. All-pay first-price (predicted hardest; weighted SMAD = 107.6)

**Rule shown to subjects:**

> In this game, you will participate in an auction for a prize against 2
> other bidders. At the start of each round, you will see your value for the
> prize, randomly drawn between $0 and $49, with all values equally likely.
> After learning your value, you will submit a bid privately at the same time
> as the other bidders. Bids must be in $0.1 increments.
> **All bidders pay their own bid, win or lose.** The highest bidder receives
> the prize. If you win, your earnings change by (your value for the prize
> minus your own bid); if you do not win, your earnings decrease by your own
> bid.
> After each auction, we will display all bids. Ties for the highest bid will
> be resolved randomly.

**Optimal bid:** Symmetric risk-neutral BNE for $n$ bidders and $U[0, V_{\max}]$
values is $b^\star(v) = v^n / V_{\max}^{n-1}$.
For $n=3$, $V_{\max}=49$: $b^\star(v) = v^3 / 2401$. At $v=30$, $b^\star\approx
\$11.2$. At $v=49$, $b^\star\approx\$49$.

**Panel signature observed.** All five strains' SMAD is high (mean $98$); even
c2 rational sits at $92.6$ because the LLM tends to shade much less than the
non-linear equilibrium prescription. Expected human behavior: substantial
overbidding relative to $b^\star(v)$, with large variance across subjects.
Sunk-cost / "I already paid, I want the prize" effects likely amplify
overbidding.

---

## H2. First-price with $15 reserve (predicted hard; weighted SMAD = 58.5)

**Rule shown to subjects:**

> In this game, you will participate in an auction for a prize against 2
> other bidders. At the start of each round, you will see your value for the
> prize, randomly drawn between $0 and $49, with all values equally likely.
> After learning your value, you will submit a bid privately at the same time
> as the other bidders. Bids must be in $0.1 increments.
> **The auction has a minimum bid (a reserve price) of $15.** Bids strictly
> below $15 are treated as non-participating and earn $0. Among bids of at
> least $15, the highest bidder wins the prize and pays their own bid (a
> first-price auction with a $15 reserve). If you win, your earnings will
> increase by your value for the prize, and decrease by your own bid. If you
> do not win, your earnings will remain unchanged.
> After each auction, we will display all bids. Ties for the highest bid will
> be resolved randomly.

**Optimal bid:**
- If $v < 15$: any bid below $15$ is equivalent; non-participation is canonical.
- If $15 \le v \le 22.5$: $b^\star(v) = 15$ (the reserve binds).
- If $v > 22.5$: $b^\star(v) = (n-1)/n \cdot v = (2/3) v$ (standard first-price
shading, above the reserve).

**Panel signature observed.** c2 rational has SMAD $43.6$ -- the BNE shading is
not always derived. Overgeneralizer's $77.6$ reflects above-value bidding that
ignores the reserve constraint. The non-linear, piecewise optimum is the
expected source of difficulty.

---

## What to report back

For each mechanism and each subject:
1. Round-level $(v, b)$ pairs.
2. Whether the subject participated (for H2).
3. Demographic / experience covariates if available.

We will use those to:
1. Compute the realized per-subject SMAD against the same $b^\star$ formulas.
2. Compare the human ranking on weighted SMAD to the predicted ranking above.
3. Refit Li-2017 mixture weights against the new human population if needed.
4. Identify mechanisms where the panel ranking disagrees with the human
ranking -- those are diagnostic of what the panel is missing.

The decisive prediction is that the easy-vs-hard cluster gap should survive
contact with humans, and that E2 should beat E1 in the easy cluster.
