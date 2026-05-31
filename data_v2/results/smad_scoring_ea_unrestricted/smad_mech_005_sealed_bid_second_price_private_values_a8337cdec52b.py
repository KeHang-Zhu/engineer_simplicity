"""Auto-generated SMAD scoring module for mechanism: mech_005_sealed_bid_second_price_private_values_a8337cdec52b

Payment rule: second_price
Optimal-bid expression: b*(v) = v
Default bidder count: n = 3

SMAD = 100 * mean_a |b_a - b*(v_a)| / mean_a |b*(v_a)|.
Lower SMAD is better; SMAD = 0 means realized bids equal the
mechanism's declared optimum.
"""

N = 3


def optimal_bid(v, n=N):
    """Risk-neutral best response for this mechanism at value v."""
    return v


def smad(value_bid_pairs):
    """Compute SMAD given a list of (value, bid) pairs."""
    errors = []
    benchmarks = []
    for v, b in value_bid_pairs:
        if v is None or b is None or v <= 0:
            continue
        b_star = optimal_bid(v)
        errors.append(abs(b - b_star))
        benchmarks.append(abs(b_star))
    if not benchmarks:
        return None
    mean_b = sum(benchmarks) / len(benchmarks)
    if mean_b == 0:
        return None
    return 100.0 * (sum(errors) / len(errors)) / mean_b
