"""
Component 4a of Stage E: selection operators.

Pure-functional, no LLM, no IO. Operates on a list of `Individual` records —
just dicts with at minimum `rule_hash` and `fitness` keys. The loop driver
(`loop.py`) is responsible for wiring these into the rest of the pipeline.

Provides:
    tournament(pop, k, rng)         - one tournament-of-k winner
    select_parents(pop, n, k, rng)  - n parents drawn by independent tournaments
    elitism(pop, n)                 - top-n by fitness (no LLM, no mutation)
    dedup(pop)                      - remove duplicates by rule_hash, keep best
"""

import random


def tournament(pop, k=3, rng=None):
    """Pick `k` distinct (when possible) individuals, return the highest-fitness one."""
    rng = rng or random.Random()
    if not pop:
        raise ValueError("tournament called on empty population")
    sample_size = min(k, len(pop))
    contenders = rng.sample(pop, sample_size)
    return max(contenders, key=lambda ind: ind["fitness"])


def select_parents(pop, n, k=3, rng=None):
    """Draw `n` parents via independent tournaments-of-k. Allows the same parent to win twice."""
    rng = rng or random.Random()
    return [tournament(pop, k=k, rng=rng) for _ in range(n)]


def elitism(pop, n=2):
    """Return the top-n individuals by fitness, unmodified."""
    return sorted(pop, key=lambda ind: -ind["fitness"])[:n]


def dedup(pop):
    """Drop duplicates by rule_hash; when collisions exist, keep the one with the highest fitness."""
    best_by_hash = {}
    for ind in pop:
        h = ind["rule_hash"]
        if h not in best_by_hash or ind["fitness"] > best_by_hash[h]["fitness"]:
            best_by_hash[h] = ind
    return list(best_by_hash.values())
