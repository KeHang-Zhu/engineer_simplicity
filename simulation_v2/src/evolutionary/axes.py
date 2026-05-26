"""
Design-space axes for single-item sealed-bid auctions.

Preliminary, hand-curated. Stage B (literature pass) will firm this up and
produce a machine-readable schema at writeup_v2/design_space_schema.yaml.

Categorical axes have a `values` list; ordinal axes have a `levels` list ordered
from "least" to "most" of the property.
"""

from dataclasses import dataclass, field
from typing import List


@dataclass(frozen=True)
class CategoricalAxis:
    name: str
    description: str
    values: List[str]
    kind: str = "categorical"


@dataclass(frozen=True)
class OrdinalAxis:
    name: str
    description: str
    levels: List[str]  # ordered low → high
    kind: str = "ordinal"


# --- The current axis set ---

DESIGN_AXES = [
    CategoricalAxis(
        name="pricing_rule",
        description=(
            "How the winner's payment is determined from the bid vector."
        ),
        values=[
            "first-price",          # winner pays own bid
            "second-price",         # winner pays second-highest bid
            "third-price",          # winner pays third-highest bid
            "all-pay",              # all bidders pay their bid; highest wins
            "average-price",        # winner pays mean of submitted bids
        ],
    ),
    CategoricalAxis(
        name="payment_modifier",
        description=(
            "Additional payment mechanics layered on top of the pricing rule."
        ),
        values=[
            "none",
            "with-reserve",         # reserve price; below-reserve bids reject
            "entry-fee",            # non-refundable fee to participate
            "refundable-deposit",   # fee refunded if you don't win
        ],
    ),
    CategoricalAxis(
        name="bid_language",
        description=(
            "What the bidder is asked to submit."
        ),
        values=[
            "continuous-bid",       # arbitrary $ amount
            "discrete-grid",        # bid from a coarse menu (e.g. $0, $10, $20)
            "binary-accept",        # accept/reject a fixed offer
            "ranking",              # rank candidate price points
        ],
    ),
    CategoricalAxis(
        name="info_disclosure",
        description=(
            "What bidders learn before/after submitting."
        ),
        values=[
            "private-only",                 # bidders see only their own value
            "post-clearing-bid-revelation", # all bids revealed after clearing
            "pre-bid-shared-signal",        # a common signal shared before bidding
        ],
    ),
    OrdinalAxis(
        name="simplicity_emphasis",
        description=(
            "How forcefully the rule prompt frames the dominant action as obvious "
            "to the bidder. Higher = more pedagogical/explicit."
        ),
        levels=["none", "low", "medium", "high"],
    ),
]


def axis_dict():
    """Flatten the axis spec into {name: list_of_levels_or_values}."""
    out = {}
    for ax in DESIGN_AXES:
        if isinstance(ax, CategoricalAxis):
            out[ax.name] = ax.values
        else:
            out[ax.name] = ax.levels
    return out
