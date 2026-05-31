"""
Single-item auction genotype contract for Digital Mice Stage D/E.

The genotype is the search object: a structured sealed-bid auction design.
LLM calls may instantiate a genotype into bidder-facing prose, but EA operators
should mutate and cross over this structured object first.

This module intentionally stays lightweight: it validates plain dicts loaded
from YAML/JSON, provides benchmark genotypes, computes static complexity
metadata, and maps genotypes to the older mechanism-generator contract.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import itertools
import json
import random
from typing import Any, Dict, Iterable, List

import yaml


SCHEMA_VERSION = "single_item_auction_genotype_v1"
DOMAIN = "single_item_auction"
IMPLEMENTATION_SCOPE = "sealed_bid_v1"

VALUE_MODELS = {"IPV", "APV", "CV"}
FEEDBACK = {"none", "post_round_outcome", "history_transcript"}
ALLOCATION_RULES = {
    "highest_bid_wins",
    "reserve_then_highest_bid",
    "posted_price_acceptance",
}
PAYMENT_RULES = {"first_price", "second_price", "third_price", "posted_price"}
RESERVE_RULES = {"none", "public_reserve", "hidden_reserve"}
BID_LANGUAGES = {
    "continuous_bid",
    "discrete_grid",
    "ranked_price_menu",
    "binary_accept",
}
DESCRIPTION_STYLES = {"traditional", "menu", "clock_framing"}
MECHANISM_SCAFFOLDS = {"none", "short_hint", "worked_example", "decision_table"}
TARGET_TYPES = {
    "truthful_bid",
    "bne_bid_ratio",
    "common_value_shading",
    "threshold_accept",
    "mechanism_specific",
}
STRATEGIC_DEPTH = {"low", "medium", "high"}
GENOTYPE_SCOPES = {"unrestricted", "first_price_ipv", "second_price_ipv"}
DEFAULT_SCOPE = "unrestricted"


DEFAULT_GENOTYPE: Dict[str, Any] = {
    "schema_version": SCHEMA_VERSION,
    "domain": DOMAIN,
    "implementation_scope": IMPLEMENTATION_SCOPE,
    "environment": {
        "value_model": "IPV",
        "n_bidders": 3,
        "value_support": {
            "private_range": [0, 49],
            "common_range": None,
            "signal_noise": None,
        },
        "increment": 0.1,
        "repetitions": "single_auction",
        "feedback": "none",
    },
    "mechanism": {
        "interaction_form": "sealed_bid",
        "allocation_rule": "highest_bid_wins",
        "payment_rule": "first_price",
        "reserve_rule": "none",
        "bid_language": "continuous_bid",
        "grid": None,
        "tie_breaking": "random",
    },
    "framing": {
        "description_style": "traditional",
        "mechanism_scaffold": "none",
        "has_chain_of_thought_demo": False,
        "language": "english",
    },
    "target_policy": {
        "type": "bne_bid_ratio",
        "benchmark_formula": "((n-1)/n)*value",
        "target_bid_value_ratio": 2 / 3,
        "tolerance": 0.10,
    },
    "complexity": {
        "action_space_size": "computed",
        "strategic_depth_required": "computed",
        "belief_required": "computed",
        "payoff_contingency_count": "computed",
        "rule_word_count": "computed",
    },
}


def _deep_update(base: Dict[str, Any], updates: Dict[str, Any]) -> Dict[str, Any]:
    out = copy.deepcopy(base)
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_update(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def normalize_genotype(raw: Dict[str, Any]) -> Dict[str, Any]:
    """Fill defaults and compute complexity metadata."""
    if not isinstance(raw, dict):
        raise ValueError("genotype must be a mapping")
    genotype = _deep_update(DEFAULT_GENOTYPE, raw)
    genotype["complexity"] = {
        **genotype.get("complexity", {}),
        **compute_complexity(genotype),
    }
    genotype["genotype_id"] = genotype.get("genotype_id") or genotype_hash(genotype)
    return genotype


def genotype_hash(genotype: Dict[str, Any]) -> str:
    """Stable hash over genotype content, excluding derived ids."""
    clean = copy.deepcopy(genotype)
    clean.pop("genotype_id", None)
    blob = json.dumps(clean, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:12]


def _require(value: Any, allowed: Iterable[Any], path: str) -> None:
    allowed_set = set(allowed)
    if value not in allowed_set:
        raise ValueError(f"{path}={value!r} must be one of {sorted(allowed_set)}")


def _range_or_none(value: Any, path: str) -> None:
    if value is None:
        return
    if (
        not isinstance(value, list)
        or len(value) != 2
        or not all(isinstance(x, (int, float)) for x in value)
        or value[0] > value[1]
    ):
        raise ValueError(f"{path} must be null or [low, high]")


def validate_genotype(raw: Dict[str, Any]) -> Dict[str, Any]:
    """Return a normalized genotype or raise ValueError with a precise reason."""
    g = normalize_genotype(raw)
    env = g["environment"]
    support = env["value_support"]
    mech = g["mechanism"]
    framing = g["framing"]
    target = g["target_policy"]

    if g["schema_version"] != SCHEMA_VERSION:
        raise ValueError(f"schema_version must be {SCHEMA_VERSION}")
    if g["domain"] != DOMAIN:
        raise ValueError(f"domain must be {DOMAIN}")
    if g["implementation_scope"] != IMPLEMENTATION_SCOPE:
        raise ValueError(f"implementation_scope must be {IMPLEMENTATION_SCOPE}")
    if mech.get("interaction_form") != "sealed_bid":
        raise ValueError("v1 genotype only supports mechanism.interaction_form=sealed_bid")

    _require(env["value_model"], VALUE_MODELS, "environment.value_model")
    if not isinstance(env["n_bidders"], int) or env["n_bidders"] < 2:
        raise ValueError("environment.n_bidders must be an integer >= 2")
    if not isinstance(env["increment"], (int, float)) or env["increment"] <= 0:
        raise ValueError("environment.increment must be positive")
    if env["repetitions"] != "single_auction":
        raise ValueError("v1 genotype only supports repetitions=single_auction")
    _require(env["feedback"], FEEDBACK, "environment.feedback")
    _range_or_none(support.get("private_range"), "environment.value_support.private_range")
    _range_or_none(support.get("common_range"), "environment.value_support.common_range")
    _range_or_none(support.get("signal_noise"), "environment.value_support.signal_noise")

    _require(mech["allocation_rule"], ALLOCATION_RULES, "mechanism.allocation_rule")
    _require(mech["payment_rule"], PAYMENT_RULES, "mechanism.payment_rule")
    _require(mech["reserve_rule"], RESERVE_RULES, "mechanism.reserve_rule")
    _require(mech["bid_language"], BID_LANGUAGES, "mechanism.bid_language")
    if mech["tie_breaking"] != "random":
        raise ValueError("v1 genotype only supports mechanism.tie_breaking=random")

    grid = mech.get("grid")
    if mech["bid_language"] in {"discrete_grid", "ranked_price_menu"}:
        if not isinstance(grid, list) or len(grid) < 2:
            raise ValueError("discrete_grid/ranked_price_menu require mechanism.grid")
        if not all(isinstance(x, (int, float)) for x in grid):
            raise ValueError("mechanism.grid must contain numeric prices")
    elif mech["bid_language"] == "continuous_bid" and grid is not None:
        raise ValueError("continuous_bid requires mechanism.grid=null")
    if mech["bid_language"] == "binary_accept":
        if mech["payment_rule"] != "posted_price":
            raise ValueError("binary_accept requires payment_rule=posted_price")
        if mech["allocation_rule"] != "posted_price_acceptance":
            raise ValueError("binary_accept requires allocation_rule=posted_price_acceptance")

    _require(framing["description_style"], DESCRIPTION_STYLES, "framing.description_style")
    _require(framing["mechanism_scaffold"], MECHANISM_SCAFFOLDS, "framing.mechanism_scaffold")
    if framing["language"] != "english":
        raise ValueError("v1 genotype only supports framing.language=english")
    if not isinstance(framing["has_chain_of_thought_demo"], bool):
        raise ValueError("framing.has_chain_of_thought_demo must be boolean")

    _require(target["type"], TARGET_TYPES, "target_policy.type")
    if not isinstance(target["tolerance"], (int, float)) or target["tolerance"] < 0:
        raise ValueError("target_policy.tolerance must be nonnegative")
    if target["type"] == "truthful_bid":
        if target["benchmark_formula"] != "value":
            raise ValueError("truthful_bid requires benchmark_formula=value")
    if target["type"] == "bne_bid_ratio":
        ratio = target.get("target_bid_value_ratio")
        if not isinstance(ratio, (int, float)) or ratio <= 0:
            raise ValueError("bne_bid_ratio requires positive target_bid_value_ratio")
    if target["type"] == "common_value_shading" and env["value_model"] != "CV":
        raise ValueError("common_value_shading requires environment.value_model=CV")
    if target["type"] == "threshold_accept" and mech["payment_rule"] != "posted_price":
        raise ValueError("threshold_accept requires payment_rule=posted_price")

    return g


def compute_complexity(genotype: Dict[str, Any]) -> Dict[str, Any]:
    env = genotype.get("environment", {})
    support = env.get("value_support", {})
    mech = genotype.get("mechanism", {})
    framing = genotype.get("framing", {})
    target = genotype.get("target_policy", {})
    private_range = support.get("private_range") or [0, 49]
    increment = float(env.get("increment", 0.1) or 0.1)

    bid_language = mech.get("bid_language")
    grid = mech.get("grid")
    if bid_language == "binary_accept":
        action_space_size: Any = 2
    elif bid_language in {"discrete_grid", "ranked_price_menu"} and isinstance(grid, list):
        action_space_size = len(grid)
    elif bid_language == "continuous_bid":
        action_space_size = int(round((private_range[1] - private_range[0]) / increment)) + 1
    else:
        action_space_size = "computed"

    payment_rule = mech.get("payment_rule")
    target_type = target.get("type")
    value_model = env.get("value_model")
    if value_model == "CV" or target_type == "common_value_shading":
        depth = "high"
        belief_required = True
    elif payment_rule in {"first_price", "third_price"} or target_type == "bne_bid_ratio":
        depth = "medium"
        belief_required = True
    elif payment_rule in {"second_price", "posted_price"}:
        depth = "low"
        belief_required = False
    else:
        depth = "medium"
        belief_required = True

    contingencies = 2  # win vs lose
    if mech.get("reserve_rule") != "none":
        contingencies += 1
    if value_model in {"APV", "CV"}:
        contingencies += 1
    if framing.get("description_style") == "clock_framing":
        contingencies += 1
    if framing.get("mechanism_scaffold") in {"worked_example", "decision_table"}:
        contingencies += 1

    return {
        "action_space_size": action_space_size,
        "strategic_depth_required": depth,
        "belief_required": belief_required,
        "payoff_contingency_count": contingencies,
        "rule_word_count": "computed",
    }


def default_target_for_payment(payment_rule: str, n_bidders: int) -> Dict[str, Any]:
    if payment_rule == "first_price":
        ratio = (n_bidders - 1) / n_bidders
        return {
            "type": "bne_bid_ratio",
            "benchmark_formula": "((n-1)/n)*value",
            "target_bid_value_ratio": ratio,
            "tolerance": 0.10,
        }
    if payment_rule == "third_price":
        ratio = (n_bidders - 1) / max(n_bidders - 2, 1)
        return {
            "type": "bne_bid_ratio",
            "benchmark_formula": "((n-1)/(n-2))*value",
            "target_bid_value_ratio": ratio,
            "tolerance": 0.15,
        }
    if payment_rule == "second_price":
        return {
            "type": "truthful_bid",
            "benchmark_formula": "value",
            "target_bid_value_ratio": 1.0,
            "tolerance": 0.10,
        }
    if payment_rule == "posted_price":
        return {
            "type": "threshold_accept",
            "benchmark_formula": "custom",
            "target_bid_value_ratio": None,
            "tolerance": 0.0,
        }
    return copy.deepcopy(DEFAULT_GENOTYPE["target_policy"])


def validate_genotype_scope(raw: Dict[str, Any],
                            scope: str | None = DEFAULT_SCOPE) -> Dict[str, Any]:
    """Validate a genotype against an optional executable search scope."""
    scope = scope or DEFAULT_SCOPE
    if scope not in GENOTYPE_SCOPES:
        raise ValueError(f"unknown genotype scope {scope!r}; expected {sorted(GENOTYPE_SCOPES)}")
    g = validate_genotype(raw)
    if scope == "unrestricted":
        return g

    env = g["environment"]
    mech = g["mechanism"]
    target = g["target_policy"]
    errors = []
    if env["value_model"] != "IPV":
        errors.append("environment.value_model must remain IPV")
    if env["n_bidders"] != 3:
        errors.append("environment.n_bidders must remain 3")
    if env["value_support"].get("common_range") is not None:
        errors.append("environment.value_support.common_range must remain null")
    if env["value_support"].get("signal_noise") is not None:
        errors.append("environment.value_support.signal_noise must remain null")
    if mech.get("interaction_form") != "sealed_bid":
        errors.append("mechanism.interaction_form must remain sealed_bid")
    if mech["allocation_rule"] != "highest_bid_wins":
        errors.append("mechanism.allocation_rule must remain highest_bid_wins")

    if scope == "first_price_ipv":
        if mech["payment_rule"] != "first_price":
            errors.append("mechanism.payment_rule must remain first_price")
        if mech["reserve_rule"] != "none":
            errors.append("mechanism.reserve_rule must remain none")
        if mech["bid_language"] == "binary_accept":
            errors.append("mechanism.bid_language=binary_accept is outside first_price_ipv")
        if target["type"] != "bne_bid_ratio":
            errors.append("target_policy.type must remain bne_bid_ratio")
        expected_ratio = (env["n_bidders"] - 1) / env["n_bidders"]
        ratio = target.get("target_bid_value_ratio")
        if not isinstance(ratio, (int, float)) or abs(float(ratio) - expected_ratio) > 0.01:
            errors.append("target_policy.target_bid_value_ratio must remain near 2/3")
    elif scope == "second_price_ipv":
        if mech["payment_rule"] != "second_price":
            errors.append("mechanism.payment_rule must remain second_price")
        if mech["bid_language"] == "binary_accept":
            errors.append("mechanism.bid_language=binary_accept is outside second_price_ipv")
        if target["type"] != "truthful_bid":
            errors.append("target_policy.type must remain truthful_bid")
        ratio = target.get("target_bid_value_ratio")
        if not isinstance(ratio, (int, float)) or abs(float(ratio) - 1.0) > 0.01:
            errors.append("target_policy.target_bid_value_ratio must remain near 1.0")

    if errors:
        raise ValueError(f"genotype scope {scope!r} failed: " + "; ".join(errors))
    return g


def apply_genotype_scope(raw: Dict[str, Any],
                         scope: str | None = DEFAULT_SCOPE) -> Dict[str, Any]:
    """Repair locked axes for a scope and return a validated genotype."""
    scope = scope or DEFAULT_SCOPE
    if scope not in GENOTYPE_SCOPES:
        raise ValueError(f"unknown genotype scope {scope!r}; expected {sorted(GENOTYPE_SCOPES)}")
    g = validate_genotype(raw)
    if scope == "unrestricted":
        return g

    env = g["environment"]
    mech = g["mechanism"]
    target = g.get("target_policy") or {}

    env["value_model"] = "IPV"
    env["n_bidders"] = 3
    env["value_support"]["private_range"] = env["value_support"].get("private_range") or [0, 49]
    env["value_support"]["common_range"] = None
    env["value_support"]["signal_noise"] = None

    mech["interaction_form"] = "sealed_bid"
    mech["allocation_rule"] = "highest_bid_wins"
    if scope == "first_price_ipv":
        mech["payment_rule"] = "first_price"
        mech["reserve_rule"] = "none"
    elif scope == "second_price_ipv":
        mech["payment_rule"] = "second_price"
        # reserve_rule is allowed to vary for second-price (no_reserve / with_reserve)
    if mech["bid_language"] == "binary_accept":
        mech["bid_language"] = "continuous_bid"
    if mech["bid_language"] in {"discrete_grid", "ranked_price_menu"} and mech.get("grid") is None:
        mech["grid"] = [0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 49]
    if mech["bid_language"] == "continuous_bid":
        mech["grid"] = None

    tolerance = target.get("tolerance", 0.10)
    if not isinstance(tolerance, (int, float)) or tolerance < 0:
        tolerance = 0.10
    g["target_policy"] = default_target_for_payment(mech["payment_rule"], env["n_bidders"])
    g["target_policy"]["tolerance"] = float(tolerance)

    g.pop("genotype_id", None)
    return validate_genotype_scope(g, scope)


def make_genotype(**overrides: Any) -> Dict[str, Any]:
    """Build a normalized genotype from nested override sections."""
    return validate_genotype(overrides)


def benchmark_genotypes() -> List[Dict[str, Any]]:
    """Ten hand-written benchmarks and scaffolding variants for v1 validation."""
    grid = [0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 49]
    examples = [
        {
            "genotype_id": "fpsb_bare_ipv_n3",
            "mechanism": {"payment_rule": "first_price"},
        },
        {
            "genotype_id": "fpsb_bne_hint_ipv_n3",
            "mechanism": {"payment_rule": "first_price"},
            "framing": {"mechanism_scaffold": "short_hint"},
        },
        {
            "genotype_id": "fpsb_worked_example_ipv_n3",
            "mechanism": {"payment_rule": "first_price"},
            "framing": {
                "mechanism_scaffold": "worked_example",
                "has_chain_of_thought_demo": True,
            },
        },
        {
            "genotype_id": "fpsb_coarse_grid_ipv_n3",
            "mechanism": {
                "payment_rule": "first_price",
                "bid_language": "discrete_grid",
                "grid": grid,
            },
        },
        {
            "genotype_id": "spsb_bare_ipv_n3",
            "mechanism": {"payment_rule": "second_price"},
            "target_policy": default_target_for_payment("second_price", 3),
        },
        {
            "genotype_id": "spsb_menu_framing_ipv_n3",
            "mechanism": {"payment_rule": "second_price"},
            "framing": {"description_style": "menu", "mechanism_scaffold": "short_hint"},
            "target_policy": default_target_for_payment("second_price", 3),
        },
        {
            "genotype_id": "spsb_clock_framing_ipv_n3",
            "mechanism": {"payment_rule": "second_price"},
            "framing": {
                "description_style": "clock_framing",
                "mechanism_scaffold": "decision_table",
            },
            "target_policy": default_target_for_payment("second_price", 3),
        },
        {
            "genotype_id": "tpsb_bare_ipv_n3",
            "mechanism": {"payment_rule": "third_price"},
            "target_policy": default_target_for_payment("third_price", 3),
        },
        {
            "genotype_id": "posted_price_binary_ipv_n3",
            "mechanism": {
                "allocation_rule": "posted_price_acceptance",
                "payment_rule": "posted_price",
                "bid_language": "binary_accept",
            },
            "target_policy": default_target_for_payment("posted_price", 3),
        },
        {
            "genotype_id": "cv_fpsb_winners_curse_n3",
            "environment": {
                "value_model": "CV",
                "value_support": {
                    "private_range": [0, 49],
                    "common_range": [20, 29],
                    "signal_noise": [-20, 20],
                },
            },
            "mechanism": {"payment_rule": "first_price"},
            "target_policy": {
                "type": "common_value_shading",
                "benchmark_formula": "custom",
                "target_bid_value_ratio": None,
                "tolerance": 0.15,
            },
        },
    ]
    return [validate_genotype(x) for x in examples]


def sample_genotypes(n: int, seed: int = 20260527,
                     include_benchmarks: bool = True) -> List[Dict[str, Any]]:
    """Deterministically sample sealed-bid v1 genotypes, excluding all-pay."""
    rng = random.Random(seed)
    out = benchmark_genotypes() if include_benchmarks else []
    axes = {
        "value_model": ["IPV", "APV", "CV"],
        "payment_rule": ["first_price", "second_price", "third_price", "posted_price"],
        "bid_language": ["continuous_bid", "discrete_grid", "ranked_price_menu", "binary_accept"],
        "description_style": ["traditional", "menu", "clock_framing"],
        "mechanism_scaffold": ["none", "short_hint", "worked_example", "decision_table"],
    }
    combos = list(itertools.product(*axes.values()))
    rng.shuffle(combos)
    for combo in combos:
        if len(out) >= n:
            break
        value_model, payment_rule, bid_language, description_style, scaffold = combo
        if bid_language == "binary_accept" and payment_rule != "posted_price":
            continue
        if payment_rule == "posted_price" and bid_language != "binary_accept":
            continue
        target = default_target_for_payment(payment_rule, 3)
        if value_model == "CV":
            target = {
                "type": "common_value_shading",
                "benchmark_formula": "custom",
                "target_bid_value_ratio": None,
                "tolerance": 0.15,
            }
        mechanism = {
            "allocation_rule": (
                "posted_price_acceptance"
                if payment_rule == "posted_price"
                else "highest_bid_wins"
            ),
            "payment_rule": payment_rule,
            "bid_language": bid_language,
            "grid": (
                [0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 49]
                if bid_language in {"discrete_grid", "ranked_price_menu"}
                else None
            ),
        }
        raw = {
            "environment": {
                "value_model": value_model,
                "value_support": {
                    "private_range": [0, 49],
                    "common_range": [20, 29] if value_model == "CV" else None,
                    "signal_noise": [-20, 20] if value_model == "CV" else None,
                },
            },
            "mechanism": mechanism,
            "framing": {
                "description_style": description_style,
                "mechanism_scaffold": scaffold,
                "has_chain_of_thought_demo": scaffold == "worked_example",
            },
            "target_policy": target,
        }
        out.append(validate_genotype(raw))
    return out[:n]


def _repair_payment_consistency(genotype: Dict[str, Any]) -> Dict[str, Any]:
    """Repair dependent fields after mutation/crossover."""
    g = normalize_genotype(genotype)
    mech = g["mechanism"]
    env = g["environment"]
    payment = mech["payment_rule"]
    if payment == "posted_price":
        mech["allocation_rule"] = "posted_price_acceptance"
        mech["bid_language"] = "binary_accept"
        mech["grid"] = None
    else:
        if mech["allocation_rule"] == "posted_price_acceptance":
            mech["allocation_rule"] = "highest_bid_wins"
        if mech["bid_language"] == "binary_accept":
            mech["bid_language"] = "continuous_bid"
        if mech["bid_language"] in {"discrete_grid", "ranked_price_menu"} and mech.get("grid") is None:
            mech["grid"] = [0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 49]
        if mech["bid_language"] == "continuous_bid":
            mech["grid"] = None

    if env["value_model"] == "CV":
        g["target_policy"] = {
            "type": "common_value_shading",
            "benchmark_formula": "custom",
            "target_bid_value_ratio": None,
            "tolerance": 0.15,
        }
        env["value_support"]["common_range"] = env["value_support"]["common_range"] or [20, 29]
        env["value_support"]["signal_noise"] = env["value_support"]["signal_noise"] or [-20, 20]
    else:
        env["value_support"]["common_range"] = None
        env["value_support"]["signal_noise"] = None
        g["target_policy"] = default_target_for_payment(payment, env["n_bidders"])
    g.pop("genotype_id", None)
    return validate_genotype(g)


def mutate_genotype(parent: Dict[str, Any], rng: random.Random | None = None,
                    scope: str | None = DEFAULT_SCOPE) -> Dict[str, Any]:
    """Small structural mutation of one genotype axis."""
    rng = rng or random.Random()
    child = validate_genotype_scope(parent, scope)
    ops = ["bid_language", "description_style", "mechanism_scaffold", "feedback"]
    if (scope or DEFAULT_SCOPE) == "unrestricted":
        ops.extend(["payment_rule", "value_model"])
    op = rng.choice(ops)
    if op == "payment_rule":
        choices = ["first_price", "second_price", "third_price", "posted_price"]
        choices.remove(child["mechanism"]["payment_rule"])
        child["mechanism"]["payment_rule"] = rng.choice(choices)
    elif op == "bid_language":
        choices = ["continuous_bid", "discrete_grid", "ranked_price_menu"]
        current = child["mechanism"]["bid_language"]
        if current in choices:
            choices.remove(current)
        child["mechanism"]["bid_language"] = rng.choice(choices)
        child["mechanism"]["grid"] = (
            [0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 49]
            if child["mechanism"]["bid_language"] in {"discrete_grid", "ranked_price_menu"}
            else None
        )
    elif op == "description_style":
        choices = ["traditional", "menu", "clock_framing"]
        choices.remove(child["framing"]["description_style"])
        child["framing"]["description_style"] = rng.choice(choices)
    elif op == "mechanism_scaffold":
        choices = ["none", "short_hint", "worked_example", "decision_table"]
        choices.remove(child["framing"]["mechanism_scaffold"])
        scaffold = rng.choice(choices)
        child["framing"]["mechanism_scaffold"] = scaffold
        child["framing"]["has_chain_of_thought_demo"] = scaffold == "worked_example"
    elif op == "value_model":
        choices = ["IPV", "APV", "CV"]
        choices.remove(child["environment"]["value_model"])
        child["environment"]["value_model"] = rng.choice(choices)
    elif op == "feedback":
        choices = ["none", "post_round_outcome", "history_transcript"]
        choices.remove(child["environment"]["feedback"])
        child["environment"]["feedback"] = rng.choice(choices)
    repaired = _repair_payment_consistency(child)
    repaired = apply_genotype_scope(repaired, scope)
    repaired["op"] = "genotype_mutation"
    repaired["parent_genotype_ids"] = [parent.get("genotype_id", genotype_hash(parent))]
    repaired["edit_summary"] = f"mutated {op}"
    return repaired


def crossover_genotypes(parent_a: Dict[str, Any], parent_b: Dict[str, Any],
                        rng: random.Random | None = None,
                        scope: str | None = DEFAULT_SCOPE) -> Dict[str, Any]:
    """Section-level crossover for compatible sealed-bid genotypes."""
    rng = rng or random.Random()
    a = validate_genotype_scope(parent_a, scope)
    b = validate_genotype_scope(parent_b, scope)
    child = copy.deepcopy(a)
    for section in ["environment", "mechanism", "framing"]:
        if rng.random() < 0.5:
            child[section] = copy.deepcopy(b[section])
    repaired = _repair_payment_consistency(child)
    repaired = apply_genotype_scope(repaired, scope)
    repaired["op"] = "genotype_crossover"
    repaired["parent_genotype_ids"] = [
        a.get("genotype_id", genotype_hash(a)),
        b.get("genotype_id", genotype_hash(b)),
    ]
    repaired["edit_summary"] = "section-level genotype crossover"
    return repaired


def genotype_to_legacy_coord(genotype: Dict[str, Any]) -> Dict[str, Any]:
    g = validate_genotype(genotype)
    mech = g["mechanism"]
    framing = g["framing"]
    payment = mech["payment_rule"].replace("_", "-")
    bid_language = {
        "continuous_bid": "continuous-bid",
        "discrete_grid": "discrete-grid",
        "ranked_price_menu": "ranking",
        "binary_accept": "binary-accept",
    }[mech["bid_language"]]
    scaffold_to_simplicity = {
        "none": "none",
        "short_hint": "medium",
        "worked_example": "high",
        "decision_table": "high",
    }
    return {
        "pricing_rule": payment,
        "payment_modifier": (
            "with-reserve" if mech["reserve_rule"] == "public_reserve" else "none"
        ),
        "bid_language": bid_language,
        "info_disclosure": (
            "post-clearing-bid-revelation"
            if g["environment"]["feedback"] == "post_round_outcome"
            else "private-only"
        ),
        "simplicity_emphasis": scaffold_to_simplicity[framing["mechanism_scaffold"]],
    }


def incentive_target_from_genotype(genotype: Dict[str, Any]) -> Dict[str, Any]:
    g = validate_genotype(genotype)
    target = g["target_policy"]
    target_type = target["type"]
    if target_type == "truthful_bid":
        return {
            "action": "truthful_bid",
            "description": "Bid exactly your value.",
            "parameters": {"absolute_tolerance": target["tolerance"]},
        }
    if target_type == "bne_bid_ratio":
        return {
            "action": "target_bid_value_ratio",
            "description": (
                "Bid near the benchmark bid/value ratio implied by the "
                "Bayes-Nash prediction."
            ),
            "parameters": {
                "target_bid_value_ratio": target["target_bid_value_ratio"],
                "ratio_tolerance": target["tolerance"],
            },
        }
    if target_type == "threshold_accept":
        return {
            "action": "threshold_accept",
            "description": "Accept the posted price iff value exceeds price.",
            "parameters": {"ratio_tolerance": target["tolerance"]},
        }
    if target_type == "common_value_shading":
        return {
            "action": "common_value_shading",
            "description": (
                "Shade below signal to account for adverse selection conditional "
                "on winning."
            ),
            "parameters": {"ratio_tolerance": target["tolerance"]},
        }
    return {
        "action": "mechanism_specific",
        "description": "Requires a mechanism-specific scorer.",
    }


def yaml_overrides_from_genotype(genotype: Dict[str, Any]) -> Dict[str, Any]:
    g = validate_genotype(genotype)
    payment = g["mechanism"]["payment_rule"]
    price_order = {
        "first_price": "first",
        "second_price": "second",
        "third_price": "third",
        "posted_price": "first",
    }[payment]
    support = g["environment"]["value_support"]
    overrides = {
        "rule.price_order": price_order,
        "rule.special_name": "",
        "value.private_range": support["private_range"][1],
        "value.increment": g["environment"]["increment"],
    }
    if g["mechanism"].get("grid") is not None:
        overrides["mechanism.grid"] = g["mechanism"]["grid"]
    return overrides


def _load_yaml(path: str) -> Any:
    with open(path) as f:
        return yaml.safe_load(f)


def _as_list(data: Any) -> List[Dict[str, Any]]:
    if isinstance(data, dict) and "genotypes" in data:
        data = data["genotypes"]
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        return [data]
    raise ValueError("expected a genotype mapping or a list of genotypes")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--validate", help="YAML file containing one or more genotypes")
    parser.add_argument("--examples", action="store_true", help="print benchmark genotypes")
    parser.add_argument("--sample", type=int, help="print N sampled genotypes")
    parser.add_argument("--scope", default=DEFAULT_SCOPE, choices=sorted(GENOTYPE_SCOPES))
    parser.add_argument("--seed", type=int, default=20260527)
    args = parser.parse_args()

    if args.validate:
        rows = _as_list(_load_yaml(args.validate))
        valid = [validate_genotype_scope(row, args.scope) for row in rows]
        print(f"validated {len(valid)} genotype(s)")
        for row in valid:
            print(f"  {row['genotype_id']}  {row['complexity']}")
        return
    if args.examples:
        print(yaml.safe_dump({"genotypes": benchmark_genotypes()}, sort_keys=False))
        return
    if args.sample:
        print(yaml.safe_dump(
            {"genotypes": sample_genotypes(args.sample, seed=args.seed)},
            sort_keys=False,
        ))
        return
    parser.print_help()


if __name__ == "__main__":
    main()
