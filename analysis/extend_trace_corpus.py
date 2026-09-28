#!/usr/bin/env python3
"""
Corpus extension for the traces re-analysis (auction-v4, "What the Traces Say").

Extends the FROZEN trace corpus (results/traces/trace_features.csv, built by
analysis/build_trace_features.py) with three additions, WITHOUT touching the
frozen file or the frozen 18-feature dictionary:

  (a) auction-level cluster IDs (`auction_id`) for every legacy row, enabling
      auction-level inference where run-level clustering is degenerate
      (one run per model x experiment);
  (b) previously unscored trace pools, scored with the frozen dictionary:
        - frontier models (gpt5, gpt5mini, claude_sonnet5, gemini25flash)
          from experiment_logs/  (~5-6k traces)
        - robustness_logs/ model x format grid (llama, gemma27b, claude_sonnet,
          gemini, gpt5mini, gpt4o temperature variants)  (optional, default on)
  (c) an ADDITIVE feature group `FEATURES_V2` (formal-derivation mode, the
      target of paper Heuristic 7) applied to ALL rows. The original 18
      regexes are never edited; new measurement is new columns only.

Frozen-dictionary discipline
----------------------------
`freeze_manifest()` records a sha256 of the frozen FEATURES source and of
FEATURES_V2, plus the frozen CSV's row count and per-feature column sums, in
results/traces/corpus_manifest.json. The legacy reload is asserted feature-by-
feature, row-by-row equal to the frozen CSV before anything else is written.

Output: results/traces/trace_features_v2.csv
  = frozen 30 columns
  + corpus in {legacy, frontier, robustness}
  + auction_id (result-file stem / combined-CSV repetition_id)
  + model_dir, price_order, seal_clock, temperature, config_name, seed_base
  + 5 FEATURES_V2 columns + clock_exit_language (convenience copy of the
    provenance-note regex from build_trace_mediation.py)

Reproducibility: deterministic (sorted globs); numpy seed 1299 via the
build_trace_features import. Run:  python3 analysis/extend_trace_corpus.py
"""

import glob
import hashlib
import json
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import build_trace_features as btf  # frozen dictionary + parser of record

REPO = os.path.dirname(HERE)
ES = REPO
ES_LOGS = os.path.join(ES, "experiment_logs")
ROBUST = os.path.join(REPO, "robustness_logs")
TRACES = os.path.join(REPO, "results", "traces")
FROZEN_CSV = os.path.join(TRACES, "trace_features.csv")
OUT_CSV = os.path.join(TRACES, "trace_features_v2.csv")
MANIFEST = os.path.join(TRACES, "corpus_manifest.json")

FRONTIER_DIRS = ["gpt5", "gpt5mini", "claude_sonnet5", "gemini25flash"]

# ---------------------------------------------------------------------------
# FEATURES_V2: formal-derivation mode (paper Heuristic 7). ADDITIVE ONLY.
# Style matches the frozen dictionary: lowercased text, binary regex presence.
# Validated read-only on a 50 legacy + 50 frontier sample before freezing
# (see corpus_manifest.json: features_v2_sha256). Do not edit after freezing.
# ---------------------------------------------------------------------------
FEATURES_V2 = {
    # Recites an equilibrium concept or the closed-form equilibrium bid.
    "equilibrium_formula": (
        r"nash equilibrium|bayes(?:ian)?[- ]nash|symmetric equilibrium|"
        r"equilibrium bid(?:ding)? (?:strategy|function)|"
        r"\(n\s*[-−–]\s*1\)\s*/\s*n|\bbne\b|\b2/3 of (?:my|the) value"),
    # States dominance as a quantified claim over all rival actions.
    "dominance_proof": (
        r"weakly dominant|regardless of (?:what )?(?:the )?other|"
        r"no matter what (?:the )?other|"
        r"for (?:any|all|every) (?:possible )?(?:rival|opponent|other|competing) bid|"
        r"in (?:all|every|either|both) (?:case|cases|scenario|scenarios)"),
    # Writes payoff algebra (profit/payoff/utility as an expression).
    "algebraic_derivation": (
        r"\bv\s*[-−–]\s*(?:p|b|price)\b|profit\s*=|payoff\s*=|"
        r"utility\s*=|surplus\s*=|\be\s*\[|=\s*value\s*[-−–]"),
    # Enumerates cases explicitly (case 1 / case 2, (i)...(ii), two cases).
    "case_enumeration": (
        r"case \d|(?:two|three|both) cases|\(i+\)|"
        r"(?:^|\W)(?:1|i)\)\s.{0,160}\W(?:2|ii)\)\s"),
    # Recites the statistical environment model (IPV/iid/uniform draws).
    "env_model_recital": (
        r"i\.?i\.?d\.?\b|independent private values|independently drawn|"
        r"uniform(?:ly)?(?: distributed)? (?:on|over|between|from)"),
}
V2_COLS = list(FEATURES_V2.keys())

# Convenience copy of the provenance-note clock-exit regex
# (build_trace_mediation.py CLOCK_EXIT_RE), so the v2 corpus is self-contained.
CLOCK_EXIT_RE = (r"clock|exit price|stage 2|drop out|exit the auction|"
                 r"automatically exit")

MECH_BY_PRICE_ORDER = {"first": "fpsb_sealed", "second": "spsb_sealed",
                       "third": "tpsb_sealed"}


def _sha(obj) -> str:
    return hashlib.sha256(repr(sorted(obj.items())).encode()).hexdigest()


def freeze_manifest(frozen: pd.DataFrame) -> dict:
    man = {
        "frozen_csv": os.path.relpath(FROZEN_CSV, REPO),
        "frozen_rows": int(len(frozen)),
        "features_sha256": _sha(btf.FEATURES),
        "features_v2_sha256": _sha(FEATURES_V2),
        "frozen_feature_sums": {c: int(frozen[c].sum())
                                for c in btf.FEATURE_COLS},
    }
    with open(MANIFEST, "w") as f:
        json.dump(man, f, indent=2)
    return man


# ---------------------------------------------------------------------------
# Legacy reload with auction_id, asserted equal to the frozen CSV.
# Loader bodies mirror build_trace_features exactly (same sort order).
# ---------------------------------------------------------------------------
def _rows_from_result_json_aligned(path, model, experiment, mechanism,
                                   source, run_id):
    """Like btf._rows_from_result_json, but aligns the bidding history to the
    value/plan arrays BY AGENT NAME instead of positionally, and records the
    winner flag.

    Rationale (verified 2026-08-02): sealed-bid logs store 'bidding history'
    in agent order (Bidder Andy, Betty, Charles -- alphabetical), so the
    positional zip is correct there. ASCENDING-CLOCK logs store it in EXIT
    order (sorted by drop price), so the positional zip mispairs values and
    exit prices (e.g. value 20 paired with the winner's 45). Name-based
    alignment is the identity on sealed logs and the fix on clock logs.
    The winner's recorded 'bid' in a clock is the final price (censored at
    the runner-up's exit), flagged via is_winner."""
    rows = []
    try:
        with open(path) as f:
            d = json.load(f)
    except (json.JSONDecodeError, OSError):
        return rows
    for _rkey, rd in d.items():
        if not isinstance(rd, dict) or "plan" not in rd:
            continue
        vals = rd.get("value", [])
        hist = (rd.get("history") or {}).get("bidding history", [])
        plans = rd.get("plan", [])
        if not (len(vals) == len(hist) == len(plans)):
            continue
        agents = sorted(h["agent"] for h in hist)
        if len(set(agents)) != len(hist):
            continue  # duplicate agent entries: cannot align by name
        bid_by_agent = {h["agent"]: h["bid"] for h in hist}
        winner = ((rd.get("history") or {}).get("winner") or {}).get("winner")
        for i, (v, p) in enumerate(zip(vals, plans)):
            agent = agents[i]
            rows.append({
                "source": source, "model": model, "experiment": experiment,
                "mechanism": mechanism, "run_id": run_id,
                "value": float(v), "bid": float(bid_by_agent[agent]),
                "plan": str(p), "is_winner": int(agent == winner),
            })
    return rows


def _rows_with_auction(path, model, experiment, mechanism, source, run_id):
    rows = _rows_from_result_json_aligned(path, model, experiment, mechanism,
                                          source, run_id)
    stem = os.path.splitext(os.path.basename(path))[0]
    for r in rows:
        r["auction_id"] = f"{run_id}|{stem}"
    return rows


def load_combined_with_auction() -> pd.DataFrame:
    df = pd.read_csv(btf.COMBINED_CSV)
    out = pd.DataFrame({
        "source": "combined_csv",
        "model": df["model"],
        "experiment": df["experiment"],
        "mechanism": np.where(df["experiment"].eq("ascending_clock_closed"),
                              "ascending_clock", "spsb_sealed"),
        "run_id": (df["model"].str[:12] + "|" + df["experiment"] + "|"
                   + df["timestamp"]),
        "value": df["player_value"].astype(float),
        "bid": df["bid"].astype(float),
        "plan": df["plan"].astype(str),
        "is_winner": df["is_winner"].astype(int),
        "auction_id": (df["model"].str[:12] + "|" + df["experiment"] + "|"
                       + df["repetition_id"].astype(str)),
        "temperature": df["temperature"],
        "seal_clock": df["seal_clock"],
        "price_order": df["price_order"],
    })
    return out


def load_menu_proxy_with_auction() -> pd.DataFrame:
    model_map = {"gpt4o": "gpt-4o", "claude": "claude-3-5-haiku-20241022",
                 "gemini": "gemini-2.0-flash", "gemma": "google/gemma-3-27b-it"}
    rows = []
    for mdir, model in model_map.items():
        for exp in ("intervention_menu", "intervention_proxy_breitmoser"):
            files = sorted(glob.glob(os.path.join(
                ES_LOGS, mdir, exp, "result_*.json")))
            run_id = f"{model[:12]}|{exp}|flat"
            for fp in files:
                rows.extend(_rows_with_auction(
                    fp, model, exp, "spsb_sealed", "es_logs_flat", run_id))
    return pd.DataFrame(rows)


def load_v12_with_auction() -> pd.DataFrame:
    rows = []
    for fam in sorted(os.listdir(btf.V12)):
        if not (fam.endswith("_first") or fam.endswith("_third")):
            continue
        mech = "fpsb_sealed" if fam.endswith("_first") else "tpsb_sealed"
        for run_dir in sorted(glob.glob(os.path.join(btf.V12, fam, "run_*"))):
            run_id = f"gpt-4o|{fam}|{os.path.basename(run_dir)}"
            for fp in sorted(glob.glob(os.path.join(
                    run_dir, "raw_data", "result_*.json"))):
                rows.extend(_rows_with_auction(
                    fp, "gpt-4o", fam, mech, "v12_recovered", run_id))
    return pd.DataFrame(rows)


def assert_legacy_equals_frozen(legacy: pd.DataFrame, frozen: pd.DataFrame):
    """Row-by-row equality of the re-scored legacy corpus vs the frozen CSV on
    every frozen feature column plus identifiers. Aborts on any mismatch."""
    assert len(legacy) == len(frozen), (
        f"row count mismatch: legacy={len(legacy)} frozen={len(frozen)}")
    for col in ["source", "model", "experiment", "mechanism", "run_id",
                "family"]:
        same = (legacy[col].astype(str).values
                == frozen[col].astype(str).values)
        assert same.all(), f"column {col}: {(~same).sum()} mismatching rows"
    for col in ["value", "bid", "deviation", "abs_dev"]:
        assert np.allclose(legacy[col].values, frozen[col].values,
                           equal_nan=True), f"float column {col} mismatch"
    for col in btf.FEATURE_COLS + ["n_words"]:
        same = legacy[col].values == frozen[col].values
        assert same.all(), (
            f"feature {col}: {(~same).sum()} mismatching rows "
            f"(sums {legacy[col].sum()} vs {frozen[col].sum()})")
    print(f"[assert] legacy reload == frozen CSV on all {len(btf.FEATURE_COLS)}"
          f" features + identifiers ({len(legacy)} rows)", file=sys.stderr)


# ---------------------------------------------------------------------------
# Frontier + robustness loaders (config.yaml-driven; never hardcode model
# names or mechanisms -- the claude_sonnet5 menu cell is the cautionary tale).
# ---------------------------------------------------------------------------
def _read_config(run_dir):
    import yaml
    cfg_path = os.path.join(run_dir, "config.yaml")
    try:
        with open(cfg_path) as f:
            return yaml.safe_load(f)
    except (OSError, Exception):
        return None


def _mechanism_from_cfg(cfg, exp_name):
    rule = (cfg or {}).get("rule", {})
    special = str(rule.get("special_name", ""))
    if str(rule.get("seal_clock", "")) == "clock" or \
            exp_name.startswith("ascending_clock"):
        return "ascending_clock"
    if "all_pay" in special or "all_pay" in exp_name:
        return "allpay_sealed"
    return MECH_BY_PRICE_ORDER.get(str(rule.get("price_order", "")),
                                   "unknown")


def _load_run_tree(base_dir, exp_name, model_dir, source, corpus,
                   coverage_rows):
    """Walk <base>/run_*/raw_data/result_*.json with config.yaml metadata."""
    rows = []
    run_dirs = sorted(glob.glob(os.path.join(base_dir, "run_*")))
    for run_dir in run_dirs:
        cfg = _read_config(run_dir)
        if cfg is None:
            coverage_rows.append(dict(corpus=corpus, cell=exp_name,
                                      run=os.path.basename(run_dir),
                                      files=0, parsed=0,
                                      note="config.yaml unreadable"))
            continue
        model = str(cfg.get("llm", {}).get("model", model_dir))
        mech = _mechanism_from_cfg(cfg, exp_name)
        run_id = f"{model}|{exp_name}|{os.path.basename(run_dir)}"
        files = sorted(glob.glob(os.path.join(run_dir, "raw_data",
                                              "result_*.json")))
        parsed = 0
        for fp in files:
            got = _rows_with_auction(fp, model, exp_name, mech, source, run_id)
            if got:
                parsed += 1
            for r in got:
                r.update(
                    model_dir=model_dir,
                    temperature=cfg.get("llm", {}).get("temperature"),
                    price_order=cfg.get("rule", {}).get("price_order"),
                    seal_clock=cfg.get("rule", {}).get("seal_clock"),
                    config_name=cfg.get("experiment", {}).get("name"),
                    seed_base=cfg.get("value", {}).get("seed_base"),
                )
            rows.extend(got)
        coverage_rows.append(dict(corpus=corpus, cell=exp_name,
                                  run=os.path.basename(run_dir),
                                  files=len(files), parsed=parsed, note=""))
    return rows


def load_frontier(coverage_rows) -> pd.DataFrame:
    rows = []
    for mdir in FRONTIER_DIRS:
        root = os.path.join(ES_LOGS, mdir)
        if not os.path.isdir(root):
            print(f"[warn] frontier dir missing: {root}", file=sys.stderr)
            continue
        for exp in sorted(os.listdir(root)):
            exp_dir = os.path.join(root, exp)
            if not os.path.isdir(exp_dir):
                continue
            rows.extend(_load_run_tree(exp_dir, exp, mdir,
                                       "frontier_logs", "frontier",
                                       coverage_rows))
    df = pd.DataFrame(rows)
    return df


def load_robustness(coverage_rows) -> pd.DataFrame:
    rows = []
    for cell in sorted(os.listdir(ROBUST)):
        if cell == "V10":  # V10/ duplicates two top-level cells (same run ids)
            continue
        cell_dir = os.path.join(ROBUST, cell)
        if not os.path.isdir(cell_dir):
            continue
        rows.extend(_load_run_tree(cell_dir, cell, cell,
                                   "robustness_logs", "robustness",
                                   coverage_rows))
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# V2 features (additive) + assembly
# ---------------------------------------------------------------------------
def add_features_v2(df: pd.DataFrame) -> pd.DataFrame:
    low = df["plan"].str.lower()
    for name, pat in FEATURES_V2.items():
        df[name] = low.str.contains(pat, regex=True).astype(int)
    df["clock_exit_language"] = (low.str.contains(CLOCK_EXIT_RE, regex=True)
                                 .astype(int))
    return df


def main():
    include_robustness = "--no-robustness" not in sys.argv

    frozen = pd.read_csv(FROZEN_CSV)
    man = freeze_manifest(frozen)
    print(f"[manifest] features sha {man['features_sha256'][:12]}..., "
          f"v2 sha {man['features_v2_sha256'][:12]}..., "
          f"frozen rows {man['frozen_rows']}", file=sys.stderr)

    # --- legacy, with auction ids, asserted equal to frozen -----------------
    print("Reloading legacy corpus with auction ids ...", file=sys.stderr)
    legacy = pd.concat([load_combined_with_auction(),
                        load_menu_proxy_with_auction(),
                        load_v12_with_auction()], ignore_index=True)
    legacy = btf.add_features(legacy)
    assert_legacy_equals_frozen(legacy, frozen)
    legacy["corpus"] = "legacy"
    legacy["model_dir"] = np.nan
    legacy["config_name"] = np.nan
    legacy["seed_base"] = np.nan

    # --- new pools -----------------------------------------------------------
    coverage_rows = []
    print("Loading frontier logs ...", file=sys.stderr)
    frontier = load_frontier(coverage_rows)
    if len(frontier):
        frontier = btf.add_features(frontier)
        frontier["corpus"] = "frontier"
    parts = [legacy, frontier]

    if include_robustness:
        print("Loading robustness logs ...", file=sys.stderr)
        robust = load_robustness(coverage_rows)
        if len(robust):
            robust = btf.add_features(robust)
            robust["corpus"] = "robustness"
        parts.append(robust)

    df = pd.concat(parts, ignore_index=True)
    df = add_features_v2(df)

    cov = pd.DataFrame(coverage_rows)
    cov_path = os.path.join(TRACES, "corpus_v2_coverage.csv")
    cov.to_csv(cov_path, index=False)
    bad = cov[(cov["files"] > 0) & (cov["parsed"] / cov["files"] < 0.95)]
    if len(bad):
        print("[warn] cells below 95% parse coverage:", file=sys.stderr)
        print(bad.to_string(index=False), file=sys.stderr)

    df.to_csv(OUT_CSV, index=False)
    print(f"\ntrace_features_v2.csv: {len(df)} rows "
          f"(legacy={len(legacy)}, frontier={len(frontier)}, "
          f"robustness={len(df) - len(legacy) - len(frontier)})",
          file=sys.stderr)

    # --- summary stats to eyeball ------------------------------------------
    print("\nRows by corpus x mechanism:", file=sys.stderr)
    print(df.groupby(["corpus", "mechanism"]).size().to_string(),
          file=sys.stderr)
    print("\nFrontier rows by model x experiment:", file=sys.stderr)
    if len(frontier):
        print(frontier.groupby(["model", "experiment"]).size().to_string(),
              file=sys.stderr)
    print("\nformal_derivation feature prevalence by corpus (sealed only):",
          file=sys.stderr)
    sealed = df[df["mechanism"].str.contains("sealed", na=False)]
    print(sealed.groupby("corpus")[V2_COLS].mean().round(4).to_string(),
          file=sys.stderr)

    # v2-regex eyeball sample: 6 frontier hits + 4 legacy hits, verbatim.
    hit = sealed[(sealed[V2_COLS].sum(axis=1) >= 2)]
    for corpus in ["frontier", "legacy"]:
        sub = hit[hit["corpus"] == corpus].head(6 if corpus == "frontier" else 4)
        print(f"\n--- sample {corpus} formal-derivation hits ---",
              file=sys.stderr)
        for _, r in sub.iterrows():
            fired = [c for c in V2_COLS if r[c] == 1]
            print(f"[{r['model']} | {r['experiment']} | v={r['value']} "
                  f"b={r['bid']}] {fired}\n  {str(r['plan'])[:300]}\n",
                  file=sys.stderr)


if __name__ == "__main__":
    main()
