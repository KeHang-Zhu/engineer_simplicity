"""
Component 4b of Stage E: SQLite persistence for the EA loop.

Schema:
    runs(
      run_id INTEGER PK,
      started_at TEXT,
      pop_size INTEGER, n_generations INTEGER,
      n_personas INTEGER, reps_per_persona INTEGER,
      model TEXT, seed INTEGER, notes TEXT
    )
    individuals(
      ind_id INTEGER PK,
      run_id INTEGER REFERENCES runs(run_id),
      generation INTEGER,
      op TEXT,                -- "init" | "mutation" | "crossover" | "elite"
      parent_hashes TEXT,     -- JSON list of parent rule_hashes
      rule_hash TEXT,
      name TEXT, one_line TEXT,
      yaml_path TEXT, tpl_path TEXT,
      fitness REAL,
      metrics_json TEXT,      -- full metrics dict from fitness.evaluate_mechanism
      coord_json TEXT,        -- design-axis coordinate or op summary
      created_at TEXT,
      UNIQUE(run_id, rule_hash, generation)
    )

`top_k(run_id, k)` returns the highest-fitness individuals across all
generations for a given run, sorted descending.
"""

import json
import os
import sqlite3
from datetime import datetime, timezone


SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    run_id INTEGER PRIMARY KEY AUTOINCREMENT,
    started_at TEXT NOT NULL,
    pop_size INTEGER, n_generations INTEGER,
    n_personas INTEGER, reps_per_persona INTEGER,
    model TEXT, seed INTEGER, notes TEXT
);

CREATE TABLE IF NOT EXISTS individuals (
    ind_id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id INTEGER NOT NULL REFERENCES runs(run_id),
    generation INTEGER NOT NULL,
    op TEXT NOT NULL,
    parent_hashes TEXT,
    rule_hash TEXT NOT NULL,
    name TEXT,
    one_line TEXT,
    yaml_path TEXT,
    tpl_path TEXT,
    fitness REAL,
    metrics_json TEXT,
    coord_json TEXT,
    created_at TEXT NOT NULL,
    UNIQUE(run_id, rule_hash, generation)
);

CREATE INDEX IF NOT EXISTS idx_individuals_run_gen
    ON individuals(run_id, generation);
CREATE INDEX IF NOT EXISTS idx_individuals_fitness
    ON individuals(run_id, fitness DESC);
"""


def _now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def connect(db_path):
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.executescript(SCHEMA)
    conn.commit()
    return conn


def start_run(conn, *, pop_size, n_generations, n_personas, reps_per_persona,
              model, seed, notes=""):
    cur = conn.execute(
        "INSERT INTO runs(started_at, pop_size, n_generations, n_personas, "
        "reps_per_persona, model, seed, notes) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        (_now(), pop_size, n_generations, n_personas, reps_per_persona,
         model, seed, notes),
    )
    conn.commit()
    return cur.lastrowid


def record_individual(conn, *, run_id, generation, mech, fitness, metrics,
                      yaml_path, tpl_path):
    """`mech` is the dict shape produced by mechanism_generator / mutation / crossover."""
    op = mech.get("op", "init")
    conn.execute(
        "INSERT OR IGNORE INTO individuals(run_id, generation, op, parent_hashes, "
        "rule_hash, name, one_line, yaml_path, tpl_path, fitness, metrics_json, "
        "coord_json, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (
            run_id, generation, op,
            json.dumps(mech.get("parent_hashes", [])),
            mech["rule_hash"], mech.get("name"), mech.get("one_line"),
            yaml_path, tpl_path, fitness,
            json.dumps(metrics, default=str),
            json.dumps(mech.get("coord", {}), default=str),
            _now(),
        ),
    )
    conn.commit()


def top_k(conn, run_id, k=10):
    cur = conn.execute(
        "SELECT ind_id, generation, op, name, one_line, rule_hash, fitness "
        "FROM individuals WHERE run_id = ? ORDER BY fitness DESC LIMIT ?",
        (run_id, k),
    )
    cols = [c[0] for c in cur.description]
    return [dict(zip(cols, row)) for row in cur.fetchall()]


def lineage(conn, run_id):
    """Return all individuals with parent links, in generation order."""
    cur = conn.execute(
        "SELECT generation, op, name, rule_hash, parent_hashes, fitness "
        "FROM individuals WHERE run_id = ? ORDER BY generation ASC, fitness DESC",
        (run_id,),
    )
    cols = [c[0] for c in cur.description]
    return [dict(zip(cols, row)) for row in cur.fetchall()]
