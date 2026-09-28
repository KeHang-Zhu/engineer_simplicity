#!/usr/bin/env python3
"""Verify released data bytes against the repository-relative source manifest."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    manifest = json.loads((ROOT / 'data/source_manifest.json').read_text())
    failures = []
    for entry in manifest['files']:
        path = ROOT / entry['path']
        if not path.is_file():
            failures.append(f"missing: {entry['path']}")
            continue
        content = path.read_bytes()
        if len(content) != entry['bytes'] or hashlib.sha256(content).hexdigest() != entry['sha256']:
            failures.append(f"changed: {entry['path']}")
    if failures:
        raise SystemExit('\n'.join(failures))
    print(f"Verified {len(manifest['files'])} released data files (SHA-256).")


if __name__ == '__main__':
    main()
