#!/usr/bin/env bash
set -euo pipefail
SRC="$(cd "$(dirname "$0")" && pwd)"
DST="$(cd "$SRC/.." && pwd)/spsb_basis_cot"
cp "$SRC"/c*_f*_b*.txt "$DST"/
printf 'Copied generated COT personas to %s
' "$DST"
