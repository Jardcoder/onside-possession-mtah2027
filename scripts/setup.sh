#!/usr/bin/env bash
# Idempotent setup of a fresh clone: venv from requirements.lock, official challenge repository, tests, hash check of the submission.
# Run through `./run.sh setup [options]`; safe to re-run, every step skips what is already in place.
#
#   --skip-tests | --skip-verify
#
# Exit code: 0 everything done, 3 a step needs something only a human can provide (the message says what), 1 a step failed.
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
if [ -f "$ROOT/.env" ]; then set -a; . "$ROOT/.env"; set +a; fi
CHALLENGE="${ONSIDE_CHALLENGE_DIR:-$ROOT/challenge}"
CHALLENGE_REPO="https://github.com/metrica-sports/more-than-a-hack-2027"
SKIP_TESTS=0; SKIP_VERIFY=0
while [ $# -gt 0 ]; do
  case "$1" in
    --skip-tests) SKIP_TESTS=1; shift ;;
    --skip-verify) SKIP_VERIFY=1; shift ;;
    -h|--help) sed -n 2,7p "${BASH_SOURCE[0]}"; exit 0 ;;
    *) echo "unknown option: $1 (see --help)" >&2; exit 2 ;;
  esac
done

NEEDS=()      # things only a human can provide
FAILS=()
say()  { printf '%s\n' "$*"; }
step() { printf '\n== %s\n' "$*"; }
need() { NEEDS+=("$*"); say "  NEEDS A HUMAN: $*"; }
fail() { FAILS+=("$*"); say "  FAILED: $*"; }

# ---------------------------------------------------------------- 1. venv
step "1/4 Python environment (.venv, Python 3.11, requirements.lock)"
VENV="$ROOT/.venv"
if [ ! -x "$VENV/bin/python" ]; then
  PYBIN=""
  for c in python3.11 python3; do
    if command -v "$c" >/dev/null 2>&1 && [ "$("$c" -c 'import sys; print("%d.%d" % sys.version_info[:2])' 2>/dev/null)" = "3.11" ]; then PYBIN="$c"; break; fi
  done
  if [ -n "$PYBIN" ]; then "$PYBIN" -m venv "$VENV"
  elif command -v uv >/dev/null 2>&1; then uv venv --seed --python 3.11 "$VENV"
  else fail "Python 3.11 not found (requirements.lock is frozen on 3.11): install python3.11 or uv, then re-run"; fi
fi
if [ -x "$VENV/bin/python" ]; then
  EXTRA=()
  if [ "$(uname -s)" = "Linux" ]; then EXTRA=(--extra-index-url https://download.pytorch.org/whl/cpu); fi   # CPU wheel of torch instead of the CUDA build
  stamp="$(shasum -a 256 requirements.lock | cut -d' ' -f1)"
  if [ "$(cat "$VENV/.onside-stamp" 2>/dev/null)" = "$stamp" ]; then say "  ok: dependencies already installed"
  elif "$VENV/bin/python" -m pip install -q -r requirements.lock ${EXTRA[@]+"${EXTRA[@]}"}; then
    echo "$stamp" > "$VENV/.onside-stamp"; say "  ok: dependencies installed in $VENV"
  else fail "pip install failed (see the output above; requirements.lock was frozen on macOS arm64, Python 3.11)"; fi
fi
PY="$VENV/bin/python"
[ -x "$PY" ] || { say; say "Cannot continue without the environment."; exit 1; }

# ---------------------------------------------------------------- 2. challenge repo
step "2/4 Official challenge repository ($CHALLENGE)"
if [ -f "$CHALLENGE/evaluation/possession_metrics.py" ]; then say "  ok: already there"
elif [ -e "$CHALLENGE" ] && [ -n "$(ls -A "$CHALLENGE" 2>/dev/null)" ]; then fail "$CHALLENGE exists but is not a clone of the challenge repository"
elif git clone -q "$CHALLENGE_REPO" "$CHALLENGE"; then say "  ok: cloned $CHALLENGE_REPO"
else fail "git clone $CHALLENGE_REPO failed (network?)"; fi

# ---------------------------------------------------------------- 3. challenge data
step "3/4 Challenge data ($CHALLENGE/data)"
have_data() { [ -f "$CHALLENGE/data/tactical/BAR-ATH_tactical_atd.txt" ] && [ -f "$CHALLENGE/data/tactical/BAR-ATH_tactical_atd.xml" ] \
  && [ -f "$CHALLENGE/data/tactical/BAR-ATH_tactical_homography.csv" ] && [ -f "$CHALLENGE/data/tactical/BAR-ATH_tactical_smart_tagging.xml" ]; }
if have_data; then say "  ok: already there"
else
  need "the challenge data bundle. Each participant gets it after enrolling at https://landing.mobileworldcapital.com/more-than-a-hack-2027 and accepting the terms of data use. It cannot be redistributed. Unpack it into $CHALLENGE/data keeping the names it arrives with (README.md, 'Challenge data'), then re-run"
fi

# ---------------------------------------------------------------- 4. tests and submission hash
step "4/4 Tests and submission hash"
if [ "$SKIP_TESTS" = 1 ]; then say "  tests skipped (--skip-tests)"
elif ONSIDE_PYTHON="$PY" "$ROOT/run.sh" test; then say "  ok: tests passed (those needing absent data are skipped, see -rs)"
else fail "tests failed"; fi
if [ "$SKIP_VERIFY" = 1 ]; then say "  hash check skipped (--skip-verify)"
elif ! have_data; then say "  hash check not run: challenge data missing"
elif ONSIDE_PYTHON="$PY" "$ROOT/run.sh" verify; then :
else fail "the regenerated CSV does not match scripts/expected_sha256.txt"; fi

# ---------------------------------------------------------------- summary
echo
if [ ${#FAILS[@]} -gt 0 ]; then say "FAILED STEPS:"; for f in "${FAILS[@]}"; do say "  - $f"; done; fi
if [ ${#NEEDS[@]} -gt 0 ]; then say "NEEDS A HUMAN:"; for n in "${NEEDS[@]}"; do say "  - $n"; done; fi
[ ${#FAILS[@]} -gt 0 ] && exit 1
[ ${#NEEDS[@]} -gt 0 ] && exit 3
say "Setup complete."
