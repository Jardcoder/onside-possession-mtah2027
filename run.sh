#!/usr/bin/env bash
# Entry point of the repository. Paths come from the environment (see src/onside_paths.py and README.md):
#   ONSIDE_PYTHON         interpreter to use; default ./.venv/bin/python (see README.md), then <challenge dir>/.venv/bin/python
#   ONSIDE_CHALLENGE_DIR  clone of the official challenge repository   (default ./challenge)
#   ONSIDE_PFF_DIR        PFF World Cup 2022 files                      (default ./data/pff)
#   ONSIDE_OUT            outputs and caches                            (default ./src/out)
#   ONSIDE_BANKS_DIR      DFL bank and its calibration caches           (default ./bancos/data)
# A `.env` file at the repository root is sourced first when it exists (gitignored).
# Installation and the command that regenerates the submission: see README.md
# Usage: ./run.sh <subcommand> [args...]   (run it without arguments for the list)
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"
if [ -f "$ROOT/.env" ]; then set -a; . "$ROOT/.env"; set +a; fi
CHALLENGE="${ONSIDE_CHALLENGE_DIR:-$ROOT/challenge}"
if [ -n "${ONSIDE_PYTHON:-}" ]; then PY="$ONSIDE_PYTHON"
elif [ -x "$ROOT/.venv/bin/python" ]; then PY="$ROOT/.venv/bin/python"
else PY="$CHALLENGE/.venv/bin/python"; fi
[ -x "$PY" ] || { echo "no interpreter at $PY: create .venv as described in README.md or set ONSIDE_PYTHON" >&2; exit 1; }

cmd_test()            { "$PY" -m pytest tests/ -q "$@"; }
cmd_baseline()        { "$PY" src/reproducir_baseline.py "$@"; }      # official smooth50 baseline: CSV, validation and score on the public window
# --- data
cmd_pff_check()      { "$PY" pff_download/pff_manifest.py --check "$@"; }   # verify the downloaded PFF files against the SHA-256 manifest
cmd_pff_build()      { "$PY" bancos/pff.py --build "$@"; }           # parse the 40 development matches into <ONSIDE_PFF_DIR>/cache (about 17 min, 2.6 GB)
# --- calibration aggregates the T1 pipeline reads
cmd_regimen()         { "$PY" bancos/regimen_reto.py "$@"; }          # detector regime of the challenge vs PFF (tracking only) -> regimen_reto.csv
cmd_calibrar_largo()  { "$PY" bancos/dfl_degradado.py --calibrar-largo "$@"; }   # P(dead | absent) by gap length -> huecos_largo_atd.json (needs the DFL bank)
cmd_verificar_huecos() { "$PY" bancos/verificar_huecos_largo.py "$@"; }  # gaps by length: challenge vs degraded bank
cmd_calibrar_balon()  { "$PY" bancos/calibrar_balon.py "$@"; }        # ball noise: challenge vs bank vs PFF smoothing candidates
# --- PFF development bank
cmd_pff()             { "$PY" bancos/evaluar_pff.py "$@"; }           # accepted ladder B0 -> B3 on the 40 PFF development matches
cmd_toques()          { "$PY" bancos/evaluar_toques.py "$@"; }        # touch rules on the DFL development banks
cmd_equipo()          { "$PY" bancos/evaluar_equipo.py "$@"; }        # team of each deflection on the DFL banks
cmd_ruido_gt()        { "$PY" bancos/ruido_gt_pff.py "$@"; }          # label noise floor: possession edges shifted +-4 frames
cmd_sobrantes()       { "$PY" bancos/diagnostico_pff_sobrantes.py "$@"; }   # extra v2 touches against the event touches
cmd_oraculo_pff()     { "$PY" bancos/oraculo_pff.py "$@"; }           # oracle ceilings on PFF (diagnostic, reads the GT on purpose)
cmd_ronda()           { "$PY" bancos/ronda_pff.py "$@"; }             # rounds 1..4 on PFF, nested 5 folds, entry rule `verdict`
cmd_aerial()          { "$PY" bancos/gt_aerial.py "$@"; }             # label_aerial stress GT against label
# --- model: M1 -> T1 -> lambda -> T1w -> challenge CSV
cmd_modelo()          { "$PY" bancos/modelo_pff.py "$@"; }            # M1 on PFF: LightGBM per frame + Viterbi, nested 5 x 8
cmd_transfer()        { "$PY" bancos/modelo_transfer.py "$@"; }       # T0 / T1: M1 with ball holes calibrated to the challenge (bancos/inyeccion_huecos.py)
cmd_barrido_lambda()  { "$PY" bancos/barrido_lambda.py "$@"; }        # nested lambda sweep of T1 on the out-of-fold probabilities -> lambda_final.json
cmd_ronda5()          { "$PY" bancos/ronda5_t1.py "$@"; }             # T1w / T1wm: per-frame weights; --lam-ha ... [--lam-d ...] writes lambda_final_<model>.json
cmd_dominio()         { "$PY" bancos/dominio.py "$@"; }               # domain shift of the T1 features, challenge vs PFF
cmd_diag_t1()         { "$PY" bancos/diagnostico_t1.py "$@"; }        # where T1 loses spellF1 on the 40 development matches
cmd_predecir_reto_m1() { "$PY" src/predecir_reto_m1.py "$@"; }        # challenge CSV; --modelo T1w (default)|T1|T1wm, --salida PATH, --solo-csv, --cache
cmd_bordes()          { "$PY" bancos/ronda_bordes.py "$@"; }          # border detector on top of T1w: nested evaluation against T1w, final weights -> models/
cmd_predecir_reto_m2() { "$PY" src/predecir_reto_m2.py "$@"; }        # challenge CSV with T1w + border detector; --salida PATH, --lam HA D, --modelos DIR
cmd_ablacion_presencia() { "$PY" bancos/ablacion_presencia.py "$@"; } # border detector without / with the ball-presence columns, paired (PFF only)
cmd_all() { cmd_test; cmd_baseline; }

SUBS="test|baseline|pff-check|pff-build|regimen|calibrar-largo|verificar-huecos|calibrar-balon|pff|toques|equipo|ruido-gt|sobrantes|oraculo-pff|ronda|aerial|modelo|transfer|barrido-lambda|ronda5|dominio|diag-t1|predecir-reto-m1|bordes|predecir-reto-m2|ablacion-presencia|all"
sub="${1:-}"
[ $# -gt 0 ] && shift
case "$sub" in
  test)             cmd_test "$@" ;;
  baseline)         cmd_baseline "$@" ;;
  pff-check)        cmd_pff_check "$@" ;;
  pff-build)        cmd_pff_build "$@" ;;
  regimen)          cmd_regimen "$@" ;;
  calibrar-largo)   cmd_calibrar_largo "$@" ;;
  verificar-huecos) cmd_verificar_huecos "$@" ;;
  calibrar-balon)   cmd_calibrar_balon "$@" ;;
  pff)              cmd_pff "$@" ;;
  toques)           cmd_toques "$@" ;;
  equipo)           cmd_equipo "$@" ;;
  ruido-gt)         cmd_ruido_gt "$@" ;;
  sobrantes)        cmd_sobrantes "$@" ;;
  oraculo-pff)      cmd_oraculo_pff "$@" ;;
  ronda)            cmd_ronda "$@" ;;
  aerial)           cmd_aerial "$@" ;;
  modelo)           cmd_modelo "$@" ;;
  transfer)         cmd_transfer "$@" ;;
  barrido-lambda)   cmd_barrido_lambda "$@" ;;
  ronda5)           cmd_ronda5 "$@" ;;
  dominio)          cmd_dominio "$@" ;;
  diag-t1)          cmd_diag_t1 "$@" ;;
  predecir-reto-m1) cmd_predecir_reto_m1 "$@" ;;
  bordes)           cmd_bordes "$@" ;;
  predecir-reto-m2) cmd_predecir_reto_m2 "$@" ;;
  ablacion-presencia) cmd_ablacion_presencia "$@" ;;
  all)              cmd_all ;;
  *) echo "usage: $0 {$SUBS}" >&2; exit 2 ;;
esac
