#!/bin/bash
# Racetrack (ybifurc-channel-bie flow, lens) study of WHICH cap/junction quad blocks D preconditions, with
# the blocks REUSED by congruence class (the 2 placed junctions are rigid copies, each 3-fold symmetric,
# the stem caps are copies -- measured 520 elems -> 41 proper-rotation classes, worst Kabsch resid ~5e-14).
#
# QJ_JACOBI_QUAD = all (cap+junction) | junction (bodies only, skip the QJ_JACOBI_NCAP trailing stem caps)
#                | off (arm-only D).  QJ_JACOBI_QUAD_REUSE=0 disables reuse (distinct per-element A/B).
# Arm is arm=single (rotated-per-panel; == full). Stage-1 M on M+D configs = corner.
#
# n_cap = 40 stem-cap elements for THIS config (order4 nref1: 520 quad elems, last 40 are the 2 stem caps),
# measured from a QJ_DUMP_QUAD dump. Change it if you change order/nref/Ncap.
#
# Usage: scripts/run-jacobi-quad-reuse-racetrack.sh [level ord nref eta NsT s_cap nAx fourier tol ncap]
set -u
. ./sctl_source >/dev/null 2>&1

LVL=${1:-1.5}; ORD=${2:-4}; NREF=${3:-1}; ETA=${4:-0.4}; NST=${5:-3}; SCAP=${6:-0.88}; NAX=${7:-3}; F=${8:-12}; TOL=${9:-1e-6}; NCAP=${10:-40}
BIN=./bin/ybifurc-channel-bie
ARGS="flow $LVL $ORD $NREF $ETA $NST $SCAP $NAX $F $TOL"
OUT=/tmp/claude-2860/-mnt-home-tli10-quad-junctions/jacobi-quad-reuse
mkdir -p "$OUT"
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-8}
export OMP_STACKSIZE=256M

echo "== racetrack (lens) flow  level=$LVL ord=$ORD nref=$NREF fourier=$F tol=$TOL  n_cap=$NCAP =="
printf "%-40s %6s %10s %11s %10s %8s %8s %8s\n" \
  "config" "iters" "solve(s)" "Dbuild(s)" "Dcap(s)" "Darm(s)" "nclass" "resid"

run () {
  local label="$1"; shift
  local log="$OUT/$(echo "$label" | tr ' /()=+' '______').log"
  env "$@" $BIN $ARGS >"$log" 2>&1
  local it so db dc da nc tr
  it=$(grep -oP 'GMRES iters=\K[0-9]+' "$log" | tail -1)
  so=$(grep -oP 'solve=\K[0-9.eE+-]+' "$log" | tail -1)
  db=$(grep -oP 'jacobi-D build \(precompute\) = \K[0-9.eE+-]+' "$log" | tail -1)
  dc=$(grep -oP 'cap/junction = \K[0-9.eE+-]+' "$log" | tail -1)
  da=$(grep -oP 'jacobi D build split.*arm = \K[0-9.eE+-]+' "$log" | tail -1)
  nc=$(grep -oP 'quadpanel: [0-9]+ element block\(s\) via \K[0-9]+' "$log" | tail -1)
  [ -z "$nc" ] && nc=$(grep -oP 'quadpanel: \K[0-9]+(?= element block\(s\), )' "$log" | tail -1)
  tr=$(grep -oP 'TRUE rel residual.*= \K[0-9.eE+-]+' "$log" | tail -1)
  printf "%-40s %6s %10s %11s %10s %8s %8s %8s\n" \
    "$label" "${it:-?}" "${so:-?}" "${db:--}" "${dc:--}" "${da:--}" "${nc:--}" "${tr:-?}"
}

run "no-precond"                          QJ_ARM_PRECOND=off
echo "--- D-only (arm=single) ---"
run "D all cap+junction (reused)"         QJ_ARM_PRECOND=off QJ_JACOBI=block QJ_JACOBI_ARM=single QJ_JACOBI_QUAD=all
run "D all cap+junction (NO reuse)"       QJ_ARM_PRECOND=off QJ_JACOBI=block QJ_JACOBI_ARM=single QJ_JACOBI_QUAD=all QJ_JACOBI_QUAD_REUSE=0
run "D junction-only (reused)"            QJ_ARM_PRECOND=off QJ_JACOBI=block QJ_JACOBI_ARM=single QJ_JACOBI_QUAD=junction QJ_JACOBI_NCAP=$NCAP
run "D arm-only (no cap/junction)"        QJ_ARM_PRECOND=off QJ_JACOBI=block QJ_JACOBI_ARM=single QJ_JACOBI_QUAD=off
echo "--- M+D mult (arm=single, M=corner) ---"
run "M+D all cap+junction (reused)"       QJ_ARM_PRECOND=corner QJ_JACOBI=block QJ_JACOBI_ARM=single QJ_JACOBI_QUAD=all      QJ_JACOBI_COMPOSE=mult
run "M+D junction-only (reused)"          QJ_ARM_PRECOND=corner QJ_JACOBI=block QJ_JACOBI_ARM=single QJ_JACOBI_QUAD=junction QJ_JACOBI_NCAP=$NCAP QJ_JACOBI_COMPOSE=mult
run "M+D arm-only (no cap/junction)"      QJ_ARM_PRECOND=corner QJ_JACOBI=block QJ_JACOBI_ARM=single QJ_JACOBI_QUAD=off      QJ_JACOBI_COMPOSE=mult
echo "logs in $OUT"
