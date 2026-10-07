#!/bin/bash
# A/B for the single-panel arm block-Jacobi D on the RACETRACK (ybifurc-channel-bie flow, lens mode).
#
# Same study as the capsule: {no precond, D-only, M+D mult} x {arm=single reused panel, arm=per-panel
# distinct} for iters, GMRES time/iter, and D precompute time (split cap/junction vs arm). The racetrack
# arm panels are BENT (walls) so the single reused block is built from the ACTUAL middle panel (the driver
# supplies no panel_len/axis), unlike the exact straight capsule arm -- so this also tests whether ignoring
# per-panel wall rotation degrades the D preconditioner. Stage-1 M on the racetrack = rotated-CORNER block
# (QJ_ARM_PRECOND=corner), since the straight whole-arm proxy HURTS the racetrack.
#
# Usage: scripts/run-jacobi-single-panel-racetrack.sh [level ord nref eta NsT s_cap nAx fourier tol]
set -u
. ./sctl_source >/dev/null 2>&1

LVL=${1:-1.5}; ORD=${2:-4}; NREF=${3:-1}; ETA=${4:-0.4}; NST=${5:-3}; SCAP=${6:-0.88}; NAX=${7:-3}; F=${8:-12}; TOL=${9:-1e-6}
BIN=./bin/ybifurc-channel-bie
ARGS="flow $LVL $ORD $NREF $ETA $NST $SCAP $NAX $F $TOL"
OUT=/tmp/claude-2860/-mnt-home-tli10-quad-junctions/jacobi-racetrack
mkdir -p "$OUT"
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-8}
export OMP_STACKSIZE=256M
# Cold-start any cached single-panel straight-proxy arm block (racetrack uses actual-panel, not cached, but be safe).
rm -f data/precond-cache/aprecond*armwhole* 2>/dev/null

echo "== racetrack (lens) flow  level=$LVL ord=$ORD nref=$NREF fourier=$F tol=$TOL =="
printf "%-34s %6s %10s %12s %10s %8s %8s %12s\n" \
  "config" "iters" "solve(s)" "t/iter(s)" "Dbuild" "Dcap" "Darm" "true_resid"

run () {
  local label="$1"; shift
  local log="$OUT/$(echo "$label" | tr ' /()' '____').log"
  env "$@" $BIN $ARGS >"$log" 2>&1
  local it so ti db dc da tr
  it=$(grep -oP 'GMRES iters=\K[0-9]+' "$log" | tail -1)
  so=$(grep -oP 'solve=\K[0-9.eE+-]+' "$log" | tail -1)
  ti=$(grep -oP 'time/iter=\K[0-9.eE+-]+' "$log" | tail -1)
  db=$(grep -oP 'jacobi-D build \(precompute\) = \K[0-9.eE+-]+' "$log" | tail -1)
  dc=$(grep -oP 'cap/junction = \K[0-9.eE+-]+' "$log" | tail -1)
  da=$(grep -oP 'arm = \K[0-9.eE+-]+' "$log" | tail -1)
  tr=$(grep -oP 'TRUE rel residual.*= \K[0-9.eE+-]+' "$log" | tail -1)
  printf "%-34s %6s %10s %12s %10s %8s %8s %12s\n" \
    "$label" "${it:-?}" "${so:-?}" "${ti:-?}" "${db:--}" "${dc:--}" "${da:--}" "${tr:-?}"
}

run "no-precond"            QJ_ARM_PRECOND=off
run "D-only  (arm=single)"  QJ_ARM_PRECOND=off    QJ_JACOBI=block QJ_JACOBI_ARM=single
run "D-only  (arm=full)"    QJ_ARM_PRECOND=off    QJ_JACOBI=block QJ_JACOBI_ARM=full
run "M+D mult (arm=single)" QJ_ARM_PRECOND=corner QJ_JACOBI=block QJ_JACOBI_ARM=single QJ_JACOBI_COMPOSE=mult
run "M+D mult (arm=full)"   QJ_ARM_PRECOND=corner QJ_JACOBI=block QJ_JACOBI_ARM=full   QJ_JACOBI_COMPOSE=mult
echo "logs in $OUT"
