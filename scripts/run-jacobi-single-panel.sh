#!/bin/bash
# A/B for the single-panel arm block-Jacobi D on the capsule flow solve.
#
# Compares {no precond, D-only, M-then-D (mult)} x {arm=single reused panel, arm=per-panel distinct}
# for iteration count, GMRES time/iter, and D precompute time. Junctions/caps stay per-element.
#
# Usage: scripts/run-jacobi-single-panel.sh [order R_shaft length n_axial fourier Ncap tol]
set -u
. ./sctl_source >/dev/null 2>&1

ORD=${1:-12}; R=${2:-0.3}; L=${3:-10}; NA=${4:-8}; F=${5:-16}; NCAP=${6:-2}; TOL=${7:-1e-10}
BIN=./bin/capsule-robin-bie
ARGS="flow $ORD $R $L $NA $F $NCAP $TOL"
OUT=/tmp/claude-2860/-mnt-home-tli10-quad-junctions/jacobi-single
mkdir -p "$OUT"
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-8}
export OMP_STACKSIZE=256M
# Cold-start the single-panel straight-proxy arm block so its precompute is measured, not warm-loaded.
rm -f data/precond-cache/aprecond*armwhole* 2>/dev/null

echo "== capsule flow  order=$ORD R=$R L=$L n_axial=$NA fourier=$F tol=$TOL  aspect=$(echo "$L/$R"|bc -l) =="
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
run "D-only  (arm=single)"  QJ_ARM_PRECOND=off   QJ_JACOBI=block QJ_JACOBI_ARM=single
run "D-only  (arm=full)"    QJ_ARM_PRECOND=off   QJ_JACOBI=block QJ_JACOBI_ARM=full
run "M+D mult (arm=single)" QJ_ARM_PRECOND=whole QJ_JACOBI=block QJ_JACOBI_ARM=single QJ_JACOBI_COMPOSE=mult
run "M+D mult (arm=full)"   QJ_ARM_PRECOND=whole QJ_JACOBI=block QJ_JACOBI_ARM=full   QJ_JACOBI_COMPOSE=mult
echo "logs in $OUT"
