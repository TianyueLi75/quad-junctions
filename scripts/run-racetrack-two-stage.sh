#!/bin/bash
# Two-stage LEFT preconditioner A/B on the racetrack (ybifurc-channel-bie flow, lens racetrack).
# Cells: baseline off, M only (stage-1 arm block), D only (per-panel block-Jacobi), and the two
# composites mult / add. M here = the ACTUAL-geometry rotated-corner block (QJ_ARM_PRECOND=corner):
# the straight whole-arm proxy (QJ_ARM_PRECOND=whole) is a wrong model for the BENT racetrack walls and
# its 15840-DOF SVD is ~2h under sequential MKL (see multiarm-arm-precond-racetrack-vessels memory).
# Serial (QJ_JACOBI=block builds Comm::Self sub-Ops). Reports GMRES iters + gmres-solve wall time.
set -u
cd "$(dirname "$0")/.."
. ./sctl_source 2>/dev/null

BIN=./bin/ybifurc-channel-bie
OUT=${OUT:-/tmp/claude-2860/-mnt-home-tli10-quad-junctions/29ea0bdf-f96f-4ca2-9008-4961a19d3f74/scratchpad}
mkdir -p "$OUT"

# coarse-but-real config: order4 nref1 fourier12 tol1e-6, small viz grid. Override via ARGS=...
ARGS=${ARGS:-"flow 1.5 4 1 0.4 3 0.88 3 12 1e-6 400 30 6 9.6 25 -1 2 6 0 1 1 40 10 10 400 0"}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-8}

run() {  # name  env-assignments...
  local name="$1"; shift
  local log="$OUT/rt2s-$name.log"
  echo "==== $name ====  ($*)"
  env "$@" $BIN $ARGS > "$log" 2>&1
  local it=$(grep -oE "GMRES iters=[0-9]+" "$log" | tail -1 | grep -oE "[0-9]+")
  local tr=$(grep -oE "TRUE rel residual.*= [0-9.eE+-]+" "$log" | tail -1 | grep -oE "[0-9.eE+-]+$")
  local gt=$(grep -E "gmres solve" "$log" | tail -1)
  echo "    iters=$it  true_resid=$tr"
  echo "    $gt"
  echo "$name|iters=$it|true_resid=$tr|$gt" >> "$OUT/rt2s-summary.txt"
}

rm -f "$OUT/rt2s-summary.txt"
run off        QJ_ARM_PRECOND=off
run M_only     QJ_ARM_PRECOND=corner
run D_only     QJ_ARM_PRECOND=off    QJ_JACOBI=block
run mult       QJ_ARM_PRECOND=corner QJ_JACOBI=block QJ_JACOBI_COMPOSE=mult
run add        QJ_ARM_PRECOND=corner QJ_JACOBI=block QJ_JACOBI_COMPOSE=add

echo; echo "==== summary ===="; cat "$OUT/rt2s-summary.txt"
