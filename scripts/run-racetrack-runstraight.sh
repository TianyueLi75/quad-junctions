#!/bin/bash
# A/B the M-stage preconditioner variants on the racetrack (ybifurc-channel-bie flow, lens).
# The wall arm is  lead | corner | run | corner | lead.  Variants:
#   off          no arm precond (baseline)
#   corner       rotated-corner shoulder blocks only (previous M)
#   runstraight  straight-cylinder self-block over ONLY the high-aspect RUN region (this change)
#   cornerstraight  shoulders + straight leads + run (whole wall)
# runstraight is the "straight arm replacement of the arm region between the two corners";
# leads/corners (regular aspect) are left as identity.  Serial. Reports GMRES iters + solve time.
set -u
cd "$(dirname "$0")/.."
. ./sctl_source 2>/dev/null

BIN=./bin/ybifurc-channel-bie
OUT=${OUT:-/tmp/claude-2860/-mnt-home-tli10-quad-junctions/29ea0bdf-f96f-4ca2-9008-4961a19d3f74/scratchpad}
mkdir -p "$OUT"

# coarse-but-real config (matches run-racetrack-two-stage.sh). Override via ARGS=...
ARGS=${ARGS:-"flow 1.5 4 1 0.4 3 0.88 3 12 1e-6 400 30 6 9.6 25 -1 2 6 0 1 1 40 10 10 400 0"}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-8}

run() {  # name  env-assignments...
  local name="$1"; shift
  local log="$OUT/rtrs-$name.log"
  echo "==== $name ====  ($*)"
  env "$@" $BIN $ARGS > "$log" 2>&1
  local it=$(grep -oE "GMRES iters=[0-9]+" "$log" | tail -1 | grep -oE "[0-9]+")
  local tr=$(grep -oE "TRUE rel residual.*= [0-9.eE+-]+" "$log" | tail -1 | grep -oE "[0-9.eE+-]+$")
  local gt=$(grep -E "gmres solve" "$log" | tail -1)
  local pl=$(grep -E "arm preconditioner =" "$log" | tail -1)
  echo "    $pl"
  echo "    iters=$it  true_resid=$tr"
  echo "    $gt"
  echo "$name|iters=$it|true_resid=$tr|$gt" >> "$OUT/rtrs-summary.txt"
}

rm -f "$OUT/rtrs-summary.txt"
run off            QJ_ARM_PRECOND=off
run corner         QJ_ARM_PRECOND=corner
run runstraight    QJ_ARM_PRECOND=runstraight
run cornerstraight QJ_ARM_PRECOND=cornerstraight

echo; echo "==== summary ===="; cat "$OUT/rtrs-summary.txt"
