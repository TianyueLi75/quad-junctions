#!/bin/bash
# LAUNCHER (run on a login node:  bash scripts/submit-vessels-deflation.sh [--dry-run]).
# Submits FIVE Slurm jobs on the SAME production geometry -- the full deflation A/B grid, with NO env vars to
# set by hand (the grid is baked in below):
#   d0         baseline, deflation OFF (the production stall; shared reference)
#   d1         deflation ON, both wall-time levers OFF (matrixfree_deflation.hpp)
#   d1-ulr     + uniform-L/R   (QJ_DEFLATE_UNIFORM_LR=1: every arm block sized from the MEAN aspect)
#   d1-bw      + block-whiten  (QJ_DEFLATE_BLOCK_WHITEN=1: per-arm block-structured whitening)
#   d1-ulr-bw  + both levers
# All five use the DL-only formulation (SL dropped; S-free -1/2 I + D + net-flux completion) by default -- set
# QJ_DL_ONLY=0 to fall back to the combined-field SL_scal=-1 operator. With uniform-LR + block-whiten both on,
# the congruent arm blocks are whitened with ONE shared transform (block 0), not one dsyev per arm.
# All run the cold-start interior Stokes flow solve at PRODUCTION resolution (order 12 / nref 2 / fourier 36 /
# naxial x2), so the [FLOW] RESULT iters + wall-clock lines are directly comparable. Each job: 4 nodes x
# 2 tasks x 32 cpus, ccm/icelake, 8 h.
#
# Deflation targets the longest CONNECTING arms only (the 11 arterial<->venous connectors). Restrict to the
# K longest with  QJ_DEFLATE_NARM=K  below; tune the near-null band with QJ_DEFLATE_THETA (default 0.05).
# Afterwards, compare:  grep -E '\[FLOW\] RESULT|\[deflate\]|\[DEFL\]|\[NETFLUX\] RESULT' out/vessels-defl-*.log
cd ~/quad-junctions
DRY=0; [ "$1" = "--dry-run" ] && DRY=1

if [ $DRY -eq 0 ]; then
    source ./sctl_source
    [ -f extern/pvfmm/lib/.libs/libpvfmm.a ] || { echo "ERROR: build libpvfmm first (CLAUDE.md 'PVFMM build')" >&2; exit 1; }
    # Build ONCE here so both jobs share an identical binary. Objects are not keyed by build flags -> force
    # the source to recompile (-march=native: build on a node of the same CPU family as the icelake nodes).
    # rm -f obj/ybifurc-vessels-flow-bie.o bin/ybifurc-vessels-flow-bie
    # make PVFMM=1 bin/ybifurc-vessels-flow-bie -j || exit 1
fi
mkdir -p out

TOL=1e-9                   # production solve tolerance (gmres_tol = 10*tol); tighten to 1e-11 if desired
ORD=12                     # production y120 order
FOURIER=36                 # production y120 arm azimuthal order
NAX=2                      # naxial multiplier (production)
# Optional: restrict / tune the deflation basis (exported so the job script's defaults are overridden).
export QJ_DEFLATE_NARM=${QJ_DEFLATE_NARM:-0}       # 0 = all 11 connectors; K = K longest connecting arms
export QJ_DEFLATE_THETA=${QJ_DEFLATE_THETA:-0.05}  # near-null band for the per-arm n_max heuristic
export QJ_DEFLATE_FLUX=${QJ_DEFLATE_FLUX:-1}       # append the global flux column

# ---- The A/B/C/D/E grid, submitted automatically (no env vars to set by hand). Five jobs: ----
#   d0           : baseline, deflation OFF (the production stall; the shared reference). Run ONCE -- the two
#                  wall-time levers below only act when deflation is ON, so there is no d0 variant.
#   d1           : deflation ON, both levers OFF (the current 883-iter / 541-col result).
#   d1-ulr       : + uniform-L/R   -- every arm block sized from the MEAN aspect (fewer columns, fewer matvecs).
#   d1-bw        : + block-whiten  -- per-arm block-structured whitening (cheap setup eigendecomposition).
#   d1-ulr-bw    : + both levers together.
# Columns of each row: DEFLATE UNIFORM_LR BLOCK_WHITEN TAG
GRID=(
  # "0 0 0 "
  # "1 0 0 "
  # "1 1 0 -ulr"
  # "1 0 1 -bw"
  "1 1 1 -ulr-bw"
)
TIME=08:00:00
for row in "${GRID[@]}"; do
  read -r DEFLATE ULR BW TAG <<< "$row"
  NAME="vdefl-t${TOL}-o${ORD}-f${FOURIER}-n${NAX}-d${DEFLATE}${TAG}"
  CMD=(sbatch --job-name=${NAME} --time=${TIME} --output=out/${NAME}-%j.log --error=out/${NAME}-%j.log
       --export=ALL,SCAN_TOL=${TOL},SCAN_ORD=${ORD},SCAN_FOURIER=${FOURIER},SCAN_NAX=${NAX},SCAN_DEFLATE=${DEFLATE},QJ_DEFLATE_UNIFORM_LR=${ULR},QJ_DEFLATE_BLOCK_WHITEN=${BW}
       scripts/submit-vessels-deflation-job.sh)
  if [ $DRY -eq 1 ]; then echo "${CMD[@]}"; else "${CMD[@]}"; fi
done
