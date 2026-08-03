#!/usr/bin/env bash

# TWIST-ANGLE SWEEP on the cubed sphere at a FIXED discretization (order=12, ppf=8), using the on-surface
# STOKES GREEN'S IDENTITY (SL[du/dn] - DL[u] = u for an exterior Stokeslet source) as the accuracy probe.
# For each tolerance (with its paired near-eval Nbeta / max_depth) the sweep runs every twist in
# {0, pi/2, pi} and prints the setup profile + Green's-identity max relative error and node count. Goal:
# see how the SL+DL near quadrature degrades as the twist-induced element shear grows.
#
# Quadrature: HYBRID scheme = RectPolar self + Adaptive near ("rp-self-adaptive-near"), driven by the
# recently updated near/self interaction path (centered graded-u self rule + split-at-foot near). Each
# stokes_greens run warms up Setup(), ClearSetup()s, then profiles a cold Setup() before the identity.
#
# Geometry: BuildTwistedSphere with theta(z) = TWIST*z: total pole-to-pole rotation = 2*TWIST.
#
# order must be a multiple of 4 in {4..48}; SCTL's singular RP rule only exists for
# Nbeta in {48,100,200,300,400,512} and max_depth in {4,8,12,30}. (48,4) is the COARSEST rule, so the
# 1e-3 row reuses it (SCTL's rule floor), differing from 1e-5 only in the tol handed to SetAccuracy.
#
# PARALLELISM: after the (serial) build, each (tol tier x twist) task is a SEPARATE single-threaded
# process (OMP_NUM_THREADS=1 + MKL_NUM_THREADS=1) pinned to its OWN physical core via `taskset -c`. Cores
# are discovered dynamically from `lscpu` and round-robin-assigned; tasks run in parallel in batches of
# NCORES so no core is ever oversubscribed. Per-task output is buffered to a temp log and printed in
# deterministic order after each batch.
#
# Then an OMP THREAD-SCALING sub-study runs a single process at OMP threads {1,2,4,8,16,32} for the 1e-9
# tier at twist=pi, written to a SEPARATE log (out/twisted-sphere-omp-<job>.log) so the main parser is
# unaffected. See the block at the bottom.
#
# Usage:  scripts/twisted_sphere_run.sh

#SBATCH --job-name=twist-sphere-sweep
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --time=01:30:00
#SBATCH --output=out/twisted-sphere-sweep-%j.log
#SBATCH --error=out/twisted-sphere-sweep-%j.log

WORK_DIR=~/quad-junctions
SOURCE_FILE=${WORK_DIR}/sctl_source
cd ${WORK_DIR}
source ${SOURCE_FILE}
set -euo pipefail
mkdir -p out

SCTL_BIN=bin/twisted-sphere-selfsetup
# The vendored include/sctl/experimental/quad_element.{cpp,hpp} are NOT tracked as Make prerequisites
# (the object rule depends only on the driver .cpp), so a fresh checkout with updated near/self
# quadrature would otherwise reuse a stale object. Force a rebuild of this TU so the NEWEST near
# changes are compiled in.
rm -f obj/twisted-sphere-selfsetup.o "$SCTL_BIN"
make "$SCTL_BIN"

COVQ=6
R=1.0

# --- twist angles to sweep (theta(z) = twist * z; total pole-to-pole rotation = 2*twist) ---
TWISTS=(0.0 1.5707963267948966 3.141592653589793)   # 0, pi/2, pi

# --- accuracy-parameter tiers (per tolerance) ---
TOLS=(1e-3 1e-5 1e-7 1e-9 1e-11)
NBS=( 48   48   100  200  400 )       # Nbeta     per tol  (48,4 is the coarsest valid rule)
MDS=( 4    4    8    12   30  )       # max_depth per tol

# --- fixed discretization ---
# order = surface polynomial order per patch (multiple of 4); PPF = PatchPerFace. Node count =
# 6 * PPF^2 * order^2. Held fixed here at order 12 / ppf 8 (the twist-angle degradation, not the
# {ppf,order} plateau, is the variable of interest).
ORDER=12
PPF=8

# --- discover physical cores (one logical CPU per physical core) ---
mapfile -t CORES < <(lscpu -p=CPU,CORE 2>/dev/null | grep -v '^#' | sort -t, -k2 -n -u | cut -d, -f1)
[ "${#CORES[@]}" -eq 0 ] && mapfile -t CORES < <(seq 0 "$(($(nproc) - 1))")
NCORES=${#CORES[@]}
echo "# discovered $NCORES physical cores: ${CORES[*]}"

# --- build the flat job list: one entry ("tol nb md twist") per (tol tier, twist) ---
JOBS=()
for k in "${!TOLS[@]}"; do
  for twist in "${TWISTS[@]}"; do
    JOBS+=("${TOLS[$k]} ${NBS[$k]} ${MDS[$k]} $twist")
  done
done
NJOBS=${#JOBS[@]}
echo "# dispatching $NJOBS tasks (order=$ORDER ppf=$PPF) in parallel batches of $NCORES"

LOGDIR=$(mktemp -d "${TMPDIR:-/tmp}/twist-greens.XXXXXX")
trap 'rm -rf "$LOGDIR"' EXIT

# --- run in batches of NCORES; each task pinned to its own physical core, output printed in order ---
i=0
while [ "$i" -lt "$NJOBS" ]; do
  batch_logs=()
  for ((c = 0; c < NCORES && i < NJOBS; c++, i++)); do
    read -r tol nb md twist <<< "${JOBS[$i]}"
    core=${CORES[$c]}
    log="$LOGDIR/job-$(printf '%03d' "$i").log"
    batch_logs+=("$log")
    {
      echo "########################################################################"
      echo "# core=$core  tol=$tol  (Nbeta=$nb  max_depth=$md)  twist=$twist  order=$ORDER ppf=$PPF"
      echo "########################################################################"
      OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OMP_PROC_BIND=true \
        taskset -c "$core" "$SCTL_BIN" stokes_greens "$ORDER" "$PPF" "$tol" "$nb" "$md" "$COVQ" "$R" "$twist"
    } > "$log" 2>&1 &
  done
  wait || true            # let the sweep continue even if a task exits non-zero (its log still prints)
  for log in "${batch_logs[@]}"; do echo ""; cat "$log"; done
done

# =====================================================================================================
# OMP THREAD-SCALING sub-study: a SINGLE MPI process, OpenMP threads = {1,2,4,8,16,32}, at the 1e-9 tier
# (Nbeta=200 max_depth=12) on the pi-twisted sphere (order 12 / ppf 8 -- same fixed discretization).
# Each run is pinned to the first <nt> discovered physical cores with OMP_NUM_THREADS=nt (thread affinity
# OMP_PLACES=cores / OMP_PROC_BIND=close). These runs are SEQUENTIAL (each wants many cores), unlike the
# parallel twist sweep above. NOTE: the binary links -lmkl_sequential, so MKL never multithreads; the
# scaling comes from SCTL's own OpenMP regions in the near/self setup. MKL_NUM_THREADS is set to match
# only for tidiness. Output goes to its OWN log so it does not perturb parse_twisted_sphere.sh (which
# anchors on '# core=' blocks) -- parse it separately (a '# omp=' header carries the thread count).
# =====================================================================================================
OMP_TWIST=3.141592653589793     # pi
OMP_TOL=1e-9; OMP_NB=200; OMP_MD=12
OMP_THREADS_LIST=(1 2 4 8 16 32)
OMP_LOG="out/twisted-sphere-omp-${SLURM_JOB_ID:-manual}.log"
: > "$OMP_LOG"
echo ""
echo "# ===== OMP thread-scaling sub-study (tol=$OMP_TOL Nbeta=$OMP_NB max_depth=$OMP_MD twist=pi order=$ORDER ppf=$PPF) -> $OMP_LOG ====="
for nt in "${OMP_THREADS_LIST[@]}"; do
  if [ "$nt" -gt "$NCORES" ]; then
    echo "# skip omp=$nt (only $NCORES physical cores available)" | tee -a "$OMP_LOG"
    continue
  fi
  nt_cores=$(IFS=,; echo "${CORES[*]:0:nt}")     # first nt discovered physical cores
  echo "# running omp=$nt on cores {$nt_cores} ..."
  {
    echo "########################################################################"
    echo "# omp=$nt  cores=$nt_cores  tol=$OMP_TOL  (Nbeta=$OMP_NB  max_depth=$OMP_MD)  twist=$OMP_TWIST  order=$ORDER ppf=$PPF"
    echo "########################################################################"
    OMP_NUM_THREADS=$nt MKL_NUM_THREADS=$nt OMP_PLACES=cores OMP_PROC_BIND=close \
      taskset -c "$nt_cores" "$SCTL_BIN" stokes_greens "$ORDER" "$PPF" "$OMP_TOL" "$OMP_NB" "$OMP_MD" "$COVQ" "$R" "$OMP_TWIST"
  } >> "$OMP_LOG" 2>&1
done
echo "# OMP sub-study complete -> $OMP_LOG"
