#!/usr/bin/env bash

# Head-to-head Laplace SL self+near SETUP timing, single-core:
#   (A) 2021 fmm3dbie getnearquad_lap_comb_dir (iquadtype=1 = GGQ self + adaptive near, dpars=(1,0) SLP)
#       on the paper's own stellarator (igeomtype=2, iasp=3/iref=2 -> N=2400 patches, p=8, 86400 nodes).
#   (B) SCTL hybrid Y-bifurcation (QuadElemList junction + CSBQ SlenderElemList arms) cold Setup().
#   (C) SCTL twisted cubed sphere (theta(z)=twist*z, twist=pi/3), order 12 / ppf 4, cold Setup().
# Both/all sweep eps/tol over 1e-3,1e-5,1e-7,1e-9,1e-11. Reported metric = Nnodes / (self+near time):
#   fmm3dbie t_slp (excludes the near-list findnear~BuildNearLst and get_far_order~SetupFarField,
#   both built once before the timer); SCTL SetupSingular+SetupNear rows of the profile.
# (C) is also runnable standalone via scripts/twisted_sphere_run.sh.
#
# Single-core enforcement = OMP_NUM_THREADS=1 + MKL_NUM_THREADS=1 (no OpenMP/MKL threads)
#                           + taskset -c $CORE (pin the whole process to one physical core).
#
# Usage:  scripts/fmm3dbie_run.sh 

#SBATCH --job-name=fmm3dbie
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --time=01:30:00
#SBATCH --output=out/fmm3dbie-%j.log
#SBATCH --error=out/fmm3dbie-%j.log

WORK_DIR=~/quad-junctions
SOURCE_FILE=${WORK_DIR}/sctl_source
cd ${WORK_DIR}
source ${SOURCE_FILE}
set -euo pipefail
mkdir -p out

# --- assign one core to each single-threaded section, from the cores actually AVAILABLE to this process
# (its CPU-affinity mask, /proc/self/status Cpus_allowed_list). Under Slurm this is EXACTLY the allocated
# cpuset; on a bare workstation it is all cores. Using the affinity mask -- not lscpu, which reports the
# whole node -- guarantees every taskset target is inside the allocation, so pinning can never fail with
# EINVAL when Slurm hands out high-numbered cores. Sections A/B/C run SEQUENTIALLY, each pinned via
# taskset; distinct cores just give clean migration-free timing, and with fewer than 3 available they
# wrap+reuse (harmless -- never two at once). The first available core is skipped only when there is slack
# (>=4 cores), to keep OS/interrupt noise off the timed core.
CORES=()
_avail=$(grep -i '^Cpus_allowed_list' /proc/self/status 2>/dev/null | awk '{print $2}')
if [ -n "$_avail" ]; then
  IFS=',' read -ra _rng <<< "$_avail"
  for r in "${_rng[@]}"; do
    if [[ $r == *-* ]]; then CORES+=( $(seq "${r%-*}" "${r#*-}") ); else CORES+=( "$r" ); fi
  done
fi
[ "${#CORES[@]}" -eq 0 ] && mapfile -t CORES < <(seq 0 "$(($(nproc) - 1))")   # fallback: all logical CPUs
NCORES=${#CORES[@]}
CORE_BASE=$([ "$NCORES" -ge 4 ] && echo 1 || echo 0)
COREFMM=${CORES[$((  CORE_BASE      % NCORES ))]}
COREYBIF=${CORES[$(( (CORE_BASE+1) % NCORES ))]}
CORETWIST=${CORES[$(( (CORE_BASE+2) % NCORES ))]}
echo "# available cores ($NCORES: ${CORES[*]}) -> COREFMM=$COREFMM COREYBIF=$COREYBIF CORETWIST=$CORETWIST"

SRC=bench/fmm3dbie/aquad_lap_stell_perf_test.f   # our harness (kept OUT of the vendored 2021 tree)
BIN=bench/fmm3dbie/aquad_lap_2021                 # build product (also outside the vendored tree)

# --- build against the paired 2021 int4 static libs (fmm3dbie + FMM3D) ---
# default-integer = int4, so it MUST link the int4 FMM3D-2021 lib; mixing with the int8
# current FMM3D segfaults in pts_tree3d. -std=legacy -O3 -march=native matches the 2021 .make.
# The vendored extern/fmm3dbie-2021 tree is left pristine (only its static lib is linked).
MKL="-L$MKLROOT/lib/intel64 -lmkl_gf_lp64 -lmkl_sequential -lmkl_core -lpthread -lm -ldl"
gfortran -fPIC -O3 -funroll-loops -march=native -fopenmp -std=legacy \
  -o "$BIN" "$SRC" \
  extern/fmm3dbie-2021/lib-static/libfmm3dbie.a \
  extern/FMM3D-2021/lib-static/libfmm3d.a \
  $MKL -lgomp -lgfortran

# --- run single-threaded, pinned to ONE core ---
#   OMP_NUM_THREADS=1  -> getnearquad's parallel loops run serial
#   MKL_NUM_THREADS=1  -> no MKL/BLAS threads
#   OMP_PROC_BIND=true -> the (single) thread does not migrate
# COREFMM assigned above from the discovered cores.
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OMP_PROC_BIND=true \
taskset -c $COREFMM "$BIN"

# ============================================================================
# (B) SCTL hybrid Y-bifurcation Laplace SL setup timing (the head-to-head number)
# ============================================================================
# QuadElemList junction + CSBQ SlenderElemList arms in ONE BoundaryIntegralOp; sets the on-surface
# targets, clears setup, then times a single COLD Setup(). From each printed profile read the
# SetupSingular (self) + SetupNear rows; speed = Nnodes / (SetupSingular + SetupNear), the same
# self+near metric as fmm3dbie's t_slp above. order 8 / nref 2 matches the fmm3dbie p=8 mesh.
# Per-tolerance (Nbeta, max_depth) are the project's canonical near-eval parameter sets. SCTL's
# singular RP rule only exists for Nbeta in {48,100,200,300,400,512} and max_depth in {4,8,12,30};
# (48,4) is the COARSEST rule, so the 1e-3 row reuses it (SCTL's rule floor -- the analog of
# fmm3dbie's discrete-tier tie), differing from 1e-5 only in the tol handed to SetAccuracy.
SCTL_BIN=bin/ybifurc-bie-selfsetup
make "$SCTL_BIN"                      

# COREYBIF assigned above from the discovered cores.
ORDER=8; NREF=2; COVQ=6; FOUR=36
TOLS=(1e-3 1e-5 1e-7 1e-9 1e-11)
NBS=( 48   48   100  200  400 )       # Nbeta  per tol  (48,4 is the coarsest valid rule)
MDS=( 4    4    8    12   30  )       # max_depth per tol
for k in "${!TOLS[@]}"; do
  tol=${TOLS[$k]}; nb=${NBS[$k]}; md=${MDS[$k]}
  echo "## SCTL hybrid Laplace SL  order=$ORDER nref=$NREF tol=$tol (Nbeta=$nb max_depth=$md)"
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OMP_PROC_BIND=true \
    taskset -c $COREYBIF "$SCTL_BIN" laplace $ORDER $NREF $tol $nb $md $COVQ $FOUR
done

# ============================================================================
# (C) SCTL twisted cubed-sphere Laplace SL setup timing
# ============================================================================
# Cubed sphere with a height-dependent twist theta(z)=TWIST*z (BuildTwistedSphere); same self+near
# cold-Setup() measurement as (B), different geometry. order 12 / ppf 4, twist=pi/3 (the largest twist
# that still preserves DL Laplace accuracy). Reuses the TOLS/NBS/MDS
# arrays from (B) (same tol sweep + rule-floor 1e-3). Also runnable via scripts/twisted_sphere_run.sh.
TWIST_BIN=bin/twisted-sphere-selfsetup
make "$TWIST_BIN"                      # one-off; up-to-date -> instant

# CORETWIST assigned above from the discovered cores.
TORDER=12; PPF=10; TR=1.0; TWIST=1.0471975511965976   # pi/3 -- largest twist that preserves DL accuracy
for k in "${!TOLS[@]}"; do
  tol=${TOLS[$k]}; nb=${NBS[$k]}; md=${MDS[$k]}
  echo "## SCTL twisted-sphere Laplace SL  order=$TORDER ppf=$PPF twist=$TWIST tol=$tol (Nbeta=$nb max_depth=$md)"
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OMP_PROC_BIND=true \
    taskset -c $CORETWIST "$TWIST_BIN" laplace $TORDER $PPF $tol $nb $md $COVQ $TR $TWIST
done
