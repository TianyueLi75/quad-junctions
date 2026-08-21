#!/bin/bash
# =============================================================================
# cilia_near_bench_sweep.sh  --  SINGLE-RANK near-setup attribution sweep (BENCH=1 build)
#
# Diagnostic for the weak-scaling SetupNear efficiency drop. It runs the cilia-carpet driver
# single-rank over an Npatch ladder (the SAME sizes the weak-scaling ladder walks) and reads the
# per-target near-setup breakdown printed by the BENCH=1 build (the "[nearbench]" line):
#   closest-point / subpanel-interp / cell-integrate seconds, mean closest-point iterations,
#   grid-fallback fraction, angle-adjusted-order fraction, mean near GL order, cells/target,
#   elevated-order cell fraction.
#
# WHY single-rank (not MPI): weak scaling grows the GLOBAL carpet (Npatch ~ sqrt(ranks)). The
# weak-scaling drop is driven by the near-setup cost PER UNIT WORK rising as the carpet densifies
# -- a geometry-density effect, NOT an MPI-partition effect. Walking Npatch single-rank isolates
# exactly that: if tClose/cpIt, elevF/mOrd/elevCF, or cellsT climb with Npatch, that stage is the
# weak-scaling culprit. Cheap (no MPI/partition noise), and cross-checked against the real MPI weak
# sweep (submit_cilia_carpet_weak.sh, same [nearbench] line) to rule out a partition artifact.
#
# Geometry matches the best-scaling configuration under study: WIGGLES ON, NO TILT, NO z-plate
# scaling (VSCALE=1, so z_plate/tips are fixed, not shrunk with Npatch).
#
# BUILD (from the repo root; PVFMM is needed for the periodic far field, BENCH for the timers):
#     . ./sctl_source
#     make clean && make PVFMM=1 BENCH=1 bin/cilia_carpet-bie
#   NOTE: object files are not keyed by flags, so `make clean` is REQUIRED when toggling BENCH.
#
# RUN (workstation; override LAUNCH / NPATCH_LIST / OMP_NUM_THREADS as needed):
#     scripts/cilia_near_bench_sweep.sh | tee out/cilia_nearbench_sweep.log
#     scripts/parse_cilia_scaling.sh --csv out/cilia_nearbench_sweep.log > nearbench.csv
#   The parser tabulates the geom(s)/tClose/tSubp/tCell/cpIt/fbFrac/elevF/mOrd/cellsT/elevCF columns.
# =============================================================================
#SBATCH --job-name=cilia_bench_nowiggle
#SBATCH --partition=gen
#SBATCH --constraint=icelake
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --time=01:30:00
#SBATCH --output=out/cilia_bench-%j.log
#SBATCH --error=out/cilia_bench-%j.log

set -euo pipefail

WORK_DIR=~/quad-junctions
cd "${WORK_DIR}"
[ -f ./sctl_source ] && source ./sctl_source || true
mkdir -p vis out

BIN=./bin/cilia_carpet-bie
if [ ! -x "${BIN}" ]; then
  echo "error: ${BIN} not found. Build it first:  make clean && make PVFMM=1 BENCH=1 bin/cilia_carpet-bie" >&2
  exit 1
fi

# --- runtime env (single rank) ----------------------------------------------
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-${SLURM_CPUS_PER_TASK:-16}}   # use the full allocation (64 here) under Slurm
export OMP_PLACES=${OMP_PLACES:-cores}
export OMP_PROC_BIND=${OMP_PROC_BIND:-spread}
export PVFMM_DIR=${PVFMM_DIR:-${WORK_DIR}/extern/pvfmm}   # must EXIST (else pvfmm exit(0)s silently)
export QJ_CILIA_SEED=${QJ_CILIA_SEED:-12345}              # deterministic geometry
export QJ_BOX_BUFFER=${QJ_BOX_BUFFER:-0.01}
# export QJ_CILIA_WIGGLE=${QJ_CILIA_WIGGLE:-1}              # WIGGLES ON (the configuration under study)
export QJ_CILIA_WIGGLE=0

# Launcher: PVFMM forces an MPI build, so run under a 1-rank launcher. `--bind-to none` is LOAD-BEARING:
# OpenMPI otherwise pins a single-rank job to ONE core (default --bind-to core for np<=2), which would
# cram all OMP_NUM_THREADS onto that core and make every timing here meaningless. With --bind-to none the
# OMP threads spread across the allocated cpuset via OMP_PLACES/OMP_PROC_BIND. Works on a workstation and
# inside a Slurm allocation alike; override with LAUNCH="srun --cpu-bind=cores -n1" if you prefer srun.
LAUNCH=${LAUNCH:-mpirun -n 1 --bind-to none}

# --- mesh params (mirror submit_cilia_carpet_weak.sh, NO tilt, NO z-plate scaling: VSCALE=1) ----
ORDER=${ORDER:-12}
TOL=${TOL:-1e-6}
NAZ=${NAZ:-8}
Z_PLATE=${Z_PLATE:-0.01}      # fixed (no VSCALE shrink) -- "no z-plate scaling"
BOT_TIP=${BOT_TIP:-0.48}
TOP_TIP=${TOP_TIP:-0.52}
TILT_DEG=${TILT_DEG:-0}       # NO tilt
PDROP=-1
NVIS=1                        # skip volume vis
FINGERS=1
FOURIER=${FOURIER:-24}
CHEB=${CHEB:-10}
N_AXIAL=${N_AXIAL:-10}

# Npatch ladder = the weak-scaling sizes at BASE_NPATCH=8 (ranks 1,2,4,8,16,32 -> round(8*sqrt(r))).
# Single-rank per-rank work grows with Npatch, which is exactly what exposes rising per-target cost.
# NPATCH_LIST=${NPATCH_LIST:-"6 8 12 17 24 34"}
NPATCH_LIST=${NPATCH_LIST:-"8 17"}

echo "NEARBENCH single-rank sweep  order=${ORDER} tol=${TOL} Naz=${NAZ} fourier=${FOURIER} n_axial=${N_AXIAL}  threads=${OMP_NUM_THREADS}"
echo "  geometry: WIGGLE=${QJ_CILIA_WIGGLE}  tilt=${TILT_DEG}deg  z_plate=${Z_PLATE} (fixed, no VSCALE)  tips=${BOT_TIP}/${TOP_TIP}"
echo "  build must be:  make clean && make PVFMM=1 BENCH=1 bin/cilia_carpet-bie   (else no [nearbench] line)"

for NPATCH in ${NPATCH_LIST}; do
  echo "==================================================================================="
  # A parser-recognized run header: '[weak]' + ranks= + Npatch= delimit each run for
  # parse_cilia_scaling.sh. ranks=1 always here (this is the single-rank density proxy, not an MPI sweep).
  echo "[weak] $(date +%H:%M:%S)  ranks=1  nodes=1  Npatch=${NPATCH}  (single-rank near-setup attribution)"
  # `|| continue` so one failing point (e.g. the largest Npatch OOMs) does NOT let `set -e` abort the
  # whole sweep -- the smaller points already printed their [nearbench] lines and must be kept.
  ${LAUNCH} "${BIN}" "${NPATCH}" "${ORDER}" "${TOL}" "${NAZ}" \
      "${BOT_TIP}" "${TOP_TIP}" "${TILT_DEG}" "${PDROP}" "${NVIS}" "${FINGERS}" \
      "${FOURIER}" "${CHEB}" "${N_AXIAL}" "${Z_PLATE}" \
    || { echo "[weak] SKIP Npatch=${NPATCH} (run exited non-zero -- see log above)"; continue; }
done
