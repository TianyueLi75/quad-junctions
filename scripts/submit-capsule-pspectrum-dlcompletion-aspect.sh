#!/bin/bash
#SBATCH --job-name=capsule-pspec-dlcomp
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --partition=gen
#SBATCH --constraint=icelake
#SBATCH --time=08:00:00
#SBATCH --mem=96G
#SBATCH --output=vis/capsule-pspectrum/comp-aspect/slurm-dlcomp-%j.out
# =============================================================================
# submit-capsule-pspectrum-dlcompletion-aspect.sh
#
# S-FREE interior-DL eigenvalue spectrum of the STRAIGHT capsule, across an aspect sweep. By DEFAULT this
# runs ONLY the FLUX-ONLY completion (the correct S-free operator); the un-completed DL-ONLY baseline is
# left as a commented-out block below (re-enable it, and re-add "straight-dlonly" to the plot loop, to
# regenerate it). Both variants DROP the single layer (QJ_DL_COMPLETION=1 assembles the pure double layer
# -1/2 I + D); the rank-1 flux completion is toggled via QJ_DL_COMP_WFLUX (include/quad_junctions/
# dl_completion.hpp):
#     FLUX-ONLY (rank-1, DEFAULT) : A = -1/2 I + D + w_flux (flux proj)             [gtag straight-dlflux]
#     DL-ONLY   (no completion)   : A = -1/2 I + D                                  [gtag straight-dlonly, disabled]
# (The earlier rank-7 flux+6-RBM completion was REMOVED 2026-09-24: the 6 rigid modes are not near-null
#  for the interior direct DL -- they live at the accumulation point λ=-1 -- so the RBM completion drove
#  them onto the origin and made the operator worse. The correct S-free operator is dlflux. See the
#  dl-completion memory.)
# The three aspect subdirs (ar17_dl / ar167_dl / ar1670_dl) hold both variants -- the gtag keeps the
# CSVs/modes from colliding. Parallel to the S-kept net-flux sweep (ar17 / ar167 / ar1670).
#
#   ar17_dl   : R=0.3   L=5  n_axial=6   aspect L/R = 16.7  (base)
#   ar167_dl  : R=0.03  L=5  n_axial=6   aspect L/R = 167   (0.1x R)
#   ar1670_dl : R=0.003 L=5  n_axial=6   aspect L/R ~ 1670  (0.01x R)
# Fourier + order fixed across the sweep; N=1920 each (seconds per dense eig/SVD).
#
# The job runs the SAME eigenvalue + eigenvector visualizations for each active variant across the
# aspect ratios at the end. Figures land in comp-aspect/:
#   pspectrum-{dlflux,dlonly}-eig.png            (complex-plane eigenvalue clustering per aspect)
#   pspectrum-{dlflux,dlonly}-nearzero-tail.png  (near-null decay vs aspect ratio)
#   <arNN_dl>/pmodes-<arNN_dl>-{dlflux,dlonly}-{surface,surface-energy,flow,flow-energy}.png
#   pmodes-{dlflux,dlonly}-aspect-montage.png    (mode-0 surface / surface energy / flow energy vs aspect)
#
# RESOURCE MODEL: serial process (dense assembly + SVD/dgeev assert comm.Size()==1), threaded MKL across
# the node. QJ_PSPECTRUM_NEG=0 (only the "none" variant is studied) halves the dgeev cost.
# =============================================================================
set -uo pipefail

root="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/.." && pwd)}"
cd "$root"
while [ ! -f sctl_source ] && [ "$PWD" != "/" ]; do cd ..; done
[ -f sctl_source ] || { echo "ERROR: cannot locate project root (no sctl_source found from '$root')"; exit 1; }
echo "project root: $PWD"

source ./sctl_source

BASE=vis/capsule-pspectrum/comp-aspect
mkdir -p "$BASE"

# ---- build the OBJECT (serial) then RELINK with threaded MKL for fast dense SVD/dgeev -------------
# Object files are not keyed by build flags; clean so a prior MPI object cannot leak in. Request the
# .o as an explicit goal (the bin rule deletes it as an intermediate otherwise). Fail loudly.
OBJ=obj/capsule-robin-bie.o
make clean >/dev/null 2>&1 || true
make "$OBJ" || { echo "ERROR: object build failed"; exit 1; }
[ -f "$OBJ" ] || { echo "ERROR: $OBJ missing after build"; exit 1; }

BIN=./bin/capsule-robin-bie-pspec
g++ "$OBJ" -o "$BIN" \
  -std=c++17 -fopenmp -O3 -march=native -DNDEBUG -rdynamic -ldl -mno-avx512fp16 \
  -DSCTL_GLOBAL_MEM_BUFF=0 -DSCTL_QUAD_T=__float128 -DSCTL_DATA_PATH=./data \
  -lmkl_intel_lp64 -lmkl_gnu_thread -lmkl_core -lgomp -lpthread -DSCTL_HAVE_BLAS -DSCTL_HAVE_LAPACK \
  -lfftw3_omp -lfftw3 -lfftw3f -lfftw3l \
  || { echo "ERROR: threaded-MKL relink failed"; exit 1; }
echo "relinked (threaded MKL): $BIN"

# ---- fixed mesh params (Fourier + order held across all runs) ------------------------------------
ORDER=${ORDER:-4}
FOURIER=${FOURIER:-8}
NCAP=${NCAP:-1}
L=${L:-5}
TOL=${TOL:-1e-6}

# ---- S-free DL completion + eig knobs ------------------------------------------------------------
# S is dropped in BOTH variants below (QJ_DL_COMPLETION=1 assembles the pure double layer -1/2 I + D);
# the rank-1 flux completion term is toggled via QJ_DL_COMP_WFLUX, which also selects the gtag the outputs
# are written under ("-dlflux" flux, "-dlonly" none).
export QJ_DL_COMPLETION=1
export QJ_PSPECTRUM_NEG=0             # only the "none" variant is plotted -> skip the caparmneg dgeev
export QJ_SPECTRUM_MAXDOF=${QJ_SPECTRUM_MAXDOF:-16000}
export QJ_PSPECTRUM_MODES=${QJ_PSPECTRUM_MODES:-5}   # VISUALIZE the 5 smallest-|lambda| modes per aspect.
# Surface/flow ENERGY aggregates sum over ALL near-null modes (count grows with aspect: ~1/14/54 at a.r.
# 17/167/1670). write_nearnull_basis auto-grows dumped eigvecs+flow to max(QJ_PSPECTRUM_MODES, n_null) so the
# energy sum is complete; only 5 are plotted per-mode. Threshold matches plot_pspectrum_dl.py nearzero-tail:
export QJ_PSPECTRUM_NULL_THRESH=${QJ_PSPECTRUM_NULL_THRESH:-1e-2}

# ---- threading: one process, all allocated cores -------------------------------------------------
NCORES=${SLURM_CPUS_PER_TASK:-64}
export OMP_NUM_THREADS="$NCORES" MKL_NUM_THREADS="$NCORES" OMP_PROC_BIND=close OMP_PLACES=cores
export OMP_STACKSIZE=256M

echo "=== capsule pspectrum S-free DL completion sweep : order=$ORDER fourier=$FOURIER Ncap=$NCAP L=$L tol=$TOL ==="
echo "    threads=$NCORES  QJ_SPECTRUM_MAXDOF=$QJ_SPECTRUM_MAXDOF  (dl_completion=1, neg=0, modes=$QJ_PSPECTRUM_MODES)"

# run() : $1=subdir  $2=R_shaft  $3=n_axial  [$4=length, defaults to global $L]
run() {
  local sub="$1" R="$2" NA="$3" LEN="${4:-$L}"
  local out="$BASE/$sub"; mkdir -p "$out"
  local log="$out/run.log"
  echo ">>> $sub : R=$R L=$LEN n_axial=$NA (aspect=$(python3 -c "print(f'{$LEN/$R:.1f}')")) -> $log"
  env QJ_PSPECTRUM_OUTDIR="$out" \
      "$BIN" pspectrum "$ORDER" "$R" "$LEN" "$NA" "$FOURIER" "$NCAP" "$TOL" > "$log" 2>&1
  grep -E "pspectrum\] int/|DOF N=|completion|===" "$log" | sed 's/^/    /'
}

# =============================================================================
# The three aspect ratios (ar17_dl / ar167_dl / ar1670_dl) share one subdir set; each operator VARIANT
# writes under its own gtag so they never collide.
# =============================================================================

# ---- FLUX-ONLY completion: A = (-1/2 I + D) + flux proj   [gtag straight-dlflux]
export QJ_DL_COMP_WFLUX=1
echo; echo "########## variant: FLUX-ONLY -> straight-dlflux ##########"
run ar17_dl   0.3   6
run ar167_dl  0.03  6
run ar1670_dl 0.003 6

# ---- NO completion, DL ONLY: A = -1/2 I + D   (S dropped, no flux)   [gtag straight-dlonly]
# DISABLED by default: this script now runs the FLUX-ONLY completion only. Uncomment this block to
# also regenerate the un-completed DL-only baseline (and re-add "straight-dlonly" to the plot loop below).
# export QJ_DL_COMP_WFLUX=0
# echo; echo "########## variant: DL-ONLY (no completion) -> straight-dlonly ##########"
# run ar17_dl   0.3   6
# run ar167_dl  0.03  6
# run ar1670_dl 0.003 6

echo
echo "=== near-null eig CSVs ==="
ls -1 "$BASE"/*_dl/straight-dlflux-int-none.eig.csv \
   2>/dev/null | sed 's/^/  /'

# ---- Eigenvalue + eigenvector visualizations for the active variant (flux-only) across aspect ratios --
PY3=$(command -v python3)
VENVPY=/mnt/home/tli10/venv/bin/python; [ -x "$VENVPY" ] || VENVPY="$PY3"
for GT in straight-dlflux; do
  echo; echo "=== plotting $GT ==="
  "$PY3"   python/plot_pspectrum_dl.py all "$BASE" "$GT" || true
  "$VENVPY" python/plot_pmodes_dl.py       "$BASE" "$GT" || true
done
echo
echo "Figures under $BASE :  pspectrum-dlflux-{eig,nearzero-tail}.png"
echo "                        <arNN_dl>/pmodes-<arNN_dl>-{dlflux,dlonly}-{surface,surface-energy,flow,flow-energy}.png"
echo "                        pmodes-{dlflux,dlonly}-aspect-montage.png"
