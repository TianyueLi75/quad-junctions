#!/bin/bash
#SBATCH --job-name=capsule-pspec-comp
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --partition=gen
#SBATCH --constraint=icelake
#SBATCH --time=08:00:00
#SBATCH --mem=96G
#SBATCH --output=vis/capsule-pspectrum/comp-aspect/slurm-%j.out
# =============================================================================
# submit-capsule-pspectrum-completion-aspect.sh
#
# NET-FLUX-COMPLETED interior eigenvalue spectrum of the STRAIGHT capsule, across an aspect /
# arm-resolution / scale sweep. Runs of src/capsule-robin-bie.cpp `pspectrum` with QJ_COMPLETION=1
# (the interior net-flux-zero completion A' = A + (w/area) n (x) ∮n·(.) dA -- see the pspectrum
# comment block + plot_pspectrum.py plot_eig_completed). Each run drops its CSVs in its OWN
# QJ_PSPECTRUM_OUTDIR subdir so the shared "straight-comp-int-none.eig.csv" name does not collide.
#
# Fourier and order are FIXED across all runs; only R_shaft (aspect), n_axial (Nelem), and (for the
# scale-invariance check) L move. The ORIGINAL sweep is commented out below; the ACTIVE runs are:
#   ar1670      : R=0.003, L=5,   n_axial=6    aspect L/R ~ 1670  (shrink R another 0.1x; base Nelem)
#   ar1670-n60  : R=0.003, L=5,   n_axial=60   aspect L/R ~ 1670  (Nelem x10 arm refinement)
#   ar17-smallr : R=0.03,  L=0.5, n_axial=6    aspect L/R = 16.7  (uniform 0.1x scaled base -- scale test)
# (Original, done: ar17 R=0.3/L=5/n6 aspect16.7 ; ar167 R=0.03/L=5/n6 aspect167 ; ar167-n60 n60.)
#
# Only the un-preconditioned "none" variant is needed for the near-null tail plot, so we set
# QJ_PSPECTRUM_NEG=0 (drops the caparmneg dgeev) -- halves the expensive dense eigensolve, which at
# the ar1670-n60 mesh (N~14880) dominates wall time (dgeev's QR tail threads poorly).
#
# After the job:  python3 python/plot_pspectrum.py tail vis/capsule-pspectrum/comp-aspect
#   -> vis/capsule-pspectrum/comp-aspect/pspectrum-comp-nearzero-tail.png
#
# RESOURCE MODEL: serial process (dense assembly + SVD/dgeev assert comm.Size()==1), threaded MKL
# across the node. Cost ~ 10-20 N^3 per dense eig/SVD; the n_axial=6 runs are N=1920 (seconds), the
# ar1670-n60 mesh is N~14880 (~1.8 GB/matrix, tens of Tflop, up to a few hours). --mem/--time sized
# for that run.
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

# ---- completion + eig knobs ----------------------------------------------------------------------
export QJ_COMPLETION=1                 # interior net-flux-zero completion (the whole point)
export QJ_PSPECTRUM_NEG=0              # only the "none" variant is plotted -> skip the caparmneg dgeev
export QJ_SPECTRUM_MAXDOF=${QJ_SPECTRUM_MAXDOF:-16000}   # ar167-n60 is N~14880; raise WITH --mem/--time
export QJ_PSPECTRUM_MODES=${QJ_PSPECTRUM_MODES:-5}       # VISUALIZE the 5 smallest-|lambda| modes per aspect
                                                        # (right eigvecs + interior flow) for plot_pmodes.py.
# The surface/flow ENERGY aggregates must sum over ALL near-null modes, whose count GROWS with aspect ratio
# (|lambda|<threshold: ~1 at a.r.17, ~14 at 167, ~54 at 1670). write_nearnull_basis therefore auto-grows the
# number of dumped eigenvectors+flow fields to max(QJ_PSPECTRUM_MODES, n_null) so the energy sum is complete;
# only 5 are plotted per-mode. Threshold matches plot_pspectrum.py's nearzero-tail THRESH:
export QJ_PSPECTRUM_NULL_THRESH=${QJ_PSPECTRUM_NULL_THRESH:-1e-2}

# ---- threading: one process, all allocated cores -------------------------------------------------
NCORES=${SLURM_CPUS_PER_TASK:-64}
export OMP_NUM_THREADS="$NCORES" MKL_NUM_THREADS="$NCORES" OMP_PROC_BIND=close OMP_PLACES=cores
export OMP_STACKSIZE=256M

echo "=== capsule pspectrum completion sweep : order=$ORDER fourier=$FOURIER Ncap=$NCAP L=$L tol=$TOL ==="
echo "    threads=$NCORES  QJ_SPECTRUM_MAXDOF=$QJ_SPECTRUM_MAXDOF  (completion=1, neg=0)"

# run() : $1=subdir  $2=R_shaft  $3=n_axial  [$4=length, defaults to global $L]
run() {
  local sub="$1" R="$2" NA="$3" LEN="${4:-$L}"
  local out="$BASE/$sub"; mkdir -p "$out"
  local log="$out/run.log"
  echo ">>> $sub : R=$R L=$LEN n_axial=$NA (aspect=$(python3 -c "print(f'{$LEN/$R:.1f}')")) -> $log"
  env QJ_PSPECTRUM_OUTDIR="$out" \
      "$BIN" pspectrum "$ORDER" "$R" "$LEN" "$NA" "$FOURIER" "$NCAP" "$TOL" > "$log" 2>&1
  grep -E "pspectrum\] int/|DOF N=|completion applied|===" "$log" | sed 's/^/    /'
}

# ---- original aspect / arm-resolution sweep (DONE -- commented out; uncomment to regenerate) ------
#   ar17      R=0.3  L=5 n_axial=6    aspect 16.7  (base)
#   ar167     R=0.03 L=5 n_axial=6    aspect 167   (0.1*R, Nelem same)
#   ar167-n60 R=0.03 L=5 n_axial=60   aspect 167   (Nelem x10, fixed-ratio arm)
# The three LOW-Nelem aspect runs are the ones plot_pmodes.py reads (near-null mode viz). N=1920 each,
# seconds -- re-enabled so QJ_PSPECTRUM_MODES=5 drops a basis bundle in each subdir.
run ar17      0.3  6
run ar167     0.03 6
# run ar167-n60 0.03 60

# ---- NEW runs -------------------------------------------------------------------------------------
#   ar1670 / ar1670-n60 : shrink R another 0.1x (0.03 -> 0.003) -> aspect ~1670. Two runs: base Nelem
#     and a x10 arm refinement (Nelem 6 -> 60). NOTE a TRUE fixed-ratio arm at aspect 1670 would need
#     n_axial = 6*100 = 600 (N~145k DOF) -- far beyond a dense dgeev, so 60 is a 10x refinement only.
#   ar17-smallr : 0.1*R (0.03) AND length shortened 1/10 (5 -> 0.5) -> aspect back to 16.7 but on a
#     10x-smaller body. A uniform 0.1x scaled copy of the base -- tests whether the spectrum is
#     scale-invariant (Stokes BIE is scale-free, so its near-null tail should match ar17 exactly).
run ar1670      0.003 6
# run ar1670-n60  0.003 60   # high-Nelem: not read by plot_pmodes.py (low-Nelem only)
# run ar17-smallr 0.03  6  0.5

echo
echo "=== near-null eig CSVs ==="
ls -1 "$BASE"/*/straight-comp-int-none.eig.csv 2>/dev/null | sed 's/^/  /'
# ---- Eigenvalue + eigenvector visualizations across aspect ratios. Same call signature as the dl script
# ---- (plot_pmodes.py <base> <gtag>); runs the plotting once here, and each is also runnable plot-only. ----
PY3=$(command -v python3)
VENVPY=/mnt/home/tli10/venv/bin/python; [ -x "$VENVPY" ] || VENVPY="$PY3"
for GT in straight-comp; do
  echo; echo "=== plotting $GT ==="
  "$PY3"    python/plot_pspectrum.py tail "$BASE"      || true
  "$VENVPY" python/plot_pmodes.py         "$BASE" "$GT" || true
done
echo
echo "Figures under $BASE :  pspectrum-comp-nearzero-tail.png"
echo "                        <arNN>/pmodes-<arNN>-{surface,surface-energy,flow,flow-energy}.png"
echo "                        pmodes-aspect-montage.png   pmodes-aspect-montage_normalized.png"
