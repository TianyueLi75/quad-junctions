#!/bin/bash
#SBATCH --job-name=capsule-pspectrum
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --partition=gen
#SBATCH --constraint=icelake
#SBATCH --time=03:00:00
#SBATCH --mem=64G
#SBATCH --output=vis/capsule-pspectrum/slurm-%j.out
# =============================================================================
# submit-capsule-pspectrum.sh
#
# PRECONDITIONED SINGULAR-VALUE + EIGENVALUE SPECTRUM of the capsule combined-field self operator
# (src/capsule-robin-bie.cpp `pspectrum` mode). For BOTH the interior and exterior operator it
# assembles the DENSE operator A = jump*I + SL*S + DL*D, forms M^{-1}A for each preconditioner
# strategy by applying the block preconditioner to every column of A, and takes the SVD + eigenvalues
# (dgeev) of each -- the direct spectral picture behind the GMRES iteration counts. One CSV (+ .eig.csv)
# per (geom, prob, variant) under vis/capsule-pspectrum/.
#
# VARIANTS: { none, seam, arm-only, cap+arm } and their SIGN-FLIPPED twins { arm-only-neg, cap+arm-neg }.
#   The interior raw operator sits at eig ~ -1/2 (jump). The usual block-Jacobi sends every inverted
#   self-block to +1, so a PARTIAL precond (arm-only: arm->+1, caps left at -1/2) straddles the origin
#   -> the GMRES stall. The *neg variants apply blk_scale = -1 (via scale_block_precond), sending the
#   inverted blocks to -1 instead -- the SAME side as the un-preconditioned caps -- so arm-only-neg no
#   longer straddles 0, and cap+arm-neg is a single cluster at -1. Toggle with QJ_PSPECTRUM_NEG (default
#   1; set 0 to skip the two extra variants and halve the dgeev cost).
#
# RESOURCE MODEL (why 1 node / all cores / a few hours, NOT MPI):
#   * The mode is SERIAL by construction -- dense global assembly + SVD assert comm.Size()==1.
#     So it is ONE process; parallelism is OpenMP/threaded-MKL across the node's cores, NOT ranks.
#     -> nodes=1, ntasks-per-node=1, cpus-per-task=<all cores>. No mpirun.
#   * Cost is dominated by the dense N x N SVD, N = 3*(cap_dof + arm_dof):
#       - memory: A and M^{-1}A plus the SVD's U,S,V^T workspace ~ 5 * N^2 * 8 bytes.
#           N= 9600 -> ~3.5 GB   N=12000 -> ~5.5 GB   N=16000 -> ~10 GB   (mem=64G is ample headroom)
#       - flops : ~20 N^3 per SVD, and there are 8 SVDs per geometry (4 variants x {int,ext})
#           plus 2 dense assemblies (int,ext, each N column self-evals).
#           N= 9600 -> ~18 Tflop/SVD ; N=12000 -> ~35 Tflop/SVD.
#     dgesvd's bidiagonal-QR tail (dbdsqr) is only partly threaded, so wall-time is conservative:
#     at the default config (N=9600, 2 geoms => 16 SVDs) this finishes well under an hour on 64
#     icelake cores; the 3 h request is headroom for a coarsening bump or a loaded node.
#   * MUST link THREADED MKL. The default Makefile links -lmkl_sequential (SVD would run on one
#     core). This script relinks the compiled object with -lmkl_gnu_thread into bin/capsule-robin-bie-pspec
#     (same recipe as bin/capsule-robin-bie-seam). If you raise the mesh, bump --time / --mem to match
#     the N^3 / N^2 scaling above (and QJ_SPECTRUM_MAXDOF below).
#
# After the job:  python python/plot_pspectrum.py straight   &&   python python/plot_pspectrum.py spiral
# =============================================================================
set -uo pipefail

# Resolve the project root. Under `sbatch`, Slurm copies this script into the spool dir, so $0's
# dirname points there (hence the earlier "./sctl_source: No such file" + "cannot create vis" errors).
# Prefer SLURM_SUBMIT_DIR (the dir you ran sbatch from), else the script's dir; then climb until we
# find sctl_source, so submitting from the repo root OR from scripts/ both land in the project root.
root="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/.." && pwd)}"
cd "$root"
while [ ! -f sctl_source ] && [ "$PWD" != "/" ]; do cd ..; done
[ -f sctl_source ] || { echo "ERROR: cannot locate project root (no sctl_source found from '$root')"; exit 1; }
echo "project root: $PWD"

source ./sctl_source

OUT=vis/capsule-pspectrum
mkdir -p "$OUT"

# ---- build the OBJECT (serial: MPI not needed) then RELINK with threaded MKL for fast SVDs --------
# Object files are not keyed by build flags; a prior MPI=1 build leaves an mpi object here, so clean.
# IMPORTANT: request obj/...o as an EXPLICIT target, NOT `make bin/...`. Make treats the .o as an
# intermediate of the bin rule and DELETES it after linking (`rm obj/...o`), so a relink against it
# then fails. Building the object as the goal keeps it on disk. Also fail loudly -- a silent build
# failure previously let the job run a stale -pspec binary.
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

# ---- mesh / solve params -------------------------------------------------------------------------
# The dense diagnostic now also computes the EIGENVALUES of M^{-1}A (dgeev), which are what GMRES
# actually tracks -- and dgeev is ~as costly as the SVD but its QR iteration threads poorly, so it
# dominates wall time. Two presets below; uncomment ONE (env still overrides either).
#
# PRODUCTION (N=9600; caps fully resolved at Ncap=2 so the interior near-null flux mode is genuine;
# the near-null sigma resolves down toward ~1e-8). dgeev at N=9600 x16 matrices is HOURS -- bump
# --time/--mem before using. Kept here commented for the real run.
# ORDER=${ORDER:-8}
# FOURIER=${FOURIER:-8}
# NCAP=${NCAP:-2}
# NAXIAL=${NAXIAL:-8}
# R=${R:-0.3}
# L=${L:-5}
# TOL=${TOL:-1e-8}
#
# COARSE (N~3k; dgeev x16 finishes in minutes). The flux null mode is TOPOLOGICAL (present even at
# low order -- see the operator-spectrum study), so the qualitative trends survive: one isolated
# interior near-null lambda~0 (a shade larger than production, e.g. ~1e-5, still gapped), and the
# eigenvalue-clustering disconnect (none/caparm tight vs armonly split despite its lower kappa).
# NOTE: capsule-robin-bie asserts BOTH `order` and `fourier` are multiples of 4 (>=4) -- the butterfly
# caps are QuadElemList O-grid patches. 6 aborts at startup; 4 is the smallest valid coarse value.
ORDER=${ORDER:-4}
FOURIER=${FOURIER:-8}
NCAP=${NCAP:-1}
NAXIAL=${NAXIAL:-6}
R=${R:-0.3}
L=${L:-5}
TOL=${TOL:-1e-6}
SEAM_RINGS=${SEAM_RINGS:-1}                       # K arm panels per seam block (user's "one ring")
export QJ_SEAM_RINGS="$SEAM_RINGS"
export QJ_SPECTRUM_MAXDOF=${QJ_SPECTRUM_MAXDOF:-16000}   # dense-SVD guard; raise WITH --time/--mem

# ---- threading: one process, all allocated cores --------------------------------------------------
NCORES=${SLURM_CPUS_PER_TASK:-64}
export OMP_NUM_THREADS="$NCORES" MKL_NUM_THREADS="$NCORES" OMP_PROC_BIND=close OMP_PLACES=cores

echo "=== capsule pspectrum : order=$ORDER fourier=$FOURIER Ncap=$NCAP n_axial=$NAXIAL R=$R L=$L tol=$TOL K=$SEAM_RINGS ==="
echo "    threads=$NCORES  (serial process, threaded MKL)  QJ_SPECTRUM_MAXDOF=$QJ_SPECTRUM_MAXDOF"

# ---- geometries : straight (amp=0) and mild spiral (amp=0.5 turns=1) ------------------------------
run_geom() {   # $1=label  $2..=extra env (spiral)
  local label="$1"; shift
  local log="$OUT/run-${label}.log"
  echo ">>> geom=$label -> $log"
  env "$@" "$BIN" pspectrum "$ORDER" "$R" "$L" "$NAXIAL" "$FOURIER" "$NCAP" "$TOL" > "$log" 2>&1
  grep -E "pspectrum\] (int|ext)/|DOF N=|===" "$log" | sed 's/^/    /'
}

run_geom straight
run_geom spiral QJ_SPIRAL_AMP=0.5 QJ_SPIRAL_TURNS=1

echo
echo "=== CSVs ==="
ls -1 "$OUT"/*.csv 2>/dev/null | sed 's/^/  /'
echo
echo "Now plot:  python python/plot_pspectrum.py straight   &&   python python/plot_pspectrum.py spiral"
