#!/bin/bash
# =============================================================================
# run-capsule-seam-precond.sh
#
# SeamCouple preconditioner sweep on the manufactured-Stokeslet Dirichlet probe (capsule mfs mode).
# The coupled block spans one cap hemisphere + the first/last K arm panels at EACH end; the middle arm
# panels are handled by QJ_SEAM_REST = none | panelexact | armwhole.
#
# Per the aspect-33 study: INTERIOR uses n_axial=64, EXTERIOR uses n_axial=32, tol=1e-10.
# Runs LOCAL, pinned to cores 8-15 (leaving 0-7 free), threaded MKL + LU-inverse blocks (fast).
# Uses the alternate binary bin/capsule-robin-bie-seam (threaded-MKL relink; see the session notes).
# =============================================================================
set -uo pipefail
cd "$(dirname "$0")/.."
. ./sctl_source

BIN=./bin/capsule-robin-bie-seam
OUT=vis/capsule-mfs-refined
mkdir -p "$OUT"
ORDER=12; FOURIER=16; NCAP=2; TOL=1e-10; K=2

export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OMP_PROC_BIND=close OMP_PLACES=cores

run(){ # $1=prob(int|ext) $2=n_axial $3=rest
  local prob=$1 na=$2 rest=$3
  local log="$OUT/${prob}-nax${na}-seamK${K}-${rest}.log"
  echo ">>> prob=$prob n_axial=$na seam K=$K rest=$rest -> $log"
  env QJ_ARM_PRECOND=seam QJ_SEAM_RINGS=$K QJ_SEAM_REST=$rest QJ_MFS_ONLY=$prob \
      taskset -c 8-15 "$BIN" mfs "$ORDER" 0.3 10 "$na" "$FOURIER" "$NCAP" "$TOL" > "$log" 2>&1
  grep -E "GMRES iters =|rel-L2|seamcouple" "$log" | sed 's/^/    /'
}

echo "=== EXTERIOR n_axial=32 (tol=$TOL) ==="
for rest in none panelexact armwhole; do run ext 32 "$rest"; done
echo "=== INTERIOR n_axial=64 (tol=$TOL) ==="
for rest in none panelexact armwhole; do run int 64 "$rest"; done
echo "=== SEAM SWEEP DONE ==="
