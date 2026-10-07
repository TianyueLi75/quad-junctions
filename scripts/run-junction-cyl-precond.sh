#!/bin/bash
# =============================================================================
# run-junction-cyl-precond.sh
#
# The JunctionCyl block preconditioner (junction_cyl_precond.hpp) on the hybrid Y-bifurcation
# inflow/outflow Stokes flow BVP (src/ybifurc-flow-bie.cpp). Each QUAD junction-body arm region
# (annular iso-surface sector + POU transition tube) is preconditioned by an EQUIVALENT STRAIGHT CSBQ
# cylinder self-block -- Nelem = Nr + Ns_trans axial, FourierOrder = Na azimuthal, radius R0 -- built
# in that seam's actual world (u,e1) frame, SVD-pseudo-inverted, and sandwiched by an order/azimuth
# interpolation R (ElemOrder is fixed at 10; the quad region is at patch `order`).
#
# A LEFT preconditioner must NOT change the solution, so the true residual and the interior velocity
# must match between off/on (correctness oracle); the GMRES iteration count is the effect being measured.
#
# Compares GMRES convergence off vs on at a modest config. Serial; OpenMP pinned to 10 physical cores.
# =============================================================================
set -euo pipefail
cd "$(dirname "$0")/.."

. ./sctl_source
make MPI=0 bin/ybifurc-flow-bie

BIN=./bin/ybifurc-flow-bie
OUT=vis/junction-cyl-precond
mkdir -p "$OUT"

export OMP_NUM_THREADS=10
export OMP_PROC_BIND=close
export OMP_PLACES=cores
# The additive block stores dense M+I (blk_quad^2) per arm region: ~23 GB at order 8. Workstations
# have >=512 GB, so lift the default 8 GB precond guard.
export QJ_PRECOND_MAXGB=${QJ_PRECOND_MAXGB:-32}
# Cylinder azimuthal FourierOrder for the coarse space (densely-cached toroidal rule).
export QJ_JUNC_CYL_FOURIER=${QJ_JUNC_CYL_FOURIER:-36}

# modest mesh: level order nref eta_join Ns_trans s_cap n_axial fourier tol ...
# order 8, nref 1 -> Na=16, Nr=2, Ns_trans=3 -> merged cylinder Nax=5, FourierOrder=16.
ARGS="1.5 8 1 0.4 3 0.88 3 12 1e-8"

for MODE in off on; do
  echo "=================== JunctionCyl precond = ${MODE} ==================="
  QJ_JUNC_CYL_PRECOND=${MODE} ${BIN} ${ARGS} 2>&1 | tee "${OUT}/flow-${MODE}.txt"
  echo
done

echo "=== summary (GMRES iters) ==="
grep -H "GMRES iters=" "${OUT}"/flow-*.txt || true
