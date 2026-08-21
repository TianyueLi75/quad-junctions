#!/bin/bash

# Slurm batch: physical interior Stokes INFLOW/OUTFLOW BVP on the LARGE vmtk-derived vessel network (the
# ".obj network": 160 quad junctions + bent CSBQ slender arms + 177 hemisphere-capped leaves), loaded
# from the pre-assembled per-junction bundles vis/network-jNNN.{mesh,arms} (bifurc-network-flow-bie.cpp).
#
# ============================ REGENERATE THE GEOMETRY FIRST ============================================
# NO bundle is committed -- vis/ is gitignored, so the per-junction bundles are build artifacts you MUST
# (re)assemble before every run. The CANONICAL geometry is bifurc-network-assemble run on
# data/vmtk/vessels_fixed.graph (TRUE per-junction angles + all watertightness fixes), which closes to
# |int n dA| rel ~1e-5. Assemble at the fixed-geometry DEFAULT PARAMETERS (order=12 / fourier=24 for 6 digits):
#   make bin/bifurc-network-assemble
#   OMP_NUM_THREADS=8 ./bin/bifurc-network-assemble \
#       data/vmtk/vessels_fixed.graph  vis/network  12 1 1.5 0.4 3 12 10 24 2
#       # order=12 nref=1 level=1.5 eta_join=0.4 Ns_trans=3 n_axial=12 cheb=10 fourier=24 lead_panels=2
# The driver prints the combined watertightness |int n dA| up front -- confirm you are on the ~1e-5 (fixed)
# geometry before trusting the solve (a rel ~2.8e-3 closure = an angle-approximated/stale build, whose
# conformity floor caps the solve regardless of quadrature). NB the fixed graph is x10 scale.
# ======================================================================================================
#
# The graph is a TREE (338 nodes / 337 edges => no lumen loops), so unlike the genus-10 SVG vessels net
# the fluid interior is simply connected and GMRES converges like a single bifurcation. A SINGLE inflow
# port (the extreme-x leaf cap = the vmtk traversal root / main trunk) drives a flux-normalized parabolic
# (Poiseuille) velocity; every other leaf cap is an outflow, split equally and normalized so the net flux
# int u.n dA = 0 (the interior incompressible-Stokes Dirichlet compatibility condition, ASSERTED by the
# driver). All tube walls / junction bodies are no-slip. Combined-field solve (-1/2 I - S + D) sigma=u_bc.
#
# SCALE: at order 12 / fourier 24 the network is ~48k panels, ~7.5M surface nodes, ~22M Stokes DOF. This is the
# "reasonable discretization for ~6-digit QUADRATURE accuracy" the request asked for (order 12 tensor
# patches + fourier 24 tube cross-section + tol 1e-7 near-eval, the 1e-7 tier's Nbeta/max_depth). CAVEAT:
# the ACHIEVABLE solve accuracy is additionally capped by the network's geometric CONFORMITY floor (arm
# terminal ring vs junction seam hole); the driver prints the combined watertightness |int n dA| up front
# so you can read that floor before trusting 6 digits. It is a large multi-node PVFMM job -- smoke-test the
# geometry first (see QJ_GEOM_ONLY below) before committing the full solve.
#
# Far field via PVFMM (`make PVFMM=1`, which forces MPI=1). Every BoundaryIntegralOp goes through the FMM
# once its global target count exceeds 40000 (sctl/fmm-wrapper.txx:858) -- this mesh is far past it.
#
# FLEXIBLE TO THE ALLOCATION: the rank/thread layout is derived from the Slurm request, so requesting more
# nodes/tasks/cpus scales it without editing the script. The #SBATCH lines are only defaults; override at
# submit time, e.g.
#   sbatch --nodes=8 --ntasks-per-node=2 --cpus-per-task=32 --time=24:00:00 scripts/submit-network-flow.sh
# For a cheap geometry+watertightness smoke test (no solve, minutes), submit with QJ_GEOM_ONLY, e.g.
#   sbatch --nodes=1 --ntasks-per-node=2 --cpus-per-task=32 --time=00:30:00 --export=ALL,QJ_GEOM_ONLY=1 \
#          scripts/submit-network-flow.sh

#SBATCH --job-name=network-flow
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=32
#SBATCH --time=24:00:00
#SBATCH --partition=gen
#SBATCH --constraint=icelake
#SBATCH --output=out/network-flow-%j.log
#SBATCH --error=out/network-flow-%j.log

WORK_DIR=~/quad-junctions
SOURCE_DIR=${WORK_DIR}/sctl_source
cd ${WORK_DIR}
source ${SOURCE_DIR}
set -eo pipefail

# libpvfmm.a must exist for the link line (built once per the CLAUDE.md "PVFMM build" recipe).
if [ ! -f extern/pvfmm/lib/.libs/libpvfmm.a ]; then
    echo "ERROR: extern/pvfmm/lib/.libs/libpvfmm.a missing -- build it first (see CLAUDE.md, 'PVFMM build'):" >&2
    echo "  cd extern/pvfmm && ./autogen.sh && ./configure --with-blas=... --with-lapack=... && make -j lib/libpvfmm.la" >&2
    exit 1
fi

# Bundle prefix (the -jNNN.{mesh,arms} set). NOT committed -- vis/ is gitignored, so (re)assemble it from the
# FIXED graph before running (see the "REGENERATE THE GEOMETRY FIRST" banner above).
PREFIX=vis/network
if [ ! -f ${PREFIX}-j001.mesh ]; then
    echo "ERROR: bundles ${PREFIX}-jNNN.{mesh,arms} missing -- (re)assemble them from the FIXED graph first:" >&2
    echo "  OMP_NUM_THREADS=8 ./bin/bifurc-network-assemble data/vmtk/vessels_fixed.graph ${PREFIX} 12 1 1.5 0.4 3 12 10 24 2" >&2
    exit 1
fi

# REBUILD ON THE COMPUTE NODE (Makefile uses -march=native; a workstation build SIGILLs on an AMD Rusty
# worker). rm first: obj/*.o is NOT keyed by build flags and the FMM is header-only template code make does
# not track as a prerequisite, so a stale object silently runs the wrong pvfmm/flags. REQUIRED, not optional.
rm -f obj/bifurc-network-flow-bie.o bin/bifurc-network-flow-bie
make PVFMM=1 bin/bifurc-network-flow-bie -j

# pvfmm precomputed translation operators (Precomp_*.data). MUST be a directory or pvfmm exit(0)s silently
# (fmm_pts.txx:248-255). Rank-0-guarded writes, so one shared directory is safe.
export PVFMM_DIR=${WORK_DIR}/extern/pvfmm
mkdir -p "${PVFMM_DIR}"

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export NTASK=$((${SLURM_NNODES}*${SLURM_NTASKS_PER_NODE}))
NCORE=$((NTASK*OMP_NUM_THREADS))

# ---- run parameters: ONE source of truth (log header + mpirun line). The bundle bakes in order 12 / cheb
# ---- 10 / fourier 24 (the fixed-geometry default; see the REGENERATE banner), so accuracy is dialed by the
# ---- bundle set + the near-eval tol tier below. tol 1e-7 is the 6-digit tier: (Nbeta,max_depth)=(100,8) at
# ---- 1e-7 per CLAUDE.md/twisted_sphere; cov_q stays 6.  (PREFIX is set above, before the bundle check.)
TOL=1e-7 ; NBETA=100 ; MAXD=8 ; COVQ=6
PIN=10                 # total inflow flux magnitude (equally split among inflows; outflow normalized to it)
GMAXIT=2000            # tree lumen => converges well below this; cap high for safety at 22M DOF
NVIS=8 ; NGRID=200     # interior point-cloud density (arm stars always kept; junction boxes filtered inside)

# Inflow/outflow port selection (all optional; defaults => single extreme-x cap inflow, rest equal outflow):
#   QJ_INFLOW_AXIS=x|y|z|x-|y-|z-   axis whose extreme cap is the default single inflow (default x = max-x).
#   QJ_INFLOW_NODES="id,id,..."     force these cap graph-node ids as inflows (split equally); overrides axis.
#   QJ_OUTFLOW_FLUX="w1,w2,..."     relative outflow weights in printed order (missing -> 1; unset -> equal).
# Exported (-x) to every rank: caps are assigned identically on all ranks.
export QJ_INFLOW_AXIS=${QJ_INFLOW_AXIS:-x}
# export QJ_INFLOW_NODES="0"
# export QJ_OUTFLOW_FLUX="1,1,1,..."
# QJ_GEOM_ONLY=1 (set via --export at submit time) loads + checks watertightness then exits before the solve.

echo "======== layout: ${SLURM_NNODES} node(s) x ${SLURM_NTASKS_PER_NODE} ranks x ${OMP_NUM_THREADS}" \
     "threads = ${NTASK} ranks / ${NCORE} cores ========"
echo "======== network inflow/outflow: bundles ${PREFIX} | p_in ${PIN} | tol ${TOL} Nbeta ${NBETA}" \
     "max_depth ${MAXD} cov_q ${COVQ} | gmres_max_iter ${GMAXIT} | Nvis ${NVIS} Ngrid ${NGRID}" \
     "| inflow_axis ${QJ_INFLOW_AXIS} geom_only ${QJ_GEOM_ONLY:-0} ========"

# Args: bundle_prefix tol p_in cov_q Nbeta max_depth gmres_max_iter Nvis Ngrid
mpirun -n ${NTASK} --map-by slot:pe=${OMP_NUM_THREADS} \
    -x QJ_INFLOW_AXIS -x QJ_INFLOW_NODES -x QJ_OUTFLOW_FLUX -x QJ_GEOM_ONLY -x PVFMM_DIR \
    ./bin/bifurc-network-flow-bie \
    ${PREFIX} ${TOL} ${PIN} ${COVQ} ${NBETA} ${MAXD} ${GMAXIT} ${NVIS} ${NGRID} 2>&1

echo "=== done, VTUs in vis/bifurc-network-flow-*.{pvtu,vtu} ==="
