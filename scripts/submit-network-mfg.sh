#!/bin/bash

# Slurm batch: MANUFACTURED-SOLUTION acceptance sweep on the LARGE vmtk-derived vessel network, loaded
# from the pre-assembled per-junction bundles geom/n298_seamsettle_tr6-jNNN.{mesh,arms} (bifurc-network-flow-bie.cpp,
# QJ_NET_MODE=manufactured). Validates the SAME coupled operator the flow solve uses, against a known
# analytic Stokeslet field -- two tests per tol:
#   mfg-int: sources EXTERIOR (on the enclosing sphere, radius padded 0.2x) => field smooth in the
#            interior lumen => interior jump -1/2 (SL=-1,DL=+1); checked at interior arm-centerline pts.
#   mfg-ext: sources INTERIOR (arm centerlines) => smooth in the exterior => exterior jump +1/2
#            (SL=+1,DL=+1); checked on the enclosing sphere.
# Only the quadrature tol is swept (order/cheb/fourier are baked into the bundles; the default Duffy
# self/near scheme ignores cov_q/Nbeta/max_depth). rel-L2 is capped by the geometry watertightness floor
# (~1e-6 for the n298_seamsettle_tr6 build): expect it to track tol down to that floor then PLATEAU -- the sub-floor
# rows (1e-7,1e-8) stop improving, which is the evidence that geometry, not quadrature, is the limiter.
#
# ============================ REGENERATE THE GEOMETRY FIRST ============================================
# The n298_seamsettle_tr6 bundle (production; bare-build reproducible, turn-refine default-on, no special env) are a build artifact -- NOT committed. (Re)assemble before a run. The CANONICAL
# geometry is bifurc-network-assemble on data/vmtk/vessels_fixed_n298.graph (TRUE per-junction angles + all
# watertightness fixes), which closes to |int n dA| rel ~1e-5. Assemble at the production discretization:
#   make bin/bifurc-network-assemble
#   OMP_NUM_THREADS=8 ./bin/bifurc-network-assemble \
#       data/vmtk/vessels_fixed_n298.graph  geom/n298_seamsettle_tr6  12 1 1.5 0.4 3 12 10 24
#       # order=12 nref=1 level=1.5 eta_join=0.4 Ns_trans=3 n_axial=12 cheb=10 fourier=24 lead_panels=2
# The driver prints the combined watertightness |int n dA| up front -- confirm you are on the ~1e-5 (fixed)
# geometry before trusting the numbers. NB the fixed graph is x10 scale.
# ======================================================================================================
#
# Far field via PVFMM (`make PVFMM=1`, which forces MPI=1). Every BoundaryIntegralOp goes through the FMM
# once its global target count exceeds 40000 (sctl/fmm-wrapper.txx:858) -- this mesh is far past it.
#
# Override the rank/thread layout at submit time if desired, e.g.
#   sbatch --nodes=4 --ntasks-per-node=2 --cpus-per-task=32 --time=12:00:00 scripts/submit-network-mfg.sh

#SBATCH --job-name=network-mfg
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=32
#SBATCH --time=12:00:00
#SBATCH --partition=ccm
#SBATCH --constraint=icelake
#SBATCH --output=out/network-mfg-%j.log
#SBATCH --error=out/network-mfg-%j.log

WORK_DIR=~/quad-junctions
SOURCE_DIR=${WORK_DIR}/sctl_source
cd ${WORK_DIR}
source ${SOURCE_DIR}
set -eo pipefail

# libpvfmm.a must exist for the link line (built once per the CLAUDE.md "PVFMM build" recipe).
if [ ! -f extern/pvfmm/lib/.libs/libpvfmm.a ]; then
    echo "ERROR: extern/pvfmm/lib/.libs/libpvfmm.a missing -- build it first (see CLAUDE.md, 'PVFMM build')." >&2
    exit 1
fi

# Bundle prefix (the -jNNN.{mesh,arms} set). A build artifact (NOT committed), so (re)assemble it from the
# FIXED graph before running (see the "REGENERATE THE GEOMETRY FIRST" banner above).
PREFIX=${PREFIX:-geom/n298_seamsettle_tr6}
if [ ! -f ${PREFIX}-j001.mesh ]; then
    echo "ERROR: bundles ${PREFIX}-jNNN.{mesh,arms} missing -- (re)assemble them from the FIXED graph first:" >&2
    echo "  OMP_NUM_THREADS=8 ./bin/bifurc-network-assemble data/vmtk/vessels_fixed_n298.graph ${PREFIX} 12 1 1.5 0.4 3 12 10 24" >&2
    exit 1
fi

# REBUILD ON THE COMPUTE NODE (Makefile uses -march=native; a workstation build SIGILLs on an AMD Rusty
# worker). rm first: obj/*.o is NOT keyed by build flags and the FMM is header-only template code make does
# not track as a prerequisite, so a stale object silently runs the wrong pvfmm/flags. REQUIRED, not optional.
rm -f obj/bifurc-network-flow-bie.o bin/bifurc-network-flow-bie
make PVFMM=1 bin/bifurc-network-flow-bie -j

# pvfmm precomputed translation operators (Precomp_*.data). MUST be a directory or pvfmm exit(0)s silently
# (fmm_pts.txx:168-255). Rank-0-guarded writes, so one shared directory is safe.
export PVFMM_DIR=${WORK_DIR}/extern/pvfmm
mkdir -p "${PVFMM_DIR}"

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export NTASK=$((${SLURM_NNODES}*${SLURM_NTASKS_PER_NODE}))
NCORE=$((NTASK*OMP_NUM_THREADS))

# ---- manufactured mode + sweep parameters (ONE source of truth: log header + mpirun line). The bundle
# ---- bakes in order 12 / cheb 10 / fourier 24 (same discretization as submit-network-greens.sh's
# ---- production baseline, assembled with CLI params "12 1 1.5 0.4 3 12 10 24"); only the quadrature tol
# ---- below is swept. The tol list brackets the ~1e-6 watertightness floor and probes a few magnitudes below.
export QJ_NET_MODE=manufactured
# mfg-int ONLY: the interior-source exterior problem (mfg-ext) is far worse conditioned (thousands of GMRES
# iters at eta=1) and starved the previous sweep at the 12h wall limit -- run it separately if needed.
export QJ_MFG_ONLY=${QJ_MFG_ONLY:-int}
# Skip 1e-4 and 1e-5: mfg-int at those tols already completed in job 7016918 (rel-L2 6.33e-4, 1.66e-4).
export QJ_MFG_TOLS=${QJ_MFG_TOLS:-1e-6,1e-7,1e-8}
# GMRES stop tol, DECOUPLED from the swept near-eval tol (new QJ_MFG_GMRES_TOL knob; unset => the historical
# tol*10). Pinned to 1e-3 here so the Krylov solve stops at a loose tolerance regardless of the quadrature tol.
export QJ_MFG_GMRES_TOL=${QJ_MFG_GMRES_TOL:-1e-3}
export QJ_MFG_NLON=${QJ_MFG_NLON:-8}         # exterior-source sphere grid (~8x8 => ~64 sources)
export QJ_MFG_NLAT=${QJ_MFG_NLAT:-8}
export QJ_MFG_JITTER=${QJ_MFG_JITTER:-0.02}  # per-source random offset as a fraction of R_encl
export QJ_MFG_SEED=${QJ_MFG_SEED:-12345}
# The positional tol below is only the single-value fallback (QJ_MFG_TOLS drives the sweep); p_in unused.
TOL=1e-7 ; PIN=10 ; NBETA=100 ; MAXD=8 ; COVQ=6
GMAXIT=5000             # GMRES max iters (>=5000); at the 1e-3 stop tol the mfg-int solve converges well inside this
NVIS=0 ; NGRID=0        # viz cloud unused in manufactured mode

echo "======== layout: ${SLURM_NNODES} node(s) x ${SLURM_NTASKS_PER_NODE} ranks x ${OMP_NUM_THREADS}" \
     "threads = ${NTASK} ranks / ${NCORE} cores ========"
echo "======== network manufactured: bundles ${PREFIX} | tests ${QJ_MFG_ONLY} | tols ${QJ_MFG_TOLS} | sphere ${QJ_MFG_NLON}x${QJ_MFG_NLAT}" \
     "jitter ${QJ_MFG_JITTER} seed ${QJ_MFG_SEED} | gmres_tol ${QJ_MFG_GMRES_TOL} gmres_max_iter ${GMAXIT} ========"

# Args: bundle_prefix tol p_in cov_q Nbeta max_depth gmres_max_iter Nvis Ngrid
mpirun -n ${NTASK} --map-by slot:pe=${OMP_NUM_THREADS} \
    -x QJ_NET_MODE -x QJ_MFG_ONLY -x QJ_MFG_TOLS -x QJ_MFG_GMRES_TOL -x QJ_MFG_NLON -x QJ_MFG_NLAT -x QJ_MFG_JITTER -x QJ_MFG_SEED -x PVFMM_DIR \
    ./bin/bifurc-network-flow-bie \
    ${PREFIX} ${TOL} ${PIN} ${COVQ} ${NBETA} ${MAXD} ${GMAXIT} ${NVIS} ${NGRID} 2>&1

echo "=== done: read the 'manufactured convergence summary' table above ==="
