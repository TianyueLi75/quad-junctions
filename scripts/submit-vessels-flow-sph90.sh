#!/bin/bash

# Slurm batch: interior Stokes INFLOW/OUTFLOW BVP on the ARTERIAL-ONLY (capped) vessels network -- the
# arterial binary tree from arterial_venous_smoothed_nolabels.svg, ended in caps where its leaf branches
# used to merge into the venous tree (arterial_only=1; venous half omitted). PLANAR (sphere_deg=0). The
# single arterial-root cap is the INFLOW (flux p_in); every leaf cap is an OUTFLOW, with the split set by
# QJ_OUTFLOW_FLUX (relative weights, normalized so the total outflow == p_in => net flux 0). It is a flux
# (velocity) BC, not a pressure BC; p_out is IGNORED in arterial_only mode. No-slip walls, combined-field
# solve. Hybrid MPI + OpenMP. NB the arterial tree is genus-0 (all caps, no lumen loops), yet GMRES still
# takes 604 iters at order8/nref1/tol1e-7 (job 6869827) -- only ~3x below the full genus-10 network's 1860
# at the SAME tier, matching the ~3x smaller node count. So the loops are NOT the bottleneck (see the
# vessels-flow-gmres-stall memory): iteration count tracks problem size/branch-count, not genus.
#
# GEOMETRY SANITY CHECK (optional, before a full solve): the arterial-only surface is a tree, no draping,
# so there is no tube-tube crowding to worry about; still worth one cheap watertight/collision pass, e.g.
#   OMP_NUM_THREADS=8 ./bin/ybifurc-vessels-bie 1.5 8 1 0.4 2 24 1 12 1e-6 100 8 6 0.06 1 1 0 0 1
# (trailing 1 = arterial_only; arg 14 = geomOnly=1 for a fast no-BIE layout pass).
#
# Far field via PVFMM (`make PVFMM=1`, which forces MPI=1). Every BoundaryIntegralOp goes through the FMM
# once its global target count exceeds 40000 (sctl/fmm-wrapper.txx:858) -- the vessels mesh is far past it.

#SBATCH --job-name=ybifurc-vessels-flow-arterial
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=32
#SBATCH --time=01:00:00
#SBATCH --partition=gen
#SBATCH --constraint=icelake
#SBATCH --output=out/vessels-flow-%j.log
#SBATCH --error=out/vessels-flow-%j.log

WORK_DIR=~/quad-junctions
SOURCE_DIR=${WORK_DIR}/sctl_source
cd ${WORK_DIR}
source ${SOURCE_DIR}
set -eo pipefail

# libpvfmm.a must exist for the link line (built once per the CLAUDE.md "PVFMM build" recipe -- it is not
# rebuilt per-node: it holds only the C/Fortran wrapper API sctl::ParticleFMM never links against).
if [ ! -f extern/pvfmm/lib/.libs/libpvfmm.a ]; then
    echo "ERROR: extern/pvfmm/lib/.libs/libpvfmm.a missing -- build it first (see CLAUDE.md, 'PVFMM build'):" >&2
    echo "  cd extern/pvfmm && ./autogen.sh && ./configure --with-blas=... --with-lapack=... && make -j lib/libpvfmm.la" >&2
    exit 1
fi

# REBUILD ON THE COMPUTE NODE (Makefile uses -march=native; a workstation build SIGILLs on an AMD Rusty
# worker). rm first: obj/*.o is NOT keyed by build flags, and the FMM that runs is header-only template code
# make does not track as a prerequisite, so without the rm a stale object silently runs the wrong pvfmm/flags.
# NB: this rebuild is REQUIRED, not optional -- job 6733231 ran with it commented out and silently reused a
# stale shared bin/ (built from pre-sphere source), so sphere_deg=90 was ignored and it re-ran the PLANAR
# problem. The rm forces a clean recompile of the CURRENT source (which parses argv 19-21 + tags -sph90).
rm -f obj/ybifurc-vessels-flow-bie.o bin/ybifurc-vessels-flow-bie
make PVFMM=1 bin/ybifurc-vessels-flow-bie -j

# pvfmm precomputed translation operators (Precomp_*.data). MUST be a directory or pvfmm exit(0)s silently
# (fmm_pts.txx:248-255). Rank-0-guarded writes, so one shared directory is safe.
export PVFMM_DIR=${WORK_DIR}/extern/pvfmm
mkdir -p "${PVFMM_DIR}"

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export NTASK=$((${SLURM_NNODES}*${SLURM_NTASKS_PER_NODE}))
NCORE=$((NTASK*OMP_NUM_THREADS))

# ---- run parameters: ONE source of truth (log header + mpirun line). 1e-7 accuracy tier (nref=1 mesh +
# ---- tol=1e-7 near-eval set); see submit-vessels-flow.sh for the full tier rationale.
LEVEL=1.5 ; ORD=8 ; NREF=1 ; ETA=0.4 ;
NSTRANS=2 ; FOURIER=24 ;
LEAD=1 ; CORNER=12
TOL=1e-7 ; NBETA=100 ; MAXD=8 ; COVQ=6 ; SVG=0.06 ;
# ARTERIAL-ONLY flux: PIN is the arterial-root inflow flux. In arterial_only mode POUT is IGNORED (the 11
# leaf caps are the outflows); the split among them is QJ_OUTFLOW_FLUX below, normalized so total outflow
# == PIN (net flux 0, the interior-Stokes compatibility the driver ASSERTS -- holds for ANY weights).
PIN=10 ; POUT=10 ;
# Outflow split: 11 relative weights, one per leaf-cap outflow in the order the driver prints them. All 1 =
# EQUAL split (each leaf gets PIN/11). Edit any weight to bias flow to that outlet (e.g. first=3 => 3x share).
OUTFLOW="1,1,1,1,1,1,1,1,1,1,1"
NGRID=80
GMAXIT=5000
# NVIS = junction-box per-axis sample count for the 3D interior point cloud (10 => 10^3/junction, ~1/3
# interior). 0 would fall back to cbrt(NGRID)=4, too sparse for a useful junction cloud.
NVIS=10
# SPHERE DRAPE: GSCALE = global similarity scale; SPHEREDEG = arc degrees the network spans on the sphere
# (0 = planar); SPHERETILT = degrees to splay the two middle connectors apart (clearance relief).
GSCALE=1 ;
# SPHEREDEG=90 ;
SPHEREDEG=0 ;
SPHERETILT=0
# ARTERIAL_ONLY=1: build only the arterial tree, capped where it would meet the venous side (arg 22).
ARTERIAL_ONLY=1
# OBSTACLE TOGGLE: 1 => add rigid spherical obstacles near the centerlines (one per axial arm element +
# one per junction; the inflow/outflow root-cap stems are skipped). 0 => baseline (no obstacles). Sphere
# radius = OBSTACLE_RADFRAC * local tube radius (default 0.2); OBSTACLE_SEED fixes the random placement.
# When ON the output tag gains a "-obst" suffix, so an obstacle run does NOT clobber the baseline VTUs.
OBSTACLE=1
OBSTACLE_SEED=2
OBSTACLE_RADFRAC=0.2

# UNPRECONDITIONED (reverted 2026-08-11). The coarse (order-4) two-grid junction preconditioner
# (quad_junctions/junction_precond.hpp) was tested on this network and HURT convergence: the
# unpreconditioned solve converges in 1860 GMRES iters, whereas the coarse junction block (19/20
# junctions covered, job 6824396) reached 2212 iters without converging inside the 3 h wall. The
# genus-10 lumen circulation modes -- not junction conditioning -- dominate, so a junction-only block
# can only perturb the already well-conditioned junction rows. Machinery kept in the header + racetrack
# driver, but deliberately NOT invoked here. Do NOT re-add QJ_PRECOND_* exports.

# Outflow split forwarded to every rank (caps are assigned on all ranks). -x exports it through OpenMPI.
export QJ_OUTFLOW_FLUX="${OUTFLOW}"

# CSBQ well-conditioned per-node single-layer scaling on the slender arms (Malhotra-Barnett 2024, Eq.33):
# arm SL coefficient eta(s)=1/(2*eps*log(1/eps)) instead of the constant SL_scal, to keep the combined
# field O(1)-conditioned as the tube radius eps->0. MUST be forwarded to every rank (-x below): the eta
# vector is built per-rank on each rank's LOCAL arms. eps_max=0.2 scales the finer/deeper-generation arms
# (this network's radii run ~0.15-0.3, so the 0.1 default would scale nothing). Set QJ_SLENDER_SCALING=0
# to A/B against the unscaled solve.
export QJ_SLENDER_SCALING=1
export QJ_SLENDER_EPS_MAX=0.2

# Spherical obstacles forwarded to every rank (geometry is built on all ranks; -x below exports them).
export QJ_OBSTACLE="${OBSTACLE}"
export QJ_OBSTACLE_SEED="${OBSTACLE_SEED}"
export QJ_OBSTACLE_RADFRAC="${OBSTACLE_RADFRAC}"

echo "======== layout: ${SLURM_NNODES} node(s) x ${SLURM_NTASKS_PER_NODE} ranks x ${OMP_NUM_THREADS}" \
     "threads = ${NTASK} ranks / ${NCORE} cores ========"
echo "======== ARTERIAL-ONLY inflow p_in=${PIN} (p_out ignored) outflow weights=[${OUTFLOW}] |" \
     "order ${ORD} nref ${NREF} fourier ${FOURIER} | tol ${TOL} Nbeta ${NBETA} max_depth ${MAXD} cov_q ${COVQ} |" \
     "Ngrid ${NGRID} Nvis ${NVIS} | Ns_trans ${NSTRANS} gmres_max_iter ${GMAXIT} |" \
     "sphere_deg ${SPHEREDEG} tilt ${SPHERETILT} gscale ${GSCALE} arterial_only ${ARTERIAL_ONLY} |" \
     "obstacle ${OBSTACLE} (seed ${OBSTACLE_SEED} radfrac ${OBSTACLE_RADFRAC}) ========"

# ARTERIAL-ONLY (planar) vessels network: inflow/outflow interior Stokes BVP.
# Args: level ord nref eta_join Ns_trans fourier lead corner tol Nbeta max_depth cov_q svg_scale p_in p_out Ngrid gmres_max_iter Nvis gscale sphere_deg sphere_tilt arterial_only
mpirun -n ${NTASK} --map-by slot:pe=${OMP_NUM_THREADS} \
    -x QJ_OUTFLOW_FLUX -x QJ_SLENDER_SCALING -x QJ_SLENDER_EPS_MAX \
    -x QJ_OBSTACLE -x QJ_OBSTACLE_SEED -x QJ_OBSTACLE_RADFRAC ./bin/ybifurc-vessels-flow-bie \
    ${LEVEL} ${ORD} ${NREF} ${ETA} ${NSTRANS} ${FOURIER} ${LEAD} ${CORNER} \
    ${TOL} ${NBETA} ${MAXD} ${COVQ} ${SVG} ${PIN} ${POUT} ${NGRID} ${GMAXIT} ${NVIS} ${GSCALE} ${SPHEREDEG} ${SPHERETILT} ${ARTERIAL_ONLY} 2>&1
# arterial_only + planar (sphere_deg=0) => the driver tag has NO -sph suffix; -obst is added when OBSTACLE=1.
OBSTAG=$([ "${OBSTACLE}" != "0" ] && echo "-obst" || echo "")
echo "=== done, VTUs in vis/ybifurc-vessels-flow-ord${ORD}-nref${NREF}${OBSTAG}-*.{pvtu,vtu} ==="
