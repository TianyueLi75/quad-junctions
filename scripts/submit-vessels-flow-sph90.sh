#!/bin/bash

# Slurm batch: interior Stokes INFLOW/OUTFLOW BVP on the 20-junction arterial/venous network DRAPED ONTO A
# SPHERE (sphere_deg=90 -- the network's overall length span subtends 90 arc degrees of a sphere; every
# junction is placed rigidly tangent to it, arms follow a transported frame). Same physics as the planar
# submit-vessels-flow.sh: parabolic inflow p_in=10 / matching outflow p_out=10 (net flux 0), no-slip walls,
# combined-field solve. The inflow/outflow BCs, Poiseuille network, DL mask, and 3D point-cloud viz are all
# geometry-agnostic, so the SAME pin/pout problem runs unchanged on the curved surface. Hybrid MPI + OpenMP.
#
# CLEARANCE CAVEAT: draping brings the two trees closer together, so at large sphere_deg tubes can nearly
# touch. Sanity-check the geometry once before spending a full solve, e.g.
#   OMP_NUM_THREADS=8 ./bin/ybifurc-vessels-bie 1.5 8 1 0.4 2 24 1 12 1e-6 100 8 6 0.06 1 1 90 0
# and look for the "*** reduce sphere_deg / increase svg_scale ***" clearance warning. Raise SVG (svg_scale)
# or SPHERETILT, or lower SPHEREDEG, if it fires.
#
# Far field via PVFMM (`make PVFMM=1`, which forces MPI=1). Every BoundaryIntegralOp goes through the FMM
# once its global target count exceeds 40000 (sctl/fmm-wrapper.txx:858) -- the vessels mesh is far past it.

#SBATCH --job-name=ybifurc-vessels-flow-sph90
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=32
#SBATCH --time=02:00:00
#SBATCH --partition=gen
#SBATCH --constraint=icelake
#SBATCH --output=out/vessels-flow-sph90-%j.log
#SBATCH --error=out/vessels-flow-sph90-%j.log

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
# p_in == p_out so net flux is 0 (interior-Stokes compatibility, which the driver ASSERTS).
PIN=10 ; POUT=10 ;
NGRID=80
GMAXIT=5000
# NVIS = junction-box per-axis sample count for the 3D interior point cloud (10 => 10^3/junction, ~1/3
# interior). 0 would fall back to cbrt(NGRID)=4, too sparse for a useful junction cloud.
NVIS=10
# SPHERE DRAPE: GSCALE = global similarity scale; SPHEREDEG = arc degrees the network spans on the sphere
# (0 = planar); SPHERETILT = degrees to splay the two middle connectors apart (clearance relief).
GSCALE=1 ; SPHEREDEG=90 ; SPHERETILT=0

echo "======== layout: ${SLURM_NNODES} node(s) x ${SLURM_NTASKS_PER_NODE} ranks x ${OMP_NUM_THREADS}" \
     "threads = ${NTASK} ranks / ${NCORE} cores ========"
echo "======== inflow/outflow p_in=${PIN} p_out=${POUT} | order ${ORD} nref ${NREF} fourier ${FOURIER} |" \
     "tol ${TOL} Nbeta ${NBETA} max_depth ${MAXD} cov_q ${COVQ} | Ngrid ${NGRID} Nvis ${NVIS}" \
     "| Ns_trans ${NSTRANS} gmres_max_iter ${GMAXIT} | SPHERE-DRAPED ${SPHEREDEG} deg tilt ${SPHERETILT} gscale ${GSCALE} ========"

# CAPPED vessels network draped on a sphere: inflow/outflow interior Stokes BVP.
# Args: level ord nref eta_join Ns_trans fourier lead corner tol Nbeta max_depth cov_q svg_scale p_in p_out Ngrid gmres_max_iter Nvis gscale sphere_deg sphere_tilt
mpirun -n ${NTASK} --map-by slot:pe=${OMP_NUM_THREADS} ./bin/ybifurc-vessels-flow-bie \
    ${LEVEL} ${ORD} ${NREF} ${ETA} ${NSTRANS} ${FOURIER} ${LEAD} ${CORNER} \
    ${TOL} ${NBETA} ${MAXD} ${COVQ} ${SVG} ${PIN} ${POUT} ${NGRID} ${GMAXIT} ${NVIS} ${GSCALE} ${SPHEREDEG} ${SPHERETILT} 2>&1
echo "=== done, VTUs in vis/ybifurc-vessels-flow-ord${ORD}-nref${NREF}-sph${SPHEREDEG}-*.{pvtu,vtu} ==="
