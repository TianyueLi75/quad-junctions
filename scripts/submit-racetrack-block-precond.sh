#!/bin/bash
# =============================================================================
# submit-racetrack-block-precond.sh
#
# PRODUCTION comparison of the CORNER-family flipped-diagonal-block preconditioners on the racetrack
# (ybifurc-channel-bie `flow`, genus-1 lens). Every preconditioned row is a single-stage block-diagonal
# LEFT preconditioner with the INTERIOR sign-flip (inverted blocks -> -I, same side as the raw -1/2 jump;
# the flip is the interior default since 2026-09-21, passed explicitly as QJ_PRECOND_SIGN=jump so each row
# is self-documenting). NO two-stage Jacobi (QJ_JACOBI is never set). The rank-1 net-flux completion is at
# its interior default (ON) and constant across rows, so this reads ONLY the block preconditioner.
#
# Rows (all flipped, no two-stage):
#   off           -- no preconditioner (reference)
#   corner        -- QJ_ARM_PRECOND=corner        : the two rotated actual-geometry shoulder self-blocks only
#   cornerstraight-- QJ_ARM_PRECOND=cornerstraight : corners + WHOLE-SEGMENT straight-cyl blocks on lead/run
#                    (all intra-section axial coupling; the run block is a large sequential-MKL SVD)
#   cornercyl     -- QJ_ARM_PRECOND=cornercyl      : corners + PER-PANEL straight-cyl blocks on lead/run
#                    (cheap 1-panel SVDs; drops intra-section coupling)
#
# NOT included: QJ_PRECOND_BLOCK=junction is INFEASIBLE at production order 8 -- its block is npj*order^2*3
# ~ 46080 DOF => dense ~17 GB, over the 8 GB QJ_PRECOND_MAXGB guard (and a ~2e15-flop SVD). panel-block and
# whole-junction-body blocks do not help the wall-dominated racetrack (coarse A/B). So the corner (wall
# shoulder) family is the right carrier here.
#
# Far field via PVFMM (make PVFMM=1 forces MPI=1); a single MPI rank is fine (EvalPVFMM engages above 40000
# global targets, which the order8/fourier24 lens is past) and keeps every block/segment offset trivially
# local. The corner block SVDs build on Comm::Self() (replicated), so single-rank is also the cheapest layout.
#
# Submit:  sbatch scripts/submit-racetrack-block-precond.sh          (from the quad-junctions repo root)
#          sbatch --cpus-per-task=32 --time=12:00:00 scripts/submit-racetrack-block-precond.sh
# =============================================================================
#SBATCH --job-name=racetrack-block-precond
#SBATCH --partition=gen
#SBATCH --constraint=icelake
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --time=08:00:00
#SBATCH --output=out/racetrack-block-precond-%j.log
#SBATCH --error=out/racetrack-block-precond-%j.log

WORK_DIR=~/quad-junctions
SOURCE_DIR=${WORK_DIR}/sctl_source
cd ${WORK_DIR}                       # MUST run from repo root so ./data/special_quad_q10_* resolve (no regen)
source ${SOURCE_DIR}
set -eo pipefail

mkdir -p vis out

# libpvfmm.a must exist for the link line (built once per CLAUDE.md "PVFMM build").
if [ ! -f extern/pvfmm/lib/.libs/libpvfmm.a ]; then
    echo "ERROR: extern/pvfmm/lib/.libs/libpvfmm.a missing -- build it first (see CLAUDE.md, 'PVFMM build')." >&2
    exit 1
fi

# REBUILD ON THE COMPUTE NODE (Makefile uses -march=native). rm first: obj/*.o is NOT keyed by build flags
# and the FMM is header-only, so a stale object silently runs the wrong pvfmm/flags. REQUIRED, not optional.
rm -f obj/ybifurc-channel-bie.o bin/ybifurc-channel-bie
make PVFMM=1 bin/ybifurc-channel-bie -j

# pvfmm precomputed-operator dir. MUST be a directory or pvfmm exit(0)s silently (fmm_pts.txx).
export PVFMM_DIR=${WORK_DIR}/extern/pvfmm
mkdir -p "${PVFMM_DIR}"

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-64}
export OMP_PLACES=cores
export OMP_PROC_BIND=spread

# ---- production solve parameters (Band D "fine" tier): order8/nref1/fourier24/tol1e-7. ONE source of truth
#      for the srun line below. flow mode = the lens racetrack (interior pressure-drop BVP). ----
# flow  level ord nref eta_join Ns_trans s_cap nAxFree fourier tol Nbeta max_depth cov_q sep tiltDeg \
#       nBent lead corner geomOnly SL_scal DL_scal Ngrid p_in p_out gmres_max_iter Nvis
LEVEL=1.5 ; ORD=8 ; NREF=1 ; ETAJOIN=0.4 ; NSTRANS=3 ; SCAP=0.88 ; NAXFREE=3 ; FOURIER=24
TOL=1e-7 ; NBETA=400 ; MAXD=30 ; COVQ=6
SEP=9.6              # lens junction-center separation (default)
TILTDEG=25 ; NBENT=-1 ; LEAD=2 ; CORNER=12 ; GEOMONLY=0
SLSCAL=1 ; DLSCAL=1 ; NGRID=80 ; PIN=10 ; POUT=10
GMAXIT=400 ; NVIS=0

# ---- rows: "TAG  env-assignments...". Each is a SEPARATE solve (its own srun step); the geometry is
#      identical so only the preconditioner differs. Flip is the interior default; QJ_PRECOND_SIGN=jump is
#      passed on block rows for self-documentation. Trim the list if the wall time is tight. ----
CONFIGS=(
  "off             QJ_ARM_PRECOND=off"
  "corner          QJ_ARM_PRECOND=corner          QJ_PRECOND_SIGN=jump"
  "cornerstraight  QJ_ARM_PRECOND=cornerstraight  QJ_PRECOND_SIGN=jump"
  "cornercyl       QJ_ARM_PRECOND=cornercyl       QJ_PRECOND_SIGN=jump"
)

echo "host=$(hostname)  nodes=${SLURM_NNODES:-1}  ranks=1  cpus/task(threads)=${OMP_NUM_THREADS}"
echo "solve: ord=${ORD} nref=${NREF} fourier=${FOURIER} tol=${TOL} Nbeta=${NBETA} max_depth=${MAXD}" \
     "cov_q=${COVQ} lead=${LEAD} corner=${CORNER} gmres_max=${GMAXIT} | completion=interior-default(ON), flipped blocks"
echo "rows: ${#CONFIGS[@]} (off + corner-family)  PVFMM_DIR=${PVFMM_DIR}"

run=0
for cfg in "${CONFIGS[@]}"; do
    run=$((run+1))
    read -r TAG ENVASSIGN <<< "${cfg}"
    echo ""
    echo "######## row ${run}/${#CONFIGS[@]}: precond=${TAG}  (env: ${ENVASSIGN}) ########"
    # A single row that aborts (e.g. an over-guard SVD) must NOT kill the remaining rows: guard with || true.
    ( env ${ENVASSIGN} srun --cpus-per-task="${OMP_NUM_THREADS}" --cpu-bind=cores \
        ./bin/ybifurc-channel-bie \
            flow ${LEVEL} ${ORD} ${NREF} ${ETAJOIN} ${NSTRANS} ${SCAP} ${NAXFREE} ${FOURIER} \
            ${TOL} ${NBETA} ${MAXD} ${COVQ} ${SEP} ${TILTDEG} ${NBENT} ${LEAD} ${CORNER} ${GEOMONLY} \
            ${SLSCAL} ${DLSCAL} ${NGRID} ${PIN} ${POUT} ${GMAXIT} ${NVIS} 2>&1 ) || \
        echo "  [row ${TAG}] returned non-zero (see log above) -- continuing to next row"
    echo "######## row ${run}/${#CONFIGS[@]} (${TAG}) done ########"
done

# ---- summary: iteration count + true residual per row (a row is valid only if its true rel resid < tol*10
#      and it did NOT hit gmres_max_iter=${GMAXIT}) ----
LOG="out/racetrack-block-precond-${SLURM_JOB_ID:-local}.log"
echo ""
echo "======== SUMMARY (grep from ${LOG}) ========"
if [ -f "${LOG}" ]; then
    grep -nE "^######## row [0-9]|GMRES iters=|TRUE rel residual|precond sign =" "${LOG}" || true
else
    echo "  (log ${LOG} not found -- grep '######## row' and 'GMRES iters=' from wherever stdout landed)"
fi
echo "=== done ==="
