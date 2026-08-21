#!/bin/bash

# Slurm batch: QUADRATURE-ACCURACY scan of the interior Stokes inflow/outflow BVP on the ARTERIAL-ONLY
# (capped, genus-0) vessels network. Purpose: isolate what drives the vessels GMRES stall. Every prior
# lever is a measured dead end -- FMM far field, genus/lumen loops, Lu2019 completed double-layer,
# junction-block preconditioner, Krylov recycling, and CSBQ Eq.33 slender SL scaling all fail to help
# (see the vessels-flow-gmres-stall memory). The prior SIZE/CONDITIONING scan (job 6901921) showed the
# ~600-iteration count is FLAT vs mesh refinement (nref/order), raw size, and lead/corner, and only ~5%
# sensitive to the 0.8->0.9 taper spread. This scan turns to the QUADRATURE axis: does the near/self
# quadrature ACCURACY (operator tol, and the POU transition resolution Ns_trans that shapes the
# junction<->arm seam) move the iteration count, or is ~600 intrinsic to the operator itself?
#
# The sweep is the FULL CROSS of Ns_trans in {2,3} x tol in {1e-7,1e-9}. Per CLAUDE.md / twisted_sphere
# the near-eval rule parameters are TIED to tol: (Nbeta,max_depth) = (100,8) at 1e-7 and (200,12) at
# 1e-9; cov_q stays 6 at every tol. So each row sets Nbeta/max_depth from its tol automatically -- tol is
# a genuine end-to-end accuracy knob here, not just SetAccuracy. Read-out: GMRES iters FALLING as tol
# tightens (or as Ns_trans rises) => residual/quadrature accuracy is the driver; FLAT => the ~600 count
# is intrinsic to the discretized operator spectrum, independent of how well it is quadratured.
#
# FLEXIBLE TO THE ALLOCATION: rank/thread layout is derived from the Slurm request (SLURM_NNODES,
# SLURM_NTASKS_PER_NODE, SLURM_CPUS_PER_TASK), so requesting more nodes/tasks/cpus scales it without
# editing the script. The #SBATCH lines below are only defaults; override at submit time, e.g.
#   sbatch --nodes=2 --ntasks-per-node=2 --cpus-per-task=32 --time=08:00:00 scripts/submit-vessels-flow-scans.sh
#
# The baseline taper stays at the production r=0.8 for every row (QJ_VESSELS_TAPER=0.8), so this scan
# reads only the quadrature axis and is directly comparable to job 6901921's baseline row.
#
# Far field via PVFMM (`make PVFMM=1`, forces MPI=1). Every BoundaryIntegralOp goes through the FMM once
# its global target count exceeds 40000 (sctl/fmm-wrapper.txx:858) -- the vessels mesh is far past it.

#SBATCH --job-name=ybifurc-vessels-flow-scans
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=32
#SBATCH --time=05:00:00
#SBATCH --partition=gen
#SBATCH --constraint=icelake
#SBATCH --output=out/vessels-flow-scans-%j.log
#SBATCH --error=out/vessels-flow-scans-%j.log

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

# REBUILD ON THE COMPUTE NODE, ONCE (the same binary serves every config -- taper is an env var, the rest
# are argv). rm first: obj/*.o is NOT keyed by build flags and the FMM is header-only template code make
# does not track, so a stale object silently runs the wrong pvfmm/flags. This rm is REQUIRED, not optional
# (job 6733231 ran with it commented out and silently reused a stale bin/).
rm -f obj/ybifurc-vessels-flow-bie.o bin/ybifurc-vessels-flow-bie
make PVFMM=1 bin/ybifurc-vessels-flow-bie -j

# pvfmm precomputed translation operators. MUST be a directory or pvfmm exit(0)s silently (fmm_pts.txx:248-255).
export PVFMM_DIR=${WORK_DIR}/extern/pvfmm
mkdir -p "${PVFMM_DIR}"

# ---- rank/thread layout from the Slurm allocation (fall back to sensible values if run outside Slurm) ----
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
NNODES=${SLURM_NNODES:-1}
NTPN=${SLURM_NTASKS_PER_NODE:-1}
export NTASK=$((NNODES*NTPN))
NCORE=$((NTASK*OMP_NUM_THREADS))

# ---- fixed baseline (ONE source of truth). The scan varies only NSTRANS and TOL per row (and Nbeta/
#      max_depth, which are DERIVED from TOL just below). TAPER is held at the production 0.8. ----
LEVEL=1.5 ; ETA=0.4 ; FOURIER=24 ;
TAPER=0.8 ; NREF=1 ; ORD=12 ; LEAD=1 ; CORNER=12 ;
COVQ=6 ; SVG=0.06 ;                 # cov_q is 6 at every tol (only Nbeta/max_depth track tol)
PIN=10 ; POUT=10 ;                 # arterial_only: POUT ignored; outflow split is QJ_OUTFLOW_FLUX
NGRID=80 ; GMAXIT=5000 ; NVIS=10 ;
GSCALE=1 ; SPHEREDEG=0 ; SPHERETILT=0 ;
ARTERIAL_ONLY=1                     # arg 22: arterial tree only, capped leaves = outflow ports (genus-0)
export QJ_VESSELS_TAPER="${TAPER}"

# tol -> (Nbeta, max_depth) near-eval rule, per CLAUDE.md / scripts/twisted_sphere_run.sh. cov_q fixed 6.
tol_params() {  # sets NBETA, MAXD for the tol in $1
  case "$1" in
    (1e-7) NBETA=100 ; MAXD=8  ;;
    (1e-9) NBETA=200 ; MAXD=12 ;;
    (*) echo "ERROR: no near-eval rule tier defined for tol=$1 (expected 1e-7 or 1e-9)" >&2; exit 1 ;;
  esac
}

# Outflow split: 11 relative weights (one per leaf-cap outflow, in the driver's print order), all 1 =
# equal split, normalized so total outflow == PIN (net flux 0). Forwarded to every rank via -x.
export QJ_OUTFLOW_FLUX="1,1,1,1,1,1,1,1,1,1,1"

# NB no QJ_SLENDER_SCALING / QJ_SLENDER_EPS_MAX here: the CSBQ Eq.33 arm SL scaling was tested and does
# NOT reduce the GMRES iteration count (dead lever, see the csbq-slender-sl-scaling memory). No QJ_PRECOND_*
# either: the junction-block preconditioner was reverted 2026-08-11 (it HURT convergence).

# ---- the 4 runs: FULL CROSS of Ns_trans x tol around the baseline (taper0.8/nref1/ord12/lead1/corner12).
# Each row is "NSTRANS TOL". Nbeta/max_depth come from TOL via tol_params(). All rows share the ord12-nref1
# vis tag and overwrite each other's VTUs -- fine, this study reads only the GMRES iteration counts.
CONFIGS=(
  "2 1e-7"   # 1 baseline (matches job 6901921 baseline row)
  "3 1e-7"   # 2 Ns_trans scan at loose tol
  "2 1e-9"   # 3 tol scan at baseline Ns_trans
  "3 1e-9"   # 4 both tightened
)

echo "======== layout: ${NNODES} node(s) x ${NTPN} ranks x ${OMP_NUM_THREADS} threads" \
     "= ${NTASK} ranks / ${NCORE} cores ========"
echo "======== SCAN: ${#CONFIGS[@]} arterial-only configs (Ns_trans in {2,3} x tol in {1e-7,1e-9}) |" \
     "baseline TAPER=${TAPER} NREF=${NREF} ORD=${ORD} LEAD=${LEAD} CORNER=${CORNER} |" \
     "fixed ETA=${ETA} FOURIER=${FOURIER} cov_q=${COVQ} GMAXIT=${GMAXIT} ========"

run=0
for cfg in "${CONFIGS[@]}"; do
    run=$((run+1))
    read -r NSTRANS TOL <<< "${cfg}"
    tol_params "${TOL}"                 # sets NBETA, MAXD from TOL (cov_q fixed at ${COVQ})
    echo ""
    echo "######## run ${run}/${#CONFIGS[@]}: NSTRANS=${NSTRANS} TOL=${TOL} (Nbeta=${NBETA} max_depth=${MAXD} cov_q=${COVQ})" \
         "(arterial_only, ${NTASK} ranks x ${OMP_NUM_THREADS} threads) ########"
    # Args: level ord nref eta_join Ns_trans fourier lead corner tol Nbeta max_depth cov_q svg_scale
    #       p_in p_out Ngrid gmres_max_iter Nvis gscale sphere_deg sphere_tilt arterial_only
    mpirun -n ${NTASK} --map-by slot:pe=${OMP_NUM_THREADS} \
        -x QJ_OUTFLOW_FLUX -x QJ_VESSELS_TAPER ./bin/ybifurc-vessels-flow-bie \
        ${LEVEL} ${ORD} ${NREF} ${ETA} ${NSTRANS} ${FOURIER} ${LEAD} ${CORNER} \
        ${TOL} ${NBETA} ${MAXD} ${COVQ} ${SVG} ${PIN} ${POUT} ${NGRID} ${GMAXIT} ${NVIS} \
        ${GSCALE} ${SPHEREDEG} ${SPHERETILT} ${ARTERIAL_ONLY} 2>&1
    echo "######## run ${run}/${#CONFIGS[@]} done ########"
done

# ---- scan summary: pull the iteration count + true-residual line for each config out of this log ----
LOG="out/vessels-flow-scans-${SLURM_JOB_ID:-local}.log"
echo ""
echo "======== SCAN SUMMARY (grep from ${LOG}) ========"
echo "  (a config is a valid measurement ONLY if its true rel residual < its own gmres_tol*10 -- tol varies"
echo "   per row (1e-7 or 1e-9); a run that hit gmres_max_iter=${GMAXIT} is a CAP, not a converged count.)"
if [ -f "${LOG}" ]; then
    grep -nE "^######## run [0-9]|GMRES iters=|TRUE rel residual" "${LOG}" || true
else
    echo "  (log ${LOG} not found -- grep '######## run' and 'GMRES iters=' from wherever stdout landed)"
fi
echo "=== done ==="
