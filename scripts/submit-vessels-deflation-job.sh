#!/bin/bash

# Slurm batch (ONE run per job; submitted by scripts/submit-vessels-deflation.sh, which passes the
# deflation state via --export and overrides --time/--job-name). Runs ybifurc-vessels-flow-bie on the full
# 20-junction vessels network at PRODUCTION resolution, cold-start flow solve, WITHOUT or WITH the layered
# per-connector-arm standing-wave deflation preconditioner (matrixfree_deflation.hpp), so the two jobs are a
# direct A/B of GMRES iteration count on the identical geometry/operator:
#   SCAN_MFS=0           -> skip the geometry tests + MFS; go straight to the cold flow solve
#   QJ_DEFLATE=0 or 1    -> baseline (production stall) vs deflated GMRES
#   QJ_XSECTION_ONLY=1   -> flow solve -> connector flux probes -> [NETFLUX] conservation -> exit (no viz)
# Everything goes to out/vessels-defl-<tag>-<jobid>.log; the [FLOW] RESULT iters line is the headline number.
#
# Required env: SCAN_TOL SCAN_ORD SCAN_FOURIER SCAN_NAX SCAN_DEFLATE
# Optional env: QJ_DEFLATE_NARM (keep K longest connectors; unset = all 11), QJ_DEFLATE_THETA (near-null band,
#               default 0.05), QJ_DEFLATE_FLUX (append flux column; default 1)

#SBATCH --job-name=vessels-defl
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=32
#SBATCH --time=08:00:00
#SBATCH --partition=ccm
#SBATCH --constraint=icelake
#SBATCH --output=out/vessels-defl-%j.log
#SBATCH --error=out/vessels-defl-%j.log

WORK_DIR=~/quad-junctions
cd ${WORK_DIR}
source ${WORK_DIR}/sctl_source
set -eo pipefail

: "${SCAN_TOL:?}" "${SCAN_ORD:?}" "${SCAN_FOURIER:?}" "${SCAN_NAX:?}" "${SCAN_DEFLATE:?}"

# Binary is built once by the launcher (make PVFMM=1); do NOT rebuild here (jobs run concurrently).
[ -x bin/ybifurc-vessels-flow-bie ] || { echo "ERROR: bin/ybifurc-vessels-flow-bie missing" >&2; exit 1; }

# pvfmm precomputed operators: MUST be an existing directory or pvfmm exit(0)s silently.
export PVFMM_DIR=${WORK_DIR}/extern/pvfmm
mkdir -p "${PVFMM_DIR}"

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
NTASK=$((${SLURM_NNODES}*${SLURM_NTASKS_PER_NODE}))

# ---- fixed PRODUCTION parameters (same as submit-vessels-scan-job.sh) ----
LEVEL=1.5 ; NREF=2 ; ETA=0.4 ; NSTRANS=3 ; LEAD=1 ; CORNER=12   # production y120 junction resolution
NBETA=100 ; MAXD=8 ; COVQ=6 ; SVG=0.06 ; PIN=10 ; POUT=10
NGRID=80 ; GMAXIT=8000 ; NVIS=10 ; GSCALE=1 ; SPHEREDEG=0 ; SPHERETILT=0 ; ARTERIAL_ONLY=0

export QJ_MFS=0                        # cold flow solve only (the thing deflation targets)
export QJ_XSECTION_ONLY=1              # flow solve -> connector flux probes -> [NETFLUX] -> exit (no viz)
export QJ_COMPLETION=1                 # net-flux completion ON (production default; deflation layers on top)
export QJ_DL_ONLY=${QJ_DL_ONLY:-1}     # DL-only formulation (SL dropped, S-free dlflux = -1/2 I + D + flux)
export QJ_XS_NRAD=12 QJ_XS_NANG=24
export QJ_VESSELS_NAX_MULT=${SCAN_NAX}
export QJ_DEFLATE=${SCAN_DEFLATE}      # 0 = baseline, 1 = layered per-connector-arm deflation
# deflation knobs (only meaningful when QJ_DEFLATE=1; harmless otherwise). Passed through to the ranks below.
export QJ_DEFLATE_NARM=${QJ_DEFLATE_NARM:-0}    # 0 = all connectors (the longest connecting arms); K = K longest
export QJ_DEFLATE_THETA=${QJ_DEFLATE_THETA:-0.05}
export QJ_DEFLATE_FLUX=${QJ_DEFLATE_FLUX:-1}
export QJ_DEFLATE_UNIFORM_LR=${QJ_DEFLATE_UNIFORM_LR:-0}    # 1 = size every arm block from the MEAN aspect L/R
export QJ_DEFLATE_BLOCK_WHITEN=${QJ_DEFLATE_BLOCK_WHITEN:-0} # 1 = per-arm block-structured whitening (cheap setup)
export QJ_TAG_SUFFIX="tol${SCAN_TOL}-nax${SCAN_NAX}-defl${SCAN_DEFLATE}"

echo "[DEFL] tol=${SCAN_TOL} order=${SCAN_ORD} fourier=${SCAN_FOURIER} nax=${SCAN_NAX} deflate=${SCAN_DEFLATE}" \
     "(narm=${QJ_DEFLATE_NARM} theta=${QJ_DEFLATE_THETA} flux=${QJ_DEFLATE_FLUX}" \
     "uniform_lr=${QJ_DEFLATE_UNIFORM_LR} block_whiten=${QJ_DEFLATE_BLOCK_WHITEN} dl_only=${QJ_DL_ONLY})" \
     "nref=${NREF} Ns_trans=${NSTRANS} job=${SLURM_JOB_ID} ranks=${NTASK} threads=${OMP_NUM_THREADS} $(date)"

# Args: level ord nref eta_join Ns_trans fourier lead corner tol Nbeta max_depth cov_q svg_scale p_in p_out Ngrid gmres_max_iter Nvis gscale sphere_deg sphere_tilt arterial_only
mpirun -n ${NTASK} --map-by slot:pe=${OMP_NUM_THREADS} \
    -x QJ_MFS -x QJ_XSECTION_ONLY -x QJ_COMPLETION -x QJ_XS_NRAD -x QJ_XS_NANG \
    -x QJ_VESSELS_NAX_MULT -x QJ_DEFLATE -x QJ_DEFLATE_NARM -x QJ_DEFLATE_THETA -x QJ_DEFLATE_FLUX \
    -x QJ_DEFLATE_UNIFORM_LR -x QJ_DEFLATE_BLOCK_WHITEN -x QJ_DL_ONLY \
    -x QJ_TAG_SUFFIX -x PVFMM_DIR \
    ./bin/ybifurc-vessels-flow-bie \
    ${LEVEL} ${SCAN_ORD} ${NREF} ${ETA} ${NSTRANS} ${SCAN_FOURIER} ${LEAD} ${CORNER} \
    ${SCAN_TOL} ${NBETA} ${MAXD} ${COVQ} ${SVG} ${PIN} ${POUT} ${NGRID} ${GMAXIT} ${NVIS} ${GSCALE} ${SPHEREDEG} ${SPHERETILT} ${ARTERIAL_ONLY} 2>&1

echo "[DEFL] done $(date)"
