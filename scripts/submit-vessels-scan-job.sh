#!/bin/bash

# Slurm batch (ONE parameter set per job; submitted by scripts/submit-vessels-scan.sh, which passes the
# set via --export and overrides --time/--job-name). Runs ybifurc-vessels-flow-bie on the full 20-junction
# vessels network, in sequence on ONE geometry/operator setup:
#   [GEOM] watertightness -> Stokes DL const-density -> Stokes Green's (exterior sources, on-surface)
#   [MFS]  manufactured interior Stokes solve, exterior Stokeslets
#   [FLOW] pressure-driven inflow/outflow solve (cold start; Krylov recycling removed), then
#   [NETFLUX] connector flux conservation check Q_inlet / Q_outlet / Q_conn_sum (QJ_XSECTION_ONLY=1 exits after this)
# Everything goes to out/vessels-scan-<tag>-<jobid>.log; extract with python3 -I python/parse_vessels_scan.py.
#
# Optional env: SCAN_MFS (0 = cold flow), SCAN_XSNRAD / SCAN_XSNANG (probe quadrature; default 12/24)
# Required env: SCAN_TOL SCAN_ORD SCAN_FOURIER SCAN_NAX (naxial multiplier: 1 = current, 2 = 2x current)

#SBATCH --job-name=vessels-scan
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=32
#SBATCH --time=04:00:00
#SBATCH --partition=ccm
#SBATCH --constraint=icelake
#SBATCH --output=out/vessels-scan-%j.log
#SBATCH --error=out/vessels-scan-%j.log

WORK_DIR=~/quad-junctions
cd ${WORK_DIR}
source ${WORK_DIR}/sctl_source
set -eo pipefail

: "${SCAN_TOL:?}" "${SCAN_ORD:?}" "${SCAN_FOURIER:?}" "${SCAN_NAX:?}"

# Binary is built once by the launcher (make PVFMM=1); do NOT rebuild here (jobs run concurrently).
[ -x bin/ybifurc-vessels-flow-bie ] || { echo "ERROR: bin/ybifurc-vessels-flow-bie missing" >&2; exit 1; }

# pvfmm precomputed operators: MUST be an existing directory or pvfmm exit(0)s silently.
export PVFMM_DIR=${WORK_DIR}/extern/pvfmm
mkdir -p "${PVFMM_DIR}"

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
NTASK=$((${SLURM_NNODES}*${SLURM_NTASKS_PER_NODE}))

# ---- fixed parameters (as submit-vessels-flow-sphscan.sh) ----
LEVEL=1.5 ; NREF=2 ; ETA=0.4 ; NSTRANS=3 ; LEAD=1 ; CORNER=12   # production y120 junction resolution: nref2 + Ns_trans>=3 (was nref1/Ns2); order/fourier from launcher
NBETA=100 ; MAXD=8 ; COVQ=6 ; SVG=0.06 ; PIN=10 ; POUT=10
NGRID=80 ; GMAXIT=8000 ; NVIS=10 ; GSCALE=1 ; SPHEREDEG=0 ; SPHERETILT=0 ; ARTERIAL_ONLY=0

export QJ_MFS=${SCAN_MFS:-1}           # SCAN_MFS=0: cold flow solve (skips geometry tests + MFS); else geometry tests + manufactured solve first (independent solve; no subspace sharing)
export QJ_XSECTION_ONLY=1              # flow solve -> connector flux probes -> [NETFLUX] -> exit (no viz)
export QJ_COMPLETION=1
export QJ_XS_NRAD=${SCAN_XSNRAD:-12} QJ_XS_NANG=${SCAN_XSNANG:-24}   # SCAN_XSNRAD/NANG: probe disk quadrature order
export QJ_VESSELS_NAX_MULT=${SCAN_NAX}
export QJ_TAG_SUFFIX="tol${SCAN_TOL}-nax${SCAN_NAX}${SCAN_XSNRAD:+-nrad${SCAN_XSNRAD}}${SCAN_MFS:+-mfs${SCAN_MFS}}"

echo "[SCAN] tol=${SCAN_TOL} order=${SCAN_ORD} fourier=${SCAN_FOURIER} nax=${SCAN_NAX} nref=${NREF} Ns_trans=${NSTRANS}" \
     "job=${SLURM_JOB_ID} ranks=${NTASK} threads=${OMP_NUM_THREADS} $(date)"

# Args: level ord nref eta_join Ns_trans fourier lead corner tol Nbeta max_depth cov_q svg_scale p_in p_out Ngrid gmres_max_iter Nvis gscale sphere_deg sphere_tilt arterial_only
mpirun -n ${NTASK} --map-by slot:pe=${OMP_NUM_THREADS} \
    -x QJ_VESSELS_NAX_MULT -x QJ_TAG_SUFFIX -x PVFMM_DIR \
    ./bin/ybifurc-vessels-flow-bie \
    ${LEVEL} ${SCAN_ORD} ${NREF} ${ETA} ${NSTRANS} ${SCAN_FOURIER} ${LEAD} ${CORNER} \
    ${SCAN_TOL} ${NBETA} ${MAXD} ${COVQ} ${SVG} ${PIN} ${POUT} ${NGRID} ${GMAXIT} ${NVIS} ${GSCALE} ${SPHEREDEG} ${SPHERETILT} ${ARTERIAL_ONLY} 2>&1

echo "[SCAN] done $(date)"
