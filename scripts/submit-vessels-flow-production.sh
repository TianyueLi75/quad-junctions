#!/bin/bash

# Slurm batch: PRODUCTION interior Stokes inflow/outflow BVP on the FULL genus-10 vessels network (the
# 20-junction arterial+venous tree from arterial_venous_smoothed_nolabels.svg; arterial_only=0). Parabolic
# inflow p_in on the arterial root cap, matching outflow p_out on the venous root cap (net flux 0), no-slip
# walls, combined-field solve. This is the full-network counterpart of the arterial-only studies
# (submit-vessels-flow-scans.sh / -sph90.sh); the historical unpreconditioned reference is ~1860 GMRES iters
# at order8/nref1/tol1e-7 (see the vessels-flow-gmres-stall memory).
#
# PURPOSE (2026-09-21): production-scale test of the interior FLIPPED-diagonal-block preconditioner now that
# it is the interior default. The full network is the ONLY vessels variant with arterial<->venous CONNECTOR
# arms, so it is the only one where the whole-arm connector block (QJ_ARM_PRECOND=whole) exists; that block
# is now sign-flipped (inverted blocks -> -I, same side as the raw -1/2 interior jump). The prior connector
# whole-arm proxy STALLED at ~400 (job in vessels-flow-gmres-stall) -- but WITHOUT the flip, i.e. an
# origin-straddle that the racetrack A/B showed the flip removes (junction-block 181 -> 66). This job re-runs
# it WITH the flip. The rank-1 net-flux completion is left at its interior default (ON) throughout.
#
# A/B rows (see CONFIGS): (1) off = new-default baseline (no block; flip inactive); (2) whole = connector
# whole-arm block, flip ON (interior default). Uncomment the third row to isolate the flip (whole, flip OFF).
#
# FLEXIBLE TO THE ALLOCATION: rank/thread layout is derived from the Slurm request (SLURM_NNODES,
# SLURM_NTASKS_PER_NODE, SLURM_CPUS_PER_TASK). The #SBATCH lines are only defaults; override at submit, e.g.
#   sbatch --nodes=8 --ntasks-per-node=2 --cpus-per-task=32 --time=24:00:00 scripts/submit-vessels-flow-production.sh
#
# Far field via PVFMM (`make PVFMM=1`, forces MPI=1). Every BoundaryIntegralOp goes through the FMM once its
# global target count exceeds 40000 (sctl/fmm-wrapper.txx:858) -- the full vessels mesh is far past it.

#SBATCH --job-name=ybifurc-vessels-flow-production
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=16
#SBATCH --time=12:00:00
#SBATCH --partition=ccm
#SBATCH --constraint=icelake
#SBATCH --output=out/vessels-flow-production-%j.log
#SBATCH --error=out/vessels-flow-production-%j.log

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

# REBUILD ON THE COMPUTE NODE, ONCE (the same binary serves every row -- precond is an env var, rest is argv).
# rm first: obj/*.o is NOT keyed by build flags and the FMM is header-only template code make does not track,
# so a stale object silently runs the wrong pvfmm/flags. REQUIRED, not optional.
rm -f obj/ybifurc-vessels-flow-bie.o bin/ybifurc-vessels-flow-bie
make PVFMM=1 bin/ybifurc-vessels-flow-bie -j

# pvfmm precomputed translation operators. MUST be a directory or pvfmm exit(0)s silently (fmm_pts.txx:248-255).
export PVFMM_DIR=${WORK_DIR}/extern/pvfmm
mkdir -p "${PVFMM_DIR}"

# ---- rank/thread layout from the Slurm allocation ----
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-32}
NNODES=${SLURM_NNODES:-4}
NTPN=${SLURM_NTASKS_PER_NODE:-2}
export NTASK=$((NNODES*NTPN))
NCORE=$((NTASK*OMP_NUM_THREADS))

# ---- production tier (ONE source of truth). order8/nref1/fourier24/tol1e-7 matches the historical
#      full-network 1860-iter reference. FULL network: arterial_only=0, so p_in/p_out set the two root caps
#      (arterial-root inflow, venous-root outflow) and QJ_OUTFLOW_FLUX is NOT used. ----
LEVEL=1.5 ; ORD=8 ; NREF=1 ; ETA=0.4 ;
NSTRANS=2 ; FOURIER=24 ;
LEAD=1 ; CORNER=12 ;
TOL=1e-7 ; NBETA=100 ; MAXD=8 ; COVQ=6 ; SVG=0.06 ;   # (Nbeta,max_depth)=(100,8) is the 1e-7 near-eval tier
PIN=10 ; POUT=10 ;
NGRID=80 ; GMAXIT=5000 ; NVIS=10 ;
GSCALE=1 ; SPHEREDEG=0 ; SPHERETILT=0 ;
ARTERIAL_ONLY=0                       # arg 22: FULL genus-10 network (arterial + venous)

# NB completion (QJ_COMPLETION) is left UNSET -> interior default ON. Set QJ_COMPLETION=0 in a row's env to
# recover the pre-2026-09-21 baseline. QJ_SLENDER_SCALING is left OFF (dead lever on vessels).

# ---- PER-ROW WALL BUDGET (added 2026-09-22 after job 7085100 TIMED OUT) ----
# Job 7085100 requested 10 h and finished only rows 1-2 (comp_off 1604 iters / comp_on 1369 iters); the
# `corner` row then RAN 8.5 h WITHOUT CONVERGING (its rotated-corner setup printed a garbage block count
# `applying 4704873078348706111 block(s)` and never entered gmres) and, because a HANG is not a non-zero
# exit, the `|| true` guard never fired -- it ate the rest of the allocation and no summary printed. Wrap
# every row in `timeout` so a stuck row is killed (exit 124 -> caught by the row guard) and the run
# continues to the next row + the summary. Override at submit, e.g. QJ_ROW_TIMEOUT=4h.
ROW_TIMEOUT=${QJ_ROW_TIMEOUT:-3h}    # per-row hard cap; a row past this is abandoned, next row still runs

# ---- A/B rows: "TAG  env-assignments...". Each is a SEPARATE full-network production solve (node-hours).
#
# FOCUS (2026-09-21, after job 7084538): the COMPLETION A/B. Job 7084538 row `off` (completion default ON)
# converged in 1369 GMRES iters -- notably below the historical ~1860 unpreconditioned reference -- so the
# clean same-config question is whether the rank-1 net-flux completion is what lowered it (the genus-10
# network, unlike the genus-1 racetrack, DOES carry net-flux/circulation modes the completion can touch).
# comp_off vs comp_on isolates exactly that.
#
# The whole-arm connector block is DROPPED here (commented below): at production fourier=24 it is
# n_rep(62)*ElemOrder(10)*fourier(24)*3 = 44640 DOF => 14.85 GB dense, over the 8 GB guard -- job 7084538
# row `whole` SIGABRT'd on it. To run it you MUST cap the prefix with QJ_ARM_PRECOND_NREP so the block fits
# (n_rep=24 => 17280 DOF => ~4.8 GB), but then it is only a LEADING-PREFIX block (not the whole connector)
# AND its ~2e14-flop SVD is a multi-hour sequential-MKL build per distinct orientation (5 here) -- known
# dead lever (see the multiarm-arm-precond + vessels-flow-gmres-stall memories). Left as an opt-in row.
# The CORNER-family rows mirror the racetrack: corner = the two rotated actual-geometry shoulder blocks per
# connector; cornercyl = corners + per-panel straight cylinders on the lead/run (cheap 1-panel blocks, deduped
# by exact size -- no rescaling). Both are single-stage flipped blocks (QJ_PRECOND_SIGN=jump; also the interior
# default) with completion left ON. They are CHEAP (corner block ~cornerP*ElemOrder*fourier*3 DOF; the crashing
# whole-arm block was 62 panels) so no 8 GB-guard issue. Planar full network only (build_corner_sublist is planar).
# The completion A/B is now ANSWERED (job 7085100: comp_off=1604, comp_on=1369 GMRES iters -- the rank-1
# net-flux completion cuts ~15% on the genus-10 network), so `comp_off` is retired from the default to give
# the block-precond rows their wall time. `comp_on` stays as the honest in-job baseline the block rows
# compare against (it re-confirms 1369, and matches job 7084538 off=1369 bit-for-bit). Uncomment comp_off to
# re-run the completion baseline.
CONFIGS=(
# "comp_off   QJ_ARM_PRECOND=off       QJ_COMPLETION=0"                # completion OFF baseline (answered: 1604)
  "comp_on    QJ_ARM_PRECOND=off"                                      # completion default ON baseline (= 1369)
  "corner     QJ_ARM_PRECOND=corner    QJ_PRECOND_SIGN=jump"           # rotated shoulder blocks, flipped, completion ON
  "cornercyl  QJ_ARM_PRECOND=cornercyl QJ_PRECOND_SIGN=jump"           # corners + per-panel straight cyl, flipped, completion ON
# "whole_flip QJ_ARM_PRECOND=whole     QJ_PRECOND_SIGN=jump QJ_ARM_PRECOND_NREP=24"  # opt-in; +raise --time & QJ_PRECOND_MAXGB
)

echo "======== layout: ${NNODES} node(s) x ${NTPN} ranks x ${OMP_NUM_THREADS} threads" \
     "= ${NTASK} ranks / ${NCORE} cores ========"
echo "======== FULL genus-10 vessels network | ${#CONFIGS[@]} precond row(s) |" \
     "order ${ORD} nref ${NREF} fourier ${FOURIER} Ns_trans ${NSTRANS} | tol ${TOL} Nbeta ${NBETA}" \
     "max_depth ${MAXD} cov_q ${COVQ} | p_in ${PIN} p_out ${POUT} | GMAXIT ${GMAXIT} |" \
     "completion=interior-default(ON), flipped blocks ========"

run=0
for cfg in "${CONFIGS[@]}"; do
    run=$((run+1))
    read -r TAG ENVASSIGN <<< "${cfg}"
    echo ""
    echo "######## row ${run}/${#CONFIGS[@]}: precond=${TAG}  (env: ${ENVASSIGN})  (${NTASK} ranks x ${OMP_NUM_THREADS} threads, wall cap ${ROW_TIMEOUT}) ########"
    # Forward the row's env vars to every rank via -x (they are read per-rank). Build the -x list from the
    # names in ENVASSIGN so only the vars this row sets are exported.
    XFLAGS=""
    for kv in ${ENVASSIGN}; do XFLAGS="${XFLAGS} -x ${kv%%=*}"; done
    # Args: level ord nref eta_join Ns_trans fourier lead corner tol Nbeta max_depth cov_q svg_scale
    #       p_in p_out Ngrid gmres_max_iter Nvis gscale sphere_deg sphere_tilt arterial_only
    # Guard with a per-row `timeout` + `|| true`: a row that ABORTS (e.g. an over-guard precond SVD -> MPI
    # SIGABRT) OR HANGS (e.g. the corner-precond setup in job 7085100) must NOT kill the remaining rows or the
    # summary under `set -e`. timeout sends SIGTERM at ROW_TIMEOUT (exit 124) and SIGKILL 60 s later; either
    # way the subshell returns non-zero and the guard continues to the next row.
    # rc capture must use `|| rc=$?`: under `set -e` a bare `( ... ) ; rc=$?` would abort before the
    # assignment when the subshell fails. `|| rc=$?` keeps rc=0 on success and captures the code on failure.
    rc=0
    ( timeout --kill-after=60s "${ROW_TIMEOUT}" \
        env ${ENVASSIGN} mpirun -n ${NTASK} --map-by slot:pe=${OMP_NUM_THREADS} ${XFLAGS} \
        ./bin/ybifurc-vessels-flow-bie \
        ${LEVEL} ${ORD} ${NREF} ${ETA} ${NSTRANS} ${FOURIER} ${LEAD} ${CORNER} \
        ${TOL} ${NBETA} ${MAXD} ${COVQ} ${SVG} ${PIN} ${POUT} ${NGRID} ${GMAXIT} ${NVIS} \
        ${GSCALE} ${SPHEREDEG} ${SPHERETILT} ${ARTERIAL_ONLY} 2>&1 ) || rc=$?
    if [ ${rc} -eq 124 ] || [ ${rc} -eq 137 ]; then
        echo "  [row ${TAG}] EXCEEDED per-row wall cap ${ROW_TIMEOUT} (rc=${rc}) -- abandoned, continuing to next row"
    elif [ ${rc} -ne 0 ]; then
        echo "  [row ${TAG}] returned non-zero (rc=${rc}, see log above) -- continuing to next row"
    fi
    echo "######## row ${run}/${#CONFIGS[@]} (${TAG}) done ########"
done

# ---- summary: iteration count + true-residual per row (a row is valid only if its true rel resid < tol*10
#      and it did NOT hit gmres_max_iter=${GMAXIT}) ----
LOG="out/vessels-flow-production-${SLURM_JOB_ID:-local}.log"
echo ""
echo "======== PRODUCTION SUMMARY (grep from ${LOG}) ========"
if [ -f "${LOG}" ]; then
    grep -nE "^######## row [0-9]|GMRES iters=|TRUE rel residual|precond sign =" "${LOG}" || true
else
    echo "  (log ${LOG} not found -- grep '######## row' and 'GMRES iters=' from wherever stdout landed)"
fi
echo "=== done ==="
