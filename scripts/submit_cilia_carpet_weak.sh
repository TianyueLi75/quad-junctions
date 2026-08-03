#!/bin/bash
# =============================================================================
# submit_cilia_carpet_weak.sh  --  WEAK-scaling study, cilia carpet (0.25*S geometry)
#
# Per-RANK work held ~constant while the MPI rank count grows: the size knob is Npatch (cilia = 2*Npatch^2,
# total surface nodes ~ Npatch^2), so cilia/rank = 2*Npatch^2/ranks. Holding that constant means
# Npatch = BASE_NPATCH * sqrt(ranks); then cilia/rank == 2*BASE_NPATCH^2 on every step. Ranks pack
# tasks/node = --ntasks-per-node per node, so nodes = ceil(ranks / tasks-per-node). Npatch is rounded to the
# nearest integer, so the load is only APPROXIMATELY constant (printed each step as Npatch^2/ranks -- watch
# it stay flat). Ideal weak scaling keeps the solve time flat across steps.
#
# One Slurm allocation of MAX_NODES; the script loops srun'ing on r = 1,2,4,... ranks. The driver's
# profiling prints, per run: a timed SETUP phase (two "Setup" blocks, SL then DL, each with SetupSingular /
# SetupNear) and a Nrep-repeat GMRES loop (per-repeat t_avg/t_max/f-per-s + averaged per-iteration solve time).
#
# (Run from the quad-junctions repo root. Rebuild first: make PVFMM=1 bin/cilia_carpet-bie)
# Submit with:   sbatch scripts/submit_cilia_carpet_weak.sh
# =============================================================================
#SBATCH --job-name=cilia_weak_1632
#SBATCH --partition=ccm
#SBATCH --constraint=icelake
#SBATCH --nodes=16
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=32
#SBATCH --time=01:00:00
#SBATCH --output=out/cilia_weak-%j.log
#SBATCH --error=out/cilia_weak-%j.log

WORK_DIR=~/quad-junctions
SOURCE_DIR=${WORK_DIR}/sctl_source
cd ${WORK_DIR}
source ${SOURCE_DIR}
set -euo pipefail

mkdir -p vis out

# --- runtime env -------------------------------------------------------------
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-32}
export OMP_PLACES=cores
export OMP_PROC_BIND=spread
export PVFMM_DIR=${WORK_DIR}/extern/pvfmm     # must EXIST (else pvfmm exit(0)s silently)
export QJ_CILIA_SEED=12345                    # deterministic geometry, identical on every rank
export QJ_BOX_BUFFER=0.01                     # required at Npatch>=8 (0.1 default over-constrains the edge column)

# --- weak-scaling size law + mesh params (CLI: Npatch order tol Naz bot_tip top_tip tilt_deg pdrop Nvis fingers fourier cheb n_axial) ---
# BASE_NPATCH sets the constant per-rank load. ranks=1 -> Npatch=BASE_NPATCH; keep BASE_NPATCH>=3 (driver
# guard requires Npatch>=3), and note BASE_NPATCH*sqrt(ranks) rounds, so cilia/rank drifts slightly.
BASE_NPATCH=4
ORDER=16
TOL=1e-9
NAZ=8
BOT_TIP=0.5
TOP_TIP=0.5
TILT_DEG=10
PDROP=-1
NVIS=1              # scaling study: skip the volume-vis grid (1^3) so timing isn't dominated by VTK I/O
FINGERS=1
FOURIER=48
CHEB=10
N_AXIAL=10

# Weak-scaling sweep BY MPI RANK COUNT (space-separated). Ranks pack tasks/node = --ntasks-per-node, so r
# ranks use ceil(r / tasks-per-node) nodes; that node need must fit SBATCH --nodes (oversize steps are
# SKIPPED below, not hard-failed). Npatch is DERIVED per step (round(BASE_NPATCH*sqrt(r))), so Npatch^2/ranks
# stays ~BASE_NPATCH^2=16 flat:
#   ranks    1   2   4    8   16   32   64   80
#   Npatch   4   6   8   11   16   23   32   36
# RANK_LIST="1 2 4 8 16 32 64 80"
# RANK_LIST="1 2 4 8"
RANK_LIST="16 32"
# RANK_LIST="64 80"

# Per-node packing straight from the allocation -- respects whatever --ntasks-per-node you submit with.
TPN=${SLURM_NTASKS_PER_NODE:-2}
ALLOC_NODES=${SLURM_JOB_NUM_NODES:-0}   # 0 => not under Slurm (no guard); else the --nodes you submitted with

echo "WEAK scaling  BASE_NPATCH=${BASE_NPATCH} (=> 2*${BASE_NPATCH}^2=$((2*BASE_NPATCH*BASE_NPATCH)) cilia/rank target)  order=${ORDER} tol=${TOL} fourier=${FOURIER} n_axial=${N_AXIAL}  tasks/node=${TPN} threads/rank=${OMP_NUM_THREADS}"
echo "  (R_shaft=0.25*S locked in the binary; QJ_BOX_BUFFER=${QJ_BOX_BUFFER})"

for r in ${RANK_LIST}; do
  # tasks/node = min(r, TPN); nodes = ceil(r / tpn) -- same layout logic as the strong-scaling script.
  tpn=$(( r < TPN ? r : TPN )); n=$(( (r + tpn - 1) / tpn ))
  # Skip (don't hard-fail) a step whose node need exceeds the allocation: srun would error and, under `set -e`,
  # abort the whole job -- taking any smaller steps listed after it down too.
  if [ "${ALLOC_NODES}" -gt 0 ] && [ "${n}" -gt "${ALLOC_NODES}" ]; then
    echo "[weak] SKIP ranks=${r} (needs ${n} nodes): exceeds allocation (--nodes=${ALLOC_NODES}). Raise SBATCH --nodes or trim RANK_LIST."
    continue
  fi
  NPATCH=$(awk "BEGIN{printf \"%d\", int(${BASE_NPATCH}*sqrt(${r})+0.5)}")
  perrank=$(awk "BEGIN{printf \"%.2f\", (${NPATCH}*${NPATCH})/${r}}")   # Npatch^2/ranks, flatness proxy (~BASE^2)
  echo "==================================================================================="
  # cores(used) = ranks*threads/rank; matches nodes*TPN*threads when every node is fully packed.
  echo "[weak] $(date +%H:%M:%S)  ranks=${r}  nodes=${n}  tasks/node=${tpn}  cores=$(( r * OMP_NUM_THREADS ))  Npatch=${NPATCH}  (Npatch^2/ranks=${perrank})"
  srun --nodes="${n}" --ntasks="${r}" --ntasks-per-node="${tpn}" \
       --cpus-per-task="${OMP_NUM_THREADS}" --cpu-bind=cores \
       ./bin/cilia_carpet-bie "${NPATCH}" "${ORDER}" "${TOL}" "${NAZ}" \
           "${BOT_TIP}" "${TOP_TIP}" "${TILT_DEG}" "${PDROP}" "${NVIS}" "${FINGERS}" \
           "${FOURIER}" "${CHEB}" "${N_AXIAL}"
done
