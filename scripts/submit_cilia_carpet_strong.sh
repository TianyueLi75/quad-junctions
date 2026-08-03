#!/bin/bash
# =============================================================================
# submit_cilia_carpet_strong.sh  --  STRONG-scaling study, cilia carpet (0.25*S geometry)
#
# One FIXED problem (NPATCH below) solved on an increasing number of nodes; ideal strong scaling
# halves the solve time when the node count doubles. 
#
# One Slurm allocation covering the LARGEST config; the script then loops, srun'ing the SAME binary/args at
# each MPI rank count in RANK_LIST. The per-node layout is taken FROM THE SBATCH HEADER you submit with:
# tasks/node = --ntasks-per-node (SLURM_NTASKS_PER_NODE) and threads/rank = --cpus-per-task
# (SLURM_CPUS_PER_TASK), so nodes used at r ranks = ceil(r / tasks-per-node). Threads/rank are FIXED across
# the sweep so the speedup isolates MPI scaling (total cores = ranks * threads/rank). The
# driver's profiling prints, per run:
#   - a timed SETUP phase  -> two "Setup" blocks (SL then DL), each with SetupSingular / SetupNear
#   - a Nrep-repeat GMRES loop -> per-repeat t_avg/t_max/f-per-s + the averaged per-iteration solve time
#
# (Run from the quad-junctions repo root. Rebuild first: make PVFMM=1 bin/cilia_carpet-bie)
# Submit with:   sbatch scripts/submit_cilia_carpet_strong.sh
# =============================================================================
#SBATCH --job-name=cilia_strong_64
#SBATCH --partition=ccm
#SBATCH --constraint=icelake
#SBATCH --nodes=32
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=32
#SBATCH --time=00:30:00
#SBATCH --output=out/cilia_strong-%j.log
#SBATCH --error=out/cilia_strong-%j.log

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

# --- FIXED problem + mesh params (CLI: Npatch order tol Naz bot_tip top_tip tilt_deg pdrop Nvis fingers fourier cheb n_axial) ---
NPATCH=8            # <-- FIXED problem size (2*8*8 = 128 cilia, ~2.3M nodes at n_axial=10)
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

# Strong-scaling sweep BY MPI RANK COUNT. Per-node packing follows YOUR sbatch --ntasks-per-node (TPN below),
# so r ranks use ceil(r/TPN) nodes; the allocation (SBATCH --nodes above) MUST cover the LARGEST entry
# (e.g. at TPN=4, 40 ranks -> 10 nodes; at TPN=2, -> 20 nodes). Trim RANK_LIST *and* --nodes together for a
# smaller/cheaper sweep.
# (r=2 and r=4 reproduce the earlier completed run: setup 318.9 s / 161.3 s -- this extends, not replaces it.)
# RANK_LIST="1 2 4 8 16 32 64 80"
# RANK_LIST="1 2 4 8" 
# RANK_LIST="16 32" 
RANK_LIST="64"

# Per-node packing straight from the allocation -- respects whatever --ntasks-per-node you submit with.
TPN=${SLURM_NTASKS_PER_NODE:-2}

echo "STRONG scaling  Npatch=${NPATCH} (FIXED)  order=${ORDER} tol=${TOL} fourier=${FOURIER} n_axial=${N_AXIAL}  tasks/node=${TPN} threads/rank=${OMP_NUM_THREADS}  seed=${QJ_CILIA_SEED}"
echo "  ranks swept: ${RANK_LIST}   (R_shaft=0.25*S locked in the binary; QJ_BOX_BUFFER=${QJ_BOX_BUFFER})"

for r in ${RANK_LIST}; do
  # tasks/node = min(r, TPN); nodes = ceil(r / tpn). A partial last node (r not a multiple of TPN) is fine.
  tpn=$(( r < TPN ? r : TPN )); n=$(( (r + tpn - 1) / tpn ))
  echo "==================================================================================="
  # cores(used) = ranks*threads/rank (cores actually working, the scaling x-axis); cores(alloc) = nodes fully
  # dressed at TPN*threads -- they match when every node is fully packed, and diverge if a node is under-filled.
  echo "[strong] $(date +%H:%M:%S)  ranks=${r}  nodes=${n}  tasks/node=${tpn}  cores(used)=$(( r * OMP_NUM_THREADS ))  cores(alloc)=$(( n * TPN * OMP_NUM_THREADS ))  Npatch=${NPATCH}"
  srun --nodes="${n}" --ntasks="${r}" --ntasks-per-node="${tpn}" \
       --cpus-per-task="${OMP_NUM_THREADS}" --cpu-bind=cores \
       ./bin/cilia_carpet-bie "${NPATCH}" "${ORDER}" "${TOL}" "${NAZ}" \
           "${BOT_TIP}" "${TOP_TIP}" "${TILT_DEG}" "${PDROP}" "${NVIS}" "${FINGERS}" \
           "${FOURIER}" "${CHEB}" "${N_AXIAL}"
done
