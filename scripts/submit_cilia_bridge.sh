#!/bin/bash
# =============================================================================
# submit_cilia_bridge.sh
#
# Doubly-periodic (XY) Stokes WALL-TO-WALL cilia-bridge FLOW run. Each of the Npatch x Npatch cells hosts a
# SINGLE cilium that bridges the bottom plate to the top plate (collar+fillet, one slender shaft, collar+
# fillet -- NO caps, NO free tips, NO tilt), with a random per-cilium sine wiggle confined to its own patch
# cell (neighbours cannot collide). The cilium shaft radius is LOCKED in the binary to R_shaft = 0.25*S
# (S = L/(2*Npatch)), r_fil = 0.1*R_shaft -- scale-invariant in Npatch.
#
# PRECOMPUTES ARE ALREADY BUILT -- this run reuses them and generates NOTHING. Do not change ORDER/CHEB/TOL
# without checking the files below exist for the new value.
#   CSBQ near-quadrature tables (SCTL_DATA_PATH = ./data, baked into the binary -> MUST run from the repo
#     root, which the `cd ${WORK_DIR}` below guarantees):
#       ./data/special_quad_q10_{Laplace3D,Stokes3D}-{DxU,FxU}   (only q10 exists  => CHEB=10 is mandatory)
#       ./data/toroidal_quad_rule_m{10,12,14,16}_*               (=> base ORDER in {10,12,14,16}; 12 used)
#   PVFMM precomputed operators (read from $PVFMM_DIR):
#       extern/pvfmm/Precomp_Stokes3D-{DxU,FxU}_m{4,6,8,10}.data (TOL=1e-7 hits an existing m8/m10 order)
# If a run ever prints "Unable to open ./data/special_quad_*" it was launched from the wrong cwd (the cd
# failed) -- fix the path, do NOT let it regenerate.
#
# Build (one-time, PVFMM required for periodicity -> forces MPI):
#       . ./sctl_source && make PVFMM=1 bin/cilia_bridge-bie
#
# Deliverables (land in the job's out/ log + vis/ dir):
#   1. Visualization           -> vis/CiliaBridge_{geom,shaft,U}.pvtu (ParaView) + density_{base,shaft}
#   2. Periodicity checks      -> "[verify-periodicity] max|u(x=0)-u(x=L)| ... y ..."
#   3. Self-convergence probes -> "[verify-probe] induced u at mid-gap probe points"
#   4. GMRES residual history  -> the "N KSP Residual norm ..." lines (SCTL_VERBOSE)
#
# Submit with:   sbatch scripts/submit_cilia_bridge.sh   (from the quad-junctions repo root)
# =============================================================================
#SBATCH --job-name=ciliabridge
#SBATCH --partition=gen
#SBATCH --constraint=icelake
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=32
#SBATCH --time=01:30:00
#SBATCH --output=out/cilia_bridge-%j.log
#SBATCH --error=out/cilia_bridge-%j.log

WORK_DIR=~/quad-junctions
SOURCE_DIR=${WORK_DIR}/sctl_source
cd ${WORK_DIR}                       # MUST run from repo root so ./data/special_quad_q10_* resolve (no regen)
source ${SOURCE_DIR}
set -euo pipefail

mkdir -p vis out

# --- runtime env -------------------------------------------------------------
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-32}
export OMP_PLACES=cores
export OMP_PROC_BIND=spread

# PVFMM precomputed-operator dir (must EXIST as a directory, else pvfmm exit(0)s silently). Reuses the
# extern/pvfmm/Precomp_Stokes3D-*_m{4,6,8,10}.data files already on disk.
export PVFMM_DIR=${WORK_DIR}/extern/pvfmm

# Cilia-wiggle RNG seed (deterministic geometry, identical on every rank).
export QJ_CILIA_SEED=12345
export QJ_CILIA_WIGGLE=1             # random per-cilium confined sine wiggle ON (=0 disables)
export QJ_BOX_BUFFER=0.01            # cell-containment buffer (the confined bound keeps tubes inside the cell)

# --- geometry / solver parameters --------------------------------------------
# CLI: [Npatch order tol Naz pdrop Nvis fingers fourier cheb n_axial z_plate]
# (R_shaft / r_fil are locked to 0.25*S / 0.1*R_shaft in the binary.)
NPATCH=8            # 8x8 grid = 64 wall-to-wall cilia
ORDER=12            # QuadElemList base order (fixed by the toroidal_quad_rule_m12_* precompute)
TOL=1e-7            # near-quadrature / FMM accuracy (hits the existing PVFMM m8/m10 operators)
NAZ=8               # collar/foot azimuthal sectors
PDROP=-1            # background pressure drop (drives the x-flow)
NVIS=100            # volume-vis grid 100^3
FINGERS=1           # volume-vis finger mask toggle
FOURIER=24          # slender azimuthal Fourier order
CHEB=10             # slender Chebyshev order (only q10 has a precomputed special_quad table -- keep 10)
N_AXIAL=10          # axial panels/fiber (wall-to-wall span ~2x the carpet half-gap => auto n_axial ~2x; 10 caps DOF)

echo "host=$(hostname)  nodes=${SLURM_NNODES:-1}  ntasks=${SLURM_NTASKS:-2}  cpus/task=${OMP_NUM_THREADS}  seed=${QJ_CILIA_SEED}"
echo "QJ_BOX_BUFFER=${QJ_BOX_BUFFER}  QJ_CILIA_WIGGLE=${QJ_CILIA_WIGGLE}  PVFMM_DIR=${PVFMM_DIR}  (R_shaft=0.25*S locked)"
echo "cmd: cilia_bridge-bie $NPATCH $ORDER $TOL $NAZ $PDROP $NVIS $FINGERS $FOURIER $CHEB $N_AXIAL"

# 2 MPI ranks total (2 per node), 32 OpenMP threads each. srun binds per SBATCH --cpus-per-task.
srun --cpus-per-task="${OMP_NUM_THREADS}" --cpu-bind=cores \
    ./bin/cilia_bridge-bie "$NPATCH" "$ORDER" "$TOL" "$NAZ" \
        "$PDROP" "$NVIS" "$FINGERS" "$FOURIER" "$CHEB" "$N_AXIAL"
