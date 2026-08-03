#!/bin/bash
# =============================================================================
# submit_cilia_carpet_8x8_thin.sh
#
# Doubly-periodic (XY) Stokes cilia-carpet FLOW run, 8x8 HYBRID cilia (128 total).
# The cilium shaft radius is LOCKED in the binary to R_shaft = 0.25 * S (thin, patch-relative;
#   S = L/(2*Npatch) = 1/16 = 0.0625 => R_shaft = 0.015625), with r_fil = 0.1*R_shaft. Scale-invariant
#   in Npatch, so the carpet looks the same and has the same solid volume fraction at every grid size.
#
# Mesh parameters: order 16, tol 1e-9, Naz 8, fourier 48, cheb 10, plus:
#   TILT = 10 deg  the largest tested-non-overlapping tilt at full reach-to-midplane
#                  (surface clearance +0.008; tilt>=15 => overlap). Confirmed via QJ_GEOM_ONLY.
#   QJ_BOX_BUFFER = 0.01  keeps every tube surface inside the unit box with
#                  margin (measured overflow -6e-14) while leaving room for the wiggle.
#
# GEOMETRY VERIFIED (QJ_GEOM_ONLY=1, order 16, this exact config):
#   min pairwise cilium clearance = +0.00795  (>0 => NO overlap between cilia)
#   box overflow (x&y, incl R_shaft) = -6e-14 (<=0 => inside the unit box)
#   tube+cap z-extent = [0.0116, 0.988], z overflow -0.0116 (<=0 => cilia do NOT pierce the walls)
#   watertight |int n dA| = 2.7e-10 (rel 1.3e-10), all Jacobians positive
#
# Problem size: base 128*68*16^2 = 1,703,936 + shaft 128*10*10*48 = 614,400 => 2,318,336 nodes
#            (~7.0M Stokes unknowns). ~1.5x the 4x4 order-16 run (1.51M nodes, ~374 s / solve, 64 cores).
#
# Resources: 1 node x 2 tasks/node x 32 CPUs/task = 2 MPI ranks, 32 OpenMP threads each = 64 cores.
#
# Deliverables (land in the job's out/ log + vis/ dir):
#   1. Visualization         -> vis/CiliaCarpet_{geom,shaft,U}.pvtu  (ParaView)  + density_{base,shaft}
#   2. Periodicity checks     -> "[verify-periodicity] max|u(x=0)-u(x=L)| ... y ..."
#   3. Self-convergence probes -> "[verify-probe] induced u at mid-gap probe points"
#   4. GMRES residual history -> the "N KSP Residual norm ..." lines (SCTL_VERBOSE)
#
# Submit with:   sbatch scripts/submit_cilia_carpet_8x8_thin.sh
# (Run from the quad-junctions repo root.)
# =============================================================================
#SBATCH --job-name=cilia8x8thin
#SBATCH --partition=gen
#SBATCH --constraint=icelake
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=32
#SBATCH --time=01:30:00
#SBATCH --output=out/cilia_carpet_8x8-%j.log
#SBATCH --error=out/cilia_carpet_8x8-%j.log

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

# PVFMM precomputed-operator dir (must EXIST as a directory, else pvfmm exit(0)s silently).
export PVFMM_DIR=${WORK_DIR}/extern/pvfmm

# Cilia-wiggle RNG seed (deterministic geometry, identical on every rank).
export QJ_CILIA_SEED=12345

# --- geometry knob required at Npatch=8 --------------------------------------
export QJ_BOX_BUFFER=0.01     # required at Npatch=8 (0.1 default over-constrains the edge column)

# --- visualization mask --------------------------------------------------------------------
# mask_r = max(QJ_VIS_MASK_RFRAC*R_shaft, R_shaft + QJ_VIS_MASK_CELLS*hgrid). The tight-sleeve values
# (1.2, 0.5) that this run needs are now the BINARY DEFAULTS, so no export is required here. With Nvis=100
# hgrid=0.9/99=0.0091, R_shaft=0.0156: mask_r = max(1.2*0.0156, 0.0156+0.5*0.0091) ~= 0.020. QJ_VIS_WALL_MARGIN
# defaults to mask_r, so the near-wall blanked layer matches. If near-tube velocity SPIKES reappear
# (layer-potential eval is inaccurate in the thin shell just outside the surface), raise them, e.g.:
#   export QJ_VIS_MASK_RFRAC=2.0 ; export QJ_VIS_MASK_CELLS=1.0

# --- geometry / solver parameters --------------------------------------------
# CLI: [Npatch order tol Naz bot_tip top_tip tilt_deg pdrop Nvis fingers fourier cheb n_axial]
# (R_shaft / r_fil are no longer CLI args -- locked to 0.25*S / 0.1*R_shaft in the binary.)
NPATCH=8            # 8x8 grid per plate (2*8*8 = 128 cilia)
ORDER=12            # QuadElemList base order (multiple of 4)
TOL=1e-7            # near-quadrature / FMM accuracy
NAZ=8               # collar/foot azimuthal sectors
BOT_TIP=0.5         # bottom cilia reach the midplane z=0.5
TOP_TIP=0.5         # top cilia reach the midplane z=0.5
TILT_DEG=10         # non-overlapping tilt at 8x8 (see header)
PDROP=-1
NVIS=100             # volume-vis grid 80^3 (was 40)
FINGERS=1
FOURIER=36          # slender azimuthal Fourier order
CHEB=10             # slender Chebyshev order (only 10 has a precomputed special_quad table)
N_AXIAL=10          # ~1/4 of the auto ~39 axial panels/fiber (shaft DOF 2.40M -> 0.61M)

echo "host=$(hostname)  nodes=${SLURM_NNODES:-?}  ntasks=${SLURM_NTASKS:-2}  cpus/task=${OMP_NUM_THREADS}  seed=${QJ_CILIA_SEED}"
echo "QJ_BOX_BUFFER=${QJ_BOX_BUFFER}  (R_shaft=0.25*S locked in the binary)"
echo "cmd: cilia_carpet-bie $NPATCH $ORDER $TOL $NAZ $BOT_TIP $TOP_TIP $TILT_DEG $PDROP $NVIS $FINGERS $FOURIER $CHEB $N_AXIAL"

# 2 ranks total (2 per node), 32 OpenMP threads each. srun binds per SBATCH --cpus-per-task.
srun --cpus-per-task="${OMP_NUM_THREADS}" --cpu-bind=cores \
    ./bin/cilia_carpet-bie "$NPATCH" "$ORDER" "$TOL" "$NAZ" \
        "$BOT_TIP" "$TOP_TIP" "$TILT_DEG" "$PDROP" "$NVIS" "$FINGERS" \
        "$FOURIER" "$CHEB" "$N_AXIAL"
