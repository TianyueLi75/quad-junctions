#!/bin/bash
# =============================================================================
# submit_cilia_carpet_8x8_thin.sh
#
# Doubly-periodic (XY) Stokes cilia-carpet FLOW run, 16x16 HYBRID cilia (512 total).
# The cilium shaft radius is LOCKED in the binary to R_shaft = 0.25 * S (thin, patch-relative;
#   S = L/(2*Npatch) = 1/32 = 0.03125 => R_shaft = 0.0078125), with r_fil = 0.1*R_shaft. Scale-invariant
#   in Npatch, so the carpet looks the same and has the same solid volume fraction at every grid size.
#
# Mesh parameters: order 12, tol 1e-7, Naz 8, fourier 36, cheb 10, plus:
#   TILT = 10 deg  = the REFERENCE tilt at Npatch=8. The binary auto-scales the tilt so the geometry
#                  stays scale-invariant: tan(tilt_eff) = tan(10deg)*8/Npatch (the lateral tip swing is
#                  reach*tan(tilt) with reach fixed at the midplane, and the pitch is 2S=L/Npatch, so a
#                  fixed swing/pitch => tan(tilt) ~ 1/Npatch). At Npatch=16 this is 5.04deg, inside the
#                  verified non-overlapping window [5,6.5]deg. (A FIXED 10deg here overlaps -- clearance
#                  -0.015. QJ_TILT_SCALE=0 disables the scaling.)
#   QJ_BOX_BUFFER = 0.01  keeps every tube surface inside the unit box with margin.
#
# GEOMETRY VERIFIED (QJ_GEOM_ONLY=1, order 4, this config, tilt auto-scaled to 5.04deg at Npatch=16):
#   min pairwise cilium clearance = +0.0056  (>0 => NO overlap between cilia; ~= the 8x8 +0.0080 margin)
#   box overflow (x&y, incl R_shaft) = 0     (<=0 => inside the unit box)
#   tube+cap z-extent = [0.0108, 0.989], z overflow -0.0108 (<=0 => cilia do NOT pierce the walls)
#
# Problem size (order 12): base 3,833,856 + shaft 1,843,200 => 5,677,056 nodes (~17M Stokes unknowns),
#   exactly 4x the 8x8 order-12 run (1,419,264 nodes). NOTE: for a TRUE weak-scaling point the core count
#   must scale 4x too -- 128 cores here is the same as the 8x8 run, so per-iter cost is ~4x from DOF/core.
#
# Resources: 2 nodes x 2 tasks/node x 32 CPUs/task = 4 MPI ranks, 32 OpenMP threads each = 128 cores.
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
#SBATCH --job-name=cilia14x14
#SBATCH --partition=gen
#SBATCH --constraint=icelake
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=32
#SBATCH --time=01:30:00
#SBATCH --output=out/cilia_carpet_16x16-%j.log
#SBATCH --error=out/cilia_carpet_16x16-%j.log

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
export QJ_CILIA_WIGGLE=1      # random per-cilium sine wiggle ON (=0 disables)

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
NPATCH=14            # 8x8 grid per plate (2*8*8 = 128 cilia)
ORDER=12            # QuadElemList base order (multiple of 4)
TOL=1e-7            # near-quadrature / FMM accuracy
NAZ=8               # collar/foot azimuthal sectors
BOT_TIP=0.5         # bottom cilia reach the midplane z=0.5
TOP_TIP=0.5         # top cilia reach the midplane z=0.5
TILT_DEG=10         # REFERENCE tilt at Npatch=8; binary auto-scales to tan(10)*8/Npatch (=5.04deg at 16). See header.
PDROP=-1
NVIS=100             # volume-vis grid 80^3 (was 40)
FINGERS=1
FOURIER=24          # slender azimuthal Fourier order
CHEB=10             # slender Chebyshev order (only 10 has a precomputed special_quad table)
N_AXIAL=10          # ~1/4 of the auto ~39 axial panels/fiber (shaft DOF 2.40M -> 0.61M)

echo "host=$(hostname)  nodes=${SLURM_NNODES:-?}  ntasks=${SLURM_NTASKS:-2}  cpus/task=${OMP_NUM_THREADS}  seed=${QJ_CILIA_SEED}"
echo "QJ_BOX_BUFFER=${QJ_BOX_BUFFER}  QJ_CILIA_WIGGLE=${QJ_CILIA_WIGGLE}  (R_shaft=0.25*S locked in the binary)"
echo "cmd: cilia_carpet-bie $NPATCH $ORDER $TOL $NAZ $BOT_TIP $TOP_TIP $TILT_DEG $PDROP $NVIS $FINGERS $FOURIER $CHEB $N_AXIAL"

# 2 ranks total (2 per node), 32 OpenMP threads each. srun binds per SBATCH --cpus-per-task.
srun --cpus-per-task="${OMP_NUM_THREADS}" --cpu-bind=cores \
    ./bin/cilia_carpet-bie "$NPATCH" "$ORDER" "$TOL" "$NAZ" \
        "$BOT_TIP" "$TOP_TIP" "$TILT_DEG" "$PDROP" "$NVIS" "$FINGERS" \
        "$FOURIER" "$CHEB" "$N_AXIAL"
