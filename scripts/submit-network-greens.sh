#!/bin/bash

# Slurm batch: GEOMETRY-FLOOR accuracy gate on the LARGE vmtk-derived vessel network, run through
# bifurc-network-flow-bie's QJ_NET_MODE=greens + QJ_GREENS_DLONLY path. NO GMRES, NO solve -- just the two
# forward diagnostics that isolate the pure quadrature + geometry-conformity floor:
#
#   1. WATERTIGHT  : combined |int n dA| (printed at load), reported BOTH absolute AND divide-by-area
#                    (rel = |int n dA| / total area -- scale-invariant closure). ~0 for a closed surface.
#   2. DL IDENTITY : constant-density double-layer principal value, checked against BOTH formulations in one
#                    apply -- EXTERIOR (-1/2, outward normals) and INTERIOR (+1/2, inward normals). Only the
#                    correctly-oriented one -> 0; mean(U/q) is printed so the constant is visible exactly.
#
# Green's third identity is deliberately NOT run here (QJ_GREENS_DLONLY short-circuits after the DL apply),
# so this is much cheaper than the old sweep -- one Laplace DL forward apply per config. 
# When QJ_GREENS_DLONLY commented out, will run Green's identity test as well.
#
# ============ GEOMETRY: baseline = geom/n298_seamsettle_tr6 (production) ====================
# Baseline bundle = seam-settle (coaxial quintic-Hermite settle of both merged arm terminals; removed the ~0.5
# cross-list free-edge DL error) + arm turn-angle refinement (QJ_ARM_TURN_REFINE=6, cap QJ_ARM_TURN_REFINE_MAX=24:
# arm panels bending >6 deg are split into ceil(bend/6) sub-panels; default-on). The production arm recipe is
# baked into the code, so a bare build reproduces it -- NO QJ_* env needed:
#     ./bin/bifurc-network-assemble data/vmtk/vessels_fixed_n298.graph geom/n298_seamsettle_tr6 12 1 1.5 0.4 3 12 10 24
# Full-network gate: DL const-density max rel err 1.33e-3, watertight rel closure 6.67e-8 (vs 4.46e-3 / 3.52e-7
# for the same recipe without turn-refine, +~15% arm panels). The residual is the uniform geometry-conformity
# floor; the next lever below that is QJ_RADIUS_SCALE=0.1 (the "nettenth" family, a separate radius-shrink knob).
#
# ---- older geometry family (1/10 arm radius + tapering, junctions scaled to match), kept for reference ----
# All bundles use QJ_RADIUS_SCALE=0.1 (shrink every node radius -- junction bodies AND arm radii together,
# node POSITIONS fixed, so centerline curvature is untouched and every rho/r ratio rescales by 1/scale,
# clearing slender self/near-touch) and QJ_ARM_RADIUS_DIP=1 (thin the tube only through tight mid-arm bends;
# seam rings pinned, watertight). This is the "nettenth" family that reaches ~1e-6 absolute closure.
#
# TWO discretization configs are checked (cheb=10 fixed per CLAUDE.md; fourier kept in the cached toroidal
# set m4..m52). "current" n_axial = 24; the second config doubles it to 48.
#   # config A (6-digit tier)  : order12 fourier16 n_axial24 tol=1e-6
#   QJ_RADIUS_SCALE=0.1 QJ_ARM_RADIUS_DIP=1 ./bin/bifurc-network-assemble \
#       data/vmtk/vessels_fixed.graph geom/nettenth_o12f16  12 1 1.5 0.4 3 24 10 16 2
#   # config B (9-digit tier)  : order16 fourier36 n_axial48 tol=1e-9
#   QJ_RADIUS_SCALE=0.1 QJ_ARM_RADIUS_DIP=1 ./bin/bifurc-network-assemble \
#       data/vmtk/vessels_fixed.graph geom/nettenth_o16f36  16 1 1.5 0.4 3 48 10 36 2
# assemble args: graph prefix  order nref level eta_join Ns_trans n_axial cheb fourier lead_panels
# The .mesh/.arms bundles are portable data files (no -march), so assembling on a workstation is fine.
# =========================================================================================================
#
# WHY MULTI-NODE: config B (order16 / n_axial48 / fourier36) is a large mesh; the near-quadrature setup +
# the PVFMM far-field apply need the memory of several nodes. Laplace DL (default) is far lighter than the
# old Stokes Green's sweep, but the 4-node request keeps config B comfortable. Set KER=stokes to ALSO run
# the Stokes DL const-density identity (heavier near-quad setup).
#
# SUBMIT (override any var via --export=ALL,NAME=VAL at submit time):
#   sbatch scripts/submit-network-greens.sh                                 # default config (baseline, below)
#   sbatch --export=ALL,CONFIGS="geom/n298_seamsettle_tr6:1e-6",QJ_GREENS_DUMP=vis/n298_seamsettle_tr6 \
#       scripts/submit-network-greens.sh                                    # baseline + ParaView error map
#   sbatch --export=ALL,KER=both   scripts/submit-network-greens.sh         # add Stokes DL const-density
#   sbatch --export=ALL,CONFIGS="geom/nettenth_o12f16:1e-6"  scripts/submit-network-greens.sh   # old config
#   sbatch --export=ALL,QJ_GREENS_DUMP=vis/dlerr  scripts/submit-network-greens.sh   # + ParaView VTU error map
#
# SPATIAL ERROR MAP: test_DLIdentity ALWAYS prints the MPI-safe global top-20 DL-error nodes (err = |U/q+1/2|,
# tagged JUNC/ARM + xyz), so the log alone localizes the hotspots. The ~2e-4 floor is NOT panel quadrature
# (that reaches ~1e-7 at tol=1e-9 even on sheared panels) but GEOMETRY CONFORMITY: the hybrid surface is
# flux-closed (~1e-9) yet not pointwise C^1 at the junction<->arm seams, and arms can near-touch/self-fold --
# both LOCAL and refinement-insensitive. A single hotspot rings a seam edge; a mirrored PAIR marks a
# near-touch / self-fold. Set QJ_GREENS_DUMP=<prefix> to ALSO write VTUs; WHICH error it colors depends on
# QJ_GREENS_DLONLY: DLONLY set => <prefix>-<ker>-{junc,arms}.vtu colored by the DL const-density error;
# DLONLY unset (full Green's run) => <prefix>-<ker>-<ext|int>-{junc,arms}.vtu colored by the Green's error.

#SBATCH --job-name=network-greens
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=32
#SBATCH --time=01:00:00
#SBATCH --partition=ccm
#SBATCH --constraint=icelake
#SBATCH --output=out/network-greens-%j.log
#SBATCH --error=out/network-greens-%j.log

WORK_DIR=~/quad-junctions
SOURCE_DIR=${WORK_DIR}/sctl_source
cd ${WORK_DIR}
source ${SOURCE_DIR}
set -eo pipefail

# libpvfmm.a must exist for the link line (built once per the CLAUDE.md "PVFMM build" recipe).
if [ ! -f extern/pvfmm/lib/.libs/libpvfmm.a ]; then
    echo "ERROR: extern/pvfmm/lib/.libs/libpvfmm.a missing -- build it first (see CLAUDE.md, 'PVFMM build')." >&2
    exit 1
fi

# REBUILD ON THE COMPUTE NODE (Makefile uses -march=native; a workstation build SIGILLs on a different arch).
# rm first: obj/*.o is NOT keyed by build flags and the header-only FMM is not tracked as a prerequisite, so a
# stale object silently runs the wrong pvfmm/flags. REQUIRED, not optional.
rm -f obj/bifurc-network-flow-bie.o bin/bifurc-network-flow-bie
# Two explicit steps (NOT `-j`): bin/<x> depends on obj/<x>.o only through a chained implicit rule, so make
# treats the object as an INTERMEDIATE and auto-deletes it after linking (the "rm obj/...o" you see in a good
# run). Under `make -j` that deletion races the link and `ld` intermittently finds the .o already gone
# ("cannot find obj/bifurc-network-flow-bie.o"). Requesting the object as its own goal makes it non-intermediate
# (never auto-removed), and a single TU gets no speedup from -j anyway.
make PVFMM=1 obj/bifurc-network-flow-bie.o
make PVFMM=1 bin/bifurc-network-flow-bie

# pvfmm precomputed translation operators. MUST be a directory or pvfmm exit(0)s silently (fmm_pts.txx).
export PVFMM_DIR=${WORK_DIR}/extern/pvfmm
mkdir -p "${PVFMM_DIR}"

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export NTASK=$((${SLURM_NNODES}*${SLURM_NTASKS_PER_NODE}))
NCORE=$((NTASK*OMP_NUM_THREADS))

# ---- run parameters (override any via --export=ALL,NAME=VAL at submit) --------------------------------
KER=${KER:-laplace}                                # laplace | stokes | both (DL const-density kernel[s])
# Config list: space-separated "bundle_prefix:tol" pairs. Each config pairs a discretization bundle with the
# near-eval tol appropriate to it. Compile any missing bundle per the GEOMETRY banner above.
# Default = production baseline (geom/n298_seamsettle_tr6: seam-settle + turn-refine 6/cap 24, gated ~1.3e-3 DL /
# 6.7e-8 closure; built with CLI params 12 1 1.5 0.4 3 12 10 24). Old nettenth configs still work,
# e.g. CONFIGS="geom/nettenth_o12f16:1e-6 geom/nettenth_o16f36:1e-9".
CONFIGS=${CONFIGS:-"geom/n298_seamsettle_tr6f36:1e-6"}

export QJ_NET_MODE=greens
# export QJ_GREENS_DLONLY=1                           # DL const-density identity only (NO Green's)
export QJ_GREENS_KER=${KER}

echo "======== layout: ${SLURM_NNODES} node(s) x ${SLURM_NTASKS_PER_NODE} ranks x ${OMP_NUM_THREADS}" \
     "threads = ${NTASK} ranks / ${NCORE} cores ========"
echo "======== geometry-floor gate: watertight (abs + /area) + DL const-density (ext & int), KER=${KER} ========"

for CFG in ${CONFIGS}; do
    PREFIX=${CFG%%:*}
    TOL=${CFG##*:}
    if [ ! -f ${PREFIX}-j001.mesh ]; then
        echo "!!!! SKIP ${PREFIX}: bundles ${PREFIX}-jNNN.{mesh,arms} missing (assemble them -- see banner) !!!!" >&2
        continue
    fi
    echo ""
    echo "############################## ${PREFIX}   tol=${TOL} ##############################"
    # bifurc-network-flow-bie args: bundle_prefix tol  (greens mode ignores the flow args)
    mpirun -n ${NTASK} --map-by slot:pe=${OMP_NUM_THREADS} \
        -x QJ_NET_MODE -x QJ_GREENS_DLONLY -x QJ_GREENS_KER -x QJ_GREENS_DUMP -x PVFMM_DIR \
        ./bin/bifurc-network-flow-bie ${PREFIX} ${TOL} 2>&1
done

echo ""
echo "=== done: watertight (abs + /area) + DL const-density identity (ext & int), per config above ==="
