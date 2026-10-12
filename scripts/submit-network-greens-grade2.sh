#!/bin/bash
# Slurm batch (PREPARED, NOT SUBMITTED): full-network geometry gate for the QJ_ARM_PANEL_GRADE=2 lever
# (docs/next-avenue-panel-grading.md). Mirrors scripts/submit-network-greens.sh (same layout / rebuild / mpirun).
#
# Bundle geom/n298_tr6_grade2 is already built from the production graph (byte-identical to production when the
# knob is off; verified 320/320 files):
#   QJ_ARM_PANEL_GRADE=2 OMP_NUM_THREADS=4 ./bin/bifurc-network-assemble data/vmtk/vessels_fixed_n298.graph \
#       geom/n298_tr6_grade2 12 1 1.5 0.4 3 12 10 24          # ~10 min serial; portable (no -march)
# Subset anchor (hot5, 5 junctions, faithful via QJ_L10): DL max rel err 1.326e-3 -> 2.48e-4 (R=2). Expect the full
# network to drop from 1.33e-3 toward ~2-3e-4 (floor then = junction side, ~7e-5 in the subset).
#
# SUBMIT:  sbatch scripts/submit-network-greens-grade2.sh                          # DL identity only (cheap gate)
#          sbatch --export=ALL,FULL=1 scripts/submit-network-greens-grade2.sh      # + Laplace Green's ext/int
#          sbatch --export=ALL,CONFIGS="geom/n298_seamsettle_tr6:1e-6 geom/n298_tr6_grade2:1e-6" \
#                 scripts/submit-network-greens-grade2.sh                          # in-job A/B vs production

#SBATCH --job-name=network-greens-g2
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=32
#SBATCH --time=01:00:00
#SBATCH --partition=ccm
#SBATCH --constraint=icelake
#SBATCH --output=out/network-greens-%j.log
#SBATCH --error=out/network-greens-%j.log

WORK_DIR=~/quad-junctions
cd ${WORK_DIR}
source ${WORK_DIR}/sctl_source
set -eo pipefail

[ -f extern/pvfmm/lib/.libs/libpvfmm.a ] || { echo "ERROR: libpvfmm.a missing" >&2; exit 1; }

# Rebuild the gate driver on the compute node (-march=native); rm first (stale .o), two explicit steps (no -j).
rm -f obj/bifurc-network-flow-bie.o bin/bifurc-network-flow-bie
make PVFMM=1 obj/bifurc-network-flow-bie.o
make PVFMM=1 bin/bifurc-network-flow-bie

export PVFMM_DIR=${WORK_DIR}/extern/pvfmm
mkdir -p "${PVFMM_DIR}"
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export NTASK=$((${SLURM_NNODES}*${SLURM_NTASKS_PER_NODE}))

CONFIGS=${CONFIGS:-"geom/n298_tr6_grade2:1e-6"}
export QJ_NET_MODE=greens
export QJ_GREENS_KER=${KER:-laplace}
if [ -z "${FULL}" ]; then export QJ_GREENS_DLONLY=1; else unset QJ_GREENS_DLONLY; fi

for CFG in ${CONFIGS}; do
    PREFIX=${CFG%%:*}; TOL=${CFG##*:}
    [ -f ${PREFIX}-j001.mesh ] || { echo "!!!! SKIP ${PREFIX}: bundle missing" >&2; continue; }
    echo; echo "############################## ${PREFIX}   tol=${TOL} ##############################"
    mpirun -n ${NTASK} --map-by slot:pe=${OMP_NUM_THREADS} \
        -x QJ_NET_MODE -x QJ_GREENS_DLONLY -x QJ_GREENS_KER -x QJ_GREENS_DUMP -x PVFMM_DIR \
        ./bin/bifurc-network-flow-bie ${PREFIX} ${TOL} 2>&1
done
echo "=== done ==="
