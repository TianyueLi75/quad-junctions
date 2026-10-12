#!/bin/bash
# Slurm batch (PREPARED, NOT SUBMITTED): in-job A/B of production vs grade2 vs grade2+lowpass256 with IDENTICAL
# interior Green's sources (QJ_GREENS_INT_XYZ=geom/greens_int_prod.xyz, taken from the production bundle).
# Why: the interior sources are stride-sampled arm centerline nodes, so a bundle with a different arm node count
# silently gets a different source set (7223686 int 5.9e-5 -> 7.9e-4 is NOT comparable to the baseline).
# Bundles (all already built, serial ~10 min each):
#   geom/n298_seamsettle_tr6      production
#   geom/n298_tr6_grade2          QJ_ARM_PANEL_GRADE=2
#   geom/n298_tr6_grade2_lp256    QJ_ARM_PANEL_GRADE=2 QJ_ARM_CL_LOWPASS_PASSES=256
# Sources file regenerate: python3 python/greens_int_sources.py geom/n298_seamsettle_tr6 geom/greens_int_prod.xyz
# SUBMIT: sbatch scripts/submit-network-greens-grade2-fixsrc.sh    (full Green's, 3 bundles; set --time=02:00:00 if needed)
# Only the 2 changed bundles need a DL pass with DLONLY=1 if the budget is tight: --export=ALL,DLONLY=1

#SBATCH --job-name=network-greens-fixsrc
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=32
#SBATCH --time=02:00:00
#SBATCH --partition=ccm
#SBATCH --constraint=icelake
#SBATCH --output=out/network-greens-%j.log
#SBATCH --error=out/network-greens-%j.log

WORK_DIR=~/quad-junctions
cd ${WORK_DIR}
source ${WORK_DIR}/sctl_source
set -eo pipefail
[ -f extern/pvfmm/lib/.libs/libpvfmm.a ] || { echo "ERROR: libpvfmm.a missing" >&2; exit 1; }
rm -f obj/bifurc-network-flow-bie.o bin/bifurc-network-flow-bie
make PVFMM=1 obj/bifurc-network-flow-bie.o
make PVFMM=1 bin/bifurc-network-flow-bie
export PVFMM_DIR=${WORK_DIR}/extern/pvfmm; mkdir -p "${PVFMM_DIR}"
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export NTASK=$((${SLURM_NNODES}*${SLURM_NTASKS_PER_NODE}))
CONFIGS=${CONFIGS:-"geom/n298_seamsettle_tr6:1e-6 geom/n298_tr6_grade2:1e-6 geom/n298_tr6_grade2_lp256:1e-6"}
export QJ_NET_MODE=greens QJ_GREENS_KER=laplace
export QJ_GREENS_INT_XYZ=${WORK_DIR}/geom/greens_int_prod.xyz
if [ -n "${DLONLY}" ]; then export QJ_GREENS_DLONLY=1; else unset QJ_GREENS_DLONLY; fi
# QJ_GREENS_DUMP=vis/<tag> would add the VTU error map (this checkout prints only the argmax node, no top-20 list).
for CFG in ${CONFIGS}; do
    PREFIX=${CFG%%:*}; TOL=${CFG##*:}
    [ -f ${PREFIX}-j001.mesh ] || { echo "!!!! SKIP ${PREFIX}: bundle missing" >&2; continue; }
    echo; echo "############################## ${PREFIX}   tol=${TOL} ##############################"
    mpirun -n ${NTASK} --map-by slot:pe=${OMP_NUM_THREADS} \
        -x QJ_NET_MODE -x QJ_GREENS_DLONLY -x QJ_GREENS_KER -x QJ_GREENS_INT_XYZ -x QJ_GREENS_DUMP -x PVFMM_DIR \
        ./bin/bifurc-network-flow-bie ${PREFIX} ${TOL} 2>&1
done
echo "=== done ==="
