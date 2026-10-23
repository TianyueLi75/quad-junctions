#!/bin/bash
# Slurm batch (PREPARED, NOT SUBMITTED): full-network geometry gate (DL-only) for the POLY-FIT arm pipeline
# (the default arm builder: wide-knot piecewise degree-9 Chebyshev body; append_polyfit_fiber in gen_network_geom.hpp).
# Mirrors scripts/submit-network-greens.sh (same layout / rebuild / mpirun). Gates a PREBUILT bundle only.
#
# The bundle geom/n298_polyfit is assembled LOCALLY (not in this job) so the gate always sees a fixed
# surface; rebuild it with:
#   OMP_NUM_THREADS=8 ./bin/bifurc-network-assemble data/vmtk/vessels_fixed_n298.graph \
#       geom/n298_polyfit 12 1 1.5 0.4 3 12 10 24     # append dumpVTU=1 as arg 16 for geometry VTU
#
# Dumps produced (all under the one QJ_GREENS_DUMP prefix, no overwrite -- distinct name suffixes):
#   * geometry VTU      geom/n298_polyfit-{junc,arms}.vtu          (from the local assemble above, dumpVTU=1)
#   * DL-error VTU      vis/ds_greens-lap-{junc,arms}.vtu             (per-target DL const-density identity err;
#                                                                      written in BOTH the DL-only and FULL gate)
#   * Green's-error VTU vis/ds_greens-lap-<ext|int>-{junc,arms}.vtu   (only with FULL=1; Laplace Green's err)
#
# SUBMIT:  sbatch scripts/submit-network-greens-grade2.sh                        # DL identity gate (+ DL-error VTU)
#          sbatch --export=ALL,FULL=1 scripts/submit-network-greens-grade2.sh    # + Laplace Green's ext/int (+ Green's VTU)
#          sbatch --export=ALL,CONFIGS="geom/n298_polyfit:1e-6 geom/n298_polyfit:1e-6" \
#                 scripts/submit-network-greens-grade2.sh                        # in-job A/B vs production

#SBATCH --job-name=network-greens-ds
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

CONFIGS=${CONFIGS:-"geom/n298_polyfit:1e-6"}
export QJ_NET_MODE=greens
export QJ_GREENS_KER=${KER:-laplace}
export QJ_GREENS_DUMP=${QJ_GREENS_DUMP:-vis/ds_greens}
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
