#!/bin/bash
# LAUNCHER (run on a login node:  bash scripts/submit-vessels-scan.sh [--dry-run]).
# Submits ONE Slurm job per parameter set. DEFAULT = PRODUCTION y120 junction resolution
#   tol 1e-9  x  order 12  x  nref 2  x  Ns_trans 3  x  fourier 36  x  naxial multiplier 2
#   (the playbook acceptance recipe: order12/nref2/fourier36, Required Ns_trans>=3, pou=1-smootherstep.
#    order/fourier/tol/nax are set here; nref/Ns_trans in scripts/submit-vessels-scan-job.sh.)
# To sweep, add values back to the TOL / ORD / NAX loop lists below (e.g. TOL in 1e-6 1e-9 1e-11).
# Each job: 4 nodes x 2 tasks x 32 cpus, partition ccm, icelake (see scripts/submit-vessels-scan-job.sh).
# Wall time: 4 h per job.
# Afterwards:  python3 -I python/parse_vessels_scan.py out/vessels-scan-*.log
cd ~/quad-junctions
DRY=0; [ "$1" = "--dry-run" ] && DRY=1

if [ $DRY -eq 0 ]; then
    source ./sctl_source
    [ -f extern/pvfmm/lib/.libs/libpvfmm.a ] || { echo "ERROR: build libpvfmm first (CLAUDE.md 'PVFMM build')" >&2; exit 1; }
    # Build ONCE here. Objects are not keyed by build flags -> force the source to recompile.
    # NB -march=native: build on a node of the same CPU family as the icelake compute nodes.
    # rm -f obj/ybifurc-vessels-flow-bie.o bin/ybifurc-vessels-flow-bie
    # make PVFMM=1 bin/ybifurc-vessels-flow-bie -j || exit 1
fi
mkdir -p out

FOURIER=36                 # production y120: arm azimuthal order (was 24); Ns_trans/nref in the job script
for TOL in 1e-11; do
  for ORD in 12; do
    for NAX in 2; do
      TIME=08:00:00        # nref2/fourier36 is ~2x the DOF of the old nref1/fourier24 run -> more wall time
      NAME="vscan-t${TOL}-o${ORD}-f${FOURIER}-n${NAX}"
      CMD=(sbatch --job-name=${NAME} --time=${TIME} --output=out/${NAME}-%j.log --error=out/${NAME}-%j.log
           --export=ALL,SCAN_TOL=${TOL},SCAN_ORD=${ORD},SCAN_FOURIER=${FOURIER},SCAN_NAX=${NAX}
           scripts/submit-vessels-scan-job.sh)
      if [ $DRY -eq 1 ]; then echo "${CMD[@]}"; else "${CMD[@]}"; fi
    done
  done
done
