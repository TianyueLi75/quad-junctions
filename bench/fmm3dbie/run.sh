#!/bin/bash
# End-to-end S_init benchmark: a p=12 row of Table 1c (Greengard, O'Neil, Rachh et al. 2021)
# on the quad-junctions Y-bifurcation surface.
#
#   bench/fmm3dbie/run.sh
#
# Steps: (1) dump order-11 RV nodes, (2) export the closed all-quad Y-bifurcation as split-quad
# triangles -> ybifurc_p12.srcvals, (3) sweep eps and time the near-quadrature precompute.
# Headline S_init is single-core (paper-comparable); an 8-core run is added for context.
set -e
cd "$(dirname "$0")"
ROOT=$(cd ../.. && pwd)

# toolchain (gfortran + MKLROOT for the harness; SCTL stack for the exporter)
. "$ROOT/sctl_source" 2>/dev/null || true

echo "==================== 1. build ===================="
make
( cd "$ROOT" && make bin/ybifurc-export-fmm3dbie )

echo "==================== 2. RV nodes + export ===================="
./dump_rvnodes
( cd "$ROOT" && OMP_NUM_THREADS=8 ./bin/ybifurc-export-fmm3dbie \
      bench/fmm3dbie/rvnodes_o11.txt bench/fmm3dbie/ybifurc_p12.srcvals 12 2 \
      2>/dev/null )

echo "==================== 3a. S_init sweep (single core, Table-1c comparable) ===================="
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 ./s_init_sweep ybifurc_p12.srcvals \
    2>/dev/null | grep -vE "entering|beginning|starting|generate near" | tee s_init_p12_1core.txt

echo "==================== 3b. S_init sweep (8 cores, context) ===================="
OMP_NUM_THREADS=8 MKL_NUM_THREADS=1 ./s_init_sweep ybifurc_p12.srcvals \
    2>/dev/null | grep -vE "entering|beginning|starting|generate near" | tee s_init_p12_8core.txt
