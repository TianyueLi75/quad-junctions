#!/bin/bash
# Self-contained build for bench-scheme-compare, mirroring the quad-junctions Makefile recipe.
#
# Deliberately does NOT touch the top-level Makefile: the whole scheme-comparison benchmark lives
# under bench/scheme-compare/. Builds into bench/scheme-compare/bin.<TAG>/ so an Icelake binary
# (AVX-512) and a Zen binary never collide on the shared home filesystem (same reasoning as the
# sbatch scripts' bin.<TAG>/ split).
#
#   bash bench/scheme-compare/build.sh [TAG]      # TAG defaults to $SLURM_JOB_CONSTRAINTS or "native"
#
# The driver #includes <sctl/experimental/quad_element.cpp> directly, so this single compile also
# exercises (and validates) the ported near/self quadrature templates.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/../.." && pwd)"                       # quad-junctions repo root
TAG="${1:-${SLURM_JOB_CONSTRAINTS:-native}}"
cd "$ROOT"

# Toolchain (gcc-13 / FFTW / OpenMPI / MKL) -- source only if not already in the environment.
[ -n "${MKLROOT:-}" ] || source ./sctl_source

mkdir -p "$HERE/bin.$TAG"

CXX=g++
# ORDER IS LOAD-BEARING: -I include must precede -I .../SCTL/include so this repo's ~3000-line
# vendored quad_element shadows upstream SCTL's older copy (see the top-level Makefile / CLAUDE.md).
INCLUDES="-I include -I extern/CSBQ/include -I extern/pvfmm/SCTL/include"
CXXFLAGS="-std=c++17 -fopenmp -Wall -Wfloat-conversion -O3 -march=native -DNDEBUG"
CXXFLAGS="$CXXFLAGS -gdwarf-4 -g -rdynamic -ldl -mno-avx512fp16"
CXXFLAGS="$CXXFLAGS -DSCTL_GLOBAL_MEM_BUFF=0 -DSCTL_PROFILE=25 -DSCTL_VERBOSE -DSCTL_SIG_HANDLER"
CXXFLAGS="$CXXFLAGS -DSCTL_QUAD_T=__float128 -DSCTL_DATA_PATH=./data"
CXXFLAGS="$CXXFLAGS -lmkl_intel_lp64 -lmkl_sequential -lmkl_core -lpthread -DSCTL_HAVE_BLAS -DSCTL_HAVE_LAPACK"
CXXFLAGS="$CXXFLAGS -lfftw3_omp -DSCTL_FFTW_THREADS -lfftw3 -DSCTL_HAVE_FFTW -lfftw3f -DSCTL_HAVE_FFTWF -lfftw3l -DSCTL_HAVE_FFTWL"

$CXX $CXXFLAGS $INCLUDES "$HERE/bench-scheme-compare.cpp" -o "$HERE/bin.$TAG/bench-scheme-compare"
echo "built: $HERE/bin.$TAG/bench-scheme-compare"
