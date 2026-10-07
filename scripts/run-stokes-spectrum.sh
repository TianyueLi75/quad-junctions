#!/bin/bash
# =============================================================================
# run-stokes-spectrum.sh
#
# Stokes slender-body BIE spectrum study: Figure 7 / Figure 8 analogs of Malhotra & Barnett 2024 (CSBQ),
# for the INTERIOR combined-field convention this project's flow solves invert.
#
# Dumps the raw pure single- and double-layer operator matrices (<prefix>s, <prefix>d) for
#   (a) the paper's constant-radius torus  (csbq-spectrum, geom=2, R=1, 8 panels, Fourier 20), and
#   (b) our capsule-with-caps               (capsule-robin-bie QJ_SPECTRUM_RAW),
# matched at the same slenderness eps, then runs python/plot_stokes_spectrum.py which forms
#   S,  D_int = -1/2 I + D,  K(eta) = -1/2 I + D - eta S
# and produces the singular-value / eigenvalue grids + the kappa(eta) scaling law
# (optimum at eta* = 1/(2 eps log 1/eps)).
#
# Serial, dense N x N matrices (numpy does the eig/svd) -- keep resolution modest; the conditioning is
# 2nd-kind / mesh-independent. QJ_SPECTRUM_MAXDOF guards the capsule dense size.
# =============================================================================
set -euo pipefail
cd "$(dirname "$0")/.."

. ./sctl_source

make MPI=0 bin/csbq-spectrum bin/capsule-robin-bie

OUT=vis/stokes-spectrum
mkdir -p "$OUT"

export OMP_NUM_THREADS=${OMP_NUM_THREADS:-10}
export OMP_PROC_BIND=close
export OMP_PLACES=cores
export QJ_SPECTRUM_MAXDOF=${QJ_SPECTRUM_MAXDOF:-16000}
QUAD_EPS=${QUAD_EPS:-1e-12}

# ---- (a) torus: constant radius (geom=2), R=1, 8 panels, Fourier 20 ; eps sweep -----------------
# Set QJ_COND_SWEEP=1 to also print the (slow) paper-exterior kappa(eta) table from the C++ side.
for EPS in 1e-2 1e-4 1e-6; do
  echo "=== torus eps=$EPS ==="
  QJ_MAT_PREFIX="$OUT/torus_eps${EPS}" \
    ./bin/csbq-spectrum -ker 2 -geom 2 -eps "$EPS" -quad_eps "$QUAD_EPS" -omp "$OMP_NUM_THREADS"
done

# ---- (b) capsule: straight tube + 2 caps, matched to the torus. Cross-sectional radius = eps (the
#         torus minor radius), centerline length fixed ~2pi (the torus circumference), so eps has the
#         SAME meaning in eta* = 1/(2 eps log 1/eps). Conditioning is 2nd-kind / mesh-independent, so a
#         coarse mesh (cap order 4, Fourier 12) is used to land N = 4800 = the torus DOF (apples-to-apples
#         dense SVD/eig). Cap order 4 => DL identity ~1e-4, plenty for a conditioning study. ------------
LEN=6.283185307
cap () {  # eps R n_axial   (L fixed = LEN)
  local eps=$1 R=$2 na=$3
  echo "=== capsule eps=$eps (R=$R L=$LEN n_axial=$na aspect=$(python -c "print(f'{$LEN/$R:.0f}')")) ==="
  QJ_SPECTRUM_RAW="$OUT/cap_eps${eps}" QJ_SPIRAL_AMP=0 \
    ./bin/capsule-robin-bie geom 4 "$R" "$LEN" "$na" 12 2 1e-10
}
cap 1e-2 0.01 8     # matched to torus eps=1e-2 (aspect ~628): the direct overlay
cap 3e-2 0.03 8     # less slender (aspect ~209): a second scaling-scatter point

echo
echo "=== plots ==="
# numpy/matplotlib live in the python Lmod module; run the plotter in a subshell so it doesn't perturb
# the compile toolchain loaded by ./sctl_source.
( module load python/3.11.11 && python python/plot_stokes_spectrum.py "$OUT" )
