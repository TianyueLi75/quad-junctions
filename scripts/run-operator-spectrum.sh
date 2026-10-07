#!/bin/bash
# =============================================================================
# run-operator-spectrum.sh
#
# Singular-value spectra of the coupled SL+DL self operator (the GMRES function)
#     A = jump*I + SL_scal*S + DL_scal*D,   jump = -0.5*DL_scal  (interior CFIE, SL=-1, DL=+1)
# assembled DENSE (unit-vector apply) and SVD'd via sctl::Matrix::SVD by the QJ_SPECTRUM gate in
# each driver (include/quad_junctions/operator_spectrum.hpp). One unified convention across all
# geometries => the spectra are directly comparable.
#
# Serial, dense N x N SVD -> keep the resolution COARSE (justified: the conditioning is 2nd-kind /
# mesh-independent, see the aspect study). QJ_SPECTRUM_MAXDOF guards the dense size.
#
# Base six geometries:
#   ybifurc-hybrid, racetrack (arm ~5 and ~10), straight capsule (L=1 and L=10, R=0.3), spiral amp0.5.
# Panel study (same-count-across-length vs refined-at-fixed-length):
#   capsule L1/L10 both n_axial=8 (same count); capsule L10 n_axial 8/16/24 (refine, fixed geometry);
#   racetrack L5/L10 fixed nBentA (same count) vs auto nBentA=-1 (panels scale with length).
#
# Output: one CSV per case in $OUT (# header carries cond/smax/smin; then index,sigma rows).
# =============================================================================
set -euo pipefail
cd "$(dirname "$0")/.."

. ./sctl_source

make MPI=0 bin/capsule-robin-bie bin/ybifurc-channel-bie bin/ybifurc-hybrid-bie

OUT=vis/spectrum
mkdir -p "$OUT"

# serial; OpenMP pinned to physical cores (SVD + assembly are the cost)
export OMP_NUM_THREADS=10
export OMP_PROC_BIND=close
export OMP_PLACES=cores
export QJ_SPECTRUM_MAXDOF=${QJ_SPECTRUM_MAXDOF:-16000}

echo "=== operator spectra (interior CFIE, SL=-1 DL=+1); MAXDOF=$QJ_SPECTRUM_MAXDOF ==="

# ---- capsule: <label> <R> <L> <n_axial> <amp> ----------------------------------------------------
# order=8 fourier=8 Ncap=2 tol=1e-8 (coarse; n_axial fixes shaft DOF independent of length, so
# L1_na8 vs L10_na8 = SAME panel count, and L10 na 8/16/24 = mesh refinement at fixed geometry).
cap () {  # label R L na amp
  local label=$1 R=$2 L=$3 na=$4 amp=$5
  echo "--- capsule $label (R=$R L=$L n_axial=$na amp=$amp) ---"
  QJ_SPECTRUM="$OUT/$label.csv" QJ_SPIRAL_AMP=$amp QJ_SPIRAL_TURNS=1 \
    ./bin/capsule-robin-bie flow 8 "$R" "$L" "$na" 8 2 1e-8
}
# cap cap_L1_na8       0.3 1  8  0      # base: straight L=1
cap cap_L10_na8      0.3 10 8  0      # base: straight L=10 (same panel count as L=1)
# cap cap_L10_na16     0.3 10 16 0      # refine L=10 (mesh-independence check)
# cap cap_L10_na24     0.3 10 24 0      # refine L=10 further
# cap cap_L10_na12     0.3 10 12 0      # straight partner for the spiral (same n_axial)
# cap cap_spiral_amp0p5 0.3 10 12 0.5   # curvature: spiral, same L/R/n_axial as cap_L10_na12

# ---- racetrack: sep sets arm length; nBentA (-1=auto, panels scale with length; >0 = fixed) -------
# coarse: order 4, nref 0, fourier 6, md 12, cov_q 6, Nbeta 48 (the junction bodies dominate DOF: at
# order 8/nref 1 this is >100k DOF -- order 4/nref 0 keeps the dense SVD ~10k, still 2nd-kind-faithful).
# fixed nBentA is the TOTAL bent-arm panel count; the arm reserves lead 2 + corner 6 each side, so the
# constraint is 2*(2+6)=16 < nBentA. Use nBentA=20 => run=4 panels for BOTH L5 and L10 (same count,
# so the per-panel length grows with arm length); auto (-1) instead holds panel SIZE (count scales).
race () {  # label sep nBentA
  local label=$1 sep=$2 nb=$3
  echo "--- racetrack $label (sep=$sep nBentA=$nb) ---"
  QJ_SPECTRUM="$OUT/$label.csv" \
    ./bin/ybifurc-channel-bie flow 1.5 4 0 0.4 3 0.88 3 6 1e-8 48 12 6 "$sep" 25 "$nb"
}
# race race_L5_auto   5  -1     # arm length ~5,  panels auto (panel size fixed, count scales w/ length)
# race race_L10_auto  10 -1     # arm length ~10, panels auto
# race race_L5_fixed  5  20     # arm length ~5,  fixed 20 panels (same count as L10_fixed)
# race race_L10_fixed 10 20     # arm length ~10, fixed 20 panels (per-panel length grows w/ length)

# ---- ybifurc-hybrid (single junction body + 3 slender arms) --------------------------------------
# echo "--- ybifurc-hybrid ---"
# args: level tol order nref eta_join Ns_trans s_cap n_axial fourier nlev pou_kind arm_kind
# order 8 / nref 0 / fourier 8 => ~5k DOF (order 8 / nref 1 / fourier 12 is 60k, too big for dense SVD).
# QJ_SPECTRUM="$OUT/hybrid.csv" ./bin/ybifurc-hybrid-bie 1.5 0 8 0 0.4 3 0.88 3 8 4 1 0

echo
echo "=== cond summary ($OUT) ==="
grep -H "^# cond=" "$OUT"/*.csv | sed 's#'"$OUT"'/##; s/.csv:# / : /'
echo
echo "Plot + qualitative comparison:  python python/plot_spectrum.py $OUT"
