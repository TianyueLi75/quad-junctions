#pragma once
// ---------------------------------------------------------------------------
// nearnull_heuristic.hpp -- aspect-ratio -> near-null-count heuristic for sizing the
// standing-wave deflation basis (nev / per-arm n_max) of the interior combined-field
// Stokes operator on slender capped tubes.
//
// The interior operator A = -1/2 I + D - (R/L) S on a slender capped tube has a
// near-null spectrum made of (i) one flux / incompressibility mode (|lambda| ~ 0) and
// (ii) a ladder of axial standing-wave pressure modes with
//        |lambda_n| ~ C (k_n R)^2 ,   k_n = n*pi/L ,   n = 1, 2, 3, ...
// so the number of modes below a threshold theta grows LINEARLY in the aspect ratio
// aspect = L/R:  count = 1 + #{ n>=1 : C (n pi R / L)^2 < theta }
//                      = 1 + floor( aspect * sqrt(theta) / (pi * sqrt(C)) ).
//
// EMPIRICAL CALIBRATION (straight capsule; dgeev of the dense interior operator; order 4
// AND order 8 -- resolution-independent; R in [0.02, 0.3], L in [2.5, 10], tol 1e-8;
// threshold |lambda| < 1e-3):
//        aspect :  16.7  25  33  50  67  100  133  167  200  250  500
//     count<1e-3 :    1   1   1   2   2    3    4    5    6    7   15
// L-varied cross-checks (aspect fixed, R and L both changed) land on the same count, i.e.
// aspect = L/R is the sole governing parameter. The fit C ~ 0.128 (= the lubrication C_lub
// = 0.125, to within the data) reproduces the whole table to within +/-1 mode. Because C is
// the theta-independent dispersion constant, the same formula sizes the basis for ANY
// threshold: count(theta) = 1 + floor(aspect * sqrt(theta)/(pi*sqrt(C))). The DEFAULT
// threshold is theta = 0.05 (a looser near-null band than the 1e-3 the table was tabulated
// at -- it deflates the shallow standing-wave rungs up to |lambda| ~ 0.05, ~sqrt(50) more
// modes per arm, matching the n_max ~ 8 that was effective on the aspect-44 racetrack arms).
//
// Usage:  per-arm block n_max = nearnull_nmax(L_arm / R_arm)         (rungs, no flux)
// ---------------------------------------------------------------------------
#include <vector>
#include <cmath>
#include <algorithm>

namespace quad_junctions {

// Dispersion constant C in |lambda_n| ~ C (k_n R)^2 (fit to the straight-capsule data above;
// equals the lubrication C_lub = 0.125 within the fit). Override at build time if desired.
#ifndef QJ_NEARNULL_DISP_C
#define QJ_NEARNULL_DISP_C 0.128
#endif

// Number of axial standing-wave rungs (n >= 1) with |lambda| < theta. This is the per-arm
// block n_max in the layered multi-arm deflation basis (the flux mode is a separate column).
inline long nearnull_nmax(double aspect, double theta = 0.05) {
  if (aspect <= 0.0 || theta <= 0.0) return 0;
  const double PI = 3.14159265358979323846;
  const double slope = std::sqrt(theta) / (PI * std::sqrt((double)QJ_NEARNULL_DISP_C));
  const long n = (long)std::floor(aspect * slope);
  return n < 0 ? 0 : n;
}

// Aspect ratio L/R of a capped tube from its collocation nodes X0 (AoS, length 3*np) and a
// unit axis: L = axial extent (max-min of the axial projection), R = median off-axis radius
// over the middle 80% of the axial span (so the caps don't bias R). Mirrors the L/R estimate
// in multiarm_deflation.hpp so a single-arm caller can size nev straight from <A>.X0.
inline double capsule_aspect_from_nodes(const std::vector<double>& X0, const double axis[3]) {
  const long np = (long)X0.size() / 3;
  if (np <= 0) return 0.0;
  std::vector<double> proj(np), rad(np);
  for (long p = 0; p < np; p++) {
    const double x = X0[3*p], y = X0[3*p+1], z = X0[3*p+2];
    proj[p] = x*axis[0] + y*axis[1] + z*axis[2];
    rad[p]  = std::sqrt(std::max(0.0, x*x + y*y + z*z - proj[p]*proj[p]));
  }
  const double smin = *std::min_element(proj.begin(), proj.end());
  const double smax = *std::max_element(proj.begin(), proj.end());
  const double L = smax - smin;
  if (L <= 0.0) return 0.0;
  std::vector<double> rr;
  for (long p = 0; p < np; p++) {
    const double s = proj[p] - smin;
    if (s >= 0.1*L && s <= 0.9*L) rr.push_back(rad[p]);
  }
  if (rr.empty()) rr = rad;
  std::nth_element(rr.begin(), rr.begin() + rr.size()/2, rr.end());
  const double R = rr[rr.size()/2];
  return (R > 0.0) ? L / R : 0.0;
}

} // namespace quad_junctions
