#pragma once
// Analytic near-null deflation basis for the capsule interior Stokes operator (no PETSc dependency).
// Columns: net-flux (outward normal) + per axial wavenumber k_n=n*pi/L: cos*normal, sin*axis
// (and with cols_per_n==4: sin*normal, cos*axis). Origin is (0,0,0); axis is the unit axial direction.
#include <vector>
#include <cmath>
#include <algorithm>

namespace quad_junctions {

inline long build_analytic_modes(const std::vector<double>& X0, const std::vector<double>& Xn0,
                                 long N, int n_max, int cols_per_n /*2|4|20(normal-only)*/, const double axis[3],
                                 std::vector<double>& Wcols, std::vector<double>* lambda,
                                 double C_lub = 0.125, bool skip_flux = false) {
  const double PI = 3.14159265358979323846;
  const long np = N / 3;
  std::vector<double> proj(np), rad(np);
  for (long p = 0; p < np; p++) {
    const double x = X0[3*p], y = X0[3*p+1], z = X0[3*p+2];
    proj[p] = x*axis[0] + y*axis[1] + z*axis[2];
    rad[p] = std::sqrt(std::max(0.0, x*x + y*y + z*z - proj[p]*proj[p]));
  }
  const double smin = *std::min_element(proj.begin(), proj.end());
  const double smax = *std::max_element(proj.begin(), proj.end());
  const double L = smax - smin;
  std::vector<double> s(np), rr;
  for (long p = 0; p < np; p++) { s[p] = proj[p] - smin;
    if (s[p] >= 0.1*L && s[p] <= 0.9*L) rr.push_back(rad[p]); }
  if (rr.empty()) rr = rad;
  std::nth_element(rr.begin(), rr.begin() + rr.size()/2, rr.end());
  const double R = rr[rr.size()/2];

  // cols_per_n == 20: NORMAL-ONLY (cosN, sinN per n, no axial columns) -> 2 cols per n.
  const bool normal_only = (cols_per_n == 20);
  if (normal_only) cols_per_n = 2;
  const long Mraw = (skip_flux ? 0 : 1) + (long)n_max * cols_per_n;
  Wcols.assign((size_t)N * Mraw, 0.0);
  if (lambda) lambda->assign(Mraw, 0.0);
  long col = 0;
  auto fill = [&](int kind, double k) {   // kind 0 flux; 1 cosN; 2 sinX; 3 sinN; 4 cosX
    double* w = &Wcols[(size_t)col * N];
    for (long p = 0; p < np; p++) for (int c = 0; c < 3; c++) {
      double v;
      switch (kind) {
        case 0: v = Xn0[3*p+c]; break;
        case 1: v = std::cos(k*s[p]) * Xn0[3*p+c]; break;
        case 2: v = std::sin(k*s[p]) * axis[c]; break;
        case 3: v = std::sin(k*s[p]) * Xn0[3*p+c]; break;
        default: v = std::cos(k*s[p]) * axis[c]; break;
      }
      w[3*p+c] = v;
    }
  };
  if (!skip_flux) {
    fill(0, 0);
    if (lambda) (*lambda)[col] = C_lub * (PI*R/L) * (PI*R/L) * 1e-2;
    col++;
  }
  for (int n = 1; n <= n_max; n++) {
    const double k = n * PI / L;
    const double lam = C_lub * (k*R) * (k*R);
    const int kinds[4] = {1, 2, 3, 4};
    const int kn[2] = {1, 3};
    for (int q = 0; q < cols_per_n; q++) {
      fill(normal_only ? kn[q] : kinds[q], k);
      if (lambda) (*lambda)[col] = lam;
      col++;
    }
  }
  return Mraw;
}

} // namespace quad_junctions
