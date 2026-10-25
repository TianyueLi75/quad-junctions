#pragma once
// Standing-wave deflation basis along a CURVED centerline (no PETSc dependency, self-contained).
// Each surface node is assigned s = arclength of its nearest point on a provided polyline centerline
// (end segments are extended linearly so hemispherical-cap nodes get s<0 / s>arc, like the straight
// axis projection does); s is shifted so min s = 0 and L = max s - min s. Columns (same layout/ordering
// as analytic_deflation.hpp): one net-flux column (outward normal) + per n=1..n_max, k=n*pi/L:
//   cols_per_n==20 : cos(k s) n_hat, sin(k s) n_hat           (normal-only, 2 cols per n)
//   cols_per_n==2  : cos(k s) n_hat, sin(k s) t_hat
//   cols_per_n==4  : cos n_hat, sin t_hat, sin n_hat, cos t_hat
// t_hat is the local centerline tangent. With an empty centerline the straight axis `axis` is used
// (s = projection onto the axis) -- this reproduces the straight-axis basis for A/B comparison.
#include <vector>
#include <cmath>
#include <algorithm>

namespace quad_junctions {

struct NodeAxial { std::vector<double> s, tx, ty, tz; double L = 0, R = 0; };

// CL: polyline points AoS (3*ncl). Returns per-node s (shifted, min=0), tangent, L, median radius.
inline NodeAxial curved_node_axial(const std::vector<double>& X0, long N, const std::vector<double>& CL,
                                   const double axis[3]) {
  const long np = N / 3;
  NodeAxial o; o.s.assign(np, 0); o.tx.assign(np, axis[0]); o.ty.assign(np, axis[1]); o.tz.assign(np, axis[2]);
  std::vector<double> rad(np, 0);
  const long ncl = (long)CL.size() / 3;
  if (ncl < 2) {   // straight-axis projection
    for (long p = 0; p < np; p++) {
      const double x = X0[3*p], y = X0[3*p+1], z = X0[3*p+2];
      o.s[p] = x*axis[0] + y*axis[1] + z*axis[2];
      rad[p] = std::sqrt(std::max(0.0, x*x + y*y + z*z - o.s[p]*o.s[p]));
    }
  } else {
    std::vector<double> cum(ncl, 0.0);   // cumulative arclength at points
    for (long i = 1; i < ncl; i++) {
      const double dx = CL[3*i]-CL[3*i-3], dy = CL[3*i+1]-CL[3*i-2], dz = CL[3*i+2]-CL[3*i-1];
      cum[i] = cum[i-1] + std::sqrt(dx*dx + dy*dy + dz*dz);
    }
    const long nseg = ncl - 1;
    std::vector<double> sl(nseg), ux(nseg), uy(nseg), uz(nseg);
    for (long i = 0; i < nseg; i++) {
      sl[i] = cum[i+1] - cum[i];
      const double inv = sl[i] > 0 ? 1.0/sl[i] : 0.0;
      ux[i] = (CL[3*i+3]-CL[3*i])*inv; uy[i] = (CL[3*i+4]-CL[3*i+1])*inv; uz[i] = (CL[3*i+5]-CL[3*i+2])*inv;
    }
    for (long p = 0; p < np; p++) {
      const double x = X0[3*p], y = X0[3*p+1], z = X0[3*p+2];
      double bd = 1e300, bs = 0; long bi = 0;
      for (long i = 0; i < nseg; i++) {
        if (sl[i] <= 0) continue;
        double t = (x-CL[3*i])*ux[i] + (y-CL[3*i+1])*uy[i] + (z-CL[3*i+2])*uz[i];
        if (i > 0) t = std::max(t, 0.0);               // extension only beyond the two ends
        if (i < nseg-1) t = std::min(t, sl[i]);
        const double qx = CL[3*i]+t*ux[i]-x, qy = CL[3*i+1]+t*uy[i]-y, qz = CL[3*i+2]+t*uz[i]-z;
        const double d2 = qx*qx + qy*qy + qz*qz;
        if (d2 < bd) { bd = d2; bs = cum[i] + t; bi = i; }
      }
      o.s[p] = bs; rad[p] = std::sqrt(bd);
      o.tx[p] = ux[bi]; o.ty[p] = uy[bi]; o.tz[p] = uz[bi];
    }
  }
  const double smin = *std::min_element(o.s.begin(), o.s.end());
  const double smax = *std::max_element(o.s.begin(), o.s.end());
  o.L = smax - smin;
  std::vector<double> rr;
  for (long p = 0; p < np; p++) { o.s[p] -= smin; if (o.s[p] >= 0.1*o.L && o.s[p] <= 0.9*o.L) rr.push_back(rad[p]); }
  if (rr.empty()) rr = rad;
  std::nth_element(rr.begin(), rr.begin() + rr.size()/2, rr.end());
  o.R = rr[rr.size()/2];
  return o;
}

inline long build_curved_modes(const std::vector<double>& X0, const std::vector<double>& Xn0, long N,
                               int n_max, int cols_per_n, const std::vector<double>& CL,
                               const double axis[3], std::vector<double>& Wcols) {
  const double PI = 3.14159265358979323846;
  const long np = N / 3;
  NodeAxial na = curved_node_axial(X0, N, CL, axis);
  const bool normal_only = (cols_per_n == 20);
  if (normal_only) cols_per_n = 2;
  const long Mraw = 1 + (long)n_max * cols_per_n;
  Wcols.assign((size_t)N * Mraw, 0.0);
  long col = 0;
  auto fill = [&](int kind, double k) {   // 0 flux; 1 cosN; 2 sinT; 3 sinN; 4 cosT
    double* w = &Wcols[(size_t)col * N];
    for (long p = 0; p < np; p++) {
      const double t3[3] = {na.tx[p], na.ty[p], na.tz[p]};
      for (int c = 0; c < 3; c++) {
        const double n = Xn0[3*p+c];
        double v;
        switch (kind) {
          case 0: v = n; break;
          case 1: v = std::cos(k*na.s[p]) * n; break;
          case 2: v = std::sin(k*na.s[p]) * t3[c]; break;
          case 3: v = std::sin(k*na.s[p]) * n; break;
          default: v = std::cos(k*na.s[p]) * t3[c]; break;
        }
        w[3*p+c] = v;
      }
    }
  };
  fill(0, 0); col++;
  for (int n = 1; n <= n_max; n++) {
    const double k = n * PI / na.L;
    const int kinds[4] = {1, 2, 3, 4};
    const int kn[2] = {1, 3};
    for (int q = 0; q < cols_per_n; q++) { fill(normal_only ? kn[q] : kinds[q], k); col++; }
  }
  return Mraw;
}

} // namespace quad_junctions
