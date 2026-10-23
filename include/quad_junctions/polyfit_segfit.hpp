// polyfit_segfit.hpp -- SegPolyFit: piecewise Chebyshev-polynomial arm-centerline body (QJ_ARM_POLYFIT).
// Faithful C++ port of python/polyfit_centerline.py (SegPolyArm / fit_arm_segmented / segment_knots /
// adaptive_body_window).  Self-contained (std only) so it can be unit-tested without sctl.
// S segments x 3 coords x (deg+1) Chebyshev coeffs in the LOCAL parameter u in [0,1] (x = 2u-1); segments are
// delimited by knot fractions f[0]=0 < ... < f[S]=1 of the body parameter tau (= polyline arclength fraction).
// The fit is a per-coordinate KKT constrained least squares (endpoints fixed, C^cont continuity at interior
// knots); the dense KKT system is solved with a plain partial-pivot LU in double (same as LAPACK gesv).
#pragma once
#include <vector>
#include <array>
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <functional>

namespace quad_junctions {
namespace pf {

using V3 = std::array<double, 3>;

// T_k^{(m)}(x), k=0..deg, m=0..mmax, via  T_{k+1}^{(m)} = 2x T_k^{(m)} + 2m T_k^{(m-1)} - T_{k-1}^{(m)}.
// out[m][k]
inline void cheb_derivs(double x, int deg, int mmax, std::vector<std::vector<double>>& out) {
  out.assign(mmax + 1, std::vector<double>(deg + 1, 0.0));
  for (int m = 0; m <= mmax; m++) {
    for (int k = 0; k <= deg; k++) {
      double v;
      if (k == 0) v = (m == 0) ? 1.0 : 0.0;
      else if (k == 1) v = (m == 0) ? x : (m == 1 ? 1.0 : 0.0);
      else {
        v = 2.0 * x * out[m][k - 1] - out[m][k - 2];
        if (m > 0) v += 2.0 * m * out[m - 1][k - 1];
      }
      out[m][k] = v;
    }
  }
}

// row of the m-th derivative wrt LOCAL u in [0,1] of each basis fn at u (includes the 2^m chain factor).
inline std::vector<double> locbasis(double u, int deg, int m) {
  std::vector<std::vector<double>> d; cheb_derivs(2.0 * u - 1.0, deg, m, d);
  std::vector<double> row = d[m]; const double sc = std::ldexp(1.0, m);
  for (auto& v : row) v *= sc;
  return row;
}

// Dense solve K X = RHS (RHS has nrhs columns) by partial-pivot LU. K is n x n row-major; returns false if singular.
inline bool lu_solve(std::vector<double> K, int n, std::vector<std::vector<double>>& rhs) {
  std::vector<int> piv(n);
  for (int c = 0; c < n; c++) {
    int p = c; double mx = std::fabs(K[(size_t)c * n + c]);
    for (int r = c + 1; r < n; r++) { double a = std::fabs(K[(size_t)r * n + c]); if (a > mx) { mx = a; p = r; } }
    if (mx < 1e-300) return false;
    piv[c] = p;
    if (p != c) { for (int k = 0; k < n; k++) std::swap(K[(size_t)c * n + k], K[(size_t)p * n + k]);
                  for (auto& b : rhs) std::swap(b[c], b[p]); }
    for (int r = c + 1; r < n; r++) {
      const double f = K[(size_t)r * n + c] / K[(size_t)c * n + c];
      if (f == 0.0) continue;
      for (int k = c; k < n; k++) K[(size_t)r * n + k] -= f * K[(size_t)c * n + k];
      for (auto& b : rhs) b[r] -= f * b[c];
    }
  }
  for (auto& b : rhs)
    for (int r = n - 1; r >= 0; r--) {
      double s = b[r];
      for (int k = r + 1; k < n; k++) s -= K[(size_t)r * n + k] * b[k];
      b[r] = s / K[(size_t)r * n + r];
    }
  return true;
}

inline std::vector<double> arclen(const std::vector<V3>& P) {
  std::vector<double> s(P.size(), 0.0);
  for (size_t k = 1; k < P.size(); k++) {
    double d = 0; for (int c = 0; c < 3; c++) { double e = P[k][c] - P[k - 1][c]; d += e * e; }
    s[k] = s[k - 1] + std::sqrt(d);
  }
  return s;
}

// np.interp(x, xp, fp)
inline double np_interp(double x, const std::vector<double>& xp, const std::vector<double>& fp) {
  if (x <= xp.front()) return fp.front();
  if (x >= xp.back()) return fp.back();
  size_t j = (size_t)(std::upper_bound(xp.begin(), xp.end(), x) - xp.begin()) - 1;
  if (j + 1 >= xp.size()) return fp.back();
  const double dx = xp[j + 1] - xp[j];
  if (dx == 0.0) return fp[j];
  return fp[j] + (x - xp[j]) / dx * (fp[j + 1] - fp[j]);
}

// segment_knots(): S from length rule and turning rule; knots at equal cumulative turning. force_single => S=1.
inline std::vector<double> segment_knots(const std::vector<V3>& body, double Lseg, double turn_seg_deg,
                                         bool force_single, double* total_turn_deg_out = nullptr) {
  const std::vector<double> s = arclen(body); const double L = s.back();
  const size_t N = body.size();
  std::vector<V3> tn;
  for (size_t k = 0; k + 1 < N; k++) {
    V3 t; double n = 0; for (int c = 0; c < 3; c++) { t[c] = body[k + 1][c] - body[k][c]; n += t[c] * t[c]; }
    n = std::sqrt(n); if (n < 1e-300) n = 1e-300; for (int c = 0; c < 3; c++) t[c] /= n; tn.push_back(t);
  }
  std::vector<double> cturn(1, 0.0);
  for (size_t k = 0; k + 1 < tn.size(); k++) {
    double d = tn[k][0] * tn[k + 1][0] + tn[k][1] * tn[k + 1][1] + tn[k][2] * tn[k + 1][2];
    d = std::max(-1.0, std::min(1.0, d)); cturn.push_back(cturn.back() + std::acos(d));
  }
  const double tot_deg = cturn.back() * 180.0 / M_PI;
  if (total_turn_deg_out) *total_turn_deg_out = tot_deg;
  int S = std::max({(int)std::ceil(L / Lseg), (int)std::ceil(tot_deg / turn_seg_deg), 1});
  if (force_single) S = 1;
  std::vector<double> f;
  if (S > 1 && cturn.back() > 1e-9) {
    std::vector<double> svtx(s.begin() + 1, s.end());       // arclength at vertices where cturn is defined
    for (int i = 0; i <= S; i++) f.push_back(np_interp(cturn.back() * (double)i / (double)S, cturn, svtx) / L);
  } else {
    for (int i = 0; i <= S; i++) f.push_back((double)i / (double)S);
  }
  f.front() = 0.0; f.back() = 1.0;
  for (auto& v : f) v = std::max(0.0, std::min(1.0, v));
  std::sort(f.begin(), f.end()); f.erase(std::unique(f.begin(), f.end()), f.end());
  return f;
}

struct SegPolyFit {
  int deg = 9, S = 0, cont = 4;
  std::vector<double> f;                               // S+1 knot fractions
  std::vector<double> coef;                            // [(seg*3 + coord)*(deg+1) + k]
  double L = 0;                                        // polyline arclength of the fitted body
  double total_turn_deg = 0;
  bool ok = false;

  int piece(double t) const {
    int i = (int)(std::upper_bound(f.begin(), f.end(), t) - f.begin()) - 1;
    return std::max(0, std::min(S - 1, i));
  }
  // order-th derivative wrt tau in [0,1] (order 0 = position)
  V3 deriv(double t, int order) const {
    const int i = piece(t); const double h = f[i + 1] - f[i], u = (t - f[i]) / h;
    std::vector<std::vector<double>> d; cheb_derivs(2.0 * u - 1.0, deg, order, d);
    const double sc = std::ldexp(1.0, order) / std::pow(h, order);
    V3 o{0, 0, 0};
    for (int c = 0; c < 3; c++) { double a = 0; const double* cc = &coef[((size_t)i * 3 + c) * (deg + 1)];
      for (int k = 0; k <= deg; k++) a += cc[k] * d[order][k]; o[c] = a * sc; }
    return o;
  }
  V3 eval(double t) const { return deriv(t, 0); }
  // min over the sampled grid of rho/r; also returns the grid minima over [0,frac) and (1-frac,1] if asked.
  std::vector<double> rho_over_r(double r, int n = 1200) const {
    std::vector<double> out(n);
    for (int q = 0; q < n; q++) {
      const double t = (double)q / (double)(n - 1);
      const V3 d1 = deriv(t, 1), d2 = deriv(t, 2);
      const V3 cr{d1[1] * d2[2] - d1[2] * d2[1], d1[2] * d2[0] - d1[0] * d2[2], d1[0] * d2[1] - d1[1] * d2[0]};
      const double num = std::sqrt(cr[0] * cr[0] + cr[1] * cr[1] + cr[2] * cr[2]);
      const double sp = std::sqrt(d1[0] * d1[0] + d1[1] * d1[1] + d1[2] * d1[2]);
      const double den = std::max(sp * sp * sp, 1e-300);
      const double kappa = num > 1e-14 ? num / den : 0.0;
      out[q] = (kappa > 1e-12 ? 1.0 / kappa : INFINITY) / r;
    }
    return out;
  }

  // fit_arm_segmented(): knots from segment_knots unless `knots` non-empty.  p0,p1 = fixed end positions.
  // end_kappa0: pin the body-poly's SECOND derivative (wrt tau) to zero at both window ends, so its end curvature
  // is zero -- matches DOWNSAMPLE's natural-C2 cubic ends and makes the corner->body join C2 (the zero-accel
  // quintic corner also has kappa=0 there). A linear constraint (RHS 0), so the 1/h^m scale is immaterial.
  bool init(const std::vector<V3>& body, const V3& p0, const V3& p1, double Lseg, double turn_seg_deg,
            int deg_, int cont_, bool force_single, const std::vector<double>& knots = {}, int n_fit_per_seg = 24,
            bool end_kappa0 = false) {
    deg = deg_; cont = cont_; ok = false;
    const std::vector<double> s = arclen(body); L = s.back();
    if (body.size() < 3 || L <= 0) return false;
    std::vector<double> t(body.size()); for (size_t k = 0; k < t.size(); k++) t[k] = s[k] / L;
    f = knots.empty() ? segment_knots(body, Lseg, turn_seg_deg, force_single, &total_turn_deg) : knots;
    S = (int)f.size() - 1; const int nc = deg + 1; const int ncol = S * nc;
    std::vector<int> pc(body.size());
    for (size_t k = 0; k < t.size(); k++) {
      int i = (int)(std::upper_bound(f.begin(), f.end(), t[k]) - f.begin()) - 1; pc[k] = std::max(0, std::min(S - 1, i)); }
    // data rows (A) as (row index into body, packed row)
    std::vector<std::vector<double>> Arows; std::vector<int> Aidx;
    for (int i = 0; i < S; i++) {
      std::vector<int> idx; for (size_t k = 0; k < t.size(); k++) if (pc[k] == i) idx.push_back((int)k);
      if ((int)idx.size() < deg + 2) {                     // borrow nearest points for thin segments
        const double mid = 0.5 * (f[i] + f[i + 1]);
        std::vector<int> all(t.size()); for (size_t k = 0; k < all.size(); k++) all[k] = (int)k;
        std::stable_sort(all.begin(), all.end(), [&](int a, int b) { return std::fabs(t[a] - mid) < std::fabs(t[b] - mid); });
        idx.assign(all.begin(), all.begin() + std::min<size_t>(deg + 3, all.size()));
      }
      if (n_fit_per_seg && (int)idx.size() > n_fit_per_seg) {
        const int n = (int)idx.size(); const double step = (double)(n - 1) / (double)(n_fit_per_seg - 1);
        std::vector<int> sel;
        for (int k = 0; k < n_fit_per_seg; k++) { double y = (k == n_fit_per_seg - 1) ? (double)(n - 1) : (double)k * step;
          int q = (int)y; if (sel.empty() || sel.back() != q) sel.push_back(q); }
        std::vector<int> ni; for (int q : sel) ni.push_back(idx[q]); idx.swap(ni);
      }
      const double h = f[i + 1] - f[i];
      for (int ii : idx) {
        const double u = (t[ii] - f[i]) / h; std::vector<double> row(ncol, 0.0);
        std::vector<double> bb = locbasis(u, deg, 0);
        for (int k = 0; k < nc; k++) row[(size_t)i * nc + k] = bb[k];
        Arows.push_back(row); Aidx.push_back(ii);
      }
    }
    // constraints
    std::vector<std::vector<double>> B; std::vector<std::array<double, 3>> dv;
    { std::vector<double> r0(ncol, 0.0); auto b0 = locbasis(0.0, deg, 0); for (int k = 0; k < nc; k++) r0[k] = b0[k];
      B.push_back(r0); dv.push_back({p0[0], p0[1], p0[2]}); }
    { std::vector<double> r1(ncol, 0.0); auto b1 = locbasis(1.0, deg, 0); for (int k = 0; k < nc; k++) r1[(size_t)(S - 1) * nc + k] = b1[k];
      B.push_back(r1); dv.push_back({p1[0], p1[1], p1[2]}); }
    if (end_kappa0 && S >= 1) {   // zero end curvature: r''(tau)=0 at both window ends (natural-C2 body, C2 to the corner)
      { std::vector<double> r(ncol, 0.0); auto b = locbasis(0.0, deg, 2); for (int k = 0; k < nc; k++) r[k] = b[k];
        B.push_back(r); dv.push_back({0, 0, 0}); }
      { std::vector<double> r(ncol, 0.0); auto b = locbasis(1.0, deg, 2); for (int k = 0; k < nc; k++) r[(size_t)(S - 1) * nc + k] = b[k];
        B.push_back(r); dv.push_back({0, 0, 0}); }
    }
    for (int i = 1; i < S; i++) {
      const double hL = f[i] - f[i - 1], hR = f[i + 1] - f[i];
      for (int m = 0; m <= cont; m++) {
        std::vector<double> row(ncol, 0.0);
        auto bl = locbasis(1.0, deg, m), br = locbasis(0.0, deg, m);
        for (int k = 0; k < nc; k++) { row[(size_t)(i - 1) * nc + k] = bl[k] / std::pow(hL, m);
                                       row[(size_t)i * nc + k] = -br[k] / std::pow(hR, m); }
        B.push_back(row); dv.push_back({0, 0, 0});
      }
    }
    const int mB = (int)B.size(); const int n = ncol + mB;
    std::vector<double> K((size_t)n * n, 0.0);
    for (const auto& row : Arows) for (int a = 0; a < ncol; a++) { if (row[a] == 0.0) continue;
      for (int b = 0; b < ncol; b++) K[(size_t)a * n + b] += 2.0 * row[a] * row[b]; }
    for (int q = 0; q < mB; q++) for (int a = 0; a < ncol; a++) { K[(size_t)(ncol + q) * n + a] = B[q][a]; K[(size_t)a * n + ncol + q] = B[q][a]; }
    std::vector<std::vector<double>> rhs(3, std::vector<double>(n, 0.0));
    for (int c = 0; c < 3; c++) {
      for (size_t q = 0; q < Arows.size(); q++) { const double y = body[Aidx[q]][c];
        for (int a = 0; a < ncol; a++) rhs[c][a] += 2.0 * Arows[q][a] * y; }
      for (int q = 0; q < mB; q++) rhs[c][ncol + q] = dv[q][c];
    }
    if (!lu_solve(K, n, rhs)) return false;
    coef.assign((size_t)S * 3 * nc, 0.0);
    for (int c = 0; c < 3; c++) for (int i = 0; i < S; i++) for (int k = 0; k < nc; k++)
      coef[((size_t)i * 3 + c) * nc + k] = rhs[c][(size_t)i * nc + k];
    ok = true; return true;
  }
};

// adaptive_body_window(): push an end deeper while its fitted rho/r (first/last end_frac of t) < rho_tol.
// sm = smoothed full centerline vertices; room = Llead + k*r.  Returns false if no valid window even at base_k.
struct PfWindow { int iA = 0, iB = 0; double kA = 0, kB = 0; };
inline bool adaptive_body_window(const std::vector<V3>& sm, double r, double Llead, SegPolyFit& arm, PfWindow& w,
                                 double base_k, double step_k, double cap_k, double rho_tol, double end_frac,
                                 int deg, int cont, double Lseg, double turn_seg_deg, bool force_single, bool end_kappa0,
                                 // optional window-index providers (room -> vertex index). Default (empty) = python reference:
                                 // arclength from the polyline ends.  The C++ arm builder supplies AXIAL-DEPTH providers
                                 // (projection past the mouth), because the vmtk centerline starts a few r BEHIND the seam
                                 // centre, so arclength-from-cl[0] can land behind the lead stub (=> hooked corner / fold).
                                 const std::function<int(double)>& idxA = nullptr,
                                 const std::function<int(double)>& idxB = nullptr,
                                 // optional extra per-end acceptance test (e.g. the mouth QUINTIC CORNER rho/r), AND-ed with
                                 // the body rho/r test; returns {okA, okB}. Same push-deeper reaction.
                                 const std::function<std::pair<bool,bool>(const SegPolyFit&, const PfWindow&)>& extra = nullptr) {
  const std::vector<double> s = arclen(sm); const double L = s.back(); const int N = (int)sm.size();
  auto fit_with = [&](double kA, double kB, SegPolyFit& out, PfWindow& ww) -> bool {
    int iA = idxA ? idxA(Llead + kA * r) : (int)(std::lower_bound(s.begin(), s.end(), Llead + kA * r) - s.begin());
    int iB = idxB ? idxB(Llead + kB * r) : (int)(std::lower_bound(s.begin(), s.end(), L - (Llead + kB * r)) - s.begin());
    iA = std::max(iA, 2); iB = std::min(iB, N - 3);
    if (iB - iA < std::max(8, deg + 3)) return false;
    std::vector<V3> body(sm.begin() + iA, sm.begin() + iB + 1);
    if (!out.init(body, body.front(), body.back(), Lseg, turn_seg_deg, deg, cont, force_single, {}, 24, end_kappa0)) return false;
    ww.iA = iA; ww.iB = iB; ww.kA = kA; ww.kB = kB; return true;
  };
  double kA = base_k, kB = base_k;
  if (!fit_with(kA, kB, arm, w)) return false;
  for (int it = 0; it < 12; it++) {
    const std::vector<double> rr = arm.rho_over_r(r, 1200);
    double loA = INFINITY, loB = INFINITY;
    for (int q = 0; q < 1200; q++) { const double t = (double)q / 1199.0;
      if (t < end_frac) loA = std::min(loA, rr[q]);
      if (t > 1.0 - end_frac) loB = std::min(loB, rr[q]); }
    std::pair<bool,bool> ex{true, true}; if (extra) ex = extra(arm, w);
    const bool okA = (loA >= rho_tol && ex.first) || kA >= cap_k, okB = (loB >= rho_tol && ex.second) || kB >= cap_k;
    if (okA && okB) break;
    if (!okA) kA = std::min(cap_k, kA + step_k);
    if (!okB) kB = std::min(cap_k, kB + step_k);
    SegPolyFit nx; PfWindow nw;
    if (!fit_with(kA, kB, nx, nw)) break;
    arm = nx; w = nw;
  }
  return true;
}

}  // namespace pf
}  // namespace quad_junctions
