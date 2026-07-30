/**
 * Y-bifurcation iso-surface geometry primitives for high-order QuadElemList meshes.
 *
 * The shared kernel extracted from SCTL_quad_element/src/test-ybifurc-geom.cpp (the fork's
 * production port of bifurcation_geom/ybifurc.py): the sum-of-Gaussians Y field (`YField`) and
 * the SWEPT-O-GRID helpers that project its iso-surface f = level via fixed-RAY root solves
 * (star-shaped, NOT gradient-flow Newton) -- `ray_root`, `vslerp`, `arm_frame`, `junction_dir`,
 * `arm_point`, and the outward-orienting node emitter `push_oriented`, plus the mesh-parameter
 * config `YSwept`. Blocks share edge curves (=> watertight) and are each locally STAR-SHAPED so
 * their param->surface map is smooth (=> spectral area self-convergence).
 *
 * The M2 hybrid split (junction-only Quad + Slender arms) is built on top of these primitives in
 * ybifurc_hybrid_geom.hpp; the standalone full-quad builder that once lived here (used only by the
 * retired ybifurc-bie/ybifurc-geom drivers) has been removed.
 */
#pragma once

#include <sctl.hpp>
#include <sctl/experimental/quad_element.hpp>
#include <sctl/experimental/quad_element.cpp>
#include <quad_junctions/mpi_utils.hpp>
#include <array>
#include <cmath>
#include <vector>

namespace quad_junctions {
using namespace sctl;

// ============================================================================
// Field config -- the arms/Gaussians that define the iso-surface. Matches the
// field parameters in bifurcation_geom/bifurcation.py Config (Field). The mesh
// (swept-O-grid) parameters live in YSwept below, not here.
// ============================================================================
struct YCfg {
  static constexpr double arm_deg[3] = {-90.0, 30.0, 150.0}; // inlet down, two branches
  static constexpr double sigma      = 0.15;   // Gaussian width (tube radius scale)
  static constexpr double amp        = 1.0;    // Gaussian amplitude
  static constexpr double gauss_ds   = 0.05;   // Gaussian spacing along an arm
  static constexpr double gauss_len  = 0.95;   // arm length covered by Gaussians
};

template <class Real> using Vec3 = std::array<Real, 3>;

// ============================================================================
// Field: f(x) = sum_i amp*exp(-|x-c_i|^2/(2 sigma^2)); exact gradient.
// ============================================================================
template <class Real> struct YField {
  std::vector<Vec3<Real>> C;
  Real inv2s2, invs2, amp;
  YField() {
    const Real ds = YCfg::gauss_ds, L = YCfg::gauss_len;
    inv2s2 = (Real)1 / (2 * YCfg::sigma * YCfg::sigma);
    invs2  = (Real)1 / (YCfg::sigma * YCfg::sigma);
    amp    = YCfg::amp;
    C.push_back(Vec3<Real>{0, 0, 0});                       // one Gaussian at the junction
    for (int a = 0; a < 3; a++) {
      const Real th = YCfg::arm_deg[a] * const_pi<Real>() / 180;
      const Vec3<Real> u{cos<Real>(th), sin<Real>(th), 0};
      for (int k = 1; k * ds <= L + (Real)1e-9; k++)
        C.push_back(Vec3<Real>{k * ds * u[0], k * ds * u[1], 0});
    }
  }
  Real f(const Vec3<Real>& x) const {
    Real s = 0;
    for (const auto& c : C) {
      const Real dx = x[0]-c[0], dy = x[1]-c[1], dz = x[2]-c[2];
      s += amp * exp<Real>(-(dx*dx+dy*dy+dz*dz) * inv2s2);
    }
    return s;
  }
  Vec3<Real> grad(const Vec3<Real>& x) const {
    Vec3<Real> g{0, 0, 0};
    for (const auto& c : C) {
      const Real dx = x[0]-c[0], dy = x[1]-c[1], dz = x[2]-c[2];
      const Real e = amp * exp<Real>(-(dx*dx+dy*dy+dz*dz) * inv2s2) * invs2;
      g[0] -= e*dx; g[1] -= e*dy; g[2] -= e*dz;
    }
    return g;
  }
};

// ============================================================================
// Swept-O-grid geometry helpers (port of bifurcation_geom/ybifurc.py).
// ============================================================================
// Star-shaped projection: from an INTERIOR center c0 along direction d, find s>0 with
// f(c0 + s*d) = level, by bisection (f is >level at c0 and decays to <level outward, so
// there is a single crossing). This is the fixed-RAY projection -- NOT gradient flow.
template <class Real> Vec3<Real> ray_root(const YField<Real>& fld, const Vec3<Real>& c0, Vec3<Real> d, Real level, Real* res_out = nullptr) {
  const Real dn = sqrt<Real>(d[0]*d[0]+d[1]*d[1]+d[2]*d[2]); d[0]/=dn; d[1]/=dn; d[2]/=dn;
  auto F = [&](Real s){ return fld.f(Vec3<Real>{c0[0]+s*d[0], c0[1]+s*d[1], c0[2]+s*d[2]}) - level; };
  Real lo = (Real)1e-6, hi = (Real)3;
  SCTL_ASSERT_MSG(F(lo) > 0 && F(hi) < 0, "ray_root: center not inside / ray does not cross the level set");
  for (int it = 0; it < 100; it++) { const Real mid = (Real)0.5*(lo+hi); if (F(mid) > 0) lo = mid; else hi = mid; }
  const Real s = (Real)0.5*(lo+hi);
  const Vec3<Real> x{c0[0]+s*d[0], c0[1]+s*d[1], c0[2]+s*d[2]};
  if (res_out) *res_out = std::fabs(fld.f(x) - level);
  return x;
}

template <class Real> Vec3<Real> vslerp(Vec3<Real> a, Vec3<Real> b, Real w) {
  auto nrm = [](Vec3<Real>& v){ const Real n = sqrt<Real>(v[0]*v[0]+v[1]*v[1]+v[2]*v[2]); v[0]/=n; v[1]/=n; v[2]/=n; };
  nrm(a); nrm(b);
  Real dot = a[0]*b[0]+a[1]*b[1]+a[2]*b[2]; dot = std::max<Real>((Real)-1, std::min<Real>((Real)1, dot));
  const Real om = acos<Real>(dot);
  if (om < (Real)1e-12) return a;
  const Real s0 = sin<Real>((1-w)*om)/sin<Real>(om), s1 = sin<Real>(w*om)/sin<Real>(om);
  return Vec3<Real>{s0*a[0]+s1*b[0], s0*a[1]+s1*b[1], s0*a[2]+s1*b[2]};
}

// Swept-O-grid parameters (validated in ybifurc.py: area->4e-13, flux->1e-16).
struct YSwept {
  static constexpr double alpha_deg = 38.0;  // junction hole half-angle
  static constexpr double Lseam     = 0.88;  // arm-axis arc-length of the tube->cap seam (pole ~1.10)
  static constexpr double zc_off    = 0.10;  // cap center offset inside the seam (along -u_k)
  static constexpr double core_frac = 0.45;  // butterfly-cap core half-size (tangent units)
  static constexpr int    Nr0       = 2;     // junction radial rings (per sector)  } base counts,
  static constexpr int    Na0       = 16;    // azimuthal panels (mult of 4)         } scaled by nref
  static constexpr int    Ns0       = 4;     // arm axial panels                     } chosen for ~square
  static constexpr int    Ncap0     = 2;     // butterfly core & arc panels/dir       } panels (aspect<1.85)
};

template <class Real> void arm_frame(int k, Vec3<Real>& u, Vec3<Real>& e1, Vec3<Real>& e2) {
  const Real th = YCfg::arm_deg[k]*const_pi<Real>()/180;
  u  = Vec3<Real>{cos<Real>(th), sin<Real>(th), 0};
  e1 = Vec3<Real>{0, 0, 1};                                  // "up"; beta=0 -> +z (matches junction hole)
  e2 = Vec3<Real>{e1[1]*u[2]-e1[2]*u[1], e1[2]*u[0]-e1[0]*u[2], e1[0]*u[1]-e1[1]*u[0]}; // e1 x u (+phi)
}

// junction (sphere-with-3-holes) direction for sector k at annulus params (t radial, s around).
template <class Real> Vec3<Real> junction_dir(int k, Real t, Real s) {
  Vec3<Real> u, e1, e2; arm_frame<Real>(k, u, e1, e2);
  const Real pi = const_pi<Real>(), alpha = (Real)YSwept::alpha_deg*pi/180;
  const Real thc = YCfg::arm_deg[k]*pi/180, phiR = thc+pi/3, phiL = thc-pi/3;
  const Vec3<Real> Ptop{0,0,1}, Pbot{0,0,-1};
  const Vec3<Real> eR{cos<Real>(phiR), sin<Real>(phiR), 0}, eL{cos<Real>(phiL), sin<Real>(phiL), 0};
  const Real beta = 2*pi*s;
  const Vec3<Real> rad{cos<Real>(beta)*e1[0]+sin<Real>(beta)*e2[0], cos<Real>(beta)*e1[1]+sin<Real>(beta)*e2[1], cos<Real>(beta)*e1[2]+sin<Real>(beta)*e2[2]};
  Vec3<Real> inner{cos<Real>(alpha)*u[0]+sin<Real>(alpha)*rad[0], cos<Real>(alpha)*u[1]+sin<Real>(alpha)*rad[1], cos<Real>(alpha)*u[2]+sin<Real>(alpha)*rad[2]};
  const Real seg = s*4;
  Vec3<Real> A, B; Real w;
  if      (seg < 1) { A = Ptop; B = eR; w = seg-0; }
  else if (seg < 2) { A = eR; B = Pbot; w = seg-1; }
  else if (seg < 3) { A = Pbot; B = eL; w = seg-2; }
  else              { A = eL; B = Ptop; w = seg-3; }
  return vslerp<Real>(inner, vslerp<Real>(A, B, w), t);
}

// arm tube point: eta in [0,1] base(=junction hole)->seam; beta azimuth around the arm.
template <class Real> Vec3<Real> arm_point(const YField<Real>& fld, int k, Real eta, Real beta, Real level, Real* res = nullptr) {
  Vec3<Real> u, e1, e2; arm_frame<Real>(k, u, e1, e2);
  const Real alpha = (Real)YSwept::alpha_deg*const_pi<Real>()/180, Ls = (Real)YSwept::Lseam;
  const Vec3<Real> c{eta*Ls*u[0], eta*Ls*u[1], eta*Ls*u[2]};
  const Vec3<Real> rad{cos<Real>(beta)*e1[0]+sin<Real>(beta)*e2[0], cos<Real>(beta)*e1[1]+sin<Real>(beta)*e2[1], cos<Real>(beta)*e1[2]+sin<Real>(beta)*e2[2]};
  const Real cu = (1-eta)*cos<Real>(alpha), cr = (1-eta)*sin<Real>(alpha)+eta;
  const Vec3<Real> d{cu*u[0]+cr*rad[0], cu*u[1]+cr*rad[1], cu*u[2]+cr*rad[2]};
  return ray_root<Real>(fld, c, d, level, res);
}

// Push one filled order x order node block (i slow, j fast) into X, oriented so the
// tensor u x v normal points OUTWARD (aligned with -grad f). GetFarFieldNodes uses the
// u x v convention with no re-orientation, so consistent winding is what makes int n dA
// cancel. If inward, transpose the block (swap u<->v) to flip the normal.
template <class Real> void push_oriented(Vector<Real>& X, const YField<Real>& fld, std::vector<Vec3<Real>>& nd, Integer order) {
  auto at = [&](Integer i, Integer j) -> Vec3<Real>& { return nd[i*order+j]; };
  const Vec3<Real> tu{at(1,0)[0]-at(0,0)[0], at(1,0)[1]-at(0,0)[1], at(1,0)[2]-at(0,0)[2]};
  const Vec3<Real> tv{at(0,1)[0]-at(0,0)[0], at(0,1)[1]-at(0,0)[1], at(0,1)[2]-at(0,0)[2]};
  const Vec3<Real> n{tu[1]*tv[2]-tu[2]*tv[1], tu[2]*tv[0]-tu[0]*tv[2], tu[0]*tv[1]-tu[1]*tv[0]};
  const Vec3<Real> g = fld.grad(at(0,0));                       // outward = -g
  const bool flip = (n[0]*g[0]+n[1]*g[1]+n[2]*g[2]) > 0;        // n . (-g) < 0  =>  n.g > 0
  for (Integer i = 0; i < order; i++)
    for (Integer j = 0; j < order; j++) {
      const Vec3<Real>& p = flip ? at(j,i) : at(i,j);
      X.PushBack(p[0]); X.PushBack(p[1]); X.PushBack(p[2]);
    }
}

} // namespace quad_junctions
