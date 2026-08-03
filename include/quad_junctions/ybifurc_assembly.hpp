/**
 * Composable hybrid Y-junction assembly.
 *
 * Turns the single-junction M2 hybrid (ybifurc_hybrid_geom.hpp) into recombinable COMPONENTS that a
 * caller can place and join into larger geometry, all fed into ONE BoundaryIntegralOp:
 *
 *   - Placement<Real>      : rigid transform (rotation + translation) placing a junction in the world.
 *   - ArmSeam<Real>        : a circle-in-world with a frame -- the ONLY handoff between a placed
 *                            junction and the cap/shaft builders (decouples them from the canonical frame).
 *   - HybridJunction<Real> : one placed junction = QuadElemList sphere-with-3-holes + 3 POU transition
 *                            tubes (NO caps -- each arm's termination is a separable choice), exposing
 *                            its 3 world ArmSeams.
 *   - HybridAssembly<Real> : the combiner. add_junction / add_free_arm (slender + hemisphere cap) /
 *                            add_shared_arm (ONE slender spanning two coaxial junction seams, no caps),
 *                            then emit one combined QuadElemList (all junctions+transitions+caps) and one
 *                            combined SlenderElemList (all shafts).
 *
 * This is ADDITIVE: it reuses the frozen canonical builders (YField, arm_frame, junction_dir, arm_point,
 * ray_root, push_oriented, arm_ring_stats, transition_point) unchanged and does NOT modify any existing
 * free function, so ybifurc-bie / ybifurc-hybrid-bie keep working bit-for-bit. A junction is built in the
 * canonical frame (junction at origin, arms at fixed angles) exactly as the single hybrid does, then
 * rigidly transformed into the world; the shafts are built directly in world coordinates from ArmSeams.
 */
#pragma once

#include <quad_junctions/ybifurc_hybrid_geom.hpp>
#include <quad_junctions/mpi_utils.hpp>
#include <functional>
#include <cstdint>
#include <cstdlib>
#include <cstdio>
#include <string>
#include <memory>
#include <limits>
#include <filesystem>
#include <sys/stat.h>
#include <unistd.h>

namespace quad_junctions {
using namespace sctl;

// Native (unscaled) seam radius of a canonical junction arm: the azimuthal-mean iso-surface radius at
// eta_join (identical for all three arms by junction symmetry). A caller that wants an arm at a chosen
// radius r resizes the junction with placement scale = r / canonical_seam_R0(level, eta_join).
template <class Real> Real canonical_seam_R0(Real level, Real eta_join, int k = 0) {
  const YField<Real> fld;
  Real R0, a0; arm_ring_stats<Real>(fld, k, eta_join, level, R0, a0, 64);
  return R0;
}

// ============================================================================
// Similarity transform x -> t + scale * (R x) (R a proper rotation, row-major 3x3; scale a uniform
// positive factor). scale != 1 uniformly resizes a placed junction about its origin (which lands at t):
// junction body, transition tubes, and seam radius/axial-station all scale together (see transform_seam),
// while seam axis/phase directions stay unit (apply_dir). Default scale = 1 => a pure rigid transform.
// ============================================================================
template <class Real> struct Placement {
  Vec3<Real> t{(Real)0, (Real)0, (Real)0};
  Real R[9] = {1,0,0, 0,1,0, 0,0,1};
  Real scale = (Real)1;

  // Rotate a DIRECTION (no translation, no scale) -- keeps unit vectors unit.
  Vec3<Real> apply_dir(const Vec3<Real>& v) const {
    return Vec3<Real>{R[0]*v[0]+R[1]*v[1]+R[2]*v[2], R[3]*v[0]+R[4]*v[1]+R[5]*v[2], R[6]*v[0]+R[7]*v[1]+R[8]*v[2]};
  }
  Vec3<Real> apply_point(const Vec3<Real>& v) const {
    const Vec3<Real> d = apply_dir(v);
    return Vec3<Real>{scale*d[0]+t[0], scale*d[1]+t[1], scale*d[2]+t[2]};
  }
  // Inverse of apply_point (R orthonormal => R^{-1} = R^T): world -> canonical/local coords.
  Vec3<Real> apply_inverse_point(const Vec3<Real>& v) const {
    const Vec3<Real> d{(v[0]-t[0])/scale, (v[1]-t[1])/scale, (v[2]-t[2])/scale};
    return Vec3<Real>{R[0]*d[0]+R[3]*d[1]+R[6]*d[2], R[1]*d[0]+R[4]*d[1]+R[7]*d[2], R[2]*d[0]+R[5]*d[1]+R[8]*d[2]};
  }

  static Placement Identity() { return Placement(); }

  // Build the placement that sends canonical arm k's axis u_k -> world_dir and the canonical
  // "up" e1(=+z) -> the component of world_up perpendicular to world_dir, translated to `trans`, with an
  // optional uniform `scale` (resizes the whole junction + its arm seams about `trans`).
  // The canonical arm triad (u, e1, e2=e1 x u) is LEFT-handed; we build the target triad (a, b, c=b x a)
  // with the SAME handedness so R = T S^T is a PROPER rotation (det = +1) => surface winding / outward
  // normals are preserved after the transform.
  static Placement AlignArm(int k, Vec3<Real> world_dir, Vec3<Real> world_up, Vec3<Real> trans,
                            Real scale = (Real)1) {
    auto nrm = [](Vec3<Real> v) -> Vec3<Real> { const Real n = sqrt<Real>(v[0]*v[0]+v[1]*v[1]+v[2]*v[2]); return Vec3<Real>{v[0]/n, v[1]/n, v[2]/n}; };
    auto cross = [](const Vec3<Real>& p, const Vec3<Real>& q) -> Vec3<Real> { return Vec3<Real>{p[1]*q[2]-p[2]*q[1], p[2]*q[0]-p[0]*q[2], p[0]*q[1]-p[1]*q[0]}; };
    Vec3<Real> u, e1c, e2c; arm_frame<Real>(k, u, e1c, e2c);                 // source triad (columns of S)
    const Vec3<Real> a = nrm(world_dir);
    const Real d = world_up[0]*a[0]+world_up[1]*a[1]+world_up[2]*a[2];
    const Vec3<Real> b = nrm(Vec3<Real>{world_up[0]-d*a[0], world_up[1]-d*a[1], world_up[2]-d*a[2]});
    const Vec3<Real> c = cross(b, a);                                        // matches source handedness
    Placement P; P.t = trans; P.scale = scale;
    for (int i = 0; i < 3; i++) {
      P.R[3*i+0] = a[i]*u[0] + b[i]*e1c[0] + c[i]*e2c[0];                    // R = a u^T + b e1^T + c e2^T
      P.R[3*i+1] = a[i]*u[1] + b[i]*e1c[1] + c[i]*e2c[1];
      P.R[3*i+2] = a[i]*u[2] + b[i]*e1c[2] + c[i]*e2c[2];
    }
    return P;
  }
};

// ============================================================================
// A terminal circle of a placed junction arm (all in WORLD coords): ring center C at axial station a0,
// outward axis u, azimuthal phase frame (e1,e2), radius R0. This is the complete interface a shaft or a
// cap needs -- no reference to the canonical field or frame.
// ============================================================================
template <class Real> struct ArmSeam {
  Vec3<Real> C{(Real)0,(Real)0,(Real)0}, u{(Real)0,(Real)0,(Real)1}, e1{(Real)1,(Real)0,(Real)0}, e2{(Real)0,(Real)1,(Real)0};
  Real R0 = 0, a0 = 0;
};

// Rotate+translate every node (X = [x0,y0,z0, x1,...]) in place.
template <class Real> void transform_nodes(Vector<Real>& X, const Placement<Real>& P) {
  const Long N = X.Dim()/3;
  for (Long i = 0; i < N; i++) {
    const Vec3<Real> q = P.apply_point(Vec3<Real>{X[3*i], X[3*i+1], X[3*i+2]});
    X[3*i] = q[0]; X[3*i+1] = q[1]; X[3*i+2] = q[2];
  }
}

template <class Real> ArmSeam<Real> transform_seam(const ArmSeam<Real>& s, const Placement<Real>& P) {
  ArmSeam<Real> o;
  o.C = P.apply_point(s.C);
  o.u = P.apply_dir(s.u); o.e1 = P.apply_dir(s.e1); o.e2 = P.apply_dir(s.e2);
  o.R0 = P.scale * s.R0; o.a0 = P.scale * s.a0;   // lengths scale with the placement (C already scaled)
  return o;
}

// ============================================================================
// Emit the CANONICAL junction (sphere-with-3-holes) + 3 POU transition tubes into X (NO caps). Mirrors
// the junction/transition loops of BuildYJunctionWithTransitions; also fills the 3 canonical ArmSeams
// (terminal ring of each transition tube at eta_join). Kept as a free function so HybridJunction can
// transform the raw node array before constructing a QuadElemList.
// ============================================================================
template <class Real> void emit_junction_transitions(
    Vector<Real>& X, Integer order, Real level, Integer nref, Real eta_join, Integer Ns_trans,
    ArmSeam<Real> seams[3], Real* max_res = nullptr) {
  const YField<Real> fld;
  const Vector<Real>& nds = QuadElemList<Real>::ParamNodes(order);
  const Integer Nr = YSwept::Nr0*nref, Na = YSwept::Na0*nref;
  const Real pi = const_pi<Real>();
  SCTL_ASSERT_MSG(Ns_trans >= 2, "Ns_trans must be >= 2 (blend panels + one pure-circle clamp panel).");
  const Real eta_w = eta_join * (Real)(Ns_trans - 1) / Ns_trans;   // clamp w=0 on the outer panel
  Real rmax = 0, r;
  std::vector<Vec3<Real>> nd(order*order);

  for (int k = 0; k < 3; k++) {
    Real R0, a_join; arm_ring_stats<Real>(fld, k, eta_join, level, R0, a_join, 64);
    Vec3<Real> u, e1, e2; arm_frame<Real>(k, u, e1, e2);
    seams[k].C = Vec3<Real>{a_join*u[0], a_join*u[1], a_join*u[2]};
    seams[k].u = u; seams[k].e1 = e1; seams[k].e2 = e2; seams[k].R0 = R0; seams[k].a0 = a_join;

    // ---- junction sector k: Nr x Na annulus (t radial 0=hole..1=arcs, s around) ----
    for (Integer ir = 0; ir < Nr; ir++)
      for (Integer ia = 0; ia < Na; ia++) {
        const Real t0 = (Real)ir/Nr, t1 = (Real)(ir+1)/Nr, s0 = (Real)ia/Na, s1 = (Real)(ia+1)/Na;
        for (Integer i = 0; i < order; i++) { const Real t = t0 + nds[i]*(t1-t0);
          for (Integer j = 0; j < order; j++) { const Real s = s0 + nds[j]*(s1-s0);
            nd[i*order+j] = ray_root<Real>(fld, Vec3<Real>{0,0,0}, junction_dir<Real>(k, t, s), level, &r); rmax = std::max(rmax, r); } }
        push_oriented<Real>(X, fld, nd, order);
      }

    // ---- transition tube k: Ns_trans x Na (eta axial in [0,eta_join], beta around), POU blended ----
    // eta depends only on (l,i), so the ring's axial center a_loc is tabulated per (l,i) ONCE instead of
    // recomputed (64 arm_point each) per node: Ns_trans*order calls per arm, not Ns_trans*Na*order^2.
    // a_tab stays inside the k loop -- arm_ring_stats depends on k via arm_frame, and the three arms are
    // only mathematically (not bit-) symmetric. Emission order of X is untouched.
    std::vector<Real> a_tab(order);
    for (Integer l = 0; l < Ns_trans; l++) {
      const Real e0 = eta_join*(Real)l/Ns_trans, e1t = eta_join*(Real)(l+1)/Ns_trans;
      for (Integer i = 0; i < order; i++) { const Real eta = e0 + nds[i]*(e1t-e0);
        Real R0_loc; arm_ring_stats<Real>(fld, k, eta, level, R0_loc, a_tab[i], 64); }
      for (Integer ia = 0; ia < Na; ia++) {
        const Real b0 = (Real)ia/Na, b1 = (Real)(ia+1)/Na;
        for (Integer i = 0; i < order; i++) { const Real eta = e0 + nds[i]*(e1t-e0);
          for (Integer j = 0; j < order; j++) { const Real beta = 2*pi*(b0 + nds[j]*(b1-b0));
            nd[i*order+j] = transition_point<Real>(fld, k, eta, beta, level, R0, eta_w, a_tab[i]); } }
        push_oriented<Real>(X, fld, nd, order);
      }
    }
  }
  if (max_res) *max_res = rmax;
}

// Push one order x order node block oriented so the tensor normal aligns with the OUTWARD radial
// (node - C) of a hemisphere centered at C. Geometric analogue of push_oriented's -grad f flip; for a
// convex dome the two agree in sign, so this reproduces the field-based cap's orientation/accuracy.
template <class Real> void push_oriented_hemisphere(Vector<Real>& X, std::vector<Vec3<Real>>& nd, Integer order, const Vec3<Real>& C) {
  auto at = [&](Integer i, Integer j) -> Vec3<Real>& { return nd[i*order+j]; };
  const Vec3<Real> tu{at(1,0)[0]-at(0,0)[0], at(1,0)[1]-at(0,0)[1], at(1,0)[2]-at(0,0)[2]};
  const Vec3<Real> tv{at(0,1)[0]-at(0,0)[0], at(0,1)[1]-at(0,0)[1], at(0,1)[2]-at(0,0)[2]};
  const Vec3<Real> n{tu[1]*tv[2]-tu[2]*tv[1], tu[2]*tv[0]-tu[0]*tv[2], tu[0]*tv[1]-tu[1]*tv[0]};
  const Vec3<Real> o{at(0,0)[0]-C[0], at(0,0)[1]-C[1], at(0,0)[2]-C[2]};
  const bool flip = (n[0]*o[0]+n[1]*o[1]+n[2]*o[2]) < 0;
  for (Integer i = 0; i < order; i++)
    for (Integer j = 0; j < order; j++) {
      const Vec3<Real>& p = flip ? at(j,i) : at(i,j);
      X.PushBack(p[0]); X.PushBack(p[1]); X.PushBack(p[2]);
    }
}

// ============================================================================
// Butterfly-dome cap on a WORLD ring: hemisphere of radius ring.R0 centered at ring.C, equator = the
// ring (=> node-conforms geometrically to the slender's terminal circle). Same non-degenerate O-grid as
// the M1 / add_arm_cap_hemisphere elevation, but expressed directly in the ring's (u,e1,e2) frame and
// oriented geometrically (push_oriented_hemisphere) so it needs no field.
// ============================================================================
template <class Real> void add_cap_hemisphere_frame(Vector<Real>& X, const ArmSeam<Real>& ring, Integer order,
                                                    Integer Ncap, Real core_frac = (Real)0.40) {
  const Real pi = const_pi<Real>();
  const Vec3<Real>& u = ring.u; const Vec3<Real>& e1 = ring.e1; const Vec3<Real>& e2 = ring.e2;
  const Vec3<Real>& C = ring.C; const Real R0 = ring.R0;
  const Real h = core_frac;
  const Integer nc = std::max<Integer>(1, Ncap);
  const Vector<Real>& nds = QuadElemList<Real>::ParamNodes(order);
  std::vector<Vec3<Real>> nd(order*order);
  auto elev = [&](Real Dx, Real Dy) -> Vec3<Real> {
    const Real q = sqrt<Real>(Dx*Dx + Dy*Dy), psi = q * pi / 2;
    const Real r_lat = R0 * sin<Real>(psi), h_ax = R0 * cos<Real>(psi);
    const Real roq = (q > (Real)1e-9) ? r_lat / q : R0 * pi / 2;
    return Vec3<Real>{C[0] + roq*(Dx*e1[0]+Dy*e2[0]) + h_ax*u[0],
                      C[1] + roq*(Dx*e1[1]+Dy*e2[1]) + h_ax*u[1],
                      C[2] + roq*(Dx*e1[2]+Dy*e2[2]) + h_ax*u[2]};
  };
  for (Integer ic = 0; ic < nc; ic++)
    for (Integer jc = 0; jc < nc; jc++) {
      const Real x0 = -h + 2*h*ic/nc, x1 = -h + 2*h*(ic+1)/nc, y0 = -h + 2*h*jc/nc, y1 = -h + 2*h*(jc+1)/nc;
      for (Integer i = 0; i < order; i++) { const Real yy = y0 + nds[i]*(y1-y0);
        for (Integer j = 0; j < order; j++) { const Real xx = x0 + nds[j]*(x1-x0);
          nd[i*order+j] = elev(xx, yy); } }
      push_oriented_hemisphere<Real>(X, nd, order, C);
    }
  for (Integer kk = 0; kk < 4; kk++) {
    const Real rot = kk * pi / 2, cr = cos<Real>(rot), sr = sin<Real>(rot);
    auto pt = [=](Real eta, Real xi) -> Vec2<Real> {
      const Real th = -pi/4 + xi * (pi/2);
      const Vec2<Real> in{h, h*(2*xi - 1)};
      const Vec2<Real> out{cos<Real>(th), sin<Real>(th)};
      const Real px = (1-eta)*in[0] + eta*out[0], py = (1-eta)*in[1] + eta*out[1];
      return Vec2<Real>{cr*px - sr*py, sr*px + cr*py};
    };
    for (Integer ir = 0; ir < nc; ir++)
      for (Integer ia = 0; ia < nc; ia++) {
        const Real e0 = (Real)ir/nc, e1c = (Real)(ir+1)/nc, a0 = (Real)ia/nc, a1 = (Real)(ia+1)/nc;
        for (Integer i = 0; i < order; i++) { const Real xi = a0 + nds[i]*(a1-a0);
          for (Integer j = 0; j < order; j++) { const Real eta = e0 + nds[j]*(e1c-e0);
            const Vec2<Real> P = pt(eta, xi); nd[i*order+j] = elev(P[0], P[1]); } }
        push_oriented_hemisphere<Real>(X, nd, order, C);
      }
  }
}

// ============================================================================
// CANONICAL-JUNCTION CACHE (memo -> file -> build).
//
// emit_junction_transitions is a PURE function of (order, level, nref, eta_join, Ns_trans) plus two
// non-argument inputs: pou_kind() and the compile-time YCfg/YSwept constants. It accumulates nothing
// across calls (X is a fresh local, rmax starts at 0, YField's ctor has no tunables, ray_root is a fixed
// 100-iteration bisection, ParamNodes(order) is a deterministic table). So every placement with the same
// tuple gets a BIT-IDENTICAL canonical mesh, and rebuilding it per placement is pure duplication --
// 20x in build_vessels_network, 11x in ybifurc-tree-bie, 2x in multi/channel/flow.
//
// This lives at free-function scope and is called from HybridJunction's ctor (NOT from add_junction and
// NOT as a HybridAssembly member) for two reasons: ybifurc-tree-bie builds a HybridJunction directly to
// read its canonical seams, and ybifurc-multi-bie builds two separate HybridAssembly objects with the
// same parameters.
//
// NOT thread-safe (function-local static, unguarded insert). Every caller is serial driver code outside
// any OMP region; do not call from inside a parallel region without adding a lock.
// ============================================================================

struct CanonKey {
  Integer order = 0, nref = 0, Ns_trans = 0;
  double level = 0, eta_join = 0;              // compared bitwise
  int pou = 0, rsize = 0;                      // pou_kind() snapshot, (int)sizeof(Real)
  bool operator==(const CanonKey& o) const {
    return order == o.order && nref == o.nref && Ns_trans == o.Ns_trans
        && level == o.level && eta_join == o.eta_join && pou == o.pou && rsize == o.rsize;
  }
};

// Callers receive this by CONST reference: transforming the cached X in place would double-transform
// every junction after the first, silently, into plausible-looking garbage.
template <class Real> struct CanonMesh {
  Vector<Real> X;
  ArmSeam<Real> seams[3];
  Real max_res = 0;
};

// Node-coordinate count: 3 arms x (Nr*Na sector + Ns_trans*Na transition) panels x order^2 nodes x 3.
inline Long canon_ncoord(Integer order, Integer nref, Integer Ns_trans) {
  const Long Nr = (Long)YSwept::Nr0*nref, Na = (Long)YSwept::Na0*nref;
  return (Long)3 * (Nr*Na + (Long)Ns_trans*Na) * (Long)order * (Long)order * 3;
}

// ---- on-disk record, all Real ----------------------------------------------------------------
//  0- 2  magic, format version, sizeof(Real)
//  3- 8  order, nref, Ns_trans, level, eta_join, pou_kind
//  9-11  Nr, Na, node count
// 12-22  YCfg::{sigma,amp,gauss_ds,gauss_len,arm_deg[0..2]}, YSwept::{alpha_deg,Lseam,Nr0*100+Na0}, algo ver
// 23-24  max_res, reserved
// 25-66  3 x ArmSeam (14 Reals each: C,u,e1,e2,R0,a0)
// 67-    node payload, in emit order
static constexpr Long   CANON_NHDR    = 25;
static constexpr Long   CANON_NSEAM   = 42;
static constexpr double CANON_MAGIC   = 20260724.0;
static constexpr double CANON_FMT_VER = 1.0;
// Bump when the canonical geometry ALGORITHM changes without any YCfg/YSwept constant changing (the
// eta_w convention, the pou_weight polynomial, ray_root's iteration count, the emit order, ...).
static constexpr double CANON_ALGO_VER = 1.0;

inline std::string canon_cache_dir() {
  const char* e = std::getenv("QJ_MESH_CACHE_DIR");
  return (e && *e) ? std::string(e) : std::string("data/mesh-cache");
}
inline bool canon_cache_enabled() {
  const char* e = std::getenv("QJ_MESH_CACHE");
  return !(e && e[0] == '0');                  // on by default; QJ_MESH_CACHE=0 disables
}
// QJ_CANON_MEMO=0 disables in-memory reuse, so every placement rebuilds (or re-reads) its canonical
// mesh. Only useful for verification: with QJ_MESH_CACHE=0 QJ_CANON_MEMO=0 this function reproduces the
// original build-per-placement behavior exactly, which is the A/B baseline for the cache.
inline bool canon_memo_enabled() {
  const char* e = std::getenv("QJ_CANON_MEMO");
  return !(e && e[0] == '0');
}
// Encoding integer counts as Reals is only exact with a >=53-bit mantissa. All drivers use double; this
// gate keeps a future float instantiation from failing validation forever.
template <class Real> constexpr bool canon_disk_ok() {
  return std::numeric_limits<Real>::digits >= 53;
}

// %.17g so distinct doubles always yield distinct filenames (the older junc-*.mesh key streamed at the
// default 6 significant digits, where 0.4 and 0.4000000001 collide onto one file).
inline std::string canon_cache_path(const CanonKey& k) {
  char lv[40], ej[40], buf[512];
  std::snprintf(lv, sizeof lv, "%.17g", k.level);
  std::snprintf(ej, sizeof ej, "%.17g", k.eta_join);
  std::snprintf(buf, sizeof buf, "%s/ycanon-ord%ld-L%s-nref%ld-eta%s-Ns%ld-pou%d-r%d.bin",
                canon_cache_dir().c_str(), (long)k.order, lv, (long)k.nref, ej,
                (long)k.Ns_trans, k.pou, k.rsize);
  return std::string(buf);
}

// Independent recomputation of the 3 canonical seams -- a fingerprint of the whole field stack
// (YField::f -> ray_root -> arm_point -> arm_frame). Mirrors the seam block of
// emit_junction_transitions; kept separate on purpose so a cache hit is validated against a fresh
// computation rather than against itself.
template <class Real> void canon_seam_fingerprint(Real level, Real eta_join, ArmSeam<Real> out[3]) {
  const YField<Real> fld;
  for (int k = 0; k < 3; k++) {
    Real R0, a_join; arm_ring_stats<Real>(fld, k, eta_join, level, R0, a_join, 64);
    Vec3<Real> u, e1, e2; arm_frame<Real>(k, u, e1, e2);
    out[k].C = Vec3<Real>{a_join*u[0], a_join*u[1], a_join*u[2]};
    out[k].u = u; out[k].e1 = e1; out[k].e2 = e2; out[k].R0 = R0; out[k].a0 = a_join;
  }
}

// Flatten / unflatten one ArmSeam as 14 consecutive Reals (field by field, never memcpy).
template <class Real> void canon_seam_put(Vector<Real>& v, Long at, const ArmSeam<Real>& s) {
  for (int c = 0; c < 3; c++) { v[at+c] = s.C[c]; v[at+3+c] = s.u[c]; v[at+6+c] = s.e1[c]; v[at+9+c] = s.e2[c]; }
  v[at+12] = s.R0; v[at+13] = s.a0;
}
template <class Real> bool canon_seam_matches(const Vector<Real>& v, Long at, const ArmSeam<Real>& s) {
  for (int c = 0; c < 3; c++)
    if (v[at+c] != s.C[c] || v[at+3+c] != s.u[c] || v[at+6+c] != s.e1[c] || v[at+9+c] != s.e2[c]) return false;
  return v[at+12] == s.R0 && v[at+13] == s.a0;
}

// Try to load a cached canonical mesh. Returns false (and leaves `out` untouched) on absence or ANY
// validation failure -- the caller then rebuilds and overwrites. Never aborts: a stale file in a shared
// data/ directory must not kill a multi-hour job. No collective calls: a concurrent job can rename the
// file into place between one rank's stat and another's, so hit/miss may legitimately differ per rank,
// and validation is what guarantees a hit equals a fresh build.
template <class Real> bool canon_try_read(const std::string& path, const CanonKey& key,
                                          Real level, Real eta_join, CanonMesh<Real>& out, const Comm& comm) {
  struct stat st;
  if (::stat(path.c_str(), &st) != 0) return false;      // stat first: Vector::Read prints per rank
  Vector<Real> v;                                        // FRESH: Read leaves its target untouched on failure
  v.Read(path.c_str());

  const Long ncoord = canon_ncoord(key.order, key.nref, key.Ns_trans);
  bool ok = true;
  const char* why = "";
  auto chk = [&](bool cond, const char* what) { if (ok && !cond) { ok = false; why = what; } };

  chk(v.Dim() == CANON_NHDR + CANON_NSEAM + ncoord, "size");
  if (ok) {
    chk((double)v[0] == CANON_MAGIC, "magic");
    chk((double)v[1] == CANON_FMT_VER, "format version");
    chk((Integer)v[2] == (Integer)sizeof(Real), "sizeof(Real)");
    chk((Integer)v[3] == key.order, "order");
    chk((Integer)v[4] == key.nref, "nref");
    chk((Integer)v[5] == key.Ns_trans, "Ns_trans");
    chk((double)v[6] == key.level, "level");
    chk((double)v[7] == key.eta_join, "eta_join");
    chk((Integer)v[8] == (Integer)key.pou, "pou_kind");
    chk((Integer)v[9] == (Integer)(YSwept::Nr0*key.nref), "Nr");
    chk((Integer)v[10] == (Integer)(YSwept::Na0*key.nref), "Na");
    chk((Long)v[11] == ncoord/3, "node count");
    chk((double)v[12] == (double)YCfg::sigma, "YCfg::sigma");
    chk((double)v[13] == (double)YCfg::amp, "YCfg::amp");
    chk((double)v[14] == (double)YCfg::gauss_ds, "YCfg::gauss_ds");
    chk((double)v[15] == (double)YCfg::gauss_len, "YCfg::gauss_len");
    chk((double)v[16] == (double)YCfg::arm_deg[0], "YCfg::arm_deg[0]");
    chk((double)v[17] == (double)YCfg::arm_deg[1], "YCfg::arm_deg[1]");
    chk((double)v[18] == (double)YCfg::arm_deg[2], "YCfg::arm_deg[2]");
    chk((double)v[19] == (double)YSwept::alpha_deg, "YSwept::alpha_deg");
    chk((double)v[20] == (double)YSwept::Lseam, "YSwept::Lseam");
    chk((Integer)v[21] == (Integer)(YSwept::Nr0*100 + YSwept::Na0), "YSwept::Nr0/Na0");
    chk((double)v[22] == CANON_ALGO_VER, "algorithm version");
  }
  if (ok) {   // the strongest check: recompute the seams and require bit equality
    ArmSeam<Real> fp[3]; canon_seam_fingerprint<Real>(level, eta_join, fp);
    for (int k = 0; k < 3; k++) chk(canon_seam_matches(v, CANON_NHDR + 14*k, fp[k]), "seam fingerprint");
  }
  if (!ok) {
    if (!comm.Rank())
      std::cout << "  [ycanon] WARNING: ignoring stale/corrupt cache file (" << why << " mismatch): "
                << path << " -- rebuilding and overwriting\n";
    return false;
  }

  out.X.ReInit(ncoord);
  for (Long i = 0; i < ncoord; i++) out.X[i] = v[CANON_NHDR + CANON_NSEAM + i];
  for (int k = 0; k < 3; k++) {
    ArmSeam<Real>& s = out.seams[k];
    const Long at = CANON_NHDR + 14*k;
    for (int c = 0; c < 3; c++) { s.C[c] = v[at+c]; s.u[c] = v[at+3+c]; s.e1[c] = v[at+6+c]; s.e2[c] = v[at+9+c]; }
    s.R0 = v[at+12]; s.a0 = v[at+13];
  }
  out.max_res = v[23];
  if (!comm.Rank()) std::cout << "  [ycanon] loaded " << path << "\n";
  return true;
}

// Persist a freshly built canonical mesh. Rank 0 only; unique temp IN THE TARGET DIRECTORY (a
// cross-device rename fails EXDEV, and a shared temp name is exactly how a half-written file gets
// renamed into place), then an atomic rename -- so concurrent jobs on the same key are harmless in every
// interleaving and no reader can ever observe a partial file.
template <class Real> void canon_write(const std::string& path, const CanonKey& key,
                                       const CanonMesh<Real>& m, const Comm& comm) {
  if (comm.Rank()) return;
  std::error_code ec;
  std::filesystem::create_directories(canon_cache_dir(), ec);   // no C++ here mkdirs; submit-*.sh don't either

  const Long ncoord = m.X.Dim();
  Vector<Real> v(CANON_NHDR + CANON_NSEAM + ncoord);
  v[0] = (Real)CANON_MAGIC;      v[1] = (Real)CANON_FMT_VER;  v[2] = (Real)sizeof(Real);
  v[3] = (Real)key.order;        v[4] = (Real)key.nref;       v[5] = (Real)key.Ns_trans;
  v[6] = (Real)key.level;        v[7] = (Real)key.eta_join;   v[8] = (Real)key.pou;
  v[9] = (Real)(YSwept::Nr0*key.nref); v[10] = (Real)(YSwept::Na0*key.nref); v[11] = (Real)(ncoord/3);
  v[12] = (Real)YCfg::sigma;     v[13] = (Real)YCfg::amp;     v[14] = (Real)YCfg::gauss_ds;
  v[15] = (Real)YCfg::gauss_len; v[16] = (Real)YCfg::arm_deg[0]; v[17] = (Real)YCfg::arm_deg[1];
  v[18] = (Real)YCfg::arm_deg[2]; v[19] = (Real)YSwept::alpha_deg; v[20] = (Real)YSwept::Lseam;
  v[21] = (Real)(YSwept::Nr0*100 + YSwept::Na0); v[22] = (Real)CANON_ALGO_VER;
  v[23] = m.max_res;             v[24] = (Real)0;
  for (int k = 0; k < 3; k++) canon_seam_put(v, CANON_NHDR + 14*k, m.seams[k]);
  for (Long i = 0; i < ncoord; i++) v[CANON_NHDR + CANON_NSEAM + i] = m.X[i];

  char host[128]; host[0] = 0; ::gethostname(host, sizeof host - 1);
  const std::string tmp = path + ".tmp." + std::to_string((long)::getpid()) + "." + host;
  v.Write(tmp.c_str());                        // reports failure only by printing -> stat to be sure
  struct stat st;
  if (::stat(tmp.c_str(), &st) != 0) {
    std::cout << "  [ycanon] WARNING: could not write " << tmp << " (cache disabled for this run)\n";
    return;
  }
  if (std::rename(tmp.c_str(), path.c_str()) != 0) {
    std::remove(tmp.c_str());
    std::cout << "  [ycanon] WARNING: could not rename into " << path << "\n";
    return;
  }
  std::cout << "  [ycanon] wrote " << path << " (" << (long)(st.st_size/1024) << " KiB)\n";
}

template <class Real> const CanonMesh<Real>&
canonical_junction(Integer order, Real level, Integer nref, Real eta_join, Integer Ns_trans,
                   const Comm& comm = Comm::Self()) {
  struct Entry { CanonKey key; CanonMesh<Real> mesh; };
  static std::vector<std::unique_ptr<Entry>> memo;      // unique_ptr => returned refs stay valid on growth
  static CanonMesh<Real> scratch;                       // storage when memoization is disabled
  const bool memoize = canon_memo_enabled();
  const CanonKey key{order, nref, Ns_trans, (double)level, (double)eta_join,
                     pou_kind(), (int)sizeof(Real)};
  if (memoize) for (const auto& e : memo) if (e->key == key) return e->mesh;

  std::unique_ptr<Entry> ent;
  CanonMesh<Real>* out;
  if (memoize) { ent.reset(new Entry); ent->key = key; out = &ent->mesh; }
  else { scratch = CanonMesh<Real>(); out = &scratch; }

  const bool use_disk = canon_disk_ok<Real>() && canon_cache_enabled();
  const std::string path = use_disk ? canon_cache_path(key) : std::string();
  if (!use_disk || !canon_try_read<Real>(path, key, level, eta_join, *out, comm)) {
    emit_junction_transitions<Real>(out->X, order, level, nref, eta_join, Ns_trans,
                                    out->seams, &out->max_res);
    SCTL_ASSERT_MSG(out->X.Dim() == canon_ncoord(order, nref, Ns_trans),
                    "canonical_junction: unexpected node count from emit_junction_transitions.");
    if (use_disk) canon_write<Real>(path, key, *out, comm);
  }
  if (!memoize) return scratch;
  memo.push_back(std::move(ent));
  return memo.back()->mesh;
}

// ============================================================================
// One placed junction: canonical junction + transitions built, transformed into world, seams exposed.
// ============================================================================
template <class Real> class HybridJunction {
  Vector<Real> X_;
  ArmSeam<Real> seams_[3];
  Integer order_;
  Placement<Real> place_;
 public:
  Real max_res = 0;
  HybridJunction(Integer order, Real level, Integer nref, Real eta_join, Integer Ns_trans,
                 const Placement<Real>& P, const Comm& comm = Comm::Self())
      : order_(order), place_(P) {
    // The canonical (pre-placement) mesh is identical for every placement of the same parameters, so it
    // comes from the cache. COPY it, then transform this instance -- transforming the cached buffer would
    // double-transform every later junction. `comm` only gates which rank writes the cache file.
    const CanonMesh<Real>& C = canonical_junction<Real>(order, level, nref, eta_join, Ns_trans, comm);
    X_ = C.X;
    max_res = C.max_res;
    transform_nodes<Real>(X_, P);
    for (int k = 0; k < 3; k++) seams_[k] = transform_seam<Real>(C.seams[k], P);
  }
  const Vector<Real>& nodes() const { return X_; }
  const ArmSeam<Real>& seam(int k) const { return seams_[k]; }
  const Placement<Real>& placement() const { return place_; }
  Integer order() const { return order_; }
};

// ============================================================================
// Combine placed junctions + capped free arms + shared arms into one QuadElemList + one SlenderElemList.
// ============================================================================
template <class Real> class HybridAssembly {
  Integer order_;
  Vector<Real> Xquad_;                                  // all junction+transition+cap nodes (world)
  Vector<Long> elem_order_, forder_;                    // per-slender-panel accumulation (built full,
  Vector<Real> coord_, radius_, orient_;                //   sliced per-rank in slender())
  Long npanel_ = 0;

  // Append one slender fiber: n_axial panels from ring center C0 along axis u, length `length`,
  // per-node orientation e1 (must be perpendicular to u). Optional `disp(t)`, t = arc fraction in [0,1],
  // adds a world-space transverse offset to the centerline (a curved fiber); it must vanish with zero
  // slope at t=0,1 so the terminal rings stay circles perpendicular to u (seam-conforming). Optional
  // `rad(t)` gives a per-node radius PROFILE (t = arc fraction); when absent every node uses the constant
  // R0. rad(t) must equal the neighbouring seam radii at t=0,1 for a seam-conforming (watertight) join.
  // Optional `orient(t)` supplies a per-node frame reference e1 (t = arc fraction), used instead of the
  // constant `e1` -- needed for a 3D-bent centerline where a single constant e1 would go parallel to the
  // tangent somewhere (CSBQ GetGeom divides by |e1_perp|). When absent, the constant `e1` is pushed exactly
  // as before, so every existing (planar) caller is byte-for-byte identical.
  void add_slender_fiber(const Vec3<Real>& C0, const Vec3<Real>& u, Real R0, const Vec3<Real>& e1,
                         Real length, Integer n_axial, Long cheb, Long fourier,
                         const std::function<Vec3<Real>(Real)>& disp = {},
                         const std::function<Real(Real)>& rad = {},
                         const std::function<Vec3<Real>(Real)>& orient = {}) {
    for (Integer p = 0; p < n_axial; p++, npanel_++) {
      elem_order_.PushBack(cheb); forder_.PushBack(fourier);
      const Vector<Real>& cn = SlenderElemList<Real>::CenterlineNodes(cheb);
      for (Long j = 0; j < cheb; j++) {
        const Real s = length * (p + cn[j]) / n_axial;
        const Real t = s / length;
        Vec3<Real> P{C0[0]+s*u[0], C0[1]+s*u[1], C0[2]+s*u[2]};
        if (disp) { const Vec3<Real> d = disp(t); P[0]+=d[0]; P[1]+=d[1]; P[2]+=d[2]; }
        coord_.PushBack(P[0]); coord_.PushBack(P[1]); coord_.PushBack(P[2]);
        radius_.PushBack(rad ? rad(t) : R0);
        const Vec3<Real> o = orient ? orient(t) : e1;
        orient_.PushBack(o[0]); orient_.PushBack(o[1]); orient_.PushBack(o[2]);
      }
    }
  }

 public:
  explicit HybridAssembly(Integer order) : order_(order) {}
  Integer order() const { return order_; }

  // Place a junction; append its quad nodes; return it so the caller reads its seams / placement.
  // `comm` is forwarded only so the canonical-mesh cache file is written by rank 0 alone.
  HybridJunction<Real> add_junction(const Placement<Real>& P, Real level, Integer nref, Real eta_join,
                                    Integer Ns_trans, const Comm& comm = Comm::Self()) {
    HybridJunction<Real> J(order_, level, nref, eta_join, Ns_trans, P, comm);
    for (auto v : J.nodes()) Xquad_.PushBack(v);
    return J;
  }

  // Free arm off `seam`: a slender fiber from the seam ring (station a0) out to arc station s_cap along
  // seam.u, then (with_cap) a hemisphere cap on the far ring. Reproduces the single-hybrid free arm.
  // `with_cap=false` emits ONLY the stub fiber, leaving the far end an OPEN circular ring -- a port for a
  // periodic / externally-closed problem (used by the singly-periodic vessels geometry).
  void add_free_arm(const ArmSeam<Real>& seam, Real s_cap, Integer n_axial, Integer Ncap,
                    Long cheb = 10, Long fourier = 12, Real core_frac = (Real)0.40, bool with_cap = true) {
    const Real L = s_cap - seam.a0;
    SCTL_ASSERT_MSG(L > 0, "add_free_arm: s_cap must exceed the seam axial station a0.");
    add_slender_fiber(seam.C, seam.u, seam.R0, seam.e1, L, n_axial, cheb, fourier);
    if (!with_cap) return;
    ArmSeam<Real> far = seam;
    far.C = Vec3<Real>{seam.C[0]+L*seam.u[0], seam.C[1]+L*seam.u[1], seam.C[2]+L*seam.u[2]};
    add_cap_hemisphere_frame<Real>(Xquad_, far, order_, Ncap, core_frac);
  }

  // Shared arm: ONE slender fiber spanning two coaxial, facing junction seams (no caps). The seams must
  // be collinear and their axes must point at each other (a.u ~ +axis, b.u ~ -axis). The radius is read
  // off the two seams: a LINEAR taper from a.R0 (at seam a) to b.R0 (at seam b), so a differently-sized
  // junction on each end is joined by a tapered tube whose terminal rings match each seam exactly
  // (seam-conforming, watertight). Equal a.R0/b.R0 => the former constant-radius tube (bit-for-bit).
  // Optional sine wiggle: a transverse displacement A*sin^2(pi*t)*sin(2*pi*periods*t) over the WHOLE arm
  // (t = arc fraction in [0,1]), perpendicular to BOTH the axis and the orient e1 (so e1 stays perp to
  // the bent tangent). The sin^2(pi*t) envelope is C-INFINITY on [0,1] and vanishes with zero slope at
  // t=0,1 -> the terminal rings stay circles perpendicular to the axis (seam-conforming), and it tapers
  // to ~0 near both connections (displacement is O(t^3) there) so the wiggle sits in the middle, away
  // from the seams. Being C-infinity everywhere, it has NO derivative kink, so its resolution is
  // INSENSITIVE to how the axial panel boundaries fall (a hard window would put a C2 kink at its edges
  // that only resolves when panel boundaries happen to align with it).
  void add_shared_arm(const ArmSeam<Real>& a, const ArmSeam<Real>& b, Integer n_axial,
                      Long cheb = 10, Long fourier = 12, Real sine_amp = 0, Real sine_periods = 0) {
    const Vec3<Real> d{b.C[0]-a.C[0], b.C[1]-a.C[1], b.C[2]-a.C[2]};
    const Real len = sqrt<Real>(d[0]*d[0]+d[1]*d[1]+d[2]*d[2]);
    SCTL_ASSERT_MSG(len > 0, "add_shared_arm: the two seams coincide.");
    const Vec3<Real> dir{d[0]/len, d[1]/len, d[2]/len};
    auto dot = [](const Vec3<Real>& p, const Vec3<Real>& q){ return p[0]*q[0]+p[1]*q[1]+p[2]*q[2]; };
    SCTL_ASSERT_MSG(dot(a.u, dir) > (Real)0.999, "add_shared_arm: seam A axis must point toward seam B.");
    SCTL_ASSERT_MSG(dot(b.u, dir) < (Real)-0.999, "add_shared_arm: seam B axis must point toward seam A.");
    SCTL_ASSERT_MSG(a.R0 > (Real)0 && b.R0 > (Real)0, "add_shared_arm: seam radii must be positive.");
    // Linear radius taper read off the two seams (constant when a.R0 == b.R0).
    const Real rA = a.R0, rB = b.R0;
    auto rad = [=](Real t) -> Real { return rA + t*(rB - rA); };
    std::function<Vec3<Real>(Real)> disp;
    if (sine_amp != (Real)0) {
      auto cross = [](const Vec3<Real>& p, const Vec3<Real>& q){ return Vec3<Real>{p[1]*q[2]-p[2]*q[1], p[2]*q[0]-p[0]*q[2], p[0]*q[1]-p[1]*q[0]}; };
      Vec3<Real> w = cross(a.e1, dir);                                    // perp to axis AND to orient e1
      const Real wn = sqrt<Real>(w[0]*w[0]+w[1]*w[1]+w[2]*w[2]);
      SCTL_ASSERT_MSG(wn > 1e-9, "add_shared_arm: orient e1 is parallel to the axis; cannot pick a wiggle plane.");
      w[0]/=wn; w[1]/=wn; w[2]/=wn;
      const Real pi = const_pi<Real>();
      disp = [=](Real t) -> Vec3<Real> {
        const Real env = sin<Real>(pi*t); const Real e2 = env*env;       // sin^2(pi t): C-inf, 0+flat at ends
        const Real dd = sine_amp * e2 * sin<Real>(2*pi*sine_periods*t);
        return Vec3<Real>{dd*w[0], dd*w[1], dd*w[2]};
      };
    }
    add_slender_fiber(a.C, dir, a.R0, a.e1, len, n_axial, cheb, fourier, disp, rad);
  }

  // Smootherstep partition-of-unity ramp W(tau): W=0 for tau<=0, W=1 for tau>=1, C2 at both ends
  // (W'=W''=0 there, so the curve joins each straight lead with continuous curvature = 0). It is the
  // DEGREE-5 polynomial 6tau^5-15tau^4+10tau^3, so on the turn span the centerline (this blended with the
  // two linear seam-axis rays) is a degree-6 polynomial that a CSBQ Chebyshev panel of order >=7
  // represents EXACTLY -- no blend representation error (same reason the junction transition uses it,
  // pou_kind=1). The C-infinity bump was tried and is catastrophic here: it is far too sharp in the middle
  // for the panels to resolve (cf. the pou_kind=0 warning for the transition tube). For that exactness the
  // straight-lead/turn junctions must fall ON panel boundaries (the driver snaps lead_frac to n/n_axial).
  static Real pou_ramp(Real tau) {
    if (tau <= (Real)0) return (Real)0;
    if (tau >= (Real)1) return (Real)1;
    const Real t = tau;
    return t*t*t*((Real)6*t*t - (Real)15*t + (Real)10);
  }

  // RACETRACK centerline joining seam a to seam b (a.C -> b.C): a straight coaxial lead along a.u, a
  // smooth POU "shoulder" corner, a STRAIGHT RUN, a second corner, then a straight coaxial lead along b.u
  // into b.C -- the stadium of the target sketch. Its layout is controlled DIRECTLY in panels:
  //   panels [0, lead_panels)                straight lead along a.u (coaxial with the junction arm)
  //   panels [lead_panels, lead_panels+cp)   corner 1 (smootherstep POU blend line A -> run line)
  //   panels [..,  n-lead_panels-cp)         straight RUN
  //   panels [.., n-lead_panels)             corner 2 (run line -> line B)
  //   panels [n-lead_panels, n)              straight lead along b.u
  // The corner VERTEX QA=a.C+L*a.u is placed at the CENTER of corner 1 (t1=(lead_panels+cp/2)/n), with
  // L=t1*|chord|/(1-t1) derived so line A and the run line cross inside the window; QB mirrors on b. Every
  // panel is thus a single line (lead/run) or a degree-6 line-blend, represented EXACTLY by a Chebyshev
  // panel of order>=7 since all window edges are panel boundaries. r(0)=a.C, r(1)=b.C, r'(0)||a.u,
  // r'(1)||b.u exactly (watertight seams, zero curvature next to each seam). Shorter run <- smaller |chord|
  // (junction separation); smaller corners <- smaller cp; shorter coaxial lead <- smaller lead_panels.
  // single_corner=true selects the HALF-racetrack "lead | corner | run" variant (intra-tree connector):
  //   panels [0, lead_panels)                straight lead along a.u (coaxial with the parent branch)
  //   panels [lead_panels, lead_panels+cp)   ONE smootherstep POU corner (line A -> line B directly)
  //   panels [lead_panels+cp, n)             straight RUN along b.u into b.C -- becomes the child stem arm
  // The single corner vertex Q is the intersection of line A (a.C + s*a.u) and line B (b.C + r*b.u); the
  // run plunges straight into b.C so r'(1)||b.u (watertight). Endpoints/tangents exact as in the racetrack.
  static Vec3<Real> bent_centerline(const ArmSeam<Real>& a, const ArmSeam<Real>& b, Real t,
                                    Integer lead_panels = 2, Integer corner_panels = 6,
                                    Integer n_axial = 40, bool single_corner = false,
                                    bool skew_safe = false) {
    auto dot = [](const Vec3<Real>& p, const Vec3<Real>& q){ return p[0]*q[0]+p[1]*q[1]+p[2]*q[2]; };
    auto sub = [](const Vec3<Real>& p, const Vec3<Real>& q){ return Vec3<Real>{p[0]-q[0],p[1]-q[1],p[2]-q[2]}; };
    auto mix = [](const Vec3<Real>& p, const Vec3<Real>& q, Real w){ return Vec3<Real>{(1-w)*p[0]+w*q[0], (1-w)*p[1]+w*q[1], (1-w)*p[2]+w*q[2]}; };
    const Integer n = std::max<Integer>(6, n_axial);
    Integer lp = std::max<Integer>(1, lead_panels), cp = std::max<Integer>(1, corner_panels);
    if (single_corner) {
      if (2*lp + cp >= n) { lp = std::max<Integer>(1, n/6); cp = std::max<Integer>(1, n/3); }   // keep leads+run
      const Vec3<Real> chord = sub(b.C, a.C);
      const Real c = dot(a.u, b.u), p = dot(chord, a.u), q = dot(chord, b.u);
      const Real denom = (Real)1 - c*c;                          // >0: axes non-parallel (add_bent_arm asserts)
      const Real r_ = (p*c - q)/denom, s_ = p + r_*c;            // Q = a.C + s_*a.u = b.C + r_*b.u  (s_,r_>0)
      const Vec3<Real> Q{a.C[0]+s_*a.u[0], a.C[1]+s_*a.u[1], a.C[2]+s_*a.u[2]};   // Q_a: lead endpoint on line A
      // Skew-safe corner: when the two seam axes and the chord are NOT coplanar (3D placement, e.g. tangent
      // on a sphere) line A and line B do not intersect, so a single vertex Q on line A would send the run
      // Q->b.C off-axis (r'(1) not || b.u) and the terminal ring would not conform to seam b -> a leak. End
      // the corner instead at Q_b = b.C + r_*b.u (the closest point on line B): then lineB leaves b.C exactly
      // along b.u (watertight) and the corner bridges the O(1/R) skew gap Q_a->Q_b. When coplanar Q_b == Q_a,
      // so with skew_safe off (every planar caller) this is byte-for-byte the original single-corner curve.
      const Vec3<Real> Qb = skew_safe ? Vec3<Real>{b.C[0]+r_*b.u[0], b.C[1]+r_*b.u[1], b.C[2]+r_*b.u[2]} : Q;
      // Place the corner at Q's TRUE arc fraction tQ=s_/(s_+r_): the lead (length s_) maps to t in [0,tQ]
      // and the run (length r_) to [tQ,1] at the SAME speed s_+r_ -> uniform panels. Center a cp-panel window
      // on the nearest panel boundary. (Forcing Q to a fixed small parameter crushes the lead panels ~10x --
      // non-uniform panels wreck the slender self-quadrature and blow up the DL identity.)
      const Real tQ = s_/(s_ + r_);
      const Integer hcpL = cp/2, hcpR = cp - hcpL;
      Integer kc = (Integer)std::lround((double)(tQ*n));
      kc = std::max<Integer>(hcpL, std::min<Integer>(n - hcpR, kc));
      const Real e1L = (Real)(kc - hcpL)/n, e1R = (Real)(kc + hcpR)/n;
      auto lineA = [&](Real x){ return mix(a.C, Q,  x/tQ); };
      auto lineB = [&](Real x){ return mix(Qb, b.C, (x-tQ)/((Real)1-tQ)); };
      if (t <= e1L) return lineA(t);
      if (t <  e1R) return mix(lineA(t), lineB(t), pou_ramp((t-e1L)/(e1R-e1L)));
      return lineB(t);
    }
    const Real Lc = sqrt<Real>(dot(sub(b.C,a.C), sub(b.C,a.C)));
    if (2*(lp+cp) >= n) { lp = std::max<Integer>(1, n/6); cp = std::max<Integer>(1, n/6); }   // keep a run
    const Real t1 = (Real)(lp + cp*(Real)0.5)/n, t2 = (Real)1 - t1;       // corner vertex params (centered)
    const Real L  = t1*Lc/((Real)1 - t1);                                 // reach so QA sits at t1
    const Vec3<Real> QA{a.C[0]+L*a.u[0], a.C[1]+L*a.u[1], a.C[2]+L*a.u[2]};
    const Vec3<Real> QB{b.C[0]+L*b.u[0], b.C[1]+L*b.u[1], b.C[2]+L*b.u[2]};
    const Real e1L=(Real)lp/n, e1R=(Real)(lp+cp)/n, e2L=(Real)(n-lp-cp)/n, e2R=(Real)(n-lp)/n;
    auto lineA = [&](Real x){ return Vec3<Real>{a.C[0]+(x/t1)*(QA[0]-a.C[0]), a.C[1]+(x/t1)*(QA[1]-a.C[1]), a.C[2]+(x/t1)*(QA[2]-a.C[2])}; };
    auto lineR = [&](Real x){ const Real u=(x-t1)/(t2-t1); return Vec3<Real>{QA[0]+u*(QB[0]-QA[0]), QA[1]+u*(QB[1]-QA[1]), QA[2]+u*(QB[2]-QA[2])}; };
    auto lineB = [&](Real x){ const Real u=(x-t2)/(1-t2); return Vec3<Real>{QB[0]+u*(b.C[0]-QB[0]), QB[1]+u*(b.C[1]-QB[1]), QB[2]+u*(b.C[2]-QB[2])}; };
    if (t <= e1L) return lineA(t);
    if (t <  e1R) return mix(lineA(t), lineR(t), pou_ramp((t-e1L)/(e1R-e1L)));
    if (t <= e2L) return lineR(t);
    if (t <  e2R) return mix(lineR(t), lineB(t), pou_ramp((t-e2L)/(e2R-e2L)));
    return lineB(t);
  }

  // Pull a centerline point radially onto the sphere of centre O so the arm HUGS the surface instead of
  // chording through it (which both looks stiff and lets neighbouring runs dip toward one another under the
  // surface). The pull is windowed to ZERO over the first/last `lead_panels` panels (so the straight,
  // seam-coaxial lead is untouched -> terminal rings stay perpendicular to the seam axes, watertight) and
  // ramps up over the next `corner_panels` via the same smootherstep POU used for the corner; the run is
  // pulled fully onto the target radius (linearly interpolated between the two seam radii, both ~= R). This
  // matches the two-corner window exactly; for the single-corner arm it still keeps the lead panels straight.
  static Vec3<Real> hug_to_sphere(Vec3<Real> r, const Vec3<Real>& aC, const Vec3<Real>& bC, Real t,
                                  Integer lead_panels, Integer corner_panels, Integer n_axial, const Vec3<Real>& O) {
    const Integer n = std::max<Integer>(6, n_axial);
    const Integer lp = std::max<Integer>(1, lead_panels), cp = std::max<Integer>(1, corner_panels);
    Real w0 = (Real)lp/n, w1 = (Real)(lp+cp)/n, w2 = (Real)(n-lp-cp)/n, w3 = (Real)(n-lp)/n;
    if (w1 >= w2) { const Real m = (w0+w3)/2; w1 = w2 = m; }        // short arm: single peak, no flat run
    Real W;
    if (t <= w0 || t >= w3) W = (Real)0;
    else if (t < w1)  W = pou_ramp((t-w0)/(w1-w0));
    else if (t <= w2) W = (Real)1;
    else              W = pou_ramp((w3-t)/(w3-w2));
    if (W <= (Real)0) return r;
    auto sub = [](const Vec3<Real>& p, const Vec3<Real>& q){ return Vec3<Real>{p[0]-q[0],p[1]-q[1],p[2]-q[2]}; };
    auto nrm = [](const Vec3<Real>& p){ return sqrt<Real>(p[0]*p[0]+p[1]*p[1]+p[2]*p[2]); };
    const Real Ra = nrm(sub(aC,O)), Rb = nrm(sub(bC,O)), Rt = Ra + t*(Rb-Ra);   // target radius ~= R
    const Vec3<Real> d = sub(r, O); const Real rr = nrm(d);
    if ((double)rr <= 1e-30) return r;
    const Real f = (Rt - rr)/rr * W;                                // radial move to reach Rt, windowed
    return Vec3<Real>{r[0]+d[0]*f, r[1]+d[1]*f, r[2]+d[2]*f};
  }

  // Racetrack connector: ONE slender fiber joining two seams whose axes need NOT be coaxial (unlike
  // add_shared_arm, which requires them collinear and facing). The centerline (see bent_centerline) leaves
  // each junction STRAIGHT along its arm axis (coaxial, zero curvature at the seam), bends smoothly through
  // a smootherstep POU shoulder, runs straight across, bends again, and enters the far seam straight -- so
  // both terminal rings stay perpendicular to their seam axis (watertight) and all curvature sits at the
  // two shoulders, far from the junctions and caps. Radius tapers a.R0 -> b.R0 (constant if equal).
  //
  // The whole connector is PLANAR in the plane perpendicular to the orient e1, so e1 stays perpendicular to
  // the tangent throughout (a valid moving frame whose rings are perpendicular to the local centerline
  // direction): we require a.e1 == b.e1 and that this common e1 is perpendicular to a.u, b.u, AND the
  // chord. The driver arranges this by rotating both junctions about their shared e1 axis (so all arm axes,
  // seam centers, QA/QB and the run lie in the plane perpendicular to e1); then r'(t) is a combination of
  // vectors all perpendicular to e1, so e1 . r'(t) == 0 for every t.
  //
  // transported=true relaxes the PLANAR-turn requirement (shared e1 perpendicular to both axes and the
  // chord): the two seams may sit on a curved surface (e.g. tangent to a sphere) so the lead/corner/run
  // centerline bends in full 3D. A single constant e1 would then go parallel to the tangent somewhere and
  // divide-by-zero in CSBQ GetGeom; instead a rotation-minimizing (double-reflection) frame is carried per
  // node, seeded by a.e1 at t=0 and re-projected onto the CSBQ tangent inside GetGeom. The straight lead/run
  // keep r'(0)||a.u, r'(1)||b.u, so the terminal rings stay perpendicular to the seam axes (watertight).
  // transported=false is byte-for-byte the original planar arm (no orient functor is passed).
  //
  // tilt_offset adds a transverse mid-arm displacement tilt_offset*sin^2(pi*t) to the centerline: a C-inf
  // bump that vanishes with zero slope at both seams (terminal rings stay perpendicular to the seam axes ->
  // watertight) and peaks (= tilt_offset) at mid-arm. Used to nudge two otherwise-touching arms apart, one
  // "up" and one "down" out of the local tangent plane. Zero (default) leaves the arm unchanged.
  // hug_O (optional): when non-null the run/corner is pulled radially onto the sphere of that centre (see
  // hug_to_sphere) so the arm follows the surface; the leads stay straight. Null (default) = chord path.
  void add_bent_arm(const ArmSeam<Real>& a, const ArmSeam<Real>& b, Integer n_axial,
                    Long cheb = 10, Long fourier = 12, Integer lead_panels = 2, Integer corner_panels = 6,
                    bool single_corner = false, bool transported = false,
                    Vec3<Real> tilt_offset = Vec3<Real>{(Real)0,(Real)0,(Real)0},
                    const Vec3<Real>* hug_O = nullptr) {
    auto dot = [](const Vec3<Real>& p, const Vec3<Real>& q){ return p[0]*q[0]+p[1]*q[1]+p[2]*q[2]; };
    const Vec3<Real> chord{b.C[0]-a.C[0], b.C[1]-a.C[1], b.C[2]-a.C[2]};
    const Real L = sqrt<Real>(dot(chord, chord));
    SCTL_ASSERT_MSG(L > 0, "add_bent_arm: the two seams coincide.");
    const Vec3<Real> cdir{chord[0]/L, chord[1]/L, chord[2]/L};
    SCTL_ASSERT_MSG(a.R0 > (Real)0 && b.R0 > (Real)0, "add_bent_arm: seam radii must be positive.");
    if (single_corner) {
      // lead|corner|run: one rounded corner at Q = line A ^ line B; the run becomes the child stem arm.
      // No chord-direction constraint (the turn may exceed 90 deg); require intersecting, forward-facing lines.
      SCTL_ASSERT_MSG(lead_panels >= 1 && corner_panels >= 1 && (2*lead_panels+corner_panels) < n_axial,
                      "add_bent_arm(single): need lead,corner>=1 and 2*lead+corner<n_axial (room for leads+corner+run).");
      const Real c = dot(a.u, b.u);
      SCTL_ASSERT_MSG(std::fabs((double)c) < 0.9999, "add_bent_arm(single): seam axes must not be (near-)parallel.");
      const Real p = dot(chord, a.u), q = dot(chord, b.u), denom = (Real)1 - c*c;
      const Real r_ = (p*c - q)/denom, s_ = p + r_*c;
      SCTL_ASSERT_MSG(s_ > (Real)0 && r_ > (Real)0,
                      "add_bent_arm(single): corner vertex must lie ahead of both seams (check placement).");
    } else {
      SCTL_ASSERT_MSG(dot(a.u, cdir) > (Real)0, "add_bent_arm: seam A axis must point toward seam B (positive chord component).");
      SCTL_ASSERT_MSG(dot(b.u, cdir) < (Real)0, "add_bent_arm: seam B axis must point toward seam A (negative chord component).");
      SCTL_ASSERT_MSG(lead_panels >= 1 && corner_panels >= 1 && 2*(lead_panels+corner_panels) < n_axial,
                      "add_bent_arm: need lead_panels,corner_panels>=1 and 2*(lead+corner)<n_axial (nonempty run).");
    }
    const Vec3<Real>& e1 = a.e1;
    if (!transported) {
      SCTL_ASSERT_MSG(std::fabs((double)dot(e1, b.e1)) > (Real)0.999, "add_bent_arm: the two seams must share the orient e1.");
      const Real tolp = (Real)1e-6;
      SCTL_ASSERT_MSG(std::fabs((double)dot(e1, a.u)) < tolp && std::fabs((double)dot(e1, b.u)) < tolp
                      && std::fabs((double)dot(e1, cdir)) < tolp,
                      "add_bent_arm: orient e1 must be perpendicular to both seam axes and the chord (planar turn).");
    }
    const ArmSeam<Real> sa = a, sb = b;
    const Integer na = n_axial, lp = lead_panels, cp = corner_panels;
    const bool sc = single_corner;
    // disp(t) = r(t) - straight_chord(t); add_slender_fiber adds this onto C0 + s*cdir (= chord(t)),
    // so the emitted centerline is exactly bent_centerline(t). Vanishes at t=0,1 (endpoints match).
    std::function<Vec3<Real>(Real)> disp = [=](Real t) -> Vec3<Real> {
      Vec3<Real> r = bent_centerline(sa, sb, t, lp, cp, na, sc, /*skew_safe*/transported);
      Real env = sin<Real>(const_pi<Real>()*t); env *= env;               // sin^2(pi t): 0+flat at ends
      r = Vec3<Real>{r[0]+tilt_offset[0]*env, r[1]+tilt_offset[1]*env, r[2]+tilt_offset[2]*env};
      if (hug_O) r = hug_to_sphere(r, a.C, b.C, t, lead_panels, corner_panels, n_axial, *hug_O);
      return Vec3<Real>{r[0]-(a.C[0]+t*chord[0]), r[1]-(a.C[1]+t*chord[1]), r[2]-(a.C[2]+t*chord[2])};
    };
    const Real rA = a.R0, rB = b.R0;
    std::function<Real(Real)> rad = [=](Real t) -> Real { return rA + t*(rB - rA); };
    if (!transported) {
      add_slender_fiber(a.C, cdir, a.R0, e1, L, n_axial, cheb, fourier, disp, rad);
      return;
    }
    // Rotation-minimizing (double-reflection, Wang et al.) reference frame along the 3D-bent centerline,
    // tabulated on a fine uniform grid and linearly interpolated (CSBQ re-orthonormalizes against its own
    // tangent, so a smooth, non-degenerate e1 is all that is required). Seed = a.e1 projected perp to a.u.
    auto sub = [](const Vec3<Real>& p, const Vec3<Real>& q){ return Vec3<Real>{p[0]-q[0],p[1]-q[1],p[2]-q[2]}; };
    auto unit = [&dot](Vec3<Real> p){ const Real n = sqrt<Real>(dot(p,p)); if ((double)n>1e-30){p[0]/=n;p[1]/=n;p[2]/=n;} return p; };
    const Integer Ng = std::max<Integer>((Integer)200, (Integer)20*n_axial);
    auto pos = [&](Real tt){ Vec3<Real> r = bent_centerline(sa, sb, tt, lp, cp, na, sc, /*skew_safe*/transported);
      Real e = sin<Real>(const_pi<Real>()*tt); e *= e;
      r = Vec3<Real>{r[0]+tilt_offset[0]*e, r[1]+tilt_offset[1]*e, r[2]+tilt_offset[2]*e};
      if (hug_O) r = hug_to_sphere(r, a.C, b.C, tt, lead_panels, corner_panels, n_axial, *hug_O);
      return r; };
    auto tang = [&](Integer i) -> Vec3<Real> {
      const Real h = (Real)1/Ng;
      if (i <= 0)  return unit(sub(pos(h), pos((Real)0)));
      if (i >= Ng) return unit(sub(pos((Real)1), pos((Real)1-h)));
      return unit(sub(pos((i+1)*h), pos((i-1)*h)));
    };
    std::vector<Vec3<Real>> Rtab((size_t)Ng+1);
    Vec3<Real> ti = tang(0);
    Vec3<Real> r = unit(sub(a.e1, Vec3<Real>{dot(a.e1,ti)*ti[0], dot(a.e1,ti)*ti[1], dot(a.e1,ti)*ti[2]}));
    Rtab[0] = r;
    for (Integer i = 0; i < Ng; i++) {
      const Vec3<Real> xi = pos(i/(Real)Ng), xi1 = pos((i+1)/(Real)Ng);
      const Vec3<Real> ti1 = tang(i+1);
      const Vec3<Real> v1 = sub(xi1, xi); const Real c1 = dot(v1, v1);
      const Vec3<Real> rL = ((double)c1>0) ? sub(r,  Vec3<Real>{2*dot(v1,r )/c1*v1[0], 2*dot(v1,r )/c1*v1[1], 2*dot(v1,r )/c1*v1[2]}) : r;
      const Vec3<Real> tL = ((double)c1>0) ? sub(ti, Vec3<Real>{2*dot(v1,ti)/c1*v1[0], 2*dot(v1,ti)/c1*v1[1], 2*dot(v1,ti)/c1*v1[2]}) : ti;
      const Vec3<Real> v2 = sub(ti1, tL); const Real c2 = dot(v2, v2);
      Vec3<Real> r1 = ((double)c2>0) ? sub(rL, Vec3<Real>{2*dot(v2,rL)/c2*v2[0], 2*dot(v2,rL)/c2*v2[1], 2*dot(v2,rL)/c2*v2[2]}) : rL;
      r1 = unit(r1);
      Rtab[(size_t)i+1] = r1; r = r1; ti = ti1;
    }
    std::function<Vec3<Real>(Real)> orient = [Rtab, Ng](Real t) -> Vec3<Real> {
      Real g = t*Ng; if ((double)g < 0) g = 0; if (g > (Real)Ng) g = (Real)Ng;
      Integer i = (Integer)g; if (i >= Ng) i = Ng-1; const Real f = g - i;
      const Vec3<Real>& p0 = Rtab[(size_t)i]; const Vec3<Real>& p1 = Rtab[(size_t)i+1];
      Vec3<Real> o{p0[0]+f*(p1[0]-p0[0]), p0[1]+f*(p1[1]-p0[1]), p0[2]+f*(p1[2]-p0[2])};
      const Real n = sqrt<Real>(o[0]*o[0]+o[1]*o[1]+o[2]*o[2]);
      if ((double)n > 1e-30) { o[0]/=n; o[1]/=n; o[2]/=n; return o; } return p0;
    };
    add_slender_fiber(a.C, cdir, a.R0, e1, L, n_axial, cheb, fourier, disp, rad, orient);
  }

  // Combined quad list: ctor replicate-then-slices X across `comm`.
  //
  // Xquad_ is built redundantly on every rank and MUST be bit-identical across them: the ctor below
  // keeps only this rank's element slice and performs NO cross-rank consistency check, so a divergence
  // would silently assemble a wrong global surface (no assert, plausible-looking output). Fold the raw
  // bytes (FNV-1a, so no alignment or type-width assumptions) and require every rank to agree.
  //
  // QJ_DUMP_QUAD=<path> additionally dumps the assembled surface via QuadElemList::Write -- 17-digit
  // ASCII, i.e. exact for double, unlike the float-truncated VTU output. That makes `cmp` a usable
  // bit-identity oracle for the whole assembly path.
  QuadElemList<Real> quad(const Comm& comm = Comm::Self()) const {
    if (comm.Size() > 1 && Xquad_.Dim() > 0) {
      const unsigned char* raw = (const unsigned char*)&Xquad_[0];
      const Long nbytes = Xquad_.Dim() * (Long)sizeof(Real);
      uint64_t h = 1469598103934665603ull;
      for (Long i = 0; i < nbytes; i++) { h ^= (uint64_t)raw[i]; h *= 1099511628211ull; }
      const Long hl = (Long)(h >> 1);                       // >>1 keeps it non-negative for MIN/MAX
      SCTL_ASSERT_MSG(GlobalReduce(hl, comm, CommOp::MIN) == GlobalReduce(hl, comm, CommOp::MAX),
                      "HybridAssembly::quad: the replicated node array differs across ranks.");
    }
    QuadElemList<Real> list(order_, Xquad_, comm);
    if (const char* p = std::getenv("QJ_DUMP_QUAD")) list.Write(p, comm);   // collective; rank 0 writes
    return list;
  }

  // Combined slender list: emit only this rank's global-panel slice (same k0=Nelem*pid/Np partition
  // as BuildYArmsSlender, since SlenderElemList has no comm-aware replicate-then-slice ctor).
  SlenderElemList<Real> slender(const Comm& comm = Comm::Self()) const {
    const Long Nelem = npanel_, Np = comm.Size(), pid = comm.Rank();
    const Long k0 = (Nelem * pid) / Np, k1 = (Nelem * (pid + 1)) / Np;
    Vector<Long> eo, fo; Vector<Real> co, ra, ori;
    Long node0 = 0;
    for (Long p = 0; p < Nelem; p++) {
      const Long cheb = elem_order_[p];
      if (p >= k0 && p < k1) {
        eo.PushBack(elem_order_[p]); fo.PushBack(forder_[p]);
        for (Long j = 0; j < cheb; j++) {
          const Long n = node0 + j;
          co.PushBack(coord_[3*n]); co.PushBack(coord_[3*n+1]); co.PushBack(coord_[3*n+2]);
          ra.PushBack(radius_[n]);
          ori.PushBack(orient_[3*n]); ori.PushBack(orient_[3*n+1]); ori.PushBack(orient_[3*n+2]);
        }
      }
      node0 += cheb;
    }
    return SlenderElemList<Real>(eo, fo, co, ra, ori);
  }
};

} // namespace quad_junctions
