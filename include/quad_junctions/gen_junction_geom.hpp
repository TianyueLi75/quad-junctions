/**
 * GENERALIZED junction iso-surface geometry: an N-arm bifurcation for ARBITRARY branch angles, an
 * arbitrary (incl. EVEN) number of branches, and NON-COPLANAR (3D) arm directions.
 *
 * This is the runtime-configurable superset of ybifurc_geom.hpp. The frozen 3-coplanar-120 kernel in
 * ybifurc_geom.hpp (YField/YCfg/arm_frame/junction_dir) is left byte-untouched -- vessels/tree/multi/
 * channel and their disk caches depend on it bit-for-bit. This header re-implements the same swept-O-grid
 * construction on top of a runtime GenSpec (an arbitrary list of unit arm directions), reusing vslerp
 * (from ybifurc_geom.hpp) and pou_weight/pou_kind (from ybifurc_hybrid_geom.hpp) verbatim, and carrying
 * its own field-generic ray_root / push_oriented (the ybifurc ones are bound to YField concretely).
 *
 * The one new idea is the junction sphere-with-N-holes partition. The old junction_dir is EXACTLY the
 * spherical-Voronoi bisector partition specialized to 3 coplanar directions: the two +-z poles are the
 * two Voronoi vertices of three coplanar directions, and the +-60 offsets are the equatorial midpoints of
 * the two bisector edges of a cell. Generalizing:
 *
 *   - COPLANAR path (arbitrary N, arbitrary in-plane angles, incl. even N): every cell is a "bigon"
 *     between poles +-n (n = common plane normal), bounded by two bisector great-circle arcs to the arm's
 *     two angular neighbours. Same 4-sub-arc O-grid as the old junction_dir but with the equatorial
 *     bisector directions eR/eL = TRUE angular bisectors normalize(u_k+u_nbr) instead of theta_k +- 60.
 *     Adjacent cells share a bisector great circle sampled with the same per-sub-arc panel count and
 *     GL-symmetric nodes -> watertight, exactly as the old kernel.
 *   - NON-COPLANAR path (generic 3D directions): the spherical Voronoi diagram of {u_k}; each cell is the
 *     Voronoi polygon around u_k, O-grid-meshed inner hole-circle -> polygon boundary, every Voronoi EDGE
 *     (keyed by the unordered pair of cells it separates) meshed with one globally-agreed panel count so
 *     the two incident cells place identical boundary nodes -> watertight.
 *
 * Everything downstream is unchanged and generic: the POU transition tube, the R0 azimuthal-mean seam
 * ring, the CSBQ slender arm, and the butterfly hemisphere cap all consume only a per-arm frame (u,e1,e2)
 * + (R0,a0). hybrid_bie_tests.hpp runs the DL / Green's / watertightness tests verbatim.
 */
#pragma once

#include <quad_junctions/ybifurc_geom.hpp>          // Vec3, vslerp (generic); YField (reference type)
#include <quad_junctions/ybifurc_hybrid_geom.hpp>    // pou_weight / pou_kind, Vec2, SlenderElemList
#include <algorithm>
#include <cmath>
#include <map>
#include <utility>
#include <vector>

namespace quad_junctions {
using namespace sctl;

// ============================================================================
// Small Vec3 helpers.
// ============================================================================
template <class Real> inline Real gv_dot(const Vec3<Real>& a, const Vec3<Real>& b) { return a[0]*b[0]+a[1]*b[1]+a[2]*b[2]; }
template <class Real> inline Vec3<Real> gv_cross(const Vec3<Real>& a, const Vec3<Real>& b) {
  return Vec3<Real>{a[1]*b[2]-a[2]*b[1], a[2]*b[0]-a[0]*b[2], a[0]*b[1]-a[1]*b[0]}; }
template <class Real> inline Real gv_norm(const Vec3<Real>& a) { return sqrt<Real>(gv_dot(a,a)); }
template <class Real> inline Vec3<Real> gv_unit(const Vec3<Real>& a) { const Real n = gv_norm(a); return Vec3<Real>{a[0]/n,a[1]/n,a[2]/n}; }
template <class Real> inline Vec3<Real> gv_add(const Vec3<Real>& a, const Vec3<Real>& b) { return Vec3<Real>{a[0]+b[0],a[1]+b[1],a[2]+b[2]}; }
template <class Real> inline Vec3<Real> gv_axpy(Real s, const Vec3<Real>& a, const Vec3<Real>& b) { return Vec3<Real>{s*a[0]+b[0], s*a[1]+b[1], s*a[2]+b[2]}; }
template <class Real> inline Vec3<Real> gv_scal(Real s, const Vec3<Real>& a) { return Vec3<Real>{s*a[0],s*a[1],s*a[2]}; }

// Field-generic star-shaped ray projection (ybifurc ray_root is bound to YField; this takes any Field
// with an f() method). From interior center c0 along direction d, bisect for s>0 with f(c0+s d)=level.
template <class Real, class Field> Vec3<Real> gen_ray_root(const Field& fld, const Vec3<Real>& c0, Vec3<Real> d, Real level, Real* res_out = nullptr) {
  const Real dn = gv_norm(d); d[0]/=dn; d[1]/=dn; d[2]/=dn;
  auto F = [&](Real s){ return fld.f(Vec3<Real>{c0[0]+s*d[0], c0[1]+s*d[1], c0[2]+s*d[2]}) - level; };
  Real lo = (Real)1e-6, hi = (Real)3;
  SCTL_ASSERT_MSG(F(lo) > 0 && F(hi) < 0, "gen_ray_root: center not inside / ray does not cross the level set");
  for (int it = 0; it < 100; it++) { const Real mid = (Real)0.5*(lo+hi); if (F(mid) > 0) lo = mid; else hi = mid; }
  const Real s = (Real)0.5*(lo+hi);
  const Vec3<Real> x{c0[0]+s*d[0], c0[1]+s*d[1], c0[2]+s*d[2]};
  if (res_out) *res_out = std::fabs((double)(fld.f(x) - level));
  return x;
}

// Field-generic outward-orienting node emitter (mirror of ybifurc push_oriented, any Field with grad()).
template <class Real, class Field> void gen_push_oriented(Vector<Real>& X, const Field& fld, std::vector<Vec3<Real>>& nd, Integer order) {
  auto at = [&](Integer i, Integer j) -> Vec3<Real>& { return nd[i*order+j]; };
  const Vec3<Real> tu{at(1,0)[0]-at(0,0)[0], at(1,0)[1]-at(0,0)[1], at(1,0)[2]-at(0,0)[2]};
  const Vec3<Real> tv{at(0,1)[0]-at(0,0)[0], at(0,1)[1]-at(0,0)[1], at(0,1)[2]-at(0,0)[2]};
  const Vec3<Real> nrm{tu[1]*tv[2]-tu[2]*tv[1], tu[2]*tv[0]-tu[0]*tv[2], tu[0]*tv[1]-tu[1]*tv[0]};
  const Vec3<Real> g = fld.grad(at(0,0));
  const bool flip = (nrm[0]*g[0]+nrm[1]*g[1]+nrm[2]*g[2]) > 0;   // outward = -g
  for (Integer i = 0; i < order; i++)
    for (Integer j = 0; j < order; j++) { const Vec3<Real>& p = flip ? at(j,i) : at(i,j);
      X.PushBack(p[0]); X.PushBack(p[1]); X.PushBack(p[2]); }
}

// ============================================================================
// Runtime field/mesh spec. arm_dir = arbitrary unit directions (N >= 2). Defaults reproduce the ybifurc
// field constants so a symmetric-120 config yields the same tube radius scale.
// ============================================================================
template <class Real> struct GenSpec {
  std::vector<Vec3<Real>> arm_dir;
  Real sigma    = (Real)0.15;
  Real amp      = (Real)1.0;
  Real gauss_ds = (Real)0.05;
  Real gauss_len= (Real)0.95;
  Real alpha_deg = (Real)38.0;               // junction hole half-angle (nominal)
  Real alpha_clamp_frac = (Real)0.8;         // hole clamped to this fraction of the min half-gap (tight-gap safety)
  Real Lseam     = (Real)0.88;               // arm-axis arc-length of the tube seam
  Real core_frac = (Real)0.40;               // butterfly-cap core half-size
  Integer Nr0 = 2, Na0 = 16, Ncap0 = 2;      // base panel counts (scaled by nref); Na0 mult of 4
};

// ============================================================================
// Sum-of-Gaussians field, N arbitrary arms.
// ============================================================================
template <class Real> struct NField {
  std::vector<Vec3<Real>> C;
  Real inv2s2, invs2, amp;
  explicit NField(const GenSpec<Real>& s) {
    inv2s2 = (Real)1 / (2 * s.sigma * s.sigma);
    invs2  = (Real)1 / (s.sigma * s.sigma);
    amp    = s.amp;
    C.push_back(Vec3<Real>{0,0,0});
    for (const auto& d0 : s.arm_dir) {
      const Vec3<Real> u = gv_unit(d0);
      for (int k = 1; k * s.gauss_ds <= s.gauss_len + (Real)1e-9; k++)
        C.push_back(Vec3<Real>{k*s.gauss_ds*u[0], k*s.gauss_ds*u[1], k*s.gauss_ds*u[2]});
    }
  }
  Real f(const Vec3<Real>& x) const {
    Real s = 0;
    for (const auto& c : C) { const Real dx=x[0]-c[0], dy=x[1]-c[1], dz=x[2]-c[2]; s += amp*exp<Real>(-(dx*dx+dy*dy+dz*dz)*inv2s2); }
    return s;
  }
  Vec3<Real> grad(const Vec3<Real>& x) const {
    Vec3<Real> g{0,0,0};
    for (const auto& c : C) { const Real dx=x[0]-c[0], dy=x[1]-c[1], dz=x[2]-c[2];
      const Real e = amp*exp<Real>(-(dx*dx+dy*dy+dz*dz)*inv2s2)*invs2; g[0]-=e*dx; g[1]-=e*dy; g[2]-=e*dz; }
    return g;
  }
};

// ============================================================================
// Junction geometry: per-arm frames + sphere-with-N-holes partition. gen_junction_dir dispatches on mode.
// ============================================================================
template <class Real> struct JuncGeom {
  Integer N = 0;
  bool coplanar = true;
  bool bigon3 = false;                        // N==3 non-coplanar: generalized-pole bigon (see gen_junction_dir_bigon)
  Vec3<Real> n{0,0,1};                        // common plane normal / poles +-n (coplanar); circumcenter axis for bigon3
  Real alpha = 0;                             // hole half-angle (rad)
  std::vector<Vec3<Real>> u, e1, e2;          // per-arm axis + azimuthal frame (beta=0 -> e1)
  std::vector<Vec3<Real>> eR, eL;             // coplanar: equatorial bisector dirs (right/left neighbour)
  // voronoi partition (non-coplanar):
  std::vector<Vec3<Real>> vtx;                        // unique Voronoi vertex directions
  std::vector<std::vector<Integer>> cell_vtx;         // per-cell ordered boundary vertices (CCW about u)
  std::vector<std::vector<Integer>> cell_np;          // per-cell per-edge panel count
  std::vector<Integer> cell_Na;                       // per-cell total azimuthal panels
};

// e1 = up made perpendicular to u (robust); e2 = e1 x u.
template <class Real> void gen_frame_from_up(const Vec3<Real>& u, const Vec3<Real>& up, Vec3<Real>& e1, Vec3<Real>& e2) {
  const Real d = gv_dot(up, u);
  Vec3<Real> e = gv_axpy<Real>(-d, u, up);
  if (gv_norm(e) < (Real)1e-8) { Vec3<Real> a{1,0,0}; if (std::fabs((double)u[0]) > 0.9) a = Vec3<Real>{0,1,0}; e = gv_axpy<Real>(-gv_dot(a,u), u, a); }
  e1 = gv_unit(e);
  e2 = gv_unit(gv_cross(e1, u));
}

// Non-coplanar spherical-Voronoi partition (implemented after the coplanar tier is validated).
template <class Real> void build_voronoi_partition(JuncGeom<Real>& jg, const GenSpec<Real>& s, Integer nref);

// Build a JuncGeom from a spec (+nref for panel density). Detects coplanarity; builds the partition.
template <class Real> JuncGeom<Real> build_junc_geom(const GenSpec<Real>& s, Integer nref) {
  JuncGeom<Real> jg;
  const Integer N = (Integer)s.arm_dir.size();
  SCTL_ASSERT_MSG(N >= 2, "build_junc_geom: need at least 2 arms.");
  jg.N = N;
  jg.u.resize(N); for (Integer k = 0; k < N; k++) jg.u[k] = gv_unit(s.arm_dir[k]);

  // Junction hole half-angle: the cone around u_k must fit INSIDE the arm's Voronoi cell, i.e. stay
  // clear of the bisector to its nearest neighbour (which sits at half the pairwise arm angle). The
  // nominal alpha_deg (38, from the symmetric 120 Y where the nearest bisector is at 60) is clamped to
  // 0.7 * (min pairwise arm half-angle) so tight gaps (e.g. the 60 gap in 150-150-60 -> bisector at 30)
  // do not push the hole through the cell boundary and fold the O-grid.
  Real min_ang = const_pi<Real>();
  for (Integer i = 0; i < N; i++) for (Integer j = i+1; j < N; j++) {
    Real d = gv_dot(jg.u[i], jg.u[j]); d = std::max<Real>((Real)-1, std::min<Real>((Real)1, d));
    min_ang = std::min(min_ang, acos<Real>(d));
  }
  jg.alpha = std::min<Real>(s.alpha_deg * const_pi<Real>() / 180, s.alpha_clamp_frac * (Real)0.5 * min_ang);

  Vec3<Real> n{0,0,0};
  for (Integer j = 1; j < N && gv_norm(n) < (Real)1e-6; j++) { const Vec3<Real> c = gv_cross(jg.u[0], jg.u[j]); if (gv_norm(c) > (Real)1e-6) n = gv_unit(c); }
  bool coplanar = (gv_norm(n) > (Real)1e-6);
  if (coplanar) for (Integer k = 0; k < N; k++) if (std::fabs((double)gv_dot(jg.u[k], n)) > 1e-7) { coplanar = false; break; }
  jg.coplanar = coplanar;

  jg.e1.resize(N); jg.e2.resize(N);
  if (coplanar) {
    jg.n = n;
    for (Integer k = 0; k < N; k++) gen_frame_from_up<Real>(jg.u[k], n, jg.e1[k], jg.e2[k]);
    const Vec3<Real> a0 = jg.u[0], b0 = gv_cross(n, jg.u[0]);
    std::vector<std::pair<double,Integer>> ang(N);
    for (Integer k = 0; k < N; k++) ang[k] = { std::atan2((double)gv_dot(jg.u[k],b0), (double)gv_dot(jg.u[k],a0)), k };
    std::sort(ang.begin(), ang.end());
    std::vector<Integer> pos(N); for (Integer i = 0; i < N; i++) pos[ang[i].second] = i;
    jg.eR.resize(N); jg.eL.resize(N);
    for (Integer k = 0; k < N; k++) {
      const Integer i = pos[k];
      const Integer kR = ang[(i+1)%N].second, kL = ang[(i+N-1)%N].second;
      SCTL_ASSERT_MSG(gv_dot(jg.u[k], jg.u[kR]) > (Real)-0.9999 && gv_dot(jg.u[k], jg.u[kL]) > (Real)-0.9999,
                      "build_junc_geom: adjacent arms ~antiparallel (gap >= 180 deg); not a valid junction.");
      jg.eR[k] = gv_unit(gv_add(jg.u[k], jg.u[kR]));
      jg.eL[k] = gv_unit(gv_add(jg.u[k], jg.u[kL]));
    }
    jg.cell_Na.assign(N, s.Na0*nref);                   // uniform 4-sub-arc bigon count (mult of 4)
  } else if (N == 3) {
    // Three NON-coplanar arms. The spherical Voronoi of 3 sites has exactly two vertices and they are
    // ANTIPODAL (both circumcenters of the great-circle triangle) -> every cell is a digon whose two
    // edges meet at those poles, which the Voronoi path (endpoint slerp) cannot represent. This is the
    // same antipodal-poles-plus-bisector-midpoint structure as the coplanar bigon, so reuse that path
    // with the pole GENERALIZED from the plane-normal to the circumcenter axis (u0-u1)x(u0-u2). For
    // coplanar 3 arms that axis is +-n, so this branch never fires for them (the coplanar path does).
    const Vec3<Real> P = gv_unit(gv_cross(gv_axpy<Real>((Real)-1, jg.u[1], jg.u[0]),
                                          gv_axpy<Real>((Real)-1, jg.u[2], jg.u[0])));
    jg.n = P; jg.bigon3 = true;
    for (Integer k = 0; k < N; k++) gen_frame_from_up<Real>(jg.u[k], P, jg.e1[k], jg.e2[k]);
    // Cyclic order about the pole P (honest azimuth in P's equatorial frame; reduces to a0=u0,b0=Pxu0
    // when coplanar). Determines each arm's left/right neighbours for the bisector walls.
    const Vec3<Real> pe1 = gv_unit(gv_axpy<Real>(-gv_dot(jg.u[0],P), P, jg.u[0])), pe2 = gv_cross(P, pe1);
    std::vector<std::pair<double,Integer>> ang(N);
    for (Integer k = 0; k < N; k++) ang[k] = { std::atan2((double)gv_dot(jg.u[k],pe2), (double)gv_dot(jg.u[k],pe1)), k };
    std::sort(ang.begin(), ang.end());
    std::vector<Integer> pos(N); for (Integer i = 0; i < N; i++) pos[ang[i].second] = i;
    jg.eR.resize(N); jg.eL.resize(N);
    // Bisector-edge waypoint for arms (k,o). The cell's outer boundary is a lune bounded by the two
    // bisector great circles, each passing through the poles +-P; voro_edge returns the point on the (k,o)
    // bisector GC that lies on the TRUE Voronoi-edge half -- the half where k,o are the two nearest sites.
    //   Bug it fixes: the old code used the near-midpoint c=U(u_k+u_o). For a WIDELY-separated pair (gap
    //   ->180 deg, common in an asymmetric non-coplanar junction) that midpoint is far from both arms and
    //   closest to the THIRD arm, so it sits on the WRONG half of the GC -> the 4-sub-arc walk enclosed the
    //   wrong lune, the cell missed u_k, and the junction leaked ~0.4-0.6 order-INDEPENDENTLY (premade
    //   j06/j08). (minWt stayed >0: a coverage bug, not a fold.)
    //   Fix: keep c=U(u_k+u_o) UNCHANGED whenever it is already on the correct half (dot(c,u_k)>=
    //   dot(c,u_third)) -- so tri3d and every already-working premade j0x stay BIT-IDENTICAL. Only on the
    //   wrong half replace c, and NOT by its antipode -c (which for a wide pair sits near a POLE -> grossly
    //   skewed sub-arcs, edge aspect ~33). Use the correct half's ARC MIDPOINT = the meridian's equator
    //   crossing t=+-unit((u_k-u_o)x P), which balances the Ptop->eR and eR->Pbot sub-arcs (aspect ~9).
    // (These wide-pair cells are aspect-limited: watertight+spectral, but run them under SCTL_SELF_SCHEME=
    // duffy to reach the wide-config DL/Green floor.) The COPLANAR path is separate and untouched.
    auto voro_edge = [&](Integer k, Integer o, Integer third) -> Vec3<Real> {
      Vec3<Real> c = gv_unit(gv_add(jg.u[k], jg.u[o]));                          // near-midpoint on the bisector GC
      if (gv_dot(c, jg.u[third]) > gv_dot(c, jg.u[k])) {                         // wrong half -> arc midpoint of correct half
        Vec3<Real> t = gv_unit(gv_cross(gv_axpy<Real>((Real)-1, jg.u[o], jg.u[k]), P));
        if (gv_dot(t, jg.u[third]) > gv_dot(t, jg.u[k])) t = gv_scal<Real>((Real)-1, t);
        c = t;
      }
      return c;
    };
    for (Integer k = 0; k < N; k++) {
      const Integer i = pos[k];
      const Integer kR = ang[(i+1)%N].second, kL = ang[(i+N-1)%N].second;
      SCTL_ASSERT_MSG(gv_dot(jg.u[k], jg.u[kR]) > (Real)-0.9999 && gv_dot(jg.u[k], jg.u[kL]) > (Real)-0.9999,
                      "build_junc_geom: adjacent arms ~antiparallel (gap >= 180 deg); not a valid junction.");
      jg.eR[k] = voro_edge(k, kR, kL);
      jg.eL[k] = voro_edge(k, kL, kR);
    }
    jg.cell_Na.assign(N, s.Na0*nref);
  } else {
    const Vec3<Real> up{0,0,1};
    for (Integer k = 0; k < N; k++) gen_frame_from_up<Real>(jg.u[k], up, jg.e1[k], jg.e2[k]);
    build_voronoi_partition<Real>(jg, s, nref);
  }
  return jg;
}

// Per-arm frame accessor.
template <class Real> void gen_arm_frame(const JuncGeom<Real>& jg, Integer k, Vec3<Real>& u, Vec3<Real>& e1, Vec3<Real>& e2) {
  u = jg.u[k]; e1 = jg.e1[k]; e2 = jg.e2[k];
}

// ============================================================================
// COPLANAR sphere-with-N-holes direction (cell k): identical structure to ybifurc junction_dir, with
// eR/eL = true angular bisectors and poles +-n. t: 0 hole -> 1 arcs; s: azimuth [0,1) (4 sub-arcs).
// ============================================================================
template <class Real> Vec3<Real> gen_junction_dir_coplanar(const JuncGeom<Real>& jg, Integer k, Real t, Real s) {
  const Real pi = const_pi<Real>();
  const Vec3<Real>& e1 = jg.e1[k]; const Vec3<Real>& e2 = jg.e2[k]; const Vec3<Real>& u = jg.u[k];
  const Vec3<Real> Ptop = jg.n, Pbot = gv_scal<Real>((Real)-1, jg.n);
  const Vec3<Real>& eR = jg.eR[k]; const Vec3<Real>& eL = jg.eL[k];
  const Real beta = 2*pi*s;
  const Vec3<Real> rad = gv_add(gv_scal<Real>(cos<Real>(beta), e1), gv_scal<Real>(sin<Real>(beta), e2));
  const Vec3<Real> inner = gv_add(gv_scal<Real>(cos<Real>(jg.alpha), u), gv_scal<Real>(sin<Real>(jg.alpha), rad));
  const Real seg = s*4; Vec3<Real> A, B; Real w;
  if      (seg < 1) { A = Ptop; B = eR; w = seg-0; }
  else if (seg < 2) { A = eR; B = Pbot; w = seg-1; }
  else if (seg < 3) { A = Pbot; B = eL; w = seg-2; }
  else              { A = eL; B = Ptop; w = seg-3; }
  return vslerp<Real>(inner, vslerp<Real>(A, B, w), t);
}

// ============================================================================
// NON-COPLANAR 3-arm (bigon3) direction (cell k): same Ptop->eR->Pbot->eL 4-sub-arc outer boundary as
// the coplanar bigon, but with the generalized circumcenter poles (jg.n) -- and because the arms are
// tilted off jg.n's equator, eR/eL no longer project to azimuth 90/270 in the (e1,e2) frame, so the
// inner hole ring's azimuth is ALIGNED to the outer boundary azimuth (as in the Voronoi path) instead of
// uniform 2pi s, or the radial O-grid lines would bow across azimuths and skew the cell.
// ============================================================================
template <class Real> Real voro_azimuth(const JuncGeom<Real>& jg, Integer k, const Vec3<Real>& d);  // defined below
template <class Real> Vec3<Real> bigon_outer(const JuncGeom<Real>& jg, Integer k, Real s) {
  const Vec3<Real> Ptop = jg.n, Pbot = gv_scal<Real>((Real)-1, jg.n);
  const Vec3<Real>& eR = jg.eR[k]; const Vec3<Real>& eL = jg.eL[k];
  const Real seg = s*4; Vec3<Real> A, B; Real w;
  if      (seg < 1) { A = Ptop; B = eR; w = seg-0; }
  else if (seg < 2) { A = eR; B = Pbot; w = seg-1; }
  else if (seg < 3) { A = Pbot; B = eL; w = seg-2; }
  else              { A = eL; B = Ptop; w = seg-3; }
  return vslerp<Real>(A, B, w);
}
template <class Real> Vec3<Real> gen_junction_dir_bigon(const JuncGeom<Real>& jg, Integer k, Real t, Real s) {
  const Vec3<Real>& e1 = jg.e1[k]; const Vec3<Real>& e2 = jg.e2[k]; const Vec3<Real>& u = jg.u[k];
  const Vec3<Real> outer = bigon_outer<Real>(jg, k, s);
  const Real beta = voro_azimuth<Real>(jg, k, outer);           // inner ring aligned to the outer azimuth
  const Vec3<Real> rad = gv_add(gv_scal<Real>(cos<Real>(beta), e1), gv_scal<Real>(sin<Real>(beta), e2));
  const Vec3<Real> inner = gv_add(gv_scal<Real>(cos<Real>(jg.alpha), u), gv_scal<Real>(sin<Real>(jg.alpha), rad));
  return vslerp<Real>(inner, outer, t);
}

// ============================================================================
// NON-COPLANAR sphere-with-N-holes direction (cell k): O-grid annulus from the inner hole ring (cone
// alpha, azimuth beta=2pi s aligned to e1_k) out to the cell's Voronoi polygon boundary, walked as
// consecutive great-circle arcs between the cell's ordered Voronoi vertices. s in [0,1) partitions into
// cell_Na[k] azimuthal panels aligned to the per-edge panel counts (so panel boundaries land on Voronoi
// vertices; adjacent cells share each edge with identical panel count -> watertight).
// ============================================================================
// The Voronoi cell's outer-boundary direction at azimuthal parameter s in [0,1) (walk the polygon edges).
template <class Real> Vec3<Real> voro_outer(const JuncGeom<Real>& jg, Integer k, Real s) {
  const std::vector<Integer>& vv = jg.cell_vtx[k]; const std::vector<Integer>& np = jg.cell_np[k];
  const Integer ne = (Integer)vv.size(), M = jg.cell_Na[k];
  Real spanel = s*M; if (spanel >= (Real)M) spanel = (Real)M - (Real)1e-12;
  Integer cum = 0, e = 0; for (; e < ne; e++) { if (spanel < cum + np[e]) break; cum += np[e]; }
  if (e >= ne) { e = ne-1; cum = M - np[e]; }
  const Real frac = (spanel - cum) / np[e];
  return vslerp<Real>(jg.vtx[vv[e]], jg.vtx[vv[(e+1)%ne]], frac);
}

// Azimuth (around u_k, in the e1_k/e2_k frame) of a direction d -- the aligned beta so the O-grid radial
// lines stay at constant azimuth and do NOT bow outside the (convex, site-containing) Voronoi cell.
template <class Real> Real voro_azimuth(const JuncGeom<Real>& jg, Integer k, const Vec3<Real>& d) {
  const Real ox = gv_dot(d, jg.e1[k]), oy = gv_dot(d, jg.e2[k]);
  Real b = atan2<Real>(oy, ox); if (b < 0) b += 2*const_pi<Real>(); return b;
}

// Junction azimuth beta(s) used by BOTH the junction inner hole ring and the transition tube (so their
// shared seam conforms). Coplanar: uniform 2pi s (the eR/eL bisectors are already azimuth-aligned).
// Voronoi: the azimuth of the outer boundary point at s (aligns inner<->outer, prevents cell overlap).
template <class Real> Real gen_azimuth(const JuncGeom<Real>& jg, Integer k, Real s) {
  if (jg.coplanar) return 2*const_pi<Real>()*s;
  if (jg.bigon3)   return voro_azimuth<Real>(jg, k, bigon_outer<Real>(jg, k, s));
  return voro_azimuth<Real>(jg, k, voro_outer<Real>(jg, k, s));
}

template <class Real> Vec3<Real> gen_junction_dir_voronoi(const JuncGeom<Real>& jg, Integer k, Real t, Real s) {
  const Vec3<Real>& e1 = jg.e1[k]; const Vec3<Real>& e2 = jg.e2[k]; const Vec3<Real>& u = jg.u[k];
  const Vec3<Real> outer = voro_outer<Real>(jg, k, s);
  const Real beta = voro_azimuth<Real>(jg, k, outer);            // inner ring aligned to the outer azimuth
  const Vec3<Real> rad = gv_add(gv_scal<Real>(cos<Real>(beta), e1), gv_scal<Real>(sin<Real>(beta), e2));
  const Vec3<Real> inner = gv_add(gv_scal<Real>(cos<Real>(jg.alpha), u), gv_scal<Real>(sin<Real>(jg.alpha), rad));
  return vslerp<Real>(inner, outer, t);
}

// Dispatch to the coplanar (bigon) or non-coplanar (Voronoi polygon) junction partition.
template <class Real> Vec3<Real> gen_junction_dir(const JuncGeom<Real>& jg, Integer k, Real t, Real s) {
  if (jg.coplanar) return gen_junction_dir_coplanar<Real>(jg, k, t, s);
  if (jg.bigon3)   return gen_junction_dir_bigon<Real>(jg, k, t, s);
  return gen_junction_dir_voronoi<Real>(jg, k, t, s);
}

// ============================================================================
// Arm tube point (star-shaped ray on NField + gen frame). Same math as ybifurc arm_point.
// ============================================================================
template <class Real> Vec3<Real> gen_arm_point(const NField<Real>& fld, const JuncGeom<Real>& jg, Integer k,
                                               Real eta, Real beta, Real level, Real Lseam, Real* res = nullptr) {
  Vec3<Real> u, e1, e2; gen_arm_frame<Real>(jg, k, u, e1, e2);
  const Vec3<Real> c = gv_scal<Real>(eta*Lseam, u);
  const Vec3<Real> rad = gv_add(gv_scal<Real>(cos<Real>(beta), e1), gv_scal<Real>(sin<Real>(beta), e2));
  const Real cu = (1-eta)*cos<Real>(jg.alpha), cr = (1-eta)*sin<Real>(jg.alpha)+eta;
  const Vec3<Real> d = gv_add(gv_scal<Real>(cu, u), gv_scal<Real>(cr, rad));
  return gen_ray_root<Real>(fld, c, d, level, res);
}

// Azimuthal-mean ring stats of arm k at axial parameter eta (periodic trapezoid over beta).
template <class Real> void gen_arm_ring_stats(const NField<Real>& fld, const JuncGeom<Real>& jg, Integer k, Real eta,
                                              Real level, Real Lseam, Real& R0, Real& a_axial, Integer Nq = 64) {
  Vec3<Real> u, e1, e2; gen_arm_frame<Real>(jg, k, u, e1, e2);
  const Real twopi = 2*const_pi<Real>(); Real sR = 0, sA = 0;
  for (Integer m = 0; m < Nq; m++) {
    const Real beta = twopi*m/Nq;
    const Vec3<Real> P = gen_arm_point<Real>(fld, jg, k, eta, beta, level, Lseam);
    const Real sax = gv_dot(P, u);
    const Vec3<Real> perp = gv_axpy<Real>(-sax, u, P);
    sR += gv_norm(perp); sA += sax;
  }
  R0 = sR/Nq; a_axial = sA/Nq;
}

// POU-blended transition-tube point (blend true field ring -> exact R0 circle). a_loc supplied per-eta.
template <class Real> Vec3<Real> gen_transition_point(const NField<Real>& fld, const JuncGeom<Real>& jg, Integer k,
                                                      Real eta, Real beta, Real level, Real Lseam, Real R0, Real eta_w, Real a_loc) {
  Vec3<Real> u, e1, e2; gen_arm_frame<Real>(jg, k, u, e1, e2);
  const Real w = pou_weight<Real>(eta / eta_w);
  const Vec3<Real> Ptrue = gen_arm_point<Real>(fld, jg, k, eta, beta, level, Lseam);
  const Real cb = cos<Real>(beta), sb = sin<Real>(beta);
  const Vec3<Real> Pcirc = gv_add(gv_scal<Real>(a_loc, u), gv_scal<Real>(R0, gv_add(gv_scal<Real>(cb, e1), gv_scal<Real>(sb, e2))));
  return Vec3<Real>{w*Ptrue[0]+(1-w)*Pcirc[0], w*Ptrue[1]+(1-w)*Pcirc[1], w*Ptrue[2]+(1-w)*Pcirc[2]};
}

// Butterfly-dome hemisphere cap for arm k (equator = R0 circle at s_cap; node-conforms to the tube end).
template <class Real> void gen_add_arm_cap_hemisphere(Vector<Real>& X, const NField<Real>& fld, const JuncGeom<Real>& jg,
                                                      Integer order, Integer k, Real R0, Real s_cap, Integer Ncap, Real core_frac) {
  const Real pi = const_pi<Real>();
  Vec3<Real> u, e1, e2; gen_arm_frame<Real>(jg, k, u, e1, e2);
  const Vec3<Real> C{s_cap*u[0], s_cap*u[1], s_cap*u[2]};
  const Real h = core_frac; const Integer nc = std::max<Integer>(1, Ncap);
  const Vector<Real>& nds = QuadElemList<Real>::ParamNodes(order);
  std::vector<Vec3<Real>> nd(order*order);
  auto elev = [&](Real Dx, Real Dy) -> Vec3<Real> {
    const Real q = sqrt<Real>(Dx*Dx+Dy*Dy), psi = q*pi/2;
    const Real r_lat = R0*sin<Real>(psi), h_ax = R0*cos<Real>(psi);
    const Real roq = (q > (Real)1e-9) ? r_lat/q : R0*pi/2;
    return Vec3<Real>{C[0]+roq*(Dx*e1[0]+Dy*e2[0])+h_ax*u[0], C[1]+roq*(Dx*e1[1]+Dy*e2[1])+h_ax*u[1], C[2]+roq*(Dx*e1[2]+Dy*e2[2])+h_ax*u[2]};
  };
  for (Integer ic = 0; ic < nc; ic++)
    for (Integer jc = 0; jc < nc; jc++) {
      const Real x0=-h+2*h*ic/nc, x1=-h+2*h*(ic+1)/nc, y0=-h+2*h*jc/nc, y1=-h+2*h*(jc+1)/nc;
      for (Integer i = 0; i < order; i++) { const Real yy = y0+nds[i]*(y1-y0);
        for (Integer j = 0; j < order; j++) { const Real xx = x0+nds[j]*(x1-x0); nd[i*order+j] = elev(xx, yy); } }
      gen_push_oriented<Real>(X, fld, nd, order);
    }
  for (Integer kk = 0; kk < 4; kk++) {
    const Real rot = kk*pi/2, cr = cos<Real>(rot), sr = sin<Real>(rot);
    auto pt = [=](Real eta, Real xi) -> Vec2<Real> {
      const Real th = -pi/4 + xi*(pi/2);
      const Vec2<Real> in{h, h*(2*xi-1)}, out{cos<Real>(th), sin<Real>(th)};
      const Real px = (1-eta)*in[0]+eta*out[0], py = (1-eta)*in[1]+eta*out[1];
      return Vec2<Real>{cr*px - sr*py, sr*px + cr*py};
    };
    for (Integer ir = 0; ir < nc; ir++)
      for (Integer ia = 0; ia < nc; ia++) {
        const Real e0=(Real)ir/nc, e1c=(Real)(ir+1)/nc, a0=(Real)ia/nc, a1=(Real)(ia+1)/nc;
        for (Integer i = 0; i < order; i++) { const Real xi = a0+nds[i]*(a1-a0);
          for (Integer j = 0; j < order; j++) { const Real eta = e0+nds[j]*(e1c-e0); const Vec2<Real> P = pt(eta, xi); nd[i*order+j] = elev(P[0], P[1]); } }
        gen_push_oriented<Real>(X, fld, nd, order);
      }
  }
}

// ============================================================================
// Build the QuadElemList half: N junction sectors + N POU transition tubes + N R0-hemisphere caps. Fills
// per-arm R0/a0/s_cap (sized N). Mirrors BuildYJunctionWithTransitions with the generalized kernel.
// ============================================================================
// emit_caps=false leaves each arm's seam ring OPEN (no hemisphere cap): the network assembler
// (gen_network_geom.hpp) attaches a bent slender arm to that open ring instead, and caps only the
// leaf tips itself. Every existing caller omits the flag and gets the capped body bit-for-bit.
template <class Real> QuadElemList<Real> BuildGenJunctionWithTransitions(
    const GenSpec<Real>& spec, const JuncGeom<Real>& jg, Integer order, Real level, Integer nref,
    Real eta_join, Integer Ns_trans, Real s_cap_arc, std::vector<Real>& R0_out, std::vector<Real>& a0_out,
    std::vector<Real>& s_cap_out, Integer Ncap = -1, Real* max_res = nullptr, const Comm& comm = Comm::Self(),
    bool emit_caps = true) {
  const NField<Real> fld(spec);
  const Vector<Real>& nds = QuadElemList<Real>::ParamNodes(order);
  const Integer Nr = spec.Nr0*nref;                         // Na is PER-ARM (jg.cell_Na[k]) below
  const Integer NcapUse = (Ncap > 0) ? Ncap : spec.Ncap0*nref;
  const Real pi = const_pi<Real>();
  SCTL_ASSERT_MSG(Ns_trans >= 2, "Ns_trans must be >= 2.");
  const Real eta_w = eta_join*(Real)(Ns_trans-1)/Ns_trans;
  const Integer N = jg.N;
  R0_out.assign(N,0); a0_out.assign(N,0); s_cap_out.assign(N,s_cap_arc);
  Real rmax = 0, r; std::vector<Vec3<Real>> nd(order*order); Vector<Real> X;

  for (Integer k = 0; k < N; k++) {
    const Integer Na = jg.cell_Na[k];                       // per-arm azimuthal panel count (cell boundary)
    Real R0, a_join; gen_arm_ring_stats<Real>(fld, jg, k, eta_join, level, spec.Lseam, R0, a_join, 64);
    R0_out[k] = R0; a0_out[k] = a_join;

    for (Integer ir = 0; ir < Nr; ir++)
      for (Integer ia = 0; ia < Na; ia++) {
        const Real t0=(Real)ir/Nr, t1=(Real)(ir+1)/Nr, s0=(Real)ia/Na, s1=(Real)(ia+1)/Na;
        for (Integer i = 0; i < order; i++) { const Real t = t0+nds[i]*(t1-t0);
          for (Integer j = 0; j < order; j++) { const Real s = s0+nds[j]*(s1-s0);
            nd[i*order+j] = gen_ray_root<Real>(fld, Vec3<Real>{0,0,0}, gen_junction_dir<Real>(jg, k, t, s), level, &r); rmax = std::max(rmax, r); } }
        gen_push_oriented<Real>(X, fld, nd, order);
      }

    std::vector<Real> a_tab(order);
    for (Integer l = 0; l < Ns_trans; l++) {
      const Real e0 = eta_join*(Real)l/Ns_trans, e1t = eta_join*(Real)(l+1)/Ns_trans;
      for (Integer i = 0; i < order; i++) { const Real eta = e0+nds[i]*(e1t-e0); Real R0_loc; gen_arm_ring_stats<Real>(fld, jg, k, eta, level, spec.Lseam, R0_loc, a_tab[i], 64); }
      for (Integer ia = 0; ia < Na; ia++) {
        const Real b0=(Real)ia/Na, b1=(Real)(ia+1)/Na;
        for (Integer i = 0; i < order; i++) { const Real eta = e0+nds[i]*(e1t-e0);
          for (Integer j = 0; j < order; j++) { const Real beta = gen_azimuth<Real>(jg, k, b0+nds[j]*(b1-b0));
            nd[i*order+j] = gen_transition_point<Real>(fld, jg, k, eta, beta, level, spec.Lseam, R0, eta_w, a_tab[i]); } }
        gen_push_oriented<Real>(X, fld, nd, order);
      }
    }
    if (emit_caps) gen_add_arm_cap_hemisphere<Real>(X, fld, jg, order, k, R0, s_cap_arc, NcapUse, spec.core_frac);
  }
  if (max_res) *max_res = rmax;
  return QuadElemList<Real>(order, X, comm);
}

// ============================================================================
// Build the SlenderElemList half: N straight constant-R0 fibers, each a0[k]*u_k -> s_cap[k]*u_k, with
// orientation e1[k] so beta=0 -> e1 matches the transition tube/cap. MPI: CSBQ k0=Nelem*pid/Np partition.
// ============================================================================
template <class Real> SlenderElemList<Real> BuildGenArmsSlender(
    const JuncGeom<Real>& jg, const std::vector<Real>& R0, const std::vector<Real>& a0, const std::vector<Real>& s_cap,
    Integer n_axial, Long cheb_order = 10, Long fourier_order = 12, const Comm& comm = Comm::Self()) {
  const Integer N = jg.N;
  const Long Nelem = (Long)N*n_axial, Np = comm.Size(), pid = comm.Rank();
  const Long k0g = (Nelem*pid)/Np, k1g = (Nelem*(pid+1))/Np;
  Vector<Long> elem_order, forder; Vector<Real> coord, radius, orient; Long eg = 0;
  for (Integer k = 0; k < N; k++) {
    Vec3<Real> u, e1, e2; gen_arm_frame<Real>(jg, k, u, e1, e2);
    const Real s0 = a0[k], s1 = s_cap[k];
    for (Integer p = 0; p < n_axial; p++, eg++) {
      if (eg < k0g || eg >= k1g) continue;
      elem_order.PushBack(cheb_order); forder.PushBack(fourier_order);
      const Vector<Real>& cn = SlenderElemList<Real>::CenterlineNodes(cheb_order);
      for (Long j = 0; j < cheb_order; j++) {
        const Real s = s0 + (s1-s0)*(p+cn[j])/n_axial;
        coord.PushBack(s*u[0]); coord.PushBack(s*u[1]); coord.PushBack(s*u[2]);
        radius.PushBack(R0[k]);
        orient.PushBack(e1[0]); orient.PushBack(e1[1]); orient.PushBack(e1[2]);
      }
    }
  }
  return SlenderElemList<Real>(elem_order, forder, coord, radius, orient);
}

// ============================================================================
// Spherical Voronoi partition of the arm directions {u_k} (non-coplanar generic case). Fills jg.vtx
// (unique Voronoi vertex directions), jg.cell_vtx[k] (CCW-ordered boundary vertices of cell k), and the
// per-edge panel counts jg.cell_np[k] / jg.cell_Na[k]. Panel counts are keyed by the (unordered) vertex
// pair so the two cells sharing an edge agree -> watertight. Assumes valence-3 vertices / polygon cells
// (degenerate coplanar digons are handled by the coplanar path, not here).
// ============================================================================
template <class Real> void build_voronoi_partition(JuncGeom<Real>& jg, const GenSpec<Real>& s, Integer nref) {
  const Integer N = jg.N;
  const Real tol = (Real)1e-6;
  std::vector<std::vector<Integer>> inc;   // incident arm set per unique vertex (parallel to jg.vtx)

  // 1+2. candidate Voronoi vertices from all arm triples, deduped, incident sets merged.
  auto add_cand = [&](Vec3<Real> d) {
    const Real dn = gv_norm(d); if (dn < (Real)1e-12) return; d = gv_scal<Real>((Real)1/dn, d);
    for (int sgn = -1; sgn <= 1; sgn += 2) {
      const Vec3<Real> dd = gv_scal<Real>((Real)sgn, d);
      Real m = (Real)-2; for (Integer p = 0; p < N; p++) m = std::max(m, gv_dot(dd, jg.u[p]));
      std::vector<Integer> ai; for (Integer p = 0; p < N; p++) if (gv_dot(dd, jg.u[p]) > m - tol) ai.push_back(p);
      if ((Integer)ai.size() < 3) continue;                  // not a real Voronoi vertex
      Integer at = -1;
      for (size_t b = 0; b < jg.vtx.size(); b++) if (gv_dot(dd, jg.vtx[b]) > (Real)1 - (Real)1e-10) { at = (Integer)b; break; }
      if (at < 0) { jg.vtx.push_back(dd); inc.push_back(ai); }
      else for (Integer p : ai) if (std::find(inc[at].begin(), inc[at].end(), p) == inc[at].end()) inc[at].push_back(p);
    }
  };
  for (Integer i = 0; i < N; i++) for (Integer j = i+1; j < N; j++) for (Integer l = j+1; l < N; l++)
    add_cand(gv_cross(gv_axpy<Real>((Real)-1, jg.u[j], jg.u[i]), gv_axpy<Real>((Real)-1, jg.u[l], jg.u[i])));
  SCTL_ASSERT_MSG(jg.vtx.size() >= 2, "build_voronoi_partition: found < 2 Voronoi vertices (degenerate config).");

  // 3. per-cell ordered boundary vertices (CCW about u_k).
  jg.cell_vtx.assign(N, {});
  for (Integer k = 0; k < N; k++) {
    std::vector<std::pair<double,Integer>> ord;
    for (size_t v = 0; v < jg.vtx.size(); v++)
      if (std::find(inc[v].begin(), inc[v].end(), k) != inc[v].end())
        ord.push_back({ std::atan2((double)gv_dot(jg.vtx[v], jg.e2[k]), (double)gv_dot(jg.vtx[v], jg.e1[k])), (Integer)v });
    std::sort(ord.begin(), ord.end());
    SCTL_ASSERT_MSG(ord.size() >= 3, "build_voronoi_partition: a cell has < 3 boundary vertices (digon / degenerate); use the coplanar path for such configs.");
    for (auto& p : ord) jg.cell_vtx[k].push_back(p.second);
  }

  // 4+5. per-edge panel counts, keyed by sorted vertex pair (shared count -> watertight).
  const Real h = const_pi<Real>() / (Real)(8*nref);        // target arc length per panel (~ coplanar density)
  std::map<std::pair<int,int>,Integer> npair;
  auto edge_np = [&](Integer a, Integer b) -> Integer {
    std::pair<int,int> key{ std::min((int)a,(int)b), std::max((int)a,(int)b) };
    auto it = npair.find(key); if (it != npair.end()) return it->second;
    Real d = gv_dot(jg.vtx[a], jg.vtx[b]); d = std::max<Real>((Real)-1, std::min<Real>((Real)1, d));
    const Real arclen = acos<Real>(d);
    const Integer n = std::max<Integer>(2, (Integer)std::lround((double)(arclen / h)));
    npair[key] = n; return n;
  };
  jg.cell_np.assign(N, {}); jg.cell_Na.assign(N, 0);
  for (Integer k = 0; k < N; k++) {
    const std::vector<Integer>& vv = jg.cell_vtx[k]; const Integer ne = (Integer)vv.size();
    Integer tot = 0;
    for (Integer e = 0; e < ne; e++) { const Integer n = edge_np(vv[e], vv[(e+1)%ne]); jg.cell_np[k].push_back(n); tot += n; }
    jg.cell_Na[k] = tot;
  }
}

} // namespace quad_junctions
