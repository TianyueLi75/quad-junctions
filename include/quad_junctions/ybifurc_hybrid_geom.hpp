/**
 * M2 HYBRID Y-bifurcation geometry: QuadElemList junction (+ POU transition tubes + caps) joined to
 * CSBQ SlenderElemList circular arms, fed into ONE BoundaryIntegralOp.
 *
 * The M1 arms (fully-meshed swept-O-grid tubes in ybifurc_geom.hpp) are replaced by constant-radius
 * slender limbs. The non-circular iso-surface cross-section near the junction is bridged to the
 * circular slender cross-section by a PARTITION OF UNITY (POU) blend inside a short QuadElemList
 * "transition tube":
 *
 *   P(eta,beta) = w(eta) * P_true(eta,beta) + (1 - w(eta)) * P_circle(eta,beta)
 *
 * with w the C-infinity bump (w=1 at the junction hole seam -> exact true field, watertight with the
 * junction; w=0 on the outer panel -> exact circle of radius R0 = azimuthal-MEAN radius of the true
 * ring at eta_join). From that circle the arm is a SlenderElemList straight cylinder of constant R0,
 * closed at the tip by a butterfly dome rebuilt on the R0 circle. This yields two circle<->circle
 * cross-list seams per arm (base transition<->slender, tip slender<->cap).
 *
 * This header reuses the frozen M1 builder machinery (YField, ray_root, arm_frame, junction_dir,
 * arm_point, push_oriented) unchanged; it only adds new free functions alongside them.
 */
#pragma once

#include <quad_junctions/ybifurc_geom.hpp>
#include <sctl/experimental/quad_element.hpp>
// CSBQ SlenderElemList (declared in namespace sctl). csbq.hpp is the umbrella header; the driver
// already includes it, but include the slender element here so this header is self-contained.
#include <csbq/slender_element.hpp>
#include <csbq/slender_element.cpp>

namespace quad_junctions {
using namespace sctl;

template <class Real> using Vec2 = std::array<Real, 2>;   // (ybifurc_geom.hpp defines only Vec3)

// ============================================================================
// Azimuthal-mean ring statistics of arm k at axial parameter eta: R0 = mean_beta r_perp,
// a_axial = mean_beta (P . u), by the periodic trapezoid rule (spectral for the 2pi-periodic ring).
// r_perp is the in-plane radial distance to the arm axis (the ray {t*u} through the ORIGIN), so the
// decomposition P = (P.u) u + perp is exact.
// ============================================================================
template <class Real> void arm_ring_stats(const YField<Real>& fld, int k, Real eta, Real level,
                                          Real& R0, Real& a_axial, Integer Nq = 64) {
  Vec3<Real> u, e1, e2; arm_frame<Real>(k, u, e1, e2);
  const Real twopi = 2 * const_pi<Real>();
  Real sR = 0, sA = 0;
  for (Integer m = 0; m < Nq; m++) {
    const Real beta = twopi * m / Nq;
    const Vec3<Real> P = arm_point<Real>(fld, k, eta, beta, level);
    const Real sax = P[0]*u[0] + P[1]*u[1] + P[2]*u[2];
    const Real px = P[0]-sax*u[0], py = P[1]-sax*u[1], pz = P[2]-sax*u[2];
    sR += sqrt<Real>(px*px + py*py + pz*pz);
    sA += sax;
  }
  R0 = sR / Nq;
  a_axial = sA / Nq;
}

// ============================================================================
// Partition-of-unity weight over tau in [0,1]: w(0)=1, w(1)=0; tau<=0 -> 1, tau>=1 -> 0.
// pou_kind() selects the profile (settable by the driver):
//   0 = C-infinity bump  g(1-tau)/(g(1-tau)+g(tau)), g(x)=exp(-1/x). All derivatives vanish at both
//       ends (best panel-boundary smoothness) BUT very sharp in the interior -> hard for a modest
//       polynomial order to represent, which caps the transition-tube geometry accuracy.
//   1 = smootherstep 1 - (6 tau^5 - 15 tau^4 + 10 tau^3): only C2 at the ends, but a degree-5
//       polynomial that an order>=6 GL panel represents EXACTLY -> no blend representation error.
// ============================================================================
inline int& pou_kind() { static int k = 1; return k; }   // default: smootherstep (order-exact)

template <class Real> Real pou_weight(Real tau) {
  if (tau <= (Real)0) return (Real)1;
  if (tau >= (Real)1) return (Real)0;
  if (pou_kind() == 0) {
    auto g = [](Real x) -> Real { return x > (Real)0 ? exp<Real>(-(Real)1 / x) : (Real)0; };
    const Real ga = g((Real)1 - tau), gb = g(tau);
    return ga / (ga + gb);
  }
  const Real t = tau, S = t*t*t*((Real)6*t*t - (Real)15*t + (Real)10);   // smootherstep
  return (Real)1 - S;
}

// ============================================================================
// Blended transition-tube point for arm k. eta in [0,eta_join]; the blend runs over [0,eta_w] and is
// clamped to the pure circle on [eta_w,eta_join] (so the outer panel(s) are an exact surface of
// revolution and the terminal edge is exactly the circle of radius R0 centered on the axis).
// R0 is FIXED (the mean radius at eta_join, passed in); the axial center a(eta) is recomputed locally
// so the circle slides smoothly along the axis.
// ============================================================================
// a_loc = axial center of the local ring (mean over beta), so P_circle rides the true ring's axial
// station. It depends ONLY on eta -- never on beta -- so the caller supplies it from a per-eta table
// (see the emit loops); that hoists the 64-sample arm_ring_stats out of the per-node inner loop.
template <class Real> Vec3<Real> transition_point(const YField<Real>& fld, int k, Real eta, Real beta,
                                                  Real level, Real R0, Real eta_w, Real a_loc) {
  Vec3<Real> u, e1, e2; arm_frame<Real>(k, u, e1, e2);
  const Real w = pou_weight<Real>(eta / eta_w);
  const Vec3<Real> Ptrue = arm_point<Real>(fld, k, eta, beta, level);
  const Real cb = cos<Real>(beta), sb = sin<Real>(beta);
  const Vec3<Real> Pcirc{a_loc*u[0] + R0*(cb*e1[0]+sb*e2[0]),
                         a_loc*u[1] + R0*(cb*e1[1]+sb*e2[1]),
                         a_loc*u[2] + R0*(cb*e1[2]+sb*e2[2])};
  return Vec3<Real>{w*Ptrue[0] + (1-w)*Pcirc[0], w*Ptrue[1] + (1-w)*Pcirc[1], w*Ptrue[2] + (1-w)*Pcirc[2]};
}

// Back-compat form: computes the ring's axial center itself (the original per-node behavior). Kept so
// the 7-arg signature of the frozen canonical builder stays available to any other caller.
template <class Real> Vec3<Real> transition_point(const YField<Real>& fld, int k, Real eta, Real beta,
                                                  Real level, Real R0, Real eta_w) {
  Real R0_loc, a_loc; arm_ring_stats<Real>(fld, k, eta, level, R0_loc, a_loc, 64);
  return transition_point<Real>(fld, k, eta, beta, level, R0, eta_w, a_loc);
}

// ============================================================================
// Butterfly-dome cap for arm k: a hemisphere of radius R0 centered on the axis at station s_cap. The
// equator (q=1) is exactly the R0 circle at s_cap => node-conforms to the slender tube's terminal
// circle. Same non-degenerate O-grid as the M1 cap / collar_mount add_cap_butterfly (central gnomonic
// square core [-h,h]^2 + 4 Coons arc blocks to the unit circle), elevated onto the hemisphere.
// Ncap panels per direction. push_oriented gives outward normals (aligned with -grad f).
// ============================================================================
template <class Real> void add_arm_cap_hemisphere(Vector<Real>& X, const YField<Real>& fld, Integer order,
                                                  int k, Real R0, Real s_cap, Integer Ncap, Real core_frac = (Real)0.40) {
  const Real pi = const_pi<Real>();
  Vec3<Real> u, e1, e2; arm_frame<Real>(k, u, e1, e2);
  const Vec3<Real> C{s_cap*u[0], s_cap*u[1], s_cap*u[2]};
  const Real h = core_frac;                              // core half-size in UNIT tangent-disk coords
  const Integer nc = std::max<Integer>(1, Ncap);
  const Vector<Real>& nds = QuadElemList<Real>::ParamNodes(order);
  std::vector<Vec3<Real>> nd(order*order);
  // Elevate a unit tangent-disk point (Dx,Dy), q=|D| in [0,1], onto the R0 hemisphere in the arm frame.
  auto elev = [&](Real Dx, Real Dy) -> Vec3<Real> {
    const Real q = sqrt<Real>(Dx*Dx + Dy*Dy), psi = q * pi / 2;
    const Real r_lat = R0 * sin<Real>(psi), h_ax = R0 * cos<Real>(psi);
    const Real roq = (q > (Real)1e-9) ? r_lat / q : R0 * pi / 2;      // r_lat/q -> R0*pi/2 as q->0
    return Vec3<Real>{C[0] + roq*(Dx*e1[0]+Dy*e2[0]) + h_ax*u[0],
                      C[1] + roq*(Dx*e1[1]+Dy*e2[1]) + h_ax*u[1],
                      C[2] + roq*(Dx*e1[2]+Dy*e2[2]) + h_ax*u[2]};
  };
  // Central gnomonic-square core, nc x nc panels over [-h,h]^2 (i=y slow, j=x fast).
  for (Integer ic = 0; ic < nc; ic++)
    for (Integer jc = 0; jc < nc; jc++) {
      const Real x0 = -h + 2*h*ic/nc, x1 = -h + 2*h*(ic+1)/nc, y0 = -h + 2*h*jc/nc, y1 = -h + 2*h*(jc+1)/nc;
      for (Integer i = 0; i < order; i++) { const Real yy = y0 + nds[i]*(y1-y0);
        for (Integer j = 0; j < order; j++) { const Real xx = x0 + nds[j]*(x1-x0);
          nd[i*order+j] = elev(xx, yy); } }
      push_oriented<Real>(X, fld, nd, order);
    }
  // 4 Coons arc blocks: core edge (unit coords) -> unit circle (= hemisphere equator = R0 ring).
  for (Integer kk = 0; kk < 4; kk++) {
    const Real rot = kk * pi / 2, cr = cos<Real>(rot), sr = sin<Real>(rot);
    auto pt = [=](Real eta, Real xi) -> Vec2<Real> {
      const Real th = -pi/4 + xi * (pi/2);
      const Vec2<Real> in{h, h*(2*xi - 1)};                 // core right edge in unit coords
      const Vec2<Real> out{cos<Real>(th), sin<Real>(th)};   // unit circle
      const Real px = (1-eta)*in[0] + eta*out[0], py = (1-eta)*in[1] + eta*out[1];
      return Vec2<Real>{cr*px - sr*py, sr*px + cr*py};
    };
    for (Integer ir = 0; ir < nc; ir++)
      for (Integer ia = 0; ia < nc; ia++) {
        const Real e0 = (Real)ir/nc, e1c = (Real)(ir+1)/nc, a0 = (Real)ia/nc, a1 = (Real)(ia+1)/nc;
        for (Integer i = 0; i < order; i++) { const Real xi = a0 + nds[i]*(a1-a0);
          for (Integer j = 0; j < order; j++) { const Real eta = e0 + nds[j]*(e1c-e0);
            const Vec2<Real> P = pt(eta, xi); nd[i*order+j] = elev(P[0], P[1]); } }
        push_oriented<Real>(X, fld, nd, order);
      }
  }
}

// ============================================================================
// Build the QuadElemList half of the hybrid: junction (3 sphere-with-3-holes sectors) + 3 POU
// transition tubes + 3 R0-hemisphere caps. Returns per-arm R0[3] and axial start a0[3] (= a(eta_join))
// for the slender builder, plus s_cap[3] (= where the cap sits, also the slender tube end).
// eta_join: end of the transition band; Ns_trans: transition axial panels (>=2; the outer panel is
// clamped to the pure circle). s_cap_arc: axial station (arc length along u) of the tube->cap seam.
// ============================================================================
template <class Real> QuadElemList<Real> BuildYJunctionWithTransitions(
    Integer order, Real level, Integer nref, Real eta_join, Integer Ns_trans, Real s_cap_arc,
    Real R0_out[3], Real a0_out[3], Real s_cap_out[3], Integer Ncap = -1, Real* max_res = nullptr,
    const Comm& comm = Comm::Self()) {
  const YField<Real> fld;
  const Vector<Real>& nds = QuadElemList<Real>::ParamNodes(order);
  const Integer Nr = YSwept::Nr0*nref, Na = YSwept::Na0*nref;
  const Integer NcapUse = (Ncap > 0) ? Ncap : YSwept::Ncap0*nref;
  const Real pi = const_pi<Real>();
  SCTL_ASSERT_MSG(Ns_trans >= 2, "Ns_trans must be >= 2 (blend panels + one pure-circle clamp panel).");
  const Real eta_w = eta_join * (Real)(Ns_trans - 1) / Ns_trans;   // clamp w=0 on the outer panel
  Real rmax = 0, r;
  std::vector<Vec3<Real>> nd(order*order);
  Vector<Real> X;

  for (int k = 0; k < 3; k++) {
    // per-arm R0 (mean radius at eta_join) and axial start a(eta_join)
    Real R0, a_join; arm_ring_stats<Real>(fld, k, eta_join, level, R0, a_join, 64);
    R0_out[k] = R0; a0_out[k] = a_join; s_cap_out[k] = s_cap_arc;

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
      const Real e0 = eta_join*(Real)l/Ns_trans, e1 = eta_join*(Real)(l+1)/Ns_trans;
      for (Integer i = 0; i < order; i++) { const Real eta = e0 + nds[i]*(e1-e0);
        Real R0_loc; arm_ring_stats<Real>(fld, k, eta, level, R0_loc, a_tab[i], 64); }
      for (Integer ia = 0; ia < Na; ia++) {
        const Real b0 = (Real)ia/Na, b1 = (Real)(ia+1)/Na;
        for (Integer i = 0; i < order; i++) { const Real eta = e0 + nds[i]*(e1-e0);
          for (Integer j = 0; j < order; j++) { const Real beta = 2*pi*(b0 + nds[j]*(b1-b0));
            nd[i*order+j] = transition_point<Real>(fld, k, eta, beta, level, R0, eta_w, a_tab[i]); } }
        push_oriented<Real>(X, fld, nd, order);
      }
    }

    // ---- arm k cap: butterfly hemisphere of radius R0 at s_cap ----
    add_arm_cap_hemisphere<Real>(X, fld, order, k, R0, s_cap_arc, NcapUse);
  }
  if (max_res) *max_res = rmax;
  // X is built identically on every rank; the ctor keeps only this rank's element slice for `comm`.
  return QuadElemList<Real>(order, X, comm);
}

// ============================================================================
// Build the SlenderElemList half: 3 straight constant-R0 fibers, each from a0[k]*u_k to s_cap[k]*u_k,
// with per-node orientation e1 (=z) so the azimuthal phase (beta=0 -> +z) matches the QuadElemList
// transition tube and cap. Both fiber ends are OPEN at radius R0 (caps and junction are QuadElemList).
// n_axial: axial panels per arm; cheb_order (10; tables) and fourier_order (mult of 4; e.g. 12).
// ============================================================================
// SlenderElemList has no comm-aware ctor (unlike QuadElemList it does NOT replicate-then-slice),
// so under MPI the caller must partition the element loop itself and pass only this rank's local
// elements to Init -- the CSBQ k0=Nelem*pid/Np pattern (see SlenderElemList::test). The 3*n_axial
// arm panels are indexed by a single global counter `eg`; each rank keeps eg in [k0g,k1g). Some
// ranks may get zero elements (fine). Default Comm::Self() keeps the full serial list.
template <class Real> SlenderElemList<Real> BuildYArmsSlender(
    const Real R0[3], const Real a0[3], const Real s_cap[3], Integer n_axial,
    Long cheb_order = 10, Long fourier_order = 12, const Comm& comm = Comm::Self()) {
  const Long Nelem = (Long)3 * n_axial, Np = comm.Size(), pid = comm.Rank();
  const Long k0g = (Nelem * pid) / Np, k1g = (Nelem * (pid + 1)) / Np;   // this rank's global element range
  Vector<Long> elem_order, forder;
  Vector<Real> coord, radius, orient;
  Long eg = 0;   // global panel index across all three arms
  for (int k = 0; k < 3; k++) {
    Vec3<Real> u, e1, e2; arm_frame<Real>(k, u, e1, e2);
    const Real s0 = a0[k], s1 = s_cap[k];
    for (Integer p = 0; p < n_axial; p++, eg++) {
      if (eg < k0g || eg >= k1g) continue;   // not owned by this rank
      elem_order.PushBack(cheb_order);
      forder.PushBack(fourier_order);
      const Vector<Real>& cn = SlenderElemList<Real>::CenterlineNodes(cheb_order);
      for (Long j = 0; j < cheb_order; j++) {
        const Real s = s0 + (s1 - s0) * (p + cn[j]) / n_axial;   // straight station along u
        coord.PushBack(s*u[0]); coord.PushBack(s*u[1]); coord.PushBack(s*u[2]);
        radius.PushBack(R0[k]);
        orient.PushBack(e1[0]); orient.PushBack(e1[1]); orient.PushBack(e1[2]);
      }
    }
  }
  return SlenderElemList<Real>(elem_order, forder, coord, radius, orient);
}

// ============================================================================
// Build the QUAD-TUBE half (M3 shaft-swap A/B counterpart of BuildYArmsSlender): 3 straight
// constant-R0 cylinders, each from a0[k]*u_k to s_cap[k]*u_k, meshed Ns x Na order-x-order GL panels.
// This is a fully-quad drop-in replacement for the slender arms -- the geometry (same axis, same
// [a0,s_cap] span, same radius R0, same beta=0 -> +z phase as the transition/cap) is IDENTICAL to the
// slender fiber's surface, only the element type differs. The tube's base ring (s=a0) and tip ring
// (s=s_cap) are exactly the R0 circles the transition terminal edge and cap equator sit on =>
// geometrically coincident (watertight) but node-non-conforming (fine: a cross-list near seam, exactly
// like the slender). push_oriented auto-orients each panel outward via -grad f (as in the cap builder).
// Na must be >= 2 (a closed tube needs at least two azimuthal panels; Na=1 leaves a beta=0/2pi gap).
// For the A/B match set Na = fourier_order/order so total azimuthal nodes Na*order == fourier_order.
// ============================================================================
template <class Real> QuadElemList<Real> BuildYArmsQuadTube(
    Integer order, const Real R0[3], const Real a0[3], const Real s_cap[3],
    Integer Ns, Integer Na, const Comm& comm = Comm::Self()) {
  SCTL_ASSERT_MSG(Na >= 2, "BuildYArmsQuadTube: Na must be >= 2 (a closed tube needs >=2 azimuthal panels).");
  SCTL_ASSERT_MSG(Ns >= 1, "BuildYArmsQuadTube: Ns must be >= 1.");
  const YField<Real> fld;   // used by push_oriented only for the outward-normal sign
  const Vector<Real>& nds = QuadElemList<Real>::ParamNodes(order);
  const Real twopi = 2 * const_pi<Real>();
  std::vector<Vec3<Real>> nd(order*order);
  Vector<Real> X;
  for (int k = 0; k < 3; k++) {
    Vec3<Real> u, e1, e2; arm_frame<Real>(k, u, e1, e2);
    const Real s0 = a0[k], s1 = s_cap[k], R = R0[k];
    for (Integer l = 0; l < Ns; l++)
      for (Integer ia = 0; ia < Na; ia++) {
        const Real sa = s0 + (s1-s0)*(Real)l/Ns, sb = s0 + (s1-s0)*(Real)(l+1)/Ns;   // axial panel [sa,sb]
        const Real b0 = (Real)ia/Na, b1 = (Real)(ia+1)/Na;                            // azimuthal panel
        for (Integer i = 0; i < order; i++) { const Real s = sa + nds[i]*(sb-sa);
          for (Integer j = 0; j < order; j++) { const Real beta = twopi*(b0 + nds[j]*(b1-b0));
            const Real cb = cos<Real>(beta), sb2 = sin<Real>(beta);
            nd[i*order+j] = Vec3<Real>{s*u[0] + R*(cb*e1[0]+sb2*e2[0]),
                                       s*u[1] + R*(cb*e1[1]+sb2*e2[1]),
                                       s*u[2] + R*(cb*e1[2]+sb2*e2[2])}; } }
        push_oriented<Real>(X, fld, nd, order);
      }
  }
  // X is built identically on every rank; the ctor keeps only this rank's element slice for `comm`.
  return QuadElemList<Real>(order, X, comm);
}

} // namespace quad_junctions
