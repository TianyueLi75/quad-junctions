#pragma once
/**
 * Spiral "flagella" centerline for the twirling-cilia hybrid geometry (a C++ port of python/flagella.py,
 * used as loose inspiration). ONE source of truth (mirrors vessels_build.hpp / hybrid_bie_tests.hpp) so
 * the CSBQ SlenderElemList shaft builder and the QuadElemList base builder (which places the butterfly
 * cap at the spiral tip) agree bit-for-bit at the seams.
 *
 * Each finger centerline is built as the ybifurc RACETRACK (ybifurc_assembly.hpp::bent_centerline):
 *   panels [0, lead_panels)                 STRAIGHT lead along -u from the fillet-bottom Cs*u
 *   panels [lead_panels, lead_panels+corner) SMOOTHERSTEP-POU corner: blend straight lead -> spiral
 *   panels [lead_panels+corner, n_axial)     the flagella SPIRAL to the tip
 * Because pou_ramp (smootherstep 6t^5-15t^4+10t^3) has zero 1st AND 2nd derivative at each window edge,
 * the join is C2 (zero-curvature continuous). The straight lead => the finger leaves EXACTLY
 * perpendicular to the sphere, C2-matching the fillet (a revolution about u). Window edges fall ON panel
 * boundaries (M8 hard rule: a panel straddling a C2-blend seam re-introduced a ~3000x blowup). NEVER use
 * the C-infinity bump for the ramp -- smootherstep only.
 *
 * The cross-section radius equals R_shaft at the foot (== the fillet-bottom / slender-foot seam ring) and
 * holds the python tube radius r_py along the shaft; if R_shaft > r_py it tapers R_shaft->r_py over the
 * first Ntaper panels (the "finger fatter than the script" branch). The default operating point sets
 * R_shaft ~ r_py (scale the base to the script) so the tube is constant radius.
 */
#include <sctl.hpp>
#include <algorithm>
#include <cmath>
#include <vector>

namespace quad_junctions {
using namespace sctl;

// Smootherstep partition-of-unity ramp (== ybifurc_assembly.hpp::pou_ramp). C2: value/1st/2nd derivative
// all vanish at tau=0 and tau=1, so a straight-lead <-> spiral blend across a panel-aligned window is C2.
template <class Real> inline Real flagella_pou_ramp(Real tau) {
  if (tau <= (Real)0) return (Real)0;
  if (tau >= (Real)1) return (Real)1;
  const Real t = tau;
  return t*t*t*((Real)6*t*t - (Real)15*t + (Real)10);
}

// Shape + geometry parameters. Shape constants default to python/flagella.py; the geometry fields
// (R_shaft/r_fil/lead_panels/corner_panels/n_axial/Ntaper) are set by the builders.
template <class Real> struct FlagellaCfg {
  Real R          = (Real)1.0;    // sphere radius
  Real L          = (Real)0.9;    // thread length (flagella.py L)
  Real turns      = (Real)2.2;    // spiral winding count (flagella.py TURNS)
  Real max_tilt   = (Real)0.35;   // axis lean by the tip (flagella.py MAX_TILT)
  Real bend_amt   = (Real)0.32;   // tangential curl magnitude (flagella.py BEND_AMT)
  Real spiral_amp = (Real)0.13;   // spiral offset amplitude (flagella.py SPIRAL_AMP)
  Real r_py       = (Real)0.05;   // python tube radius (target shaft radius, flagella.py TUBE_RADIUS)
  bool bias       = true;         // reproduce the +z-anchor bias-toward-+y (flagella.py req 8)

  Real R_shaft    = (Real)0.05;   // foot ring radius (== fillet-bottom / slender-foot seam ring)
  Real r_fil      = (Real)0.005;  // fillet radius (sets the foot station Cs)
  Integer lead_panels   = 3;      // straight normal-lead panels
  Integer corner_panels = 8;      // smootherstep-POU corner panels (M8: the Green's accuracy limiter)
  Integer n_axial       = 40;     // total axial panels per finger
  Integer Ntaper        = 5;      // radius taper panels (only used when R_shaft > r_py)
  Integer tip_lead_panels   = 2;  // straight tip-lead panels (=> exact slender-tip<->cap seam, like the foot)
  Integer tip_corner_panels = 6;  // smootherstep-POU corner blending the spiral into the straight tip lead
};

// One finger's centerline, frames and radius profile, built from the patch axis u (= flagella anchor n).
template <class Real> struct FlagellumCurve {
  FlagellaCfg<Real> cfg;
  Real u[3], u_fl[3], v_fl[3];            // anchor axis + the flagella.py tangent frame
  Real bias_dir[3]; Real bias_amt;        // per-anchor bias (nonzero only for the +z anchor)
  Real Cs;                                // fillet-bottom station along u (foot start of the slender)
  Real e1L, e1R;                          // foot corner window edges in arc-fraction f (panel boundaries)
  Real e2L, e2R;                          // tip  corner window edges in arc-fraction f (panel boundaries)
  Real P2[3], Sp2[3];                     // spiral point + df-derivative at e2L (the straight tip-lead ray)

  static Real dot3(const Real* a, const Real* b) { return a[0]*b[0]+a[1]*b[1]+a[2]*b[2]; }
  static void cross3(const Real* a, const Real* b, Real* o) { o[0]=a[1]*b[2]-a[2]*b[1]; o[1]=a[2]*b[0]-a[0]*b[2]; o[2]=a[0]*b[1]-a[1]*b[0]; }
  static void nrm3(Real* a) { const Real m = std::sqrt(dot3(a,a)); if (m>0){a[0]/=m;a[1]/=m;a[2]/=m;} }
  static Real smoother(Real x) { if (x<=(Real)0) return (Real)0; if (x>=(Real)1) return (Real)1; return x*x*x*(x*((Real)6*x-(Real)15)+(Real)10); }

  FlagellumCurve(const FlagellaCfg<Real>& c, const Real axis[3]) : cfg(c) {
    for (int k=0;k<3;k++) u[k]=axis[k];
    nrm3(u);
    // flagella.py frame: ref = (0,1,0) if |u_y|<0.9 else (1,0,0); u_fl = u x ref; v_fl = u x u_fl.
    Real ref[3] = { (Real)0, (Real)1, (Real)0 };
    if (std::fabs(u[1]) >= (Real)0.9) { ref[0]=(Real)1; ref[1]=(Real)0; }
    cross3(u, ref, u_fl); nrm3(u_fl);
    cross3(u, u_fl, v_fl); nrm3(v_fl);
    // bias only on the +z anchor: toward +y, projected perpendicular to u (flagella.py BIAS_*).
    bias_amt = (Real)0; bias_dir[0]=bias_dir[1]=bias_dir[2]=(Real)0;
    if (cfg.bias && std::fabs(u[0]) < (Real)1e-6 && std::fabs(u[1]) < (Real)1e-6 && u[2] > (Real)0.999) {
      Real bd[3] = { (Real)0, (Real)1, (Real)0 };
      const Real d = dot3(bd, u); for (int k=0;k<3;k++) bd[k] -= d*u[k]; nrm3(bd);
      for (int k=0;k<3;k++) bias_dir[k]=bd[k];
      bias_amt = (Real)0.18;
    }
    // Foot station: the exact tangent-arc fillet-bottom station along u (matches add_cilium_stud /
    // BuildAllFingerSphereBase), so the slender foot ring == the fillet-bottom ring.
    const Real R = cfg.R, R_foot = cfg.R_shaft + cfg.r_fil, a0 = std::sqrt(R*R - R_foot*R_foot);
    const Real rho_arc = cfg.r_fil / (1 + R_foot/R);
    Cs = a0 * (1 - rho_arc/R);
    const Integer n = std::max<Integer>(6, cfg.n_axial);
    const Integer lp = std::max<Integer>(1, cfg.lead_panels), cp = std::max<Integer>(1, cfg.corner_panels);
    const Integer tlp = std::max<Integer>(1, cfg.tip_lead_panels), tcp = std::max<Integer>(1, cfg.tip_corner_panels);
    e1L = (Real)lp / n; e1R = (Real)(lp + cp) / n;
    e2L = (Real)(n - tlp - tcp) / n; e2R = (Real)(n - tlp) / n;
    // Straight tip-lead ray: the spiral point P2 and its df-derivative Sp2 at e2L. Over [e2R,1] the
    // centerline runs straight along Sp2, so the slender's tip ring is EXACTLY perpendicular to Sp2 -- the
    // butterfly cap (built on that same direction) then conforms exactly (the tip analogue of the foot lead).
    const Vec3<Real> Pm = spiral(e2L - (Real)1e-4), Pp = spiral(e2L + (Real)1e-4), P0 = spiral(e2L);
    for (int k=0;k<3;k++) { P2[k] = P0[k]; Sp2[k] = (Pp[k]-Pm[k]) / (Real)2e-4; }
  }

  // The flagella spiral (== python/flagella.py::centerline), base anchored ON the surface at Cs*u, marched
  // inward with tilt/bend/(bias)/spiral. Reach ~ L, so the tip separation matches the script (no inward
  // overshoot -- the straight lead is the axial PROJECTION of this curve, not an extra inward segment).
  Vec3<Real> spiral(Real t) const {
    const Real tilt = cfg.max_tilt * t * t;
    Real inw[3]; for (int k=0;k<3;k++) inw[k] = -std::cos(tilt)*u[k] + std::sin(tilt)*u_fl[k]; nrm3(inw);
    Real c[3]; for (int k=0;k<3;k++) c[k] = Cs*u[k] + cfg.L*t*inw[k];
    const Real sm = smoother(t);
    for (int k=0;k<3;k++) c[k] += v_fl[k] * (cfg.bend_amt * sm * cfg.L);
    if (bias_amt > 0) for (int k=0;k<3;k++) c[k] += bias_dir[k] * (bias_amt * sm * cfg.L);
    // Spiral amplitude ramps in over the first half via SMOOTHERSTEP (not flagella.py's min(1,4t), whose
    // C1 kink at t=0.25 spikes the centerline curvature -> radius-of-curvature 0.006 << tube radius -> the
    // swept tube self-intersects there). The smooth ramp lifts min radius-of-curvature to ~0.06.
    const Real ang = cfg.turns * 2 * const_pi<Real>() * t, grow = smoother(2*t), sr = cfg.spiral_amp * grow;
    for (int k=0;k<3;k++) c[k] += u_fl[k]*(std::cos(ang)*sr) + v_fl[k]*(std::sin(ang)*sr);
    return Vec3<Real>{c[0], c[1], c[2]};
  }

  // Composite centerline in arc-fraction f in [0,1]:
  //   [0,e1L]   straight FOOT lead   -- spiral's axial projection onto u (Cs*u + <S-Cs*u,u> u): a straight
  //             line along u (zero curvature) leaving the surface EXACTLY perpendicular, so it C2-matches
  //             the fillet (a revolution about u) AND the slender foot ring is perpendicular to u.
  //   [e1L,e1R] smootherstep-POU corner  (foot lead -> spiral; C2 at both panel-aligned edges)
  //   [e1R,e2L] the untouched flagella SPIRAL (tip position / inter-finger separation = the script's)
  //   [e2L,e2R] smootherstep-POU corner  (spiral -> straight tip lead; C2)
  //   [e2R,1]   straight TIP lead   -- along Sp2 from P2, so the slender tip ring is EXACTLY perpendicular
  //             to Sp2 and the butterfly cap conforms exactly (the connection fix).
  Vec3<Real> point(Real f) const {
    const Vec3<Real> S = spiral(f);
    const Real a = (S[0]-Cs*u[0])*u[0] + (S[1]-Cs*u[1])*u[1] + (S[2]-Cs*u[2])*u[2];   // <S - Cs*u, u>
    const Real footx = Cs*u[0]+a*u[0], footy = Cs*u[1]+a*u[1], footz = Cs*u[2]+a*u[2]; // straight foot lead (axis u)
    const Real tipx = P2[0]+(f-e2L)*Sp2[0], tipy = P2[1]+(f-e2L)*Sp2[1], tipz = P2[2]+(f-e2L)*Sp2[2]; // straight tip lead (Sp2)
    if (f <= e1L) return Vec3<Real>{footx, footy, footz};
    if (f < e1R) { const Real w = flagella_pou_ramp((f-e1L)/(e1R-e1L));
      return Vec3<Real>{(1-w)*footx+w*S[0], (1-w)*footy+w*S[1], (1-w)*footz+w*S[2]}; }
    if (f <= e2L) return S;
    if (f < e2R) { const Real w = flagella_pou_ramp((f-e2L)/(e2R-e2L));
      return Vec3<Real>{(1-w)*S[0]+w*tipx, (1-w)*S[1]+w*tipy, (1-w)*S[2]+w*tipz}; }
    return Vec3<Real>{tipx, tipy, tipz};
  }

  // Unit travel tangent by central finite difference (the composite is C2 -> smooth FD).
  Vec3<Real> tangent(Real f) const {
    const Real h = (Real)1e-4;
    Real a = f - h, b = f + h;
    if (a < 0) { a = 0; b = 2*h; }
    if (b > 1) { b = 1; a = 1 - 2*h; }
    const Vec3<Real> pa = point(a), pb = point(b);
    Real T[3] = { pb[0]-pa[0], pb[1]-pa[1], pb[2]-pa[2] }; nrm3(T);
    return Vec3<Real>{T[0], T[1], T[2]};
  }

  // Cross-section radius: R_shaft at the foot (seam), holding r_py along the shaft, tapering R_shaft->r_py
  // over the first Ntaper panels when R_shaft > r_py (otherwise constant R_shaft).
  Real radius(Real f) const {
    if (cfg.R_shaft <= cfg.r_py) return cfg.R_shaft;
    const Real ft = (Real)std::max<Integer>(1, cfg.Ntaper) / std::max<Integer>(6, cfg.n_axial);
    if (f >= ft) return cfg.r_py;
    return cfg.R_shaft + (cfg.r_py - cfg.R_shaft) * (f / ft);
  }

  // Phase reference e1(f): u_fl projected perpendicular to the tangent (smooth, minimal twist since the
  // tangent stays mostly axial; falls back to v_fl if u_fl aligns with the tangent). Only sets the
  // azimuthal origin -- the tube surface is frame-independent (a full circle), so twist is harmless.
  Vec3<Real> e1_at(const Vec3<Real>& T) const {
    Real w[3] = { u_fl[0], u_fl[1], u_fl[2] };
    Real d = w[0]*T[0]+w[1]*T[1]+w[2]*T[2];
    Real e[3] = { w[0]-d*T[0], w[1]-d*T[1], w[2]-d*T[2] };
    if (std::sqrt(dot3(e,e)) < (Real)0.1) {
      Real w2[3] = { v_fl[0], v_fl[1], v_fl[2] }; d = w2[0]*T[0]+w2[1]*T[1]+w2[2]*T[2];
      for (int k=0;k<3;k++) e[k] = w2[k]-d*T[k];
    }
    nrm3(e); return Vec3<Real>{e[0], e[1], e[2]};
  }
  Vec3<Real> e1(Real f) const { return e1_at(tangent(f)); }

  // Tip frame for the butterfly cap. Ttip is EXACTLY the straight tip-lead direction Sp2 (not a finite-
  // difference tangent), so the cap's equator plane matches the slender's straight-tip-section ring plane
  // exactly -- a watertight cap<->slender seam. Ctip = tip point, (w1,w2) an orthonormal frame perp to Ttip.
  void tip_frame(Real Ctip[3], Real Ttip[3], Real w1[3], Real w2[3], Real& rho_tip) const {
    const Vec3<Real> C = point(1);
    Real T[3] = { Sp2[0], Sp2[1], Sp2[2] }; nrm3(T);
    const Vec3<Real> Tv{T[0],T[1],T[2]}, W1 = e1_at(Tv);
    for (int k=0;k<3;k++) { Ctip[k]=C[k]; Ttip[k]=T[k]; w1[k]=W1[k]; }
    cross3(Ttip, w1, w2); nrm3(w2);
    rho_tip = radius(1);
  }
};

// Approximate centerline arc length (for auto axial-panel sizing); insensitive to n_axial since the arc
// is dominated by the spiral shape, not the tiny lead/corner windows.
template <class Real> Real flagella_arclen(const FlagellaCfg<Real>& cfg, const Real axis[3], Integer nsamp = 400) {
  FlagellumCurve<Real> fc(cfg, axis);
  Real arc = 0; Vec3<Real> prev = fc.point((Real)0);
  for (Integer i = 1; i <= nsamp; i++) {
    const Vec3<Real> p = fc.point((Real)i / nsamp);
    Real d[3] = { p[0]-prev[0], p[1]-prev[1], p[2]-prev[2] };
    arc += std::sqrt(d[0]*d[0]+d[1]*d[1]+d[2]*d[2]); prev = p;
  }
  return arc;
}

// Minimum surface-to-surface clearance between finger centerlines (like flagella.py::min_separation):
// min pairwise centerline distance minus the two radii. curves sampled as polylines.
template <class Real> Real flagella_min_clearance(const std::vector<FlagellumCurve<Real>>& curves,
                                                  Long& pi_out, Long& pj_out, Integer nsamp = 120) {
  const Long Np = (Long)curves.size(); Real best = (Real)1e30; pi_out = pj_out = -1;
  std::vector<std::vector<Vec3<Real>>> P(Np);
  std::vector<std::vector<Real>> Rr(Np);
  for (Long i = 0; i < Np; i++) { P[i].resize(nsamp+1); Rr[i].resize(nsamp+1);
    for (Integer s = 0; s <= nsamp; s++) { const Real f = (Real)s/nsamp; P[i][s] = curves[i].point(f); Rr[i][s] = curves[i].radius(f); } }
  for (Long i = 0; i < Np; i++) for (Long j = i+1; j < Np; j++)
    for (Integer si = 0; si <= nsamp; si++) for (Integer sj = 0; sj <= nsamp; sj++) {
      Real d[3] = { P[i][si][0]-P[j][sj][0], P[i][si][1]-P[j][sj][1], P[i][si][2]-P[j][sj][2] };
      const Real gap = std::sqrt(d[0]*d[0]+d[1]*d[1]+d[2]*d[2]) - (Rr[i][si] + Rr[j][sj]);
      if (gap < best) { best = gap; pi_out = i; pj_out = j; }
    }
  return best;
}

} // namespace quad_junctions
