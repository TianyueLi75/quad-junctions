#pragma once
/**
 * HYBRID flat-plane cilia carpet: the QuadElemList BASE (two flat walls, each tiled with an
 * Npatch x Npatch grid of collar + fillet + butterfly-cap FEET, NO shaft) joined to a CSBQ
 * SlenderElemList that supplies one bent-tube SHAFT per cilium, all fed into ONE BoundaryIntegralOp.
 * The flat-carpet hybrid: QuadElemList collar/cap feet + CSBQ slender shafts (the `flagella`/`allfinger` pattern).
 *
 * Each cilium's slender shaft follows a CiliumCurveFlat centerline: a straight base, a smootherstep-POU
 * tilt into `tilt_rad` (in the +dir_x*x / z plane), then straight-tilted to a tip at height H_reach, with
 * an OPTIONAL random transverse SINE WIGGLE (one full period over the arc, windowed to zero value+slope at
 * BOTH the foot -- clean collar seam -- and the tip -- cap stays on the nominal axis). The QuadElemList
 * base foot (collar + fillet, NO cap, NO shaft) is built canonically and the butterfly cap is appended at
 * the (wiggled) canonical tip, then the whole stud is placed onto its plane and oriented -- exactly the
 * order add_cilium_stud_flat used, so the geometry is bit-identical when the wiggle is off.
 *
 * generate_cilia_carpet() produces the placed curve set with: (1) per-column TILT REDUCTION so no tip
 * leaves the unit box (0.01 buffer), (2) a random wiggle per cilium (amp < 0.4*collar-side, orientation in
 * [0,pi)), and (3) COLLISION RESOLUTION -- if a cilium's tube+cap overlaps another, regenerate its wiggle
 * (up to 3x), else shorten it (retract the tip toward its own wall). Deterministic given the seed (same on
 * every MPI rank => identical replicated geometry).
 *
 * NORMAL ORIENTATION: the CSBQ slender shaft normal is ALWAYS radially OUTWARD and cannot be flipped, so the
 * WHOLE base is oriented to match it -- out-of-solid / INTO the fluid (wall points into the gap, tube points
 * radially outward). A carpet cilium is a protruding tube (fluid OUTSIDE), so out-of-fluid would point
 * radially INWARD = opposite the slender; matching means adopting the out-of-solid convention on the base too
 * (BuildCiliaCarpetHybridBase flips the wall uz for this). The DL identity is then +1/2 and run_flow uses
 * NormalOrient=-1 (+1/2 jump). [Earlier "no inversion / native outward matches the walls" was WRONG: the base
 * ended up out-of-fluid, opposite the slender -> periodic DL spread ~1.0, wrong near-cilia flow. See
 * [[stud-sphere-hybrid]] for the same slender-cannot-be-flipped gotcha.]
 */

#include <quad_junctions/plane_cilia_geom.hpp>
#include <csbq/slender_element.hpp>
#include <csbq/slender_element.cpp>
#include <cstdlib>
#include <random>
#include <vector>

namespace quad_junctions {
using namespace sctl;

// One cilium's bent (and optionally wiggled) centerline / frame. The base centerline replicates
// add_cilium_stud_flat's math EXACTLY (same az/s0/s_t/s1/Sarc); the wiggle is an additive transverse
// displacement. Placement onto a plane: translate (bottom, reflect=false, z -> z_base+z) or z-reflection
// (top, reflect=true, z -> z_base-z). Arc s in [0,Sarc], s=0 = fillet-top / slender-foot ring, s=Sarc = tip.
template <class Real> struct CiliumCurveFlat {
  Real cx, cy, R_shaft, r_fil, tilt_rad, dir_x, H_reach;
  Real ct0, st0, s0, s_t, s1, Sarc;
  Real Cs0[3], Cs0t[3];
  Integer Ns;                 // pure-quad axial panel count (nstr+ntr+Ntil); slender uses its own n_axial
  Real z_base; bool reflect;  // placement onto a plane
  // random transverse sine wiggle (one period over the arc, windowed to zero at both ends)
  Real wig_amp = 0, wig_theta = 0;   // amplitude, orientation angle in [0,pi)
  Real wig_p1[3], wig_p2[3];         // transverse basis (canonical), perpendicular to the tilted axis
  Real wig_ramp = (Real)0.20;        // window ramp fraction at each end

  static Real smoother(Real x) { if (x<=(Real)0) return (Real)0; if (x>=(Real)1) return (Real)1;
    return x*x*x*(x*((Real)6*x-(Real)15)+(Real)10); }

  // ---- nominal (unwiggled) centerline pieces (== add_cilium_stud_flat) ----
  Real phi(Real s) const {
    if (s <= s0 || tilt_rad <= 0) return 0;
    if (s >= s0 + s_t) return tilt_rad;
    return tilt_rad * smoother((s - s0) / s_t);
  }
  void tang0_canon(Real s, Real T[3]) const { const Real p = phi(s); T[0] = dir_x*sin<Real>(p); T[1] = 0; T[2] = cos<Real>(p); }
  void integ(Real sa, Real sb, Real I[3]) const {   // fixed-N Simpson (deterministic => watertight seams)
    const Integer n = 1000; const Real h = (sb - sa) / n; Real acc[3] = {0,0,0}, T[3];
    for (Integer k = 0; k <= n; k++) { tang0_canon(sa + k*h, T); const Real w = (k==0||k==n) ? 1 : (k%2 ? 4 : 2);
      for (int c=0;c<3;c++) acc[c] += w*T[c]; }
    for (int c=0;c<3;c++) I[c] = acc[c]*h/3;
  }
  void C0_canon(Real s, Real C[3]) const {   // nominal centerline (no wiggle)
    if (tilt_rad <= 0 || s <= s0) { C[0]=cx; C[1]=cy; C[2]=r_fil+s; return; }
    if (s <= s0 + s_t) { Real I[3]; integ(s0, s, I); C[0]=Cs0[0]+I[0]; C[1]=cy; C[2]=Cs0[2]+I[2]; return; }
    const Real ds = s - (s0 + s_t); C[0]=Cs0t[0]+dir_x*st0*ds; C[1]=cy; C[2]=Cs0t[2]+ct0*ds;
  }
  void e1of_canon(Real s, Real e1[3]) const { const Real p = phi(s); e1[0]=cos<Real>(p); e1[1]=0; e1[2]=-dir_x*sin<Real>(p); }

  // ---- wiggle displacement (canonical): amp * window(f) * sin(2*pi*f) * w_hat ----
  Real window(Real f) const { return smoother(f/wig_ramp) * smoother((1-f)/wig_ramp); }   // 0 (value+slope) at both ends
  void wiggle_canon(Real s, Real D[3]) const {
    D[0]=D[1]=D[2]=0;
    if (wig_amp <= 0) return;
    const Real f = s / Sarc;
    const Real a = wig_amp * window(f) * sin<Real>(2*const_pi<Real>()*f);
    const Real c = cos<Real>(wig_theta), sn = sin<Real>(wig_theta);
    for (int k=0;k<3;k++) D[k] = a * (c*wig_p1[k] + sn*wig_p2[k]);
  }
  void C_canon(Real s, Real C[3]) const { C0_canon(s, C); Real D[3]; wiggle_canon(s, D); for (int k=0;k<3;k++) C[k]+=D[k]; }
  void tang_canon(Real s, Real T[3]) const {   // FD of the wiggled centerline
    const Real h = Sarc*(Real)1e-4; Real a=s-h,b=s+h; if(a<0){a=0;b=2*h;} if(b>Sarc){b=Sarc;a=Sarc-2*h;}
    Real Ca[3],Cb[3]; C_canon(a,Ca); C_canon(b,Cb); Real n=0;
    for (int k=0;k<3;k++){ T[k]=Cb[k]-Ca[k]; n+=T[k]*T[k]; } n=std::sqrt(n); if(n>0) for(int k=0;k<3;k++) T[k]/=n;
  }

  // ---- placement (canonical -> world) ----
  void to_world_pt(const Real c[3], Real w[3]) const { w[0]=c[0]; w[1]=c[1]; w[2]= reflect ? (z_base - c[2]) : (z_base + c[2]); }
  void to_world_vec(const Real v[3], Real w[3]) const { w[0]=v[0]; w[1]=v[1]; w[2]= reflect ? -v[2] : v[2]; }

  Vec3<Real> point(Real f) const { Real c[3], w[3]; C_canon(Sarc*f, c); to_world_pt(c, w); return Vec3<Real>{w[0],w[1],w[2]}; }
  Vec3<Real> e1(Real f) const { Real c[3], w[3]; e1of_canon(Sarc*f, c); to_world_vec(c, w); return Vec3<Real>{w[0],w[1],w[2]}; }

  // CANONICAL tip frame for the butterfly cap (appended before placement, so the whole stud is placed &
  // oriented together -- identical to add_cilium_stud_flat when wig_amp=0). Ttip is the (wiggled) tip
  // tangent; (w1,w2) an orthonormal frame perpendicular to it (Gram-Schmidt from e1of, then Ttip x w1).
  void tip_frame_canon(Real Ctip[3], Real Ttip[3], Real w1[3], Real w2[3]) const {
    C_canon(Sarc, Ctip); tang_canon(Sarc, Ttip);
    Real e[3]; e1of_canon(Sarc, e); const Real d = e[0]*Ttip[0]+e[1]*Ttip[1]+e[2]*Ttip[2];
    Real n=0; for(int k=0;k<3;k++){ w1[k]=e[k]-d*Ttip[k]; n+=w1[k]*w1[k]; } n=std::sqrt(n); for(int k=0;k<3;k++) w1[k]/=n;
    w2[0]=Ttip[1]*w1[2]-Ttip[2]*w1[1]; w2[1]=Ttip[2]*w1[0]-Ttip[0]*w1[2]; w2[2]=Ttip[0]*w1[1]-Ttip[1]*w1[0];
  }

  Integer auto_n_axial(Long fourier) const {
    const Real az = 2 * const_pi<Real>() * R_shaft / (Real)fourier;
    return std::max<Integer>(2, (Integer)std::llround((double)(Sarc / az)));
  }

  CiliumCurveFlat(Real cx_, Real cy_, Real R_shaft_, Real H_reach_, Real r_fil_, Integer Naz,
                  Real tilt_rad_, Real dir_x_, Integer n_straight, Integer n_trans,
                  Real z_base_, bool reflect_, Real wig_amp_ = 0, Real wig_theta_ = 0)
      : cx(cx_), cy(cy_), R_shaft(R_shaft_), r_fil(r_fil_), tilt_rad(tilt_rad_), dir_x(dir_x_), H_reach(H_reach_),
        z_base(z_base_), reflect(reflect_), wig_amp(wig_amp_), wig_theta(wig_theta_) {
    const Real pi = const_pi<Real>();
    const Real az = 2 * pi * R_shaft / Naz;                     // panel arc size (== add_cilium_stud_flat)
    ct0 = cos<Real>(tilt_rad); st0 = sin<Real>(tilt_rad);
    const Integer nstr = std::max<Integer>(0, n_straight), ntr = (tilt_rad > 0 ? std::max<Integer>(1, n_trans) : 0);
    s0 = nstr * az; s_t = ntr * az;
    Real Itr[3]; integ(s0, s0 + s_t, Itr);
    Cs0[0]=cx; Cs0[1]=cy; Cs0[2]=r_fil + s0;
    Cs0t[0]=Cs0[0]+Itr[0]; Cs0t[1]=cy; Cs0t[2]=Cs0[2]+Itr[2];
    s1 = (H_reach - R_shaft*ct0 - Cs0t[2]) / ct0;
    const Integer Ntil = std::max<Integer>(2, (Integer)std::llround((double)(s1 / az)));
    s1 = Ntil * az;
    Sarc = s0 + s_t + s1;
    Ns = nstr + ntr + Ntil;
    // transverse wiggle basis (canonical): p1 perpendicular to the tilted axis in the x-z plane, p2 = binormal.
    wig_p1[0]=ct0; wig_p1[1]=0; wig_p1[2]=-dir_x*st0;
    wig_p2[0]=0;   wig_p2[1]=1; wig_p2[2]=0;
  }
};

// Enumerate all cilia (both planes, Npatch x Npatch grid) as placed CiliumCurveFlat objects (NOMINAL: tilt,
// no wiggle). generate_cilia_carpet() below layers tilt-fit + wiggle + collision resolution on top.
template <class Real> std::vector<CiliumCurveFlat<Real>> cilia_carpet_curves(
    Integer Npatch, Real z_bottom, Real z_top, Real L, Real R_shaft, Real bot_tip, Real top_tip,
    Real r_fil, Integer Naz, Real tilt_rad, Integer n_straight, Integer n_trans) {
  std::vector<CiliumCurveFlat<Real>> curves; curves.reserve(2 * Npatch * Npatch);
  for (int plane = 0; plane < 2; plane++) {
    const Real z_base   = plane == 0 ? z_bottom : z_top;
    const bool reflect  = plane != 0;
    const Real H_reach  = plane == 0 ? (bot_tip - z_bottom) : (z_top - top_tip);
    const Real dir_x    = plane == 0 ? (Real)+1 : (Real)-1;   // bottom tilts +x (with flow), top -x
    for (Integer iu = 0; iu < Npatch; iu++)
      for (Integer iv = 0; iv < Npatch; iv++) {
        const Real cx = (iu + (Real)0.5) * L / Npatch, cy = (iv + (Real)0.5) * L / Npatch;
        curves.emplace_back(cx, cy, R_shaft, H_reach, r_fil, Naz, tilt_rad, dir_x, n_straight, n_trans, z_base, reflect);
      }
  }
  return curves;
}

// World sample points of a cilium: the centerline polyline (tip included at f=1). The box/clearance checks
// use these +- R_shaft; the hemispherical cap lies within the ball of radius R_shaft about the tip, so the
// tip point already bounds it. (An earlier version added a separate cap-apex point tip + R_shaft*tangent,
// which the +-R_shaft margin then DOUBLE-COUNTED -- spuriously flagging +x-tilted edge cilia as leaving the
// box, inconsistent with the tilt-fit which uses tip +- R_shaft.)
template <class Real> void cilium_samples(const CiliumCurveFlat<Real>& c, std::vector<Vec3<Real>>& pts, Integer nsamp = 48) {
  pts.clear(); pts.reserve(nsamp + 1);
  for (Integer i = 0; i <= nsamp; i++) pts.push_back(c.point((Real)i / nsamp));
}

// Surface clearance between two cilia (min sampled distance - 2*R_shaft). <0 => the tubes/caps overlap.
template <class Real> Real cilium_clearance(const std::vector<Vec3<Real>>& A, const std::vector<Vec3<Real>>& B, Real R_shaft) {
  Real best = (Real)1e30;
  for (const auto& a : A) for (const auto& b : B) {
    const Real dx=a[0]-b[0], dy=a[1]-b[1], dz=a[2]-b[2]; const Real d = std::sqrt(dx*dx+dy*dy+dz*dz);
    if (d < best) best = d;
  }
  return best - 2*R_shaft;
}

// Tube+cap Z-EXTENT of a cilium (world). Unlike the x/y box check (which uses centerline +- R_shaft, a
// ball), the z-extent of the swept TUBE at a station is centerline_z +- R_shaft*|T_horizontal| (0 for a
// vertical tube, R_shaft for a horizontal one) -- so a fat TILTED tube's surface dips below its centerline.
// The hemispherical TIP CAP adds tip_z +- R_shaft. Needed because the periodic FMM requires the whole
// geometry's z-extent < L (else `periodic_wrap_max_k` asserts): a thick tilted cilium near a wall can push
// its surface outside [0,L] even though its centerline stays inside.
template <class Real> void cilium_z_bounds(const CiliumCurveFlat<Real>& c, Real R_shaft, Real& zmin, Real& zmax, Integer nsamp = 48) {
  zmin = (Real)1e30; zmax = (Real)-1e30;
  const Real df = (Real)1e-3;
  for (Integer i = 0; i <= nsamp; i++) {
    const Real f = (Real)i / nsamp;
    const auto p = c.point(f);
    const auto pa = c.point(std::max<Real>(0, f - df)), pb = c.point(std::min<Real>(1, f + df));  // world tangent via FD
    const Real T[3] = {pb[0]-pa[0], pb[1]-pa[1], pb[2]-pa[2]};
    const Real Tn = std::sqrt(T[0]*T[0]+T[1]*T[1]+T[2]*T[2]) + (Real)1e-30;
    const Real h = std::sqrt(T[0]*T[0]+T[1]*T[1]) / Tn;   // horizontal fraction of the unit tangent
    const Real zdip = R_shaft * h;                        // tube cross-section z half-extent at this station
    zmin = std::min(zmin, p[2] - zdip); zmax = std::max(zmax, p[2] + zdip);
  }
  const auto tip = c.point(1);                            // tip cap hemisphere (radius R_shaft about the tip)
  zmin = std::min(zmin, tip[2] - R_shaft); zmax = std::max(zmax, tip[2] + R_shaft);
}

// Nominal tip world-x for a candidate tilt (no wiggle: the windowed sine is 0 at the tip, so the wiggled
// tip == the nominal tip; box-fit can use the nominal tip).
template <class Real> Real tip_world_x(Real cx, Real cy, Real R_shaft, Real H_reach, Real r_fil, Integer Naz,
                                       Real tilt, Real dir_x, Integer n_straight, Integer n_trans, Real z_base, bool reflect) {
  CiliumCurveFlat<Real> c(cx, cy, R_shaft, H_reach, r_fil, Naz, tilt, dir_x, n_straight, n_trans, z_base, reflect);
  return c.point(1)[0];
}

// The FULL randomized carpet: tilt-reduction for box containment, random sine wiggle, collision resolution.
// Deterministic given `seed` (identical on every MPI rank). Prints a report on rank 0.
template <class Real> std::vector<CiliumCurveFlat<Real>> generate_cilia_carpet(
    Integer Npatch, Real z_bottom, Real z_top, Real L, Real R_shaft, Real bot_tip, Real top_tip,
    Real r_fil, Integer Naz, Real tilt_rad, Integer n_straight, Integer n_trans,
    uint64_t seed, const Comm& comm, Real box_buffer = (Real)0.01, Real coll_buffer = (Real)0.005,
    bool wiggle = true) {
  const Real S = L / (Real)(2 * Npatch);
  const Real pi = const_pi<Real>();
  // Wiggle amplitude (transverse, windowed to 0 at foot+tip). Bounded by FOUR things:
  //  (1) PATCH pitch: wig_frac*2S (env QJ_WIG_AMP_FRAC default 0.4).
  //  (2)+(3) RADIUS+TILT: the free lane between adjacent tubes (2S apart, radius R_shaft each => surface gap
  //      2S-2*R_shaft, minus the tilt's lateral swing R_shaft*sin(tilt)); cap at HALF the lane.
  //  (4) CURVATURE vs shaft radius: the wiggle is ONE sine period over the arc (length ~ Sarc), so its peak
  //      curvature is kappa = amp*(2*pi/Sarc)^2. The swept tube of radius R_shaft only stays
  //      self-intersection-free with FULLY RESOLVED inner-bend surface panels if the centerline radius of
  //      curvature 1/kappa >= k_curv*R_shaft, i.e. amp <= Sarc^2 / (4*pi^2 * k_curv * R_shaft). Use the
  //      shorter reach as a conservative Sarc proxy (arc >= vertical reach); k_curv default 2 (env QJ_WIG_KCURV).
  const char* wa_env = std::getenv("QJ_WIG_AMP_FRAC");
  const Real wig_frac = wa_env ? (Real)std::atof(wa_env) : (Real)0.4;
  const Real lane = std::max((Real)0, (Real)0.5*(2*S - 2*R_shaft) - R_shaft*std::sin(tilt_rad));
  const Real Sarc_lo = std::min(bot_tip - z_bottom, z_top - top_tip);   // conservative arc-length proxy
  const Real k_curv = std::getenv("QJ_WIG_KCURV") ? (Real)std::atof(std::getenv("QJ_WIG_KCURV")) : (Real)2;
  const Real curv_amp = (R_shaft > 0) ? Sarc_lo*Sarc_lo / (4*pi*pi*k_curv*R_shaft) : (Real)1e30;
  // wiggle knob (default on): OFF forces amp==0 for every cilium, so each curve reduces to the nominal
  // tilted add_cilium_stud_flat centerline (bit-identical wiggle-off). Tilt-reduction / box-containment /
  // cap-overlap safeguards below still run -- only the random transverse sine is removed.
  const Real max_amp = wiggle ? std::min(std::min(wig_frac * (2 * S), lane), curv_amp) : (Real)0;
  const Real edge_amp = (Real)0.2 * (L / (Real)Npatch);  // Phase-C box amp-cut STEP size (0.1*edge_amp per step)

  // 1. nominal curves (tilt, no wiggle)
  std::vector<CiliumCurveFlat<Real>> C = cilia_carpet_curves<Real>(
      Npatch, z_bottom, z_top, L, R_shaft, bot_tip, top_tip, r_fil, Naz, tilt_rad, n_straight, n_trans);
  const Long N = (Long)C.size();

  // per-cilium mutable spec (rebuild the curve when tilt / H_reach / wiggle change)
  std::vector<Real> tilt(N), Hreach(N), amp(N, (Real)0), theta(N, (Real)0);
  for (Long i = 0; i < N; i++) { tilt[i] = C[i].tilt_rad; Hreach[i] = C[i].H_reach; }
  auto rebuild = [&](Long i) {
    const auto& c = C[i];
    C[i] = CiliumCurveFlat<Real>(c.cx, c.cy, R_shaft, Hreach[i], r_fil, Naz, tilt[i], c.dir_x,
                                 n_straight, n_trans, c.z_base, c.reflect, amp[i], theta[i]);
  };

  // 2. TILT REDUCTION: bisect each cilium's tilt down until its (nominal) tip + cap radius fits [buffer, L-buffer].
  Long n_tilt_cut = 0;
  for (Long i = 0; i < N; i++) {
    const auto& c = C[i];
    auto fits = [&](Real t) { const Real x = tip_world_x<Real>(c.cx, c.cy, R_shaft, Hreach[i], r_fil, Naz, t, c.dir_x, n_straight, n_trans, c.z_base, c.reflect);
      return (x - R_shaft) >= box_buffer && (x + R_shaft) <= (L - box_buffer); };
    if (fits(tilt[i])) continue;
    Real lo = 0, hi = tilt[i];                 // tilt=0 always fits (tip x = cx, interior)
    for (int it = 0; it < 40; it++) { const Real mid = (Real)0.5*(lo+hi); if (fits(mid)) lo = mid; else hi = mid; }
    tilt[i] = lo; rebuild(i); n_tilt_cut++;
  }

  // 3. random wiggle per cilium (amp in [amp_min,max_amp), orientation in [0,pi)); deterministic RNG.
  //    amp_min is a FLOOR (0.5*max_amp) so every cilium visibly wiggles -- a plain [0,max) draw lets some
  //    cilia land near zero and render dead straight ("both straight and new shaft").
  const Real amp_min = (Real)0.5 * max_amp;
  std::mt19937_64 gen(seed);
  std::uniform_real_distribution<double> amp_d((double)amp_min, (double)max_amp), th_d(0.0, (double)const_pi<Real>());
  auto draw = [&](Long i) { amp[i] = (Real)amp_d(gen); theta[i] = (Real)th_d(gen); rebuild(i); };
  for (Long i = 0; i < N; i++) draw(i);

  // 4. COLLISION + BOX-CONTAINMENT RESOLUTION (greedy, single pass). A cilium is "bad" if its wiggled
  //    tube+cap overlaps another OR leaves the unit box (both x AND y, tube radius included). Fix: regenerate
  //    the wiggle up to 3x (a new orientation usually points the bulge inward, fixing the box without losing
  //    the wiggle); still bad => last resort: shrink the wiggle amplitude (box overflow), then reduce tilt
  //    (box+collision), then shorten (collision). Containment wins over the amp floor, so amp may drop below amp_min.
  auto samples_of = [&](Long i, std::vector<Vec3<Real>>& s) { cilium_samples<Real>(C[i], s); };
  std::vector<std::vector<Vec3<Real>>> smp(N);
  for (Long i = 0; i < N; i++) samples_of(i, smp[i]);
  auto min_clear = [&](Long i) { Real best = (Real)1e30;
    for (Long j = 0; j < N; j++) if (j != i) best = std::min(best, cilium_clearance<Real>(smp[i], smp[j], R_shaft));
    return best; };
  // Whole tube+cap (centerline +- R_shaft, in BOTH x and y) must stay within [box_buffer, L-box_buffer], AND
  // the tube+cap Z-surface must stay within [0, L] (periodic-FMM needs the geometry z-extent < L -- a thick
  // TILTED cilium can otherwise dip below the wall, see cilium_z_bounds).
  auto exceeds_box = [&](Long i) {
    Real zmin, zmax; cilium_z_bounds<Real>(C[i], R_shaft, zmin, zmax);
    if (zmin < (Real)0 || zmax > L) return true;
    for (const auto& p : smp[i])
      if (p[0]-R_shaft < box_buffer || p[0]+R_shaft > L-box_buffer ||
          p[1]-R_shaft < box_buffer || p[1]+R_shaft > L-box_buffer) return true;
    return false;
  };
  auto bad = [&](Long i) { return (min_clear(i) < coll_buffer) || exceeds_box(i); };

  // EDGE cilia keep their (tilt-fit) lighter tilt and still wiggle, wall-parallel (pure-y) so x is unaffected,
  // at the LARGEST amplitude the y-box admits (capped at the interior max_amp) -- see Phase B. The wiggle
  // excurses +-amp in y (full sine period), so the box limit is min(cy,L-cy) - box_buffer - R_shaft.
  Long n_regen = 0, n_edge = 0, n_ampcut = 0;
  for (Long i = 0; i < N; i++) {
    if (!bad(i)) continue;
    // Phase A: regenerate full-amplitude wiggle (new orientation often fixes box + collision), up to 3x.
    int tries = 0;
    while (tries < 3 && bad(i)) { draw(i); samples_of(i, smp[i]); tries++; n_regen++; }
    if (!bad(i)) continue;
    // Phase B: if the cilium overflows the box, give it the EDGE treatment -- keep tilt, drop to the small
    //   edge amplitude, and search for a BOX-fitting orientation (wall-parallel). Decoupled from collision:
    //   this keeps a visible wiggle on edge cilia instead of cutting it to zero; a remaining COLLISION is
    //   then resolved by shortening below (which preserves the wiggle), not by killing the amplitude.
    if (exceeds_box(i)) {
      // keep the current (tilt-fit) tilt; wall-parallel wiggle (theta=pi/2 => pure-y, wig_p2=(0,1,0), adds
      // nothing in the tilt/x direction, so x still fits). Size the amplitude as BIG as the y-box allows: the
      // nominal centerline sits at y=cy and the pure-y wiggle excurses +-amp, so amp <= min(cy,L-cy) - box_buffer
      // - R_shaft keeps it inside; cap at the interior max_amp so it matches (and is not more collision-prone
      // than) the interior wiggle. Makes the edge wiggle clearly visible, vs the old tiny fixed 0.5*max_amp.
      const Real yroom = std::min(C[i].cy, L - C[i].cy) - box_buffer - R_shaft;   // max |y-excursion| that fits
      amp[i] = std::max((Real)0, std::min(max_amp, yroom)); theta[i] = const_pi<Real>() / 2;
      rebuild(i); samples_of(i, smp[i]);
      n_edge++;
    }
    // Phase C: shrink the wiggle amplitude to fit the box (box overflow only). NO shortening -- every cilium
    //   must keep its full reach to z=0.5 regardless of radius; residual TIP/CAP collisions are handled by
    //   the tilt-adjust in step 5, not by retracting the tip.
    bool ampcut = false;
    for (int guard = 0; exceeds_box(i) && amp[i] > (Real)1e-4 && guard < 120; guard++) {
      amp[i] = std::max((Real)0, amp[i] - (Real)0.1*edge_amp); ampcut = true;
      rebuild(i); samples_of(i, smp[i]);
    }
    if (ampcut) n_ampcut++;
  }

  // 5. CAP-OVERLAP SAFEGUARD. The steps above resolve on the centerline; the hemispherical TIP CAPS (radius
  //    R_shaft about each tip) can still overlap when two tips are within 2*R_shaft -- e.g. same-column
  //    top/bottom cilia both reaching the midplane. Resolve by tilting the offending pair 5 deg further APART
  //    (each in its own dir_x, which are OPPOSITE for such a pair, so the tips move in opposite x directions)
  //    until the caps clear -- but never accept a shift that pushes a cilium out of the box (x,y OR z).
  auto caps_overlap = [&](Long i, Long j) {
    const auto a = C[i].point(1), b = C[j].point(1);
    const Real dx=a[0]-b[0], dy=a[1]-b[1], dz=a[2]-b[2];
    return std::sqrt(dx*dx+dy*dy+dz*dz) < 2*R_shaft + coll_buffer;
  };
  const Real dtilt = (Real)5 * const_pi<Real>() / 180;   // 5 degree shift per sweep
  Long n_capshift = 0;
  for (int sweep = 0; sweep < 12; sweep++) {
    bool any = false;
    for (Long i = 0; i < N; i++) for (Long j = i+1; j < N; j++) {
      if (!caps_overlap(i, j)) continue;
      any = true;
      for (Long k : {i, j}) {   // shift each 5 deg further in ITS OWN dir_x (opposite for a top/bottom pair);
        const Real saved = tilt[k];                 // this keeps the tip at z=0.5 (H_reach unchanged) and moves it
        tilt[k] += dtilt; rebuild(k); samples_of(k, smp[k]);   // laterally, separating the two caps
        if (exceeds_box(k)) { tilt[k] = saved; rebuild(k); samples_of(k, smp[k]); }   // never leave the box (x,y,z)
        else n_capshift++;
      }
    }
    if (!any) break;
  }

  // report: min clearance + max box overflow (both should be within tolerance / <= 0)
  Real gmin = (Real)1e30; Long gi = -1, gj = -1;
  for (Long i = 0; i < N; i++) for (Long j = i+1; j < N; j++) { const Real g = cilium_clearance<Real>(smp[i], smp[j], R_shaft);
    if (g < gmin) { gmin = g; gi = i; gj = j; } }
  Real box_over = -(Real)1e30;   // max amount any tube surface point exceeds [box_buffer, L-box_buffer] (x or y); <=0 => contained
  for (Long i = 0; i < N; i++) for (const auto& p : smp[i]) {
    box_over = std::max(box_over, box_buffer - (p[0]-R_shaft)); box_over = std::max(box_over, (p[0]+R_shaft) - (L-box_buffer));
    box_over = std::max(box_over, box_buffer - (p[1]-R_shaft)); box_over = std::max(box_over, (p[1]+R_shaft) - (L-box_buffer));
  }
  Real z_over = -(Real)1e30, zlo_g = (Real)1e30, zhi_g = -(Real)1e30;  // tube+cap z outside [0,L]; <=0 => periodic-FMM-safe (dz<L)
  for (Long i = 0; i < N; i++) { Real zmn, zmx; cilium_z_bounds<Real>(C[i], R_shaft, zmn, zmx);
    z_over = std::max(z_over, std::max((Real)0 - zmn, zmx - L)); zlo_g = std::min(zlo_g, zmn); zhi_g = std::max(zhi_g, zmx); }
  if (!comm.Rank())
    std::cout << "  generate_cilia_carpet: " << N << " cilia  seed=" << seed
              << "  wiggle=" << (wiggle ? "ON" : "OFF") << "  amp in [" << amp_min
              << ", " << max_amp << "] (bounds: patch " << wig_frac*(2*S) << ", lane " << lane
              << ", curvature " << curv_amp << " [R_curv>=" << k_curv << "*R_shaft])\n"
              << "    tilt-reduced (box-fit): " << n_tilt_cut << "   wiggle regenerations: " << n_regen
              << "   edge cilia (wall-|| wiggle, box-max<=" << max_amp << ", kept tilt): " << n_edge
              << "   box amp-cut: " << n_ampcut << "   cap-shifts(5deg): " << n_capshift << "   (cilia keep full reach to z=0.5)\n"
              << "    final min pairwise clearance = " << gmin << " (pair " << gi << "," << gj << ")  [>0 => no overlap]\n"
              << "    final box overflow (x&y, incl. R_shaft) = " << box_over << "  [<=0 => contained within " << box_buffer << " buffer]\n"
              << "    tube+cap z-extent = [" << zlo_g << "," << zhi_g << "] (need in [0," << L << "]); z overflow = " << z_over << "  [<=0 => periodic-FMM-safe]\n";
  return C;
}

// ============================================================================
// QuadElemList base: every cilium = collar + fillet + butterfly cap (NO shaft). The foot (collar+fillet) is
// built canonically (add_cilium_stud_flat with_cap=false, with_shaft=false), the butterfly cap is appended
// at the (possibly wiggled) CANONICAL tip, then the whole stud is placed onto its plane and oriented so ALL
// normals point OUT-OF-SOLID / INTO the fluid (wall points into the gap; tube points radially OUTWARD).
// This MATCHES the CSBQ SlenderElemList shaft, whose normal is ALWAYS radially outward and CANNOT be flipped
// (see [[stud-sphere-hybrid]]): a protruding tube has fluid OUTSIDE, so out-of-fluid points radially INWARD
// -- opposite the slender -- hence the whole base must adopt the slender's out-of-solid convention, NOT the
// wall's usual out-of-fluid one. The DL identity is then +1/2 and run_flow uses NormalOrient=-1 (+1/2 jump).
// (Earlier this oriented the base out-of-fluid, opposite the slender at the collar<->shaft seam -> periodic
//  DL spread ~1.0 instead of ~1e-6, wrong near-cilia flow insensitive to order/tol.)
// ============================================================================
template <class Real> QuadElemList<Real> BuildCiliaCarpetHybridBase(
    Integer order, const std::vector<CiliumCurveFlat<Real>>& curves, Real R_shaft, Real r_fil, Integer Naz,
    Real S, Real core_frac, Real grade_exp, Integer n_straight, Integer n_trans, const Comm& comm) {
  const Real R0 = R_shaft + r_fil;
  SCTL_ASSERT(R0 < S);
  Vector<Real> Xall;
  Long nflip = 0;
  for (const auto& c : curves) {
    const Real z_plane = c.z_base;
    // uz = wall-normal direction that makes the whole stud point OUT-OF-SOLID / into the fluid (to match the
    // slender shaft): bottom plane (reflect=false) -> +z (up into the gap); top plane (reflect=true) -> -z.
    // (This is the OPPOSITE sign of the out-of-fluid wall convention -- see the header note above.)
    const Real uz = c.reflect ? (Real)-1 : (Real)+1;
    Vector<Real> Xp;   // canonical foot: collar + fillet (no cap, no shaft)
    add_cilium_stud_flat<Real>(Xp, order, c.cx, c.cy, R_shaft, c.H_reach, r_fil, S, Naz, /*Nc*/-1, /*Ns*/-1, /*Nf*/-1,
                               grade_exp, core_frac, /*with_cap=*/false, c.tilt_rad, c.dir_x, n_straight, n_trans,
                               /*with_shaft=*/false);
    // butterfly cap at the (wiggled) canonical tip, appended BEFORE placement (whole stud placed together)
    Real Ctip[3], Ttip[3], w1[3], w2[3]; c.tip_frame_canon(Ctip, Ttip, w1, w2);
    add_tip_cap_butterfly<Real>(Xp, order, Ctip, Ttip, w1, w2, R_shaft, Naz, core_frac);
    // place: bottom translate (z += z_base), top z-reflect (z -> z_base - z)
    for (Long i = 0; i < Xp.Dim() / 3; i++) Xp[i*3+2] = c.reflect ? (c.z_base - Xp[i*3+2]) : (c.z_base + Xp[i*3+2]);
    if (orient_group_flat<Real>(Xp, order, z_plane, uz)) nflip++;
    for (Long i = 0; i < Xp.Dim(); i++) Xall.PushBack(Xp[i]);
  }
  if (!comm.Rank())
    std::cout << "  cilia-carpet HYBRID base: " << curves.size() << " feet (collar+fillet+cap, no shaft)"
              << " Naz=" << Naz << "; " << nflip << " groups flipped outward; nodes=" << (Xall.Dim()/3) << "\n";
  return QuadElemList<Real>(order, Xall, comm);
}

// ============================================================================
// SlenderElemList shafts: one bent (wiggled) fiber per cilium along its CiliumCurveFlat centerline, radius
// R_shaft, n_axial panels (uniform in arc), foot (f=0) -> tip (f=1). MPI-partitioned by global panel index.
// ============================================================================
template <class Real> SlenderElemList<Real> BuildCiliaCarpetHybridShafts(
    const std::vector<CiliumCurveFlat<Real>>& curves, Integer n_axial, Real R_shaft,
    Long cheb_order, Long fourier_order, const Comm& comm) {
  const Long Np = (Long)curves.size();
  const Long Nelem = Np * n_axial, Npr = comm.Size(), pid = comm.Rank();
  const Long k0g = (Nelem * pid) / Npr, k1g = (Nelem * (pid + 1)) / Npr;
  Vector<Long> elem_order, forder;
  Vector<Real> coord, radius, orient;
  Long eg = 0;
  for (Long p = 0; p < Np; p++) {
    const CiliumCurveFlat<Real>& fc = curves[p];
    for (Integer k = 0; k < n_axial; k++, eg++) {
      if (eg < k0g || eg >= k1g) continue;
      elem_order.PushBack(cheb_order); forder.PushBack(fourier_order);
      const Vector<Real>& cn = SlenderElemList<Real>::CenterlineNodes(cheb_order);
      for (Long j = 0; j < cheb_order; j++) {
        const Real f = ((Real)k + cn[j]) / n_axial;             // 0 (foot) .. 1 (tip), uniform in arc
        const Vec3<Real> P = fc.point(f), e = fc.e1(f);
        coord.PushBack(P[0]); coord.PushBack(P[1]); coord.PushBack(P[2]);
        radius.PushBack(R_shaft);
        orient.PushBack(e[0]); orient.PushBack(e[1]); orient.PushBack(e[2]);
      }
    }
  }
  return SlenderElemList<Real>(elem_order, forder, coord, radius, orient);
}

// Minimum surface-to-surface clearance across all cilia (tube + cap), for reporting. <=0 => overlap.
template <class Real> Real cilia_min_clearance(const std::vector<CiliumCurveFlat<Real>>& curves,
                                               Long& pi_out, Long& pj_out, Integer nsamp = 48) {
  const Long Np = (Long)curves.size(); Real best = (Real)1e30; pi_out = pj_out = -1;
  std::vector<std::vector<Vec3<Real>>> smp(Np);
  for (Long i = 0; i < Np; i++) cilium_samples<Real>(curves[i], smp[i], nsamp);
  for (Long i = 0; i < Np; i++) for (Long j = i+1; j < Np; j++) { const Real g = cilium_clearance<Real>(smp[i], smp[j], curves[0].R_shaft);
    if (g < best) { best = g; pi_out = i; pj_out = j; } }
  return best;
}

} // namespace quad_junctions
