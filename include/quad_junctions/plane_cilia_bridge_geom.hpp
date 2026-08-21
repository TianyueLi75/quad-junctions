#pragma once
/**
 * WALL-TO-WALL cilia "bridge" carpet: a variant of the flat cilia carpet (plane_cilia_hybrid_geom.hpp)
 * in which each of the Npatch x Npatch cells hosts a SINGLE cilium that BRIDGES the bottom plate to the
 * top plate -- collar + fillet in the bottom wall, one slender shaft, collar + fillet in the top wall.
 * There are NO butterfly caps and NO free tips (both shaft ends seam into a fillet-top ring), and NO tilt.
 *
 * Each shaft carries a random single-period SINE WIGGLE (transverse displacement windowed to zero VALUE
 * and SLOPE at both ends, so the end rings stay horizontal and conform to the fillet openings -- clean,
 * watertight collar<->shaft seams at BOTH walls). The wiggle amplitude is bounded so the whole tube
 * surface stays inside its own patch cell (half-width S), hence neighbouring cilia can never collide by
 * construction. The collision resolver from the carpet is kept purely as a VERIFIER: it prints a loud
 * `COLLISION TRIGGERED` line if any pair ever overlaps (and then regenerates the wiggle as a fallback),
 * which the confined-wiggle configuration should never do.
 *
 * NORMAL ORIENTATION: as in the carpet hybrid, the CSBQ slender shaft normal is radially OUTWARD and
 * cannot be flipped, so BOTH feet are oriented OUT-OF-SOLID / into the fluid to match it (bottom wall
 * points +z into the gap, top wall points -z into the gap). See plane_cilia_hybrid_geom.hpp:22-29.
 */

#include <quad_junctions/plane_cilia_geom.hpp>
#include <csbq/slender_element.hpp>
#include <csbq/slender_element.cpp>
#include <cstdlib>
#include <random>
#include <vector>

namespace quad_junctions {
using namespace sctl;

// One wall-to-wall cilium centerline in WORLD coordinates: a vertical line from the bottom fillet-top ring
// (z0 = z_bottom + r_fil) to the top fillet-top ring (z1 = z_top - r_fil) at (cx,cy), plus an additive
// transverse SINE WIGGLE (one period over the span, windowed to zero value+slope at both ends). f in [0,1]
// runs bottom (f=0) -> top (f=1).
template <class Real> struct BridgeCurve {
  Real cx, cy, R_shaft;
  Real z0, z1, span;
  Real wig_amp = 0, wig_theta = 0;          // amplitude, in-plane orientation angle in [0,pi)
  Real wig_ramp = (Real)0.20;               // window ramp fraction at each end

  static Real smoother(Real x) { if (x<=(Real)0) return (Real)0; if (x>=(Real)1) return (Real)1;
    return x*x*x*(x*((Real)6*x-(Real)15)+(Real)10); }
  Real window(Real f) const { return smoother(f/wig_ramp) * smoother((1-f)/wig_ramp); }  // 0 (value+slope) at both ends

  BridgeCurve(Real cx_, Real cy_, Real R_shaft_, Real z0_, Real z1_, Real wig_amp_ = 0, Real wig_theta_ = 0)
      : cx(cx_), cy(cy_), R_shaft(R_shaft_), z0(z0_), z1(z1_), span(z1_ - z0_),
        wig_amp(wig_amp_), wig_theta(wig_theta_) {}

  Vec3<Real> point(Real f) const {
    const Real disp = wig_amp * window(f) * sin<Real>(2*const_pi<Real>()*f);
    const Real c = cos<Real>(wig_theta), sn = sin<Real>(wig_theta);
    return Vec3<Real>{cx + disp*c, cy + disp*sn, z0 + f*span};
  }
  Vec3<Real> tang(Real f) const {   // FD of the wiggled centerline (unit)
    const Real h = (Real)1e-4; Real a=f-h, b=f+h; if(a<0){a=0;b=2*h;} if(b>1){b=1;a=1-2*h;}
    const Vec3<Real> Ca = point(a), Cb = point(b);
    Real T[3] = {Cb[0]-Ca[0], Cb[1]-Ca[1], Cb[2]-Ca[2]}, n=0;
    for (int k=0;k<3;k++) n += T[k]*T[k]; n = std::sqrt(n); if (n>0) for(int k=0;k<3;k++) T[k]/=n;
    return Vec3<Real>{T[0],T[1],T[2]};
  }
  // A frame vector perpendicular to the tangent (Gram-Schmidt of x_hat against T; CSBQ re-orthogonalizes).
  Vec3<Real> e1(Real f) const {
    const Vec3<Real> T = tang(f);
    Real e[3] = {1,0,0}; const Real d = e[0]*T[0]+e[1]*T[1]+e[2]*T[2];
    Real n=0; for(int k=0;k<3;k++){ e[k]-=d*T[k]; n+=e[k]*e[k]; } n=std::sqrt(n);
    if (n < (Real)1e-12) { e[0]=0; e[1]=1; e[2]=0; }   // T ~ x_hat: fall back to y_hat
    else for(int k=0;k<3;k++) e[k]/=n;
    return Vec3<Real>{e[0],e[1],e[2]};
  }
  Integer auto_n_axial(Long fourier) const {
    const Real az = 2 * const_pi<Real>() * R_shaft / (Real)fourier;
    return std::max<Integer>(2, (Integer)std::llround((double)(span / az)));
  }
};

// World sample points along a bridge cilium (centerline polyline).
template <class Real> void bridge_samples(const BridgeCurve<Real>& c, std::vector<Vec3<Real>>& pts, Integer nsamp = 64) {
  pts.clear(); pts.reserve(nsamp + 1);
  for (Integer i = 0; i <= nsamp; i++) pts.push_back(c.point((Real)i / nsamp));
}
// Surface clearance between two cilia (min sampled distance - 2*R_shaft). <0 => tubes overlap.
template <class Real> Real bridge_clearance(const std::vector<Vec3<Real>>& A, const std::vector<Vec3<Real>>& B, Real R_shaft) {
  Real best = (Real)1e30;
  for (const auto& a : A) for (const auto& b : B) {
    const Real dx=a[0]-b[0], dy=a[1]-b[1], dz=a[2]-b[2]; const Real d = std::sqrt(dx*dx+dy*dy+dz*dz);
    if (d < best) best = d;
  }
  return best - 2*R_shaft;
}

// Build the Npatch x Npatch wall-to-wall cilia: one BridgeCurve per cell with a confined random wiggle.
// The wiggle amplitude is capped so the whole tube surface stays inside the cell (half-width S), so the
// collision verifier below should always report 0 overlaps. Deterministic given `seed` (identical on every
// MPI rank). Prints a report on rank 0; prints `COLLISION TRIGGERED` (and regenerates) if any pair overlaps.
template <class Real> std::vector<BridgeCurve<Real>> generate_cilia_bridge(
    Integer Npatch, Real z_bottom, Real z_top, Real L, Real R_shaft, Real r_fil,
    uint64_t seed, const Comm& comm, Real box_buffer = (Real)0.02, Real coll_buffer = (Real)0.005,
    bool wiggle = true) {
  const Real pi = const_pi<Real>();
  const Real S = L / (Real)(2 * Npatch);              // patch half-width (cell = 2S pitch)
  const Real z0 = z_bottom + r_fil, z1 = z_top - r_fil, span = z1 - z0;
  // Confined-to-cell amplitude bound. Keeping the tube surface (centerline +- R_shaft) inside the cell
  // half-width S requires amp <= S - R_shaft - box_buffer. Also honour the carpet's patch-fraction and
  // self-intersection-curvature bounds (one sine period over the span => peak curvature amp*(2pi/span)^2,
  // radius of curvature >= k_curv*R_shaft => amp <= span^2/(4*pi^2*k_curv*R_shaft)).
  const char* wa_env = std::getenv("QJ_WIG_AMP_FRAC");
  const Real wig_frac = wa_env ? (Real)std::atof(wa_env) : (Real)0.4;
  const Real cell_amp = std::max((Real)0, S - R_shaft - box_buffer);
  const Real k_curv = std::getenv("QJ_WIG_KCURV") ? (Real)std::atof(std::getenv("QJ_WIG_KCURV")) : (Real)2;
  const Real curv_amp = (R_shaft > 0) ? span*span / (4*pi*pi*k_curv*R_shaft) : (Real)1e30;
  const Real max_amp = wiggle ? std::min(std::min(wig_frac*(2*S), cell_amp), curv_amp) : (Real)0;
  const Real amp_min = (Real)0.5 * max_amp;           // floor so every cilium visibly wiggles

  std::vector<BridgeCurve<Real>> C; C.reserve(Npatch * Npatch);
  for (Integer iu = 0; iu < Npatch; iu++)
    for (Integer iv = 0; iv < Npatch; iv++) {
      const Real cx = (iu + (Real)0.5) * L / Npatch, cy = (iv + (Real)0.5) * L / Npatch;
      C.emplace_back(cx, cy, R_shaft, z0, z1);
    }
  const Long N = (Long)C.size();

  std::mt19937_64 gen(seed);
  std::uniform_real_distribution<double> amp_d((double)amp_min, (double)max_amp), th_d(0.0, (double)pi);
  auto draw = [&](Long i) { C[i].wig_amp = (Real)amp_d(gen); C[i].wig_theta = (Real)th_d(gen); };
  for (Long i = 0; i < N; i++) draw(i);

  // sampled surface clearance + collision VERIFIER (should never fire with the confined bound).
  std::vector<std::vector<Vec3<Real>>> smp(N);
  for (Long i = 0; i < N; i++) bridge_samples<Real>(C[i], smp[i]);
  auto min_clear = [&](Long i) { Real best = (Real)1e30;
    for (Long j = 0; j < N; j++) if (j != i) best = std::min(best, bridge_clearance<Real>(smp[i], smp[j], R_shaft));
    return best; };
  Long n_trigger = 0;
  for (Long i = 0; i < N; i++) {
    if (min_clear(i) >= coll_buffer) continue;
    // The confined-wiggle config must never reach here. Print loudly, then FALL BACK to regeneration so the
    // geometry stays valid (grant more buffer if this ever prints -- raise box_buffer / lower QJ_WIG_AMP_FRAC).
    if (!comm.Rank())
      std::cout << "  *** COLLISION TRIGGERED *** cilium " << i << " clearance " << min_clear(i)
                << " < " << coll_buffer << " (should be impossible with the confined wiggle bound)\n";
    n_trigger++;
    for (int tries = 0; tries < 3 && min_clear(i) < coll_buffer; tries++) { draw(i); bridge_samples<Real>(C[i], smp[i]); }
  }

  // report: min pairwise clearance + box containment (tube surface in [box_buffer, L-box_buffer], x & y).
  Real gmin = (Real)1e30; Long gi = -1, gj = -1;
  for (Long i = 0; i < N; i++) for (Long j = i+1; j < N; j++) { const Real g = bridge_clearance<Real>(smp[i], smp[j], R_shaft);
    if (g < gmin) { gmin = g; gi = i; gj = j; } }
  Real box_over = -(Real)1e30;
  for (Long i = 0; i < N; i++) for (const auto& p : smp[i]) {
    box_over = std::max(box_over, box_buffer - (p[0]-R_shaft)); box_over = std::max(box_over, (p[0]+R_shaft) - (L-box_buffer));
    box_over = std::max(box_over, box_buffer - (p[1]-R_shaft)); box_over = std::max(box_over, (p[1]+R_shaft) - (L-box_buffer));
  }
  if (!comm.Rank())
    std::cout << "  generate_cilia_bridge: " << N << " wall-to-wall cilia  seed=" << seed
              << "  wiggle=" << (wiggle ? "ON" : "OFF") << "  amp in [" << amp_min << ", " << max_amp
              << "] (bounds: patch " << wig_frac*(2*S) << ", cell " << cell_amp << ", curvature " << curv_amp << ")\n"
              << "    collisions triggered: " << n_trigger << "  [0 => confined wiggle held, no overlap possible]\n"
              << "    final min pairwise clearance = " << gmin << " (pair " << gi << "," << gj << ")  [>0 => no overlap]\n"
              << "    box overflow (x&y, incl. R_shaft) = " << box_over << "  [<=0 => each tube contained in its cell]\n";
  return C;
}

// ============================================================================
// QuadElemList base: TWO feet per cilium -- a collar+fillet in the bottom wall and one in the top wall (NO
// cap, NO shaft). Each foot is built canonically "up" (add_cilium_stud_flat with_cap=false, with_shaft=
// false, tilt=0), placed onto its plane (bottom = translate z+z_bottom; top = reflect z->z_top-z) and
// oriented OUT-OF-SOLID / into the fluid (bottom +z, top -z) to match the un-flippable slender shaft normal.
// ============================================================================
template <class Real> QuadElemList<Real> BuildCiliaBridgeBase(
    Integer order, const std::vector<BridgeCurve<Real>>& curves, Real R_shaft, Real r_fil, Integer Naz,
    Real S, Real z_bottom, Real z_top, Real core_frac, Real grade_exp, Integer n_straight, Integer n_trans,
    const Comm& comm) {
  const Real R0 = R_shaft + r_fil;
  SCTL_ASSERT(R0 < S);
  const Real H_dummy = z_top - z_bottom;   // add_cilium_stud_flat asserts H_reach > r_fil+R_shaft; shaft unused
  Vector<Real> Xall;
  Long nflip = 0;
  for (const auto& c : curves) {
    for (int plane = 0; plane < 2; plane++) {
      const Real z_plane = plane == 0 ? z_bottom : z_top;
      const Real uz      = plane == 0 ? (Real)+1 : (Real)-1;   // out-of-solid: bottom +z, top -z (into the gap)
      Vector<Real> Xp;   // canonical foot: collar + fillet (no cap, no shaft, no tilt)
      add_cilium_stud_flat<Real>(Xp, order, c.cx, c.cy, R_shaft, H_dummy, r_fil, S, Naz, /*Nc*/-1, /*Ns*/-1, /*Nf*/-1,
                                 grade_exp, core_frac, /*with_cap=*/false, /*tilt_rad=*/0, /*dir_x=*/1,
                                 n_straight, n_trans, /*with_shaft=*/false);
      // place: bottom translate (z += z_bottom); top z-reflect (z -> z_top - z)
      for (Long i = 0; i < Xp.Dim() / 3; i++) Xp[i*3+2] = (plane == 0) ? (z_bottom + Xp[i*3+2]) : (z_top - Xp[i*3+2]);
      if (orient_group_flat<Real>(Xp, order, z_plane, uz)) nflip++;
      for (Long i = 0; i < Xp.Dim(); i++) Xall.PushBack(Xp[i]);
    }
  }
  if (!comm.Rank())
    std::cout << "  cilia-bridge base: " << curves.size() << " cilia x2 feet (collar+fillet, no cap, no shaft)"
              << " Naz=" << Naz << "; " << nflip << " groups flipped outward; nodes=" << (Xall.Dim()/3) << "\n";
  return QuadElemList<Real>(order, Xall, comm);
}

// ============================================================================
// SlenderElemList shafts: one wall-to-wall fiber per cilium along its BridgeCurve centerline, radius
// R_shaft, n_axial panels (uniform in arc), bottom foot (f=0) -> top foot (f=1). MPI-partitioned by global
// panel index. (Mirrors BuildCiliaCarpetHybridShafts.)
// ============================================================================
template <class Real> SlenderElemList<Real> BuildCiliaBridgeShafts(
    const std::vector<BridgeCurve<Real>>& curves, Integer n_axial, Real R_shaft,
    Long cheb_order, Long fourier_order, const Comm& comm) {
  const Long Np = (Long)curves.size();
  const Long Nelem = Np * n_axial, Npr = comm.Size(), pid = comm.Rank();
  const Long k0g = (Nelem * pid) / Npr, k1g = (Nelem * (pid + 1)) / Npr;
  Vector<Long> elem_order, forder;
  Vector<Real> coord, radius, orient;
  Long eg = 0;
  for (Long p = 0; p < Np; p++) {
    const BridgeCurve<Real>& fc = curves[p];
    for (Integer k = 0; k < n_axial; k++, eg++) {
      if (eg < k0g || eg >= k1g) continue;
      elem_order.PushBack(cheb_order); forder.PushBack(fourier_order);
      const Vector<Real>& cn = SlenderElemList<Real>::CenterlineNodes(cheb_order);
      for (Long j = 0; j < cheb_order; j++) {
        const Real f = ((Real)k + cn[j]) / n_axial;             // 0 (bottom foot) .. 1 (top foot), uniform in arc
        const Vec3<Real> P = fc.point(f), e = fc.e1(f);
        coord.PushBack(P[0]); coord.PushBack(P[1]); coord.PushBack(P[2]);
        radius.PushBack(R_shaft);
        orient.PushBack(e[0]); orient.PushBack(e[1]); orient.PushBack(e[2]);
      }
    }
  }
  return SlenderElemList<Real>(elem_order, forder, coord, radius, orient);
}

// Minimum surface-to-surface clearance across all bridge cilia (tube), for reporting. <=0 => overlap.
template <class Real> Real cilia_bridge_min_clearance(const std::vector<BridgeCurve<Real>>& curves,
                                                      Long& pi_out, Long& pj_out, Integer nsamp = 64) {
  const Long Np = (Long)curves.size(); Real best = (Real)1e30; pi_out = pj_out = -1;
  std::vector<std::vector<Vec3<Real>>> smp(Np);
  for (Long i = 0; i < Np; i++) bridge_samples<Real>(curves[i], smp[i], nsamp);
  for (Long i = 0; i < Np; i++) for (Long j = i+1; j < Np; j++) { const Real g = bridge_clearance<Real>(smp[i], smp[j], curves[0].R_shaft);
    if (g < best) { best = g; pi_out = i; pj_out = j; } }
  return best;
}

} // namespace quad_junctions
