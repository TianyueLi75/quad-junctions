#pragma once
/**
 * Flat-plane cilia carpet geometry (pure QuadElemList), for the doubly-periodic (XY) Stokes flow
 * driver src/cilia_carpet-bie.cpp.
 *
 * The cilium-stud machinery in stud_sphere_geom.hpp mounts collar+fingers on a cubed SPHERE: its
 * collar/disk/finger builders derive the local frame as u = P0/|P0| (sphere radial), project every
 * node onto Rsph = |P0|, and place the foot ring at station a0 = sqrt(Rsph^2 - R_foot^2). On a FLAT
 * plane there is no sagitta, so this header provides flat-native analogues:
 *   - local frame hardcoded u = (0,0,1), e1 = (1,0,0), e2 = (0,1,0) (never calls mount_local_frame),
 *   - foot ring at station 0 (in the plane), collar/disk purely in-plane (no Rsph projection),
 *   - shaft/fillet/cap as EXACT bodies of revolution about u, re-origined so station = z above the plane.
 *
 * Everything else is reused from stud_sphere_geom.hpp: the 2D collar map collar_point + add_collar_block
 * (NOT the sphere-projecting collar_world / add_collar_block3 / emit_collar_sector family), collar_Nc,
 * the generalized-tip butterfly cap add_tip_cap_butterfly, coons/cline/carc, Vec2/Vec3/Mount, report_area.
 *
 * Canonical build: add_cilium_stud_flat builds one stud in a canonical "up" configuration (foot at z=0,
 * axis +z) -- exactly the sphere north-pole case (SphereMount has u = +z), so the shaft/fillet/collar/cap
 * sub-parts inherit the sphere code's verified mutual normal-orientation consistency. BuildCiliaCarpet
 * then PLACES each stud by pure translation (bottom plane) or z-reflection+translation (top plane); a
 * z-reflection reverses every element's normal uniformly, so a single group-flip still orients the whole
 * group. The group is oriented so its normals point OUTWARD-FROM-SOLID (bottom wall -z, top wall +z --
 * away from each other), matching the project's convention (DL constant-density identity = -1/2).
 */

#include <quad_junctions/stud_sphere_geom.hpp>

namespace quad_junctions {
using namespace sctl;

// Affine flat mount: (X,Y,depth) -> world, ignoring depth (the collar/disk are in-plane at z_plane).
template <class Real> Mount<Real> FlatMount(Real cx, Real cy, Real z_plane) {
  return [=](Real X, Real Y, Real /*depth*/) { return Vec3<Real>{cx + X, cy + Y, z_plane}; };
}

// One collar sector (circle R_foot -> square half-width S annulus) emitted PURELY in the tangent plane
// via the 2D collar_point + add_collar_block (mnt(X,Y,0), no sphere projection). Winding matches the
// legacy add_collar_sector: u-slow = azimuth, v-fast = radial. Nc rings, geometric radial grading.
template <class Real> void add_collar_sector_flat(Vector<Real>& X, Integer order, const Mount<Real>& mnt,
                                                  Real R_foot, Real S, Integer Naz, Integer Nc, Integer m, Real grade_exp = 1) {
  const Real pi = const_pi<Real>();
  const Real tha = pi / 4 + m * 2 * pi / Naz, thb = tha + 2 * pi / Naz;
  auto P = [=](Real t, Real phi) { return collar_point<Real>(R_foot, S, tha, thb, t, phi); };
  for (Integer ir = 0; ir < Nc; ir++) {
    const Real t0 = pow<Real>((Real)ir / Nc, grade_exp), t1 = pow<Real>((Real)(ir + 1) / Nc, grade_exp);
    auto Eb = [=](Real phi) { return P(t0, phi); };
    auto Et = [=](Real phi) { return P(t1, phi); };
    auto El = [=](Real xi)  { return P(t0 + xi * (t1 - t0), (Real)0); };
    auto Er = [=](Real xi)  { return P(t0 + xi * (t1 - t0), (Real)1); };
    add_collar_block<Real>(X, order, mnt, Eb, Et, El, Er);
  }
}

// Flat butterfly disk of radius R_disk centered at (cx,cy) in the plane z=z_plane (fills the foot hole).
// Copy of add_disk_fill with the sphere cap removed: place(Px,Qy) = (cx+Px, cy+Qy, z_plane) (aa=0).
// Central square core [-h,h]^2 (nc x nc) + 4 Coons arc caps to the exact R_disk circle. Same winding
// as add_disk_fill / the collar (-z inward before the group flip). core_frac = core half-size fraction.
template <class Real> void add_disk_fill_flat(Vector<Real>& X, Integer order, Real cx, Real cy, Real z_plane,
                                             Real R_disk, Integer Ndisk, Real core_frac = (Real)0.40) {
  const Real pi = const_pi<Real>();
  const Real h = core_frac * R_disk;
  const Integer nc = std::max<Integer>(1, Ndisk);
  const Vector<Real>& nds = QuadElemList<Real>::ParamNodes(order);
  auto place = [&](Real Px, Real Qy) -> Vec3<Real> { return Vec3<Real>{cx + Px, cy + Qy, z_plane}; };
  // Central square core, nc x nc panels (u=y slow, v=x => -z inward), matching the collar/caps.
  for (Integer ic = 0; ic < nc; ic++)
    for (Integer jc = 0; jc < nc; jc++) {
      const Real x0 = -h + 2*h*ic/nc, x1 = -h + 2*h*(ic+1)/nc, y0 = -h + 2*h*jc/nc, y1 = -h + 2*h*(jc+1)/nc;
      for (Integer i = 0; i < order; i++) { const Real yy = y0 + nds[i]*(y1-y0);
        for (Integer j = 0; j < order; j++) { const Real xx = x0 + nds[j]*(x1-x0);
          const Vec3<Real> w = place(xx, yy); X.PushBack(w[0]); X.PushBack(w[1]); X.PushBack(w[2]); } }
    }
  // 4 caps: core right edge (x=h) -> R_disk arc [-pi/4,pi/4], rotated by k*90deg.
  for (Integer k = 0; k < 4; k++) {
    const Real rot = k * pi / 2, cr = cos<Real>(rot), sr = sin<Real>(rot);
    auto pt = [=](Real eta, Real xi) -> Vec2<Real> {
      const Real th = -pi/4 + xi * (pi/2);
      const Vec2<Real> in{h, h*(2*xi - 1)};
      const Vec2<Real> out{R_disk*cos<Real>(th), R_disk*sin<Real>(th)};
      const Real px = (1-eta)*in[0] + eta*out[0], py = (1-eta)*in[1] + eta*out[1];
      return Vec2<Real>{cr*px - sr*py, sr*px + cr*py};
    };
    for (Integer ir = 0; ir < nc; ir++)
      for (Integer ia = 0; ia < nc; ia++) {
        const Real e0 = (Real)ir/nc, e1 = (Real)(ir+1)/nc, a0 = (Real)ia/nc, a1 = (Real)(ia+1)/nc;
        for (Integer i = 0; i < order; i++) { const Real xi = a0 + nds[i]*(a1-a0);
          for (Integer j = 0; j < order; j++) { const Real eta = e0 + nds[j]*(e1-e0);
            const Vec2<Real> P = pt(eta, xi); const Vec3<Real> w = place(P[0], P[1]);
            X.PushBack(w[0]); X.PushBack(w[1]); X.PushBack(w[2]); } }
      }
  }
}

// One canonical "up" cilium stud centered at (cx,cy): foot ring in the plane z=0, occupying z in
// [0, H_reach]. The base (fillet + first n_straight shaft panels) is straight-vertical for clean
// near-quadrature; the shaft centerline then POU-transitions (smootherstep of the tilt angle over
// n_trans panels) into a `tilt_rad` tilt in the +dir_x*x / z plane, and continues straight-tilted to the
// tip. The shaft is an exact swept tube of radius R_shaft with cross-sections perpendicular to the
// centerline tangent (planar bend => rotation-minimizing frame is analytic: binormal e2=(0,1,0), and
// e1 = e2 x t). Flat re-derivation of add_cilium_stud (R0 = R_shaft + r_fil):
//   foot ring: r=R0, z=0.  fillet: quarter-arc center (R0,r_fil) radius r_fil (straight, at base).
//   shaft:     swept tube along the (straight->tilt) centerline, arc s in [0,S], s=0 at fillet-top (z=r_fil).
//   cap:       hemisphere radius R_shaft oriented along the tip tangent, pole at z ~= H_reach.
// Winding (u-slow = along-centerline, decreasing arc so dX/du = -tangent; v-fast = azimuth) matches the
// straight case's cap->fillet direction, so the shaft normal stays consistent with collar/fillet/cap and
// one group-flip (by the caller) orients the whole stud. tilt_rad=0 recovers the straight finger.
template <class Real> void add_cilium_stud_flat(Vector<Real>& Xout, Integer order, Real cx, Real cy,
                                               Real R_shaft, Real H_reach, Real r_fil, Real S, Integer Naz,
                                               Integer Nc_in = -1, Integer Ns_in = -1, Integer Nf_in = -1,
                                               Real grade_exp = 1, Real core_frac = (Real)0.40, bool with_cap = true,
                                               Real tilt_rad = 0, Real dir_x = 1, Integer n_straight = 3, Integer n_trans = 3,
                                               bool with_shaft = true) {
  const Real pi = const_pi<Real>(), R0 = R_shaft + r_fil;
  SCTL_ASSERT(R0 < S && H_reach > r_fil + R_shaft && Naz >= 4 && Naz % 4 == 0);
  const Real az = 2 * pi * R_shaft / Naz;                       // panel arc size (~square panels)
  const Integer Nf = (Nf_in >= 1) ? Nf_in : std::max<Integer>(1, (Integer)std::llround((pi / 2 * r_fil) / az));
  const Integer Nc = (Nc_in >= 1) ? Nc_in : collar_Nc<Real>(R0, S, Naz);
  const Real ct0 = cos<Real>(tilt_rad), st0 = sin<Real>(tilt_rad);
  const Integer nstr = std::max<Integer>(0, n_straight), ntr = (tilt_rad > 0 ? std::max<Integer>(1, n_trans) : 0);
  const Real s0 = nstr * az, s_t = ntr * az;                    // straight arc, transition arc (panel-aligned)
  // Tilt angle vs arc length: 0 (straight) -> tilt_rad (smootherstep over the transition) -> tilt_rad.
  auto phi = [=](Real s) -> Real {
    if (s <= s0 || tilt_rad <= 0) return 0;
    if (s >= s0 + s_t) return tilt_rad;
    const Real x = (s - s0) / s_t, sm = x*x*x*(x*(x*(Real)6 - (Real)15) + (Real)10);
    return tilt_rad * sm;
  };
  auto tang = [=](Real s, Real T[3]) { const Real p = phi(s); T[0] = dir_x*sin<Real>(p); T[1] = 0; T[2] = cos<Real>(p); };
  // Integral of the tangent over [sa,sb] (fixed-N Simpson => a deterministic pure function of (sa,sb), so
  // adjacent panels sharing a boundary arc get bit-identical centerline points => watertight).
  auto integ = [&](Real sa, Real sb, Vec3<Real>& I) {
    const Integer n = 1000; const Real h = (sb - sa) / n; Real acc[3] = {0,0,0}, T[3];
    for (Integer k = 0; k <= n; k++) { tang(sa + k*h, T); const Real w = (k==0||k==n) ? 1 : (k%2 ? 4 : 2);
      for (int c=0;c<3;c++) acc[c] += w*T[c]; }
    for (int c=0;c<3;c++) I[c] = acc[c]*h/3;
  };
  // Centerline breakpoints and the tilted-segment length s1 solved so the cap pole reaches z=H_reach.
  Vec3<Real> Itr; integ(s0, s0 + s_t, Itr);                     // tang already carries dir_x, so don't re-apply it
  const Vec3<Real> Cs0{cx, cy, r_fil + s0};                     // centerline at end of straight segment
  const Vec3<Real> Cs0t{Cs0[0] + Itr[0], cy, Cs0[2] + Itr[2]};  // ... and end of transition
  Real s1 = (H_reach - R_shaft*ct0 - Cs0t[2]) / ct0;            // tilted arc so pole z = H_reach
  const Integer Ntil = std::max<Integer>(2, (Integer)std::llround(s1 / az));
  s1 = Ntil * az;
  const Real Sarc = s0 + s_t + s1;                             // total shaft arc
  const Integer Ns = (Ns_in >= 1) ? Ns_in : (nstr + ntr + Ntil);
  // Centerline point C(s): analytic on the straight/tilted segments, Simpson on the transition.
  auto Cof = [&](Real s, Vec3<Real>& C) {
    if (tilt_rad <= 0 || s <= s0) { C = Vec3<Real>{cx, cy, r_fil + s}; return; }             // straight base
    if (s <= s0 + s_t) { Vec3<Real> I; integ(s0, s, I); C = Vec3<Real>{Cs0[0] + I[0], cy, Cs0[2] + I[2]}; return; } // POU transition
    const Real ds = s - (s0 + s_t); C = Vec3<Real>{Cs0t[0] + dir_x*st0*ds, cy, Cs0t[2] + ct0*ds};  // straight tilted
  };
  auto e1of = [=](Real s, Real e1[3]) { const Real p = phi(s); e1[0] = cos<Real>(p); e1[1] = 0; e1[2] = -dir_x*sin<Real>(p); };
  const Real e2[3] = {0, 1, 0};
  const Vector<Real>& nds = QuadElemList<Real>::ParamNodes(order);
  // Exact swept-tube ring: for panel [tau0,tau1] (u-slow), arc s = Sarc*(1-tau) so tau=0 is the cap end and
  // arc DECREASES as i increases (dX/du = -tangent), matching the straight cap->fillet winding.
  auto sweep = [&](Real tau0, Real tau1, Real tha, Real thb) {
    for (Integer i = 0; i < order; i++) {
      const Real tau = tau0 + nds[i] * (tau1 - tau0), s = Sarc * (1 - tau);
      Vec3<Real> C; Real e1[3]; Cof(s, C); e1of(s, e1);
      for (Integer j = 0; j < order; j++) {
        const Real th = thb + nds[j] * (tha - thb), c = cos<Real>(th), sn = sin<Real>(th);
        Xout.PushBack(C[0] + R_shaft*(c*e1[0] + sn*e2[0]));
        Xout.PushBack(C[1] + R_shaft*(c*e1[1] + sn*e2[1]));
        Xout.PushBack(C[2] + R_shaft*(c*e1[2] + sn*e2[2]));
      }
    }
  };
  // Fillet (straight, at the base): quarter-arc revolution about +z (unchanged from the straight finger).
  auto rev = [&](const std::function<Real(Real)>& rF, const std::function<Real(Real)>& sF, Real t0, Real t1, Real tha, Real thb) {
    for (Integer i = 0; i < order; i++) {
      const Real t = t0 + nds[i] * (t1 - t0), r = rF(t), s = sF(t);
      for (Integer j = 0; j < order; j++) {
        const Real th = thb + nds[j] * (tha - thb), c = cos<Real>(th), sn = sin<Real>(th);
        Xout.PushBack(cx + r * c); Xout.PushBack(cy + r * sn); Xout.PushBack(s);
      }
    }
  };
  auto angF = [=](Real t) { return -pi/2 - (1 - t) * pi/2; };
  auto rFil = [=](Real t) { return R0 + r_fil * cos<Real>(angF(t)); };
  auto sFil = [=](Real t) { return r_fil + r_fil * sin<Real>(angF(t)); };
  const Mount<Real> collar_mnt = FlatMount<Real>(cx, cy, (Real)0);
  for (Integer m = 0; m < Naz; m++) {
    const Real tha = pi / 4 + m * 2 * pi / Naz, thb = tha + 2 * pi / Naz;
    if (with_shaft) for (Integer l = 0; l < Ns; l++) sweep((Real)l / Ns, (Real)(l + 1) / Ns, tha, thb);  // bent shaft (omit for the hybrid: a SlenderElemList replaces it)
    for (Integer l = 0; l < Nf; l++) rev(rFil, sFil, (Real)l / Nf, (Real)(l + 1) / Nf, tha, thb); // fillet
    add_collar_sector_flat<Real>(Xout, order, collar_mnt, R0, S, Naz, Nc, m, grade_exp);
  }
  if (with_cap) {  // hemisphere at the tip, oriented along the (tilted) tip tangent
    Vec3<Real> CS; Real e1S[3], T[3]; Cof(Sarc, CS); e1of(Sarc, e1S); tang(Sarc, T);
    const Real Ctip[3] = {CS[0], CS[1], CS[2]}, Ttip[3] = {T[0], T[1], T[2]}, w1[3] = {e1S[0], e1S[1], e1S[2]}, w2[3] = {0, 1, 0};
    add_tip_cap_butterfly<Real>(Xout, order, Ctip, Ttip, w1, w2, R_shaft, Naz, core_frac);
  }
}

// Transpose every order x order element of a group (swap u<->v) to negate its normals.
template <class Real> void flip_group(Vector<Real>& Xp, Integer order) {
  const Long nn = (Long)order * order, ne = Xp.Dim() / (nn * 3);
  for (Long e = 0; e < ne; e++)
    for (Integer i = 0; i < order; i++)
      for (Integer j = i + 1; j < order; j++)
        for (int c = 0; c < 3; c++) std::swap(Xp[(e*nn + i*order + j)*3 + c], Xp[(e*nn + j*order + i)*3 + c]);
}

// Orient a placed group so its normals align with u_out=(0,0,uz) (uz=+1 or -1): transpose if the
// accumulated n.u_out over the in-plane (collar/disk) nodes is negative. Returns true if flipped.
template <class Real> bool orient_group_flat(Vector<Real>& Xp, Integer order, Real z_plane, Real uz) {
  QuadElemList<Real> pc(order, Xp);
  Vector<Real> Xc, Xnc; pc.GetNodeCoord(&Xc, &Xnc, nullptr);
  Real acc = 0;
  for (Long i = 0; i < Xc.Dim() / 3; i++)
    if (std::fabs((double)(Xc[i*3+2] - z_plane)) < 1e-9) acc += Xnc[i*3+2] * uz;  // in-plane nodes only
  if (acc < 0) { flip_group<Real>(Xp, order); return true; }
  return false;
}

// Two flat walls at z=z_bottom and z=z_top spanning [0,L]x[0,L], each tiled by an Npatch x Npatch grid of
// collar+finger cilium studs pointing toward the other wall (outward normals point AWAY from each other).
// Fingers are elongated + tilted: the bottom finger tip reaches z=bot_tip tilting +x (along the pressure
// drop), the top finger tip reaches z=top_tip tilting -x (against it), each by tilt_rad, with a straight
// base (n_straight panels + fillet) before a smootherstep POU into the tilt (n_trans panels).
// fingers=false => flat collar+disk-fill control walls (literally-flat closed periodic slab).
template <class Real> QuadElemList<Real> BuildCiliaCarpet(Integer order, Integer Npatch,
    Real z_bottom, Real z_top, Real L, Real R_shaft, Real bot_tip, Real top_tip, Real r_fil, Integer Naz,
    Real core_frac = (Real)0.40, Real grade_exp = 1, bool fingers = true,
    Real tilt_rad = 0, Integer n_straight = 3, Integer n_trans = 3,
    Integer Nc_in = -1, Integer Ndisk_in = -1, const Comm& comm = Comm::Self()) {
  const Real pi = const_pi<Real>(), R0 = R_shaft + r_fil, S = L / (Real)(2 * Npatch), az = 2*pi*R_shaft/Naz;
  const Integer Nc    = (Nc_in    >= 1) ? Nc_in    : collar_Nc<Real>(R0, S, Naz);
  const Integer Ndisk = (Ndisk_in >= 1) ? Ndisk_in : std::max<Integer>(1, (Integer)std::llround((double)(R0/az)));
  // Containment / non-overlap asserts (PVFMM XY periodicity: z must stay in (0,L)).
  SCTL_ASSERT(R0 < S && z_bottom > 0 && z_top < L && z_bottom < z_top);
  if (fingers) {
    SCTL_ASSERT(bot_tip < top_tip);                              // opposite finger tips don't cross in z
    SCTL_ASSERT(z_bottom < bot_tip && bot_tip < z_top && z_bottom < top_tip && top_tip < z_top);
  }
  Vector<Real> Xall;
  Long nflip = 0;
  // Build the canonical "up" cell once per (it depends only on cx,cy which shift the whole cell), but cx,cy
  // differ per cell, so build per cell into a temp Xp (cheap: seconds total for a small carpet).
  for (int plane = 0; plane < 2; plane++) {
    const Real z_plane = plane == 0 ? z_bottom : z_top;
    const Real uz = plane == 0 ? (Real)-1 : (Real)+1;            // outward-from-solid: bottom -z, top +z
    const Real H_reach = plane == 0 ? (bot_tip - z_bottom) : (z_top - top_tip);  // canonical tip height
    const Real dir_x   = plane == 0 ? (Real)+1 : (Real)-1;       // bottom tilts +x (with flow), top -x (against)
    for (Integer iu = 0; iu < Npatch; iu++)
      for (Integer iv = 0; iv < Npatch; iv++) {
        const Real cx = (iu + (Real)0.5) * L / Npatch, cy = (iv + (Real)0.5) * L / Npatch;
        Vector<Real> Xp;  // canonical: foot in plane z=0, finger axis +z, tilting +dir_x*x with height
        if (fingers) {
          add_cilium_stud_flat<Real>(Xp, order, cx, cy, R_shaft, H_reach, r_fil, S, Naz, Nc, -1, -1, grade_exp, core_frac, true,
                                     tilt_rad, dir_x, n_straight, n_trans);
        } else {
          for (Integer m = 0; m < Naz; m++) add_collar_sector_flat<Real>(Xp, order, FlatMount<Real>(cx, cy, (Real)0), R0, S, Naz, Nc, m, grade_exp);
          add_disk_fill_flat<Real>(Xp, order, cx, cy, (Real)0, R0, Ndisk, core_frac);
        }
        // Place: bottom = translate (z += z_bottom); top = z-reflect about z_top (z -> z_top - z), so
        // fingers point -z into the gap. Reflection reverses all normals uniformly (still consistent).
        for (Long i = 0; i < Xp.Dim() / 3; i++) {
          Xp[i*3+2] = (plane == 0) ? (z_bottom + Xp[i*3+2]) : (z_top - Xp[i*3+2]);
        }
        if (orient_group_flat<Real>(Xp, order, z_plane, uz)) nflip++;
        for (Long i = 0; i < Xp.Dim(); i++) Xall.PushBack(Xp[i]);
      }
  }
  const Long ncell = 2 * Npatch * Npatch;
  if (!comm.Rank()) {
    std::cout << "  cilia-carpet: " << ncell << " cells (" << Npatch << "x" << Npatch << " x2 planes)"
              << " Naz=" << Naz << " Nc=" << Nc << (fingers ? "" : (" disk 5*" + std::to_string((long)Ndisk) + "^2"))
              << "; " << nflip << " groups flipped outward; nodes=" << (Xall.Dim()/3) << "\n";
    if (fingers) std::cout << "    fingers: tilt=" << (tilt_rad*180/pi) << " deg, bottom tip z=" << bot_tip
                           << " (+x), top tip z=" << top_tip << " (-x), straight base " << n_straight
                           << " panels + POU " << n_trans << " panels\n";
  }
  return QuadElemList<Real>(order, Xall, comm);
}

} // namespace quad_junctions
