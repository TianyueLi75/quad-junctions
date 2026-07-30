/**
 * HYBRID cilium-finger geometry: the studded-sphere QuadElemList BASE (cubed sphere with one pole
 * patch replaced by collar + fillet + butterfly cap, but NO shaft) joined to a CSBQ SlenderElemList
 * straight-cylinder SHAFT, fed into ONE BoundaryIntegralOp. The stud_sphere counterpart of
 * ybifurc_hybrid_geom.hpp.
 *
 * Unlike the Y-bifurcation hybrid, NO partition-of-unity transition is needed: the meshed finger's
 * shaft is already a perfect straight cylinder and both of its junction rings are already exact
 * circles of one radius, so the connection is trivial ("both sides circular"):
 *
 *   - fillet-bottom ring  (fillet t=0, depth r_fil):   circle of radius rho at z = z_hi
 *   - cap-equator ring    (add_cap_butterfly q=1, depth H_shaft): circle of radius rho at z = z_lo
 *
 * Both rings, in world coordinates, are the SAME circle rho = R_shaft*s (s the gnomonic factor of
 * SphereMount at the R_shaft-radius off-axis point) centered on the pole axis. A SlenderElemList
 * constant-radius cylinder spanning [z_lo, z_hi] at radius rho abuts both exactly -> two
 * circle<->circle cross-list seams (fillet<->slender, slender<->cap), conforming geometrically.
 *
 * This header reuses stud_sphere_geom.hpp unchanged (via the with_shaft=false toggle on
 * BuildCiliumStuddedSphere) and only adds two free functions.
 */
#pragma once

#include <quad_junctions/stud_sphere_geom.hpp>
#include <quad_junctions/flagella_centerline.hpp>   // spiral centerline (twirling-cilia "flagella" mode)
// CSBQ SlenderElemList (declared in namespace sctl). csbq.hpp is the umbrella header; the driver
// already includes it, but include the slender element here so this header is self-contained.
#include <csbq/slender_element.hpp>
#include <csbq/slender_element.cpp>
#include <vector>

namespace quad_junctions {
using namespace sctl;

// ============================================================================
// QuadElemList half of the hybrid finger: the studded sphere with the shaft OMITTED (collar + fillet
// + butterfly cap remain; the fillet bottom is left as an open circle and the cap floats at the deep
// end). Also returns the shaft seam geometry (rho, z_lo, z_hi) for the slender builder, computed by
// evaluating the mount at the two shaft junction depths -- both give the circle rho = R_shaft*s at
// z = R*s - depth (same s => a true world-space cylinder). H_shaft matches the value hardcoded in
// BuildCiliumStuddedSphere (0.05).
// ============================================================================
// invert_normals=true flips the whole base (sphere + collar + fillet + cap) to INWARD normals, so the
// combined surface adopts CSBQ's native slender convention (radially OUTWARD from the tube axis = into
// the sphere solid on the shaft wall). This is REQUIRED for a consistently-oriented hybrid surface:
// CSBQ's slender normal is always radially-outward-from-axis and cannot be flipped via the orientation
// argument, whereas the finger's outward-from-solid normal points toward the axis -- opposite. Flipping
// the base to inward makes both agree. The BIE then works with the inward (interior) convention.
template <class Real> QuadElemList<Real> BuildCiliumStuddedSphereBase(
    Integer order, Long PatchPerFace, Real R, Integer Naz, Real r_fil,
    Real& rho_out, Real& z_lo_out, Real& z_hi_out,
    Integer Nc = -1, Real grade_exp = 1, Real R_shaft = 0.015, const Comm& comm = Comm::Self(), bool invert_normals = true,
    Real H_shaft = 0.05) {
  const Mount<Real> mnt = SphereMount<Real>(R);
  // EXACT-circle foot (2026-07-24): the new fillet/cap are exact revolutions about u=z anchored at
  // a0 = sqrt(R^2 - R_foot^2). The slender attaches to the exact fillet-bottom ring (radius R_shaft,
  // z = a0-r_fil) and terminates at the exact cap-equator ring (radius R_shaft, z = a0-H_shaft) -- a
  // CONSTANT-radius fiber matching add_cilium_stud. Replaces the old gnomonic R_shaft*s / R*s-depth rings.
  const Real R_foot = R_shaft + r_fil, a0 = std::sqrt(R*R - R_foot*R_foot);
  const Real rho_arc = r_fil / (1 + R_foot/R), Cs = a0 * (1 - rho_arc/R);  // matches add_cilium_stud tangent-arc fillet
  rho_out  = R_shaft;
  z_hi_out = Cs;                                         // fillet-bottom / slender-top ring (tangent-arc shaft top)
  z_lo_out = a0 - H_shaft;                               // cap-equator / slender-bottom ring
  return BuildCiliumStuddedSphere<Real>(order, PatchPerFace, R, Naz, r_fil,
      /*Ns=*/-1, /*Nf=*/-1, Nc, /*Ncap=*/-1, grade_exp, R_shaft, comm, /*with_shaft=*/false, invert_normals, H_shaft);
}

// ============================================================================
// SlenderElemList half: a single straight constant-radius cylinder on the pole axis (x=y=0),
// spanning z in [z_lo, z_hi], radius rho. Both ends OPEN (fillet and cap are QuadElemList).
//
// NORMAL ORIENTATION: the finger is a concave indentation, so its wall normal must point TOWARD the
// axis (outward-from-solid = into the tube cavity) -- opposite CSBQ's default radial-OUTWARD fiber
// normal. GetGeom builds n = (dX/ds x dX/dtheta) with e2 = e1 x dx, so NEGATING the orientation
// vector e1 flips BOTH e1 and e2 and hence the entire normal n -> -n (the cylinder point set is
// unchanged since we integrate over all theta). flip_normal=true passes orient = -x_hat -> normal
// points toward the axis. NOTE: reversing the CENTERLINE direction does NOT flip n (it only swaps the
// sin-theta component -- a reparametrization). Verify with the driver's DL identity (-> -1/2); the
// divergence_check is INSENSITIVE to this sign (a cylinder wall has int n dA = 0 either way).
//
// SlenderElemList has no comm-aware ctor (it does NOT replicate-then-slice like QuadElemList), so
// under MPI the caller partitions the element loop itself -- the CSBQ k0=Nelem*pid/Np pattern.
// cheb_order (10; tied to CSBQ quadrature tables) and fourier_order (multiple of 4; e.g. 12).
// ============================================================================
template <class Real> SlenderElemList<Real> BuildCiliumShaftSlender(
    Real rho, Real z_lo, Real z_hi, Integer n_axial, bool flip_normal = true,
    Long cheb_order = 10, Long fourier_order = 12, const Comm& comm = Comm::Self()) {
  const Long Nelem = n_axial, Np = comm.Size(), pid = comm.Rank();
  const Long k0g = (Nelem * pid) / Np, k1g = (Nelem * (pid + 1)) / Np;   // this rank's global element range
  Vector<Long> elem_order, forder;
  Vector<Real> coord, radius, orient;
  const Real s0 = z_lo, s1 = z_hi;   // centerline direction is irrelevant to the normal sign
  // Negate the orientation vector to flip the surface normal (concave finger -> normal toward axis).
  const Real ex = flip_normal ? (Real)-1 : (Real)1;
  const Real e1[3] = {ex, (Real)0, (Real)0};   // beta=0 phase reference; e2 = e1 x dx; -e1 => n -> -n
  Long eg = 0;
  for (Integer p = 0; p < n_axial; p++, eg++) {
    if (eg < k0g || eg >= k1g) continue;   // not owned by this rank
    elem_order.PushBack(cheb_order);
    forder.PushBack(fourier_order);
    const Vector<Real>& cn = SlenderElemList<Real>::CenterlineNodes(cheb_order);
    for (Long j = 0; j < cheb_order; j++) {
      const Real z = s0 + (s1 - s0) * (p + cn[j]) / n_axial;   // straight station along z
      coord.PushBack((Real)0); coord.PushBack((Real)0); coord.PushBack(z);
      radius.PushBack(rho);
      orient.PushBack(e1[0]); orient.PushBack(e1[1]); orient.PushBack(e1[2]);
    }
  }
  return SlenderElemList<Real>(elem_order, forder, coord, radius, orient);
}

// Auto axial-panel count for ~unit-aspect slender panels: axial length / azimuthal node spacing.
template <class Real> Integer cilium_shaft_n_axial(Real rho, Real z_lo, Real z_hi, Long fourier_order) {
  const Real len = std::fabs(z_hi - z_lo), az = 2 * const_pi<Real>() * rho / (Real)fourier_order;
  return std::max<Integer>(1, (Integer)std::llround((double)(len / az)));
}

// ============================================================================
// MULTI-finger hybrid: EVERY cubed-sphere patch is a cilium finger. The QuadElemList base is all patch
// FEET (collar + fillet + cap, no shaft), and one SlenderElemList holds one straight radial cylinder
// SHAFT per patch. Each shaft is fitted to its foot's true ring by azimuthally averaging the mount at
// the shaft depths (handles the gnomonic distortion of off-pole patches; the ybifurc arm_ring_stats
// idea). Both rings' mean radius may differ slightly off-pole -> the fiber radius varies linearly.
// ============================================================================

// Azimuthal-mean ring of a stud (built by `mnt`) at shaft depth `depth`, decomposed about axis `u`
// (the patch-centre radial): R0 = mean_theta |P_perp|, a_axial = mean_theta (P . u). The ring points are
// mnt(R_shaft cos th, R_shaft sin th, depth) -- exactly the fillet-bottom (depth=r_fil) / cap-equator
// (depth=H_shaft) rings of add_cilium_stud.
template <class Real> void patch_shaft_ring(const Mount<Real>& mnt, const Real u[3], Real R_shaft, Real depth,
                                            Real& R0, Real& a_axial, Integer Nq = 64) {
  const Real twopi = 2 * const_pi<Real>();
  Real sR = 0, sA = 0;
  for (Integer m = 0; m < Nq; m++) {
    const Real th = twopi * m / Nq;
    const Vec3<Real> P = mnt(R_shaft * cos<Real>(th), R_shaft * sin<Real>(th), depth);
    const Real a = P[0]*u[0] + P[1]*u[1] + P[2]*u[2];
    const Real px = P[0]-a*u[0], py = P[1]-a*u[1], pz = P[2]-a*u[2];
    sR += sqrt<Real>(px*px + py*py + pz*pz); sA += a;
  }
  R0 = sR / Nq; a_axial = sA / Nq;
}

// Build the QuadElemList base of the all-finger sphere: every patch is a finger foot (collar+fillet+cap,
// NO shaft), oriented consistently and (if invert_normals) flipped INWARD to match the slender convention.
// Fills per-patch shaft data: axis[3*Np], a_bot/a_top[Np] (deep/near-surface axial stations),
// rho_bot/rho_top[Np] (mean ring radii). Np = 6*PatchPerFace^2.
template <class Real> QuadElemList<Real> BuildAllFingerSphereBase(
    Integer order, Long PatchPerFace, Real R, Integer Naz, Real r_fil,
    Vector<Real>& axis, Vector<Real>& a_bot, Vector<Real>& a_top, Vector<Real>& rho_bot, Vector<Real>& rho_top,
    Real grade_exp = 1, Real R_shaft = 0.015, Real H_shaft = 0.4, const Comm& comm = Comm::Self(), bool invert_normals = true,
    Real core_frac = (Real)0.40, Integer Nc_in = -1, Integer cap_Naz = -1, Integer Nf_in = -1,
    Real trans_depth = 0, Integer Ns_trans = 3) {
  const Real S = R / (Real)PatchPerFace;
  const Long nn = (Long)order*order, Np = 6*PatchPerFace*PatchPerFace;
  axis.ReInit(3*Np); a_bot.ReInit(Np); a_top.ReInit(Np); rho_bot.ReInit(Np); rho_top.ReInit(Np);
  Vector<Real> Xall;
  Long p = 0;
  for (Integer face = 0; face < 6; face++)
    for (Long iu = 0; iu < PatchPerFace; iu++)
      for (Long iv = 0; iv < PatchPerFace; iv++, p++) {
        const Real a_c = 2*(iu + (Real)0.5)/PatchPerFace - 1, b_c = 2*(iv + (Real)0.5)/PatchPerFace - 1;
        const Mount<Real> mnt = PatchMount<Real>(face, a_c, b_c, R);
        // patch-centre radial axis u
        Real px, py, pz; FacePoint<Real>(px, py, pz, face, a_c, b_c, R);
        const Real pr = std::sqrt(px*px+py*py+pz*pz); const Real u[3] = {px/pr, py/pr, pz/pr};
        axis[3*p+0]=u[0]; axis[3*p+1]=u[1]; axis[3*p+2]=u[2];
        // EXACT-circle foot (2026-07-24): the new collar/fillet/cap build exact revolutions about u anchored
        // at a0 = sqrt(R^2 - R_foot^2), so every finger is CONGRUENT (no gnomonic distortion) and the seam
        // rings are analytic. The slender attaches to the exact fillet-bottom ring (radius R_shaft, station
        // a0-r_fil) and terminates at the exact cap-equator ring (radius R_shaft, station a0-H_shaft) => a
        // CONSTANT-radius fiber (rho_top=rho_bot=R_shaft), like the ybifurc arms. Replaces patch_shaft_ring
        // (azimuthal mean of the gnomonic mount) which was only approximately circular off-axis.
        const Real R_foot = R_shaft + r_fil, a0 = std::sqrt(R*R - R_foot*R_foot);
        const Real rho_arc = r_fil / (1 + R_foot/R), Cs = a0 * (1 - rho_arc/R);  // matches add_cilium_stud tangent-arc fillet
        rho_top[p] = R_shaft; a_top[p] = Cs;              // exact fillet-bottom / slender-foot ring (tangent-arc shaft top)
        rho_bot[p] = R_shaft; a_bot[p] = a0 - H_shaft;    // exact cap-equator / slender-tip ring
        Vector<Real> Xp;
        add_cilium_stud<Real>(Xp, order, mnt, R_shaft, H_shaft, r_fil, S, Naz, /*flip=*/false,
                              /*Ns*/-1, /*Nf*/Nf_in, /*Nc*/Nc_in, -1, grade_exp, /*with_shaft=*/false, /*circularize=*/false, core_frac, cap_Naz, trans_depth, Ns_trans);
        // add_cilium_stud recomputes the SAME a0 chain internally (R=|mnt(0,0,0)|), so its cap ring == a_bot[p].
        { QuadElemList<Real> pc(order, Xp); Vector<Real> Xc, Xnc; pc.GetNodeCoord(&Xc, &Xnc, nullptr);
          Real acc = 0; for (Long i = 0; i < Xc.Dim()/3; i++) { const Real x=Xc[i*3],y=Xc[i*3+1],z=Xc[i*3+2],rr=std::sqrt(x*x+y*y+z*z); acc += (Xnc[i*3]*x+Xnc[i*3+1]*y+Xnc[i*3+2]*z)/rr; }
          if (acc < 0) { const Long ne = Xp.Dim()/(nn*3);
            for (Long e=0;e<ne;e++) for (Integer i=0;i<order;i++) for (Integer j=i+1;j<order;j++) for (int c=0;c<3;c++) std::swap(Xp[(e*nn+i*order+j)*3+c], Xp[(e*nn+j*order+i)*3+c]); }
        }
        for (Long i = 0; i < Xp.Dim(); i++) Xall.PushBack(Xp[i]);
      }
  if (invert_normals) {  // flip the WHOLE base inward to match CSBQ's radial-outward-from-axis slender normal
    const Long ne = Xall.Dim()/(nn*3);
    for (Long e=0;e<ne;e++) for (Integer i=0;i<order;i++) for (Integer j=i+1;j<order;j++) for (int c=0;c<3;c++) std::swap(Xall[(e*nn+i*order+j)*3+c], Xall[(e*nn+j*order+i)*3+c]);
  }
  if (!comm.Rank()) std::cout << "  all-finger base: " << Np << " feet (collar+fillet+cap, no shaft; exact-circular holes)"
                              << (invert_normals ? ", INVERTED -> inward" : "") << "\n";
  return QuadElemList<Real>(order, Xall, comm);
}

// Minimum clearance between the radial shaft segments (each t*u for t in [a_bot,a_top]); the closest
// approach of two rays through the origin is at their smallest common radius. Returns the min gap
// (centreline distance minus the two radii) and the offending pair.
template <class Real> Real finger_min_clearance(const Vector<Real>& axis, const Vector<Real>& a_bot,
                                                const Vector<Real>& rho_top, Long& pi_out, Long& pj_out) {
  const Long Np = a_bot.Dim(); Real best = 1e30; pi_out = pj_out = -1;
  for (Long i = 0; i < Np; i++) for (Long j = i+1; j < Np; j++) {
    const Real c = axis[3*i]*axis[3*j] + axis[3*i+1]*axis[3*j+1] + axis[3*i+2]*axis[3*j+2];
    const Real d2 = a_bot[i]*a_bot[i] + a_bot[j]*a_bot[j] - 2*a_bot[i]*a_bot[j]*c;   // dist^2 at deepest corner
    const Real gap = std::sqrt(std::max<Real>(0, d2)) - (rho_top[i] + rho_top[j]);   // subtract both radii
    if (gap < best) { best = gap; pi_out = i; pj_out = j; }
  }
  return best;
}

// One SlenderElemList holding Np straight radial cylinders: fiber p runs from a_bot[p]*u_p to a_top[p]*u_p
// with radius linearly interpolated rho_bot[p] -> rho_top[p]. MPI-partitioned by global panel index.
template <class Real> SlenderElemList<Real> BuildAllFingerShaftsSlender(
    const Vector<Real>& axis, const Vector<Real>& a_bot, const Vector<Real>& a_top,
    const Vector<Real>& rho_bot, const Vector<Real>& rho_top, Integer n_axial,
    Long cheb_order = 10, Long fourier_order = 12, const Comm& comm = Comm::Self(), Real axial_grade = 0) {
  const Long Np = a_bot.Dim();
  // Axial panel-boundary clustering: axial_grade in [0,1] blends uniform (0) with a cosine map that
  // packs panels toward BOTH fiber ends -- f=0 is the cap/tip seam, f=1 the foot/fillet seam.
  const Real pi_ = const_pi<Real>();
  auto gcluster = [axial_grade, pi_](Real t) { return (1-axial_grade)*t + axial_grade*(Real)0.5*(1 - cos<Real>(pi_*t)); };
  const Long Nelem = Np * n_axial, Npr = comm.Size(), pid = comm.Rank();
  const Long k0g = (Nelem * pid) / Npr, k1g = (Nelem * (pid + 1)) / Npr;
  Vector<Long> elem_order, forder;
  Vector<Real> coord, radius, orient;
  Long eg = 0;
  for (Long p = 0; p < Np; p++) {
    const Real u[3] = {axis[3*p], axis[3*p+1], axis[3*p+2]};
    // a fixed e1 perpendicular to u (phase reference); e2 = u x e1 is formed inside CSBQ
    Real w[3] = {1,0,0}; if (std::fabs(u[0]) > 0.9) { w[0]=0; w[1]=1; }
    const Real wd = w[0]*u[0]+w[1]*u[1]+w[2]*u[2];
    Real e1[3] = {w[0]-wd*u[0], w[1]-wd*u[1], w[2]-wd*u[2]};
    const Real e1n = std::sqrt(e1[0]*e1[0]+e1[1]*e1[1]+e1[2]*e1[2]); e1[0]/=e1n; e1[1]/=e1n; e1[2]/=e1n;
    const Real s0 = a_bot[p], s1 = a_top[p], r0 = rho_bot[p], r1 = rho_top[p];
    for (Integer k = 0; k < n_axial; k++, eg++) {
      if (eg < k0g || eg >= k1g) continue;
      elem_order.PushBack(cheb_order); forder.PushBack(fourier_order);
      const Vector<Real>& cn = SlenderElemList<Real>::CenterlineNodes(cheb_order);
      const Real flo = gcluster((Real)k / n_axial), fhi = gcluster((Real)(k + 1) / n_axial);
      for (Long j = 0; j < cheb_order; j++) {
        const Real f = flo + cn[j] * (fhi - flo);             // 0..1 along the fiber (cap/tip -> foot), graded toward both ends
        const Real s = s0 + (s1 - s0) * f, rr = r0 + (r1 - r0) * f;
        coord.PushBack(s*u[0]); coord.PushBack(s*u[1]); coord.PushBack(s*u[2]);
        radius.PushBack(rr);
        orient.PushBack(e1[0]); orient.PushBack(e1[1]); orient.PushBack(e1[2]);
      }
    }
  }
  return SlenderElemList<Real>(elem_order, forder, coord, radius, orient);
}

// ============================================================================
// TWIRLING CILIA ("flagella") hybrid: every patch is a cilium finger whose CSBQ slender shaft follows a
// SPIRAL centerline (flagella_centerline.hpp), not a straight radial line. The QuadElemList base is the
// patch feet (collar + fillet, NO shaft, NO axis-cap) plus a butterfly cap placed at each SPIRAL TIP with
// the tip travel-frame; one SlenderElemList holds the Np spiral fibers. See flagella_centerline.hpp for
// the straight-lead -> smootherstep-POU-corner -> spiral construction (C2 at the foot seam).
// ============================================================================

// Auto axial-panel count for a spiral finger: arc length / azimuthal node spacing (~unit-aspect panels).
// Arc is measured with a provisional n_axial (window placement is a negligible fraction of the arc).
template <class Real> Integer flagella_n_axial(const FlagellaCfg<Real>& cfg, const Real axis[3], Real rho, Long fourier) {
  FlagellaCfg<Real> c2 = cfg; c2.n_axial = 200;
  const Real arc = flagella_arclen<Real>(c2, axis);
  const Real az = 2 * const_pi<Real>() * rho / (Real)fourier;
  const Integer n = (Integer)std::llround((double)(arc / az));
  return std::max<Integer>(cfg.lead_panels + cfg.corner_panels + 2, n);
}

// Build the QuadElemList base of the flagella all-finger sphere: every patch is a finger foot
// (collar + fillet, NO shaft) + a butterfly cap at the spiral tip. Also returns the per-patch
// FlagellumCurve objects (one source of truth, reused by the slender builder + clearance + dump) and the
// patch axes. cfg.n_axial MUST already be set (the curve windows depend on it).
template <class Real> QuadElemList<Real> BuildAllFingerFlagellaSphereBase(
    Integer order, Long PatchPerFace, Real R, Integer Naz, const FlagellaCfg<Real>& cfg,
    std::vector<FlagellumCurve<Real>>& curves, Vector<Real>& axis,
    Real grade_exp = 1, const Comm& comm = Comm::Self(), bool invert_normals = true,
    Real core_frac = (Real)0.40, Integer Nc_in = -1, Integer cap_Naz = -1, Integer Nf_in = -1) {
  const Real S = R / (Real)PatchPerFace;
  const Long nn = (Long)order*order, Np = 6*PatchPerFace*PatchPerFace;
  const Integer cNaz = (cap_Naz > 0 ? cap_Naz : Naz);
  const Real H_dummy = std::max<Real>((Real)0.1, 2*cfg.r_fil);   // unused (with_cap=false) but must satisfy H_shaft>r_fil
  curves.clear(); curves.reserve(Np); axis.ReInit(3*Np);
  Vector<Real> Xall;
  Long p = 0;
  for (Integer face = 0; face < 6; face++)
    for (Long iu = 0; iu < PatchPerFace; iu++)
      for (Long iv = 0; iv < PatchPerFace; iv++, p++) {
        const Real a_c = 2*(iu + (Real)0.5)/PatchPerFace - 1, b_c = 2*(iv + (Real)0.5)/PatchPerFace - 1;
        const Mount<Real> mnt = PatchMount<Real>(face, a_c, b_c, R);
        Real px, py, pz; FacePoint<Real>(px, py, pz, face, a_c, b_c, R);
        const Real pr = std::sqrt(px*px+py*py+pz*pz); const Real u[3] = {px/pr, py/pr, pz/pr};
        axis[3*p+0]=u[0]; axis[3*p+1]=u[1]; axis[3*p+2]=u[2];
        curves.emplace_back(cfg, u);
        // transpose (swap u<->v) every element of X to flip its normals.
        auto transpose_all = [&](Vector<Real>& X) { const Long ne = X.Dim()/(nn*3);
          for (Long e=0;e<ne;e++) for (Integer i=0;i<order;i++) for (Integer j=i+1;j<order;j++) for (int c=0;c<3;c++) std::swap(X[(e*nn+i*order+j)*3+c], X[(e*nn+j*order+i)*3+c]); };
        // FOOT: collar + fillet, NO shaft (slender replaces it), NO axis cap (placed at the spiral tip).
        // Oriented OUTWARD-from-solid via the radial flux sign (n.rhat > 0), like the straight all-finger base.
        Vector<Real> Xfoot;
        add_cilium_stud<Real>(Xfoot, order, mnt, cfg.R_shaft, H_dummy, cfg.r_fil, S, Naz, /*flip=*/false,
                              /*Ns*/-1, /*Nf*/Nf_in, /*Nc*/Nc_in, -1, grade_exp, /*with_shaft=*/false,
                              /*circularize=*/false, core_frac, cap_Naz, /*trans_depth=*/0, /*Ns_trans=*/3,
                              /*cap_rho=*/-1, /*cap_a=*/0, /*with_cap=*/false);
        { QuadElemList<Real> pc(order, Xfoot); Vector<Real> Xc, Xnc; pc.GetNodeCoord(&Xc, &Xnc, nullptr);
          Real acc = 0; for (Long i = 0; i < Xc.Dim()/3; i++) { const Real x=Xc[i*3],y=Xc[i*3+1],z=Xc[i*3+2],rr=std::sqrt(x*x+y*y+z*z); acc += (Xnc[i*3]*x+Xnc[i*3+1]*y+Xnc[i*3+2]*z)/rr; }
          if (acc < 0) transpose_all(Xfoot); }
        // TIP CAP: butterfly dome at the spiral tip, oriented by the tip travel-frame; equator == slender tip
        // ring. Orient it "OUT-OF-SOLID" INDEPENDENTLY, matching the collar's pre-inversion convention (+rhat)
        // so the single base inversion flips BOTH to into-solid (== the slender's natural outward-from-axis).
        // The cap bulges along +Ttip (deeper, into the sphere solid), so out-of-solid = toward the cavity =
        // -Ttip: a hemisphere's flux integral must be NEGATIVE along +Ttip. The foot's radial-flux sign can't
        // orient the cap (it is deep inside, ~perpendicular to rhat), and one group-flip can't reconcile the
        // two if their windings differ -- so each is oriented separately before the base inversion.
        Real Ctip[3], Ttip[3], w1[3], w2[3], rho_tip;
        curves.back().tip_frame(Ctip, Ttip, w1, w2, rho_tip);
        Vector<Real> Xcap;
        add_tip_cap_butterfly<Real>(Xcap, order, Ctip, Ttip, w1, w2, rho_tip, cNaz, core_frac);
        { QuadElemList<Real> pc(order, Xcap); Vector<Real> Xc, Xnc; pc.GetNodeCoord(&Xc, &Xnc, nullptr);
          Real accT = 0; for (Long i = 0; i < Xc.Dim()/3; i++) accT += Xnc[i*3]*Ttip[0] + Xnc[i*3+1]*Ttip[1] + Xnc[i*3+2]*Ttip[2];
          if (accT > 0) transpose_all(Xcap); }
        for (Long i = 0; i < Xfoot.Dim(); i++) Xall.PushBack(Xfoot[i]);
        for (Long i = 0; i < Xcap.Dim();  i++) Xall.PushBack(Xcap[i]);
      }
  if (invert_normals) {  // flip the WHOLE base inward to match CSBQ's radial-outward-from-axis slender normal
    const Long ne = Xall.Dim()/(nn*3);
    for (Long e=0;e<ne;e++) for (Integer i=0;i<order;i++) for (Integer j=i+1;j<order;j++) for (int c=0;c<3;c++) std::swap(Xall[(e*nn+i*order+j)*3+c], Xall[(e*nn+j*order+i)*3+c]);
  }
  if (!comm.Rank()) std::cout << "  flagella all-finger base: " << Np << " feet (collar+fillet, no shaft; butterfly cap at spiral tip)"
                              << (invert_normals ? ", INVERTED -> inward" : "") << "\n";
  return QuadElemList<Real>(order, Xall, comm);
}

// One SlenderElemList holding the Np SPIRAL fibers: fiber p follows curves[p].point(f), radius
// curves[p].radius(f), phase curves[p].e1(f), over cfg.n_axial uniform panels (f uniform so the
// lead/corner windows land on panel boundaries). MPI-partitioned by global panel index.
template <class Real> SlenderElemList<Real> BuildAllFingerFlagellaShaftsSlender(
    const std::vector<FlagellumCurve<Real>>& curves, Integer n_axial,
    Long cheb_order = 10, Long fourier_order = 12, const Comm& comm = Comm::Self()) {
  const Long Np = (Long)curves.size();
  const Long Nelem = Np * n_axial, Npr = comm.Size(), pid = comm.Rank();
  const Long k0g = (Nelem * pid) / Npr, k1g = (Nelem * (pid + 1)) / Npr;
  Vector<Long> elem_order, forder;
  Vector<Real> coord, radius, orient;
  Long eg = 0;
  for (Long p = 0; p < Np; p++) {
    const FlagellumCurve<Real>& fc = curves[p];
    for (Integer k = 0; k < n_axial; k++, eg++) {
      if (eg < k0g || eg >= k1g) continue;
      elem_order.PushBack(cheb_order); forder.PushBack(fourier_order);
      const Vector<Real>& cn = SlenderElemList<Real>::CenterlineNodes(cheb_order);
      for (Long j = 0; j < cheb_order; j++) {
        const Real f = ((Real)k + cn[j]) / n_axial;
        const Vec3<Real> P = fc.point(f); const Vec3<Real> e = fc.e1(f);
        coord.PushBack(P[0]); coord.PushBack(P[1]); coord.PushBack(P[2]);
        radius.PushBack(fc.radius(f));
        orient.PushBack(e[0]); orient.PushBack(e[1]); orient.PushBack(e[2]);
      }
    }
  }
  return SlenderElemList<Real>(elem_order, forder, coord, radius, orient);
}

} // namespace quad_junctions
