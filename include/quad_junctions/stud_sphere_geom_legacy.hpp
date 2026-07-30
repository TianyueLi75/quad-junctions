#pragma once
/**
 * ============================================================================================
 * ARCHIVED / FROZEN (2026-07-24): the ORIGINAL gnomonic tangent-plane collar + mount scheme, kept
 * verbatim as an A/B reference. The live `stud_sphere_geom.hpp` was rewritten to an on-sphere
 * exact-circle foot (small-circle inner ring about the patch axis). This copy is included NOWHERE and
 * is inert; it exists only to compare the new scheme against the old elliptical-foot behaviour. Do NOT
 * include or edit — treat as read-only history.
 * ============================================================================================
 *
 * Cilia / collar / mount surface-mesh generator (pure QuadElemList) for cubed-sphere geometries.
 * Ported verbatim from SCTL_quad_element/src/test-gmsh-geom.cpp (the cilia-mount code migrated here).
 *
 * Three layers of order x order GL patches (AoS, u-slow), concatenated into a QuadElemList:
 *   1. Mount  : (X,Y,depth) -> world functor placing a tangent-plane patch on a host surface
 *               (SphereMount = gnomonic north pole; PatchMount = gnomonic on any cube face).
 *   2. Collar : collar_point circle(R_foot)->square(S) annulus (add_collar_sector) + add_disk_fill
 *               butterfly O-grid closing the foot hole. ONE shared collar mesh.
 *   3. Cilia  : add_cilium_stud stacks shaft + fillet + collar + butterfly cap (add_cap_butterfly).
 *
 * Builders: BuildCiliumStuddedSphere (single stud), BuildSphereWithCollarFill (one collar patch, no
 * finger), BuildAllCollarFillSphere (every patch = collar+disk, optional one finger). report_area is a
 * geometry/watertightness summary helper. Observed-best defaults: disk core_frac=0.40; collar Naz=4
 * (single, conforming outer seam), Naz=8 (finger shaft + all-collar rim). See the drivers
 * src/stud_sphere-bie.cpp (DL + Green's identity) and src/stud_sphere-geom.cpp (geometry checks).
 */

#include <sctl.hpp>
#include <sctl/experimental/quad_element.hpp>
#include <sctl/experimental/quad_element.cpp>
#include <quad_junctions/mpi_utils.hpp>
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <functional>
#include <iomanip>
#include <string>
#include <vector>

namespace quad_junctions {
using namespace sctl;

// ---- Verification helpers ------------------------------------------------
// Surface area + pure-geometry closure checks via the far-field Jacobian weights. Under MPI each
// rank sees only its element slice, so the area/flux/volume sums and the min-weight are reduced
// across ranks (SUM/MIN) and the summary is printed on rank 0 only.
template <class Real> void report_area(const QuadElemList<Real>& elem_lst, const Comm& comm = Comm::Self()) {
  Vector<Real> wts, Xtemp, Xntemp, dist_far;
  Vector<Long> elem_wise_temp;
  elem_lst.GetFarFieldNodes(Xtemp, Xntemp, wts, dist_far, elem_wise_temp, 1);
  Real Area = 0, wmin = wts.Dim() ? wts[0] : (Real)1e30;
  Real flux[3] = {0, 0, 0}, xdotn = 0;  // pure-geometry closure checks (far-field only, no singular quad)
  for (Long i = 0; i < wts.Dim(); i++) {
    Area += wts[i]; wmin = std::min(wmin, wts[i]);
    for (int k = 0; k < 3; k++) flux[k] += wts[i] * Xntemp[i*3+k];              // want ~0 for closed surface
    for (int k = 0; k < 3; k++) xdotn += wts[i] * Xtemp[i*3+k] * Xntemp[i*3+k]; // = 3*Volume by divergence thm
  }
  // Reduce partial sums across ranks (flux is a vector so its magnitude must be taken AFTER reduction).
  Area  = GlobalReduce((double)Area,  comm, CommOp::SUM);
  xdotn = GlobalReduce((double)xdotn, comm, CommOp::SUM);
  wmin  = GlobalReduce((double)wmin,  comm, CommOp::MIN);
  for (int k = 0; k < 3; k++) flux[k] = GlobalReduce((double)flux[k], comm, CommOp::SUM);
  const Long npatch = GlobalReduce((Long)elem_lst.Size(), comm, CommOp::SUM);
  const Long nnode  = GlobalReduce((Long)(Xntemp.Dim()/3), comm, CommOp::SUM);
  const Real flux_mag = std::sqrt(flux[0]*flux[0]+flux[1]*flux[1]+flux[2]*flux[2]);
  if (comm.Rank()) return;
  std::cout << "  patches=" << npatch << "  nodes=" << nnode
            << "  surface area=" << std::setprecision(10) << Area
            << "  min quad weight=" << wmin << (wmin > 0 ? "  (all Jacobians positive)" : "  (WARNING: non-positive Jacobian!)") << std::endl;
  std::cout << "  [geom-closure] |int n dA| = " << std::setprecision(4) << flux_mag
            << "   (rel " << flux_mag / Area << ")   volume=int(x.n)/3 = " << std::setprecision(10) << xdotn / 3 << std::endl;
}

// ---- cubed sphere ----
template <class Real> void FacePoint(Real& x, Real& y, Real& z, Integer face, Real a, Real b, Real R) {
  switch (face) {
    case 0: x =  1; y =  a; z =  b; break;
    case 1: x = -1; y = -a; z =  b; break;
    case 2: x =  a; y =  1; z = -b; break;
    case 3: x =  a; y = -1; z =  b; break;
    case 4: x =  a; y =  b; z =  1; break;
    case 5: x = -a; y =  b; z = -1; break;
    default: SCTL_ASSERT(false);
  }
  const Real r = sqrt<Real>(x * x + y * y + z * z);
  x *= R / r; y *= R / r; z *= R / r;
}

// Append cubed-sphere patches (order x order GL nodes, u slow), skipping one patch.
template <class Real> void add_cubedsphere(Vector<Real>& X, Integer order, Long PatchPerFace, Real R, Integer skipFace, Long skipIu, Long skipIv) {
  const Vector<Real>& nds = QuadElemList<Real>::ParamNodes(order);
  for (Integer face = 0; face < 6; face++)
    for (Long iu = 0; iu < PatchPerFace; iu++)
      for (Long iv = 0; iv < PatchPerFace; iv++) {
        if (face == skipFace && iu == skipIu && iv == skipIv) continue;
        for (Integer i = 0; i < order; i++) {
          const Real a = 2 * (iu + nds[i]) / (Real)PatchPerFace - 1;
          for (Integer j = 0; j < order; j++) {
            const Real b = 2 * (iv + nds[j]) / (Real)PatchPerFace - 1;
            Real x, y, z; FacePoint<Real>(x, y, z, face, a, b, R);
            X.PushBack(x); X.PushBack(y); X.PushBack(z);
          }
        }
      }
}

// ---- cilium stud (adapted from test-cilia-sphere.cpp; appends AoS coords) ----
template <class Real> using Vec2 = std::array<Real, 2>;
template <class Real> using Vec3 = std::array<Real, 3>;
template <class Real> using Curve = std::function<Vec2<Real>(Real)>;
template <class Real> using Mount = std::function<Vec3<Real>(Real, Real, Real)>;

// Collar gnomonically projected onto a radius-R sphere at the north pole; shaft along -z.
template <class Real> Mount<Real> SphereMount(Real R) {
  return [R](Real X, Real Y, Real depth) {
    const Real s = R / sqrt<Real>(X * X + Y * Y + R * R);
    return Vec3<Real>{X * s, Y * s, R * s - depth};
  };
}
// Forward decl (FacePoint is defined further down); PatchMount places a collar/disk tangent (X,Y) in
// [-S,S]^2 onto the cubed-sphere patch centred at gnomonic (a_c,b_c) of `face` -- so ANY cube patch can
// host the collarfill pattern (used by BuildAllCollarFillSphere). depth pushes inward along the radial.
template <class Real> void FacePoint(Real& x, Real& y, Real& z, Integer face, Real a, Real b, Real R);
template <class Real> Mount<Real> PatchMount(Integer face, Real a_c, Real b_c, Real R) {
  return [=](Real X, Real Y, Real depth) {
    Real x, y, z; FacePoint<Real>(x, y, z, face, a_c + X, b_c + Y, R);
    if (depth != (Real)0) { const Real r = sqrt<Real>(x*x + y*y + z*z); x -= depth*x/r; y -= depth*y/r; z -= depth*z/r; }
    return Vec3<Real>{x, y, z};
  };
}
// Tangent-plane distortion correction A (2x2, row-major) for a mount: gnomonic PatchMount maps a small
// tangent circle to an ELLIPSE, so a circular CSBQ slender cannot match an off-pole finger's foot ring.
// A pre-distorts the "circle" tangent coords so mnt(A*(p,q)) is a true world circle for small radii:
// with M=[dmnt/dX | dmnt/dY] at (0,0) and an orthonormal frame (e1,e2) spanning its columns,
// A = (M^T M)^{-1} M^T [e1|e2]  =>  mnt(A*(R cos th, R sin th)) ~= centre + R(cos th e1 + sin th e2).
// (Exact to linear order; residual ~ (R/Rsphere)^2 over the tiny foot.) At the pole A = identity.
template <class Real> void mount_tangent_correction(const Mount<Real>& mnt, Real A[4]) {
  const Real hh = (Real)1e-3;
  const Vec3<Real> pa = mnt(hh,0,0), ma = mnt(-hh,0,0), pb = mnt(0,hh,0), mb = mnt(0,-hh,0);
  Real Ta[3], Tb[3];
  for (int k = 0; k < 3; k++) { Ta[k] = (pa[k]-ma[k])/(2*hh); Tb[k] = (pb[k]-mb[k])/(2*hh); }
  auto dot = [](const Real* a, const Real* b) { return a[0]*b[0]+a[1]*b[1]+a[2]*b[2]; };
  Real e1[3], e2[3];
  const Real na = std::sqrt(dot(Ta,Ta)); for (int k=0;k<3;k++) e1[k] = Ta[k]/na;
  const Real tbe1 = dot(Tb,e1); Real t2[3]; for (int k=0;k<3;k++) t2[k] = Tb[k]-tbe1*e1[k];
  const Real n2 = std::sqrt(dot(t2,t2)); for (int k=0;k<3;k++) e2[k] = t2[k]/n2;
  const Real G00=dot(Ta,Ta), G01=dot(Ta,Tb), G11=dot(Tb,Tb), det=G00*G11-G01*G01;
  const Real Q00=dot(Ta,e1), Q01=dot(Ta,e2), Q10=dot(Tb,e1), Q11=dot(Tb,e2);
  const Real Gi00=G11/det, Gi01=-G01/det, Gi11=G00/det;
  A[0]=Gi00*Q00+Gi01*Q10; A[1]=Gi00*Q01+Gi01*Q11;
  A[2]=Gi01*Q00+Gi11*Q10; A[3]=Gi01*Q01+Gi11*Q11;
}

// Local orthonormal frame of a mount at the patch centre: P0 = mnt(0,0,0) (on the sphere), u = P0/|P0|
// (outward radial), and (e1,e2) an orthonormal tangent basis (from the mount's tangent Jacobian). Used to
// build the finger's fillet/cap as EXACT bodies of revolution about u so their rings are true circles
// (a circular CSBQ slender then conforms exactly, even for off-pole/gnomonically-distorted patches).
template <class Real> void mount_local_frame(const Mount<Real>& mnt, Real P0[3], Real u[3], Real e1[3], Real e2[3]) {
  const Real hh = (Real)1e-3;
  const Vec3<Real> c = mnt(0,0,0), pa = mnt(hh,0,0), ma = mnt(-hh,0,0), pb = mnt(0,hh,0), mb = mnt(0,-hh,0);
  Real Ta[3], Tb[3];
  for (int k = 0; k < 3; k++) { P0[k] = c[k]; Ta[k] = (pa[k]-ma[k])/(2*hh); Tb[k] = (pb[k]-mb[k])/(2*hh); }
  auto dot = [](const Real* a, const Real* b) { return a[0]*b[0]+a[1]*b[1]+a[2]*b[2]; };
  const Real Rr = std::sqrt(dot(P0,P0)); for (int k=0;k<3;k++) u[k] = P0[k]/Rr;
  const Real na = std::sqrt(dot(Ta,Ta)); for (int k=0;k<3;k++) e1[k] = Ta[k]/na;
  const Real be1 = dot(Tb,e1); Real t2[3]; for (int k=0;k<3;k++) t2[k] = Tb[k]-be1*e1[k];
  const Real n2 = std::sqrt(dot(t2,t2)); for (int k=0;k<3;k++) e2[k] = t2[k]/n2;
}

template <class Real> Vec2<Real> coons(const Curve<Real>& Eb, const Curve<Real>& Et, const Curve<Real>& El, const Curve<Real>& Er, Real xi, Real eta) {
  const Vec2<Real> b = Eb(xi), t = Et(xi), l = El(eta), r = Er(eta);
  const Vec2<Real> c00 = Eb(0), c10 = Eb(1), c01 = Et(0), c11 = Et(1);
  Vec2<Real> P;
  for (int k = 0; k < 2; k++)
    P[k] = (1 - eta) * b[k] + eta * t[k] + (1 - xi) * l[k] + xi * r[k]
         - ((1 - xi) * (1 - eta) * c00[k] + xi * (1 - eta) * c10[k] + (1 - xi) * eta * c01[k] + xi * eta * c11[k]);
  return P;
}
template <class Real> Curve<Real> cline(Vec2<Real> A, Vec2<Real> B) {
  return [A, B](Real s) { return Vec2<Real>{A[0] + s * (B[0] - A[0]), A[1] + s * (B[1] - A[1])}; };
}
template <class Real> Curve<Real> carc(Real rad, Real th0, Real th1) {
  return [rad, th0, th1](Real s) { const Real th = th0 + s * (th1 - th0); return Vec2<Real>{rad * cos<Real>(th), rad * sin<Real>(th)}; };
}
template <class Real> void add_collar_block(Vector<Real>& X, Integer order, const Mount<Real>& mnt, const Curve<Real>& Eb, const Curve<Real>& Et, const Curve<Real>& El, const Curve<Real>& Er) {
  const Vector<Real>& nds = QuadElemList<Real>::ParamNodes(order);
  for (Integer i = 0; i < order; i++)
    for (Integer j = 0; j < order; j++) {
      const Vec2<Real> P = coons<Real>(Eb, Et, El, Er, nds[i], nds[j]);
      const Vec3<Real> w = mnt(P[0], P[1], (Real)0);
      X.PushBack(w[0]); X.PushBack(w[1]); X.PushBack(w[2]);
    }
}

// ---- Shared collar map (ONE source of truth for the circle->square annulus) ---------------------
// 2D map (tangent-plane, before Mount) of the collar annulus for azimuthal sector [tha,thb]:
//   t   in [0,1]: radial, 0 = inner circle R_foot, 1 = outer square edge (half-width S).
//   phi in [0,1]: azimuthal within the sector.
// Square-cell design (three rules, each addressing a specific cell-quality/seam requirement):
//   * OUTER boundary is LINEAR along the straight square edge (interp of the two sector-corner square
//     points) => node density matches the neighbouring cubed-sphere patch (whose edge nodes are linear
//     in the gnomonic coord), so the outer collar<->sphere seam has no density jump.
//   * INNER boundary is uniform ANGLE on the R_foot circle (= uniform arc length).
//   * per-radial-LINE geometric grading to that line's own outer radius rho_out=|O(phi)| (rho_out ranges
//     S..sqrt2*S) so cells stay ~unit-aspect even on the longer corner lines.
template <class Real> Vec2<Real> collar_point(Real R_foot, Real S, Real tha, Real thb, Real t, Real phi, const Real* Acorr = nullptr) {
  auto sq = [&](Real th) { const Real c = cos<Real>(th), s = sin<Real>(th), m = std::max(std::fabs(c), std::fabs(s)); return Vec2<Real>{S * c / m, S * s / m}; };
  const Real th = tha + phi * (thb - tha);
  Vec2<Real> I{R_foot * cos<Real>(th), R_foot * sin<Real>(th)};                       // inner: uniform angle
  if (Acorr) I = Vec2<Real>{Acorr[0]*I[0]+Acorr[1]*I[1], Acorr[2]*I[0]+Acorr[3]*I[1]}; // circularize the foot hole (off-pole gnomonic fix)
  const Vec2<Real> A = sq(tha), B = sq(thb);
  const Vec2<Real> O{(1 - phi) * A[0] + phi * B[0], (1 - phi) * A[1] + phi * B[1]};   // outer: linear along the square edge
  const Real rho_out = sqrt<Real>(O[0] * O[0] + O[1] * O[1]);
  const Real r = R_foot * pow<Real>(rho_out / R_foot, t);                             // geometric grading per line
  const Real la = (r - R_foot) / (rho_out - R_foot);
  return Vec2<Real>{(1 - la) * I[0] + la * O[0], (1 - la) * I[1] + la * O[1]};
}
// Ring count sized from the longest (corner) radial line, so aspect~1 there too.
template <class Real> Integer collar_Nc(Real R_foot, Real S, Integer Naz) {
  const double pi = (double)const_pi<Real>();
  return std::max<Integer>(1, (Integer)std::ceil(std::log((double)(std::sqrt((Real)2) * S / R_foot)) / std::log(1.0 + 2 * pi / Naz)));
}
// Emit the Nc collar rings for azimuthal sector m via the shared map. ksub subdivides each ring AND
// the sector (re-sampling the true map at finer nodes) for the collar-resolution diagnostics.
template <class Real> void add_collar_sector(Vector<Real>& X, Integer order, const Mount<Real>& mnt, Real R_foot, Real S, Integer Naz, Integer Nc, Integer m, Real grade_exp = 1, Integer ksub = 1, const Real* Acorr = nullptr) {
  const Real pi = const_pi<Real>();
  const Real tha = pi / 4 + m * 2 * pi / Naz, thb = tha + 2 * pi / Naz;
  auto P = [=](Real t, Real phi) { return collar_point<Real>(R_foot, S, tha, thb, t, phi, Acorr); };
  for (Integer ia = 0; ia < ksub; ia++) {
    const Real q0 = (Real)ia / ksub, q1 = (Real)(ia + 1) / ksub;
    for (Integer ir = 0; ir < Nc * ksub; ir++) {
      const Real t0 = pow<Real>((Real)ir / (Nc * ksub), grade_exp), t1 = pow<Real>((Real)(ir + 1) / (Nc * ksub), grade_exp);
      auto Eb = [=](Real eta) { return P(t0, q0 + eta * (q1 - q0)); };
      auto Et = [=](Real eta) { return P(t1, q0 + eta * (q1 - q0)); };
      auto El = [=](Real xi)  { return P(t0 + xi * (t1 - t0), q0); };
      auto Er = [=](Real xi)  { return P(t0 + xi * (t1 - t0), q1); };
      add_collar_block<Real>(X, order, mnt, Eb, Et, El, Er);
    }
  }
}
template <class Real> void add_rev_block(Vector<Real>& X, Integer order, const Mount<Real>& mnt, const std::function<Real(Real)>& rF, const std::function<Real(Real)>& zF, Real t0, Real t1, Real tha, Real thb) {
  const Vector<Real>& nds = QuadElemList<Real>::ParamNodes(order);
  for (Integer i = 0; i < order; i++) {
    const Real t = t0 + nds[i] * (t1 - t0);
    const Real r = rF(t), z = zF(t);
    for (Integer j = 0; j < order; j++) {
      const Real th = thb + nds[j] * (tha - thb);
      const Vec3<Real> w = mnt(r * cos<Real>(th), r * sin<Real>(th), -z);
      X.PushBack(w[0]); X.PushBack(w[1]); X.PushBack(w[2]);
    }
  }
}
// (The shaft-bottom cap is the butterfly dome add_cap_butterfly below; the earlier squircle add_cap was
// removed — its 4 corners had a degenerate Jacobian, ~1e-6 discrete normal error, the cap-region limiter.)
// Butterfly-dome cap closing the shaft bottom (hemisphere of radius R_shaft; equator ring at depth
// H_shaft, pole at depth H_shaft+R_shaft). Same non-degenerate O-grid as add_disk_fill (central gnomonic
// square core [-h,h]^2 + 4 Coons arc blocks to the UNIT circle) but elevated onto the hemisphere by the
// add_cap map (psi = q*pi/2). This replaces the squircle add_cap, whose 4 corners had a degenerate
// Jacobian (~1e-6 discrete normal error). The outer ring carries Naz/4 azimuthal panels per side at
// uniform angle with the shaft's pi/4 sector offset, so the cap<->shaft seam is node-conforming: both
// resolve to mnt(R_shaft*cos th, R_shaft*sin th, H_shaft) at the same GL angles. Winding matches
// add_disk_fill / the collar (-z inward before the stud's group-flip).
template <class Real> void add_cap_butterfly(Vector<Real>& X, Integer order, const Mount<Real>& mnt, Real R_shaft, Real H_shaft, Integer Naz, Real core_frac = (Real)0.40) {
  const Real pi = const_pi<Real>();
  const Real h = core_frac;                             // core half-size in UNIT tangent-disk coords (q in [0,1])
  const Integer nc = std::max<Integer>(1, Naz / 4);     // Naz/4 azimuthal panels per cap => outer ring = Naz (shaft-conforming)
  const Vector<Real>& nds = QuadElemList<Real>::ParamNodes(order);
  // Elevate a unit tangent-disk point (Dx,Dy), q=|D| in [0,1], onto the hemisphere (add_cap map).
  auto elev = [&](Real Dx, Real Dy) -> Vec3<Real> {
    const Real q = sqrt<Real>(Dx * Dx + Dy * Dy), psi_ = q * pi / 2;
    const Real r = R_shaft * sin<Real>(psi_), depth = H_shaft + R_shaft * cos<Real>(psi_);
    const Real roq = (q > (Real)1e-9) ? r / q : R_shaft * pi / 2;
    return mnt(roq * Dx, roq * Dy, depth);
  };
  // Central gnomonic square core, nc x nc panels over [-h,h]^2 (u=y slow, v=x => -z inward).
  for (Integer ic = 0; ic < nc; ic++)
    for (Integer jc = 0; jc < nc; jc++) {
      const Real x0 = -h + 2*h*ic/nc, x1 = -h + 2*h*(ic+1)/nc, y0 = -h + 2*h*jc/nc, y1 = -h + 2*h*(jc+1)/nc;
      for (Integer i = 0; i < order; i++) { const Real yy = y0 + nds[i]*(y1-y0);
        for (Integer j = 0; j < order; j++) { const Real xx = x0 + nds[j]*(x1-x0);
          const Vec3<Real> w = elev(xx, yy); X.PushBack(w[0]); X.PushBack(w[1]); X.PushBack(w[2]); } }
    }
  // 4 caps to the UNIT circle; corners at +-pi/4 + k*pi/2 (= the shaft's pi/4 sector offset).
  for (Integer k = 0; k < 4; k++) {
    const Real rot = k * pi / 2, cr = cos<Real>(rot), sr = sin<Real>(rot);
    auto pt = [=](Real eta, Real xi) -> Vec2<Real> {       // eta: 0=core edge -> 1=unit circle; xi: along the arc
      const Real th = -pi/4 + xi * (pi/2);
      const Vec2<Real> in{h, h*(2*xi - 1)};                // core right edge in unit coords
      const Vec2<Real> out{cos<Real>(th), sin<Real>(th)};  // unit circle (q=1 -> hemisphere equator = shaft ring)
      const Real px = (1-eta)*in[0] + eta*out[0], py = (1-eta)*in[1] + eta*out[1];
      return Vec2<Real>{cr*px - sr*py, sr*px + cr*py};     // rotate into cap k
    };
    for (Integer ir = 0; ir < nc; ir++)
      for (Integer ia = 0; ia < nc; ia++) {
        const Real e0 = (Real)ir/nc, e1 = (Real)(ir+1)/nc, a0 = (Real)ia/nc, a1 = (Real)(ia+1)/nc;
        for (Integer i = 0; i < order; i++) { const Real xi = a0 + nds[i]*(a1-a0);
          for (Integer j = 0; j < order; j++) { const Real eta = e0 + nds[j]*(e1-e0);
            const Vec2<Real> P = pt(eta, xi); const Vec3<Real> w = elev(P[0], P[1]); X.PushBack(w[0]); X.PushBack(w[1]); X.PushBack(w[2]); } }
      }
  }
}
// Flat disk of radius R_disk placed ON the sphere by `mnt` (depth 0), as a 5-block BUTTERFLY / O-grid
// (central square core [-h,h]^2 + 4 arc-transition caps) => near-square cells, no squircle center
// distortion. Its outer edge is the exact R_disk circle so it is watertight with the collar's inner
// foot circle. Ndisk = panels per direction (core: Ndisk x Ndisk; each cap: Ndisk radial x Ndisk azim).
// core_frac sets the core half-size as a fraction of R_disk; 0.40 keeps the butterfly core-corner cells
// ~square (aspect = 2h/(R-sqrt2 h) ~1.8 at 0.40; 0.55 gave ~5 and was the disk-limiter) -- observed best.
template <class Real> void add_disk_fill(Vector<Real>& X, Integer order, const Mount<Real>& mnt, Real R_disk, Integer Ndisk, Real core_frac = (Real)0.40) {
  const Real pi = const_pi<Real>();
  const Real h = core_frac * R_disk;                     // core half-size (beta); ~square core+caps
  const Integer nc = std::max<Integer>(1, Ndisk);
  const Vector<Real>& nds = QuadElemList<Real>::ParamNodes(order);
  // Central square core, nc x nc bilinear panels (uniform GL nodes) -> exactly square cells.
  for (Integer ic = 0; ic < nc; ic++)
    for (Integer jc = 0; jc < nc; jc++) {
      const Real x0 = -h + 2*h*ic/nc, x1 = -h + 2*h*(ic+1)/nc, y0 = -h + 2*h*jc/nc, y1 = -h + 2*h*(jc+1)/nc;
      // u=y (slow), v=x => core normal dX/dy x dX/dx = -z, MATCHING the collar's default (-z inward)
      // and the caps below, so the single group-flip in BuildSphereWithCollarFill makes all outward.
      for (Integer i = 0; i < order; i++) { const Real yy = y0 + nds[i]*(y1-y0);
        for (Integer j = 0; j < order; j++) { const Real xx = x0 + nds[j]*(x1-x0);
          const Vec3<Real> w = mnt(xx, yy, (Real)0); X.PushBack(w[0]); X.PushBack(w[1]); X.PushBack(w[2]); } }
    }
  // 4 caps: right cap spans arc [-pi/4, pi/4] from the core right edge (x=h) to the R_disk arc; rotate
  // by k*90deg for the others. Adjacent caps share the diagonal radial edge (core corner -> arc corner)
  // => watertight; outer edge is the exact arc.
  for (Integer k = 0; k < 4; k++) {
    const Real rot = k * pi / 2, cr = cos<Real>(rot), sr = sin<Real>(rot);
    auto pt = [=](Real eta, Real xi) -> Vec2<Real> {        // eta: 0=core edge -> 1=arc; xi: along the arc
      const Real th = -pi/4 + xi * (pi/2);
      const Vec2<Real> in{h, h*(2*xi - 1)};                 // core right edge (x=h, y from -h to h)
      const Vec2<Real> out{R_disk*cos<Real>(th), R_disk*sin<Real>(th)};
      const Real px = (1-eta)*in[0] + eta*out[0], py = (1-eta)*in[1] + eta*out[1];
      return Vec2<Real>{cr*px - sr*py, sr*px + cr*py};      // rotate into cap k
    };
    for (Integer ir = 0; ir < nc; ir++)
      for (Integer ia = 0; ia < nc; ia++) {
        const Real e0 = (Real)ir/nc, e1 = (Real)(ir+1)/nc, a0 = (Real)ia/nc, a1 = (Real)(ia+1)/nc;
        // u=azimuthal (xi, slow), v=radial (eta) => cap normal azimuthal x radial = -z, matching core+collar.
        for (Integer i = 0; i < order; i++) { const Real xi = a0 + nds[i]*(a1-a0);
          for (Integer j = 0; j < order; j++) { const Real eta = e0 + nds[j]*(e1-e0);
            const Vec2<Real> P = pt(eta, xi); const Vec3<Real> w = mnt(P[0], P[1], (Real)0);
            X.PushBack(w[0]); X.PushBack(w[1]); X.PushBack(w[2]); } }
      }
  }
}
// Append one capped cilium stud placed by `mnt` (square panels: Naz azimuthal panels, a
// multiple of 4; along-shaft/fillet/cap/collar counts derived for ~square panels; collar
// graded geometrically in radius). If flip, transpose each element (swap u<->v) to negate
// the normals. Mirrors BuildCiliumConnector in test-cilia-sphere.cpp.
// Partition-of-unity blend weight for the fillet's ellipse(sphere-foot)->circle(slender) transition,
// ported from the ybifurc hybrid (which reached ~1e-10 with it). tau in [0,1]: w=1 at the foot end
// (true elliptical ring), w=0 at the slender end (exact R0 circle). fillet_pou_kind(): 1 = smootherstep
// (degree-5, represented EXACTLY by an order>=6 GL panel -> no blend representation error; the ybifurc
// default and the one that works), 0 = C-infinity bump (all derivatives vanish at the ends but too sharp
// in the interior for a modest order to resolve -> caps accuracy; kept only for comparison).
inline int& fillet_pou_kind() { static int k = 1; return k; }
template <class Real> Real fillet_pou_weight(Real tau) {
  if (tau <= (Real)0) return (Real)1;
  if (tau >= (Real)1) return (Real)0;
  if (fillet_pou_kind() == 0) {
    auto g = [](Real x) -> Real { return x > (Real)0 ? exp<Real>(-(Real)1 / x) : (Real)0; };
    const Real ga = g((Real)1 - tau), gb = g(tau); return ga / (ga + gb);
  }
  const Real t = tau, Sm = t*t*t*((Real)6*t*t - (Real)15*t + (Real)10); return (Real)1 - Sm;  // smootherstep
}

template <class Real> void add_cilium_stud(Vector<Real>& Xout, Integer order, const Mount<Real>& mnt, Real R_shaft, Real H_shaft, Real r_fil, Real S, Integer Naz, bool flip,
                                           Integer Ns_in = -1, Integer Nf_in = -1, Integer Nc_in = -1, Integer Ncap_in = -1, Real grade_exp = 1, bool with_shaft = true, bool circularize = false, Real core_frac = (Real)0.40, Integer cap_Naz = -1,
                                           Real trans_depth = 0, Integer Ns_trans = 3, Real cap_rho = -1, Real cap_a = 0) {
  const Real pi = const_pi<Real>(), R_foot = R_shaft + r_fil;
  SCTL_ASSERT(R_foot < S && H_shaft > r_fil && Naz >= 4 && Naz % 4 == 0);
  const Real az = 2 * pi * R_shaft / Naz;
  // Per-region panel counts: auto-derive for ~square panels unless the caller overrides (>=1). The collar
  // shares the shaft/cap azimuthal count Naz (coupled): it stays node-conforming with the fillet at the
  // foot circle, and its outer square edge conforms to the neighbour cubed-sphere patch.
  const Integer Ns   = (Ns_in   >= 1) ? Ns_in   : std::max<Integer>(1, (Integer)std::llround((H_shaft - r_fil) / az));
  const Integer Nf   = (Nf_in   >= 1) ? Nf_in   : std::max<Integer>(1, (Integer)std::llround((pi / 2 * r_fil) / az));
  const Integer Nc   = (Nc_in   >= 1) ? Nc_in   : collar_Nc<Real>(R_foot, S, Naz);
  (void)Ncap_in;  // cap panel count now derived from Naz by the butterfly dome (add_cap_butterfly)
  // circularize (hybrid feet): build the fillet + cap as bodies of revolution about the patch radial u,
  // sized to the mount's GNOMONIC-MEAN ring radius rho(r,depth) (~ tangent_r * local metric) and mean axial
  // station a(r,depth). This makes the fillet-bottom / cap-equator rings TRUE circles a circular CSBQ slender
  // conforms to, AND keeps the correct off-pole SIZE (the gnomonic metric < 1 shrinks the foot; using raw
  // R_shaft/R_foot with unit vectors made the collar end NARROWER than the shaft -> an inverted, translated
  // foot). The fillet MORPHS its top ring back to the collar's actual (elliptical) inner ring E so the collar
  // is left UNTOUCHED and the collar<->fillet seam is watertight; the morph vanishes at the bottom (exact
  // slender seam). At the pole the metric is 1 and rings are already circular, so this is a no-op there.
  Real P0[3], u[3], e1v[3], e2v[3];
  Real rho1 = R_foot, a1 = 0, rhoCap = R_shaft, aCap = 0;   // mean ring at collar-inner / cap-equator
  Mount<Real> mnt_cap = mnt;
  auto ring_mean = [&](const Mount<Real>& mm, Real r, Real depth, Real& rho, Real& a) {  // azimuthal-mean radius/station about u
    const Integer Nq = 64; Real sR = 0, sA = 0;   // match patch_shaft_ring (slender) so the seam circles agree
    for (Integer q = 0; q < Nq; q++) { const Real th = 2*pi*q/Nq; const Vec3<Real> P = mm(r*cos<Real>(th), r*sin<Real>(th), depth);
      const Real aa = P[0]*u[0]+P[1]*u[1]+P[2]*u[2]; const Real dx=P[0]-aa*u[0], dy=P[1]-aa*u[1], dz=P[2]-aa*u[2];
      sR += sqrt<Real>(dx*dx+dy*dy+dz*dz); sA += aa; }
    rho = sR/Nq; a = sA/Nq;
  };
  if (circularize) {
    mount_local_frame<Real>(mnt, P0, u, e1v, e2v);
    ring_mean(mnt, R_foot,  (Real)0,  rho1,   a1);
    ring_mean(mnt, R_shaft, H_shaft,  rhoCap, aCap);
    // Tie the cap equator to the slender body's DEEP-END PANEL ring (rho_bot/a_bot) when the caller
    // supplies it, so the cap<->slender seam is the SAME circle the CSBQ fiber terminates on -- not an
    // independently recomputed ring that can drift off-axis. Identical formula (Nq=64 azimuthal mean),
    // so it is a no-op numerically today, but makes the coupling explicit and robust.
    if (cap_rho > 0) { rhoCap = cap_rho; aCap = cap_a; }
    const Real u0=u[0],u1=u[1],u2=u[2], a=e1v[0],b=e1v[1],c=e1v[2], d=e2v[0],e=e2v[1],f=e2v[2], AC=aCap, HS=H_shaft;
    mnt_cap = [u0,u1,u2,a,b,c,d,e,f,AC,HS](Real Xc, Real Yc, Real dep) {   // cap dome about u, equator at station aCap
      const Real s = AC - (dep - HS); return Vec3<Real>{s*u0 + Xc*a + Yc*d, s*u1 + Xc*b + Yc*e, s*u2 + Xc*c + Yc*f}; };
  }
  const Vector<Real>& nds = QuadElemList<Real>::ParamNodes(order);
  Vector<Real> X;
  for (Integer m = 0; m < Naz; m++) {
    const Real tha = pi / 4 + m * 2 * pi / Naz, thb = tha + 2 * pi / Naz;
    auto aOf  = [=](Real t) { return pi + t * (pi / 2 - pi); };
    auto rFil = [=](Real t) { return R_foot + r_fil * cos<Real>(aOf(t)); };
    auto zFil = [=](Real t) { return -r_fil + r_fil * sin<Real>(aOf(t)); };
    if (with_shaft) {  // the hybrid base (BuildCiliumStuddedSphereBase) omits the shaft -> a CSBQ SlenderElemList replaces it
      auto rShaft = [=](Real) { return R_shaft; };
      auto zShaft = [=](Real t) { return -H_shaft + t * (H_shaft - r_fil); };
      for (Integer l = 0; l < Ns; l++) add_rev_block<Real>(X, order, mnt, rShaft, zShaft, (Real)l / Ns, (Real)(l + 1) / Ns, tha, thb);
    }
    if (!circularize) {
      for (Integer l = 0; l < Nf; l++) add_rev_block<Real>(X, order, mnt, rFil, zFil, (Real)l / Nf, (Real)(l + 1) / Nf, tha, thb);
    } else {
      // POU transition TUBE (ybifurc mechanism): blend the LOCAL true (elliptical, gnomonic) ring with the
      // LOCAL-mean circle via a smootherstep weight of physical DEPTH. w=1 at the foot (depth 0 -> true
      // ring, matches the collar exactly); w=0 at the deep end (depth eta_w -> exact circle, matches the
      // slender). The tube = rounded fillet arc [0,r_fil] + a straight cylinder extension of depth
      // trans_depth ([r_fil, r_fil+trans_depth], Ns_trans panels) so the ellipse->circle blend spans a
      // RESOLVABLE axial length (>=3 comparable-size panels), the ybifurc Ns_trans>=3 recipe -- not a
      // single micro-fillet panel. trans_depth=0 recovers the fillet-only tube. The slender attaches at
      // the tube's circular deep end (depth r_fil+trans_depth; set by BuildAllFingerSphereBase).
      const Real d_trans = r_fil + trans_depth;                               // total tube depth (foot -> slender)
      const Real f_frac  = r_fil / d_trans;                                    // fraction of tube occupied by the rounded fillet
      // g_tube in [0,1]: 0 = foot (w=1, true ring, matches collar), 1 = slender end (w=0, exact circle).
      auto pou_push = [&](Real R_prof, Real s, Real g_tube) {                  // one azimuthal ring at (R_prof, depth s)
        const Real w = fillet_pou_weight<Real>(g_tube);
        Real rhot, at; ring_mean(mnt, R_prof, s, rhot, at);                    // LOCAL mean radius + axial station
        for (Integer j = 0; j < order; j++) {
          const Real th = thb + nds[j] * (tha - thb), ct = cos<Real>(th), st = sin<Real>(th);
          const Vec3<Real> Ptrue = mnt(R_prof*ct, R_prof*st, s);              // TRUE local (elliptical) ring
          for (int k = 0; k < 3; k++) {
            const Real Pcirc = at*u[k] + rhot*(ct*e1v[k] + st*e2v[k]);         // LOCAL-mean circle at station (keeps the profile)
            X.PushBack(w*Ptrue[k] + ((Real)1 - w)*Pcirc);
          }
        }
      };
      // rounded fillet arc: t 0->1 (t=1 foot -> t=0 fillet bottom); occupies g_tube [0, f_frac].
      // t=0 (trans_depth=0 => f_frac=1) recovers g_tube=(1-t): the working fillet-only POU.
      for (Integer l = 0; l < Nf; l++) {
        const Real T0 = (Real)l / Nf, T1 = (Real)(l + 1) / Nf;
        for (Integer i = 0; i < order; i++) { const Real t = T0 + nds[i] * (T1 - T0); pou_push(rFil(t), -zFil(t), ((Real)1 - t) * f_frac); }
      }
      // straight cylinder extension: depth [r_fil, d_trans], profile R_shaft; occupies g_tube [f_frac, 1].
      // s DECREASES with i (deep->shallow) so panel orientation matches the fillet (consistent normals).
      if (trans_depth > 0) {
        for (Integer l = 0; l < Ns_trans; l++) {
          const Real S0 = r_fil + trans_depth * l / Ns_trans, S1 = r_fil + trans_depth * (l + 1) / Ns_trans;
          for (Integer i = 0; i < order; i++) {
            const Real s = S1 - nds[i] * (S1 - S0);
            const Real g_tube = f_frac + ((s - r_fil) / trans_depth) * ((Real)1 - f_frac);
            pou_push(R_shaft, s, g_tube);
          }
        }
      }
    }
    add_collar_sector<Real>(X, order, mnt, R_foot, S, Naz, Nc, m, grade_exp); // shared circle->square collar map (UNTOUCHED)
  }
  add_cap_butterfly<Real>(X, order, circularize ? mnt_cap : mnt, circularize ? rhoCap : R_shaft, H_shaft, (cap_Naz > 0 ? cap_Naz : Naz), core_frac); // dome; cap_Naz decouples cap azimuthal from the foot Naz (>0 overrides)
  const Integer cNaz = (cap_Naz > 0 ? cap_Naz : Naz);
  const Integer cap_pan = 5 * (cNaz/4) * (cNaz/4);                  // butterfly: core (cNaz/4)^2 + 4 caps (cNaz/4)^2
  const Integer Ns_rep = with_shaft ? Ns : 0;                       // shaft omitted for the hybrid base
  const Integer Nt_rep = (circularize && trans_depth > 0) ? Ns_trans : 0;   // POU transition-tube cylinder panels
  std::cout << "  stud panels: Naz=" << Naz << " cap_Naz=" << cNaz << " Ns=" << Ns_rep << " Nf=" << Nf << " Ntrans=" << Nt_rep << " Nc=" << Nc << " cap(butterfly)=" << cap_pan
            << " -> " << (Naz * (Ns_rep + Nf + Nt_rep + Nc) + cap_pan) << "\n";
  if (flip) {
    const Long nn = (Long)order * order, ne = X.Dim() / (nn * 3);
    for (Long e = 0; e < ne; e++)
      for (Integer i = 0; i < order; i++)
        for (Integer j = i + 1; j < order; j++)
          for (int c = 0; c < 3; c++) std::swap(X[(e * nn + i * order + j) * 3 + c], X[(e * nn + j * order + i) * 3 + c]);
  }
  for (Long i = 0; i < X.Dim(); i++) Xout.PushBack(X[i]);
}

// Does the stud (unflipped) have inward collar normals w.r.t. the sphere? If so we must
// flip to match the cubed sphere's outward normals.
template <class Real> bool stud_needs_flip(Integer order, Real R, Real S, Integer Naz, Real r_fil, Real R_shaft = 0.015) {
  Vector<Real> Xs;
  add_cilium_stud<Real>(Xs, order, SphereMount<Real>(R), R_shaft, (Real)0.05, r_fil, S, Naz, /*flip=*/false);
  QuadElemList<Real> stud(order, Xs);
  Vector<Real> X, Xn; stud.GetNodeCoord(&X, &Xn, nullptr);
  Real acc = 0; Long n = 0;
  for (Long i = 0; i < X.Dim() / 3; i++) {
    const Real x = X[i*3], y = X[i*3+1], z = X[i*3+2], rr = std::sqrt(x*x + y*y + z*z);
    if (rr > R - (Real)1e-6) { acc += (Xn[i*3]*x + Xn[i*3+1]*y + Xn[i*3+2]*z) / rr; n++; } // collar (on sphere)
  }
  return (n > 0 && acc < 0);
}

// Cubed sphere with the center patch of face 4 (north-pole face) replaced by a stud.
template <class Real> QuadElemList<Real> BuildCiliumStuddedSphere(Integer order, Long PatchPerFace, Real R, Integer Naz, Real r_fil,
                                                                 Integer Ns = -1, Integer Nf = -1, Integer Nc = -1, Integer Ncap = -1, Real grade_exp = 1, Real R_shaft = 0.015,
                                                                 const Comm& comm = Comm::Self(), bool with_shaft = true, bool invert_normals = false, Real H_shaft = 0.05) {
  SCTL_ASSERT_MSG(PatchPerFace % 2 == 1, "PatchPerFace must be odd so the replaced patch is centered on the pole");
  const Real S = R / (Real)PatchPerFace; // stud collar half-width = center-patch half-extent
  const bool flip = stud_needs_flip<Real>(order, R, S, Naz, r_fil, R_shaft);
  if (!comm.Rank()) std::cout << "  stud normals " << (flip ? "FLIPPED" : "kept") << " to match sphere outward"
                              << (invert_normals ? " (then whole surface INVERTED -> inward)" : "") << "\n";
  Vector<Real> X;   // built identically on every rank; the ctor slices per `comm`
  add_cubedsphere<Real>(X, order, PatchPerFace, R, /*skipFace=*/4, PatchPerFace / 2, PatchPerFace / 2);
  add_cilium_stud<Real>(X, order, SphereMount<Real>(R), R_shaft, H_shaft, r_fil, S, Naz, flip, Ns, Nf, Nc, Ncap, grade_exp, with_shaft);
  if (invert_normals) {  // transpose EVERY element (sphere + stud) -> flip all normals to point into the sphere,
    const Long nn = (Long)order * order, ne = X.Dim() / (nn * 3);  // aligning with CSBQ's native radial-OUTWARD slender normal
    for (Long e = 0; e < ne; e++)
      for (Integer i = 0; i < order; i++)
        for (Integer j = i + 1; j < order; j++)
          for (int c = 0; c < 3; c++) std::swap(X[(e * nn + i * order + j) * 3 + c], X[(e * nn + j * order + i) * 3 + c]);
  }
  return QuadElemList<Real>(order, X, comm);
}

// Watertight sphere with ONE pole patch replaced by [collar annulus: circle R_foot -> square S] +
// [inner disk fill: square -> circle R_foot], ALL on the sphere (NO finger, NO fillet, NO cap dome).
// Uses the same shared collar map (add_collar_sector) and disk (add_disk_fill) as the studded finger,
// so the whole surface is the exact radius-R sphere and the collar/disk mesh is exercised in isolation.
template <class Real> QuadElemList<Real> BuildSphereWithCollarFill(Integer order, Long PatchPerFace, Real R, Integer Naz, Real r_fil, Real grade_exp, Real R_shaft, Integer Nc_in = -1, Integer Ndisk_in = -1, Real core_frac = (Real)0.40, const Comm& comm = Comm::Self()) {
  SCTL_ASSERT_MSG(PatchPerFace % 2 == 1, "PatchPerFace must be odd so the replaced patch is centered on the pole");
  const Real pi = const_pi<Real>(), R_foot = R_shaft + r_fil, S = R / (Real)PatchPerFace, az = 2*pi*R_shaft/Naz;
  const Mount<Real> mnt = SphereMount<Real>(R);
  const Integer Nc    = (Nc_in    >= 1) ? Nc_in    : collar_Nc<Real>(R_foot, S, Naz);
  const Integer Ndisk = (Ndisk_in >= 1) ? Ndisk_in : std::max<Integer>(1, (Integer)std::llround((double)(R_foot/az)));

  // Collar + disk built into their own buffer so we can orient them outward as a group.
  Vector<Real> Xcd;
  for (Integer m = 0; m < Naz; m++) add_collar_sector<Real>(Xcd, order, mnt, R_foot, S, Naz, Nc, m, grade_exp);
  add_disk_fill<Real>(Xcd, order, mnt, R_foot, Ndisk, core_frac);

  // Orient the collar+disk group OUTWARD (n.rhat > 0) to match the cubed sphere; transpose if inward.
  // The orientation test runs on the FULL replicated buffer, so this temporary stays Comm::Self()
  // (slicing it per rank would make ranks disagree on the flip).
  { QuadElemList<Real> cd(order, Xcd); Vector<Real> Xc, Xnc; cd.GetNodeCoord(&Xc, &Xnc, nullptr);
    Real acc = 0; for (Long i = 0; i < Xc.Dim()/3; i++) { const Real x=Xc[i*3],y=Xc[i*3+1],z=Xc[i*3+2],r=std::sqrt(x*x+y*y+z*z); acc += (Xnc[i*3]*x+Xnc[i*3+1]*y+Xnc[i*3+2]*z)/r; }
    if (acc < 0) { const Long nn=(Long)order*order, ne=Xcd.Dim()/(nn*3);
      for (Long e=0;e<ne;e++) for (Integer i=0;i<order;i++) for (Integer j=i+1;j<order;j++) for (int c=0;c<3;c++) std::swap(Xcd[(e*nn+i*order+j)*3+c], Xcd[(e*nn+j*order+i)*3+c]);
      if (!comm.Rank()) std::cout << "  collar+disk normals FLIPPED to match sphere outward\n";
    } else if (!comm.Rank()) std::cout << "  collar+disk normals kept\n";
  }
  if (!comm.Rank()) std::cout << "  collar+disk panels: Naz=" << Naz << " Nc=" << Nc << " (collar) + butterfly 5*" << Ndisk << "^2 (disk) -> " << (Naz*Nc + 5*Ndisk*Ndisk) << "\n";

  Vector<Real> X;   // full mesh built on every rank; ctor keeps this rank's slice
  add_cubedsphere<Real>(X, order, PatchPerFace, R, /*skipFace=*/4, PatchPerFace / 2, PatchPerFace / 2);
  for (Long i = 0; i < Xcd.Dim(); i++) X.PushBack(Xcd[i]);
  return QuadElemList<Real>(order, X, comm);
}

// Sphere tiled with the collarfill pattern in EVERY cubed-sphere patch (collar circle→square + butterfly
// disk), instead of just one. All patch outer boundaries are the same square-edge map at the same Naz, so
// every patch↔patch seam has matched node density (within a face exactly; across cube faces same-density,
// like the plain cubed sphere). Tests whether the single-patch collarfill accuracy holds when the whole
// surface is collar-tiled — and, at higher Naz, whether a uniformly finer outer rim (no longer a mismatch
// against plain neighbours) recovers the floor.
template <class Real> QuadElemList<Real> BuildAllCollarFillSphere(Integer order, Long PatchPerFace, Real R, Integer Naz, Real r_fil, Real grade_exp, Real R_shaft, Integer Nc_in = -1, Integer Ndisk_in = -1, Real core_frac = (Real)0.40, bool with_finger = false, bool circularize = false, const Comm& comm = Comm::Self()) {
  const Real pi = const_pi<Real>(), R_foot = R_shaft + r_fil, S = R / (Real)PatchPerFace, az = 2*pi*R_shaft/Naz;
  const Integer Nc    = (Nc_in    >= 1) ? Nc_in    : collar_Nc<Real>(R_foot, S, Naz);
  const Integer Ndisk = (Ndisk_in >= 1) ? Ndisk_in : std::max<Integer>(1, (Integer)std::llround((double)(R_foot/az)));
  const Long nn = (Long)order*order;
  Vector<Real> Xall;
  Long nflip = 0;
  for (Integer face = 0; face < 6; face++)
    for (Long iu = 0; iu < PatchPerFace; iu++)
      for (Long iv = 0; iv < PatchPerFace; iv++) {
        const Real a_c = 2*(iu + (Real)0.5)/PatchPerFace - 1, b_c = 2*(iv + (Real)0.5)/PatchPerFace - 1;
        const Mount<Real> mnt = PatchMount<Real>(face, a_c, b_c, R);
        // Optionally replace ONE patch (face-4 centre, like the studded sphere) with a full cilium finger.
        // Its collar shares Naz with the surrounding all-collar patches, so the finger's outer seam is
        // node-conforming against them (no 2:1 plain-neighbour mismatch).
        // const bool is_finger = with_finger && face == 4 && iu == PatchPerFace/2 && iv == PatchPerFace/2;
        const bool is_finger = with_finger;
        Vector<Real> Xp;
        if (is_finger) {
          add_cilium_stud<Real>(Xp, order, mnt, R_shaft, (Real)0.05, r_fil, S, Naz, /*flip=*/false,
                                /*Ns*/-1, /*Nf*/-1, /*Nc*/Nc, /*Ncap*/-1, grade_exp, /*with_shaft*/true, circularize, core_frac);
        } else {
          for (Integer m = 0; m < Naz; m++) add_collar_sector<Real>(Xp, order, mnt, R_foot, S, Naz, Nc, m, grade_exp);
          add_disk_fill<Real>(Xp, order, mnt, R_foot, Ndisk, core_frac);
        }
        // Orient this patch's collar+disk group OUTWARD (n·rhat > 0); transpose (swap u<->v) if inward.
        { QuadElemList<Real> pc(order, Xp); Vector<Real> Xc, Xnc; pc.GetNodeCoord(&Xc, &Xnc, nullptr);
          Real acc = 0; for (Long i = 0; i < Xc.Dim()/3; i++) { const Real x=Xc[i*3],y=Xc[i*3+1],z=Xc[i*3+2],r=std::sqrt(x*x+y*y+z*z); acc += (Xnc[i*3]*x+Xnc[i*3+1]*y+Xnc[i*3+2]*z)/r; }
          if (acc < 0) { const Long ne = Xp.Dim()/(nn*3); nflip++;
            for (Long e=0;e<ne;e++) for (Integer i=0;i<order;i++) for (Integer j=i+1;j<order;j++) for (int c=0;c<3;c++) std::swap(Xp[(e*nn+i*order+j)*3+c], Xp[(e*nn+j*order+i)*3+c]); }
        }
        for (Long i = 0; i < Xp.Dim(); i++) Xall.PushBack(Xp[i]);
      }
  const Long npatch = 6*PatchPerFace*PatchPerFace, per = Naz*Nc + 5*Ndisk*Ndisk;
  if (!comm.Rank()) std::cout << "  all-collarfill: " << npatch << " patches x (collar Naz=" << Naz << "*Nc=" << Nc << " + disk 5*" << Ndisk << "^2 = "
            << per << ") -> " << (npatch*per) << " panels; " << nflip << " patch-groups flipped outward\n";
  // Xall is built identically on every rank; the ctor keeps only this rank's element slice.
  return QuadElemList<Real>(order, Xall, comm);
}

} // namespace quad_junctions
