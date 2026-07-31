#pragma once
/**
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
        if ((skipFace < 0 || face == skipFace) && iu == skipIu && iv == skipIv) continue;  // skipFace<0 => skip this (iu,iv) on EVERY face
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

// PATCH-RELATIVE cilium sizing (single source of truth). A cilium is made self-similar to its patch by
// tying the shaft radius to the patch half-width S: R_shaft = frac*S (frac 0.25 => the thin cilium, a
// slender collar with ~2 rings at Naz=8, INDEPENDENT of patch count). r_fil and the (sphere) shaft depth
// H_shaft scale with R_shaft so the finger keeps constant aspect at any density. Carpet H_reach is
// box-driven and NOT set here. Drivers pass a positive absolute R_shaft to override.
template <class Real> struct CiliumScale { Real R_shaft, r_fil, H_shaft; };
template <class Real> CiliumScale<Real> cilium_scale_from_patch(Real S, Real frac = (Real)0.25,
                                                                Real rfil_k = (Real)0.1, Real hshaft_k = (Real)3) {
  const Real Rs = frac * S;
  return CiliumScale<Real>{Rs, rfil_k * Rs, hshaft_k * Rs};
}
// ---- NEW on-sphere exact-circle collar (2026-07-24) ----------------------------------------------
// The legacy collar (collar_point above) builds the annulus in TANGENT-plane 2D coords and applies the
// gnomonic `mnt(X,Y,0)` once, so an inner circle of radius R_foot becomes a world ELLIPSE off-axis (the
// ~3e-6 foot-seam floor). The new scheme defines the inner foot ring DIRECTLY in world space as an exact
// small circle about the patch axis u, keeps the outer square edge unchanged (node-conforming), and keeps
// the whole collar on the sphere. See stud_sphere_geom_legacy.hpp for the frozen original.
template <class Real> using Curve3 = std::function<Vec3<Real>(Real)>;
// World-space transfinite (Coons) blend of four boundary curves (component-wise).
template <class Real> Vec3<Real> coons3(const Curve3<Real>& Eb, const Curve3<Real>& Et, const Curve3<Real>& El, const Curve3<Real>& Er, Real xi, Real eta) {
  const Vec3<Real> b = Eb(xi), t = Et(xi), l = El(eta), r = Er(eta);
  const Vec3<Real> c00 = Eb(0), c10 = Eb(1), c01 = Et(0), c11 = Et(1);
  Vec3<Real> P;
  for (int k = 0; k < 3; k++)
    P[k] = (1 - eta) * b[k] + eta * t[k] + (1 - xi) * l[k] + xi * r[k]
         - ((1 - xi) * (1 - eta) * c00[k] + xi * (1 - eta) * c10[k] + (1 - xi) * eta * c01[k] + xi * eta * c11[k]);
  return P;
}
// Emit an order x order block from four WORLD edge curves (Coons-blended) then PROJECT each node onto the
// sphere of radius Rsph. Shared edges are exact edge curves (already on-sphere => projection is identity
// there), so adjacent blocks stay watertight.
template <class Real> void add_collar_block3(Vector<Real>& X, Integer order, Real Rsph, const Curve3<Real>& Eb, const Curve3<Real>& Et, const Curve3<Real>& El, const Curve3<Real>& Er) {
  const Vector<Real>& nds = QuadElemList<Real>::ParamNodes(order);
  for (Integer i = 0; i < order; i++)
    for (Integer j = 0; j < order; j++) {
      Vec3<Real> P = coons3<Real>(Eb, Et, El, Er, nds[i], nds[j]);
      const Real n = sqrt<Real>(P[0]*P[0] + P[1]*P[1] + P[2]*P[2]);
      X.PushBack(Rsph * P[0] / n); X.PushBack(Rsph * P[1] / n); X.PushBack(Rsph * P[2] / n);
    }
}
// Invert mnt(X,Y,0) = W for the tangent coords (X,Y). Gauss-Newton from (0,0); the target is the tiny foot
// circle near the patch centre so the mount is near-linear there and this converges in a few steps. Works
// for any smooth mount (SphereMount pole or off-axis PatchMount) -- no face/(a_c,b_c) plumbing needed.
template <class Real> Vec2<Real> mnt_inverse(const Mount<Real>& mnt, const Vec3<Real>& W) {
  const Real h = (Real)1e-5;
  Real x = 0, y = 0;
  for (int it = 0; it < 8; it++) {
    const Vec3<Real> F = mnt(x, y, (Real)0), Fx = mnt(x + h, y, (Real)0), Fy = mnt(x, y + h, (Real)0);
    Real c0[3], c1[3], r[3];
    for (int k = 0; k < 3; k++) { c0[k] = (Fx[k]-F[k])/h; c1[k] = (Fy[k]-F[k])/h; r[k] = W[k]-F[k]; }
    const Real a = c0[0]*c0[0]+c0[1]*c0[1]+c0[2]*c0[2], b = c0[0]*c1[0]+c0[1]*c1[1]+c0[2]*c1[2], d = c1[0]*c1[0]+c1[1]*c1[1]+c1[2]*c1[2];
    const Real g0 = c0[0]*r[0]+c0[1]*r[1]+c0[2]*r[2], g1 = c1[0]*r[0]+c1[1]*r[1]+c1[2]*r[2];
    const Real det = a*d - b*b; if (std::fabs((double)det) < 1e-300) break;
    const Real dx = (d*g0 - b*g1)/det, dy = (a*g1 - b*g0)/det;
    x += dx; y += dy;
    if (dx*dx + dy*dy < (Real)1e-28) break;
  }
  return Vec2<Real>{x, y};
}
// Collar map in the TANGENT plane (2D): inner = mnt-preimage of the exact small circle a0*u+R0*circle
// (a0=sqrt(Rsph^2-R0^2), on the sphere); outer = linear along the gnomonic square edge (node-conforming);
// geometric per-line grading, blended in 2D. add_collar_block then applies mnt to the whole block, giving an
// accurate gnomonic mesh whose inner ring comes out EXACTLY circular on- and off-axis.
template <class Real> Vec2<Real> collar_point_2d(const Mount<Real>& mnt, Real R0, Real a0, Real S,
    const Real u[3], const Real e1[3], const Real e2[3], Real tha, Real thb, Real t, Real phi) {
  auto sq = [&](Real th) { const Real c = cos<Real>(th), s = sin<Real>(th), m = std::max(std::fabs(c), std::fabs(s)); return Vec2<Real>{S * c / m, S * s / m}; };
  const Real th = tha + phi * (thb - tha);
  const Vec3<Real> Wc{a0*u[0] + R0*(cos<Real>(th)*e1[0] + sin<Real>(th)*e2[0]),
                      a0*u[1] + R0*(cos<Real>(th)*e1[1] + sin<Real>(th)*e2[1]),
                      a0*u[2] + R0*(cos<Real>(th)*e1[2] + sin<Real>(th)*e2[2])};
  const Vec2<Real> I = mnt_inverse<Real>(mnt, Wc);                                   // inner: preimage of the exact circle
  const Vec2<Real> A = sq(tha), B = sq(thb);
  const Vec2<Real> O{(1 - phi) * A[0] + phi * B[0], (1 - phi) * A[1] + phi * B[1]};  // outer: linear along square edge
  const Real rho_in = sqrt<Real>(I[0]*I[0] + I[1]*I[1]), rho_out = sqrt<Real>(O[0]*O[0] + O[1]*O[1]);
  const Real r = rho_in * pow<Real>(rho_out / rho_in, t);
  const Real la = (rho_out > rho_in) ? (r - rho_in) / (rho_out - rho_in) : t;
  return Vec2<Real>{(1 - la) * I[0] + la * O[0], (1 - la) * I[1] + la * O[1]};
}

// ================= Regular partition-of-unity collar map (2026-07-26) =============================
// Overhauls the off-axis MOUNTING of the collar annulus. Instead of the pure gnomonic 2D-blend-then-mount
// (whose off-axis interior skews near the foot), the collar map is a REGULAR (radial-only) partition-of-
// unity blend, on the sphere, of two constructions (see collar_world):
//   * ISOTROPIC interior -- an EXACT concentric circle about the patch axis u (uniform angle, graded
//     radius). Keeps the near-foot region circular with no gnomonic skew ("interior isotropic").
//   * ANISOTROPIC exterior -- the gnomonic conforming map whose outer ring is the node-conforming square
//     edge ("exterior anisotropic, conforming").
// Blend weight = a plain radial smootherstep (NOT corner-aware -- a corner-aware variant and an iterative
// Winslow/metric-Laplacian smoother were prototyped and dropped: on this thin graded annulus the iterative
// smoother near-folded the inner rings and blew up the near-singular BIE quadrature; the analytic blend is
// C-infinity, watertight, and fold-free by construction). See stud_sphere_geom_legacy2.hpp for the frozen
// pre-overhaul (2D-preimage) collar (the QJ_COLLAR_ENABLE=0 A/B baseline).
//
// Watertightness: emission is via add_collar_block3/coons3 whose four edge curves are slices of the ONE
// deterministic map collar_world(t,s) (plus EXACT analytic inner/outer boundary rings). Adjacent elements
// evaluate that same map at identical shared-boundary (t,s), so shared 3D edges are bit-identical ->
// watertight exactly (the GL nodes are OPEN, so watertightness is a shared-EDGE-CURVE property, not a
// shared-node one).

// Config (settable via env; mirrors the fillet_pou_kind() static-accessor pattern). Only `enable` remains:
// 1 (default) = regular PoU collar; 0 = the legacy 2D-preimage collar (the A/B baseline).
struct CollarPouCfg { int enable = 1; };
inline CollarPouCfg& collar_pou_cfg() {
  static CollarPouCfg c = [] { CollarPouCfg d;
    if (const char* e = std::getenv("QJ_COLLAR_ENABLE")) d.enable = atoi(e);
    return d; }();
  return c;
}
// degree-5 smootherstep (C2 at both ends, order-exact for order>=6); S5(0)=0, S5(1)=1.
template <class Real> Real collar_s5(Real x) {
  if (x <= (Real)0) return (Real)0;
  if (x >= (Real)1) return (Real)1;
  return x*x*x*((Real)6*x*x - (Real)15*x + (Real)10);
}

// Collar field = just the (u,e1,e2) frame + sphere radius + collar params. The collar map is now an
// ANALYTIC regular partition-of-unity blend (no grid, no iterative solve) -- see collar_world.
template <class Real> struct CollarField {
  Real u[3], e1[3], e2[3], Rsph, R0, a0, grade, S;
  Mount<Real> mnt;                                            // gnomonic mount (for the anisotropic conforming part)
  Integer Naz, Nc, order;
};
// REGULAR (non-corner-aware) partition-of-unity collar map, analytic and on the sphere. t in [0,1] radial
// (0 inner foot circle, 1 outer square seam), s in [0,1) global azimuth. Blends, with a plain radial
// smootherstep weight phi(t):
//   * P_iso  -- ISOTROPIC interior: an EXACT concentric circle about the patch axis u at uniform angle and
//     graded perpendicular radius. This keeps the near-foot region circular with NO gnomonic skew (the
//     source of the off-axis foot error) -- "interior isotropic, circle stays circular".
//   * P_ani  -- ANISOTROPIC exterior: the gnomonic conforming map (collar_point_2d then mnt), whose outer
//     ring is the node-conforming square edge -- "exterior anisotropic, conforming".
// phi(0)=0 => pure isotropic circle at the foot; phi(1)=1 => pure conforming square at the seam; C-infinity
// and watertight/fold-free by construction (a convex blend of two smooth on-sphere maps, then projected).
// enable=0 returns P_ani alone = the legacy 2D-preimage collar (the A/B baseline).
template <class Real> Vec3<Real> collar_world(const CollarField<Real>& F, Real t, Real s) {
  const Real pi = const_pi<Real>();
  s = s - std::floor((double)s);
  Integer m = (Integer)std::floor((double)(s*F.Naz)); if (m >= F.Naz) m = F.Naz-1;
  Real phi_s = s*F.Naz - m; if (phi_s < 0) phi_s = 0; if (phi_s > 1) phi_s = 1;
  const Real tha = pi/4 + m*2*pi/F.Naz, thb = tha + 2*pi/F.Naz;
  // anisotropic conforming part (world)
  const Vec2<Real> P2 = collar_point_2d<Real>(F.mnt, F.R0, F.a0, F.S, F.u, F.e1, F.e2, tha, thb, t, phi_s);
  const Vec3<Real> P_ani = F.mnt(P2[0], P2[1], (Real)0);
  auto proj = [&](Vec3<Real> W) { const Real nn = sqrt<Real>(W[0]*W[0]+W[1]*W[1]+W[2]*W[2]); return Vec3<Real>{F.Rsph*W[0]/nn, F.Rsph*W[1]/nn, F.Rsph*W[2]/nn}; };
  if (!collar_pou_cfg().enable) return proj(P_ani);           // baseline
  // isotropic interior part: exact concentric circle, uniform angle, geometric radius R0 -> S
  const Real th = pi/4 + 2*pi*s;
  const Real Rref = std::max<Real>(F.S, F.R0*(Real)1.0001);
  Real rho = F.R0 * pow<Real>(Rref/F.R0, t); rho = std::min<Real>(rho, (Real)0.999*F.Rsph);
  const Real aiso = sqrt<Real>(std::max<Real>(F.Rsph*F.Rsph - rho*rho, (Real)0));
  const Vec3<Real> P_iso{aiso*F.u[0] + rho*(cos<Real>(th)*F.e1[0] + sin<Real>(th)*F.e2[0]),
                         aiso*F.u[1] + rho*(cos<Real>(th)*F.e1[1] + sin<Real>(th)*F.e2[1]),
                         aiso*F.u[2] + rho*(cos<Real>(th)*F.e1[2] + sin<Real>(th)*F.e2[2])};
  const Real w = collar_s5<Real>(t);                          // regular radial PoU weight
  return proj(Vec3<Real>{(1-w)*P_iso[0]+w*P_ani[0], (1-w)*P_iso[1]+w*P_ani[1], (1-w)*P_iso[2]+w*P_ani[2]});
}
// Build the collar field (frame + params). No iterative solve -- the map is analytic (collar_world).
template <class Real> CollarField<Real> build_collar_field(const Mount<Real>& mnt, Real R_foot, Real S,
    Integer Naz, Integer Nc, Integer order, Real grade) {
  CollarField<Real> F;
  Real P0[3]; mount_local_frame<Real>(mnt, P0, F.u, F.e1, F.e2);
  F.Rsph = sqrt<Real>(P0[0]*P0[0]+P0[1]*P0[1]+P0[2]*P0[2]); F.R0 = R_foot; F.a0 = sqrt<Real>(F.Rsph*F.Rsph - F.R0*F.R0);
  F.Naz = Naz; F.Nc = Nc; F.order = order; F.grade = grade; F.S = S; F.mnt = mnt;
  return F;
}
// Emit sector m's Nc collar rings from the smoothed field, via add_collar_block3 (world Coons + project).
// Same edge layout / winding as the legacy add_collar_sector (u-slow=azimuth, v-fast=radial), so the
// existing collar+disk group-orientation flip in the builders applies unchanged.
//
// The two FIXED boundary rings are emitted from EXACT analytic curves (NOT the Catmull-Rom interpolant),
// so the cross-patch outer seam and the collar<->disk inner seam stay watertight to machine precision:
//   * inner (ir=0, t=0): the exact small circle a0*u + R0*(cos e1 + sin e2)  -> matches the disk fill.
//   * outer (ir=Nc-1, t=1): mnt(linear-along-square-edge) -> node-conforming with the neighbour patch,
//     byte-identical to the legacy construction.
// All interior ring boundaries + the sector-boundary radial edges come from collar_world (their shared
// copies are bit-identical between neighbours because it is one deterministic map; the interpolation error
// there is internal to the collar and cancels).
template <class Real> void emit_collar_sector(Vector<Real>& X, Integer order, const CollarField<Real>& F, Integer m) {
  const Real pi = const_pi<Real>();
  const Integer Nc = F.Nc; const Real grade = F.grade, s0 = (Real)m/F.Naz, s1 = (Real)(m+1)/F.Naz;
  auto W = [&](Real t, Real s) { return collar_world<Real>(F, t, s); };
  auto inner_exact = [&](Real s) {                            // exact small circle on the sphere
    const Real th = pi/4 + 2*pi*s;
    return Vec3<Real>{F.a0*F.u[0] + F.R0*(cos<Real>(th)*F.e1[0] + sin<Real>(th)*F.e2[0]),
                      F.a0*F.u[1] + F.R0*(cos<Real>(th)*F.e1[1] + sin<Real>(th)*F.e2[1]),
                      F.a0*F.u[2] + F.R0*(cos<Real>(th)*F.e1[2] + sin<Real>(th)*F.e2[2])}; };
  auto outer_exact = [&](Real s) {                            // exact linear-along-square-edge, node-conforming
    Integer mm = (Integer)std::floor((double)(s*F.Naz)); if (mm >= F.Naz) mm = F.Naz-1;
    Real phi = s*F.Naz - mm; if (phi < 0) phi = 0; if (phi > 1) phi = 1;
    const Real tha = pi/4 + mm*2*pi/F.Naz, thb = tha + 2*pi/F.Naz;
    auto sq = [&](Real th) { const Real c = cos<Real>(th), s2 = sin<Real>(th), mx = std::max(std::fabs(c), std::fabs(s2)); return Vec2<Real>{F.S*c/mx, F.S*s2/mx}; };
    const Vec2<Real> A = sq(tha), B = sq(thb); const Vec2<Real> O{(1-phi)*A[0]+phi*B[0], (1-phi)*A[1]+phi*B[1]};
    return F.mnt(O[0], O[1], (Real)0); };
  for (Integer ir = 0; ir < Nc; ir++) {
    const Real t0 = pow<Real>((Real)ir/Nc, grade), t1 = pow<Real>((Real)(ir+1)/Nc, grade);  // geometric radial grading
    const bool inner = (ir == 0), outer = (ir == Nc-1);
    auto Eb = [=](Real eta) { const Real s = s0 + eta*(s1-s0); return inner ? inner_exact(s) : W(t0, s); };
    auto Et = [=](Real eta) { const Real s = s0 + eta*(s1-s0); return outer ? outer_exact(s) : W(t1, s); };
    auto El = [=](Real xi)  { return W(t0 + xi*(t1-t0), s0); };
    auto Er = [=](Real xi)  { return W(t0 + xi*(t1-t0), s1); };
    add_collar_block3<Real>(X, order, F.Rsph, Eb, Et, El, Er);
  }
}

// Emit the Nc collar rings for azimuthal sector m via the new on-sphere map. The (u,e1,e2) frame + sphere
// radius are derived from `mnt` (works for SphereMount pole AND any off-axis PatchMount). ksub subdivides
// each ring AND the sector for the collar-resolution diagnostics.
template <class Real> void add_collar_sector(Vector<Real>& X, Integer order, const Mount<Real>& mnt, Real R_foot, Real S, Integer Naz, Integer Nc, Integer m, Real grade_exp = 1, Integer ksub = 1) {
  const Real pi = const_pi<Real>();
  Real P0[3], u[3], e1[3], e2[3]; mount_local_frame<Real>(mnt, P0, u, e1, e2);
  const Real Rsph = sqrt<Real>(P0[0]*P0[0] + P0[1]*P0[1] + P0[2]*P0[2]);
  const Real R0 = R_foot, a0 = sqrt<Real>(Rsph*Rsph - R0*R0);
  (void)Rsph;
  const Real tha = pi / 4 + m * 2 * pi / Naz, thb = tha + 2 * pi / Naz;
  auto P = [=](Real t, Real phi) { return collar_point_2d<Real>(mnt, R0, a0, S, u, e1, e2, tha, thb, t, phi); };
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
// Butterfly-dome cap at an ARBITRARY tip: a hemisphere of radius rho_tip whose EQUATOR is the circle of
// radius rho_tip perpendicular to the travel tangent Ttip, centered at the tip point Ctip, in the frame
// (w1,w2); the pole bulges to Ctip + rho_tip*Ttip. Generalizes the axis-aligned mnt_cap in add_cilium_stud
// (recovered by Ctip=aCap*u, Ttip=-u, (w1,w2)=(e1,e2)). The H_ref offset cancels because the butterfly's
// depth enters only as (depth - H_ref) = rho*cos(psi). Used to cap the end of a curved (flagella) slender
// shaft; the equator conforms to the slender's tip ring (separate lists -> geometric coincidence suffices).
template <class Real> void add_tip_cap_butterfly(Vector<Real>& X, Integer order,
    const Real Ctip[3], const Real Ttip[3], const Real w1[3], const Real w2[3],
    Real rho_tip, Integer Naz, Real core_frac = (Real)0.40) {
  const Real Cx=Ctip[0],Cy=Ctip[1],Cz=Ctip[2], t0=Ttip[0],t1=Ttip[1],t2=Ttip[2];
  const Real a0=w1[0],a1=w1[1],a2=w1[2], b0=w2[0],b1=w2[1],b2=w2[2], HR=rho_tip;  // H_ref cancels
  Mount<Real> mnt_cap = [=](Real Xc, Real Yc, Real dep) {
    const Real g = dep - HR;   // bulge along Ttip: equator (dep=HR) -> 0, pole (dep=HR+rho_tip) -> rho_tip
    return Vec3<Real>{ Cx + Xc*a0 + Yc*b0 + g*t0, Cy + Xc*a1 + Yc*b1 + g*t1, Cz + Xc*a2 + Yc*b2 + g*t2 }; };
  add_cap_butterfly<Real>(X, order, mnt_cap, rho_tip, /*H_shaft=*/HR, Naz, core_frac);
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
  // Place butterfly (P,Q) coords ON the sphere via the u-frame so the outer ring (|(P,Q)|=R_disk=R_foot)
  // is a0*u + R_foot*circle = the NEW collar inner ring (exact-circle match), and the interior stays on the
  // sphere (a=sqrt(R^2-P^2-Q^2)). Replaces the old gnomonic mnt(P,Q,0) whose outer ring was elliptical.
  Real P0f[3], uf[3], e1f[3], e2f[3]; mount_local_frame<Real>(mnt, P0f, uf, e1f, e2f);
  const Real Rsph = sqrt<Real>(P0f[0]*P0f[0] + P0f[1]*P0f[1] + P0f[2]*P0f[2]);
  auto place = [&](Real Px, Real Qy) -> Vec3<Real> {
    const Real aa = sqrt<Real>(std::max<Real>((Real)0, Rsph*Rsph - Px*Px - Qy*Qy));
    return Vec3<Real>{aa*uf[0] + Px*e1f[0] + Qy*e2f[0], aa*uf[1] + Px*e1f[1] + Qy*e2f[1], aa*uf[2] + Px*e1f[2] + Qy*e2f[2]}; };
  // Central square core, nc x nc bilinear panels (uniform GL nodes) -> exactly square cells.
  for (Integer ic = 0; ic < nc; ic++)
    for (Integer jc = 0; jc < nc; jc++) {
      const Real x0 = -h + 2*h*ic/nc, x1 = -h + 2*h*(ic+1)/nc, y0 = -h + 2*h*jc/nc, y1 = -h + 2*h*(jc+1)/nc;
      // u=y (slow), v=x => core normal dX/dy x dX/dx = -z, MATCHING the collar's default (-z inward)
      // and the caps below, so the single group-flip in BuildSphereWithCollarFill makes all outward.
      for (Integer i = 0; i < order; i++) { const Real yy = y0 + nds[i]*(y1-y0);
        for (Integer j = 0; j < order; j++) { const Real xx = x0 + nds[j]*(x1-x0);
          const Vec3<Real> w = place(xx, yy); X.PushBack(w[0]); X.PushBack(w[1]); X.PushBack(w[2]); } }
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
            const Vec2<Real> P = pt(eta, xi); const Vec3<Real> w = place(P[0], P[1]);
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
                                           Real trans_depth = 0, Integer Ns_trans = 3, Real cap_rho = -1, Real cap_a = 0, bool with_cap = true) {
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
  // EXACT-circle foot (2026-07-24, replaces the old `circularize` ellipse->circle morph + gnomonic fillet
  // + POU transition tube). Build the shaft, fillet and cap as EXACT bodies of revolution about the patch
  // axis u (from mount_local_frame), anchored at station a0 = sqrt(R^2 - R_foot^2) so the fillet-TOP ring
  // == the collar inner ring (a0*u + R_foot*circle, ON the sphere), and the fillet-bottom / shaft / cap
  // rings are exact circles of radius R_shaft. No morph, no ring_mean, no Acorr: circles are exact by
  // construction on- AND off-axis. circularize/trans_depth/Ns_trans are now inert (kept for API/CLI compat).
  (void)circularize; (void)trans_depth; (void)Ns_trans;
  Real P0[3], u[3], e1v[3], e2v[3]; mount_local_frame<Real>(mnt, P0, u, e1v, e2v);
  const Real Rsph = sqrt<Real>(P0[0]*P0[0] + P0[1]*P0[1] + P0[2]*P0[2]);
  const Real R0 = R_foot, a0 = sqrt<Real>(Rsph*Rsph - R0*R0);
  // Fillet meridian arc (in the (radius, station-along-u) plane), TANGENT to the sphere at the foot and
  // TANGENT to the vertical shaft at the bottom -- otherwise a ~R0/R crease at the foot wrecks the
  // near-quadrature (base ~1.75e-5). Circular arc of radius rho_arc = r_fil/(1+R0/R) whose centre lies on
  // the inward sphere normal at the foot F=(R0,a0): C=(Cr,Cs)=F*(1-rho_arc/R). It sweeps from the foot
  // (angle phi0 = atan2(a0,R0), radius R0 station a0, tangent = sphere) to the shaft (angle pi, radius
  // R_shaft station Cs, tangent = vertical). The shaft-top / slender-foot station is a_top = Cs.
  const Real rho_arc = r_fil / (1 + R0/Rsph);
  const Real Cr = R0 * (1 - rho_arc/Rsph), Cs = a0 * (1 - rho_arc/Rsph);
  const Real phi0 = std::atan2((double)a0, (double)R0);   // foot-normal polar angle in the meridian plane
  // cap equator ring: exact (radius R_shaft, station a0-H_shaft) unless the caller pins it (hybrid slender end).
  Real rhoCap = R_shaft, aCap = a0 - H_shaft;
  if (cap_rho > 0) { rhoCap = cap_rho; aCap = cap_a; }
  const Real u0=u[0],u1=u[1],u2=u[2], a_=e1v[0],b_=e1v[1],c_=e1v[2], d_=e2v[0],e_=e2v[1],f_=e2v[2], AC=aCap, HS=H_shaft;
  Mount<Real> mnt_cap = [u0,u1,u2,a_,b_,c_,d_,e_,f_,AC,HS](Real Xc, Real Yc, Real dep) {   // cap dome about u, equator at station aCap
    const Real s = AC - (dep - HS); return Vec3<Real>{s*u0 + Xc*a_ + Yc*d_, s*u1 + Xc*b_ + Yc*e_, s*u2 + Xc*c_ + Yc*f_}; };
  const Vector<Real>& nds = QuadElemList<Real>::ParamNodes(order);
  Vector<Real> X;
  // Exact revolution about u of a profile given by (radius rF(t), station sF(t)); world = sF*u + rF*circle.
  auto rev_frame = [&](const std::function<Real(Real)>& rF, const std::function<Real(Real)>& sF, Real t0, Real t1, Real tha, Real thb) {
    for (Integer i = 0; i < order; i++) {
      const Real t = t0 + nds[i] * (t1 - t0), r = rF(t), s = sF(t);
      for (Integer j = 0; j < order; j++) {
        const Real th = thb + nds[j] * (tha - thb), ct = cos<Real>(th), st = sin<Real>(th);
        X.PushBack(s*u0 + r*(ct*a_ + st*d_)); X.PushBack(s*u1 + r*(ct*b_ + st*e_)); X.PushBack(s*u2 + r*(ct*c_ + st*f_));
      }
    }
  };
  // Fillet tangent arc: t=1 foot (angle phi0, radius R0, station a0) -> t=0 shaft top (angle pi, radius R_shaft, station Cs).
  auto angF  = [=](Real t) { return phi0 + (1 - t) * (pi - phi0); };
  auto rFil  = [=](Real t) { return Cr + rho_arc * cos<Real>(angF(t)); };  // R0(t=1) -> R_shaft(t=0)
  auto sFil  = [=](Real t) { return Cs + rho_arc * sin<Real>(angF(t)); };  // a0(t=1) -> Cs(t=0)
  const Real a_shaft_top = Cs;                                             // fillet-bottom / shaft-top station
  // Whole-collar corner-aware PoU + Winslow smoother, built ONCE (periodic azimuthally). Its fixed inner
  // ring == the fillet top (a0*u + R_foot*circle), so the fillet<->collar seam is unchanged/watertight.
  const CollarField<Real> CF = build_collar_field<Real>(mnt, R_foot, S, Naz, Nc, order, grade_exp);
  for (Integer m = 0; m < Naz; m++) {
    const Real tha = pi / 4 + m * 2 * pi / Naz, thb = tha + 2 * pi / Naz;
    if (with_shaft) {  // the hybrid base (BuildCiliumStuddedSphereBase) omits the shaft -> a CSBQ SlenderElemList replaces it
      auto rShaft = [=](Real) { return R_shaft; };
      auto sShaft = [=](Real t) { return aCap + t * (a_shaft_top - aCap); };  // cap station(t=0) -> fillet-bottom(t=1)
      for (Integer l = 0; l < Ns; l++) rev_frame(rShaft, sShaft, (Real)l / Ns, (Real)(l + 1) / Ns, tha, thb);
    }
    for (Integer l = 0; l < Nf; l++) rev_frame(rFil, sFil, (Real)l / Nf, (Real)(l + 1) / Nf, tha, thb);   // tangent fillet revolution
    emit_collar_sector<Real>(X, order, CF, m);                             // corner-aware PoU+Winslow collar (inner ring == fillet top)
  }
  if (with_cap) add_cap_butterfly<Real>(X, order, mnt_cap, rhoCap, H_shaft, (cap_Naz > 0 ? cap_Naz : Naz), core_frac); // dome anchored to the exact cap ring
  const Integer cNaz = (cap_Naz > 0 ? cap_Naz : Naz);
  const Integer cap_pan = with_cap ? 5 * (cNaz/4) * (cNaz/4) : 0;   // butterfly: core (cNaz/4)^2 + 4 caps (cNaz/4)^2 (0 if the cap is placed externally at a curved tip)
  const Integer Ns_rep = with_shaft ? Ns : 0;                       // shaft omitted for the hybrid base
  std::cout << "  stud panels: Naz=" << Naz << " cap_Naz=" << cNaz << " Ns=" << Ns_rep << " Nf=" << Nf << " Nc=" << Nc << " cap(butterfly)=" << cap_pan
            << " -> " << (Naz * (Ns_rep + Nf + Nc) + cap_pan) << "\n";
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
template <class Real> bool stud_needs_flip(Integer order, Real R, Real S, Integer Naz, Real r_fil, Real R_shaft = 0.015, Real H_shaft = 0.05) {
  Vector<Real> Xs;
  add_cilium_stud<Real>(Xs, order, SphereMount<Real>(R), R_shaft, H_shaft, r_fil, S, Naz, /*flip=*/false);
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
  const bool flip = stud_needs_flip<Real>(order, R, S, Naz, r_fil, R_shaft, H_shaft);
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
  const CollarField<Real> CF = build_collar_field<Real>(mnt, R_foot, S, Naz, Nc, order, grade_exp);
  for (Integer m = 0; m < Naz; m++) emit_collar_sector<Real>(Xcd, order, CF, m);
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
template <class Real> QuadElemList<Real> BuildAllCollarFillSphere(Integer order, Long PatchPerFace, Real R, Integer Naz, Real r_fil, Real grade_exp, Real R_shaft, Integer Nc_in = -1, Integer Ndisk_in = -1, Real core_frac = (Real)0.40, bool with_finger = false, bool circularize = false, const Comm& comm = Comm::Self(), Real H_shaft = (Real)0.05) {
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
          add_cilium_stud<Real>(Xp, order, mnt, R_shaft, H_shaft, r_fil, S, Naz, /*flip=*/false,
                                /*Ns*/-1, /*Nf*/-1, /*Nc*/Nc, /*Ncap*/-1, grade_exp, /*with_shaft*/true, circularize, core_frac);
        } else {
          const CollarField<Real> CF = build_collar_field<Real>(mnt, R_foot, S, Naz, Nc, order, grade_exp);
          for (Integer m = 0; m < Naz; m++) emit_collar_sector<Real>(Xp, order, CF, m);
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
