/**
 * Stud/collar-on-sphere BIE driver (cilia mount migrated from SCTL_quad_element/src/test-gmsh-geom.cpp).
 * Runs the double-layer constant-density identity (= -1/2) and Green's identity (Laplace + Stokes),
 * on-surface and off-surface, for the collarfill / all-collar / plain-sphere geometries + a manufactured BVP.
 *
 * csbq.hpp is included to keep the hybrid include path (CSBQ SlenderElemList + the fork's enhanced
 * QuadElemList) compiling in this project; SlenderElemList is used from a later milestone.
 *
 *   make bin/stud_sphere-bie
 *   OMP_NUM_THREADS=8 ./bin/stud_sphere-bie <mode> [tol Nbeta max_depth R_shaft Nc Naz order Ndisk core_frac*100 trg_dist PatchPerFace]
 * modes (a keyword is REQUIRED): sphere | collarfill | allcollar | greenoff | manufactured
 *   (the single-finger studded-sphere modes -- default finger, dump, collarsrc -- and allcollarfinger were
 *    removed 2026-07-31; the single studded finger now lives only in the hybrid driver's centerfinger mode.)
 *   manufactured: exterior + interior Stokes BVPs on the all-finger cilia sphere from a point-source
 *     Stokeslet (Dirichlet BC -> CFIE via GMRES -> near/far shells vs the exact field). Defaults ppf=1;
 *     trg_dist is reused as the near-shell offset. Exterior: source inside, shells at R+off & 2R. Interior:
 *     source outside, shells at R-off & 0.5R (clear of the inward cilia pits). Default off=0.1.
 */
#include <csbq.hpp>
#include <stokes_bio.hpp>                     // StokesBIO: combined-field Stokes op, PVFMM-safe FMM kernels
#include <quad_junctions/fmm_kernels.hpp>     // SetPVFMMKer for the identity/Green's operators
#include <quad_junctions/stud_sphere_geom.hpp>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>

using namespace sctl;
using namespace quad_junctions;

namespace {
// Collar-only surface: the annular sphere patch between the foot circle (R_foot) and the square
// half-width S, with each nominal panel subdivided ksub x ksub. The subdivision RE-SAMPLES the
// true Coons(circle->square) o gnomonic map at finer nodes (not a re-interpolation of a coarse
// panel), and splits BOTH directions equally so sub-panels stay proportioned (no radial slivers,
// unlike bumping Nc). Used to isolate the collar's DL contribution at off-collar (sphere) targets
// and watch it converge as its geometry is resolved. Mirrors the collar block in add_cilium_stud.
template <class Real> QuadElemList<Real> BuildCollarOnly(Integer order, Real R_foot, Real S, Integer Naz, Integer Nc, Real grade_exp, Real R, Integer ksub, bool flip, const Comm& comm = Comm::Self()) {
  const Mount<Real> mnt = SphereMount<Real>(R);
  Vector<Real> X;
  for (Integer m = 0; m < Naz; m++) add_collar_sector<Real>(X, order, mnt, R_foot, S, Naz, Nc, m, grade_exp, ksub); // shared map, ksub-resampled
  if (flip) {
    const Long nn = (Long)order*order, ne = X.Dim()/(nn*3);
    for (Long e = 0; e < ne; e++)
      for (Integer i = 0; i < order; i++)
        for (Integer j = i+1; j < order; j++)
          for (int c = 0; c < 3; c++) std::swap(X[(e*nn+i*order+j)*3+c], X[(e*nn+j*order+i)*3+c]);
  }
  return QuadElemList<Real>(order, X, comm);   // X replicated on every rank; ctor keeps this rank's slice
}

// Tangent-alignment check: the collar/cubed-sphere nodes lie exactly on the sphere, but the
// surface is an order-N polynomial interpolant. If that interpolant's parametric tangents
// dX/du, dX/dv tilt OFF the sphere's tangent plane (a nonzero radial component), the normal
// dX/du x dX/dv is wrong even though positions are exact -- and that normal error feeds the DL
// kernel. Reports, per region, max |t . rhat|/|t| (tangent tilt, rhat=X/|X|) and 1-|n . rhat|
// (normal deviation from radial). Geometry-only (no BIE solve).

// (removed 2026-07-31: diagnose_collar_source -- the one-finger collar-source DL diagnostic)

// =====================================================================================================
// ACCURACY TESTS + DRIVERS (DL-identity and Green's-identity; the test_* drivers below build a mesh via
// the production builders above and run these checks). test_DLIdentity/test_greens_* copied from
// test-quad-elem.cpp.
// =====================================================================================================
template <class Real, class KerDL> void test_DLIdentity(const QuadElemList<Real>& elem_lst, const Comm& comm, const Real quad_tol = 1e-8, const std::string& dumpfile = "") {
  const KerDL kernel_dl;
  BoundaryIntegralOp<Real, KerDL> BIOp(kernel_dl, false, comm);
  SetPVFMMKer(BIOp);
  BIOp.SetAccuracy(quad_tol);
  BIOp.AddElemList(elem_lst);
  const Real c_expect = -0.5; // outward surface normals
  const Long KDIM0 = KerDL::SrcDim();
  Vector<Real> X, Xn;
  elem_lst.GetNodeCoord(&X, &Xn, nullptr);
  const Long Nnode = X.Dim() / 3;
  Vector<Real> q(Nnode * KDIM0), U;
  for (Long i = 0; i < Nnode; i++) for (Long k = 0; k < KDIM0; k++) q[i*KDIM0 + k] = k + 1;
  BIOp.ComputePotential(U, q);
  Vector<Real> cx_maxerr(KDIM0); cx_maxerr = 0.;
  Long argmax_i = 0; Real emax = 0, emax_finger = 0, emax_surf = 0; // localization: where does the error live?
  for (Long i = 0; i < Nnode; i++) {
    const Real x = X[i*3], y = X[i*3+1], z = X[i*3+2], rr = std::sqrt(x*x + y*y + z*z);
    for (Long k = 0; k < KDIM0; k++) {
      const Real e = std::fabs(U[i*KDIM0+k] / q[i*KDIM0+k] - c_expect);
      cx_maxerr[k] = std::max(cx_maxerr[k], e);
      if (e > emax) { emax = e; argmax_i = i; }
      if (rr < (Real)0.999) emax_finger = std::max(emax_finger, e); else emax_surf = std::max(emax_surf, e); // finger pokes inward (r<1); collar+sphere on r~1
    }
  }
  // Reduce across ranks: per-component maxima (for avg), the finger/surface split, and the argmax loc.
  Vector<double> cxd(KDIM0); for (Long k = 0; k < KDIM0; k++) cxd[k] = (double)cx_maxerr[k];
  GlobalReduce(cxd, comm, CommOp::MAX);
  emax_finger = GlobalReduce((double)emax_finger, comm, CommOp::MAX);
  emax_surf   = GlobalReduce((double)emax_surf,   comm, CommOp::MAX);
  Real avg = 0.; for (Long k = 0; k < KDIM0; k++) avg += cxd[k] / std::fabs(c_expect);
  avg /= KDIM0;
  const double loc_xyz[3] = { (double)(Nnode?X[argmax_i*3]:0), (double)(Nnode?X[argmax_i*3+1]:0), (double)(Nnode?X[argmax_i*3+2]:0) };
  double gloc[3]; const double gemax = GlobalMaxLoc((double)emax, loc_xyz, comm, gloc);
  const Real ax = gloc[0], ay = gloc[1], az = gloc[2];
  if (!comm.Rank()) {
    std::cout << std::setprecision(8) << "  DL constant-density identity: max relative error = " << avg << std::endl;
    std::cout << "    [loc] argmax node |err|=" << gemax << " at r=" << std::sqrt(ax*ax+ay*ay+az*az)
              << " rho=" << std::hypot(ax, ay) << " z=" << az
              << "  |  max err: finger(r<0.999)=" << emax_finger << "  surface(r>=0.999)=" << emax_surf << std::endl;
  }
  if (!dumpfile.empty()) { // per-node error map: x,y,z,rho,r,err (max over kernel components)
    // Serial-only: each rank owns only a node slice, so a single flat CSV would be partial/clobbered.
    if (comm.Size() == 1) {
      std::ofstream fout(dumpfile);
      fout << "x,y,z,rho,r,err\n";
      for (Long i = 0; i < Nnode; i++) {
        const Real x = X[i*3], y = X[i*3+1], z = X[i*3+2];
        Real e = 0; for (Long k = 0; k < KDIM0; k++) e = std::max(e, std::fabs(U[i*KDIM0+k] / q[i*KDIM0+k] - c_expect));
        fout << std::setprecision(10) << x << "," << y << "," << z << "," << std::hypot(x, y) << "," << std::sqrt(x*x+y*y+z*z) << "," << e << "\n";
      }
      std::cout << "    [dump] wrote per-node DL error map -> " << dumpfile << std::endl;
    } else if (!comm.Rank()) std::cout << "    [dump] per-node DL CSV skipped under MPI (use the .vtu)\n";
  }
}

template <class Real, class KerSL, class KerDL, class KerGrad> void test_greens_identity(const QuadElemList<Real>& elem_lst, const Comm& comm, const Real tol, const Vector<Real> X0) {
  static constexpr Integer COORD_DIM = 3;
  const Long pid = comm.Rank();
  KerSL kernel_sl; KerDL kernel_dl; KerGrad kernel_grad;
  BoundaryIntegralOp<Real,KerSL> BIOpSL(kernel_sl, false, comm);
  BoundaryIntegralOp<Real,KerDL> BIOpDL(kernel_dl, false, comm);
  SetPVFMMKer(BIOpSL); SetPVFMMKer(BIOpDL);
  BIOpSL.AddElemList(elem_lst); BIOpDL.AddElemList(elem_lst);
  BIOpSL.SetAccuracy(tol); BIOpDL.SetAccuracy(tol);
  Vector<Real> X, Xn, Fs, Fd, Uref, Us, Ud;
  elem_lst.GetNodeCoord(&X, &Xn, nullptr);
  {
    Vector<Real> Xn0{0,0,0}, F0(KerSL::SrcDim()), dU;
    for (auto& x : F0) x = drand48() - 0.5;
    kernel_sl.Eval(Uref, X, X0, Xn0, F0);
    kernel_grad.Eval(dU, X, X0, Xn0, F0);
    Fd = Uref;
    constexpr Integer KDIM0 = KerSL::SrcDim();
    const Long N = X.Dim()/COORD_DIM;
    Fs.ReInit(N * KDIM0);
    for (Long i = 0; i < N; i++) for (Integer j = 0; j < KDIM0; j++) {
      Real d = 0; for (Long k = 0; k < COORD_DIM; k++) d += dU[(i*KDIM0+j)*COORD_DIM+k] * Xn[i*COORD_DIM+k];
      Fs[i*KDIM0+j] = d;
    }
  }
  BIOpSL.ComputePotential(Us,Fs); BIOpDL.ComputePotential(Ud,Fd);
  Ud -= 0.5*Fd;
  Vector<Real> Uerr = (Us - Ud) - Uref;
  StaticArray<Real,2> max_err{0,0}, max_val{0,0};
  for (auto x : Uerr) max_err[0] = std::max<Real>(max_err[0], fabs(x));
  for (auto x : Uref) max_val[0] = std::max<Real>(max_val[0], fabs(x));
  comm.Allreduce(max_err+0, max_err+1, 1, CommOp::MAX);
  comm.Allreduce(max_val+0, max_val+1, 1, CommOp::MAX);
  if (!pid) std::cout << "  Green's identity error = " << max_err[1]/max_val[1] << '\n';

  // Save err distri to file
  Vector<Real> Urelerr = Uerr;
  for (auto x : Urelerr) x = fabs(x) / max_val[1];
  elem_lst.WriteVTK("CiliumErrGreens", Urelerr, comm);
}

// OFF-surface Green's identity: targets are placed a fixed distance `trg_dist` INSIDE the surface
// (X - trg_dist * n, n outward), so the interior representation u = SL[du/dn] - DL[u] holds with NO
// jump term. For small trg_dist this is a pure near-singular-quadrature test of the adaptive near
// evaluation. Reports the relative error distribution: max, and split near-pole (collar/disk) vs the
// rest of the sphere; optionally dumps per-target (target xyz, surface rho/z, r, err) to CSV.
template <class Real, class KerSL, class KerDL, class KerGrad> void test_greens_offsurface(const QuadElemList<Real>& elem_lst, const Comm& comm, const Real tol, const Vector<Real> X0, const Real trg_dist, const Real S, const std::string& dumpfile = "") {
  static constexpr Integer COORD_DIM = 3;
  KerSL kernel_sl; KerDL kernel_dl; KerGrad kernel_grad;
  Vector<Real> X, Xn;
  elem_lst.GetNodeCoord(&X, &Xn, nullptr);
  const Long N = X.Dim()/COORD_DIM;

  // Targets: a distance trg_dist inside the surface along the (outward) normal.
  Vector<Real> Xt(N*COORD_DIM);
  for (Long i = 0; i < N; i++) for (Integer k = 0; k < COORD_DIM; k++) Xt[i*COORD_DIM+k] = X[i*COORD_DIM+k] - trg_dist * Xn[i*COORD_DIM+k];

  // Surface densities from an exterior point source X0: Fd = u|_surf (DL density), Fs = du/dn (SL density).
  constexpr Integer KDIM0 = KerSL::SrcDim();
  Vector<Real> Xn0{0,0,0}, F0(KDIM0), dU, UrefSurf, Fs, Fd, Uref_t, Us, Ud;
  for (auto& x : F0) x = drand48() - 0.5;
  kernel_sl.Eval(UrefSurf, X, X0, Xn0, F0);         // u at surface nodes -> DL density
  kernel_grad.Eval(dU, X, X0, Xn0, F0);             // grad u at surface nodes
  Fd = UrefSurf;
  Fs.ReInit(N * KDIM0);
  for (Long i = 0; i < N; i++) for (Integer j = 0; j < KDIM0; j++) {
    Real d = 0; for (Integer k = 0; k < COORD_DIM; k++) d += dU[(i*KDIM0+j)*COORD_DIM+k] * Xn[i*COORD_DIM+k];
    Fs[i*KDIM0+j] = d;
  }
  kernel_sl.Eval(Uref_t, Xt, X0, Xn0, F0);          // reference field AT the off-surface targets

  // Off-surface potentials at Xt (near-singular quadrature via the adaptive near path).
  BoundaryIntegralOp<Real,KerSL> BIOpSL(kernel_sl, false, comm);
  BoundaryIntegralOp<Real,KerDL> BIOpDL(kernel_dl, false, comm);
  SetPVFMMKer(BIOpSL); SetPVFMMKer(BIOpDL);
  BIOpSL.AddElemList(elem_lst); BIOpDL.AddElemList(elem_lst);
  BIOpSL.SetAccuracy(tol); BIOpDL.SetAccuracy(tol);
  BIOpSL.SetTargetCoord(Xt); BIOpDL.SetTargetCoord(Xt);
  BIOpSL.ComputePotential(Us, Fs); BIOpDL.ComputePotential(Ud, Fd);

  const Integer KDIM1 = KerSL::TrgDim();
  Vector<Real> Uerr = (Us - Ud) - Uref_t;
  Real max_val = 0; for (auto v : Uref_t) max_val = std::max<Real>(max_val, std::fabs(v));
  max_val = GlobalReduce((double)max_val, comm, CommOp::MAX);   // normalize by the GLOBAL max reference value

  // Distribution: per-target error (max over components), localized by the SURFACE node's rho/z.
  // The flat CSV is serial-only (each rank owns a target slice); the .vtu is not written here.
  Real emax = 0, emax_pole = 0, emax_rest = 0; Long argmax = 0;
  std::ofstream fout; if (!dumpfile.empty() && comm.Size() == 1) { fout.open(dumpfile); fout << "xt,yt,zt,rho_surf,z_surf,r_trg,err\n"; }
  for (Long i = 0; i < N; i++) {
    Real e = 0; for (Integer k = 0; k < KDIM1; k++) e = std::max<Real>(e, std::fabs(Uerr[i*KDIM1+k]));
    e /= max_val;
    if (e > emax) { emax = e; argmax = i; }
    const Real rho = std::hypot(X[i*3], X[i*3+1]), zs = X[i*3+2];
    const bool pole = (zs > (Real)0.9 && rho < (Real)1.3 * S);   // collar+disk footprint at the north pole
    if (pole) emax_pole = std::max(emax_pole, e); else emax_rest = std::max(emax_rest, e);
    if (fout.is_open()) fout << std::setprecision(10) << Xt[i*3] << "," << Xt[i*3+1] << "," << Xt[i*3+2] << ","
                             << rho << "," << zs << "," << std::sqrt(Xt[i*3]*Xt[i*3]+Xt[i*3+1]*Xt[i*3+1]+Xt[i*3+2]*Xt[i*3+2]) << "," << e << "\n";
  }
  emax_pole = GlobalReduce((double)emax_pole, comm, CommOp::MAX);
  emax_rest = GlobalReduce((double)emax_rest, comm, CommOp::MAX);
  const double loc_xyz[3] = { (double)(N?X[argmax*3]:0), (double)(N?X[argmax*3+1]:0), (double)(N?X[argmax*3+2]:0) };
  double gloc[3]; const double gemax = GlobalMaxLoc((double)emax, loc_xyz, comm, gloc);
  const Real ax = gloc[0], ay = gloc[1], az = gloc[2];
  if (!comm.Rank()) {
    std::cout << std::setprecision(6) << "  off-surface Green's (trg_dist=" << trg_dist << ") max rel error = " << gemax
              << "  |  near-pole(collar/disk)=" << emax_pole << "  rest-of-sphere=" << emax_rest << "\n";
    std::cout << "    [loc] argmax surface node rho=" << std::hypot(ax,ay) << " z=" << az
              << " (" << (az > 0.9 && std::hypot(ax,ay) < 1.3*S ? "collar/disk" : "sphere") << ")\n";
    if (comm.Size() == 1 && !dumpfile.empty()) std::cout << "    [dump] wrote per-target off-surface error map -> " << dumpfile << "\n";
    else if (!dumpfile.empty()) std::cout << "    [dump] per-target CSV skipped under MPI\n";
  }
}

// (removed 2026-07-31: test_CiliumStuddedSphere -- the single studded-finger DL/Green's test)

// Control: the SAME cubed sphere with NO finger (nothing skipped/replaced). Establishes the
// achievable identity floor for this discretization+quad settings, isolating the finger's cost.
template <class Real> void test_PlainSphere(const Comm& comm, Integer order, Long PatchPerFace, Real R, Real tol,
                                            Integer Nbeta = 400, Integer max_depth = 30, Integer cov_q = 10) {
  if (!comm.Rank())
    std::cout << "\n=== Control: plain cubed sphere " << PatchPerFace << "x" << PatchPerFace
              << "/face, NO finger (order=" << order << " tol=" << tol << " Nbeta=" << Nbeta
              << " max_depth=" << max_depth << " cov_q=" << cov_q << ") ===\n";
  Vector<Real> X;
  add_cubedsphere<Real>(X, order, PatchPerFace, R, /*skipFace=*/-1, 0, 0); // skipFace=-1 => skip nothing
  QuadElemList<Real> elem_lst(order, X, comm);   // X replicated per rank; ctor keeps this rank's slice
  report_area<Real>(elem_lst, comm);
  elem_lst.SetQuadScheme(QuadElemList<Real>::QuadScheme::Hybrid, cov_q, Nbeta, max_depth);
  const Vector<Real> X0{1.3, 1.2, 0.2};
  if (!comm.Rank()) { std::cout << "[Laplace] "; }test_DLIdentity<Real, Laplace3D_DxU>(elem_lst, comm, tol);
  if (!comm.Rank()) { std::cout << "[Stokes]  "; }test_DLIdentity<Real, Stokes3D_DxU>(elem_lst, comm, tol);
  if (!comm.Rank()) { std::cout << "[Laplace] "; }test_greens_identity<Real, Laplace3D_FxU, Laplace3D_DxU, Laplace3D_FxdU>(elem_lst, comm, tol, X0);
  if (!comm.Rank()) { std::cout << "[Stokes]  "; }test_greens_identity<Real, Stokes3D_FxU, Stokes3D_DxU, Stokes3D_FxT>(elem_lst, comm, tol, X0);
}

// Mesh-isolation driver: watertight sphere whose pole patch is the collar (circle->square) + inner disk
// (square->circle), NO finger. The whole surface is the exact sphere, so DL identity must be -1/2 and
// Green's identity ~machine precision -- this isolates the collar/disk MESH accuracy from any finger
// near-field. Best config: Naz=4 (conforming outer seam), core_frac=0.40.
template <class Real> void test_SphereCollarFill(const Comm& comm, Integer order, Long PatchPerFace, Real R, Real tol,
                                                 Integer Naz, Real r_fil, Real grade_exp, Integer Nbeta, Integer max_depth, Integer cov_q, Real R_shaft,
                                                 Integer Nc_in = -1, Integer Ndisk_in = -1, Real core_frac = (Real)0.40) {
  if (!comm.Rank())
    std::cout << "\n=== Mesh-isolation: sphere with collar(circle->square)+disk(square->circle) pole patch, NO finger (order="
              << order << " Naz=" << Naz << " r_fil=" << r_fil << " R_shaft=" << R_shaft << " tol=" << tol
              << " Nbeta=" << Nbeta << " max_depth=" << max_depth << " cov_q=" << cov_q << " Nc=" << Nc_in << " Ndisk=" << Ndisk_in
              << " core_frac=" << core_frac << ") ===\n";
  QuadElemList<Real> elem_lst = BuildSphereWithCollarFill<Real>(order, PatchPerFace, R, Naz, r_fil, grade_exp, R_shaft, Nc_in, Ndisk_in, core_frac, comm);
  report_area<Real>(elem_lst, comm); // watertightness: |int n dA| should be ~machine-zero for the closed sphere
  elem_lst.SetQuadScheme(QuadElemList<Real>::QuadScheme::Hybrid, cov_q, Nbeta, max_depth);
  elem_lst.WriteVTK("sphere-collarfill", Vector<Real>(), comm);
  const Vector<Real> X0{1.3, 1.2, 0.2};
  if (!comm.Rank()) { std::cout << "[Laplace] "; }test_DLIdentity<Real, Laplace3D_DxU>(elem_lst, comm, tol);
  if (!comm.Rank()) { std::cout << "[Stokes]  "; }test_DLIdentity<Real, Stokes3D_DxU>(elem_lst, comm, tol);
  if (!comm.Rank()) { std::cout << "[Laplace] "; }test_greens_identity<Real, Laplace3D_FxU, Laplace3D_DxU, Laplace3D_FxdU>(elem_lst, comm, tol, X0);
  if (!comm.Rank()) { std::cout << "[Stokes]  "; }test_greens_identity<Real, Stokes3D_FxU, Stokes3D_DxU, Stokes3D_FxT>(elem_lst, comm, tol, X0);
}

// Off-surface Green's identity on the collarfill sphere: targets a distance trg_dist inside the surface
// (near-singular near-eval stress test), with per-target error-distribution dumps localizing collar/disk
// vs the rest of the sphere.
template <class Real> void test_SphereCollarFillOffSurface(const Comm& comm, Integer order, Long PatchPerFace, Real R, Real tol,
                                                           Integer Naz, Real r_fil, Real grade_exp, Integer Nbeta, Integer max_depth, Integer cov_q, Real R_shaft, Real trg_dist,
                                                           Integer Nc_in = -1, Integer Ndisk_in = -1, Real core_frac = (Real)0.40) {
  const Real S = R / (Real)PatchPerFace;
  if (!comm.Rank())
    std::cout << "\n=== OFF-surface Green's identity on collarfill sphere (trg_dist=" << trg_dist << " order=" << order
              << " Naz=" << Naz << " tol=" << tol << " Nbeta=" << Nbeta << " max_depth=" << max_depth << " core_frac=" << core_frac << ") ===\n";
  QuadElemList<Real> elem_lst = BuildSphereWithCollarFill<Real>(order, PatchPerFace, R, Naz, r_fil, grade_exp, R_shaft, Nc_in, Ndisk_in, core_frac, comm);
  report_area<Real>(elem_lst, comm);
  elem_lst.SetQuadScheme(QuadElemList<Real>::QuadScheme::Hybrid, cov_q, Nbeta, max_depth);
  const Vector<Real> X0{1.3, 1.2, 0.2}; // exterior point source -> interior representation at the targets
  if (!comm.Rank()) { std::cout << "[Laplace] "; }test_greens_offsurface<Real, Laplace3D_FxU, Laplace3D_DxU, Laplace3D_FxdU>(elem_lst, comm, tol, X0, trg_dist, S, "greenoff_Laplace.csv");
  if (!comm.Rank()) { std::cout << "[Stokes]  "; }test_greens_offsurface<Real, Stokes3D_FxU, Stokes3D_DxU, Stokes3D_FxT>(elem_lst, comm, tol, X0, trg_dist, S, "greenoff_Stokes.csv");
}

// All-collarfill sphere driver: on-surface DL + Green's + off-surface Green's, so it can be compared
// directly to the single-patch collarfill numbers. Whole sphere tiled with collar+disk patches.
template <class Real> void test_AllCollarFillSphere(const Comm& comm, Integer order, Long PatchPerFace, Real R, Real tol,
                                                    Integer Naz, Real r_fil, Real grade_exp, Integer Nbeta, Integer max_depth, Integer cov_q, Real R_shaft, Real trg_dist,
                                                    Integer Nc_in = -1, Integer Ndisk_in = -1, Real core_frac = (Real)0.40, bool with_finger = false, bool circularize = false) {
  const Real S = R / (Real)PatchPerFace;
  if (!comm.Rank())
    std::cout << "\n=== ALL-collarfill sphere: EVERY cubed-sphere patch = " << (with_finger ? "cilium FINGER" : "collar+disk")
              << " (PatchPerFace=" << PatchPerFace
              << " order=" << order << " Naz=" << Naz << " tol=" << tol << " Nbeta=" << Nbeta << " max_depth=" << max_depth
              << " r_fil=" << r_fil << " grade_exp=" << grade_exp << " circularize=" << circularize
              << " core_frac=" << core_frac << " trg_dist=" << trg_dist << ") ===\n";
  QuadElemList<Real> elem_lst = BuildAllCollarFillSphere<Real>(order, PatchPerFace, R, Naz, r_fil, grade_exp, R_shaft, Nc_in, Ndisk_in, core_frac, with_finger, circularize, comm);
  report_area<Real>(elem_lst, comm);
  elem_lst.SetQuadScheme(QuadElemList<Real>::QuadScheme::Hybrid, cov_q, Nbeta, max_depth);
  const Vector<Real> X0{1.3, 1.2, 0.2};
  const std::string dlerr = std::getenv("STUD_DUMP_ERR") ? "allcollar_dlerr_Laplace.csv" : "";  // per-node DL error map (serial)
  if (!comm.Rank()) { std::cout << "[Laplace] "; }test_DLIdentity<Real, Laplace3D_DxU>(elem_lst, comm, tol, dlerr);
  if (!comm.Rank()) { std::cout << "[Stokes]  "; }test_DLIdentity<Real, Stokes3D_DxU>(elem_lst, comm, tol);
  if (!comm.Rank()) { std::cout << "[Laplace] "; }test_greens_identity<Real, Laplace3D_FxU, Laplace3D_DxU, Laplace3D_FxdU>(elem_lst, comm, tol, X0);
  if (!comm.Rank()) { std::cout << "[Stokes]  "; }test_greens_identity<Real, Stokes3D_FxU, Stokes3D_DxU, Stokes3D_FxT>(elem_lst, comm, tol, X0);
  if (!comm.Rank()) { std::cout << "[Laplace] "; }test_greens_offsurface<Real, Laplace3D_FxU, Laplace3D_DxU, Laplace3D_FxdU>(elem_lst, comm, tol, X0, trg_dist, S, "allcollar_off_Laplace.csv");
  if (!comm.Rank()) { std::cout << "[Stokes]  "; }test_greens_offsurface<Real, Stokes3D_FxU, Stokes3D_DxU, Stokes3D_FxT>(elem_lst, comm, tol, X0, trg_dist, S, "allcollar_off_Stokes.csv");
}

// Manufactured-solution Stokes BVPs on the all-finger cilia sphere (EXTERIOR + INTERIOR), on one built
// mesh. A set of Stokeslets placed on the side OPPOSITE the solution domain generates an exact field
// there; its Dirichlet trace on the surface is the RHS. Solve the combined-field BIE
// (jump*I + SL_scal*S + DL_scal*D) sigma = u|_surface via GMRES, then evaluate the represented flow on two
// target shells and report rel-L2 vs the exact Stokeslet field. Targets are the surface nodes radially
// projected onto a fixed radius, so:
//   - EXTERIOR: sources inside the solid (near origin); shells at r > R (r=R+near_off, r=2R) -> exterior
//     fluid, clear of the inward finger cavities (no targets inside the shafts).
//   - INTERIOR: sources outside the surface (|x|>R); shells at r < R (r=R-near_off, r=0.5R). The pits
//     occupy only r in [~0.935, R] near the 6 patch axes, so any interior shell at r < 0.9 is in the solid
//     core, clear of every cilium (no targets inside the shafts).
// Interior CFIE with same-sign SL/DL has an artificial null space -> SL sign is flipped (SL_scal=-1),
// jump = -1/2; exterior uses SL_scal=+1, jump = +1/2 (outward normals). Mirrors TestManufactured /
// test_StokesManufactured (both branches) in ../SCTL_quad_element/src/test-quad-elem.cpp.
template <class Real> void test_ManufacturedStokes(const Comm& comm, Integer order, Long PatchPerFace, Real R, Real tol,
                                                   Integer Naz, Real r_fil, Real grade_exp, Integer Nbeta, Integer max_depth, Integer cov_q,
                                                   Real R_shaft, Real near_off, Integer Nc_in = -1, Real core_frac = (Real)0.40, bool circularize = false) {
  using KerSL = Stokes3D_FxU;   // Stokeslet single-layer (also the exact-field kernel)
  if (!comm.Rank())
    std::cout << "\n=== MANUFACTURED Stokes BVPs (exterior + interior): all-finger cilia sphere (point-source"
              << " Stokeslet -> Dirichlet BC -> CFIE via GMRES -> shell of off-domain targets)\n    (PatchPerFace="
              << PatchPerFace << " order=" << order << " Naz=" << Naz << " tol=" << tol << " Nbeta=" << Nbeta
              << " max_depth=" << max_depth << " cov_q=" << cov_q << " R_shaft=" << R_shaft << " r_fil=" << r_fil
              << " near_off=" << near_off << ") ===\n";

  QuadElemList<Real> elem_lst = BuildAllCollarFillSphere<Real>(order, PatchPerFace, R, Naz, r_fil, grade_exp, R_shaft, Nc_in, -1, core_frac, /*with_finger=*/true, circularize, comm);
  report_area<Real>(elem_lst, comm);
  elem_lst.SetQuadScheme(QuadElemList<Real>::QuadScheme::Hybrid, cov_q, Nbeta, max_depth);

  // Surface nodes (local slice under MPI). Normals are outward -> jump = +1/2 (ext) / -1/2 (int).
  Vector<Real> Xs, Xns;
  elem_lst.GetNodeCoord(&Xs, &Xns, nullptr);
  const Long Nnode = Xs.Dim() / 3;
  KerSL ker_sl;   // exact-field kernel; the operator itself is assembled by StokesBIO below

  const Vector<Real> Fsrc{(Real)1.0, (Real)0.5, (Real)-0.3,  (Real)-1.0, (Real)-0.5, (Real)0.3};

  // One combined-field solve + two-shell evaluation, parametrized by domain (interior/exterior).
  auto run = [&](bool interior, const Vector<Real>& Xsrc, Real Rn, Real Rf, const char* label) {
    const Real SL_scal = interior ? (Real)-1 : (Real)1;  // interior: opposite-sign SL/DL (no null space)
    const Real DL_scal = 1;
    const Real jump    = (interior ? (Real)-0.5 : (Real)0.5) * DL_scal;  // outward-normal orientation

    // Dirichlet data: Stokeslet velocity at the surface nodes (SL kernel ignores the src-normal arg).
    Vector<Real> bc;
    ker_sl.Eval(bc, Xs, Xsrc, Xsrc, Fsrc);

    // StokesBIO holds the SL and DL operators together (U = SL_scal*S + DL_scal*D) and installs the
    // PVFMM-safe translation kernels in its ctor -- the plain BoundaryIntegralOp default would register
    // the DL kernel for M2M/M2L/L2L and corrupt the heap under PVFMM.
    StokesBIO<Real> Op(SL_scal, DL_scal, comm);
    Op.SetAccuracy(tol);
    Op.AddElemList(elem_lst);

    const auto ApplyK = [&](Vector<Real>* U, const Vector<Real>& sigma) {
      Vector<Real> Uc;
      Op.ComputePotential(Uc, sigma);
      if (U->Dim() != sigma.Dim()) U->ReInit(sigma.Dim());
      (*U) = Uc + jump*sigma;
    };

    GMRES<Real> solver(comm);
    KrylovPrecond<Real> krylov;
    Vector<Real> sigma;
    Long iter = 0;
    solver(&sigma, ApplyK, bc, tol * (Real)10, /*max_iter=*/400, false, &iter, &krylov);
    if (!comm.Rank()) std::cout << "  [" << label << "] GMRES iters = " << iter << std::endl;

    // Evaluate the recovered flow on two shells (surface nodes radially projected to a fixed radius);
    // compare to the exact Stokeslet field.
    const Real Rt_list[2] = { Rn, Rf };
    const char* nm[2] = { "near", "far" };
    for (int s = 0; s < 2; s++) {
      const Real Rt = Rt_list[s];
      Vector<Real> Xtrg(Nnode * 3);
      for (Long i = 0; i < Nnode; i++) {
        const Real x = Xs[i*3], y = Xs[i*3+1], z = Xs[i*3+2], rr = std::sqrt(x*x + y*y + z*z);
        const Real sc = (rr > 0 ? Rt / rr : (Real)0);
        Xtrg[i*3] = x*sc; Xtrg[i*3+1] = y*sc; Xtrg[i*3+2] = z*sc;
      }
      Op.SetTargetCoord(Xtrg);
      Vector<Real> U;
      Op.ComputePotential(U, sigma);                    // no jump term off-surface
      Vector<Real> Uref;
      ker_sl.Eval(Uref, Xtrg, Xsrc, Xsrc, Fsrc);

      Real err2 = 0, ref2 = 0;
      for (Long i = 0; i < U.Dim(); i++) { const Real e = U[i] - Uref[i]; err2 += e*e; ref2 += Uref[i]*Uref[i]; }
      err2 = GlobalReduce((double)err2, comm, CommOp::SUM);   // targets distributed across ranks
      ref2 = GlobalReduce((double)ref2, comm, CommOp::SUM);
      const Real rel_l2 = std::sqrt(err2 / ref2);
      if (!comm.Rank())
        std::cout << std::setprecision(6) << "  [" << label << "] " << nm[s] << " shell (R=" << Rt
                  << ", GMRES iters=" << iter << "): rel-L2 error = " << rel_l2 << std::endl;
    }
  };

  // EXTERIOR: Stokeslets inside the solid near the origin (both off-axis, |x|~0.29 << finger-tip r~0.935);
  // shells at r > R (exterior fluid, clear of the inward pits).
  const Vector<Real> Xsrc_ext{(Real)0.10, (Real)0.20, (Real)0.15,  (Real)-0.20, (Real)0.10, (Real)-0.10};
  run(/*interior=*/false, Xsrc_ext, R + near_off, (Real)2 * R, "exterior");

  // INTERIOR: Stokeslets outside the surface (|x|~1.5 > R); shells at r < R (solid core, r < 0.9 clears
  // every pit, which live only at r in [~0.935, R]).
  const Vector<Real> Xsrc_int{(Real)1.50, (Real)0.40, (Real)0.30,  (Real)-1.20, (Real)0.80, (Real)-0.60};
  run(/*interior=*/true, Xsrc_int, R - near_off, (Real)0.5 * R, "interior");
}

} // anonymous namespace

int main(int argc, char** argv) {
  Comm::MPI_Init(&argc, &argv);
  using Real = double;
  {
    const Comm comm = Comm::World();
    // Positional CLI: argv[2..12] = tol Nbeta max_depth R_shaft Nc Naz order Ndisk core_frac*100 trg_dist PatchPerFace
    const bool control    = (argc > 1 && std::string(argv[1]) == "sphere");           // finger-free plain cubed sphere (floor)
    const bool collarfill = (argc > 1 && std::string(argv[1]) == "collarfill");       // single collar+disk patch, NO finger
    const bool greenoff   = (argc > 1 && std::string(argv[1]) == "greenoff");         // off-surface Green's on the single-collar sphere
    const bool allcollar  = (argc > 1 && std::string(argv[1]) == "allcollar");        // every patch = collar+disk
    const bool manufac    = (argc > 1 && std::string(argv[1]) == "manufactured");     // manufactured exterior Stokes BVP (interior Stokeslet)
    // REMOVED (2026-07-31): the single-finger studded-sphere modes -- the default (no-keyword) studded finger,
    // "dump" (its per-node DL CSV), "collarsrc" (its collar-source diagnostic) -- and "allcollarfinger"
    // (all-collar sphere + one finger). A mode keyword is now REQUIRED (see the dispatch below).
    const Integer ppf_cli   = (argc > 12) ? (Integer)atoi(argv[12]) : 7;
    // PATCH-RELATIVE shaft: default R_shaft = frac*S (S=1/ppf), r_fil = 0.1*R_shaft; frac from env
    // QJ_RSHAFT_FRAC (default 0.25 -- the thin cilium). A POSITIVE argv[5] (R_shaft) or argv[13] (r_fil)
    // overrides with an absolute value (back-compat). See cilium_scale_from_patch() in stud_sphere_geom.hpp.
    const Real rshaft_frac  = std::getenv("QJ_RSHAFT_FRAC") ? (Real)atof(std::getenv("QJ_RSHAFT_FRAC")) : (Real)0.25;
    const Real S_ref        = (Real)1 / (Real)ppf_cli;   // patch half-width at the primary PPF
    const Real R_shaft_arg  = (argc > 5) ? (Real)atof(argv[5]) : (Real)-1;   // >0 => absolute override
    const Real R_shaft      = (R_shaft_arg > 0) ? R_shaft_arg : rshaft_frac * S_ref;
    const Real r_fil      = (Real)0.1 * R_shaft;   // (the numeric-argv[1] r_fil override went with the removed default finger mode)
    const Real tol          = (argc > 2) ? (Real)atof(argv[2]) : (Real)1e-8;
    const Integer Nbeta     = (argc > 3) ? (Integer)atoi(argv[3]) : 400;
    const Integer max_depth = (argc > 4) ? (Integer)atoi(argv[4]) : 30;
    const Integer Nc_cli    = (argc > 6) ? (Integer)atoi(argv[6]) : -1;
    // Azimuthal panel count (mult of 4), decoupled per mode; argv[7] overrides all.
    const Integer naz_arg      = (argc > 7) ? (Integer)atoi(argv[7]) : -1;
    const Integer naz_collar   = (naz_arg > 0) ? naz_arg : 4;   // single collar patch: collarfill / greenoff
    const Integer naz_allcollar= (naz_arg > 0) ? naz_arg : 8;   // every-patch collar sphere
    const Integer ord_cli   = (argc > 8) ? (Integer)atoi(argv[8]) : 16;
    const Integer Ndisk_cli = (argc > 9) ? (Integer)atoi(argv[9]) : -1;
    const Real corefr_cli   = (argc > 10) ? (Real)atof(argv[10])/100 : (Real)0.40;
    const Real trgdist_cli  = (argc > 11) ? (Real)atof(argv[11]) : (Real)1e-4;
    // New trailing knobs (keyword modes can't set r_fil via argv[1]): r_fil override, radial grading,
    // and the fillet/cap circularization toggle. Defaults keep pre-existing invocations bit-identical.
    const Real rfil_cli     = (argc > 13) ? (Real)atof(argv[13]) : r_fil;
    const Real grade_cli    = (argc > 14) ? (Real)atof(argv[14]) : (Real)1;
    const bool circ_cli     = (argc > 15) ? (atoi(argv[15]) != 0) : false;
    // Manufactured-mode local defaults: ppf=1 (6 well-separated fingers, the accurate/cheap regime) and a
    // near-shell offset of 0.1 (the global trgdist_cli default of 1e-4 is far too near for coarse ppf=1
    // patches; the near-singular target eval on whole-face order-16 panels converges rapidly with offset:
    // 0.05->1.4e-4, 0.1->5.3e-6, 0.2->2.0e-7, so 0.1 lands in the 1e-5..1e-7 band). Near shell = R +
    // near_off; far shell fixed at 2R (matches test-quad-elem).
    const Integer ppf_mfg   = (argc > 12) ? ppf_cli : 1;
    const Real near_off     = (argc > 11) ? trgdist_cli : (Real)0.1;

    if (allcollar)
      test_AllCollarFillSphere<Real>(comm, ord_cli, ppf_cli, 1.0, tol, naz_allcollar, rfil_cli, grade_cli, Nbeta, max_depth, 10, R_shaft, trgdist_cli, Nc_cli, -1, corefr_cli, false, circ_cli);
    else if (manufac)
      test_ManufacturedStokes<Real>(comm, ord_cli, ppf_mfg, 1.0, tol, naz_allcollar, rfil_cli, grade_cli, Nbeta, max_depth, 10, R_shaft, near_off, Nc_cli, corefr_cli, circ_cli);
    else if (greenoff)
      test_SphereCollarFillOffSurface<Real>(comm, ord_cli, ppf_cli, 1.0, tol, naz_collar, r_fil, 1, Nbeta, max_depth, 10, R_shaft, trgdist_cli, Nc_cli, -1, corefr_cli);
    else if (collarfill)
      test_SphereCollarFill<Real>(comm, ord_cli, ppf_cli, 1.0, tol, naz_collar, r_fil, 1, Nbeta, max_depth, 10, R_shaft, Nc_cli, Ndisk_cli, corefr_cli);
    else if (control)
      test_PlainSphere<Real>(comm, 16, 7, 1.0, tol, Nbeta, max_depth);
    else
      SCTL_ASSERT_MSG(false, "stud_sphere-bie: a mode keyword is required -- one of "
          "sphere | collarfill | allcollar | greenoff | manufactured "
          "(the single-finger default, 'dump', 'collarsrc', and 'allcollarfinger' modes were removed)");
  }
  Comm::MPI_Finalize();
  return 0;
}
