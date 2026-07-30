/**
 * Doubly-periodic (XY) Stokes periodicity-solver TEST: a rigid sphere between two no-slip plates.
 *
 * The simplest possible periodic obstacle problem, built to sanity-check this repo's periodic Stokes
 * boundary-integral machinery (the only other user is the elaborate cilia_carpet-bie). Geometry (ONE
 * QuadElemList):
 *   - two flat plates, each a SINGLE order-`order` patch spanning [0,L]x[0,L], at z=z_bottom / z=z_top,
 *   - one cubed sphere (PatchPerFace^2 * 6 patches, radius R) centered in the cell at (L/2,L/2,(z_b+z_t)/2).
 * All normals point OUT of the fluid domain (plates: bottom -z / top +z; sphere: toward its own center),
 * so the flow solve uses NormalOrient=+1 everywhere (=> -I/2 DL jump), exactly like cilia_carpet-bie.
 *
 * Physics: pressure-driven plane-Poiseuille background flow between the walls, with the sphere and walls
 * held no-slip (zero total velocity). Combined-field BIE solves for the surface density cancelling the
 * background flow; the induced field u = BIO(sigma) - u_bg is evaluated for the periodicity check + VTK.
 *
 * Modes:
 *   dl   : DL constant-density identity (Laplace + Stokes), free-space AND periodic. The surface is NOT
 *          closed in free space (open plates), so the free-space value is only a diagnostic. The PERIODIC
 *          Laplace acceptance is that D_per[const]/q is spatially UNIFORM (spread -> 0): the walls become
 *          an infinite closed slab and the sphere a closed body, so a non-uniform result flags an
 *          inconsistent plate/sphere normal orientation. (Stokes periodic DL is NOT a clean gate -- it
 *          lacks the volume-potential correction that only StokesBIO's SL carries; see cilia_carpet-bie.)
 *   flow : the deliverable -- periodic pressure-driven Stokes solve + x/y periodicity check + volume VTK.
 *
 * Periodicity requires PVFMM (make PVFMM=1). NB: EvalPVFMM's "< 40000 targets -> direct summation"
 * fallback is gated on periodicity==NONE (fmm-wrapper.txx:858-861); with SetPeriodicity(XY) the periodic
 * FMM ALWAYS engages, so this small (~8k-node) surface is a valid periodicity test. Run with
 *   PVFMM_DIR=extern/pvfmm OMP_NUM_THREADS=8 ./bin/periodic-sphere-bie <mode> ...
 *
 *   ./bin/periodic-sphere-bie [mode] [order R tol pdrop Nvis PatchPerFace]
 */
#include <csbq.hpp>
#include <stokes_bio.hpp>
#include <quad_junctions/fmm_kernels.hpp>
#include <quad_junctions/plane_cilia_geom.hpp>   // pulls stud_sphere_geom.hpp: add_cubedsphere, report_area, flip_group, orient_group_flat
#include <quad_junctions/periodic_flow_utils.hpp>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>

using namespace sctl;
using namespace quad_junctions;

namespace {

// Canonical tol -> (Nbeta, max_depth) near-singular quadrature map (identical to cilia_carpet-bie.cpp):
//   tol {1e-5,1e-7,1e-9,1e-11} -> Nbeta {48,100,200,400}, max_depth {4,8,12,30}. Intermediate tol rounds
//   UP to the finer scheme; beyond the ends it clamps.
template <class Real> void quad_scheme_for_tol(Real tol, Integer& Nbeta, Integer& max_depth) {
  const Real    tolL[4] = {(Real)1e-5, (Real)1e-7, (Real)1e-9, (Real)1e-11};
  const Integer NbL[4]  = {48, 100, 200, 400};
  const Integer mdL[4]  = {4, 8, 12, 30};
  int idx = 3;
  for (int k = 0; k < 4; k++) if (tolL[k] <= tol * (Real)(1 + 1e-6)) { idx = k; break; }
  Nbeta = NbL[idx]; max_depth = mdL[idx];
}

// Flat plate at z=z_plane spanning [0,L]x[0,L], tiled into plate_np x plate_np order x order patches
// (u=y slow, v=x fast, AoS), each oriented so its normal points along (0,0,uz) (uz=-1 bottom wall / +1 top
// wall = out of the fluid). plate_np=1 -> a single patch (original behavior). Appended to Xall.
template <class Real> void add_plate(Vector<Real>& Xall, Integer order, Real L, Real z_plane, Real uz, Integer plate_np = 1) {
  const Vector<Real>& nds = QuadElemList<Real>::ParamNodes(order);
  const Real h = L / (Real)plate_np;
  for (Integer pi = 0; pi < plate_np; pi++) for (Integer pj = 0; pj < plate_np; pj++) {
    const Real y0 = pi * h, x0 = pj * h;
    Vector<Real> Xp;   // one order x order patch over the sub-square [x0,x0+h] x [y0,y0+h]
    for (Integer i = 0; i < order; i++) { const Real yy = y0 + nds[i] * h;
      for (Integer j = 0; j < order; j++) { const Real xx = x0 + nds[j] * h;
        Xp.PushBack(xx); Xp.PushBack(yy); Xp.PushBack(z_plane); } }
    orient_group_flat<Real>(Xp, order, z_plane, uz);   // transpose (negate normals) if not already along uz
    for (auto v : Xp) Xall.PushBack(v);
  }
}

// A cubed sphere of radius R centered at c, with normals pointing TOWARD c (out of the fluid = into the
// solid obstacle). add_cubedsphere is origin-centered; we orient (flip if the emitter winds outward-from-
// center), then translate to c. Appended to Xall.
template <class Real> void add_obstacle_sphere(Vector<Real>& Xall, Integer order, Long PatchPerFace,
                                               Real R, const Real c[3]) {
  Vector<Real> Xs;
  add_cubedsphere<Real>(Xs, order, PatchPerFace, R, /*skipFace=*/-1, 0, 0);   // origin-centered
  {   // orient toward center: for origin-centered nodes, x is the outward radial => n.x > 0 means outward.
    QuadElemList<Real> tmp(order, Xs);
    Vector<Real> Xc, Xnc; tmp.GetNodeCoord(&Xc, &Xnc, nullptr);
    Real acc = 0;
    for (Long i = 0; i < Xc.Dim() / 3; i++) for (int k = 0; k < 3; k++) acc += Xnc[i*3+k] * Xc[i*3+k];
    if (acc > 0) flip_group<Real>(Xs, order);   // make normals point toward center (out of fluid)
  }
  for (Long i = 0; i < Xs.Dim() / 3; i++) { Xs[i*3+0] += c[0]; Xs[i*3+1] += c[1]; Xs[i*3+2] += c[2]; }
  for (auto v : Xs) Xall.PushBack(v);
}

// Surface mean of a KDIM-per-node density over one QuadElemList (far-field nodes are 1:1 with collocation
// for QuadElemList, so GetFarFieldDensity is empty and we integrate sigma directly). Returns per-component
// mean = int(sigma dA) / area. Single-list analogue of cilia_carpet-bie's combined_surface_mean.
template <class Real> void surface_mean(const QuadElemList<Real>& surf, const Vector<Real>& sigma,
    const Integer KDIM, const Real tol, const Comm& comm, Vector<Real>& mean_out, Real& total_area) {
  Vector<Real> Xff, Xnff, wts, dist, Fff; Vector<Long> cnt;
  surf.GetFarFieldNodes(Xff, Xnff, wts, dist, cnt, tol);
  surf.GetFarFieldDensity(Fff, sigma);
  const Vector<Real>& dens = (Fff.Dim() ? Fff : sigma);
  const Long Nq = wts.Dim();
  Vector<Real> sm(KDIM); sm = 0; Real area = 0;
  for (Long i = 0; i < Nq; i++) { area += wts[i]; for (Integer k = 0; k < KDIM; k++) sm[k] += wts[i] * dens[i*KDIM+k]; }
  total_area = GlobalReduce((double)area, comm, CommOp::SUM);
  mean_out.ReInit(KDIM);
  for (Integer k = 0; k < KDIM; k++) mean_out[k] = GlobalReduce((double)sm[k], comm, CommOp::SUM) / total_area;
}

// DL constant-density identity D[q]. Closed FREE-SPACE surfaces give -1/2; here the surface is open in
// free space (two square plates) so free-space is diagnostic only. Under XY periodicity the walls close
// into an infinite slab: D_per[const]/q must be spatially UNIFORM (spread -> 0) -- the geometry+orientation
// acceptance gate. Reports mean, max rel err vs -0.5, and spread (with a wall-vs-sphere split).
template <class Real, class KerDL>
void test_DLIdentity(const QuadElemList<Real>& surf, const Comm& comm, const Real tol, const bool periodic,
                     const Real L, const Real z_bottom, const Real z_top) {
  const KerDL kernel_dl;
  BoundaryIntegralOp<Real, KerDL> BIOp(kernel_dl, false, comm);
  SetPVFMMKer(BIOp);
  BIOp.SetAccuracy(tol);
  BIOp.AddElemList(surf, "0_surf");
  if (periodic) BIOp.SetPeriodicity(sctl::Periodicity::XY, L);
  const Real c_expect = -0.5;
  const Long KDIM0 = KerDL::SrcDim();
  Vector<Real> X, Xn; surf.GetNodeCoord(&X, &Xn, nullptr);
  const Long Nnode = X.Dim() / 3;
  Vector<Real> q(Nnode * KDIM0), U;
  for (Long i = 0; i < Nnode; i++) for (Long k = 0; k < KDIM0; k++) q[i*KDIM0 + k] = k + 1;
  BIOp.ComputePotential(U, q);
  const Real eps = 1e-9;
  Real emax = 0, mean_val = 0, vmax = -1e30, vmin = 1e30;
  Real vmax_w = -1e30, vmin_w = 1e30, vmax_s = -1e30, vmin_s = 1e30;
  for (Long i = 0; i < Nnode; i++) {
    const Real z = X[i*3+2];
    const bool on_wall = std::fabs((double)(z - z_bottom)) < eps || std::fabs((double)(z - z_top)) < eps;
    for (Long k = 0; k < KDIM0; k++) {
      const Real v = U[i*KDIM0+k] / q[i*KDIM0+k];
      mean_val += v; vmax = std::max(vmax, v); vmin = std::min(vmin, v);
      emax = std::max(emax, std::fabs(v - c_expect));
      if (on_wall) { vmax_w = std::max(vmax_w, v); vmin_w = std::min(vmin_w, v); }
      else         { vmax_s = std::max(vmax_s, v); vmin_s = std::min(vmin_s, v); }
    }
  }
  const Real spread_w = GlobalReduce((double)vmax_w, comm, CommOp::MAX) - GlobalReduce((double)vmin_w, comm, CommOp::MIN);
  const Real spread_s = GlobalReduce((double)vmax_s, comm, CommOp::MAX) - GlobalReduce((double)vmin_s, comm, CommOp::MIN);
  const Real avg    = GlobalReduce((double)emax, comm, CommOp::MAX) / std::fabs(c_expect);
  const Real spread = GlobalReduce((double)vmax, comm, CommOp::MAX) - GlobalReduce((double)vmin, comm, CommOp::MIN);
  const Long Ntot   = GlobalReduce((Long)(Nnode*KDIM0), comm, CommOp::SUM);
  const Real mean_all = GlobalReduce((double)mean_val, comm, CommOp::SUM) / Ntot;
  if (!comm.Rank()) {
    std::cout << std::setprecision(6)
              << "  [" << (periodic ? "PERIODIC" : "free-space") << "] DL const-density: mean D[const]/q = " << mean_all
              << "   max rel err vs -0.5 = " << avg << "\n";
    std::cout << "    spread(vmax-vmin) = " << spread << "   (wall " << spread_w << ", sphere " << spread_s
              << ")   " << (periodic ? "<- PERIODIC acceptance: spread->0" : "") << "\n";
  }
}

// Pressure-driven doubly-periodic Stokes solve + volume-flow VTK. Combined-field BIE with the surface-mean
// projection (rank-deficiency fix), replicating cilia_carpet-bie's run_flow for a single QuadElemList.
template <class Real>
void run_flow(const QuadElemList<Real>& surf, const Comm& comm, const Real tol, const Real L,
              const Real pdrop, const Long Nvis, const Real R, const Real c[3],
              const Real z_bottom, const Real z_top) {
  const Real SL_scal = 1.0, DL_scal = 1.0;
  const Real gmres_tol = std::max(tol, (Real)1e-8);
  const Long gmres_max_iter = 400;
  constexpr Integer KDIM = 3;

  Vector<Real> X0, Xn0; surf.GetNodeCoord(&X0, &Xn0, nullptr);
  const Long Nnode = X0.Dim() / 3;

  StokesBIO<Real> Op(SL_scal, DL_scal, comm);
  Op.SetAccuracy(tol);
  Op.AddElemList(surf, "0_surf");
  Op.SetPeriodicity(sctl::Periodicity::XY, L);
  Op.SetTargetCoord(X0);

  Vector<Real> NormalOrient(3 * Nnode); NormalOrient = 1.0;   // outward-from-fluid => -I/2 jump

  const auto BIO = [&](Vector<Real>* U, const Vector<Real>& sigma) {
    Vector<Real> sm; Real total_area;
    surface_mean<Real>(surf, sigma, KDIM, tol, comm, sm, total_area);
    Vector<Real> sigma0 = sigma; AddConstVec(sigma0, sm * (Real)-1);
    U->SetZero();
    Op.ComputePotential(*U, sigma0);
    if (DL_scal && U->Dim() == sigma0.Dim()) (*U) -= sigma0 * (Real)0.5 * NormalOrient * DL_scal;  // surface only
    AddConstVec(*U, sm);
  };

  GMRES<Real> solver(comm);
  KrylovPrecond<Real> krylov;
  Vector<Real> sigma;
  Long niter = 0;
  Vector<Real> rhs = bg_flow_2peri(X0); rhs *= (pdrop / L);
  Profile::Enable(true); Profile::reset();
  Profile::Tic("periodic_sphere_gmres_solve", &comm, true);
  solver(&sigma, BIO, rhs, gmres_tol, gmres_max_iter, false, &niter, &krylov);
  Profile::Toc();
  if (!comm.Rank()) std::cout << "  flow: GMRES converged in " << niter << " iters\n";
  Profile::print(&comm, {"t_avg", "t_max", "f/s_avg", "f/s_max"});

  const Real zc = (Real)0.5 * (z_bottom + z_top);
  auto eval_induced = [&](const Vector<Real>& Xt) {
    Op.SetTargetCoord(Xt);
    Vector<Real> U(Xt.Dim()); U = 0; BIO(&U, sigma);
    Vector<Real> Ub = bg_flow_2peri(Xt); Ub *= (pdrop / L);
    U -= Ub; return U;
  };

  // ===== Verification 1: periodicity across the x and y faces =====
  // Induced velocity at matching points on opposite faces (x=0 vs x~=L; y=0 vs y~=L), same other coords, at
  // the mid-gap. These are periodic IMAGES, so the difference should be at the solve/FMM tolerance.
  {
    const Long n = 6; const Real eL = L * (Real)0.9999999999;
    Vector<Real> Xa(3*n), Xb(3*n), Ya(3*n), Yb(3*n);
    for (Long i = 0; i < n; i++) { const Real t = (i + (Real)0.5) / n * L;
      Xa[i*3+0]=0;  Xa[i*3+1]=t; Xa[i*3+2]=zc;   Xb[i*3+0]=eL; Xb[i*3+1]=t; Xb[i*3+2]=zc;
      Ya[i*3+0]=t;  Ya[i*3+1]=0; Ya[i*3+2]=zc;   Yb[i*3+0]=t;  Yb[i*3+1]=eL; Yb[i*3+2]=zc; }
    Vector<Real> Ua=eval_induced(Xa), Ub=eval_induced(Xb), Va=eval_induced(Ya), Vb=eval_induced(Yb);
    Real dx=0, dy=0, umag=1e-30;
    for (Long i=0;i<3*n;i++){ dx=std::max(dx,std::fabs((double)(Ua[i]-Ub[i]))); dy=std::max(dy,std::fabs((double)(Va[i]-Vb[i])));
      umag=std::max(umag,std::max(std::fabs((double)Ua[i]),std::fabs((double)Va[i]))); }
    dx=GlobalReduce((double)dx,comm,CommOp::MAX); dy=GlobalReduce((double)dy,comm,CommOp::MAX); umag=GlobalReduce((double)umag,comm,CommOp::MAX);
    if(!comm.Rank()) std::cout << std::setprecision(4)
       << "  [verify-periodicity] max|u(x=0)-u(x=L)| = " << dx << " (rel " << dx/umag << "),  "
       << "max|u(y=0)-u(y=L)| = " << dy << " (rel " << dy/umag << ")   |u|~" << umag << "\n";
  }

  // ===== Verification 2: induced velocity at fixed mid-gap probe points (self-convergence) =====
  {
    const Real P[][3] = { {0.25*L,0.25*L,zc},{0.75*L,0.25*L,zc},{0.25*L,0.75*L,zc},
                          {0.75*L,0.75*L,zc},{0.50*L,0.15*L,zc},{0.50*L,0.85*L,zc} };
    const Long np = sizeof(P)/sizeof(P[0]);
    Vector<Real> Xp(3*np); for(Long i=0;i<np;i++) for(int cc=0;cc<3;cc++) Xp[i*3+cc]=P[i][cc];
    Vector<Real> Up = eval_induced(Xp);
    if(!comm.Rank()){ std::cout << std::setprecision(12) << "  [verify-probe] induced u at mid-gap probe points (for self-convergence):\n";
      for(Long i=0;i<np;i++) std::cout << "    P"<<i<<" ("<<P[i][0]<<","<<P[i][1]<<","<<P[i][2]<<") u = "
        << Up[i*3] << ", " << Up[i*3+1] << ", " << Up[i*3+2] << "\n"; }
  }

  // ---- Volume-flow VTU (targets outside the slab or inside the sphere masked to 0) ----
  CubeVolumeVisShifted<Real> vv(Nvis, (Real)0.9, comm);
  Vector<Real> Xv = vv.GetCoord();
  const Long Ntrg = Xv.Dim() / 3;
  if (!comm.Rank()) std::cout << "  flow: volume grid " << Nvis << "^3, " << (Nvis*Nvis*Nvis) << " targets\n";
  Vector<Real> U = eval_induced(Xv);
  Vector<Real> Ubg = bg_flow_2peri(Xv); Ubg *= (pdrop / L);
  Real umax = 0, uxsum = 0, uysum = 0, uzsum = 0; Long nmid = 0;
  for (Long i = 0; i < Ntrg; i++) {
    const Real x = Xv[i*3], y = Xv[i*3+1], z = Xv[i*3+2];
    const Real rr = std::sqrt((x-c[0])*(x-c[0]) + (y-c[1])*(y-c[1]) + (z-c[2])*(z-c[2]));
    const bool masked = (z < z_bottom || z > z_top || rr < R * (Real)1.02);
    if (masked) for (int cc = 0; cc < 3; cc++) U[i*3+cc] = 0;
    umax = std::max(umax, std::sqrt(U[i*3]*U[i*3] + U[i*3+1]*U[i*3+1] + U[i*3+2]*U[i*3+2]));
    if (std::fabs((double)(z - zc)) < 0.03 && !masked) { uxsum += U[i*3]+Ubg[i*3]; uysum += U[i*3+1]+Ubg[i*3+1]; uzsum += U[i*3+2]+Ubg[i*3+2]; nmid++; }
  }
  umax = GlobalReduce((double)umax, comm, CommOp::MAX);
  uxsum = GlobalReduce((double)uxsum, comm, CommOp::SUM); uysum = GlobalReduce((double)uysum, comm, CommOp::SUM);
  uzsum = GlobalReduce((double)uzsum, comm, CommOp::SUM); nmid = GlobalReduce((Long)nmid, comm, CommOp::SUM);
  if (!comm.Rank())
    std::cout << std::setprecision(6) << "  flow: max|U_induced| = " << umax
              << "   mid-gap mean total u = (" << (nmid?uxsum/nmid:0) << ", " << (nmid?uysum/nmid:0) << ", " << (nmid?uzsum/nmid:0) << ")\n";
  vv.WriteVTK("vis/PeriodicSphere_U", U);
  if (!comm.Rank()) std::cout << "  wrote vis/PeriodicSphere_U.pvtu\n";

  // ---- 2D TOTAL-velocity slice CSVs (rank 0 holds the full grid; other ranks pass empty targets so the
  // collective FMM eval stays valid). For quick browser/matplotlib visualization of the flow. ----
  auto dump_slice = [&](const std::string& fname, int axis /*2=z-plane, 1=y-plane*/, Real fixed, Long Ns) {
    // Build the plane grid on rank 0 only. axis=2: vary (x,y) at z=fixed. axis=1: vary (x,z) at y=fixed.
    Vector<Real> Xs;
    if (!comm.Rank()) {
      for (Long i = 0; i < Ns; i++) for (Long j = 0; j < Ns; j++) {
        const Real u = (i + (Real)0.5) / Ns * L;                                 // first in-plane axis (x)
        const Real w = (axis == 2) ? ((j + (Real)0.5) / Ns * L)                  // z-plane: second axis = y in [0,L]
                                    : (z_bottom + (j + (Real)0.5) / Ns * (z_top - z_bottom)); // y-plane: second = z in gap
        const Real x = u;
        const Real y = (axis == 2) ? w : fixed;
        const Real z = (axis == 2) ? fixed : w;
        Xs.PushBack(x); Xs.PushBack(y); Xs.PushBack(z);
      }
    }
    Vector<Real> Ui = eval_induced(Xs);                                          // collective; empty on ranks>0
    Vector<Real> Ub = bg_flow_2peri(Xs); Ub *= (pdrop / L);
    if (comm.Rank()) return;
    std::ofstream f(fname);
    f << "# a1 a2 ux uy uz mask   (axis=" << (axis==2?"z-plane a1=x a2=y":"y-plane a1=x a2=z")
      << " fixed=" << fixed << " Ns=" << Ns << " L=" << L << ")\n";
    f << std::setprecision(8);
    for (Long k = 0; k < Xs.Dim()/3; k++) {
      const Real x = Xs[k*3], y = Xs[k*3+1], z = Xs[k*3+2];
      const Real rr = std::sqrt((x-c[0])*(x-c[0]) + (y-c[1])*(y-c[1]) + (z-c[2])*(z-c[2]));
      const int mask = (rr < R) ? 1 : 0;                                         // inside the sphere obstacle
      const Real ux = Ui[k*3]+Ub[k*3], uy = Ui[k*3+1]+Ub[k*3+1], uz = Ui[k*3+2]+Ub[k*3+2];
      const Real a1 = x, a2 = (axis==2) ? y : z;
      f << a1 << " " << a2 << " " << ux << " " << uy << " " << uz << " " << mask << "\n";
    }
    std::cout << "  wrote " << fname << "\n";
  };
  dump_slice("vis/PeriodicSphere_slice_z.csv", 2, (Real)0.5*(z_bottom+z_top), 128);  // mid-gap z-plane (flow around sphere)
  dump_slice("vis/PeriodicSphere_slice_xz.csv", 1, (Real)0.5*L, 128);               // x-z plane through center (Poiseuille + blockage)
}

} // namespace

int main(int argc, char** argv) {
  Comm::MPI_Init(&argc, &argv);
  using Real = double;
  {
    Comm comm = Comm::World();

    // CLI: [mode] [order R tol pdrop Nvis PatchPerFace]
    const std::string mode = (argc > 1) ? argv[1] : "flow";
    const Integer order    = (argc > 2) ? std::atoi(argv[2]) : 12;    // QuadElemList order (multiple of 4)
    const Real    R        = (argc > 3) ? std::atof(argv[3]) : 0.25;  // sphere radius
    const Real    tol      = (argc > 4) ? std::atof(argv[4]) : 1e-7;
    const Real    pdrop    = (argc > 5) ? std::atof(argv[5]) : -1.0;  // background pressure drop
    const Long    Nvis     = (argc > 6) ? std::atol(argv[6]) : 60;    // volume-vis grid resolution
    const Long    PatchPerFace = (argc > 7) ? std::atol(argv[7]) : 3; // cubed-sphere patches per face-side
    const Integer PlateNp      = (argc > 8) ? std::atoi(argv[8]) : 1; // plate tiling: PlateNp x PlateNp patches/plate

    const Real L = 1.0, z_bottom = 0.01, z_top = 0.99;
    const Real c[3] = {(Real)0.5 * L, (Real)0.5 * L, (Real)0.5 * (z_bottom + z_top)};
    // Containment: sphere fully inside the gap and not touching the cell edge / its periodic image.
    SCTL_ASSERT_MSG(c[2] - R > z_bottom && c[2] + R < z_top, "sphere must fit between the plates in z");
    SCTL_ASSERT_MSG(2 * R < L && R < (Real)0.5 * L, "sphere must not touch the periodic cell boundary/image");

    Integer Nbeta, max_depth; quad_scheme_for_tol<Real>(tol, Nbeta, max_depth);

    // Build the merged surface (replicated per rank; the QuadElemList ctor slices per rank).
    Vector<Real> Xall;
    add_plate<Real>(Xall, order, L, z_bottom, /*uz=*/-1, PlateNp);   // bottom wall, normal -z
    add_plate<Real>(Xall, order, L, z_top,    /*uz=*/+1, PlateNp);   // top wall,    normal +z
    add_obstacle_sphere<Real>(Xall, order, PatchPerFace, R, c);
    QuadElemList<Real> surf(order, Xall, comm);
    surf.SetQuadScheme(QuadElemList<Real>::QuadScheme::Hybrid, 10, Nbeta, max_depth);

    if (!comm.Rank()) {
      const Long nplate = 2 * PlateNp * PlateNp * order * order;
      const Long nsph   = 6 * PatchPerFace * PatchPerFace * order * order;
      std::cout << "periodic-sphere [" << mode << "]  order=" << order << " tol=" << tol
                << " (Nbeta=" << Nbeta << " max_depth=" << max_depth << ")"
                << " sphere R=" << R << " PatchPerFace=" << PatchPerFace << " center=(" << c[0] << "," << c[1] << "," << c[2] << ")\n"
                << "  plates z=" << z_bottom << "/" << z_top << " (each 1 patch), L=" << L
                << "   nodes: plates=" << nplate << " sphere=" << nsph << " total=" << (nplate + nsph) << "\n";
    }
    report_area<Real>(surf, comm);
    surf.WriteVTK("vis/PeriodicSphere_geom", Vector<Real>(), comm);
    if (!comm.Rank()) std::cout << "  wrote vis/PeriodicSphere_geom.pvtu\n";

    const Vector<Real> X0src{1.3, 1.2, 0.2};  // (unused placeholder for symmetry with cilia driver)
    (void)X0src;

    if (mode == "geom") {
      // geometry only: build + VTK + report already done above.
    } else if (mode == "dl") {
      if (!comm.Rank()) std::cout << "-- Laplace --\n";
      test_DLIdentity<Real, Laplace3D_DxU>(surf, comm, tol, false, L, z_bottom, z_top);
      test_DLIdentity<Real, Laplace3D_DxU>(surf, comm, tol, true,  L, z_bottom, z_top);
      if (!comm.Rank()) std::cout << "-- Stokes (periodic DL not a clean gate; see header) --\n";
      test_DLIdentity<Real, Stokes3D_DxU>(surf, comm, tol, false, L, z_bottom, z_top);
      test_DLIdentity<Real, Stokes3D_DxU>(surf, comm, tol, true,  L, z_bottom, z_top);
    } else if (mode == "flow") {
      run_flow<Real>(surf, comm, tol, L, pdrop, Nvis, R, c, z_bottom, z_top);
    } else {
      if (!comm.Rank()) std::cout << "unknown mode '" << mode << "'; use geom | dl | flow\n";
    }
  }
  Comm::MPI_Finalize();
  return 0;
}
