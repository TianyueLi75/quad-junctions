/**
 * SETUP-TIME driver for the SCTL BIE solver on the TWISTED cubed sphere (pure QuadElemList),
 * for the Laplace OR Stokes single-layer kernel. Same measurement as src/ybifurc-bie-selfsetup.cpp:
 * set up the SL BoundaryIntegralOp, set the on-surface targets, clear any setup, then TIME a single
 * cold Setup(); the SCTL profiler reports the SetupSingular (self) and SetupNear phases. Setup speed
 * compared against fmm3dbie is
 *     Nnodes / t_avg(SetupSingular + SetupNear).
 *
 *   make bin/twisted-sphere-selfsetup
 *   OMP_NUM_THREADS=1 ./bin/twisted-sphere-selfsetup <laplace|stokes|dl_laplace|stokes_greens> [order] [ppf] [tol] [Nbeta] [max_depth] [cov_q] [R] [theta_twist]
 *     defaults: order 12, ppf 5, tol 1e-8, Nbeta 200, max_depth 8, cov_q 6, R 1, theta_twist 1
 *
 * The `stokes_greens` mode runs the on-surface STOKES Green's identity (SL[du/dn] - DL[u] = u for an
 * exterior Stokeslet source), and is instrumented like the timing modes: a warm-up Setup(), then
 * ClearSetup(), then a profiled cold Setup() (SetupSingular + SetupNear), then the identity residual.
 *
 * Read the "SetupSingular" and "SetupNear" t_avg rows from the printed profile; Nnodes is echoed below.
 *
 * The `dl_laplace` mode instead runs the double-layer Laplace CONSTANT-DENSITY IDENTITY (D[1] = -1/2 on
 * a closed outward-oriented surface): it applies the DL operator to q=1 at the on-surface nodes and
 * reports max|U + 1/2| / (1/2), the standard near-quadrature accuracy probe. Used to find the smallest
 * {ppf, order} whose DL error tracks the requested tol (within ~1-2 magnitudes). Same geometry/args as
 * the timing modes; prints one "DL-LAPLACE-IDENTITY ..." line per invocation.
 */
#include <sctl.hpp>
#include <quad_junctions/twisted_sphere_geom.hpp>   // BuildTwistedSphere
#include <quad_junctions/fmm_kernels.hpp>            // SetPVFMMKer (no-op without PVFMM)
#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <string>

using namespace sctl;
using namespace quad_junctions;

namespace {
template <class Real, class KerSL>
void run_setup(const char* kername, Integer order, Long ppf, Real tol, Integer Nb, Integer md,
               Integer cov_q, Real R, Real theta_twist, const Comm& comm) {
  using Ker = KerSL;

  QuadElemList<Real> surf = BuildTwistedSphere<Real>(order, ppf, R, theta_twist, comm);
  surf.SetQuadScheme(QuadElemList<Real>::QuadScheme::Hybrid, cov_q, Nb, md);

  // ---- 1. set up the Laplace/Stokes single-layer BIO on the twisted sphere ----
  BoundaryIntegralOp<Real, Ker> BIOp((Ker()), false, comm);
  SetPVFMMKer(BIOp);                  // normal-free translation kernels (no-op without PVFMM)
  BIOp.SetAccuracy(tol);
  BIOp.AddElemList(surf, "0_sphere");

  // ---- 2. set the targets = the on-surface discretization nodes (same as fmm3dbie's
  //         targs = srcvals positions; established on-surface pattern GetNodeCoord -> SetTargetCoord) ----
  Vector<Real> Xs;
  surf.GetNodeCoord(&Xs, nullptr, nullptr);
  BIOp.SetTargetCoord(Xs);

  const Long Nelem  = surf.Size();
  const Long Nnodes = Xs.Dim() / 3;

  // ---- 3. clear any setup state, then TIME a single cold Setup() ----
  // Setup() = SetupBasic + SetupFar + SetupSelf(=SetupSingular) + SetupNear. The fair apples-to-apples
  // against fmm3dbie's getnearquad is SetupSingular + SetupNear (self + near); SetupFarField is the
  // get_far_order analog and is excluded from that metric.
  BIOp.ClearSetup();
  Profile::Enable(true);
  Profile::reset();
  BIOp.Setup();
  Profile::print(&comm, {"t_avg", "f/s_avg"});   // t_avg = wall time (s); f/s_avg = FLOP rate (GFLOP/s)

  std::cout << "\n" << kername << "-SL SETUP (SCTL twisted cubed sphere)"
            << "  order=" << order << " ppf=" << ppf << " R=" << R << " theta_twist=" << theta_twist
            << "  Nelem=" << Nelem << "  Nnodes=" << Nnodes
            << "  (Hybrid cov_q=" << cov_q << " Nbeta=" << Nb << " max_depth=" << md
            << " tol=" << tol << ")\n"
            << "  -> speed = Nnodes / t_avg(SetupSingular + SetupNear)\n";
}

// DL Laplace constant-density identity on the twisted sphere: D[1] = -1/2 on a closed, outward-oriented
// surface. Builds the geometry, applies the double-layer Laplace operator to q=1 at the on-surface nodes
// (no SetTargetCoord -> self-eval with the singular correction), and reports the max relative error
// max|U + 1/2| / (1/2). This is the near-quadrature accuracy probe used to size {ppf, order} against tol.
template <class Real, class KerDL>
void run_dl_accuracy(const char* kername, Integer order, Long ppf, Real tol, Integer Nb, Integer md,
                     Integer cov_q, Real R, Real theta_twist, const Comm& comm) {
  QuadElemList<Real> surf = BuildTwistedSphere<Real>(order, ppf, R, theta_twist, comm);
  surf.SetQuadScheme(QuadElemList<Real>::QuadScheme::Hybrid, cov_q, Nb, md);

  BoundaryIntegralOp<Real, KerDL> BIOp((KerDL()), false, comm);
  SetPVFMMKer(BIOp);
  BIOp.SetAccuracy(tol);
  BIOp.AddElemList(surf, "0_sphere");

  Vector<Real> X;
  surf.GetNodeCoord(&X, nullptr, nullptr);
  const Long Nelem  = surf.Size();
  const Long Nnode  = X.Dim() / 3;

  const Real c_expect = -0.5;                 // outward surface normals (cubed-sphere / FacePoint orientation)
  Vector<Real> q(Nnode), U;
  q = 1;
  BIOp.ComputePotential(U, q);                // default targets = the surface's own nodes (on-surface identity)

  Real emax = 0;
  for (Long i = 0; i < U.Dim(); i++) emax = std::max<Real>(emax, std::fabs(U[i] - c_expect));
  emax = GlobalReduce((double)emax, comm, CommOp::MAX);
  const Real erel = emax / std::fabs(c_expect);

  if (!comm.Rank()) {
    std::cout << std::scientific << std::setprecision(3)
              << "DL-LAPLACE-IDENTITY  order=" << order << " ppf=" << ppf
              << " twist=" << theta_twist << " tol=" << tol
              << " (Nbeta=" << Nb << " max_depth=" << md << " cov_q=" << cov_q << ")"
              << "  Nelem=" << Nelem << " Nnodes=" << Nnode
              << "  max_rel_err=" << erel << "\n";
  }
}
// On-surface STOKES GREEN'S IDENTITY on the twisted sphere. For a Stokeslet field u generated by an
// exterior point source X0, the interior representation gives  SL[du/dn] - DL[u] = u  on the surface
// (with the -1/2 jump of the DL self term), so the residual max|(SL[Fs] - (DL[Fd]-0.5 Fd)) - u| / max|u|
// is the standard SL+DL near-quadrature accuracy probe. Densities: Fd = u|_surf, Fs = (grad u . n).
// Kernels: KerSL=Stokes3D_FxU, KerDL=Stokes3D_DxU, KerGrad=Stokes3D_FxT (traction tensor). Ported from
// src/stud_sphere-bie.cpp test_greens_identity.
//
// Instrumented for setup timing: SL and DL BIOps are each Setup() ONCE to warm caches, then ClearSetup()'d,
// then the profiler is reset and a cold Setup() of both is TIMED and printed (SetupSingular + SetupNear)
// before the Green's-identity potentials are evaluated.
template <class Real, class KerSL, class KerDL, class KerGrad>
void run_stokes_greens(Integer order, Long ppf, Real tol, Integer Nb, Integer md, Integer cov_q,
                       Real R, Real theta_twist, const Comm& comm) {
  static constexpr Integer COORD_DIM = 3;
  const Long pid = comm.Rank();

  QuadElemList<Real> surf = BuildTwistedSphere<Real>(order, ppf, R, theta_twist, comm);
  surf.SetQuadScheme(QuadElemList<Real>::QuadScheme::Hybrid, cov_q, Nb, md);

  KerSL kernel_sl; KerDL kernel_dl; KerGrad kernel_grad;
  BoundaryIntegralOp<Real, KerSL> BIOpSL(kernel_sl, false, comm);
  BoundaryIntegralOp<Real, KerDL> BIOpDL(kernel_dl, false, comm);
  SetPVFMMKer(BIOpSL); SetPVFMMKer(BIOpDL);   // normal-free translation kernels (no-op without PVFMM)
  BIOpSL.AddElemList(surf); BIOpDL.AddElemList(surf);
  BIOpSL.SetAccuracy(tol); BIOpDL.SetAccuracy(tol);

  const Long Nelem = surf.Size();
  Vector<Real> Xs; surf.GetNodeCoord(&Xs, nullptr, nullptr);
  const Long Nnodes = Xs.Dim() / COORD_DIM;

  // ---- warm-up Setup() (populate any cached quadrature/geometry state), then clear ----
  BIOpSL.Setup(); BIOpDL.Setup();
  BIOpSL.ClearSetup(); BIOpDL.ClearSetup();

  // ---- TIME a single cold Setup() of both operators (SetupSingular = self, SetupNear = near) ----
  Profile::Enable(true);
  Profile::reset();
  BIOpSL.Setup(); BIOpDL.Setup();
  Profile::print(&comm, {"t_avg", "f/s_avg"});   // t_avg = wall time (s); f/s_avg = FLOP rate (GFLOP/s)

  if (!pid) {
    std::cout << "\nSTOKES-GREENS SETUP (SCTL twisted cubed sphere)"
              << "  order=" << order << " ppf=" << ppf << " R=" << R << " theta_twist=" << theta_twist
              << "  Nelem=" << Nelem << "  Nnodes=" << Nnodes
              << "  (Hybrid cov_q=" << cov_q << " Nbeta=" << Nb << " max_depth=" << md
              << " tol=" << tol << ")\n"
              << "  -> setup speed = Nnodes / t_avg(SetupSingular + SetupNear)\n";
  }

  // ---- Green's-identity accuracy: exterior Stokeslet source -> surface densities -> SL+DL residual ----
  const Vector<Real> X0{(Real)1.3 * R, (Real)1.2 * R, (Real)0.2 * R};   // exterior point source (|X0| > R)
  Vector<Real> X, Xn, Fs, Fd, Uref, Us, Ud;
  surf.GetNodeCoord(&X, &Xn, nullptr);
  {
    Vector<Real> Xn0{0, 0, 0}, F0(KerSL::SrcDim()), dU;
    for (auto& x : F0) x = drand48() - 0.5;
    kernel_sl.Eval(Uref, X, X0, Xn0, F0);
    kernel_grad.Eval(dU, X, X0, Xn0, F0);
    Fd = Uref;
    constexpr Integer KDIM0 = KerSL::SrcDim();
    const Long N = X.Dim() / COORD_DIM;
    Fs.ReInit(N * KDIM0);
    for (Long i = 0; i < N; i++) for (Integer j = 0; j < KDIM0; j++) {
      Real d = 0; for (Long k = 0; k < COORD_DIM; k++) d += dU[(i*KDIM0+j)*COORD_DIM+k] * Xn[i*COORD_DIM+k];
      Fs[i*KDIM0+j] = d;
    }
  }
  BIOpSL.ComputePotential(Us, Fs); BIOpDL.ComputePotential(Ud, Fd);
  Ud -= 0.5 * Fd;
  Vector<Real> Uerr = (Us - Ud) - Uref;
  StaticArray<Real,2> max_err{0,0}, max_val{0,0};
  for (auto x : Uerr) max_err[0] = std::max<Real>(max_err[0], std::fabs(x));
  for (auto x : Uref) max_val[0] = std::max<Real>(max_val[0], std::fabs(x));
  comm.Allreduce(max_err+0, max_err+1, 1, CommOp::MAX);
  comm.Allreduce(max_val+0, max_val+1, 1, CommOp::MAX);

  if (!pid) {
    std::cout << std::scientific << std::setprecision(3)
              << "STOKES-GREENS-IDENTITY  order=" << order << " ppf=" << ppf
              << " twist=" << theta_twist << " tol=" << tol
              << " (Nbeta=" << Nb << " max_depth=" << md << " cov_q=" << cov_q << ")"
              << "  Nelem=" << Nelem << " Nnodes=" << Nnodes
              << "  max_rel_err=" << max_err[1] / max_val[1] << "\n";
  }
}
}  // namespace

int main(int argc, char** argv) {
  Comm::MPI_Init(&argc, &argv);
  using Real = double;
  {
    const Comm comm = Comm::Self();
    const std::string kernel = (argc > 1) ? argv[1] : "laplace";
    const Integer order = (argc > 2) ? (Integer)atoi(argv[2]) : 12;
    const Long    ppf   = (argc > 3) ? (Long)atoi(argv[3]) : 5;      // PatchPerFace
    const Real    tol   = (argc > 4) ? (Real)atof(argv[4]) : (Real)1e-8;
    const Integer Nb    = (argc > 5) ? (Integer)atoi(argv[5]) : 200;
    const Integer md    = (argc > 6) ? (Integer)atoi(argv[6]) : 8;
    const Integer cov_q = (argc > 7) ? (Integer)atoi(argv[7]) : 6;
    const Real    R     = (argc > 8) ? (Real)atof(argv[8]) : (Real)1.0;
    const Real    twist = (argc > 9) ? (Real)atof(argv[9]) : (Real)1.0;   // theta(z) = twist * z

    if (kernel == "laplace")         run_setup<Real, Laplace3D_FxU>("LAPLACE", order, ppf, tol, Nb, md, cov_q, R, twist, comm);
    else if (kernel == "stokes")     run_setup<Real, Stokes3D_FxU >("STOKES",  order, ppf, tol, Nb, md, cov_q, R, twist, comm);
    else if (kernel == "dl_laplace") run_dl_accuracy<Real, Laplace3D_DxU>("LAPLACE-DL", order, ppf, tol, Nb, md, cov_q, R, twist, comm);
    else if (kernel == "stokes_greens") run_stokes_greens<Real, Stokes3D_FxU, Stokes3D_DxU, Stokes3D_FxT>(order, ppf, tol, Nb, md, cov_q, R, twist, comm);
    else { std::cerr << "kernel must be 'laplace', 'stokes', 'dl_laplace', or 'stokes_greens' (got '" << kernel << "')\n"; Comm::MPI_Finalize(); return 1; }
  }
  Comm::MPI_Finalize();
  return 0;
}
