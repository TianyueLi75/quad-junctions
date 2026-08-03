/**
 * SETUP-TIME driver for the SCTL BIE solver on the HYBRID closed Y-bifurcation surface:
 * QuadElemList junction+transitions+caps  +  CSBQ SlenderElemList arms, both in one
 * BoundaryIntegralOp, for the Laplace OR Stokes single-layer kernel. Sets the on-surface targets,
 * clears any setup, then TIMES a single cold Setup(); the SCTL profiler reports the SetupSingular
 * (self) and SetupNear phases. The setup speed compared against fmm3dbie is
 *     Nnodes / t_avg(SetupSingular + SetupNear).
 *
 *   make bin/ybifurc-bie-selfsetup
 *   OMP_NUM_THREADS=1 ./bin/ybifurc-bie-selfsetup <laplace|stokes> [order] [nref] [tol] [Nbeta] [max_depth] [cov_q] [fourier]
 *     defaults: order 12, nref 2, tol 1e-8, Nbeta 200, max_depth 8, cov_q 6, fourier 12
 *
 * Read the "SetupSingular" and "SetupNear" t_avg rows from the printed profile; Nnodes is echoed below.
 */
#include <sctl.hpp>
#include <csbq.hpp>                                  // CSBQ SlenderElemList (hybrid arms)
#include <quad_junctions/ybifurc_hybrid_geom.hpp>   // BuildYJunctionWithTransitions, BuildYArmsSlender, YSwept
#include <quad_junctions/fmm_kernels.hpp>            // SetPVFMMKer (no-op without PVFMM)
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <string>

using namespace sctl;
using namespace quad_junctions;

namespace {
template <class Real, class KerSL>
void run_setup(const char* kername, Integer order, Integer nref, Real tol,
               Integer Nb, Integer md, Integer cov_q, Long fourier, const Comm& comm) {
  using Ker = KerSL;
  // HYBRID geometry: QuadElemList junction+transitions+caps  +  CSBQ SlenderElemList arms,
  // exactly as assembled by src/ybifurc-hybrid-bie.cpp (arm_kind=0, the slender default).
  const Real level = 1.5, etajoin = 0.4, s_cap = 0.88;
  const Integer NsTrans = 3, nAxial = 3, Ncap = -1;
  const Long cheb_order = 10;

  Real R0[3], a0[3], sc[3], max_res = 0;
  QuadElemList<Real>    junc = BuildYJunctionWithTransitions<Real>(order, level, nref, etajoin, NsTrans, s_cap,
                                                                   R0, a0, sc, Ncap, &max_res, comm);
  SlenderElemList<Real> arms = BuildYArmsSlender<Real>(R0, a0, sc, nAxial, cheb_order, fourier, comm);
  // Junction quad list gets the Hybrid RP scheme; the slender arm list schemes itself off the
  // operator tol (set_arm_scheme is a no-op for SlenderElemList), so it takes no SetQuadScheme.
  junc.SetQuadScheme(QuadElemList<Real>::QuadScheme::Hybrid, cov_q, Nb, md);

  // ---- 1. set up the Laplace/Stokes single-layer BIO on this hybrid surface ----
  BoundaryIntegralOp<Real, Ker> BIOp((Ker()), false, comm);
  SetPVFMMKer(BIOp);                  // normal-free translation kernels (no-op without PVFMM)
  BIOp.SetAccuracy(tol);
  BIOp.AddElemList(junc, "0_junc");
  BIOp.AddElemList(arms, "1_arms");

  // ---- 2. set the targets = the on-surface discretization nodes (same as fmm3dbie's
  //         targs = srcvals positions). This is the established on-surface pattern
  //         (junction_precond.hpp): GetNodeCoord -> AddElemList -> SetTargetCoord. ----
  Vector<Real> Xj, Xa;
  junc.GetNodeCoord(&Xj, nullptr, nullptr);
  arms.GetNodeCoord(&Xa, nullptr, nullptr);
  Vector<Real> Xtrg(Xj.Dim() + Xa.Dim());
  for (Long i = 0; i < Xj.Dim(); i++) Xtrg[i] = Xj[i];
  for (Long i = 0; i < Xa.Dim(); i++) Xtrg[Xj.Dim() + i] = Xa[i];
  BIOp.SetTargetCoord(Xtrg);

  const Long Nelem  = junc.Size() + arms.Size();
  const Long Nnodes = (Xj.Dim() + Xa.Dim()) / 3;   // junc order^2/elem + slender cheb*fourier/elem

  // ---- 3. clear any setup state, then TIME a single cold Setup() ----
  // Setup() = SetupBasic + SetupFar + SetupSelf(=SetupSingular) + SetupNear. The fair
  // apples-to-apples against fmm3dbie's getnearquad is SetupSingular + SetupNear (self + near);
  // SetupFarField is the get_far_order analog and is excluded from that metric.
  BIOp.ClearSetup();
  Profile::Enable(true);
  Profile::reset();
  BIOp.Setup();
  Profile::print(&comm, {"t_avg"});

  std::cout << "\n" << kername << "-SL SETUP (SCTL hybrid: QuadElemList junction + SlenderElemList arms)"
            << "  order=" << order << " nref=" << nref << " fourier=" << fourier
            << "  Nelem=" << Nelem << "  Nnodes=" << Nnodes
            << "  (junction Hybrid cov_q=" << cov_q << " Nbeta=" << Nb << " max_depth=" << md
            << " tol=" << tol << ")\n"
            << "  -> speed = Nnodes / t_avg(SetupSingular + SetupNear)\n";
}
}  // namespace

int main(int argc, char** argv) {
  Comm::MPI_Init(&argc, &argv);
  using Real = double;
  {
    const Comm comm = Comm::Self();
    const std::string kernel = (argc > 1) ? argv[1] : "laplace";
    const Integer order = (argc > 2) ? (Integer)atoi(argv[2]) : 12;
    const Integer nref  = (argc > 3) ? (Integer)atoi(argv[3]) : 2;
    const Real    tol   = (argc > 4) ? (Real)atof(argv[4]) : (Real)1e-8;
    const Integer Nb    = (argc > 5) ? (Integer)atoi(argv[5]) : 200;
    const Integer md    = (argc > 6) ? (Integer)atoi(argv[6]) : 8;
    const Integer cov_q = (argc > 7) ? (Integer)atoi(argv[7]) : 6;
    const Long    four  = (argc > 8) ? (Long)atoi(argv[8]) : 12;

    if (kernel == "laplace")      run_setup<Real, Laplace3D_FxU>("LAPLACE", order, nref, tol, Nb, md, cov_q, four, comm);
    else if (kernel == "stokes")  run_setup<Real, Stokes3D_FxU >("STOKES",  order, nref, tol, Nb, md, cov_q, four, comm);
    else { std::cerr << "kernel must be 'laplace' or 'stokes' (got '" << kernel << "')\n"; Comm::MPI_Finalize(); return 1; }
  }
  Comm::MPI_Finalize();
  return 0;
}
