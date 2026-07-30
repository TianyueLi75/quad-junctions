/**
 * SETUP-TIME driver for the SCTL/QuadElemList BIE solver on the closed Y-bifurcation surface
 * (junction+transitions+caps + quad-tube arms), for the Laplace OR Stokes single-layer kernel.
 * Builds a single-layer BoundaryIntegralOp, triggers Setup(), and lets the SCTL profiler report the
 * SetupSingular (self) and SetupNear phases. The setup speed compared against fmm3dbie is
 *     Nnodes / t_avg(SetupSingular + SetupNear).
 *
 *   make bin/ybifurc-bie-selfsetup
 *   OMP_NUM_THREADS=1 ./bin/ybifurc-bie-selfsetup <laplace|stokes> [order] [nref] [tol] [Nbeta] [max_depth] [cov_q]
 *     defaults: order 12, nref 2, tol 1e-8, Nbeta 200, max_depth 8, cov_q 6
 *
 * Read the "SetupSingular" and "SetupNear" t_avg rows from the printed profile; Nnodes is echoed below.
 */
#include <sctl.hpp>
#include <quad_junctions/ybifurc_hybrid_geom.hpp>   // BuildYJunctionWithTransitions, BuildYArmsQuadTube, YSwept
#include <quad_junctions/fmm_kernels.hpp>            // SetPVFMMKer (no-op without PVFMM)
#include <quad_junctions/hybrid_bie_tests.hpp>       // test_DLIdentity (DL const-density identity accuracy check)
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <string>

using namespace sctl;
using namespace quad_junctions;

namespace {
template <class Real, class KerSL, class KerDL>
void run_setup(const char* kername, Integer order, Integer nref, Real tol,
               Integer Nb, Integer md, Integer cov_q, const Comm& comm) {
  using Ker = KerSL;
  // Geometry identical to src/ybifurc-export-fmm3dbie.cpp (order-matched to the fmm3dbie mesh).
  const Real level = 1.5, etajoin = 0.4, s_cap = 0.88;
  const Integer NsTrans = 3, nAxial = 3, Ncap = -1;
  const Integer NsShaft = std::max<Integer>(1, (Integer)std::lround((double)nAxial * 10.0 / order));
  const Integer NaShaft = (Integer)(YSwept::Na0 * nref);

  Real R0[3], a0[3], sc[3], max_res = 0;
  QuadElemList<Real> junc = BuildYJunctionWithTransitions<Real>(order, level, nref, etajoin, NsTrans, s_cap,
                                                                R0, a0, sc, Ncap, &max_res, comm);
  QuadElemList<Real> arms = BuildYArmsQuadTube<Real>(order, R0, a0, sc, NsShaft, NaShaft, comm);
  junc.SetQuadScheme(QuadElemList<Real>::QuadScheme::Hybrid, cov_q, Nb, md);
  arms.SetQuadScheme(QuadElemList<Real>::QuadScheme::Hybrid, cov_q, Nb, md);

  const Long Nelem  = junc.Size() + arms.Size();
  const Long Nnodes = Nelem * (Long)order * order;

  BoundaryIntegralOp<Real, Ker> BIOp((Ker()), false, comm);
  SetPVFMMKer(BIOp);
  BIOp.SetAccuracy(tol);
  BIOp.AddElemList(junc, "0_junc");
  BIOp.AddElemList(arms, "1_arms");

  Profile::Enable(true);
  Profile::reset();
  std::cout << "  === PASS 1 (COLD: first Setup in this process; builds the static self/near rule caches) ===\n";
  BIOp.Setup();                       // SetupBasic + SetupFar + SetupSelf(=SetupSingular) + SetupNear
  Profile::print(&comm, {"t_avg"});

  // ---- PASS 2 (WARM): a second BoundaryIntegralOp on the SAME geometry in the SAME process.
  // The process-local static rule caches (RPSelfRule / near-adaptive rules) are already built,
  // so pass-2 SetupSingular/SetupNear recompute ONLY the geometry-specific correction matrices.
  // pass1 - pass2 = the one-time rule-cache build; pass1 is the cold first-setup that is the fair
  // apples-to-apples against fmm3dbie's getnearquad (which is always cold, no such cache).
  BoundaryIntegralOp<Real, Ker> BIOp2((Ker()), false, comm);
  SetPVFMMKer(BIOp2);
  BIOp2.SetAccuracy(tol);
  BIOp2.AddElemList(junc, "0_junc");
  BIOp2.AddElemList(arms, "1_arms");
  Profile::reset();
  std::cout << "  === PASS 2 (WARM: rule caches already built; recomputes only the geometry corrections) ===\n";
  BIOp2.Setup();
  Profile::print(&comm, {"t_avg"});

  std::cout << "\n" << kername << "-SL SELF SETUP (SCTL QuadElemList)"
            << "  order=" << order << " nref=" << nref
            << "  Nelem=" << Nelem << "  Nnodes=" << Nnodes
            << "  (Hybrid cov_q=" << cov_q << " Nbeta=" << Nb << " max_depth=" << md
            << " tol=" << tol << ")\n"
            << "  -> speed = Nnodes / t_avg(SetupSingular + SetupNear)\n";

  // ---- DL const-density identity accuracy check (untimed): D[1] -> -1/2 on-surface ----
  // Verifies the near-quad scheme actually delivers accuracy at (tol,Nbeta,max_depth). Uses the DL
  // kernel; same Hybrid scheme + tol as the timed SL setup above.
  std::cout << "  [accuracy] ";
  test_DLIdentity<Real, KerDL>(junc, arms, comm, tol);
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

    if (kernel == "laplace")      run_setup<Real, Laplace3D_FxU, Laplace3D_DxU>("LAPLACE", order, nref, tol, Nb, md, cov_q, comm);
    else if (kernel == "stokes")  run_setup<Real, Stokes3D_FxU,  Stokes3D_DxU >("STOKES",  order, nref, tol, Nb, md, cov_q, comm);
    else { std::cerr << "kernel must be 'laplace' or 'stokes' (got '" << kernel << "')\n"; Comm::MPI_Finalize(); return 1; }
  }
  Comm::MPI_Finalize();
  return 0;
}
