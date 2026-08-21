/**
 * LOAD-AND-VERIFY consumer for a generalized-bifurcation geometry bundle.
 *
 * This is the "similar script" that consumes the mesher's final product: it reads an EXACT reloadable
 * bundle (<prefix>.mesh + <prefix>.arms, written to GenGeomDir()/geom/ by bifurc-general-{bie,geom}) via
 * ReadGenGeom -- WITHOUT any field / Voronoi / GenSpec machinery -- and runs the SAME coupled BIE identity
 * suite (watertightness + DL const-density identity + Green's identity, Laplace & Stokes) that
 * bifurc-general-bie ran on the freshly-built surface. A matching pass here proves the on-disk geometry is
 * directly usable by any downstream BIE / flow driver that follows the ybifurc-hybrid "0_junc"/"1_arms"
 * layout (the operator name-sorts the two lists, so the ordering is identical to a built run).
 *
 *   make bin/bifurc-general-load-bie
 *   OMP_NUM_THREADS=8 ./bin/bifurc-general-load-bie <prefix> [nlev]
 *     <prefix>  bundle prefix (loads <prefix>.mesh + <prefix>.arms). A bare name with no '/' is resolved
 *               under GenGeomDir() (geom/), so e.g. "y120-ord12-nref2" finds geom/y120-ord12-nref2.*
 *     nlev      BIE sweep levels 1..4 over tol {1e-5,1e-7,1e-9,1e-11} (default 4)
 */
#include <csbq.hpp>                                 // CSBQ SlenderElemList
#include <quad_junctions/gen_geom_io.hpp>           // ReadGenGeom (exact bundle loader)
#include <quad_junctions/quad_scheme.hpp>           // QJDefaultScheme (Duffy default, SCTL_SELF_SCHEME=hybrid opt-out)
#include <quad_junctions/hybrid_bie_tests.hpp>      // shared BIE identity/watertightness tests
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <string>

using namespace sctl;
using namespace quad_junctions;

int main(int argc, char** argv) {
  Comm::MPI_Init(&argc, &argv);
  using Real = double;
  {
    const Comm comm = Comm::World();
    SCTL_ASSERT_MSG(argc > 1, "usage: bifurc-general-load-bie <prefix> [nlev]");
    // A bare bundle name (no '/') resolves under GenGeomDir() (geom/); an explicit path is used as-is.
    std::string prefix(argv[1]);
    if (prefix.find('/') == std::string::npos) prefix = GenGeomDir() + prefix;
    const Integer nlev = (argc > 2) ? (Integer)atoi(argv[2]) : 4;
    const Integer cov_q = 6;

    // ---- Load the exact coupled surface from disk (no rebuild). ----
    QuadElemList<Real> junc; SlenderElemList<Real> arms; GenArmTable<Real> tab;
    ReadGenGeom<Real>(prefix, junc, arms, &tab, comm);
    const Integer N = tab.N, ord = tab.order;

    if (!comm.Rank()) {
      std::cout << "\n=== LOADED generalized bifurcation bundle \"" << prefix << "\" (N=" << N
                << " arms, order=" << ord << ", fourier=" << tab.fourier_order << ") ===\n";
      for (Integer k = 0; k < N; k++)
        std::cout << "  arm " << k << " dir=(" << std::setprecision(4) << tab.u[k][0] << "," << tab.u[k][1] << "," << tab.u[k][2]
                  << ")  R0=" << std::setprecision(6) << tab.R0[k] << " axial[" << tab.a0[k] << "," << tab.s_cap[k] << "]\n";
    }

    // ---- Exterior Green's sources: one beyond each arm cap tip (same placement as bifurc-general-bie). ----
    Vector<Real> X0;
    for (Integer k = 0; k < N; k++) {
      const Real d = tab.s_cap[k] + (Real)0.6;
      X0.PushBack(d*tab.u[k][0]); X0.PushBack(d*tab.u[k][1]); X0.PushBack(d*tab.u[k][2]);
    }

    const Long njp = GlobalReduce((Long)junc.Size(), comm, CommOp::SUM), nap = GlobalReduce((Long)arms.Size(), comm, CommOp::SUM);
    if (!comm.Rank()) std::cout << "  loaded junction panels=" << njp << " | arm panels=" << nap << "\n";

    const Real    tolL[4] = {(Real)1e-5, (Real)1e-7, (Real)1e-9, (Real)1e-11};
    const Integer NbL[4]  = {48, 100, 200, 400};
    const Integer mdL[4]  = {4, 8, 12, 30};

    junc.SetQuadScheme(quad_junctions::QJDefaultScheme<Real>(), cov_q, NbL[0], mdL[0]);
    divergence_check<Real>(junc, arms, tolL[0], comm);
    for (Integer idx = 0; idx < nlev; idx++) {
      junc.SetQuadScheme(quad_junctions::QJDefaultScheme<Real>(), cov_q, NbL[idx], mdL[idx]);
      if (!comm.Rank()) std::cout << "  [tol=" << tolL[idx] << " Nbeta=" << NbL[idx] << " max_depth=" << mdL[idx] << "]\n";
      if (!comm.Rank()) std::cout << "    [Laplace] "; test_DLIdentity<Real, Laplace3D_DxU>(junc, arms, comm, tolL[idx], "", RegionReport<Real>{});
      if (!comm.Rank()) std::cout << "    [Stokes]  "; test_DLIdentity<Real, Stokes3D_DxU>(junc, arms, comm, tolL[idx], "", RegionReport<Real>{});
      if (!comm.Rank()) std::cout << "    [Laplace] "; test_greens_identity<Real, Laplace3D_FxU, Laplace3D_DxU, Laplace3D_FxdU>(junc, arms, comm, tolL[idx], X0, "");
      if (!comm.Rank()) std::cout << "    [Stokes]  "; test_greens_identity<Real, Stokes3D_FxU, Stokes3D_DxU, Stokes3D_FxT>(junc, arms, comm, tolL[idx], X0, "");
    }
  }
  Comm::MPI_Finalize();
  return 0;
}
