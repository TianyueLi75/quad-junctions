/**
 * ybifurc-vessels-bie: a 20-junction arterial/venous vascular network from
 * `arterial_venous_smoothed_nolabels.svg` (topology in include/quad_junctions/vessels_tree_data.hpp,
 * emitted by python/build_vessels_topology.py).
 *
 * ONE closed watertight surface = an arterial binary tree (root far left, grows left->right) + a mirror
 * venous binary tree (root far right, grows right->left), their 11 leaves joined across the middle. All
 * junctions are the canonical 120-degree Y (ybifurc_assembly HybridJunction), placed planar in z=0 at
 * their SVG positions and uniformly scaled by 0.8^generation (radius shrinks root->leaf). Connectors:
 *   - intra-tree parent->child: SINGLE-CORNER bent arm (lead|corner|run) -- the run plunges straight into
 *     and becomes the child junction's stem (add_bent_arm single_corner=true).
 *   - leaf connectors A_i<->V_i: the two-corner RACETRACK (add_bent_arm single_corner=false). Four pairs
 *     ((1,2),(3,4),(6,7),(9,10)) close into racetrack LENSES; 0,5,8 are lone connectors.
 * All arms taper linearly between the two (scaled) seam radii. The two tree roots are hemisphere-capped.
 *
 * Reuses the HybridAssembly component API and the shared BIE identity tests unchanged.
 * Verification: constant-density DL identity (=-1/2, watertightness) + Green's identity with EXTERIOR
 * point sources (one per junction, validated exterior), Laplace + Stokes.
 *
 *   make bin/ybifurc-vessels-bie
 *   OMP_NUM_THREADS=8 ./bin/ybifurc-vessels-bie \
 *       [level] [order(mult4)] [nref] [eta_join] [Ns_trans] [fourier] [lead] [corner] \
 *       [tol] [Nbeta] [max_depth] [cov_q] [svg_scale] [geomOnly] [gscale]
 *
 *   gscale (default 1) uniformly rescales the WHOLE network (positions AND junction sizes), i.e. a
 *   similarity transform. The DL identity is scale-invariant and SCTL renormalizes into PVFMM's
 *   [0,1]^3 box, so results must not depend on it -- use it to test for absolute-size bugs in the FMM.
 *
 *   geomOnly=1 builds the geometry, runs the collision/watertightness checks, writes the VTU meshes and
 *   exits WITHOUT the BIE solve (and WITHOUT requiring an exterior Green source) -- for layout iteration.
 */

#include <csbq.hpp>                                  // CSBQ SlenderElemList
#include <quad_junctions/ybifurc_assembly.hpp>       // composable component API (add_bent_arm single_corner)
#include <quad_junctions/hybrid_bie_tests.hpp>       // shared BIE identity / watertightness tests
#include <quad_junctions/vessels_build.hpp>          // shared network build (dot3/ArmSeg + build_vessels_network)
#include <quad_junctions/vessels_tree_data.hpp>      // arterial/venous junction + connector tables
#include <array>
#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <limits>
#include <string>
#include <vector>

using namespace sctl;
using namespace quad_junctions;

namespace {

// dot3/nrm3/add3/sub3/mul3/unit3 and struct ArmSeg now live in quad_junctions/vessels_build.hpp
// (shared with the flow driver) and are visible here via `using namespace quad_junctions`.

// Closest distance between two 3D segments [p1,q1] and [p2,q2] (clamped parametric solve; degenerate
// segments give point-to-segment). Same routine as ybifurc-tree-bie.cpp.
template <class Real>
Real seg_seg_dist(const Vec3<Real>& p1, const Vec3<Real>& q1, const Vec3<Real>& p2, const Vec3<Real>& q2) {
  const Vec3<Real> d1 = sub3(q1,p1), d2 = sub3(q2,p2), r = sub3(p1,p2);
  const Real a = dot3(d1,d1), e = dot3(d2,d2), f = dot3(d2,r);
  Real s, t;
  const Real eps = (Real)1e-12;
  if (a <= eps && e <= eps) return nrm3(r);
  if (a <= eps) { s = 0; t = std::min<Real>(1, std::max<Real>(0, f/e)); }
  else {
    const Real c = dot3(d1,r);
    if (e <= eps) { t = 0; s = std::min<Real>(1, std::max<Real>(0, -c/a)); }
    else {
      const Real b = dot3(d1,d2), den = a*e-b*b;
      s = (den > eps) ? std::min<Real>(1, std::max<Real>(0,(b*f-c*e)/den)) : (Real)0;
      t = (b*s+f)/e;
      if (t < 0)      { t = 0; s = std::min<Real>(1, std::max<Real>(0, -c/a)); }
      else if (t > 1) { t = 1; s = std::min<Real>(1, std::max<Real>(0, (b-c)/a)); }
    }
  }
  const Vec3<Real> c1 = add3(p1, mul3(s,d1)), c2 = add3(p2, mul3(t,d2));
  return nrm3(sub3(c1,c2));
}

} // anonymous namespace

int main(int argc, char** argv) {
  Comm::MPI_Init(&argc, &argv);
  using Real = double;
  {
    const Comm comm = Comm::World();
    const Real    level   = (argc > 1)  ? (Real)atof(argv[1])  : (Real)1.5;
    const Integer ord     = (argc > 2)  ? (Integer)atoi(argv[2])  : 8;
    const Integer nref    = (argc > 3)  ? (Integer)atoi(argv[3])  : 1;
    const Real    etajoin = (argc > 4)  ? (Real)atof(argv[4])  : (Real)0.4;
    const Integer NsTrans = (argc > 5)  ? (Integer)atoi(argv[5])  : 3;
    const Long    fourier = (argc > 6)  ? (Long)atoi(argv[6])  : 8;
    const Integer leadP   = (argc > 7)  ? (Integer)atoi(argv[7])  : 1;    // single-corner / racetrack lead panels
    const Integer cornerP = (argc > 8)  ? (Integer)atoi(argv[8])  : 12;   // corner panels
    const Real    tol     = (argc > 9)  ? (Real)atof(argv[9])  : (Real)1e-6;
    const Integer Nbeta   = (argc > 10) ? (Integer)atoi(argv[10]) : 100;
    const Integer maxdep  = (argc > 11) ? (Integer)atoi(argv[11]) : 4;
    const Integer cov_q   = (argc > 12) ? (Integer)atoi(argv[12]) : 6;
    const Real    svgs    = (argc > 13) ? (Real)atof(argv[13]) : (Real)0.06;  // model units per SVG pixel
    const Integer geomOnly= (argc > 14) ? (Integer)atoi(argv[14]) : 0;
    const Real    gscale  = (argc > 15) ? (Real)atof(argv[15]) : (Real)1;   // global similarity scale
    const Integer Ncap    = (Integer)(YSwept::Ncap0 * std::max<Integer>(1, nref));
    const Long    cheb    = 10;
    const Integer nAxFree = 3;
    const Real    tipLen  = (Real)3.0;   // root-cap free-arm length (x junction scale)
    pou_kind() = 1;                       // smootherstep POU (what the assembly transitions expect)

    SCTL_ASSERT_MSG(ord >= 4 && ord <= 48 && ord % 4 == 0, "order must be a multiple of 4 in {4,...,48}.");

    namespace vd = vessels_data;
    const int NJ = vd::n_junc, NC = vd::n_conn;

    if (!comm.Rank()) {
      std::cout << "\n=== ybifurc-vessels: 20-junction arterial/venous network ===\n";
      std::cout << "  order=" << ord << " level=" << level << " nref=" << nref << " eta_join=" << etajoin
                << " Ns_trans=" << NsTrans << " fourier=" << fourier << " lead=" << leadP << " corner=" << cornerP
                << " svg_scale=" << svgs << " gscale=" << gscale << (geomOnly ? "  [GEOM-ONLY]\n" : "\n");
    }

    std::vector<Real> scale(NJ);   // per-junction 0.8^gen (also used in the collision report below)
    for (int i = 0; i < NJ; i++) scale[i] = std::pow((Real)0.8, (Real)vd::juncs[i].gen);

    // Build the whole network (orientation + intra-tree arms + leaf connectors + root caps) via the
    // shared builder; the identity tests below run on the SAME geometry the flow driver uses.
    HybridAssembly<Real> A(ord);
    const VesselsBuild<Real> vb = build_vessels_network<Real>(A, level, nref, etajoin, NsTrans, fourier,
        leadP, cornerP, svgs, Ncap, cheb, nAxFree, tipLen, comm, gscale);
    const std::vector<Placement<Real>>& P = vb.P;
    const std::vector<ArmSeg<Real>>& segs = vb.segs;
    const int n_caps = vb.n_caps;

    QuadElemList<Real> junc = A.quad(comm);
    SlenderElemList<Real> arms = A.slender(comm);
    const std::string tag = "vis/ybifurc-vessels-ord" + std::to_string((long)ord) + "-nref" + std::to_string((long)nref);

    // ---- geometry report + collision guard (min clearance between non-adjacent tubes / junction bodies) ----
    if (!comm.Rank()) {
      Vector<Real> Xj, Xa; junc.GetNodeCoord(&Xj, nullptr, nullptr); arms.GetNodeCoord(&Xa, nullptr, nullptr);
      std::cout << "\n[geometry] junctions=" << NJ << " intra-tree arms=" << (NJ-2) << " connectors=" << NC
                << " capped roots=" << n_caps << "\n  quad panels=" << junc.Size() << " nodes=" << Xj.Dim()/3
                << " | slender panels=" << arms.Size() << " nodes=" << Xa.Dim()/3 << "\n";
      const Real Jrad_can = (Real)1.3;   // canonical junction blob outer-radius proxy
      Real minclear = std::numeric_limits<Real>::max(); int mi=-1, mj=-1;
      for (size_t a = 0; a < segs.size(); a++)
        for (size_t b = a+1; b < segs.size(); b++) {
          const bool adjacent = (segs[a].j0==segs[b].j0)||(segs[a].j0==segs[b].j1)||
                                (segs[a].j1==segs[b].j0)||(segs[a].j1==segs[b].j1);
          if (adjacent) continue;
          const Real d = seg_seg_dist<Real>(segs[a].A, segs[a].B, segs[b].A, segs[b].B);
          const Real clear = d - (segs[a].rtube + segs[b].rtube);
          if (clear < minclear) { minclear = clear; mi=(int)a; mj=(int)b; }
        }
      Real minJclear = std::numeric_limits<Real>::max();
      for (int i = 0; i < NJ; i++) {
        const Vec3<Real> C = P[i].apply_point(Vec3<Real>{0,0,0}); const Real Rb = scale[i]*Jrad_can;
        for (const auto& s : segs) {
          if (s.j0==i || s.j1==i) continue;
          minJclear = std::min(minJclear, seg_seg_dist<Real>(C, C, s.A, s.B) - (Rb + s.rtube));
        }
      }
      std::cout << "  [collision] min arm-arm clearance=" << std::setprecision(4) << minclear
                << " (segs " << mi << "," << mj << ")   min junction-arm clearance=" << minJclear
                << std::setprecision(6) << "\n";
      if (minclear <= 0 || minJclear <= 0)
        std::cout << "  *** WARNING: geometry self-intersection (clearance <= 0) -- increase svg_scale ***\n";
    }

    junc.WriteVTK(tag + "-junc", Vector<Real>(), comm);
    arms.WriteVTK(tag + "-arms", Vector<Real>(), comm);

    if (geomOnly) {
      if (!comm.Rank()) std::cout << "\n[geom-only] meshes written to " << tag << "-{junc,arms}.vtu; skipping BIE.\n";
      Comm::MPI_Finalize();
      return 0;
    }

    // ---- exterior Green sources: one candidate per junction, kept only if exterior to the WHOLE assembly ----
    Vector<Real> X0;
    {
      const YField<Real> fld;
      auto exterior = [&](const Vec3<Real>& Xs) -> bool {
        for (int i = 0; i < NJ; i++) if (fld.f(P[i].apply_inverse_point(Xs)) >= level) return false;
        for (const auto& s : segs) if (seg_seg_dist<Real>(Xs, Xs, s.A, s.B) <= s.rtube) return false;
        return true;
      };
      for (int i = 0; i < NJ; i++) {
        const Vec3<Real> Xs = P[i].apply_point(Vec3<Real>{1.6, 1.4, 0.9});
        if (exterior(Xs)) { X0.PushBack(Xs[0]); X0.PushBack(Xs[1]); X0.PushBack(Xs[2]); }
      }
      if (!comm.Rank()) std::cout << "  [green] kept " << X0.Dim()/3 << " / " << NJ << " exterior sources\n";
      SCTL_ASSERT_MSG(X0.Dim() > 0, "no exterior Green source survived validation.");
    }

    // ---- BIE verification: divergence + DL(=-1/2) watertightness + Green's identity, Laplace & Stokes ----
    const RegionReport<Real> region_report = [](const Vector<Real>& err, Long Nj, Long Na) {
      Real mj = 0, ma = 0;
      for (Long i = 0; i < Nj; i++) mj = std::max(mj, err[i]);
      for (Long i = 0; i < Na; i++) ma = std::max(ma, err[Nj+i]);
      std::cout << "    [region max] quad(junctions+transitions+caps)=" << mj << " slender(arms)=" << ma << "\n";
    };

    junc.SetQuadScheme(QuadElemList<Real>::QuadScheme::Hybrid, cov_q, Nbeta, maxdep);
    if (!comm.Rank())
      std::cout << "\n---- BIE verification [tol=" << tol << " Nbeta=" << Nbeta << " max_depth=" << maxdep
                << " cov_q=" << cov_q << "] ----\n";
    // QJ_VESSELS_TESTS is a comma-separated subset of {div,dl_lap,dl_stk,gr_lap,gr_stk} (default: all).
    // Added for the SCTL_FMM_CHECK diagnostic, where every BoundaryIntegralOp apply also runs an
    // O(Ns*Nt) direct sum -- Stokes costs ~9x Laplace per pair, so being able to run the Laplace
    // identities alone turns a multi-hour job into a few minutes without changing what is measured.
    const std::string tests = std::getenv("QJ_VESSELS_TESTS") ? std::getenv("QJ_VESSELS_TESTS")
                                                              : "div,dl_lap,dl_stk,gr_lap,gr_stk";
    auto want = [&tests](const std::string& t) { return tests.find(t) != std::string::npos; };
    if (!comm.Rank() && tests != "div,dl_lap,dl_stk,gr_lap,gr_stk")
      std::cout << "  [QJ_VESSELS_TESTS] running subset: " << tests << "\n";

    if (want("div")) divergence_check<Real>(junc, arms, tol, comm);

    if (want("dl_lap")) {
      if (!comm.Rank()) std::cout << "  [DL const  Laplace] ";
      test_DLIdentity<Real, Laplace3D_DxU>(junc, arms, comm, tol, tag+"-dl-laplace", region_report);
    }
    if (want("dl_stk")) {
      if (!comm.Rank()) std::cout << "  [DL const  Stokes ] ";
      test_DLIdentity<Real, Stokes3D_DxU>(junc, arms, comm, tol, tag+"-dl-stokes", region_report);
    }
    if (want("gr_lap")) {
      if (!comm.Rank()) std::cout << "  [Green     Laplace] ";
      test_greens_identity<Real, Laplace3D_FxU, Laplace3D_DxU, Laplace3D_FxdU>(junc, arms, comm, tol, X0, tag+"-green-laplace");
    }
    if (want("gr_stk")) {
      if (!comm.Rank()) std::cout << "  [Green     Stokes ] ";
      test_greens_identity<Real, Stokes3D_FxU, Stokes3D_DxU, Stokes3D_FxT>(junc, arms, comm, tol, X0, tag+"-green-stokes");
    }
  }
  Comm::MPI_Finalize();
  return 0;
}
