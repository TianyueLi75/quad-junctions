/**
 * bifurc-network-flow-bie -- physical Stokes inflow/outflow BVP on the LARGE vmtk-derived vessel network
 * (the ".obj network": 160 quad junctions + bent CSBQ slender arms + 177 hemisphere-capped leaves),
 * loaded from the per-junction bundles that bifurc-network-assemble wrote (<prefix>-jNNN.{mesh,arms}).
 *
 * This is the network analogue of src/ybifurc-vessels-flow-bie.cpp (the hand-authored 20-junction SVG
 * network). The graph is a TREE (338 nodes, 337 edges => no independent cycles), so the fluid lumen is
 * simply connected and GMRES converges like a single bifurcation -- not the genus-10 stall of the SVG net.
 *
 *   - LOAD the whole assembled network from the bundles (no re-meshing) into one coupled operator:
 *       QuadElemList "0_junc" (all junction bodies + their hemisphere leaf caps) and
 *       SlenderElemList "1_arms" (all bent tapered centerline tubes). Both MPI-partitioned.
 *   - Identify the inflow/outflow PORTS by connectivity: every degree-1 leaf is a hemisphere cap. Its
 *     dome center C / outward axis u / radius R0 are reconstructed EXACTLY from the cap arm (Lagrange
 *     extrapolation of the terminal panel to s=1 -- the cap dome is centered at that centerline endpoint).
 *   - By default a SINGLE inflow = the extreme-coordinate cap (the vmtk traversal root / main trunk),
 *     every other cap an outflow; the outflow flux is split (equal by default) and normalized so the net
 *     flux int u.n dA = 0 (the interior incompressible-Stokes Dirichlet compatibility condition). Each
 *     port carries a flux-normalized PARABOLIC (Poiseuille) velocity profile; all other walls are no-slip.
 *   - SOLVE the combined-field interior Stokes Dirichlet BVP ( -1/2 I - S + D ) sigma = u_bc via GMRES
 *     (the shared solve_dirichlet_bvp in hybrid_bie_tests.hpp), evaluate the interior velocity at a
 *     geometry-aware cloud (arm cross-section stars + per-junction boxes filtered to the interior).
 *
 * TEST MODES (QJ_NET_MODE, default "flow"):
 *   "flow"          the physical inflow/outflow BVP described above (stages 4-9).
 *   "manufactured"  (alias "mfg") an analytic-field acceptance sweep on the SAME coupled operator:
 *                   two manufactured-solution tests over a quadrature-tol sweep --
 *                     mfg-int: sources EXTERIOR (on the enclosing sphere, radius padded 0.2x) => the
 *                              field is smooth in the interior lumen => interior jump -1/2 (SL=-1,DL=+1,
 *                              the flow operator); checked at interior arm-centerline points.
 *                     mfg-ext: sources INTERIOR (arm centerlines) => smooth in the exterior => exterior
 *                              jump +1/2 (SL=+1,DL=+1); checked on the enclosing sphere.
 *                   The RHS is built by test_manufactured (hybrid_bie_tests.hpp) from combined_nodes,
 *                   so it is automatically in the "0_junc"->"1_arms" node order. rel-L2 is capped by
 *                   the geometry watertightness floor (~1e-6 for the netfix build). Only tol is swept
 *                   (order/cheb/fourier are baked into the bundles; Duffy ignores cov_q/Nbeta/max_depth).
 *                   Env: QJ_MFG_TOLS (comma list; default 1e-4,1e-5,1e-6,1e-7,1e-8,1e-9), QJ_MFG_NLON,
 *                   QJ_MFG_NLAT (sphere grid, ~8x8), QJ_MFG_JITTER (source offset frac of R_encl, 0.02),
 *                   QJ_MFG_SEED. QJ_MFG_DRYRUN builds+validates the clouds then exits before the solve
 *                   (cheap local smoke). Runs on the cluster via scripts/submit-network-mfg.sh (PVFMM).
 *
 * Env overrides for the "flow" port assignment:
 *   QJ_INFLOW_NODES="id,id,..."   graph node ids (the cap `other_node` in the bundle) to force as INFLOWs
 *                                 (split equally); default = the single extreme-axis cap.
 *   QJ_INFLOW_AXIS=x|y|z|x-|...   axis whose EXTREME cap is the default inflow (default "x" = max-x).
 *   QJ_OUTFLOW_FLUX="w1,w2,..."   relative outflow weights in printed order (missing -> 1; unset -> equal).
 *
 *   make MPI=1 bin/bifurc-network-flow-bie          # or: make PVFMM=1 bin/bifurc-network-flow-bie
 *   OMP_NUM_THREADS=8 mpirun -n <ranks> ./bin/bifurc-network-flow-bie \
 *       [bundle_prefix] [tol] [p_in] [cov_q] [Nbeta] [max_depth] [gmres_max_iter] [Nvis] [Ngrid]
 *   e.g.  ... ./bin/bifurc-network-flow-bie geom/netfix 1e-7 10   (prefix = whatever you assembled to)
 *
 * The bundles bake in the DISCRETIZATION (order + fourier), so accuracy is dialed by choosing the bundle
 * set plus the near-eval tol / cov_q / Nbeta / max_depth here.
 *
 * GEOMETRY PROVENANCE -- the production bundles live at geom/netfix-jNNN.{mesh,arms} (a build artifact,
 * NOT committed -- only the geom/ generalized-bifurcation presets are; regenerate before a production run):
 *   The CANONICAL geometry is the assembler run on data/vmtk/vessels_fixed.graph (TRUE per-junction branch
 *   angles + every watertightness fix -- bigon3, lead-corner arm bend, leaf-arm fix, turn-adaptive corners,
 *   sigma-floor 0.075->0.05, auto size-shrink). It closes to |int n dA| ~1e-3 on area ~5e5 (rel ~1e-5).
 *   PRODUCTION ASSEMBLE PARAMETERS (bifurc-network-assemble): order=12 cheb=10 fourier=24 for 6-digit quadrature:
 *       data/vmtk/vessels_fixed.graph  geom/netfix  12 1 1.5 0.4 3 12 10 24 2
 *       (nref=1 level=1.5 eta_join=0.4 Ns_trans=3 n_axial=12 cheb=10 lead_panels=2; all other flags default)
 *   The driver prints the combined watertightness |int n dA| up front so you can confirm you built the fixed
 *   geometry (~1e-5) before trusting the solve -- a rel ~2.8e-3 closure means an angle-approximated/stale
 *   build, whose conformity floor caps the solve regardless of quadrature.
 */

#include <csbq.hpp>                                  // CSBQ SlenderElemList
#include <quad_junctions/gen_network_geom.hpp>       // ReadNetworkBundle + NetworkArmBundle
#include <quad_junctions/quad_scheme.hpp>            // QJDefaultScheme (Duffy default)
#include <quad_junctions/hybrid_bie_tests.hpp>       // combined_nodes + divergence_check + solve_dirichlet_bvp
#include <quad_junctions/interior_viz.hpp>           // build_arm_panel_targets / build_box_targets / write_points_vtu
#include <quad_junctions/vessels_build.hpp>          // dot3/nrm3/sub3/unit3/add3/mul3
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

using namespace sctl;
using namespace quad_junctions;

namespace {
using Real = double;

// One inflow/outflow port (a leaf hemisphere cap): dome center C, outward axis u, radius R0, signed
// amplitude amp (= sgn*p/g so the flux through the cap is sgn*p), prescribed flux magnitude p, sign
// (-1 inflow / +1 outflow), and the graph node id (`other_node` of the cap arm) for env selection.
struct FlowCap { Vec3<Real> C, u; Real R0 = 0, amp = 0, p = 0; int sgn = 0; Integer node = -1; Integer owner = -1; };

// Parabolic axial profile prof(X) = 1 - (r/R0)^2 for X on the cap dome (|X-C| ~ R0 on the +u side), else
// 0. Same 5% shell band as ybifurc-vessels-flow-bie: arm wall nodes at the equator have prof~0, so the
// benign overlap with the cap ring does not double-drive the seam.
bool cap_profile(const Vec3<Real>& X, const FlowCap& c, Real& prof) {
  const Vec3<Real> d = sub3(X, c.C);
  const Real ax = dot3(d, c.u), dist2 = dot3(d, d), dist = sqrt<Real>(dist2);
  if (std::fabs((double)(dist - c.R0)) < (double)((Real)0.05*c.R0) && ax > (Real)-0.05*c.R0) {
    Real r2 = dist2 - ax*ax; if (r2 < 0) r2 = 0;
    prof = (Real)1 - r2/(c.R0*c.R0); if (prof < 0) prof = 0;
    return true;
  }
  prof = 0; return false;
}

// Prescribed boundary velocity at X: v = amp*prof*u on the owning cap, else 0 (no-slip).
Vec3<Real> flow_bc_vel(const Vec3<Real>& X, const std::vector<FlowCap>& caps) {
  for (const auto& c : caps) { Real prof; if (cap_profile(X, c, prof)) { const Real s = c.amp*prof; return mul3(s, c.u); } }
  return Vec3<Real>{(Real)0, (Real)0, (Real)0};
}

// Lagrange extrapolation weights from `cheb` Chebyshev centerline nodes to a target local coord s.
Vector<Real> extrap_weights(Long cheb, Real s) {
  Vector<Real> wts, trg(1); trg[0] = s;
  LagrangeInterp<Real>::Interpolate(wts, SlenderElemList<Real>::CenterlineNodes((Integer)cheb), trg);
  return wts;
}

// Smallest axis-aligned-bbox-centered sphere enclosing the whole (MPI-partitioned) network: center =
// bbox midpoint, radius = max node distance to it. Reduced across ranks so every rank agrees.
void enclosing_sphere(const QuadElemList<Real>& junc, const SlenderElemList<Real>& arms,
                      const Comm& comm, Vec3<Real>& center, Real& radius) {
  Vector<Real> X, Xn; Long Nj, Na; combined_nodes(junc, arms, X, Xn, Nj, Na);
  const Long N = X.Dim()/3;
  double lo[3] = {1e300, 1e300, 1e300}, hi[3] = {-1e300, -1e300, -1e300};
  for (Long i = 0; i < N; i++) for (int k = 0; k < 3; k++) { const double v = (double)X[3*i+k]; lo[k] = std::min(lo[k], v); hi[k] = std::max(hi[k], v); }
  for (int k = 0; k < 3; k++) { lo[k] = GlobalReduce(lo[k], comm, CommOp::MIN); hi[k] = GlobalReduce(hi[k], comm, CommOp::MAX); center[k] = (Real)(0.5*(lo[k]+hi[k])); }
  double r2 = 0;
  for (Long i = 0; i < N; i++) { double d2 = 0; for (int k = 0; k < 3; k++) { const double d = (double)X[3*i+k] - (double)center[k]; d2 += d*d; } r2 = std::max(r2, d2); }
  radius = (Real)std::sqrt(GlobalReduce(r2, comm, CommOp::MAX));
}

// Manufactured-solution acceptance on the loaded network -- TWO tests over a quadrature-tol sweep:
//   test 1 "mfg-int": point sources placed EXTERIOR to the whole network (on the enclosing sphere,
//                     radius padded 0.2x) => field smooth in the interior lumen => interior jump -1/2
//                     (SL=-1, DL=+1: the SAME operator the flow solve uses); checked at interior
//                     arm-centerline points.
//   test 2 "mfg-ext": point sources placed INTERIOR (arm centerlines = tube axes) => field smooth in
//                     the exterior => exterior jump +1/2 (SL=+1, DL=+1); checked on the enclosing
//                     sphere (exterior to every tube).
// The Dirichlet RHS is built INSIDE test_manufactured from combined_nodes(junc,arms), so it is
// automatically in the "0_junc"->"1_arms" node ordering (no manual RHS assembly). Xsrc/Fsrc are built
// identically on every rank (fixed seed) since test_manufactured uses them in the per-rank RHS Eval;
// the check clouds Xtrg live on rank 0 only (the routine GlobalReduce-sums, expecting each target once).
// Only tol is swept (order/cheb/fourier are baked into the bundles; the Duffy near-eval ignores
// cov_q/Nbeta/max_depth). rel-L2 is capped by the geometry watertightness floor (~1e-6 for the netfix
// build): expect it to track tol down to that floor, then plateau.
// Env: QJ_MFG_TOLS (comma list), QJ_MFG_NLON/NLAT (sphere grid, ~8x8), QJ_MFG_JITTER (source offset as
// a fraction of R_encl, default 0.02), QJ_MFG_SEED.
void run_network_manufactured(const QuadElemList<Real>& junc, const SlenderElemList<Real>& arms,
                              const Vector<Real>& a_coord, const Comm& comm, const Real tol_fallback,
                              const Long gmaxit, const Integer order, const Long cheb, const Long fourier) {
  const Integer pid = comm.Rank();
  const Real PI = (Real)3.14159265358979323846;

  // (a) enclosing sphere, padded 0.2x.
  Vec3<Real> ctr{(Real)0,(Real)0,(Real)0}; Real rad = 0; enclosing_sphere(junc, arms, comm, ctr, rad);
  const Real R_encl = (Real)1.2 * rad;

  // (b) knobs.
  const Integer Nlon = std::getenv("QJ_MFG_NLON") ? (Integer)atoi(std::getenv("QJ_MFG_NLON")) : 8;
  const Integer Nlat = std::getenv("QJ_MFG_NLAT") ? (Integer)atoi(std::getenv("QJ_MFG_NLAT")) : 8;
  const Real jitter  = std::getenv("QJ_MFG_JITTER") ? (Real)atof(std::getenv("QJ_MFG_JITTER")) : (Real)0.02;
  const long seed    = std::getenv("QJ_MFG_SEED") ? atol(std::getenv("QJ_MFG_SEED")) : 12345L;
  // GMRES stop tol, decoupled from the swept quadrature tol. Default (<=0) keeps test_manufactured's
  // historical tol*10; set QJ_MFG_GMRES_TOL to pin the Krylov tolerance regardless of the near-eval tol.
  const Real gmres_tol = std::getenv("QJ_MFG_GMRES_TOL") ? (Real)atof(std::getenv("QJ_MFG_GMRES_TOL")) : (Real)-1;
  // Sweep list: QJ_MFG_TOLS if set, else a default sweep bracketing the ~1e-6 floor and probing below.
  // (The CLI positional tol is not used to drive the sweep -- it always has a default; kept for signature
  // stability and as a last-resort single value only if the default list is ever emptied.)
  (void)tol_fallback;
  std::vector<Real> tols;
  if (const char* e = std::getenv("QJ_MFG_TOLS")) { std::stringstream ss(e); std::string t;
    while (std::getline(ss, t, ',')) if (!t.empty()) tols.push_back((Real)atof(t.c_str())); }
  if (tols.empty()) { const double d[] = {1e-4,1e-5,1e-6,1e-7,1e-8,1e-9}; for (double x : d) tols.push_back((Real)x); }

  // unit sphere direction from a (lat,lon) grid index; the +0.5 lat offset avoids the poles.
  auto sph_dir = [PI](Integer ila, Integer Nla, Integer ilo, Integer Nlo) -> Vec3<Real> {
    const Real phi = PI * ((Real)ila + (Real)0.5) / (Real)Nla;      // polar 0..pi
    const Real th  = (Real)2 * PI * (Real)ilo / (Real)Nlo;          // azimuth
    return Vec3<Real>{ std::sin(phi)*std::cos(th), std::sin(phi)*std::sin(th), std::cos(phi) };
  };

  // (c) EXTERIOR sources (test 1): sphere grid @ R_encl + jitter; strengths ~ U[-5,5]. Fixed seed +
  //     identical loop order => byte-identical on every rank.
  Vector<Real> Xs_ext, Fs_ext; srand48(seed);
  for (Integer ila = 0; ila < Nlat; ila++) for (Integer ilo = 0; ilo < Nlon; ilo++) {
    const Vec3<Real> d = sph_dir(ila, Nlat, ilo, Nlon);
    for (int k = 0; k < 3; k++) Xs_ext.PushBack(ctr[k] + R_encl*d[k] + (Real)(jitter*R_encl*(Real)(2*drand48()-1)));
    for (int k = 0; k < 3; k++) Fs_ext.PushBack((Real)(10.0*drand48()-5.0));
  }
  { const Long M = Xs_ext.Dim()/3; double dmin = 1e300;
    for (Long i = 0; i < M; i++) { double d2 = 0; for (int k = 0; k < 3; k++) { const double dd = (double)Xs_ext[3*i+k]-(double)ctr[k]; d2 += dd*dd; } dmin = std::min(dmin, std::sqrt(d2)); }
    SCTL_ASSERT_MSG(dmin > (double)rad, "manufactured exterior source fell inside the network bounding sphere"); }

  // (d) INTERIOR sources (test 2): stride-sampled arm centerlines (a_coord = tube axes). EVERY stored
  //     centerline node is interior to the closed surface by construction: nodes lie on the tube axis at
  //     s<1 (the cap dome center is the terminal panel extrapolated to s=1, BEYOND the last node), so they
  //     end inside the tube before the convex cap is attached -- and the junction-end nodes sit inside the
  //     junction solid the arm plugs into. No containment test / end-trim needed. Strengths ~ U[-5,5] under
  //     a disjoint seed; replicated on every rank (fixed seed + identical loop order => byte-identical).
  const Long Ncen = a_coord.Dim()/3;
  const Long Mint = (Long)Nlon*(Long)Nlat;
  Vector<Real> Xs_int, Fs_int; srand48(seed + 7);
  if (Ncen > 0) { const Long stride = std::max<Long>(1, Ncen/std::max<Long>(1, Mint));
    for (Long i = 0; i < Ncen && (Long)(Xs_int.Dim()/3) < Mint; i += stride) {
      for (int k = 0; k < 3; k++) Xs_int.PushBack(a_coord[3*i+k]);
      for (int k = 0; k < 3; k++) Fs_int.PushBack((Real)(10.0*drand48()-5.0)); } }

  // (e) check clouds -- rank 0 ONLY.
  Vector<Real> Xt_int, Xt_ext;
  if (!pid) {
    const Long want = std::min<Long>(256, Ncen);   // interior checks: centerlines, offset from the sources.
    if (Ncen > 0 && want > 0) { const Long stride = std::max<Long>(1, Ncen/want);
      for (Long i = stride/2; i < Ncen; i += stride) for (int k = 0; k < 3; k++) Xt_int.PushBack(a_coord[3*i+k]); }
    const Integer Cla = 16, Clo = 16;              // exterior checks: sphere grid @ R_encl.
    for (Integer ila = 0; ila < Cla; ila++) for (Integer ilo = 0; ilo < Clo; ilo++) {
      const Vec3<Real> d = sph_dir(ila, Cla, ilo, Clo);
      for (int k = 0; k < 3; k++) Xt_ext.PushBack(ctr[k] + R_encl*d[k]); }
  }

  if (!pid)
    std::cout << "\n=== manufactured-solution acceptance on the loaded network (2 tests, tol sweep) ===\n"
              << "  geometry (baked, NOT swept): order=" << order << " cheb=" << cheb << " fourier=" << fourier << "\n"
              << "  enclosing sphere: center=(" << std::setprecision(4) << ctr[0] << "," << ctr[1] << "," << ctr[2]
              << ") radius=" << rad << "  => sources/checks @ R_encl=" << R_encl << " (pad 0.2x)\n"
              << "  test 1 mfg-int: " << (Xs_ext.Dim()/3) << " EXTERIOR sources, interior jump -1/2 (SL=-1,DL=+1), "
              << (Xt_int.Dim()/3) << " interior checks\n"
              << "  test 2 mfg-ext: " << (Xs_int.Dim()/3) << " INTERIOR sources, exterior jump +1/2 (SL=+1,DL=+1), "
              << (Xt_ext.Dim()/3) << " exterior checks\n"
              << "  NOTE: rel-L2 is capped by the geometry watertightness floor (~1e-6 for the netfix build);\n"
              << "        expect it to track tol down to that floor then plateau. GMRES iters print per test.\n";

  // QJ_MFG_DRYRUN: build + validate the source/target clouds and exit BEFORE the GMRES solve. The full
  // solve is a ~22M-DOF multi-node PVFMM job (not local); this is the cheap local smoke of the setup.
  if (std::getenv("QJ_MFG_DRYRUN")) {
    if (!pid) std::cout << "\n[QJ_MFG_DRYRUN] clouds built + sources validated exterior/interior; skipping solve.\n";
    return;
  }

  // QJ_MFG_ONLY selects which test(s) run: "int" (mfg-int only), "ext" (mfg-ext only), or "both" (default).
  // The exterior-source interior problem (mfg-int) converges in tens of GMRES iters; the interior-source
  // exterior problem (mfg-ext) is far worse conditioned (thousands of iters at eta=1) and can starve a
  // time-limited sweep -- run them separately when that matters.
  const char* only_env = std::getenv("QJ_MFG_ONLY");
  const std::string only = only_env ? std::string(only_env) : std::string("both");
  SCTL_ASSERT_MSG(only=="int"||only=="ext"||only=="both", "QJ_MFG_ONLY must be 'int', 'ext', or 'both'");
  const bool do_int = (only != "ext"), do_ext = (only != "int");
  if (!pid) std::cout << "  QJ_MFG_ONLY=" << only << " -> running "
                      << (do_int ? "mfg-int " : "") << (do_ext ? "mfg-ext" : "") << "\n";
  if (!pid) {
    std::cout << "  GMRES stop tol = ";
    if (gmres_tol > (Real)0) std::cout << "QJ_MFG_GMRES_TOL " << std::setprecision(1) << (double)gmres_tol;
    else                     std::cout << "tol*10 (default, per swept tol)";
    std::cout << "  (max_iter=" << gmaxit << ")\n";
  }

  std::vector<Real> r1(tols.size(), (Real)-1), r2(tols.size(), (Real)-1);
  for (size_t it = 0; it < tols.size(); it++) {
    const Real tl = tols[it];
    if (!pid) std::cout << "\n----- tol = " << std::setprecision(1) << (double)tl << " -----\n";
    if (do_int)
      r1[it] = test_manufactured<Real, Stokes3D_FxU, Stokes3D_DxU>(junc, arms, comm, tl, Xs_ext, Fs_ext,
                   /*interior=*/true,  Xt_int, /*SL_scal=*/(Real)-1., /*DL_scal=*/(Real)1., "mfg-int", gmaxit, gmres_tol);
    if (do_ext)
      r2[it] = test_manufactured<Real, Stokes3D_FxU, Stokes3D_DxU>(junc, arms, comm, tl, Xs_int, Fs_int,
                   /*interior=*/false, Xt_ext, /*SL_scal=*/(Real) 1., /*DL_scal=*/(Real)1., "mfg-ext", gmaxit, gmres_tol);
  }

  if (!pid) {
    std::cout << "\n=== manufactured convergence summary (rel-L2; GMRES iters in the per-test lines above) ===\n"
              << "           tol      mfg-int (exterior src)   mfg-ext (interior src)\n";
    for (size_t it = 0; it < tols.size(); it++) {
      std::cout << "  " << std::setw(12) << std::setprecision(2) << (double)tols[it] << "         ";
      if (r1[it] < (Real)0) std::cout << std::setw(12) << "(skipped)";
      else                  std::cout << std::setw(12) << std::setprecision(6) << (double)r1[it];
      std::cout << "            ";
      if (r2[it] < (Real)0) std::cout << std::setw(12) << "(skipped)";
      else                  std::cout << std::setw(12) << std::setprecision(6) << (double)r2[it];
      std::cout << "\n";
    }
  }
}

// On-surface Green's third-identity acceptance on the loaded network -- a DIRECT quadrature check (NO
// GMRES, NO solve). For a known field u (the single-layer potential of point sources) we verify the
// on-surface representation  S[du/dn] - D_pv[u] = +-0.5 u  over the WHOLE coupled surface:
//   * EXTERIOR source: sources on the padded enclosing sphere (field regular in the interior lumen).
//                      test_greens_identity default jump -0.5.
//   * INTERIOR source: sources on the arm centerlines (= tube axes, inside the closed surface by
//                      construction; field regular in the exterior). jump -1.5.
// The reported number is a RELATIVE error (max|err|/max|u|), hence scale-invariant -- identical on the
// 10x-down geometry. Env: QJ_GREENS_KER=laplace|stokes|both (default both), QJ_GREENS_NLON/NLAT (source
// grid, default 6x6), QJ_GREENS_JITTER (0.02), QJ_GREENS_SEED.
void run_network_greens(const QuadElemList<Real>& junc, const SlenderElemList<Real>& arms,
                        const Vector<Real>& a_coord, const Comm& comm, const Real tol) {
  const Integer pid = comm.Rank();
  const Real PI = (Real)3.14159265358979323846;

  Vec3<Real> ctr{(Real)0,(Real)0,(Real)0}; Real rad = 0; enclosing_sphere(junc, arms, comm, ctr, rad);
  const Real R_encl = (Real)1.2 * rad;
  const Integer Nlon = std::getenv("QJ_GREENS_NLON") ? (Integer)atoi(std::getenv("QJ_GREENS_NLON")) : 6;
  const Integer Nlat = std::getenv("QJ_GREENS_NLAT") ? (Integer)atoi(std::getenv("QJ_GREENS_NLAT")) : 6;
  const Real jitter  = std::getenv("QJ_GREENS_JITTER") ? (Real)atof(std::getenv("QJ_GREENS_JITTER")) : (Real)0.02;
  const long seed    = std::getenv("QJ_GREENS_SEED") ? atol(std::getenv("QJ_GREENS_SEED")) : 12345L;

  auto sph_dir = [PI](Integer ila, Integer Nla, Integer ilo, Integer Nlo) -> Vec3<Real> {
    const Real phi = PI * ((Real)ila + (Real)0.5) / (Real)Nla;
    const Real th  = (Real)2 * PI * (Real)ilo / (Real)Nlo;
    return Vec3<Real>{ std::sin(phi)*std::cos(th), std::sin(phi)*std::sin(th), std::cos(phi) };
  };

  // EXTERIOR sources: padded enclosing sphere (validated strictly outside the network bbox sphere).
  Vector<Real> Xs_ext; srand48(seed);
  for (Integer ila = 0; ila < Nlat; ila++) for (Integer ilo = 0; ilo < Nlon; ilo++) {
    const Vec3<Real> d = sph_dir(ila, Nlat, ilo, Nlon);
    for (int k = 0; k < 3; k++) Xs_ext.PushBack(ctr[k] + R_encl*d[k] + (Real)(jitter*R_encl*(Real)(2*drand48()-1)));
  }
  { const Long M = Xs_ext.Dim()/3; double dmin = 1e300;
    for (Long i = 0; i < M; i++) { double d2 = 0; for (int k = 0; k < 3; k++) { const double dd = (double)Xs_ext[3*i+k]-(double)ctr[k]; d2 += dd*dd; } dmin = std::min(dmin, std::sqrt(d2)); }
    SCTL_ASSERT_MSG(dmin > (double)rad, "greens exterior source fell inside the network bounding sphere"); }

  // INTERIOR sources: stride-sampled arm centerlines (tube axes -- interior by construction, see mfg note).
  const Long Ncen = a_coord.Dim()/3, Mint = (Long)Nlon*(Long)Nlat;
  Vector<Real> Xs_int;
  if (Ncen > 0) { const Long stride = std::max<Long>(1, Ncen/std::max<Long>(1, Mint));
    for (Long i = 0; i < Ncen && (Long)(Xs_int.Dim()/3) < Mint; i += stride)
      for (int k = 0; k < 3; k++) Xs_int.PushBack(a_coord[3*i+k]); }

  const char* ker_env = std::getenv("QJ_GREENS_KER");
  const std::string ker = ker_env ? std::string(ker_env) : std::string("both");
  SCTL_ASSERT_MSG(ker=="laplace"||ker=="stokes"||ker=="both", "QJ_GREENS_KER must be 'laplace','stokes','both'");
  const bool do_lap = (ker != "stokes"), do_stk = (ker != "laplace");

  if (!pid) std::cout << "\n=== on-surface Green's identity acceptance on the loaded network (direct, no solve) ===\n"
                      << "  tol=" << std::setprecision(1) << (double)tol << "  kernels=" << ker
                      << "  enclosing sphere r=" << std::setprecision(4) << rad << " => exterior src @ R_encl=" << R_encl << "\n"
                      << "  " << (Xs_ext.Dim()/3) << " EXTERIOR sources (jump -1/2), "
                      << (Xs_int.Dim()/3) << " INTERIOR sources (jump +1/2)\n";

  // QJ_GREENS_DLONLY: run ONLY the constant-density DL identity (both ext -1/2 and int +1/2 formulations,
  // reported together by test_DLIdentity) for the requested kernel(s), then return -- no Green's third
  // identity. This is the cheap geometry-floor gate (watertightness ran before the mode switch): a single
  // forward DL apply per kernel, so Laplace fits on far fewer nodes than the Stokes Green's sweep.
  if (std::getenv("QJ_GREENS_DLONLY")) {
    // QJ_GREENS_DUMP=<prefix>: also write <prefix>-<ker>-{junc,arms}.vtu colored by the per-target DL error,
    // for a ParaView spatial map. test_DLIdentity always prints the MPI-safe global top-20 error nodes.
    const char* dp = std::getenv("QJ_GREENS_DUMP");
    const std::string dbase = dp ? std::string(dp) : std::string();
    if (do_lap) {
      if (!pid) std::cout << "\n--- Laplace DL const-density identity (ext -1/2 & int +1/2) ---" << std::endl;
      test_DLIdentity<Real, Laplace3D_DxU>(junc, arms, comm, tol, dbase.empty()?"":dbase+"-lap");
    }
    if (do_stk) {
      if (!pid) std::cout << "\n--- Stokes DL const-density identity (ext -1/2 & int +1/2) ---" << std::endl;
      test_DLIdentity<Real, Stokes3D_DxU>(junc, arms, comm, tol, dbase.empty()?"":dbase+"-stk");
    }
    return;
  }

  const Real J_EXT = (Real)-0.5, J_INT = (Real)-1.5;   // BIOp returns PV DL; add the interior/exterior jump
  const bool warm = std::getenv("QJ_GREENS_WARMUP");   // default: skip the benchmark warm-up (halves cost)
  // QJ_GREENS_DUMP=<prefix> (full-Green's path only): write <prefix>-<ker>-<ext|int>-{junc,arms}.vtu colored by
  // the per-target Green's error, one pair per (kernel, source) config so none overwrites another. The DLONLY
  // branch above instead dumps the DL const-density error (unchanged) and never reaches here.
  const char* gdp = std::getenv("QJ_GREENS_DUMP");
  const std::string gbase = gdp ? std::string(gdp) : std::string();
  const auto gtag = [&gbase](const std::string& suf) {
    return gbase.empty() ? std::string() : gbase + suf;
  };
  if (do_lap) {
    if (!pid) std::cout << "\n--- Laplace DL const-density identity (closed outward surface -> -1/2) ---" << std::endl;
    test_DLIdentity<Real, Laplace3D_DxU>(junc, arms, comm, tol);
    if (!pid) std::cout << "\n--- Laplace, EXTERIOR source ---" << std::endl;
    test_greens_identity<Real, Laplace3D_FxU, Laplace3D_DxU, Laplace3D_FxdU>(junc, arms, comm, tol, Xs_ext, gtag("-lap-ext"), J_EXT, warm);
    if (!pid) std::cout << "\n--- Laplace, INTERIOR source ---" << std::endl;
    test_greens_identity<Real, Laplace3D_FxU, Laplace3D_DxU, Laplace3D_FxdU>(junc, arms, comm, tol, Xs_int, gtag("-lap-int"), J_INT, warm);
  }
  if (do_stk) {
    if (!pid) std::cout << "\n--- Stokes, EXTERIOR source ---" << std::endl;
    test_greens_identity<Real, Stokes3D_FxU, Stokes3D_DxU, Stokes3D_FxT>(junc, arms, comm, tol, Xs_ext, gtag("-stk-ext"), J_EXT, warm);
    if (!pid) std::cout << "\n--- Stokes, INTERIOR source ---" << std::endl;
    test_greens_identity<Real, Stokes3D_FxU, Stokes3D_DxU, Stokes3D_FxT>(junc, arms, comm, tol, Xs_int, gtag("-stk-int"), J_INT, warm);
  }
}

} // anonymous namespace

int main(int argc, char** argv) {
  Comm::MPI_Init(&argc, &argv);
  {
    const Comm comm = Comm::World();
    const Integer Np = comm.Size(), pid = comm.Rank();

    const std::string prefix = (argc > 1) ? std::string(argv[1]) : std::string("geom/netfix");
    const Real    tol     = (argc > 2) ? (Real)atof(argv[2]) : (Real)1e-7;
    const Real    p_in    = (argc > 3) ? (Real)atof(argv[3]) : (Real)10;   // total inflow flux magnitude
    const Integer cov_q   = (argc > 4) ? (Integer)atoi(argv[4]) : 6;
    const Integer Nbeta   = (argc > 5) ? (Integer)atoi(argv[5]) : 200;
    const Integer maxdep  = (argc > 6) ? (Integer)atoi(argv[6]) : 12;
    const Long    gmaxit  = (argc > 7) ? (Long)atoi(argv[7]) : 60000;
    const Long    Nvis    = (argc > 8) ? (Long)atoi(argv[8]) : 0;          // junction-box per-axis samples; 0 => cbrt(Ngrid)
    const Long    Ngrid   = (argc > 9) ? (Long)atoi(argv[9]) : 200;

    if (!pid) std::cout << "\n=== Stokes inflow/outflow BVP on the vmtk vessel network (loaded bundles) ===\n"
                        << "  bundles=" << prefix << "-jNNN.{mesh,arms}   tol=" << std::setprecision(1) << tol
                        << "  p_in=" << std::setprecision(4) << p_in << "  gmres_max_iter=" << gmaxit
                        << "\n  near-eval: Hybrid(cov_q=" << cov_q << ", Nbeta=" << Nbeta << ", max_depth=" << maxdep << ")\n";

    // ----------------------------------------------------------------------------------------------
    // (1) Enumerate the junction bundles by file existence (every rank scans identically).
    // ----------------------------------------------------------------------------------------------
    std::vector<Integer> jids;
    for (Integer i = 0; i < 100000; i++) {
      std::ostringstream nm; nm << prefix << "-j" << std::setw(3) << std::setfill('0') << i;
      std::ifstream probe(nm.str() + ".mesh");
      if (probe.good()) jids.push_back(i);
    }
    if (jids.empty()) { if (!pid) std::cerr << "no bundles at '" << prefix << "-jNNN.mesh'\n"; Comm::MPI_Finalize(); return 1; }

    // ----------------------------------------------------------------------------------------------
    // (2) Load the WHOLE network on every rank (replicated coords): global junction coord0 + global arm
    //     arrays, and reconstruct each leaf cap (C,u,R0,owner,node). QuadElemList replicate-then-slices
    //     across `comm`; SlenderElemList has no comm ctor, so we slice the global panels manually below.
    // ----------------------------------------------------------------------------------------------
    Vector<Real> Xjunc_all;                                             // all junction-body coord0 (world), AoS
    Vector<Long> a_elem, a_forder; Vector<Real> a_coord, a_radius, a_orient;   // global concatenated arms
    std::vector<FlowCap> caps;
    Vector<Real> jctr;                                                  // per-junction centroid (viz), AoS
    std::vector<Real> jhalf;                                            // per-junction bounding radius (viz)
    Integer order = 0; Long cheb = 0, fourier = 0;

    for (size_t k = 0; k < jids.size(); k++) {
      Vector<Real> Xj; Integer ord_j = 0; NetworkArmBundle<Real> B;
      const bool ok = ReadNetworkBundle<Real>(prefix, jids[k], Xj, ord_j, B, Comm::Self());
      SCTL_ASSERT_MSG(ok, "bundle probe/read mismatch");
      if (!order) { order = ord_j; cheb = B.cheb; fourier = B.fourier; }
      else SCTL_ASSERT_MSG(order == ord_j && cheb == B.cheb && fourier == B.fourier, "mixed discretization across bundles");

      // body: append + centroid/bounding-radius for the interior viz box.
      const Long nn = Xj.Dim()/3; Vec3<Real> c{0,0,0};
      for (Long m = 0; m < Xj.Dim(); m++) Xjunc_all.PushBack(Xj[m]);
      for (Long q = 0; q < nn; q++) { c[0]+=Xj[3*q]; c[1]+=Xj[3*q+1]; c[2]+=Xj[3*q+2]; }
      if (nn) c = mul3((Real)1/nn, c);
      Real hr = 0; for (Long q = 0; q < nn; q++) hr = std::max(hr, nrm3(sub3(Vec3<Real>{Xj[3*q],Xj[3*q+1],Xj[3*q+2]}, c)));
      jctr.PushBack(c[0]); jctr.PushBack(c[1]); jctr.PushBack(c[2]); jhalf.push_back(hr);

      // arms: reconstruct caps from each cap arm's LAST panel, then append the raw arrays wholesale.
      const Vector<Real> w1 = extrap_weights(cheb, (Real)1);            // extrapolate terminal panel to s=1
      Long node0 = 0, panel0 = 0;                                       // running offsets within this bundle
      const Integer narm = (Integer)B.npanel.size();
      for (Integer a = 0; a < narm; a++) {
        const Long npan = B.npanel[a];
        // node offset of this arm's LAST panel (each panel has elem_order[.] centerline nodes).
        Long nlast = node0; for (Long p = 0; p < npan - 1; p++) nlast += B.elem_order[panel0 + p];
        const Long clast = B.elem_order[panel0 + npan - 1];             // cheb of the last panel
        if (B.is_cap[a]) {
          Vec3<Real> C{0,0,0}; Real R0 = 0;
          for (Long j = 0; j < clast; j++) {
            const Long n = nlast + j;
            C = add3(C, mul3(w1[j], Vec3<Real>{B.coord[3*n], B.coord[3*n+1], B.coord[3*n+2]}));
            R0 += w1[j]*B.radius[n];
          }
          // outward axis = centerline tangent near the tip (last two nodes point toward increasing t).
          const Long nA = nlast + clast - 2, nB = nlast + clast - 1;
          Vec3<Real> u = unit3(sub3(Vec3<Real>{B.coord[3*nB],B.coord[3*nB+1],B.coord[3*nB+2]},
                                    Vec3<Real>{B.coord[3*nA],B.coord[3*nA+1],B.coord[3*nA+2]}));
          FlowCap fc; fc.C = C; fc.u = u; fc.R0 = R0; fc.owner = jids[k]; fc.node = B.other_node[a];
          caps.push_back(fc);
        }
        // advance node/panel offsets over this arm.
        for (Long p = 0; p < npan; p++) node0 += B.elem_order[panel0 + p];
        panel0 += npan;
      }
      for (Long m = 0; m < B.elem_order.Dim(); m++) a_elem.PushBack(B.elem_order[m]);
      for (Long m = 0; m < B.forder.Dim();     m++) a_forder.PushBack(B.forder[m]);
      for (Long m = 0; m < B.coord.Dim();      m++) a_coord.PushBack(B.coord[m]);
      for (Long m = 0; m < B.radius.Dim();     m++) a_radius.PushBack(B.radius[m]);
      for (Long m = 0; m < B.orient.Dim();     m++) a_orient.PushBack(B.orient[m]);
    }

    // ----------------------------------------------------------------------------------------------
    // (2b) QJ_GEOM_SCALE: uniform similarity scale of the WHOLE assembled network. A pure similarity
    //     multiplies every LENGTH by s -- junction-body node coords, arm centerline coords, and the arm
    //     cross-section radius -- but leaves the unit cross-section orientation frame (a_orient) alone
    //     (it is a direction, not a length). The surface is exactly self-similar, so every normal is
    //     unchanged, area -> s^2*area, and the watertightness residual |int n dA| (units of area) ->
    //     s^2 * (original). At s=0.1 that is 1/100 of the s=1 floor -- confirming the loose watertight
    //     value is the large surface AREA, not a geometric gap. Also scale the viz centroid/bounding
    //     radius and the cap frames so a downstream solve stays consistent.
    // ----------------------------------------------------------------------------------------------
    const Real gscale = std::getenv("QJ_GEOM_SCALE") ? (Real)atof(std::getenv("QJ_GEOM_SCALE")) : (Real)1;
    if (gscale != (Real)1) {
      for (Long m = 0; m < Xjunc_all.Dim(); m++) Xjunc_all[m] *= gscale;
      for (Long m = 0; m < a_coord.Dim();   m++) a_coord[m]   *= gscale;
      for (Long m = 0; m < a_radius.Dim();  m++) a_radius[m]  *= gscale;
      for (Long m = 0; m < jctr.Dim();      m++) jctr[m]      *= gscale;
      for (size_t m = 0; m < jhalf.size();  m++) jhalf[m]     *= gscale;
      for (size_t m = 0; m < caps.size();   m++) { caps[m].C = mul3(gscale, caps[m].C); caps[m].R0 *= gscale; }
      if (!pid) std::cout << "  [QJ_GEOM_SCALE] uniform similarity scale s=" << (double)gscale
                          << " applied (lengths & radii x s; orientation frame unchanged)\n";
    }

    // ----------------------------------------------------------------------------------------------
    // (3) Build the coupled MPI-partitioned lists. QuadElemList(order,coord,comm) keeps this rank's
    //     element slice; slice the slender panels the same way HybridAssembly::slender does.
    // ----------------------------------------------------------------------------------------------
    QuadElemList<Real> junc(order, Xjunc_all, comm);
    SlenderElemList<Real> arms;
    {
      const Long Nelem = a_elem.Dim(), k0 = (Nelem*pid)/Np, k1 = (Nelem*(pid+1))/Np;
      Vector<Long> eo, fo; Vector<Real> co, ra, ori; Long n0 = 0;
      for (Long p = 0; p < Nelem; p++) {
        const Long ce = a_elem[p];
        if (p >= k0 && p < k1) {
          eo.PushBack(a_elem[p]); fo.PushBack(a_forder[p]);
          for (Long j = 0; j < ce; j++) { const Long n = n0 + j;
            co.PushBack(a_coord[3*n]); co.PushBack(a_coord[3*n+1]); co.PushBack(a_coord[3*n+2]);
            ra.PushBack(a_radius[n]);
            ori.PushBack(a_orient[3*n]); ori.PushBack(a_orient[3*n+1]); ori.PushBack(a_orient[3*n+2]); }
        }
        n0 += ce;
      }
      arms = SlenderElemList<Real>(eo, fo, co, ra, ori);
    }
    junc.SetQuadScheme(quad_junctions::QJDefaultScheme<Real>(), cov_q, Nbeta, maxdep);

    {
      const Long njp = GlobalReduce((Long)junc.Size(), comm, CommOp::SUM);
      const Long nap = GlobalReduce((Long)arms.Size(), comm, CommOp::SUM);
      if (!pid) std::cout << "\n[geometry] " << jids.size() << " junctions loaded (order=" << order
                          << ", cheb=" << cheb << ", fourier=" << fourier << "), " << caps.size() << " leaf caps\n"
                          << "  quad panels=" << njp << " | slender panels=" << nap
                          << " | TOTAL panels=" << (njp+nap) << " (global, over " << Np << " ranks)\n";
    }
    divergence_check(junc, arms, tol, comm);   // watertightness sanity (combined |int n dA| should be small)

    // QJ_GEOM_ONLY: cheap load + watertightness pass, no solve (this network is ~22M Stokes DOF -- validate
    // the geometry loads/closes before committing a multi-node PVFMM solve).
    if (std::getenv("QJ_GEOM_ONLY")) { if (!pid) std::cout << "\n[QJ_GEOM_ONLY] geometry loaded + checked; skipping solve.\n"; Comm::MPI_Finalize(); return 0; }

    // Mode select (QJ_NET_MODE, default "flow"). "manufactured"/"mfg" runs the analytic-field
    // acceptance sweep on the SAME coupled operator instead of the physical flow BVP below.
    const char* mode_env = std::getenv("QJ_NET_MODE");
    const std::string mode = mode_env ? std::string(mode_env) : std::string("flow");
    if (mode == "manufactured" || mode == "mfg") {
      run_network_manufactured(junc, arms, a_coord, comm, tol, gmaxit, order, cheb, fourier);
      Comm::MPI_Finalize(); return 0;
    }
    if (mode == "greens") {
      run_network_greens(junc, arms, a_coord, comm, tol);
      Comm::MPI_Finalize(); return 0;
    }
    SCTL_ASSERT_MSG(mode == "flow", "QJ_NET_MODE must be 'flow' (default), 'manufactured'/'mfg', or 'greens'");

    // ----------------------------------------------------------------------------------------------
    // (4) [flow mode] Assign inflow/outflow ports (connectivity: every leaf cap is a port).
    //     default: single inflow = extreme-axis cap; QJ_INFLOW_NODES overrides (graph node ids).
    //     outflow split: equal, or QJ_OUTFLOW_FLUX relative weights; normalized so net flux = 0.
    // ----------------------------------------------------------------------------------------------
    std::vector<int> inflow_idx;
    if (const char* env = std::getenv("QJ_INFLOW_NODES")) {
      std::stringstream ss(env); std::string tok;
      while (std::getline(ss, tok, ',')) { if (tok.empty()) continue; const Integer id = (Integer)atoi(tok.c_str());
        for (size_t i = 0; i < caps.size(); i++) if (caps[i].node == id) inflow_idx.push_back((int)i); }
      SCTL_ASSERT_MSG(!inflow_idx.empty(), "QJ_INFLOW_NODES matched no cap `other_node` id");
    } else {
      const char* ax = std::getenv("QJ_INFLOW_AXIS"); std::string axs = ax ? ax : "x";
      const int comp = (axs[0]=='y') ? 1 : (axs[0]=='z') ? 2 : 0;
      const Real sgn = (axs.size() > 1 && axs[1]=='-') ? (Real)-1 : (Real)1;
      int best = 0; Real bestv = -1e300;
      for (size_t i = 0; i < caps.size(); i++) { const Real v = sgn*caps[i].C[comp]; if (v > bestv) { bestv = v; best = (int)i; } }
      inflow_idx.push_back(best);
    }
    std::vector<char> is_in(caps.size(), 0); for (int i : inflow_idx) is_in[(size_t)i] = 1;
    std::vector<int> outflow_idx; for (int i = 0; i < (int)caps.size(); i++) if (!is_in[(size_t)i]) outflow_idx.push_back(i);
    SCTL_ASSERT_MSG(!outflow_idx.empty(), "no outflow caps -- need >= 1 outflow");

    // inflow split (equal); outflow split (weights).
    for (int i : inflow_idx) { caps[(size_t)i].sgn = -1; caps[(size_t)i].p = p_in/(Real)inflow_idx.size(); }
    std::vector<Real> w(outflow_idx.size(), (Real)1);
    if (const char* env = std::getenv("QJ_OUTFLOW_FLUX")) {
      std::stringstream ss(env); std::string tok; size_t k = 0;
      while (k < w.size() && std::getline(ss, tok, ',')) { if (!tok.empty()) w[k] = (Real)atof(tok.c_str()); k++; }
    }
    Real wsum = 0; for (Real x : w) wsum += x; SCTL_ASSERT_MSG((double)wsum > 0, "outflow weights must sum > 0");
    for (size_t k = 0; k < outflow_idx.size(); k++) { caps[(size_t)outflow_idx[k]].sgn = +1; caps[(size_t)outflow_idx[k]].p = p_in*w[k]/wsum; }

    if (!pid) {
      std::cout << "\n  [ports] " << inflow_idx.size() << " inflow + " << outflow_idx.size() << " outflow (net flux 0)\n";
      for (int i : inflow_idx)
        std::cout << "    INFLOW  cap node " << caps[(size_t)i].node << " (junc " << caps[(size_t)i].owner << ", R0="
                  << std::setprecision(4) << caps[(size_t)i].R0 << ") C=(" << caps[(size_t)i].C[0] << ","
                  << caps[(size_t)i].C[1] << "," << caps[(size_t)i].C[2] << ")  p=" << std::setprecision(6) << caps[(size_t)i].p << "\n";
      if (outflow_idx.size() <= 8) for (int i : outflow_idx)
        std::cout << "    outflow cap node " << caps[(size_t)i].node << " p=" << std::setprecision(6) << caps[(size_t)i].p << "\n";
      else std::cout << "    " << outflow_idx.size() << " outflow caps, each p~" << std::setprecision(6) << (p_in/(Real)outflow_idx.size()) << " (equal split)\n";
    }

    // ----------------------------------------------------------------------------------------------
    // (5) Geometric flux factor g_c = int prof*(u.n) dA per cap, then amp_c = sgn_c*p_c/g_c so the signed
    //     flux through cap c is exactly sgn_c*p_c (caps live in the quad list -- hemisphere domes).
    // ----------------------------------------------------------------------------------------------
    {
      Vector<Real> Xf, Xnf, wts, dist; Vector<Long> cnt; junc.GetFarFieldNodes(Xf, Xnf, wts, dist, cnt, tol);
      Vector<Real> g((Long)caps.size()); g = 0;
      for (Long i = 0; i < wts.Dim(); i++) {
        const Vec3<Real> X{Xf[3*i],Xf[3*i+1],Xf[3*i+2]}, n{Xnf[3*i],Xnf[3*i+1],Xnf[3*i+2]};
        for (size_t c = 0; c < caps.size(); c++) { Real pr; if (cap_profile(X, caps[c], pr)) { g[(Long)c] += wts[i]*pr*dot3(caps[c].u, n); break; } }
      }
      for (size_t c = 0; c < caps.size(); c++) g[(Long)c] = GlobalReduce((double)g[(Long)c], comm, CommOp::SUM);
      for (size_t c = 0; c < caps.size(); c++) { SCTL_ASSERT_MSG(std::fabs((double)g[(Long)c]) > 1e-30, "degenerate cap flux factor");
        caps[c].amp = (Real)caps[c].sgn * caps[c].p / g[(Long)c]; }
    }

    // ----------------------------------------------------------------------------------------------
    // (6) Boundary velocity RHS over the combined "0_junc"+"1_arms" node ordering + flux verification.
    // ----------------------------------------------------------------------------------------------
    Vector<Real> X, Xn; Long Nj, Na; combined_nodes(junc, arms, X, Xn, Nj, Na);
    const Long Nnode = Nj + Na;
    Vector<Real> bc(Nnode*3);
    for (Long i = 0; i < Nnode; i++) { const Vec3<Real> v = flow_bc_vel(Vec3<Real>{X[3*i],X[3*i+1],X[3*i+2]}, caps);
      bc[3*i]=v[0]; bc[3*i+1]=v[1]; bc[3*i+2]=v[2]; }
    {
      Vector<Real> Xf, Xnf, wts, dist; Vector<Long> cnt; junc.GetFarFieldNodes(Xf, Xnf, wts, dist, cnt, tol);
      Vector<Real> flux((Long)caps.size()); flux = 0;
      for (Long i = 0; i < wts.Dim(); i++) {
        const Vec3<Real> X0{Xf[3*i],Xf[3*i+1],Xf[3*i+2]}, n{Xnf[3*i],Xnf[3*i+1],Xnf[3*i+2]};
        const Vec3<Real> v = flow_bc_vel(X0, caps);
        for (size_t c = 0; c < caps.size(); c++) { Real pr; if (cap_profile(X0, caps[c], pr)) { flux[(Long)c] += wts[i]*dot3(v, n); break; } }
      }
      Real total = 0, in_sum = 0, out_sum = 0;
      for (size_t c = 0; c < caps.size(); c++) { flux[(Long)c] = GlobalReduce((double)flux[(Long)c], comm, CommOp::SUM);
        total += flux[(Long)c]; (caps[c].sgn < 0 ? in_sum : out_sum) += flux[(Long)c]; }
      if (!pid) std::cout << "\n  [flux check] sum inflow=" << std::setprecision(6) << in_sum << " (target " << -p_in
                          << ")  sum outflow=" << out_sum << " (target " << p_in << ")  NET=" << total << " (target 0)\n";
      SCTL_ASSERT_MSG(std::fabs((double)total) < 1e-6*(double)std::max((Real)1, p_in), "net flux != 0 -- BVP incompatible");
    }

    // ----------------------------------------------------------------------------------------------
    // (7) Interior viz targets: arm cross-section stars (interior by construction) + per-junction boxes
    //     (filtered to interior by the Laplace-DL indicator after the solve).
    // ----------------------------------------------------------------------------------------------
    Vector<Real> Xarm; build_arm_panel_targets<Real>(arms, comm, cheb, Xarm);   // collective
    Long Narm = 0, Njunc = 0, Nax = 0; Vector<Real> Xgrid;
    if (!pid) {
      for (Long i = 0; i < Xarm.Dim(); i++) Xgrid.PushBack(Xarm[i]);
      Narm = Xgrid.Dim()/3;
      Nax = (Nvis > 0) ? Nvis : std::max<Long>(3, (Long)std::lround(std::cbrt((double)Ngrid)));
      Vector<Real> half((Long)jhalf.size()); for (size_t i = 0; i < jhalf.size(); i++) half[(Long)i] = (Real)0.6*jhalf[i];
      Vector<Real> Xjb; build_box_targets<Real>(jctr, half, Nax, Xjb);
      for (Long i = 0; i < Xjb.Dim(); i++) Xgrid.PushBack(Xjb[i]);
      Njunc = Xgrid.Dim()/3 - Narm;
      std::cout << "\n  [viz] " << Narm << " arm cross-section + " << Njunc << " junction-box ("
                << jhalf.size() << " x " << Nax << "^3) candidates\n";
    }
    const Long Ngrid_pts = Xgrid.Dim()/3;

    // ----------------------------------------------------------------------------------------------
    // (8) Solve the interior Stokes Dirichlet BVP (SL=-1, DL=+1 => jump -1/2) + evaluate at the cloud.
    // ----------------------------------------------------------------------------------------------
    // CSBQ well-conditioned per-node single-layer scaling on the slender arms (Malhotra-Barnett 2024,
    // Eq. 33): replace the constant arm SL coefficient with eta(s)=1/(2*eps*log(1/eps)) so the combined
    // field stays O(1)-conditioned as the tube radius eps->0. Env-gated (default OFF -> results identical):
    //   QJ_SLENDER_SCALING=1   enable
    //   QJ_SLENDER_EPS_MAX=<r> slender-regime cutoff (default 0.1); nodes with eps>cutoff keep SL_scal.
    // Computed ONCE here from each arm node's radius; solve_dirichlet_bvp multiplies it into the arm slice
    // of the density once per GMRES iteration.
    Vector<Real> arm_sl_eta;                                          // empty => scaling OFF
    {
      const char* sbenv = std::getenv("QJ_SLENDER_SCALING");
      if (sbenv && atoi(sbenv) != 0) {
        const char* emenv = std::getenv("QJ_SLENDER_EPS_MAX");
        const Real eps_max = emenv ? (Real)atof(emenv) : (Real)0.1;
        quad_junctions::arm_slender_sl_eta<Real>(arms, cheb, arm_sl_eta, eps_max);
        // Diagnostics: how many arm nodes fall in the slender regime, and the eta range applied.
        Long n_scaled = 0; Real emin = 1e300, emax = 0;
        for (Long i = 0; i < arm_sl_eta.Dim(); i++) if (arm_sl_eta[i] > 0) { n_scaled++; emin = std::min(emin, (Real)arm_sl_eta[i]); emax = std::max(emax, (Real)arm_sl_eta[i]); }
        const Long n_arm = arm_sl_eta.Dim();
        const long g_scaled = (long)GlobalReduce((double)n_scaled, comm, CommOp::SUM);
        const long g_arm    = (long)GlobalReduce((double)n_arm,    comm, CommOp::SUM);
        const double g_emin = (n_scaled ? GlobalReduce((double)emin, comm, CommOp::MIN) : GlobalReduce(1e300, comm, CommOp::MIN));
        const double g_emax = GlobalReduce((double)emax, comm, CommOp::MAX);
        if (!pid)
          std::cout << "  [slender-scaling] ON  eps_max=" << (double)eps_max << "  scaled "
                    << g_scaled << " / " << g_arm << " arm nodes  eta in ["
                    << (g_scaled ? g_emin : 0.0) << ", " << g_emax << "]\n";
      } else if (!pid) {
        std::cout << "  [slender-scaling] OFF (set QJ_SLENDER_SCALING=1 to enable CSBQ Eq.33 arm SL scaling)\n";
      }
    }
    Vector<Real> Ugrid;
    const Vector<Real> sigma = solve_dirichlet_bvp<Real, Stokes3D_FxU, Stokes3D_DxU>(
        junc, arms, comm, tol, bc, /*interior=*/true, /*SL_scal=*/(Real)-1., /*DL_scal=*/(Real)1.,
        Xgrid, &Ugrid, "network inflow/outflow", /*gmres_max_iter=*/gmaxit,
        /*precond=*/nullptr, /*arm_sl_eta=*/arm_sl_eta);

    // ----------------------------------------------------------------------------------------------
    // (9) Interior filter (Laplace DL const-density indicator ~ -1 interior / ~0 exterior) + output.
    // ----------------------------------------------------------------------------------------------
    const std::string tag = "vis/bifurc-network-flow";
    Vector<Real> Xvis, Uvis;
    {
      BoundaryIntegralOp<Real, Laplace3D_DxU> IndOp((Laplace3D_DxU()), false, comm);
      SetPVFMMKer(IndOp); IndOp.SetAccuracy(tol);
      IndOp.AddElemList(junc, "0_junc"); IndOp.AddElemList(arms, "1_arms");
      Vector<Real> ones(Nnode); ones = 1; IndOp.SetTargetCoord(Xgrid);
      Vector<Real> ind; IndOp.ComputePotential(ind, ones);
      if (!pid) {
        Long n_in = 0;
        for (Long i = 0; i < Narm; i++) for (Integer k = 0; k < 3; k++) { Xvis.PushBack(Xgrid[3*i+k]); Uvis.PushBack(Ugrid[3*i+k]); }
        for (Long i = Narm; i < Ngrid_pts; i++) if (std::fabs((double)ind[i]) > 0.5) {
          for (Integer k = 0; k < 3; k++) { Xvis.PushBack(Xgrid[3*i+k]); Uvis.PushBack(Ugrid[3*i+k]); } n_in++; }
        std::cout << "  [grid] kept " << Narm << " arm + " << n_in << " / " << Njunc << " junction-box interior = "
                  << (Xvis.Dim()/3) << " points\n";
      }
    }
    {
      Vector<Real> sj(Nj*3), sa(Na*3), bcj(Nj*3);
      for (Long i = 0; i < Nj*3; i++) { sj[i] = sigma[i]; bcj[i] = bc[i]; }
      for (Long i = 0; i < Na*3; i++) sa[i] = sigma[Nj*3 + i];
      junc.WriteVTK(tag + "-sigma-junc", sj,  comm);
      arms.WriteVTK(tag + "-sigma-arms", sa,  comm);
      junc.WriteVTK(tag + "-bc-junc",    bcj, comm);
      if (!pid) { write_points_vtu<Real>(tag + "-flow-box", Xvis, Uvis, Xvis.Dim()/3);
        std::cout << "\n  [dump] " << tag << "-sigma-{junc,arms}.pvtu, " << tag << "-bc-junc.pvtu, "
                  << tag << "-flow-box.vtu (interior velocity)\n"; }
    }
  }
  Comm::MPI_Finalize();
  return 0;
}
