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
 * Env overrides for the port assignment:
 *   QJ_INFLOW_NODES="id,id,..."   graph node ids (the cap `other_node` in the bundle) to force as INFLOWs
 *                                 (split equally); default = the single extreme-axis cap.
 *   QJ_INFLOW_AXIS=x|y|z|x-|...   axis whose EXTREME cap is the default inflow (default "x" = max-x).
 *   QJ_OUTFLOW_FLUX="w1,w2,..."   relative outflow weights in printed order (missing -> 1; unset -> equal).
 *
 *   make MPI=1 bin/bifurc-network-flow-bie          # or: make PVFMM=1 bin/bifurc-network-flow-bie
 *   OMP_NUM_THREADS=8 mpirun -n <ranks> ./bin/bifurc-network-flow-bie \
 *       [bundle_prefix] [tol] [p_in] [cov_q] [Nbeta] [max_depth] [gmres_max_iter] [Nvis] [Ngrid]
 *   e.g.  ... ./bin/bifurc-network-flow-bie vis/network 1e-7 10   (prefix = whatever you assembled to)
 *
 * The bundles bake in the DISCRETIZATION (order + fourier), so accuracy is dialed by choosing the bundle
 * set plus the near-eval tol / cov_q / Nbeta / max_depth here.
 *
 * GEOMETRY PROVENANCE -- REGENERATE BEFORE EVERY RUN (no bundle is committed; vis/ is gitignored):
 *   The CANONICAL geometry is the assembler run on data/vmtk/vessels_fixed.graph (TRUE per-junction branch
 *   angles + every watertightness fix -- bigon3, lead-corner arm bend, leaf-arm fix, turn-adaptive corners,
 *   sigma-floor 0.075->0.05, auto size-shrink). It closes to |int n dA| ~1e-3 on area ~5e5 (rel ~1e-5).
 *   DEFAULT ASSEMBLE PARAMETERS (bifurc-network-assemble); use order=12 fourier=24 for 6-digit quadrature:
 *       data/vmtk/vessels_fixed.graph  <prefix>  <order> 1 1.5 0.4 3 12 10 <fourier> 2
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

} // anonymous namespace

int main(int argc, char** argv) {
  Comm::MPI_Init(&argc, &argv);
  {
    const Comm comm = Comm::World();
    const Integer Np = comm.Size(), pid = comm.Rank();

    const std::string prefix = (argc > 1) ? std::string(argv[1]) : std::string("vis/network");
    const Real    tol     = (argc > 2) ? (Real)atof(argv[2]) : (Real)1e-7;
    const Real    p_in    = (argc > 3) ? (Real)atof(argv[3]) : (Real)10;   // total inflow flux magnitude
    const Integer cov_q   = (argc > 4) ? (Integer)atoi(argv[4]) : 6;
    const Integer Nbeta   = (argc > 5) ? (Integer)atoi(argv[5]) : 200;
    const Integer maxdep  = (argc > 6) ? (Integer)atoi(argv[6]) : 12;
    const Long    gmaxit  = (argc > 7) ? (Long)atoi(argv[7]) : 800;
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

    // ----------------------------------------------------------------------------------------------
    // (4) Assign inflow/outflow ports (connectivity: every leaf cap is a port).
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
    Vector<Real> Ugrid;
    const Vector<Real> sigma = solve_dirichlet_bvp<Real, Stokes3D_FxU, Stokes3D_DxU>(
        junc, arms, comm, tol, bc, /*interior=*/true, /*SL_scal=*/(Real)-1., /*DL_scal=*/(Real)1.,
        Xgrid, &Ugrid, "network inflow/outflow", /*gmres_max_iter=*/gmaxit);

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
