/**
 * Interior Stokes inflow/outflow BVP on a SINGLE Y-bifurcation -- quad-only vs CSBQ slender arms.
 *
 * Builds the single Y-bifurcation via the standalone-builder path (junction body + 3 POU transition
 * tubes + 3 hemisphere caps from BuildYJunctionWithTransitions, joined to 3 straight R0 arms) and drives
 * it with ONE inlet + TWO outlet Poiseuille caps whose fluxes conserve total volume:
 *
 *   - arm 0 = parabolic INFLOW,  volumetric flux p_in  (sgn -1, flow into the domain along -u0).
 *   - arms 1,2 = parabolic OUTFLOW, fluxes p_out1, p_out2 (sgn +1).
 *   - No-slip (u = 0) everywhere else (tube walls, junction body).
 *   - Conservation by construction: p_in = p_out1 + p_out2  =>  net flux -p_in+p_out1+p_out2 = 0, the
 *     compatibility condition int u.n dA = 0 for the interior incompressible Stokes Dirichlet problem.
 *
 * The whole point is to compare the TWO arm discretizations on the SAME junction mesh:
 *   - arm_kind 0: CSBQ SlenderElemList arms (BuildYArmsSlender).
 *   - arm_kind 1: full-quad QuadElemList tube arms (BuildYArmsQuadTube), with the azimuthal panel count
 *                 held at the node-CONFORMING value Na = YSwept::Na0*nref (=16*nref) -- the setting that
 *                 makes the quad<->quad seam machine-exact and matches CSBQ accuracy (see the M3 shaft-swap
 *                 investigation: a non-conforming Na corrupts the adjacent transition/cap).
 * Both solve the SAME combined-field ( -1/2 I - S + D ) sigma = u_bc via the shared solve_dirichlet_bvp
 * in hybrid_bie_tests.hpp. We evaluate the represented velocity at ONE interior point on the inlet-arm
 * axis (both solves must agree there to the discretization floor) and report the GMRES iteration counts.
 *
 *   make bin/ybifurc-yflow-bie
 *   OMP_NUM_THREADS=8 ./bin/ybifurc-yflow-bie \
 *       [level] [order(mult4)] [nref] [eta_join] [Ns_trans] [s_cap] [n_axial] [fourier] \
 *       [tol] [Nbeta] [max_depth] [cov_q] [p_out1] [p_out2] [arm_mode]
 *   defaults: 1.5 12 1 0.4 3 0.88 3 12 1e-8 200 8 6 10 10 2
 *   arm_mode: 0 = CSBQ only, 1 = quad-tube only, 2 = both (default, runs the comparison).
 */

#include <csbq.hpp>                                  // CSBQ SlenderElemList
#include <quad_junctions/ybifurc_hybrid_geom.hpp>    // BuildYJunctionWithTransitions / BuildYArms* / arm_frame / YSwept / pou_kind
#include <quad_junctions/quad_scheme.hpp>            // QJDefaultScheme (Duffy default, SCTL_SELF_SCHEME=hybrid opt-out)
#include <quad_junctions/hybrid_bie_tests.hpp>       // combined_nodes + solve_dirichlet_bvp
#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

using namespace sctl;
using namespace quad_junctions;

namespace {

// One inflow/outflow cap: dome-equator center C, outward axis u, radius R0, signed amplitude `amp`
// (= sgn * p / g, with g the geometric flux factor so flux through the cap = sgn*p), flux magnitude p,
// and sign sgn (-1 inflow / +1 outflow). (Same struct/machinery as src/ybifurc-flow-bie.cpp.)
template <class Real> struct FlowCap {
  Vec3<Real> C, u;
  Real R0 = 0, amp = 0, p = 0;
  int sgn = 0;
};

template <class Real> inline Real dot3(const Vec3<Real>& a, const Vec3<Real>& b) {
  return a[0]*b[0] + a[1]*b[1] + a[2]*b[2];
}

// Parabolic axial profile on a cap dome: prof(X) = 1 - (r/R0)^2 with r the transverse distance from the
// cap axis, for nodes lying on that cap's hemisphere (|X-C| ~ R0, on the +u side). 0 off any cap.
template <class Real>
bool cap_profile(const Vec3<Real>& X, const FlowCap<Real>& c, Real& prof) {
  const Vec3<Real> d{X[0]-c.C[0], X[1]-c.C[1], X[2]-c.C[2]};
  const Real ax = dot3(d, c.u);
  const Real dist2 = dot3(d, d);
  const Real dist = sqrt<Real>(dist2);
  if (std::fabs((double)(dist - c.R0)) < (double)((Real)0.05*c.R0) && ax > (Real)-0.05*c.R0) {
    Real r2 = dist2 - ax*ax; if (r2 < 0) r2 = 0;
    prof = (Real)1 - r2/(c.R0*c.R0); if (prof < 0) prof = 0;
    return true;
  }
  prof = 0;
  return false;
}

// Prescribed boundary velocity at a point X: v = amp*prof*u on the owning cap, else 0 (no-slip).
template <class Real>
Vec3<Real> flow_bc_vel(const Vec3<Real>& X, const std::vector<FlowCap<Real>>& caps) {
  for (const auto& c : caps) {
    Real prof;
    if (cap_profile<Real>(X, c, prof)) {
      const Real s = c.amp*prof;
      return Vec3<Real>{s*c.u[0], s*c.u[1], s*c.u[2]};
    }
  }
  return Vec3<Real>{(Real)0, (Real)0, (Real)0};
}

// Solve the flow BVP for one arm list. Assembles the RHS from the caps over the combined "0_junc"+
// "1_arms" node ordering (caps live only in the junction, so all arm nodes get v=0), solves the
// combined-field system, and evaluates the velocity at the interior probe Xpt (supplied on rank 0).
// Returns the probe velocity (rank 0) and the GMRES iteration count.
template <class Real, class ArmList>
void run_flow(const QuadElemList<Real>& junc, const ArmList& arms, const Comm& comm,
              const std::vector<FlowCap<Real>>& caps, const Real tol, const Vector<Real>& Xpt,
              const std::string& name, Vec3<Real>& Upt_out, Long& iters_out) {
  Vector<Real> X, Xn; Long Nj = 0, Na = 0;
  combined_nodes(junc, arms, X, Xn, Nj, Na);
  const Long Nnode = Nj + Na;
  Vector<Real> bc(Nnode*3);
  for (Long i = 0; i < Nnode; i++) {
    const Vec3<Real> v = flow_bc_vel<Real>(Vec3<Real>{X[3*i], X[3*i+1], X[3*i+2]}, caps);
    bc[3*i] = v[0]; bc[3*i+1] = v[1]; bc[3*i+2] = v[2];
  }
  Vector<Real> Upt; Long iters = 0;
  solve_dirichlet_bvp<Real, Stokes3D_FxU, Stokes3D_DxU>(
      junc, arms, comm, tol, bc, /*interior=*/true, /*SL_scal=*/(Real)-1, /*DL_scal=*/(Real)1,
      Xpt, &Upt, name, /*gmres_max_iter=*/400, /*precond=*/nullptr, /*arm_sl_eta=*/Vector<Real>(),
      /*obstacles=*/(const ArmList*)nullptr, /*n_iter=*/&iters);
  iters_out = iters;
  Upt_out = Vec3<Real>{(Real)0, (Real)0, (Real)0};
  if (!comm.Rank() && Upt.Dim() >= 3) Upt_out = Vec3<Real>{Upt[0], Upt[1], Upt[2]};
}

} // anonymous namespace

int main(int argc, char** argv) {
  Comm::MPI_Init(&argc, &argv);
  using Real = double;
  {
    const Comm comm = Comm::World();
    const Real    level   = (argc > 1)  ? (Real)atof(argv[1])  : (Real)1.5;
    const Integer ord     = (argc > 2)  ? (Integer)atoi(argv[2]) : 12;
    const Integer nref    = (argc > 3)  ? (Integer)atoi(argv[3]) : 1;
    const Real    etajoin = (argc > 4)  ? (Real)atof(argv[4])  : (Real)0.4;
    const Integer NsTrans = (argc > 5)  ? (Integer)atoi(argv[5]) : 3;
    const Real    s_cap   = (argc > 6)  ? (Real)atof(argv[6])  : (Real)0.88;
    const Integer nAxial  = (argc > 7)  ? (Integer)atoi(argv[7]) : 3;
    const Long    fourier = (argc > 8)  ? (Long)atoi(argv[8])  : 12;
    const Real    tol     = (argc > 9)  ? (Real)atof(argv[9])  : (Real)1e-8;
    const Integer Nbeta   = (argc > 10) ? (Integer)atoi(argv[10]) : 200;
    const Integer maxdep  = (argc > 11) ? (Integer)atoi(argv[11]) : 8;
    const Integer cov_q   = (argc > 12) ? (Integer)atoi(argv[12]) : 6;
    const Real    p_out1  = (argc > 13) ? (Real)atof(argv[13]) : (Real)10;
    const Real    p_out2  = (argc > 14) ? (Real)atof(argv[14]) : (Real)10;
    const Integer armmode = (argc > 15) ? (Integer)atoi(argv[15]) : 2;   // 0=csbq / 1=quad / 2=both
    const Real    p_in    = p_out1 + p_out2;                             // conservation by construction
    const Integer Ncap    = (Integer)(YSwept::Ncap0 * nref);
    // Quad-tube arm discretization: axial panels match the slender axial node count; azimuthal panels
    // at the node-CONFORMING count so the quad<->quad seam is machine-exact (the M3 shaft-swap finding).
    const Integer NsShaft = std::max<Integer>(1, (Integer)std::lround((double)nAxial*10.0/ord));
    const Integer NaShaft = (Integer)(YSwept::Na0 * nref);
    pou_kind() = 1;   // smootherstep POU (order-exact) -- what the transition builder expects

    SCTL_ASSERT_MSG(ord >= 4 && ord <= 48 && ord % 4 == 0, "order must be a multiple of 4 in {4,...,48}.");
    SCTL_ASSERT_MSG(armmode >= 0 && armmode <= 2, "arm_mode must be 0 (csbq) / 1 (quad) / 2 (both).");

    if (!comm.Rank()) {
      std::cout << "\n=== Stokes inflow/outflow BVP on a single Y-bifurcation (quad vs CSBQ arms) ===\n";
      std::cout << "  order=" << ord << " level=" << level << " nref=" << nref << " eta_join=" << etajoin
                << " Ns_trans=" << NsTrans << " s_cap=" << s_cap << " n_axial=" << nAxial
                << " fourier=" << fourier << "\n";
      std::cout << "  near-eval: Hybrid(cov_q=" << cov_q << ", Nbeta=" << Nbeta << ", max_depth=" << maxdep
                << ") tol=" << std::setprecision(1) << tol << "\n";
      std::cout << "  quad-tube arms: Ns_shaft=" << NsShaft << " Na_shaft=" << NaShaft << " (conforming)\n";
      std::cout << "  prescribed flux: inflow p_in=" << std::setprecision(4) << p_in
                << "  outflow p_out={" << p_out1 << "," << p_out2 << "}  net="
                << (-p_in + p_out1 + p_out2) << "\n";
    }

    // ----------------------------------------------------------------------------------------------
    // (1) Build the single Y-bifurcation junction (+ transitions + caps). Unplaced => local == world.
    // ----------------------------------------------------------------------------------------------
    Real R0[3], a0[3], sc[3], maxres = 0;
    QuadElemList<Real> junc = BuildYJunctionWithTransitions<Real>(
        ord, level, nref, etajoin, NsTrans, s_cap, R0, a0, sc, Ncap, &maxres, comm);
    junc.SetQuadScheme(quad_junctions::QJDefaultScheme<Real>(), cov_q, Nbeta, maxdep);

    // ----------------------------------------------------------------------------------------------
    // (2) Caps (in the junction quad list): arm 0 inflow, arms 1,2 outflow. Cap dome-equator center is
    //     at station s_cap on the arm axis u_k (see add_arm_cap_hemisphere).
    // ----------------------------------------------------------------------------------------------
    std::vector<FlowCap<Real>> caps;
    Vec3<Real> u0, e1_0, e2_0; arm_frame<Real>(0, u0, e1_0, e2_0);   // inlet-arm axis (for the probe)
    for (int k = 0; k < 3; k++) {
      Vec3<Real> u, e1, e2; arm_frame<Real>(k, u, e1, e2);
      FlowCap<Real> c;
      c.C  = Vec3<Real>{s_cap*u[0], s_cap*u[1], s_cap*u[2]};
      c.u  = u; c.R0 = R0[k];
      if (k == 0)      { c.sgn = -1; c.p = p_in;   }             // inflow
      else if (k == 1) { c.sgn = +1; c.p = p_out1; }             // outflow
      else             { c.sgn = +1; c.p = p_out2; }             // outflow
      caps.push_back(c);
    }
    if (!comm.Rank()) {
      std::cout << "\n  [caps] R0=" << std::setprecision(6) << caps[0].R0 << "  (3 hemisphere caps)\n";
      for (size_t i = 0; i < caps.size(); i++)
        std::cout << "    cap " << i << ": center=(" << std::setprecision(4) << caps[i].C[0] << ","
                  << caps[i].C[1] << "," << caps[i].C[2] << ")  axis=(" << caps[i].u[0] << ","
                  << caps[i].u[1] << "," << caps[i].u[2] << ")  "
                  << (caps[i].sgn < 0 ? "INFLOW" : "OUTFLOW") << " p=" << caps[i].p << "\n";
    }

    // ----------------------------------------------------------------------------------------------
    // (3) Geometric flux factor g_c = int prof*(u.n) dA per cap (far-field quadrature), then amplitude
    //     amp_c = sgn_c * p_c / g_c so the signed flux through cap c is exactly sgn_c * p_c. Caps live in
    //     the junction quad list, so this is identical for both arm types -- compute once.
    // ----------------------------------------------------------------------------------------------
    {
      Vector<Real> Xf, Xnf, wts, dist; Vector<Long> cnt;
      junc.GetFarFieldNodes(Xf, Xnf, wts, dist, cnt, tol);
      const Long Nf = wts.Dim();
      Vector<Real> g((Long)caps.size()); g = 0;
      for (Long i = 0; i < Nf; i++) {
        const Vec3<Real> X{Xf[3*i], Xf[3*i+1], Xf[3*i+2]};
        const Vec3<Real> n{Xnf[3*i], Xnf[3*i+1], Xnf[3*i+2]};
        for (size_t c = 0; c < caps.size(); c++) {
          Real prof;
          if (cap_profile<Real>(X, caps[c], prof)) { g[(Long)c] += wts[i]*prof*dot3(caps[c].u, n); break; }
        }
      }
      for (size_t c = 0; c < caps.size(); c++) g[(Long)c] = GlobalReduce((double)g[(Long)c], comm, CommOp::SUM);
      for (size_t c = 0; c < caps.size(); c++) {
        SCTL_ASSERT_MSG(std::fabs((double)g[(Long)c]) > 1e-30, "degenerate cap flux factor");
        caps[c].amp = (Real)caps[c].sgn * caps[c].p / g[(Long)c];
      }
    }

    // ----------------------------------------------------------------------------------------------
    // (4) Verify the flux: int v.n dA per cap (should be +-p) and the total (should be ~0).
    // ----------------------------------------------------------------------------------------------
    {
      Vector<Real> Xf, Xnf, wts, dist; Vector<Long> cnt;
      junc.GetFarFieldNodes(Xf, Xnf, wts, dist, cnt, tol);
      const Long Nf = wts.Dim();
      Vector<Real> flux((Long)caps.size()); flux = 0;
      for (Long i = 0; i < Nf; i++) {
        const Vec3<Real> X0{Xf[3*i], Xf[3*i+1], Xf[3*i+2]};
        const Vec3<Real> n{Xnf[3*i], Xnf[3*i+1], Xnf[3*i+2]};
        const Vec3<Real> v = flow_bc_vel<Real>(X0, caps);
        for (size_t c = 0; c < caps.size(); c++) {
          Real prof;
          if (cap_profile<Real>(X0, caps[c], prof)) { flux[(Long)c] += wts[i]*dot3(v, n); break; }
        }
      }
      Real total = 0;
      for (size_t c = 0; c < caps.size(); c++) { flux[(Long)c] = GlobalReduce((double)flux[(Long)c], comm, CommOp::SUM); total += flux[(Long)c]; }
      if (!comm.Rank()) {
        std::cout << "\n  [flux check] int v.n dA per cap (target +-p):\n";
        for (size_t c = 0; c < caps.size(); c++)
          std::cout << "    cap " << c << " (" << (caps[c].sgn < 0 ? "in " : "out") << "): flux="
                    << std::setprecision(6) << flux[(Long)c] << "  (target " << (Real)caps[c].sgn*caps[c].p << ")\n";
        std::cout << "    TOTAL net flux = " << total << "  (compatibility condition int u.n dA = 0)\n";
      }
      SCTL_ASSERT_MSG(std::fabs((double)total) < 1e-6, "net flux not zero -- interior Stokes BVP incompatible");
    }

    // ----------------------------------------------------------------------------------------------
    // (5) Interior probe point: on the inlet-arm axis, mid-arm (deep in the fluid, no mask needed). The
    //     inlet arm carries the full flux Q=p_in; fully-developed on-axis Poiseuille umax=2Q/(pi R0^2)
    //     is an APPROXIMATE reference (arms are short ~3*R0). Supplied on rank 0 only.
    // ----------------------------------------------------------------------------------------------
    const Real s_mid = (Real)0.5*(a0[0] + s_cap);
    const Real umax  = 2*p_in/(const_pi<Real>()*R0[0]*R0[0]);
    Vector<Real> Xpt;
    if (!comm.Rank()) { Xpt.PushBack(s_mid*u0[0]); Xpt.PushBack(s_mid*u0[1]); Xpt.PushBack(s_mid*u0[2]); }
    if (!comm.Rank())
      std::cout << "\n  [probe] inlet-arm axis point s_mid=" << std::setprecision(4) << s_mid
                << " X=(" << s_mid*u0[0] << "," << s_mid*u0[1] << "," << s_mid*u0[2]
                << ")  Poiseuille umax=" << std::setprecision(6) << umax << "\n";

    // ----------------------------------------------------------------------------------------------
    // (6) Solve for each requested arm type and collect the probe velocity + GMRES iterations.
    // ----------------------------------------------------------------------------------------------
    Vec3<Real> U_csbq{0,0,0}, U_quad{0,0,0};
    Long it_csbq = -1, it_quad = -1;
    bool have_csbq = false, have_quad = false;

    if (armmode == 0 || armmode == 2) {
      if (!comm.Rank()) std::cout << "\n--- CSBQ slender arms ---\n";
      SlenderElemList<Real> arms = BuildYArmsSlender<Real>(R0, a0, sc, nAxial, 10, fourier, comm);
      run_flow<Real>(junc, arms, comm, caps, tol, Xpt, "csbq", U_csbq, it_csbq);
      have_csbq = true;
    }
    if (armmode == 1 || armmode == 2) {
      if (!comm.Rank()) std::cout << "\n--- quad-tube arms (Na=" << NaShaft << " conforming) ---\n";
      QuadElemList<Real> arms = BuildYArmsQuadTube<Real>(ord, R0, a0, sc, NsShaft, NaShaft, comm);
      arms.SetQuadScheme(quad_junctions::QJDefaultScheme<Real>(), cov_q, Nbeta, maxdep);
      run_flow<Real>(junc, arms, comm, caps, tol, Xpt, "quad-tube", U_quad, it_quad);
      have_quad = true;
    }

    // ----------------------------------------------------------------------------------------------
    // (7) Comparison table (rank 0). Axial component = U.u0 (inflow => flow toward junction along -u0,
    //     so U.u0 < 0 with |U.u0| ~ umax).
    // ----------------------------------------------------------------------------------------------
    if (!comm.Rank()) {
      auto axial = [&](const Vec3<Real>& U) { return dot3(U, u0); };
      auto mag   = [&](const Vec3<Real>& U) { return sqrt<Real>(dot3(U, U)); };
      std::cout << "\n=== interior velocity at inlet-arm axis probe + GMRES iterations ===\n"
                << std::scientific << std::setprecision(8);
      if (have_csbq)
        std::cout << "  CSBQ slender : U=(" << U_csbq[0] << "," << U_csbq[1] << "," << U_csbq[2] << ")"
                  << "  axial(U.u0)=" << axial(U_csbq) << "  |U|=" << mag(U_csbq)
                  << "  GMRES iters=" << it_csbq << "\n";
      if (have_quad)
        std::cout << "  quad-tube    : U=(" << U_quad[0] << "," << U_quad[1] << "," << U_quad[2] << ")"
                  << "  axial(U.u0)=" << axial(U_quad) << "  |U|=" << mag(U_quad)
                  << "  GMRES iters=" << it_quad << "\n";
      if (have_csbq && have_quad) {
        const Vec3<Real> d{U_quad[0]-U_csbq[0], U_quad[1]-U_csbq[1], U_quad[2]-U_csbq[2]};
        const Real dn = sqrt<Real>(dot3(d, d));
        const Real ref = mag(U_csbq);
        std::cout << "  --> ||U_quad - U_csbq|| = " << dn
                  << "   rel = " << (ref > 0 ? dn/ref : dn)
                  << "   (agreement of the two discretizations at the same interior point)\n";
      }
      std::cout << std::defaultfloat
                << "  Poiseuille reference umax = " << std::setprecision(6) << umax
                << " (approximate; short arms => entrance effects)\n";
    }
  }
  Comm::MPI_Finalize();
  return 0;
}
