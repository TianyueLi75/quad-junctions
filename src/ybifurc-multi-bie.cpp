/**
 * Composable multi-junction hybrid BIE demo + accuracy check.
 *
 * Exercises the HybridAssembly component API (include/quad_junctions/ybifurc_assembly.hpp): a placed
 * junction + POU transitions, free (capped) slender arms, and shared slender arms joining two junctions,
 * all fed into ONE BoundaryIntegralOp via the shared identity tests (hybrid_bie_tests.hpp).
 *
 * It runs two cases through the SAME sweep so their accuracy is directly comparable:
 *   (1) single junction with 3 free capped arms  -- the parity anchor: must reproduce ybifurc-hybrid-bie.
 *   (2) two junctions at (-10,0,0) and (5,0,0) joined by ONE shared arm on the x-axis, the other four
 *       arms free/capped -- the requested example. Each junction is a rigid placement of the SAME
 *       canonical iso-surface (Gaussians decay to ~0 over the 15-unit gap), so both junctions are
 *       CONGRUENT to the single one and the only new object is a long constant-R0 cylinder. The DL/Green
 *       identity errors must therefore match case (1); the shared-arm slender region stays clean.
 *
 *   make bin/ybifurc-multi-bie                 # or: make MPI=1 bin/ybifurc-multi-bie
 *   OMP_NUM_THREADS=8 ./bin/ybifurc-multi-bie \
 *       [level] [order(mult4)] [nref] [eta_join] [Ns_trans] [s_cap] [n_axial_free] [fourier] [nlev] \
 *       [n_axial_shared(auto if <=0)] [sine_amp] [sine_periods] \
 *       [caseSel] [tolOv] [NbOv] [mdOv] [cov_q] [rL] [rR]
 *
 *   rL/rR (case 2): shared-arm terminal radii at junctions A/B (<=0 => native R0 ~0.269, scale 1). Each
 *   junction is uniformly scaled to match the arm end it touches; rL != rR gives a tapered shared arm.
 */

#include <csbq.hpp>                                  // CSBQ SlenderElemList
#include <quad_junctions/ybifurc_assembly.hpp>       // composable component API
#include <quad_junctions/hybrid_bie_tests.hpp>       // shared BIE identity / watertightness tests
#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <string>

using namespace sctl;
using namespace quad_junctions;

namespace {

// Divergence check + coupled DL/Green identity sweep on a combined (quad junctions + slender arms) pair.
// Region breakdown reports the max error on the quad half vs the slender half so the shared-arm region
// is visibly clean; the argmax location print localizes where the residual actually sits.
template <class Real>
void run_case(QuadElemList<Real>& junc, SlenderElemList<Real>& arms, const Comm& comm,
              const Vector<Real>& X0, const std::string& tag, const std::string& label,
              const Integer nlev, const Integer cov_q,
              const Real tolOv = (Real)0, const Integer NbOv = 0, const Integer mdOv = 0) {
  // Default coupled 4-level sweep, OR a single custom near-eval level when tolOv>0 (near-eval scan).
  Real    tolL[4] = {(Real)1e-5, (Real)1e-7, (Real)1e-9, (Real)1e-11};
  Integer NbL[4]  = {48, 100, 200, 400};
  Integer mdL[4]  = {4, 8, 12, 30};
  Integer nlevels = nlev;
  if (tolOv > (Real)0) { tolL[0] = tolOv; NbL[0] = NbOv; mdL[0] = mdOv; nlevels = 1; }
  const RegionReport<Real> region_report = [](const Vector<Real>& err, Long Nj, Long Na) {
    Real mj = 0, ma = 0;
    for (Long i = 0; i < Nj; i++) mj = std::max(mj, err[i]);
    for (Long i = 0; i < Na; i++) ma = std::max(ma, err[Nj+i]);
    std::cout << "    [region max] quad(junctions+transitions+caps)=" << mj << " slender(arms)=" << ma << "\n";
  };
  const Long njp = GlobalReduce((Long)junc.Size(), comm, CommOp::SUM), nap = GlobalReduce((Long)arms.Size(), comm, CommOp::SUM);
  Vector<Real> Xj, Xa; junc.GetNodeCoord(&Xj, nullptr, nullptr); arms.GetNodeCoord(&Xa, nullptr, nullptr);
  const Long njn = GlobalReduce((Long)(Xj.Dim()/3), comm, CommOp::SUM), nan = GlobalReduce((Long)(Xa.Dim()/3), comm, CommOp::SUM);
  if (!comm.Rank())
    std::cout << "\n---- BIE sweep [" << label << "]: quad panels=" << njp << " nodes=" << njn
              << " | slender panels=" << nap << " nodes=" << nan << " ----\n";
  junc.SetQuadScheme(QuadElemList<Real>::QuadScheme::Hybrid, cov_q, NbL[0], mdL[0]);
  divergence_check<Real>(junc, arms, tolL[0], comm);
  for (int idx = 0; idx < nlevels; idx++) {
    junc.SetQuadScheme(QuadElemList<Real>::QuadScheme::Hybrid, cov_q, NbL[idx], mdL[idx]);
    if (!comm.Rank()) std::cout << "  [tol=" << tolL[idx] << " Nbeta=" << NbL[idx] << " max_depth=" << mdL[idx] << "]\n";
    const bool dump = (idx == nlevels-1) && (tolOv <= (Real)0);  // no VTU clobber during a near-eval scan
    const std::string dt = dump ? tag : std::string();
    if (!comm.Rank()) { std::cout << "    [Laplace] "; } test_DLIdentity<Real, Laplace3D_DxU>(junc, arms, comm, tolL[idx], dump ? dt+"-dl-laplace" : "", region_report);
    if (!comm.Rank()) { std::cout << "    [Stokes]  "; } test_DLIdentity<Real, Stokes3D_DxU>(junc, arms, comm, tolL[idx], dump ? dt+"-dl-stokes" : "", region_report);
    if (!comm.Rank()) { std::cout << "    [Laplace] "; } test_greens_identity<Real, Laplace3D_FxU, Laplace3D_DxU, Laplace3D_FxdU>(junc, arms, comm, tolL[idx], X0, dump ? dt+"-green-laplace" : "");
    if (!comm.Rank()) { std::cout << "    [Stokes]  "; } test_greens_identity<Real, Stokes3D_FxU, Stokes3D_DxU, Stokes3D_FxT>(junc, arms, comm, tolL[idx], X0, dump ? dt+"-green-stokes" : "");
  }
}

} // anonymous namespace

int main(int argc, char** argv) {
  Comm::MPI_Init(&argc, &argv);
  using Real = double;
  {
    const Comm comm = Comm::World();
    const Real    level   = (argc > 1) ? (Real)atof(argv[1]) : (Real)1.5;
    const Integer ord     = (argc > 2) ? (Integer)atoi(argv[2]) : 12;
    const Integer nref    = (argc > 3) ? (Integer)atoi(argv[3]) : 1;
    const Real    etajoin = (argc > 4) ? (Real)atof(argv[4]) : (Real)0.4;
    const Integer NsTrans = (argc > 5) ? (Integer)atoi(argv[5]) : 3;
    const Real    s_cap   = (argc > 6) ? (Real)atof(argv[6]) : (Real)0.88;
    const Integer nAxial  = (argc > 7) ? (Integer)atoi(argv[7]) : 3;    // free-arm axial panels
    const Long    fourier = (argc > 8) ? (Long)atoi(argv[8]) : 12;
    const Integer nlev    = (argc > 9) ? (Integer)atoi(argv[9]) : 4;
    const Integer nShared = (argc > 10) ? (Integer)atoi(argv[10]) : -1; // shared-arm axial panels (auto if <=0)
    const Real    sineAmp = (argc > 11) ? (Real)atof(argv[11]) : (Real)0.4;  // shared-arm sine wiggle amplitude (~1.5 R0)
    const Real    sinePer = (argc > 12) ? (Real)atof(argv[12]) : (Real)1.0;  // ... number of sine periods in the window
    // Near-eval scan controls: caseSel (0=both,1=case1,2=case2); a single custom near-eval level
    // (tolOv>0 => run ONLY that level with Nbeta=NbOv, max_depth=mdOv, cov_q=covqOv) for probing.
    const Integer caseSel = (argc > 13) ? (Integer)atoi(argv[13]) : 0;
    const Real    tolOv   = (argc > 14) ? (Real)atof(argv[14]) : (Real)0;
    const Integer NbOv    = (argc > 15) ? (Integer)atoi(argv[15]) : 0;
    const Integer mdOv    = (argc > 16) ? (Integer)atoi(argv[16]) : 0;
    const Integer cov_q   = (argc > 17) ? (Integer)atoi(argv[17]) : 6;
    // Shared-arm terminal radii (case 2): rL at junction A, rR at junction B. Each connecting junction is
    // uniformly scaled so its seam radius matches the arm end it touches (scale = r / native_R0). rL != rR
    // => a tapered shared arm between two differently-sized junctions. Sentinel <=0 => the native R0
    // (scale 1), so omitting these reproduces the previous case-2 geometry bit-for-bit.
    const Real    rL      = (argc > 18) ? (Real)atof(argv[18]) : (Real)-1;
    const Real    rR      = (argc > 19) ? (Real)atof(argv[19]) : (Real)-1;
    const Integer Ncap    = (Integer)(YSwept::Ncap0 * nref);
    pou_kind() = 1;   // smootherstep POU (order-exact) -- what the assembly's transitions expect

    SCTL_ASSERT_MSG(ord >= 4 && ord <= 48 && ord % 4 == 0, "order must be a multiple of 4 in {4,...,48}.");

    if (!comm.Rank()) {
      std::cout << "\n=== COMPOSABLE multi-junction hybrid Y-bifurcation ===\n";
      std::cout << "  order=" << ord << " level=" << level << " nref=" << nref << " eta_join=" << etajoin
                << " Ns_trans=" << NsTrans << " s_cap=" << s_cap << " n_axial(free)=" << nAxial
                << " fourier=" << fourier << " (POU=smootherstep)\n";
    }

    // ----------------------------------------------------------------------------------------------
    // (1) PARITY ANCHOR: a single junction at the origin with 3 free capped arms == the M2 single hybrid.
    // ----------------------------------------------------------------------------------------------
    if (caseSel == 0 || caseSel == 1) {
      HybridAssembly<Real> A(ord);
      const HybridJunction<Real> J = A.add_junction(Placement<Real>::Identity(), level, nref, etajoin, NsTrans);
      for (int k = 0; k < 3; k++) A.add_free_arm(J.seam(k), s_cap, nAxial, Ncap, 10, fourier);
      QuadElemList<Real> junc = A.quad(comm);
      SlenderElemList<Real> arms = A.slender(comm);
      const std::string tag = "vis/ybifurc-multi-single-ord" + std::to_string((long)ord) + "-nref" + std::to_string((long)nref);
      if (!comm.Rank()) {
        std::cout << "\n[case 1] single junction + 3 free arms  (junction ray max|f-level|=" << std::setprecision(3) << J.max_res << ")\n";
        std::cout << "  arm R0=" << std::setprecision(6) << J.seam(0).R0 << " axial [" << J.seam(0).a0 << ", " << s_cap << "]\n";
      }
      junc.WriteVTK(tag + "-junc", Vector<Real>(), comm);
      arms.WriteVTK(tag + "-arms", Vector<Real>(), comm);
      const Vector<Real> X0{(Real)1.6, (Real)1.4, (Real)0.9};
      run_case<Real>(junc, arms, comm, X0, tag, "single-junction (3 free arms)", nlev, cov_q, tolOv, NbOv, mdOv);
    }

    // ----------------------------------------------------------------------------------------------
    // (2) TWO junctions at (-10,0,0) and (5,0,0) joined by ONE shared arm on the x-axis; other 4 arms free.
    //     Junction A's arm0 points +x (toward B), junction B's arm0 points -x (toward A); both "up"=+z.
    // ----------------------------------------------------------------------------------------------
    if (caseSel == 0 || caseSel == 2) {
      HybridAssembly<Real> A(ord);
      // Arm-driven radius: the shared arm's terminal radii (rLuse at A, rRuse at B) set each junction's
      // uniform scale so its seam radius matches the arm end (scale = r/native_R0). The whole junction and
      // its free arms scale accordingly; the shared arm tapers rLuse->rRuse between them.
      const Real R0n = canonical_seam_R0<Real>(level, etajoin);
      const Real rLuse = (rL > (Real)0) ? rL : R0n, rRuse = (rR > (Real)0) ? rR : R0n;
      const Real scaleA = rLuse / R0n, scaleB = rRuse / R0n;
      const Real sineAmpS = sineAmp * (Real)0.5 * (scaleA + scaleB);   // wiggle is a length -> scale it too
      const Placement<Real> PA = Placement<Real>::AlignArm(0, Vec3<Real>{1,0,0},  Vec3<Real>{0,0,1}, Vec3<Real>{-10,0,0}, scaleA);
      const Placement<Real> PB = Placement<Real>::AlignArm(0, Vec3<Real>{-1,0,0}, Vec3<Real>{0,0,1}, Vec3<Real>{5,0,0}, scaleB);
      const HybridJunction<Real> JA = A.add_junction(PA, level, nref, etajoin, NsTrans);
      const HybridJunction<Real> JB = A.add_junction(PB, level, nref, etajoin, NsTrans);

      // shared arm (arm0 of each): auto-pick axial panels for ~unit aspect (panel length ~ tube diameter)
      const ArmSeam<Real>& sa = JA.seam(0); const ArmSeam<Real>& sb = JB.seam(0);
      const Real dx = sb.C[0]-sa.C[0], dy = sb.C[1]-sa.C[1], dz = sb.C[2]-sa.C[2];
      const Real len = std::sqrt(dx*dx+dy*dy+dz*dz);
      // Auto axial-panel count: a straight tube only needs ~unit-aspect panels (spacing ~2*R0), but a
      // wiggled centerline must be resolved to its CURVATURE -- target ~50 panels per wiggle wavelength
      // (wavelength_arc = len/periods since the sin^2-tapered sine spans the whole arm). The C-infinity
      // envelope makes convergence panel-placement-insensitive, so this need not be tuned precisely.
      Real pspac = (Real)(sa.R0 + sb.R0);   // ~mean tube diameter (handles a taper)
      if (sineAmpS != (Real)0) { const Real wl = len/std::max((Real)1e-9, sinePer); pspac = std::min(pspac, wl/50); }
      const Integer ns = (nShared > 0) ? nShared : std::max<Integer>(4, (Integer)std::lround((double)len/pspac));
      A.add_shared_arm(sa, sb, ns, 10, fourier, sineAmpS, sinePer);

      // the other two arms of each junction are free/capped; s_cap is an axial station measured from the
      // junction origin, so it scales with the junction (keeps L = s_cap - seam.a0 > 0 under scaling).
      for (int k = 1; k < 3; k++) {
        A.add_free_arm(JA.seam(k), scaleA*s_cap, nAxial, Ncap, 10, fourier);
        A.add_free_arm(JB.seam(k), scaleB*s_cap, nAxial, Ncap, 10, fourier);
      }

      QuadElemList<Real> junc = A.quad(comm);
      SlenderElemList<Real> arms = A.slender(comm);
      const std::string tag = "vis/ybifurc-multi-two-ord" + std::to_string((long)ord) + "-nref" + std::to_string((long)nref);
      // TWO Green sources, one inside EACH junction (the single-hybrid interior point transformed by that
      // junction's placement), so the error distribution lights up both bifurcations symmetrically rather
      // than only the source-bearing one. x0a inside junction A (-x side) mirrors x0b inside junction B.
      const Vec3<Real> x0b = PB.apply_point(Vec3<Real>{1.6, 1.4, 0.9});
      const Vec3<Real> x0a = PA.apply_point(Vec3<Real>{1.6, 1.4, 0.9});
      const Vector<Real> X0{x0b[0], x0b[1], x0b[2], x0a[0], x0a[1], x0a[2]};
      if (!comm.Rank()) {
        std::cout << "\n[case 2] two junctions + shared arm  (max|f-level|: A=" << std::setprecision(3) << JA.max_res << " B=" << JB.max_res << ")\n";
        std::cout << "  junction A at (-10,0,0) arm0->+x ; junction B at (5,0,0) arm0->-x\n";
        std::cout << "  arm-driven radius: native R0=" << std::setprecision(6) << R0n
                  << "  target rL=" << rLuse << " rR=" << rRuse
                  << "  -> junction scale A=" << std::setprecision(4) << scaleA << " B=" << scaleB
                  << (std::fabs((double)(rLuse-rRuse)) > 1e-12 ? "  (tapered)" : "  (uniform)") << "\n";
        std::cout << "  shared arm: R0=" << std::setprecision(6) << sa.R0 << " length=" << len << " axial panels=" << ns
                  << "  from (" << std::setprecision(4) << sa.C[0] << "," << sa.C[1] << "," << sa.C[2]
                  << ") to (" << sb.C[0] << "," << sb.C[1] << "," << sb.C[2] << ")\n";
        std::cout << "  shared-arm sine wiggle: amp=" << std::setprecision(3) << sineAmpS << " periods=" << sinePer
                  << " with a sin^2 taper -> 0 (zero slope) at both seams, concentrated mid-arm/away from the connections\n";
        std::cout << "  Green sources: X0_B=(" << std::setprecision(4) << x0b[0] << "," << x0b[1] << "," << x0b[2] << ") near junction B"
                  << " ; X0_A=(" << x0a[0] << "," << x0a[1] << "," << x0a[2] << ") near junction A\n";
      }

      // Verify each Green source is in free space OUTSIDE the solid (the interior identity's -1/2 jump
      // form needs exterior sources; one landing inside would break it). Two guards: (a) field < level in
      // each junction's local frame (outside both blobs); (b) distance to the arm axis segment exceeds
      // R0 + |wiggle amp| (outside the whole swept tube, bent or not).
      {
        const YField<Real> fldchk;
        auto seg_dist = [](const Vec3<Real>& P, const Vec3<Real>& Aa, const Vec3<Real>& Bb) {
          const Vec3<Real> ab{Bb[0]-Aa[0], Bb[1]-Aa[1], Bb[2]-Aa[2]}, ap{P[0]-Aa[0], P[1]-Aa[1], P[2]-Aa[2]};
          const Real L2 = ab[0]*ab[0]+ab[1]*ab[1]+ab[2]*ab[2];
          Real tp = (L2 > 0) ? (ap[0]*ab[0]+ap[1]*ab[1]+ap[2]*ab[2])/L2 : (Real)0;
          tp = std::max<Real>(0, std::min<Real>(1, tp));
          const Real dx=P[0]-(Aa[0]+tp*ab[0]), dy=P[1]-(Aa[1]+tp*ab[1]), dz=P[2]-(Aa[2]+tp*ab[2]);
          return std::sqrt(dx*dx+dy*dy+dz*dz);
        };
        const Real rtube = std::max(sa.R0, sb.R0) + std::fabs((double)sineAmpS);   // taper-safe
        auto check = [&](const char* name, const Vec3<Real>& Xs) {
          const Real fA = fldchk.f(PA.apply_inverse_point(Xs)), fB = fldchk.f(PB.apply_inverse_point(Xs));
          const Real dseg = seg_dist(Xs, sa.C, sb.C);
          const bool inside = (fA >= level) || (fB >= level) || (dseg <= rtube);
          if (!comm.Rank())
            std::cout << "  [source check] " << name << ": f_A=" << std::setprecision(3) << fA << " f_B=" << fB
                      << " (<" << level << "?)  dist_to_arm=" << dseg << " (>" << rtube << "?)  => "
                      << (inside ? "INSIDE geometry (BAD)" : "exterior OK") << "\n";
          SCTL_ASSERT_MSG(!inside, "Green source landed inside the geometry.");
        };
        check("X0_B", x0b); check("X0_A", x0a);
      }
      junc.WriteVTK(tag + "-junc", Vector<Real>(), comm);
      arms.WriteVTK(tag + "-arms", Vector<Real>(), comm);
      run_case<Real>(junc, arms, comm, X0, tag, "two junctions + shared arm", nlev, cov_q, tolOv, NbOv, mdOv);

      // ------------------------------------------------------------------------------------------
      // MANUFACTURED-SOLUTION CFIE solve (GMRES) on this same two-junction geometry.
      // The exact field is the SL potential of the SAME two EXTERIOR sources used above (x0b, x0a).
      // Since the sources are exterior, the field is harmonic INSIDE the solid -> interior Dirichlet
      // problem: solve ( c*I - S + D ) sigma = u_e|surface, evaluate at interior probe points, compare.
      // Default accurate near-eval: Hybrid(cov_q=6, Nbeta=200, max_depth=8) + SetAccuracy(1e-8).
      // ------------------------------------------------------------------------------------------
      {
        const Real man_tol = (Real)1e-8;
        junc.SetQuadScheme(QuadElemList<Real>::QuadScheme::Hybrid, 6, 200, 8);

        // Two exterior sources (reuse the Green-identity points), ordered [x0b, x0a].
        const Vector<Real> Xsrc{x0b[0], x0b[1], x0b[2], x0a[0], x0a[1], x0a[2]};

        // Interior probe targets, built on rank 0 only (so GlobalReduce counts each once). All lie on a
        // tube/arm centerline (radius 0 < R0) -> guaranteed interior by construction; junction centers
        // are additionally validated by the YField inside-check before inclusion.
        Vector<Real> Xtrg;
        if (!comm.Rank()) {
          auto push = [&](const Vec3<Real>& P) { Xtrg.PushBack(P[0]); Xtrg.PushBack(P[1]); Xtrg.PushBack(P[2]); };
          // shared-arm axis (on the straight seam-to-seam segment; the sine wiggle displaces the surface,
          // not this axis, so these stay interior)
          for (const Real t : {(Real)0.25, (Real)0.5, (Real)0.75})
            push(Vec3<Real>{sa.C[0]+t*(sb.C[0]-sa.C[0]), sa.C[1]+t*(sb.C[1]-sa.C[1]), sa.C[2]+t*(sb.C[2]-sa.C[2])});
          // each free arm: a couple of on-axis stations stepped into the arm from the seam ring
          const ArmSeam<Real>* freearms[4] = {&JA.seam(1), &JA.seam(2), &JB.seam(1), &JB.seam(2)};
          for (const ArmSeam<Real>* s : freearms)
            for (const Real k : {(Real)1, (Real)2})
              push(Vec3<Real>{s->C[0]+k*s->R0*s->u[0], s->C[1]+k*s->R0*s->u[1], s->C[2]+k*s->R0*s->u[2]});
          // junction centers (canonical origin), included only if confirmed inside a junction blob
          const YField<Real> fldchk;
          for (const Placement<Real>* P : {&PA, &PB}) {
            const Vec3<Real> C = P->apply_point(Vec3<Real>{0,0,0});
            if (fldchk.f(P->apply_inverse_point(C)) >= level) push(C);
          }
          SCTL_ASSERT_MSG(Xtrg.Dim() > 0, "no interior probe targets for the manufactured test");
          std::cout << "\n[case 2] MANUFACTURED-SOLUTION CFIE solve (interior Dirichlet, 2 exterior sources)\n"
                    << "  near-eval: Hybrid(cov_q=6, Nbeta=200, max_depth=8) tol=" << std::setprecision(1) << man_tol
                    << " (GMRES tol=" << man_tol*10 << ")  interior probe targets=" << Xtrg.Dim()/3 << "\n";
        }

        // Laplace CFIE: recover the two point-charge potential (strengths +1, -1).
        const Vector<Real> Fsrc_lap{(Real)1, (Real)-1};
        test_manufactured<Real, Laplace3D_FxU, Laplace3D_DxU>(
            junc, arms, comm, man_tol, Xsrc, Fsrc_lap, /*interior=*/true, Xtrg,
            /*SL_scal=*/(Real)-1, /*DL_scal=*/(Real)1, "Laplace manufactured");

        // Stokes CFIE: recover the two-Stokeslet velocity field.
        const Vector<Real> Fsrc_stk{(Real)1, (Real)0.5, (Real)-0.3, (Real)-1, (Real)-0.5, (Real)0.3};
        test_manufactured<Real, Stokes3D_FxU, Stokes3D_DxU>(
            junc, arms, comm, man_tol, Xsrc, Fsrc_stk, /*interior=*/true, Xtrg,
            /*SL_scal=*/(Real)-1, /*DL_scal=*/(Real)1, "Stokes manufactured");
      }
    }
  }
  Comm::MPI_Finalize();
  return 0;
}
