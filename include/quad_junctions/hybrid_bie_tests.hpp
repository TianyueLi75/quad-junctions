/**
 * Shared BIE identity / watertightness tests for the hybrid geometries.
 *
 * Extracted verbatim (behavior-preserving) from src/ybifurc-hybrid-bie.cpp so that both the
 * single-junction hybrid driver and the composable multi-junction driver (ybifurc-multi-bie.cpp)
 * reuse the same coupled QuadElemList+ArmList identity machinery:
 *
 *   - combined_nodes      : concatenate the two lists' node coords/normals in NAME-SORTED order
 *   - divergence_check     : int n dA = 0 watertightness / orientation diagnostic
 *   - test_DLIdentity      : DL constant-density identity (-> -1/2 on a closed, outward surface)
 *   - test_greens_identity : interior Green's third identity for a source X0 inside the surface
 *
 * All are templated on the arm list type (SlenderElemList or QuadElemList) since a BoundaryIntegralOp
 * couples any two element lists. The DRIVER-SPECIFIC per-region error breakdown is injected through an
 * optional `region_report` callback (invoked serial-only, where the node<->region map is meaningful),
 * so this header carries no assumptions about how a particular geometry is laid out.
 */
#pragma once

#include <sctl.hpp>
#include <stokes_bio.hpp>                     // StokesBIO: combined-field Stokes op with PVFMM-safe FMM kernels
#include <quad_junctions/fmm_kernels.hpp>     // SetPVFMMKer for the non-solve (Laplace + Stokes) operators
#include <quad_junctions/junction_precond.hpp> // block-diagonal left preconditioner on the junction rows
#include <cmath>
#include <functional>
#include <iomanip>
#include <iostream>
#include <string>

namespace quad_junctions {
using namespace sctl;

// A region-breakdown callback: given the per-node error vector and the junction/arm split
// (err has Nj junction nodes followed by Na arm nodes), print whatever per-region summary the
// caller wants. Invoked ONLY on a single serial rank, where the global node ordering is intact.
template <class Real> using RegionReport = std::function<void(const Vector<Real>& err, Long Nj, Long Na)>;

// Concatenate the two lists' node coords/normals in NAME-SORTED order (junc "0_..." then arms "1_...").
template <class Real, class ArmList>
void combined_nodes(const QuadElemList<Real>& junc, const ArmList& arms,
                    Vector<Real>& X, Vector<Real>& Xn, Long& Nj, Long& Na) {
  Vector<Real> Xj, Xnj, Xa, Xna;
  junc.GetNodeCoord(&Xj, &Xnj, nullptr);
  arms.GetNodeCoord(&Xa, &Xna, nullptr);
  Nj = Xj.Dim()/3; Na = Xa.Dim()/3;
  X.ReInit(0); Xn.ReInit(0);
  for (auto v : Xj)  X.PushBack(v);
  for (auto v : Xa)  X.PushBack(v);
  for (auto v : Xnj) Xn.PushBack(v);
  for (auto v : Xna) Xn.PushBack(v);
}

// ---- Watertightness / orientation check: int n dA = 0 for any closed, consistently-oriented surface.
//      Reports combined flux vector (should be ~0), per-list flux, and areas. A nonzero combined flux
//      localizes a gap or a flipped-normal region. Uses the far-field smooth quadrature of each list. ----
template <class Real, class ArmList>
void divergence_check(const QuadElemList<Real>& junc, const ArmList& arms, const Real tol, const Comm& comm) {
  auto flux_area = [&](const auto& lst, Real f[3]) -> Real {
    Vector<Real> X, Xn, wts, dist; Vector<Long> cnt;
    lst.GetFarFieldNodes(X, Xn, wts, dist, cnt, tol);
    const Long N = wts.Dim(); Real A = 0; f[0]=f[1]=f[2]=0;
    for (Long i = 0; i < N; i++) { A += wts[i]; for (int k=0;k<3;k++) f[k] += wts[i]*Xn[i*3+k]; }
    return A;
  };
  Real fj[3], fa[3];
  Real Aj = flux_area(junc, fj), Aa = flux_area(arms, fa);
  // Local slices only -> reduce areas and the (vector) fluxes across ranks before taking magnitudes.
  Aj = GlobalReduce((double)Aj, comm, CommOp::SUM);
  Aa = GlobalReduce((double)Aa, comm, CommOp::SUM);
  for (int k = 0; k < 3; k++) { fj[k] = GlobalReduce((double)fj[k], comm, CommOp::SUM); fa[k] = GlobalReduce((double)fa[k], comm, CommOp::SUM); }
  const Real fc[3] = {fj[0]+fa[0], fj[1]+fa[1], fj[2]+fa[2]};
  auto nrm = [](const Real f[3]){ return std::sqrt(f[0]*f[0]+f[1]*f[1]+f[2]*f[2]); };
  if (!comm.Rank())
    std::cout << std::setprecision(4)
              << "  [watertight] area: junc=" << Aj << " arms=" << Aa << " total=" << (Aj+Aa) << "\n"
              << "  [watertight] |int n dA|: junc=" << nrm(fj) << " arms=" << nrm(fa)
              << "  COMBINED=" << nrm(fc) << "  (combined should be ~0 for a closed surface)\n";
}

// ---- DL constant-density identity: -1/2 on a closed surface with outward normals ----
template <class Real, class KerDL, class ArmList>
void test_DLIdentity(const QuadElemList<Real>& junc, const ArmList& arms, const Comm& comm, const Real tol,
                     const std::string& dump_tag = "", const RegionReport<Real>& region_report = {}) {
  BoundaryIntegralOp<Real, KerDL> BIOp((KerDL()), false, comm);
  SetPVFMMKer(BIOp);
  BIOp.SetAccuracy(tol);
  BIOp.AddElemList(junc, "0_junc");
  BIOp.AddElemList(arms, "1_arms");
  const Long KDIM0 = KerDL::SrcDim();
  Vector<Real> X, Xn; Long Nj, Na; combined_nodes(junc, arms, X, Xn, Nj, Na);
  const Long Nnode = Nj + Na;
  Vector<Real> q(Nnode*KDIM0), U;
  for (Long i = 0; i < Nnode; i++) for (Long k = 0; k < KDIM0; k++) q[i*KDIM0+k] = k+1;
  BIOp.ComputePotential(U, q);
  SCTL_ASSERT_MSG(U.Dim() == Nnode*KerDL::TrgDim(), "hybrid node-count/ordering mismatch");
  Real emax = 0; Long argmax = 0; Vector<Real> ce(KDIM0); ce = 0; Vector<Real> err(Nnode);
  for (Long i = 0; i < Nnode; i++) { Real ei = 0;
    for (Long k = 0; k < KDIM0; k++) { const Real e = std::fabs(U[i*KDIM0+k]/q[i*KDIM0+k] + (Real)0.5); ce[k]=std::max(ce[k],e); ei=std::max(ei,e); if (e>emax){emax=e;argmax=i;} }
    err[i] = ei; }
  // Reduce per-component maxima -> global avg; reduce overall max + its location across ranks.
  Vector<double> ced(KDIM0); for (Long k = 0; k < KDIM0; k++) ced[k] = (double)ce[k];
  GlobalReduce(ced, comm, CommOp::MAX);
  Real avg = 0; for (Long k = 0; k < KDIM0; k++) avg += ced[k]/(double)0.5; avg /= KDIM0;
  const double loc_xyz[3] = { (double)(Nnode?X[argmax*3]:0), (double)(Nnode?X[argmax*3+1]:0), (double)(Nnode?X[argmax*3+2]:0) };
  double gloc[3]; const double gemax = GlobalMaxLoc((double)emax, loc_xyz, comm, gloc);
  if (!comm.Rank()) {
    std::cout << std::setprecision(6) << "  DL const-density identity: max rel err = " << avg << "  (max abs = " << gemax << ")\n";
    std::cout << "    [loc] argmax node at (" << std::setprecision(3) << gloc[0] << "," << gloc[1] << "," << gloc[2]
              << ")  r_xy=" << std::hypot(gloc[0],gloc[1]) << "  |z|=" << std::fabs(gloc[2]) << "\n";
  }
  // per-region max: only meaningful in serial (the local node index no longer maps to a region under
  // MPI once the lists are partitioned). Delegated to the caller-supplied region_report closure.
  if (comm.Size() == 1 && !comm.Rank() && region_report) region_report(err, Nj, Na);
  if (!dump_tag.empty()) {
    Vector<Real> ej(Nj), ea(Na);
    for (Long i = 0; i < Nj; i++) ej[i] = err[i];
    for (Long i = 0; i < Na; i++) ea[i] = err[Nj+i];
    junc.WriteVTK(dump_tag + "-junc", ej, comm);   // collective
    arms.WriteVTK(dump_tag + "-arms", ea, comm);   // collective
    if (!comm.Rank()) std::cout << "    [dump] wrote " << dump_tag << "-{junc,arms}.vtu (colored by DL error)\n";
  }
}

// ---- Interior Green's identity over the combined surface. X0 is a flat list of M >= 1 interior point
//      sources ([x0,y0,z0, x1,y1,z1, ...]); the reference field is the SUM of their single-layer
//      potentials (each with an independent random strength), so the error distribution localizes near
//      EVERY source. M=1 is byte-identical to a single-source run (same drand48 draw sequence). ----
template <class Real, class KerSL, class KerDL, class KerGrad, class ArmList>
void test_greens_identity(const QuadElemList<Real>& junc, const ArmList& arms,
                          const Comm& comm, const Real tol, const Vector<Real> X0, const std::string& dump_tag = "") {
  static constexpr Integer CDIM = 3;
  KerSL ksl; KerDL kdl; KerGrad kgr;
  BoundaryIntegralOp<Real,KerSL> BIOpSL(ksl, false, comm); BoundaryIntegralOp<Real,KerDL> BIOpDL(kdl, false, comm);
  SetPVFMMKer(BIOpSL); SetPVFMMKer(BIOpDL);
  BIOpSL.SetAccuracy(tol); BIOpDL.SetAccuracy(tol);
  BIOpSL.AddElemList(junc, "0_junc"); BIOpSL.AddElemList(arms, "1_arms");
  BIOpDL.AddElemList(junc, "0_junc"); BIOpDL.AddElemList(arms, "1_arms");

  Vector<Real> X, Xn; Long Nj, Na; combined_nodes(junc, arms, X, Xn, Nj, Na);
  const Long N = Nj + Na;
  const Long Msrc = X0.Dim()/CDIM;   // number of interior point sources (fields are summed)
  Vector<Real> Fs, Fd, Uref, Us, Ud;
  { constexpr Integer KDIM0 = KerSL::SrcDim();
    const Vector<Real> Xn0{0,0,0};
    Vector<Real> dU;
    for (Long s = 0; s < Msrc; s++) {
      Vector<Real> Xs(CDIM); for (Integer d = 0; d < CDIM; d++) Xs[d] = X0[s*CDIM+d];
      Vector<Real> F0(KDIM0); for (auto& x : F0) x = drand48()-0.5;
      Vector<Real> Ur, dUr; ksl.Eval(Ur, X, Xs, Xn0, F0); kgr.Eval(dUr, X, Xs, Xn0, F0);
      if (s == 0) { Uref = Ur; dU = dUr; }
      else { for (Long i = 0; i < Ur.Dim(); i++) Uref[i] += Ur[i]; for (Long i = 0; i < dUr.Dim(); i++) dU[i] += dUr[i]; }
    }
    Fd = Uref; Fs.ReInit(N*KDIM0);
    for (Long i = 0; i < N; i++) for (Integer j = 0; j < KDIM0; j++) { Real d=0; for (Long k=0;k<CDIM;k++) d += dU[(i*KDIM0+j)*CDIM+k]*Xn[i*CDIM+k]; Fs[i*KDIM0+j]=d; } }
  // Warm-up run (warms caches/allocations), then clear setup so the timed run re-measures Setup+Eval
  BIOpSL.ComputePotential(Us, Fs);
  BIOpDL.ComputePotential(Ud, Fd);
  BIOpSL.ClearSetup();
  BIOpDL.ClearSetup();
  Us = 0; Ud = 0;

  Profile::Enable(true);
  Profile::reset();   // clear warm-up counters so only the timed Setup+Eval region is reported
  Profile::Tic("Setup+Eval", &comm);
  BIOpSL.ComputePotential(Us, Fs);
  BIOpDL.ComputePotential(Ud, Fd);
  Profile::Toc();
  Profile::print(&comm, {"t_avg", "f/s_avg"});
  Profile::reset();
  Ud -= 0.5*Fd;
  Vector<Real> Uerr = (Us - Ud) - Uref;
  Real me = 0, mv = 0; for (auto x : Uerr) me = std::max<Real>(me, std::fabs(x)); for (auto x : Uref) mv = std::max<Real>(mv, std::fabs(x));
  me = GlobalReduce((double)me, comm, CommOp::MAX);   // local slices -> global max error / value
  mv = GlobalReduce((double)mv, comm, CommOp::MAX);
  if (!comm.Rank()) std::cout << "  Green's identity error = " << me/mv << '\n';
  if (!dump_tag.empty()) {
    constexpr Integer KDIM1 = KerSL::TrgDim();
    Vector<Real> err(N);
    for (Long i = 0; i < N; i++) { Real ei = 0; for (Integer k = 0; k < KDIM1; k++) ei = std::max<Real>(ei, std::fabs(Uerr[i*KDIM1+k])); err[i] = ei/mv; }
    Vector<Real> ej(Nj), ea(Na);
    for (Long i = 0; i < Nj; i++) ej[i] = err[i];
    for (Long i = 0; i < Na; i++) ea[i] = err[Nj+i];
    junc.WriteVTK(dump_tag + "-junc", ej, comm);   // collective
    arms.WriteVTK(dump_tag + "-arms", ea, comm);   // collective
    if (!comm.Rank()) std::cout << "    [dump] wrote " << dump_tag << "-{junc,arms}.vtu (colored by Green's error)\n";
  }
}

// ---- Manufactured-solution combined-field BIE (CFIE) solve on the coupled quad+slender surface.
//      The exact field is the single-layer potential of point sources placed on the side OPPOSITE the
//      solution domain (Xsrc, strengths Fsrc); its Dirichlet trace on the surface is the RHS. Solve
//          ( c*I + SL_scal*S + DL_scal*D ) sigma = u_e|surface     via GMRES,
//      then evaluate the represented potential at off-surface targets Xtrg and report the relative-L2
//      error against the exact field there. Returns that rel-L2.
//
//      This is the FIRST routine here that solves a linear system (all others are forward-apply
//      identities). It reuses combined_nodes + the AddElemList("0_junc")/("1_arms") coupling so the
//      density spans the same combined node ordering as the identity tests.
//
//      jump c = (interior ? -1/2 : +1/2)*DL_scal, fixed from the KNOWN OUTWARD normal orientation of
//      this geometry (the divergence_check / DL identity that run earlier confirm it). We deliberately
//      do NOT use a sum(x.n) heuristic to detect orientation: for an off-origin assembly (e.g. a
//      junction at x=-10) the raw node-coordinate.normal sum is dominated by the offset and misleads.
//
//      Interior Dirichlet with same-sign SL/DL has an artificial null space; the SL sign is flipped
//      (matching the fork's TestManufactured). MPI: supply Xtrg on rank 0 only (empty elsewhere) so the
//      GlobalReduce counts each target once. ----
template <class Real, class KerSL, class KerDL, class ArmList>
Real test_manufactured(const QuadElemList<Real>& junc, const ArmList& arms, const Comm& comm,
                       const Real tol, const Vector<Real>& Xsrc, const Vector<Real>& Fsrc,
                       const bool interior, const Vector<Real>& Xtrg,
                       Real SL_scal, Real DL_scal, const std::string& name,
                       const Long gmres_max_iter = 400) {
  // The solve runs through StokesBIO, which hardcodes the Stokes3D kernel family; KerSL/KerDL stay in the
  // signature because callers name them explicitly and ker_sl still supplies the exact reference field.
  static_assert(std::is_same<KerSL, Stokes3D_FxU>::value && std::is_same<KerDL, Stokes3D_DxU>::value,
                "test_manufactured solves via StokesBIO -- KerSL/KerDL must be Stokes3D_FxU/Stokes3D_DxU");
  KerSL ker_sl;

  // Combined surface nodes/normals (local slice). Density sigma spans these in "0_junc" then "1_arms" order.
  Vector<Real> X, Xn; Long Nj, Na; combined_nodes(junc, arms, X, Xn, Nj, Na);

  // Fixed jump from the known outward orientation; interior needs opposite-sign SL/DL (else null space).
  const Real jump = (interior ? (Real)-0.5 : (Real)0.5) * DL_scal;
  if (interior && SL_scal*DL_scal > (Real)0) {
    if (!comm.Rank()) std::cout << "    [" << name << "] interior CFIE null space -> flipping SL sign\n";
    SL_scal = -SL_scal;
  }

  // Dirichlet data: SL field of the exterior sources at the surface nodes (SL kernel ignores src normal).
  Vector<Real> bc;
  ker_sl.Eval(bc, X, Xsrc, Xsrc, Fsrc);

  // Combined-field operator on the SAME coupled surface (junction quad + slender arms). StokesBIO holds
  // the SL and DL BoundaryIntegralOps together and applies SL_scal*S + DL_scal*D; its ctor installs the
  // PVFMM-safe translation kernels (DL: M2M/M2L/M2T=FSxU, L2L/L2T=FxU), which the default
  // BoundaryIntegralOp setup does not -- see include/quad_junctions/fmm_kernels.hpp for why that matters.
  StokesBIO<Real> Op(SL_scal, DL_scal, comm);
  Op.SetAccuracy(tol);
  Op.AddElemList(junc, "0_junc"); Op.AddElemList(arms, "1_arms");
  Op.SetTargetCoord(X);

  const auto ApplyK = [&](Vector<Real>* U, const Vector<Real>& sigma) {
    Vector<Real> Uc;
    Op.ComputePotential(Uc, sigma);   // = SL_scal*S[sigma] + DL_scal*D[sigma]
    if (U->Dim() != sigma.Dim()) U->ReInit(sigma.Dim());
    (*U) = Uc + jump*sigma;
  };

  GMRES<Real> solver(comm);
  KrylovPrecond<Real> krylov;
  Vector<Real> sigma;
  Long iter = 0;
  const Real gmres_tol = tol * (Real)10;
  Profile::Enable(true); Profile::reset();
  Profile::Tic("gmres solve", &comm);
  solver(&sigma, ApplyK, bc, gmres_tol, gmres_max_iter, false, &iter, &krylov);
  Profile::Toc();
  Profile::print(&comm, {"t_avg", "f/s_avg"});
  Profile::reset();

  // Evaluate the recovered potential at the (off-surface, interior) targets; compare to the exact field.
  Op.SetTargetCoord(Xtrg);
  Vector<Real> U;
  Op.ComputePotential(U, sigma);                    // no jump term off-surface
  Vector<Real> Uref;
  ker_sl.Eval(Uref, Xtrg, Xsrc, Xsrc, Fsrc);

  Real err2 = 0, ref2 = 0;
  for (Long i = 0; i < U.Dim(); i++) { const Real e = U[i]-Uref[i]; err2 += e*e; ref2 += Uref[i]*Uref[i]; }
  err2 = GlobalReduce((double)err2, comm, CommOp::SUM);   // targets live on rank 0 only
  ref2 = GlobalReduce((double)ref2, comm, CommOp::SUM);
  const Real rel_l2 = std::sqrt(err2 / ref2);
  if (!comm.Rank())
    std::cout << std::setprecision(6) << "  " << name << ": GMRES iters=" << iter
              << "  rel-L2 error = " << rel_l2 << "\n";
  return rel_l2;
}

// ---- Solve a Dirichlet BVP with the combined-field representation on the coupled quad+arm surface.
//      Unlike test_manufactured (which builds a manufactured RHS from point sources and returns only an
//      error), this takes an ARBITRARY RHS `bc` given over the combined "0_junc"+"1_arms" node ordering
//      (KDIM values per node, matching KerSL::SrcDim()) and RETURNS the solved density sigma in that same
//      ordering. It solves
//          ( jump*I + SL_scal*S + DL_scal*D ) sigma = bc     via GMRES,
//      with jump = (interior ? -1/2 : +1/2)*DL_scal fixed from the KNOWN outward normal orientation of
//      this geometry (same convention as test_manufactured; deliberately NOT a sum(x.n) heuristic).
//      Interior Dirichlet with same-sign SL/DL has an artificial null space -> the SL sign is flipped.
//
//      If Xtrg is non-empty (supply on rank 0 only), the represented field SL_scal*S + DL_scal*D is
//      evaluated at those off-surface targets into *U_trg (no jump term off-surface). This is the shared
//      solver used by the physical inflow/outflow flow driver (src/ybifurc-flow-bie.cpp). ----
template <class Real, class KerSL, class KerDL, class ArmList>
Vector<Real> solve_dirichlet_bvp(const QuadElemList<Real>& junc, const ArmList& arms, const Comm& comm,
                                 const Real tol, const Vector<Real>& bc, const bool interior,
                                 Real SL_scal, Real DL_scal, const Vector<Real>& Xtrg, Vector<Real>* U_trg,
                                 const std::string& name, const Long gmres_max_iter = 400,
                                 const JunctionPrecondSpec<Real>* precond = nullptr) {
  // Solved through StokesBIO (Stokes3D kernels only); KerSL/KerDL remain in the signature because every
  // caller names them explicitly (ybifurc-flow / -vessels-flow / -channel all pass the Stokes pair).
  static_assert(std::is_same<KerSL, Stokes3D_FxU>::value && std::is_same<KerDL, Stokes3D_DxU>::value,
                "solve_dirichlet_bvp solves via StokesBIO -- KerSL/KerDL must be Stokes3D_FxU/Stokes3D_DxU");

  // Fixed jump from the known outward orientation; interior needs opposite-sign SL/DL (else null space).
  const Real jump = (interior ? (Real)-0.5 : (Real)0.5) * DL_scal;
  if (interior && SL_scal*DL_scal > (Real)0) {
    if (!comm.Rank()) std::cout << "    [" << name << "] interior CFIE null space -> flipping SL sign\n";
    SL_scal = -SL_scal;
  }

  // Combined-field operator on the SAME coupled surface (junction quad + slender arms). StokesBIO applies
  // SL_scal*S + DL_scal*D and installs the PVFMM-safe translation kernels in its ctor (the default
  // BoundaryIntegralOp setup would register the DL kernel for M2M/M2L/L2L and corrupt the heap under
  // PVFMM -- see include/quad_junctions/fmm_kernels.hpp).
  // Combined [base ; shaft] nodes (name-sorted "0_base"<"1_shaft"); the DL jump/mean act on these DOFs.
  Vector<Real> X0, Xn0; Long Nb = 0, Ns = 0;
  combined_nodes(junc, arms, X0, Xn0, Nb, Ns);

  StokesBIO<Real> Op(SL_scal, DL_scal, comm);
  Op.SetAccuracy(tol);
  Op.AddElemList(junc, "0_junc"); Op.AddElemList(arms, "1_arms");
  Op.SetTargetCoord(X0);

  const auto ApplyK = [&](Vector<Real>* U, const Vector<Real>& sigma) {
    Vector<Real> Uc;
    Op.ComputePotential(Uc, sigma);   // = SL_scal*S[sigma] + DL_scal*D[sigma]
    if (U->Dim() != sigma.Dim()) U->ReInit(sigma.Dim());
    (*U) = Uc + jump*sigma;
  };

  // ---- Optional block-diagonal LEFT preconditioner on the junction rows (see
  // ---- quad_junctions/junction_precond.hpp). Left-preconditioning means we solve
  // ----     A11^+ A sigma = A11^+ b
  // ---- so `sigma` is unchanged but the residual GMRES minimises is the PRECONDITIONED
  // ---- one -- gmres_tol therefore applies to ||A11^+(b - A sigma)||, not ||b - A sigma||.
  // ---- Same construction as ../stokes-periodize-numtest test/examples.cpp (BIO_precond +
  // ---- A11invF). Built here rather than by the caller so the jump/SL_scal it preconditions
  // ---- are BY CONSTRUCTION the ones this solve applies (incl. the SL sign flip above).
  BlockPrecond<Real> P;
  if (precond && precond->kind != PrecondBlockKind::Off) {
    P = build_block_precond<Real>(precond->kind, precond->order, precond->level, precond->nref,
                                  precond->eta_join, precond->Ns_trans, SL_scal, DL_scal, jump,
                                  tol, comm);
    precond_set_local_rows<Real>(P, junc.Size(), precond->njunc, precond->order, precond->nref,
                                 precond->Ns_trans, comm);
  }
  const auto ApplyK_precond = [&](Vector<Real>* U, const Vector<Real>& sigma) {
    Vector<Real> Uraw;
    ApplyK(&Uraw, sigma);
    P.Apply(*U, Uraw);
  };
  Vector<Real> bc_use;
  if (P.active()) P.Apply(bc_use, bc); else bc_use = bc;

  // verbose=true -> SCTL prints "%3lld KSP Residual norm %.12e" per iteration (rank 0 only). On a solve
  // that can run hundreds of matvecs at minutes each, the residual history is the only way to tell slow
  // convergence from stagnation, and it costs one line of log per iteration.
  GMRES<Real> solver(comm, true);
  KrylovPrecond<Real> krylov;
  Vector<Real> sigma;
  Long iter = 0;
  const Real gmres_tol = tol * (Real)10;
  Profile::Enable(true); Profile::reset();
  Profile::Tic("gmres solve", &comm);
  if (P.active()) solver(&sigma, ApplyK_precond, bc_use, gmres_tol, gmres_max_iter, false, &iter, &krylov);
  else            solver(&sigma, ApplyK,         bc_use, gmres_tol, gmres_max_iter, false, &iter, &krylov);
  Profile::Toc();
  Profile::print(&comm, {"t_avg", "f/s_avg"});
  Profile::reset();
  if (!comm.Rank()) std::cout << "  " << name << ": GMRES iters=" << iter << "\n";

  // TRUE (unpreconditioned) relative residual ||b - A sigma|| / ||b||. Mandatory once left
  // preconditioning is in play: the per-iteration "KSP Residual norm" history is then the
  // PRECONDITIONED residual, so comparing it against an unpreconditioned run -- or against
  // gmres_tol -- is meaningless. One extra matvec.
  {
    Vector<Real> Ax, r;
    ApplyK(&Ax, sigma);
    r = bc - Ax;
    Real rn = 0, bn = 0;
    for (Long i = 0; i < r.Dim();  i++) rn += r[i]*r[i];
    for (Long i = 0; i < bc.Dim(); i++) bn += bc[i]*bc[i];
    rn = GlobalReduce(rn, comm, CommOp::SUM);
    bn = GlobalReduce(bn, comm, CommOp::SUM);
    if (!comm.Rank())
      std::cout << "  " << name << ": TRUE rel residual ||b-A x||/||b|| = " << std::scientific
                << std::setprecision(6) << sqrt<Real>(rn) / (bn > 0 ? sqrt<Real>(bn) : (Real)1)
                << (P.active() ? "   (precond: ON)" : "   (precond: off)") << std::defaultfloat << "\n";
  }

  // Evaluate the represented field at the (off-surface) targets (no jump term off-surface). This is
  // COLLECTIVE: every rank must call SetTargetCoord/ComputePotential even if its local Xtrg slice is
  // empty (targets are typically supplied on rank 0 only), so the guard is on U_trg, NOT on Xtrg.Dim().
  if (U_trg) {
    Op.SetTargetCoord(Xtrg);
    Op.ComputePotential(*U_trg, sigma);
  }
  return sigma;
}

} // namespace quad_junctions
