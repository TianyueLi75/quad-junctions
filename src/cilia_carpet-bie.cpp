/**
 * Doubly-periodic (XY) Stokes "cilia carpet" driver.
 *
 * Two flat walls at z=z_bottom and z=z_top spanning the unit box [0,1]x[0,1], each tiled with an
 * Npatch x Npatch grid of collar-finger cilium studs pointing toward each other (outward normals point
 * AWAY from each other = the project's outward-from-solid convention, DL identity = -1/2). Geometry built
 * by BuildCiliaCarpet (include/quad_junctions/plane_cilia_geom.hpp) as ONE QuadElemList.
 *
 * Periodicity requires PVFMM (make PVFMM=1). EvalPVFMM falls back to direct summation below 40000 global
 * targets, so periodic runs must exceed that (the banner prints node/target counts). Run with
 *   PVFMM_DIR=extern/pvfmm OMP_NUM_THREADS=8 ./bin/cilia_carpet-bie ...
 *
 *   ./bin/cilia_carpet-bie [Npatch order tol Naz bot_tip top_tip tilt_deg pdrop Nvis fingers fourier cheb n_axial]
 *
 * The cilium shaft radius is LOCKED to R_shaft = 0.25*S (S = L/(2*Npatch)); r_fil = 0.1*R_shaft. This
 * thin patch-relative shaft is scale-invariant in Npatch (same look / same volume fraction at every grid).
 */
#include <csbq.hpp>
#include <stokes_bio.hpp>
#include <quad_junctions/fmm_kernels.hpp>
#include <quad_junctions/plane_cilia_geom.hpp>
#include <quad_junctions/plane_cilia_hybrid_geom.hpp>   // hybrid base (collar+fillet+cap, no shaft) + slender shafts
#include <quad_junctions/periodic_flow_utils.hpp>
#include <chrono>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <string>

using namespace sctl;
using namespace quad_junctions;

namespace {

// The cilia carpet is HYBRID: a QuadElemList base (walls + collar + fillet + cap, NO shaft) plus a CSBQ
// SlenderElemList that supplies one bent tube SHAFT per cilium. Both lists go into one operator, added as
// "0_base" then "1_shaft", so every combined array below is [base nodes ; shaft nodes] in that order.
// combined_nodes / combined_surface_mean are the only additions the solve needs -- otherwise the pure-quad
// flow/identity code is unchanged.

// Canonical tol -> (Nbeta, max_depth) near-singular quadrature map (matches ybifurc-hybrid-bie.cpp):
//   tol {1e-5, 1e-7, 1e-9, 1e-11} -> Nbeta {48, 100, 200, 400}, max_depth {4, 8, 12, 30}.
// The near-singular setup (SetupSingular) cost scales steeply with these, so tying them to tol avoids
// paying the tightest scheme's setup at loose tol. An intermediate tol rounds UP to the tighter (finer)
// scheme; tol beyond the ends clamps (>=1e-5 -> coarsest, <=1e-11 -> finest).
template <class Real> void quad_scheme_for_tol(Real tol, Integer& Nbeta, Integer& max_depth) {
  const Real    tolL[4] = {(Real)1e-5, (Real)1e-7, (Real)1e-9, (Real)1e-11};
  const Integer NbL[4]  = {48, 100, 200, 400};
  const Integer mdL[4]  = {4, 8, 12, 30};
  int idx = 3;   // default (also covers tol < 1e-11)
  for (int k = 0; k < 4; k++) if (tolL[k] <= tol * (Real)(1 + 1e-6)) { idx = k; break; }
  Nbeta = NbL[idx]; max_depth = mdL[idx];
}

// Concatenate base + shaft node coords / normals in NAME-SORTED order (base then shaft).
template <class Real> void combined_nodes(const QuadElemList<Real>& base, const SlenderElemList<Real>& shaft,
                                          Vector<Real>& X, Vector<Real>& Xn, Long& Nb, Long& Ns) {
  Vector<Real> Xb, Xnb, Xs, Xns;
  base.GetNodeCoord(&Xb, &Xnb, nullptr);
  shaft.GetNodeCoord(&Xs, &Xns, nullptr);
  Nb = Xb.Dim()/3; Ns = Xs.Dim()/3;
  X.ReInit(0); Xn.ReInit(0);
  for (auto v : Xb)  X.PushBack(v);
  for (auto v : Xs)  X.PushBack(v);
  for (auto v : Xnb) Xn.PushBack(v);
  for (auto v : Xns) Xn.PushBack(v);
}

// Combined surface mean of a combined-ordered density sigma (KDIM per node): for each list map its
// collocation density to the far-field nodes (GetFarFieldDensity) and integrate with the far-field
// weights, so the mean is correct for BOTH the QuadElemList base (no-op map) and the SlenderElemList
// shaft (real interpolation). Returns the per-component mean; fills total_area. This is the combined-list
// generalization of run_flow's original SurfaceIntegral(sm, sigma, wts) (which assumed a single 1:1 list).
template <class Real> void combined_surface_mean(const QuadElemList<Real>& base, const SlenderElemList<Real>& shaft,
    const Vector<Real>& sigma, const Long Nb, const Long Ns, const Integer KDIM, const Real tol,
    const Comm& comm, Vector<Real>& mean_out, Real& total_area) {
  Vector<Real> sm(KDIM); sm = 0; Real area = 0;
  auto accum = [&](const auto& lst, const Vector<Real>& sig_lst) {
    Vector<Real> Xff, Xnff, wts, dist, Fff; Vector<Long> cnt;
    lst.GetFarFieldNodes(Xff, Xnff, wts, dist, cnt, tol);
    lst.GetFarFieldDensity(Fff, sig_lst);       // collocation -> far-field density
    // GetFarFieldDensity returns EMPTY when the far nodes are 1:1 with the collocation nodes (the
    // ElementListBase default, used by QuadElemList); the operator itself then reads the collocation
    // density directly (boundary_integral.txx:1205-1216). SlenderElemList overrides it and fills Fff.
    const Vector<Real>& dens = (Fff.Dim() ? Fff : sig_lst);
    const Long Nq = wts.Dim();
    SCTL_ASSERT_MSG(dens.Dim() == Nq * KDIM, "combined_surface_mean: far-field density/weight size mismatch");
    for (Long i = 0; i < Nq; i++) { area += wts[i]; for (Integer k = 0; k < KDIM; k++) sm[k] += wts[i] * dens[i*KDIM+k]; }
  };
  Vector<Real> sb(Nb*KDIM), ss(Ns*KDIM);
  for (Long i = 0; i < Nb*KDIM; i++) sb[i] = sigma[i];
  for (Long i = 0; i < Ns*KDIM; i++) ss[i] = sigma[Nb*KDIM + i];
  accum(base, sb); accum(shaft, ss);
  total_area = GlobalReduce((double)area, comm, CommOp::SUM);
  mean_out.ReInit(KDIM);
  for (Integer k = 0; k < KDIM; k++) mean_out[k] = GlobalReduce((double)sm[k], comm, CommOp::SUM) / total_area;
}

// Pressure-driven doubly-periodic Stokes solve + volume-flow VTK. Combined-field BIE with the surface-mean
// projection (rank-deficiency fix), replicating planes_with_loops. NormalOrient=+1 (outward-from-solid).
template <class Real>
void run_flow(const QuadElemList<Real>& base, const SlenderElemList<Real>& shaft, const Comm& comm,
              const Real tol, const Real L,
              const Real pressure_drop, const Long Nvis, const bool fingers,
              const Real z_bottom, const Real z_top, const Real R_shaft,
              const std::vector<CiliumCurveFlat<Real>>& curves) {
  const Real SL_scal = 1.0, DL_scal = 1.0;
  const Real gmres_tol = std::max(tol, (Real)1e-8);
  const Long gmres_max_iter = 400;
  constexpr Integer COORD_DIM = 3;   // geometric coordinate dimension (always 3)
  constexpr Integer KDIM = 3;        // KERNEL TARGET dimension: components of the potential field per point
                                     // (Stokes velocity = 3; a Laplace kernel would be 1). StokesBIO fixes
                                     // Stokes so KDIM==3 here, but the visualization below is written in terms
                                     // of KDIM (not a hardcoded 3) so it follows the kernel's target dimension.

  // Combined [base ; shaft] nodes (name-sorted "0_base"<"1_shaft"); the DL jump/mean act on these DOFs.
  Vector<Real> X0, Xn0; Long Nb = 0, Ns = 0;
  combined_nodes(base, shaft, X0, Xn0, Nb, Ns);
  const Long Nnode = Nb + Ns;

  StokesBIO<Real> Op(SL_scal, DL_scal, comm);
  Op.SetAccuracy(tol);
  Op.AddElemList(base,  "0_base");    // <-- the only structural change: a second element list (the slender shafts)
  Op.AddElemList(shaft, "1_shaft");
  Op.SetPeriodicity(sctl::Periodicity::XY, L);
  Op.SetTargetCoord(X0);

  // All normals point OUT-OF-SOLID / into the fluid (the base is flipped to match the slender shaft, whose
  // radially-outward normal cannot be flipped -- see plane_cilia_hybrid_geom.hpp / [[stud-sphere-hybrid]]).
  // Fluid on the +n side => the fluid-side DL limit is D_pv + 1/2 sigma, i.e. a +I/2 jump => NormalOrient=-1
  // (the jump term below is `U -= 0.5*sigma*NormalOrient`). This is the sign that makes the periodic Laplace
  // DL identity spatially uniform (spread->0); NormalOrient=+1 here gave spread ~1.0 and wrong near-cilia flow.
  Vector<Real> NormalOrient(3 * Nnode); NormalOrient = -1.0;  // out-of-solid (into fluid) => +I/2 jump

  const auto BIO = [&](Vector<Real>* U, const Vector<Real>& sigma) {
    // surface-mean of sigma over the COMBINED surface (each list mapped to its far-field nodes then
    // weight-integrated -- generalizes the pure-quad SurfaceIntegral(sm,sigma,wts) to two lists).
    Vector<Real> sm; Real total_area;
    combined_surface_mean<Real>(base, shaft, sigma, Nb, Ns, KDIM, tol, comm, sm, total_area);
    Vector<Real> sigma0 = sigma; AddConstVec(sigma0, sm * (Real)-1);
    U->SetZero();
    Op.ComputePotential(*U, sigma0);
    if (DL_scal && U->Dim() == sigma0.Dim()) (*U) -= sigma0 * (Real)0.5 * NormalOrient * DL_scal;  // DL jump (surface only)
    AddConstVec(*U, sm);
  };

  GMRES<Real> solver(comm);
  Vector<Real> sigma;
  Long niter = 0;
  Vector<Real> rhs = bg_flow_2peri(X0); rhs *= (pressure_drop / L);

  // ===== PROFILING: separately-timed SETUP phase + n_rep-repeat GMRES solve (averaged per-iteration time). =====
  // (1) A warm-up solve at loose tol first pages in the PVFMM precomputed operators / CSBQ special-quad tables
  //     from disk so the timed Setup below measures COMPUTE, not one-off I/O.
  // (2) ClearSetup -> Tic -> Op.Setup() -> Toc times the near-singular + FMM setup. StokesBIO::Setup() runs
  //     LayerPotenSL.Setup() then LayerPotenDL.Setup(); each BoundaryIntegralOp::Setup prints child
  //     SetupSingular / SetupNear timers, so Profile::print shows two "Setup" blocks in order -- SL first,
  //     then DL -- each with its own SetupSingular / SetupNear (the four setup numbers).
  // (3) The solve is repeated n_rep times with a FRESH (cleared) Krylov basis each time; each repeat is
  //     Tic'd as cilia_carpet_gmres_solve<k> so Profile::print reports per-repeat t_avg/t_max/f-per-s, and
  //     we also print the wall-time average and the averaged per-ITERATION solve time (avg_solve / niter).
  Profile::Enable(true);
  { Vector<Real> sigma_warm; KrylovPrecond<Real> kw;                       // warm-up: load matrices (not timed)
    solver(&sigma_warm, BIO, rhs, (Real)1e-2, gmres_max_iter, false, nullptr, &kw); }

  Profile::reset(); Op.ClearSetup();
  Profile::Tic("cilia_carpet_setup", &comm, true);
  Op.Setup();                                                             // SL then DL (see note above)
  Profile::Toc();
  Profile::print(&comm, {"t_avg", "t_max"});                             // SetupSingular/SetupNear per SL,DL
  Profile::reset();

  // const Integer n_rep = 5;
  const Integer n_rep = 1;
  double t_solve_sum = 0;
  for (Integer rep = 0; rep < n_rep; rep++) {
    KrylovPrecond<Real> krylov;                                          // cleared each repeat
    sigma.ReInit(0);
    const std::string lbl = "cilia_carpet_gmres_solve" + std::to_string(rep);
    const auto t0 = std::chrono::high_resolution_clock::now();
    Profile::Tic(lbl.c_str(), &comm, true);
    solver(&sigma, BIO, rhs, gmres_tol, gmres_max_iter, false, &niter, &krylov);
    Profile::Toc();
    t_solve_sum += std::chrono::duration<double>(std::chrono::high_resolution_clock::now() - t0).count();
  }
  Profile::print(&comm, {"t_avg", "t_max", "f/s_avg", "f/s_max"});
  const double t_solve_avg = GlobalReduce(t_solve_sum, comm, CommOp::MAX) / n_rep;   // max over ranks, per repeat
  if (!comm.Rank())
    std::cout << std::setprecision(6) << "  flow: GMRES converged in " << niter << " iters (" << n_rep
              << " repeats, cleared Krylov each);  avg solve = " << t_solve_avg
              << " s,  avg per-iter = " << (niter ? t_solve_avg / niter : 0) << " s\n";

  // ---- Dump the SOLVED density sigma onto BOTH element lists for inspection (per-rank layout matches
  // combined_nodes: first Nb*KDIM = base collocation density, next Ns*KDIM = shaft). A qualitatively wrong
  // density -- sign flip / seam discontinuity at the collar<->shaft ring / spurious oscillation -- localizes
  // a FORMULATION bug (e.g. an inconsistent normal orientation between the two lists) that refining order/tol
  // cannot fix. WriteVTK's F is [Nnode*dof] AoS, same node order as GetNodeCoord. ----
  {
    constexpr Integer KD = 3;
    Vector<Real> sb(Nb*KD), ss(Ns*KD);
    for (Long i = 0; i < Nb*KD; i++) sb[i] = sigma[i];
    for (Long i = 0; i < Ns*KD; i++) ss[i] = sigma[Nb*KD + i];
    base.WriteVTK("vis/CiliaCarpet_density_base", sb, comm);
    shaft.WriteVTK("vis/CiliaCarpet_density_shaft", ss, comm);
    // Per-rank density magnitude summary (base vs shaft) -- a huge shaft |sigma| vs base is another tell.
    Real bmax = 0, smax = 0;
    for (Long i = 0; i < Nb; i++) { Real m = std::sqrt(sb[i*KD]*sb[i*KD]+sb[i*KD+1]*sb[i*KD+1]+sb[i*KD+2]*sb[i*KD+2]); bmax = std::max(bmax, m); }
    for (Long i = 0; i < Ns; i++) { Real m = std::sqrt(ss[i*KD]*ss[i*KD]+ss[i*KD+1]*ss[i*KD+1]+ss[i*KD+2]*ss[i*KD+2]); smax = std::max(smax, m); }
    bmax = GlobalReduce((double)bmax, comm, CommOp::MAX); smax = GlobalReduce((double)smax, comm, CommOp::MAX);
    if (!comm.Rank()) std::cout << std::setprecision(6) << "  wrote vis/CiliaCarpet_density_{base,shaft}.pvtu"
                                << "  (max|sigma| base=" << bmax << " shaft=" << smax << ")\n";
  }

  // Induced velocity u = BIO(sigma) - u_bg at arbitrary targets (both are XY-periodic).
  const Real zc = (Real)0.5 * (z_bottom + z_top);
  auto eval_induced = [&](const Vector<Real>& Xt) {
    const Long Nt = Xt.Dim() / COORD_DIM;               // #targets (COORD_DIM coords each)
    Op.SetTargetCoord(Xt);
    Vector<Real> U(Nt * KDIM); U = 0; BIO(&U, sigma);   // potential field: KDIM comps/target (Laplace 1, Stokes 3)
    if (KDIM == COORD_DIM) {                            // background flow is a VELOCITY (vector) field -- Stokes only
      Vector<Real> Ub = bg_flow_2peri(Xt); Ub *= (pressure_drop / L);
      U -= Ub;
    }
    return U;
  };

  // ===== Verification 1: periodicity across the x and y faces =====
  // Induced velocity at matching points on opposite faces (x=0 vs x~=L; y=0 vs y~=L), same other coords, in
  // the mid-gap. These are periodic IMAGES, so the difference should be at the solve/FMM tolerance.
  {
    const Long n = 6; const Real eL = L * (Real)0.9999999999;
    Vector<Real> Xa(3*n), Xb(3*n), Ya(3*n), Yb(3*n);
    for (Long i = 0; i < n; i++) { const Real t = (i + (Real)0.5) / n * L;
      Xa[i*3+0]=0;  Xa[i*3+1]=t; Xa[i*3+2]=zc;   Xb[i*3+0]=eL; Xb[i*3+1]=t; Xb[i*3+2]=zc;
      Ya[i*3+0]=t;  Ya[i*3+1]=0; Ya[i*3+2]=zc;   Yb[i*3+0]=t;  Yb[i*3+1]=eL; Yb[i*3+2]=zc; }
    Vector<Real> Ua=eval_induced(Xa), Ub=eval_induced(Xb), Va=eval_induced(Ya), Vb=eval_induced(Yb);
    Real dx=0, dy=0, umag=1e-30;
    for (Long i=0;i<3*n;i++){ dx=std::max(dx,std::fabs((double)(Ua[i]-Ub[i]))); dy=std::max(dy,std::fabs((double)(Va[i]-Vb[i])));
      umag=std::max(umag,std::max(std::fabs((double)Ua[i]),std::fabs((double)Va[i]))); }
    dx=GlobalReduce((double)dx,comm,CommOp::MAX); dy=GlobalReduce((double)dy,comm,CommOp::MAX); umag=GlobalReduce((double)umag,comm,CommOp::MAX);
    if(!comm.Rank()) std::cout << std::setprecision(4)
       << "  [verify-periodicity] max|u(x=0)-u(x=L)| = " << dx << " (rel " << dx/umag << "),  "
       << "max|u(y=0)-u(y=L)| = " << dy << " (rel " << dy/umag << ")   |u|~" << umag << "\n";
  }

  // ===== Verification 2: induced velocity at fixed probe points far from surfaces & edges =====
  // Mid-gap (z=zc, ~0.1 above the nearest finger tip), at cell-center columns (0.25 from the periodic
  // edges). Print full precision so a rerun at higher order/tol can be diffed for self-convergence.
  {
    const Real P[][3] = { {0.25*L,0.25*L,zc},{0.75*L,0.25*L,zc},{0.25*L,0.75*L,zc},
                          {0.75*L,0.75*L,zc},{0.40*L,0.60*L,zc},{0.60*L,0.40*L,zc} };
    const Long np = sizeof(P)/sizeof(P[0]);
    Vector<Real> Xp(3*np); for(Long i=0;i<np;i++) for(int c=0;c<3;c++) Xp[i*3+c]=P[i][c];
    Vector<Real> Up = eval_induced(Xp);
    if(!comm.Rank()){ std::cout << std::setprecision(12) << "  [verify-probe] induced u at mid-gap probe points (for self-convergence):\n";
      for(Long i=0;i<np;i++) std::cout << "    P"<<i<<" ("<<P[i][0]<<","<<P[i][1]<<","<<P[i][2]<<") u = "
        << Up[i*3] << ", " << Up[i*3+1] << ", " << Up[i*3+2] << "\n"; }
  }

  // ---- Volume-flow VTU (targets outside the slab / inside a cilium masked to 0) ----
  // Nvis<=1 means "no volume grid" (scaling/timing runs pass Nvis=1 to skip VTK I/O). CubeVolumeVisShifted
  // returns EMPTY coords for N<2 (its N0<2 early-out), and an empty off-surface target set makes BIO fall
  // back to the on-surface apply -> U sized Nsurf*KDIM while bg_flow is dim 0 -> `U -= Ub` dim-mismatch abort.
  // So bail out here rather than feed the degenerate grid through eval_induced.
  if (Nvis <= 1) {
    if (!comm.Rank()) std::cout << "  flow: Nvis<=1, skipping the volume-vis slice (timing/scaling run)\n";
    return;
  }
  // Geometry-accurate finger mask: a target is inside solid iff it lies within ~R_shaft of some cilium's
  // (tilted, wiggled) centerline. The tubes have radius R_shaft and each hemispherical cap lies within
  // R_shaft of its tip -- both bounded by the sampled centerline polyline (same points cilium_clearance
  // uses). This REPLACES the old cell-center vertical-cylinder mask, which was written for straight fingers
  // and was wrong for the tilted/wiggly defaults: it blanked empty fluid at the cell centers (spurious
  // zero-velocity holes) while leaving the real slanted tubes un-blanked (near-singular blowups).
  std::vector<Vec3<Real>> fpts;   // all cilia centerline samples (world coords)
  if (fingers) { std::vector<Vec3<Real>> tmp; for (const auto& c : curves) { cilium_samples<Real>(c, tmp, 128); for (const auto& p : tmp) fpts.push_back(p); } }
  // Mask radius (distance from the tube CENTERLINE). The layer-potential evaluation is inaccurate in a thin
  // near-field shell just OUTSIDE the surface -> velocity spikes at grid points adjacent to a shaft. The
  // buffer is tied to the grid resolution (not just R_shaft) so it covers that shell regardless of tube
  // size: mask_r = max(rfrac*R_shaft, R_shaft + QJ_VIS_MASK_CELLS * h) (default 0.5 cell). The h term matters
  // most when the tube is sub-grid (R_shaft < h). Also absorbs the centerline-polyline sampling gap.
  const Real hgrid = (Real)0.9 * L / (Real)(Nvis > 1 ? Nvis - 1 : 1);   // CubeVolumeVisShifted node spacing
  const char* mc_env = std::getenv("QJ_VIS_MASK_CELLS");
  const Real mask_cells = mc_env ? (Real)std::atof(mc_env) : (Real)0.5;
  // The near-field shell (where the layer-potential FMM eval is inaccurate) scales with the TUBE, not a fixed
  // grid-cell count, so scale the mask with R: mask_r = max(rfrac*R_shaft, R_shaft + mask_cells*h). rfrac~1.2
  // (env QJ_VIS_MASK_RFRAC) covers the shell (~R past the surface) while the h term keeps a sub-grid tube masked.
  // Defaults (1.2, 0.5) blank a tight ~R sleeve (was 2.0/1.0, a ~2R sleeve) -- show more of the near-tube field;
  // if near-tube velocity SPIKES reappear (inaccurate layer-potential eval just outside the surface), raise them.
  const char* rf_env = std::getenv("QJ_VIS_MASK_RFRAC");
  const Real mask_rfrac = rf_env ? (Real)std::atof(rf_env) : (Real)1.2;
  const Real mask_r  = std::max(mask_rfrac * R_shaft, R_shaft + mask_cells * hgrid);
  const Real mask_r2 = mask_r * mask_r;
  // Near-wall shell: the fat cilium FEET (collar+fillet+cap on the base, not on the slender centerline) and
  // the wall panels leak the same near-field shell just inside the slab. Blank a layer of thickness mask_r
  // above the bottom wall / below the top wall. Scales with R, so the thin tube (whose vis grid never samples
  // this close) is unaffected; the true velocity there is ~0 by no-slip anyway. Env QJ_VIS_WALL_MARGIN.
  const char* wm_env = std::getenv("QJ_VIS_WALL_MARGIN");
  const Real wall_margin = wm_env ? (Real)std::atof(wm_env) : mask_r;
  if (!comm.Rank()) std::cout << std::setprecision(4) << "  flow: vis finger mask radius = " << mask_r
                              << " (=" << (mask_r/R_shaft) << "*R_shaft; R=" << R_shaft << ", h=" << hgrid
                              << ")   near-wall margin = " << wall_margin << "\n";

  CubeVolumeVisShifted<Real> vv(Nvis, (Real)0.9, comm);
  Vector<Real> Xv = vv.GetCoord();
  const Long Ntrg = Xv.Dim() / COORD_DIM;
  if (!comm.Rank()) std::cout << "  flow: volume grid " << Nvis << "^3, " << (Nvis*Nvis*Nvis) << " targets\n";
  Vector<Real> U = eval_induced(Xv);                                 // field: Ntrg*KDIM (KDIM comps/target)
  Vector<Real> Ubg = bg_flow_2peri(Xv); Ubg *= (pressure_drop / L);  // background VELOCITY (used only for KDIM==3)
  Real umax = 0, uxsum = 0, uysum = 0, uzsum = 0; Long nmid = 0;
  for (Long i = 0; i < Ntrg; i++) {
    const Real x = Xv[i*COORD_DIM], y = Xv[i*COORD_DIM+1], z = Xv[i*COORD_DIM+2];
    bool masked = (z < z_bottom + wall_margin || z > z_top - wall_margin);
    if (!masked) for (const auto& p : fpts) {   // inside any cilium tube/cap?
      const Real dx = x-p[0], dy = y-p[1], dz = z-p[2];
      if (dx*dx + dy*dy + dz*dz < mask_r2) { masked = true; break; }
    }
    if (masked) for (Integer c = 0; c < KDIM; c++) U[i*KDIM+c] = 0;   // blank all KDIM field components
    Real u2 = 0; for (Integer c = 0; c < KDIM; c++) u2 += U[i*KDIM+c]*U[i*KDIM+c];
    umax = std::max(umax, std::sqrt(u2));
    // mid-gap mean TOTAL velocity (induced + background) -- a Stokes-velocity (KDIM==3) diagnostic only.
    if (KDIM == 3 && std::fabs((double)(z - zc)) < 0.03 && !masked) {
      uxsum += U[i*KDIM]+Ubg[i*3]; uysum += U[i*KDIM+1]+Ubg[i*3+1]; uzsum += U[i*KDIM+2]+Ubg[i*3+2]; nmid++;
    }
  }
  umax = GlobalReduce((double)umax, comm, CommOp::MAX);
  uxsum = GlobalReduce((double)uxsum, comm, CommOp::SUM); uysum = GlobalReduce((double)uysum, comm, CommOp::SUM);
  uzsum = GlobalReduce((double)uzsum, comm, CommOp::SUM); nmid = GlobalReduce((Long)nmid, comm, CommOp::SUM);
  if (!comm.Rank())
    std::cout << std::setprecision(6) << "  flow: max|U_induced| = " << umax
              << "   mid-gap mean total u = (" << (nmid?uxsum/nmid:0) << ", " << (nmid?uysum/nmid:0) << ", " << (nmid?uzsum/nmid:0) << ")\n";
  vv.WriteVTK("vis/CiliaCarpet_U", U);
  if (!comm.Rank()) std::cout << "  wrote vis/CiliaCarpet_U.pvtu\n";
}

} // namespace

int main(int argc, char** argv) {
  Comm::MPI_Init(&argc, &argv);
  using Real = double;
  {
    Comm comm = Comm::World();

    // CLI: [Npatch order tol Naz bot_tip top_tip tilt_deg pdrop Nvis fingers fourier cheb n_axial]
    // The slender-shaft carpet: cilia REACH the midplane z=0.5 (both planes), tilted +-x, with a random
    // per-cilium sine wiggle + collision resolution (env QJ_CILIA_SEED sets the RNG seed).
    const Integer Npatch   = (argc > 1)  ? std::atoi(argv[1])  : 4;   // >=3 required (see guard below)
    // GUARD: require at least 3x3 patches per wall. At 2x2 the same-column top/bottom cilia both reaching
    // z=0.5 cannot be reliably separated within the box by the tilt-adjust; >=3x3 gives the cap tilt-adjust +
    // bounded wiggle enough room to keep them disjoint.
    SCTL_ASSERT_MSG(Npatch >= 3, "cilia carpet requires Npatch >= 3 (2x2 cilia cannot be separated at the midplane)");
    const Integer order    = (argc > 2)  ? std::atoi(argv[2])  : 12;    // QuadElemList base order (multiple of 4)
    const Real    tol      = (argc > 3)  ? std::atof(argv[3])  : 1e-8;
    const Integer Naz      = (argc > 4)  ? std::atoi(argv[4])  : 8;
    // PATCH-RELATIVE shaft (LOCKED): R_shaft = 0.25*S (S = L/(2*Npatch)) -- the thin cilium; r_fil = 0.1*R_shaft.
    // Scale-invariant: the shaft radius tracks the patch pitch, so the carpet looks the same at every Npatch and
    // its solid volume fraction is Npatch-independent. H_reach stays box-driven (bot_tip/top_tip). See
    // cilium_scale_from_patch() in stud_sphere_geom.hpp.
    const Real    L_box    = 1.0, S_patch = L_box / (Real)(2 * Npatch);
    const Real    R_shaft  = (Real)0.25 * S_patch;
    const Real    r_fil    = (Real)0.1 * R_shaft;
    const Real    bot_tip  = (argc > 5)  ? std::atof(argv[5])  : 0.50;  // bottom cilia reach the midplane z=0.5
    const Real    top_tip  = (argc > 6)  ? std::atof(argv[6])  : 0.50;  // top cilia reach the midplane z=0.5
    const Real    tilt_deg = (argc > 7)  ? std::atof(argv[7])  : 10.0;  // nominal tilt (reduced per-column to fit the box).
                                                                        // 10deg is the largest non-overlapping tilt at the dense
                                                                        // 8x8 pitch (lateral swing < pitch); larger tilts overlap
                                                                        // same-plane neighbours. Eases as Npatch drops (wider pitch).
    const Real    pdrop    = (argc > 8)  ? std::atof(argv[8])  : -1.0;
    const Long    Nvis     = (argc > 9)  ? std::atol(argv[9])  : 60;
    const bool    fingers  = (argc > 10) ? (std::atoi(argv[10]) != 0) : true;  // volume-vis finger mask toggle
    const Long    fourier  = (argc > 11) ? std::atol(argv[11]) : 36;    // slender azimuthal Fourier order
    const Long    cheb     = (argc > 12) ? std::atol(argv[12]) : 10;    // slender Chebyshev order (CSBQ tables)
    const Integer n_axial_in = (argc > 13) ? std::atoi(argv[13]) : -1;  // slender axial panels/fiber (-1 = auto)

    const Real L = L_box, z_bottom = 0.01, z_top = 0.99, S = S_patch;
    const Real core_frac = 0.40, grade_exp = 1.0;
    const Real tilt_rad = tilt_deg * const_pi<Real>() / 180;
    // Straight base panels, then POU-transition panels. Each is n*az long with az = 2*pi*R_shaft/Naz.
    // Env-tunable QJ_N_STRAIGHT / QJ_N_TRANS (default 3).
    const Integer n_straight = std::getenv("QJ_N_STRAIGHT") ? std::atoi(std::getenv("QJ_N_STRAIGHT")) : 3;
    const Integer n_trans    = std::getenv("QJ_N_TRANS")    ? std::atoi(std::getenv("QJ_N_TRANS"))    : 3;
    const char* seed_env = std::getenv("QJ_CILIA_SEED");
    const uint64_t seed = seed_env ? (uint64_t)std::strtoull(seed_env, nullptr, 10) : (uint64_t)12345;
    // x&y box-containment buffer: keep every cilium tube+cap at least this far from the periodic cell edges.
    // Raised from 0.01 to 0.1 so edge-column cilia (and their near-field / periodic-image interactions) sit
    // well inside the cell -- cleaner flow near the x/y boundaries. Env-tunable (QJ_BOX_BUFFER). NOTE: a larger
    // buffer forces the tilt-reduction step to straighten edge columns more; at high Npatch (cells near the
    // edge) 0.1 can drive the outer column toward 0 deg tilt -- watch the geom report's tilt-cut/edge counts.
    const char* bb_env = std::getenv("QJ_BOX_BUFFER");
    const Real box_buffer = bb_env ? (Real)std::atof(bb_env) : (Real)0.1;

    // Shared curve set (ONE source of truth: base butterfly-cap placement + slender fibers -> conforming
    // seams). generate_cilia_carpet applies per-column tilt reduction (box-fit, box_buffer), a random sine
    // wiggle per cilium, and collision resolution (regenerate wiggle up to 3x, then shorten). Deterministic
    // in `seed` (identical on every rank => consistent replicated base + partitioned shaft).
    std::vector<CiliumCurveFlat<Real>> curves = generate_cilia_carpet<Real>(
        Npatch, z_bottom, z_top, L, R_shaft, bot_tip, top_tip, r_fil, Naz, tilt_rad, n_straight, n_trans, seed, comm, box_buffer);
    // Default axial panels = the curve's own Ns (Naz-based az). Uniform-in-arc slender panels then land on
    // the tilt-transition POU window edges (M8 "windows align to panels"), and Ns scales as 1/R_shaft.
    const Integer n_axial = (n_axial_in >= 1) ? n_axial_in : curves[0].Ns;
    // Near-singular quadrature scheme (base QuadElemList) tied to tol via the canonical map.
    Integer Nbeta, max_depth; quad_scheme_for_tol<Real>(tol, Nbeta, max_depth);

    if (!comm.Rank()) {
      std::cout << "cilia-carpet HYBRID Npatch=" << Npatch << " (cilia=" << 2*Npatch*Npatch << ")"
                << " order=" << order << " tol=" << tol << " (Nbeta=" << Nbeta << " max_depth=" << max_depth << ")"
                << " Naz=" << Naz << " R_shaft=" << R_shaft
                << " r_fil=" << r_fil << " tips(z)=" << bot_tip << "/" << top_tip << " tilt=" << tilt_deg << "deg\n"
                << "  slender shaft: fourier=" << fourier << " cheb=" << cheb << " n_axial/fiber=" << n_axial
                << " (Sarc=" << curves[0].Sarc << ")" << std::endl;
    }

    // Hybrid geometry: QuadElemList base (walls + collar + fillet + cap, NO shaft) + one SlenderElemList of
    // bent tube shafts. Base kept OUTWARD-from-solid (invert=false): the carpet cilium is a PROTRUDING tube
    // (fluid outside), so its outward normal is radially-outward-from-axis = CSBQ's native slender normal --
    // consistent without inverting, so run_flow's -1/2 DL jump / NormalOrient=+1 are unchanged.
    QuadElemList<Real> base = BuildCiliaCarpetHybridBase<Real>(order, curves, R_shaft, r_fil, Naz, S,
        core_frac, grade_exp, n_straight, n_trans, comm);
    SlenderElemList<Real> shaft = BuildCiliaCarpetHybridShafts<Real>(curves, n_axial, R_shaft, cheb, fourier, comm);
    base.SetQuadScheme(QuadElemList<Real>::QuadScheme::Hybrid, 10, Nbeta, max_depth);
    report_area<Real>(base, comm);
    base.WriteVTK("vis/CiliaCarpet_geom", Vector<Real>(), comm);
    shaft.WriteVTK("vis/CiliaCarpet_shaft", Vector<Real>(), comm);
    {
      const Long nbp = GlobalReduce((Long)base.Size(),  comm, CommOp::SUM);
      const Long nsp = GlobalReduce((Long)shaft.Size(), comm, CommOp::SUM);
      Vector<Real> Xb, Xnb, Xs, Xns; base.GetNodeCoord(&Xb,&Xnb,nullptr); shaft.GetNodeCoord(&Xs,&Xns,nullptr);
      const Long nbn = GlobalReduce((Long)(Xb.Dim()/3), comm, CommOp::SUM), nsn = GlobalReduce((Long)(Xs.Dim()/3), comm, CommOp::SUM);
      if (!comm.Rank())
        std::cout << "  wrote vis/CiliaCarpet_{geom(base),shaft}.pvtu/.vtu\n"
                  << "  base panels=" << nbp << " (" << nbn << " nodes)   shaft slender-panels=" << nsp
                  << " (" << nsn << " nodes)   TOTAL nodes=" << (nbn+nsn) << "\n";
      Long pci=-1, pcj=-1; const Real gap = cilia_min_clearance<Real>(curves, pci, pcj);
      if (!comm.Rank()) std::cout << std::setprecision(6) << "  [overlap] min cilium surface clearance = " << gap
                                  << " (>0 => no overlap)  closest pair (" << pci << "," << pcj << ")\n";
    }

    // QJ_GEOM_ONLY=1: stop after the geometry build + overlap/watertight/containment report and the
    // geom/shaft VTK dump (no BIE solve). Lets the (cheap, serial-ish) geometry be validated -- clearance,
    // box/z containment, node counts -- without paying for the multi-million-node periodic Stokes solve.
    if (std::getenv("QJ_GEOM_ONLY") && std::atoi(std::getenv("QJ_GEOM_ONLY")) != 0) {
      if (!comm.Rank()) std::cout << "  QJ_GEOM_ONLY set -- geometry only, skipping the flow solve.\n";
    } else {
      run_flow<Real>(base, shaft, comm, tol, L, pdrop, Nvis, fingers, z_bottom, z_top, R_shaft, curves);
    }
  }
  Comm::MPI_Finalize();
  return 0;
}
