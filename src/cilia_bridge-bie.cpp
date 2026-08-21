/**
 * WALL-TO-WALL cilia "bridge" carpet driver: doubly-periodic (XY) Stokes flow past an Npatch x Npatch grid
 * of cilia, each BRIDGING the bottom plate to the top plate (collar+fillet in the bottom wall, one slender
 * wall-to-wall shaft, collar+fillet in the top wall -- NO caps, NO free tips, NO tilt). Each shaft carries a
 * random single-period sine wiggle bounded to stay inside its own patch cell, so neighbouring cilia can
 * never collide (a collision verifier prints `COLLISION TRIGGERED` if they ever do).
 *
 * Geometry: plane_cilia_bridge_geom.hpp -- a QuadElemList base (both walls' collars+fillets) + one
 * SlenderElemList of wall-to-wall tubes, both into ONE StokesBIO. The flow solve, verification and volume
 * visualization are ported from src/cilia_carpet-bie.cpp (run_flow), adapted to BridgeCurve.
 *
 * Periodicity requires PVFMM (make PVFMM=1). EvalPVFMM falls back to direct summation below 40000 global
 * targets, so periodic runs must exceed that. Run from the repo root (so ./data/special_quad_q10_* resolve)
 * with PVFMM_DIR pointing at the precomputed-operator dir:
 *   PVFMM_DIR=extern/pvfmm OMP_NUM_THREADS=8 mpirun -n 2 ./bin/cilia_bridge-bie ...
 *
 *   ./bin/cilia_bridge-bie [Npatch order tol Naz pdrop Nvis fingers fourier cheb n_axial z_plate]
 *
 * The cilium shaft radius is LOCKED to R_shaft = 0.25*S (S = L/(2*Npatch)); r_fil = 0.1*R_shaft.
 * Env: QJ_GEOM_ONLY=1 stops after geometry+VTK (no solve). QJ_CILIA_SEED, QJ_CILIA_WIGGLE, QJ_WIG_AMP_FRAC,
 *      QJ_BOX_BUFFER, QJ_N_STRAIGHT/QJ_N_TRANS, QJ_VIS_MASK_*, QJ_VIS_WALL_MARGIN.
 */
#include <csbq.hpp>
#include <stokes_bio.hpp>
#include <quad_junctions/fmm_kernels.hpp>
#include <quad_junctions/quad_scheme.hpp>     // QJDefaultScheme (Duffy default, SCTL_SELF_SCHEME=hybrid opt-out)
#include <quad_junctions/plane_cilia_geom.hpp>
#include <quad_junctions/plane_cilia_bridge_geom.hpp>
#include <quad_junctions/periodic_flow_utils.hpp>
#include <sctl/experimental/bench_quad.hpp>   // per-target near-setup breakdown (BENCH=1); no-op otherwise
#include <chrono>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <string>

using namespace sctl;
using namespace quad_junctions;

namespace {

// Canonical tol -> (Nbeta, max_depth) near-singular quadrature map (matches cilia_carpet-bie.cpp).
template <class Real> void quad_scheme_for_tol(Real tol, Integer& Nbeta, Integer& max_depth) {
  const Real    tolL[4] = {(Real)1e-5, (Real)1e-7, (Real)1e-9, (Real)1e-11};
  const Integer NbL[4]  = {48, 100, 200, 400};
  const Integer mdL[4]  = {4, 8, 12, 30};
  int idx = 3;
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

// Combined surface mean of a combined-ordered density sigma (KDIM per node): each list maps its collocation
// density to the far-field nodes (GetFarFieldDensity) and integrates with the far-field weights. Correct for
// BOTH the QuadElemList base (no-op map -> empty Fff -> read sig_lst directly) and the SlenderElemList shaft
// (real interpolation). Copied verbatim from cilia_carpet-bie.cpp.
template <class Real> void combined_surface_mean(const QuadElemList<Real>& base, const SlenderElemList<Real>& shaft,
    const Vector<Real>& sigma, const Long Nb, const Long Ns, const Integer KDIM, const Real tol,
    const Comm& comm, Vector<Real>& mean_out, Real& total_area) {
  Vector<Real> sm(KDIM); sm = 0; Real area = 0;
  auto accum = [&](const auto& lst, const Vector<Real>& sig_lst) {
    Vector<Real> Xff, Xnff, wts, dist, Fff; Vector<Long> cnt;
    lst.GetFarFieldNodes(Xff, Xnff, wts, dist, cnt, tol);
    lst.GetFarFieldDensity(Fff, sig_lst);       // collocation -> far-field density
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
// projection (rank-deficiency fix). NormalOrient=-1 (all normals out-of-solid, matching the slender shaft).
// Ported from cilia_carpet-bie.cpp::run_flow; the only geometry-specific change is the finger mask, which
// samples the BridgeCurve centerline polyline (no cap/tilt -- polyline +- R_shaft bounds the tube exactly).
template <class Real>
void run_flow(const QuadElemList<Real>& base, const SlenderElemList<Real>& shaft, const Comm& comm,
              const Real tol, const Real L,
              const Real pressure_drop, const Long Nvis, const bool fingers,
              const Real z_bottom, const Real z_top, const Real R_shaft,
              const std::vector<BridgeCurve<Real>>& curves) {
  const Real SL_scal = 1.0, DL_scal = 1.0;
  const Real gmres_tol = std::max(tol, (Real)1e-8);
  const Long gmres_max_iter = 400;
  constexpr Integer COORD_DIM = 3;
  constexpr Integer KDIM = 3;        // Stokes velocity components/target

  Vector<Real> X0, Xn0; Long Nb = 0, Ns = 0;
  combined_nodes(base, shaft, X0, Xn0, Nb, Ns);
  const Long Nnode = Nb + Ns;

  StokesBIO<Real> Op(SL_scal, DL_scal, comm);
  Op.SetAccuracy(tol);
  Op.AddElemList(base,  "0_base");
  Op.AddElemList(shaft, "1_shaft");
  Op.SetPeriodicity(sctl::Periodicity::XY, L);
  Op.SetTargetCoord(X0);

  // All normals point OUT-OF-SOLID / into the fluid (base flipped to match the un-flippable slender normal).
  // Fluid on the +n side => fluid-side DL limit is D_pv + 1/2 sigma (a +I/2 jump) => NormalOrient=-1
  // (the jump term below is `U -= 0.5*sigma*NormalOrient`).
  Vector<Real> NormalOrient(3 * Nnode); NormalOrient = -1.0;

  const auto BIO = [&](Vector<Real>* U, const Vector<Real>& sigma) {
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

  // ===== separately-timed SETUP phase + GMRES solve (see cilia_carpet-bie.cpp for the rationale). =====
  Profile::Enable(true);
  { Vector<Real> sigma_warm;                       // warm-up: page in PVFMM/CSBQ tables (not timed)
    solver(&sigma_warm, BIO, rhs, (Real)1e-2, gmres_max_iter); }

  Profile::reset(); Op.ClearSetup();
#ifdef BENCH_QUAD
  sctl::bench::Reset();
  const auto t_setup0 = std::chrono::high_resolution_clock::now();
#endif
  Profile::Tic("cilia_bridge_setup", &comm, true);
  Op.Setup();
  Profile::Toc();
  Profile::print(&comm, {"t_avg", "t_max"});
#ifdef BENCH_QUAD
  const double setup_wall = std::chrono::duration<double>(std::chrono::high_resolution_clock::now() - t_setup0).count();
  sctl::bench::ReportMPI(
      [&](double* buf, int m) {
        Vector<double> v(m); for (int i = 0; i < m; i++) v[i] = buf[i];
        GlobalReduce(v, comm, CommOp::SUM);
        for (int i = 0; i < m; i++) buf[i] = v[i];
      },
      !comm.Rank(), "cilia_bridge_near_setup", setup_wall);
#endif
  Profile::reset();

  const Integer n_rep = 1;
  double t_solve_sum = 0;
  for (Integer rep = 0; rep < n_rep; rep++) {
    sigma.ReInit(0);
    const std::string lbl = "cilia_bridge_gmres_solve" + std::to_string(rep);
    const auto t0 = std::chrono::high_resolution_clock::now();
    Profile::Tic(lbl.c_str(), &comm, true);
    solver(&sigma, BIO, rhs, gmres_tol, gmres_max_iter, false, &niter);
    Profile::Toc();
    t_solve_sum += std::chrono::duration<double>(std::chrono::high_resolution_clock::now() - t0).count();
  }
  Profile::print(&comm, {"t_avg", "t_max", "f/s_avg", "f/s_max"});
  const double t_solve_avg = GlobalReduce(t_solve_sum, comm, CommOp::MAX) / n_rep;
  if (!comm.Rank())
    std::cout << std::setprecision(6) << "  flow: GMRES converged in " << niter << " iters (" << n_rep
              << " repeats, );  avg solve = " << t_solve_avg
              << " s,  avg per-iter = " << (niter ? t_solve_avg / niter : 0) << " s\n";

  // ---- Dump the SOLVED density onto BOTH element lists (base collocation density then shaft). ----
  {
    constexpr Integer KD = 3;
    Vector<Real> sb(Nb*KD), ss(Ns*KD);
    for (Long i = 0; i < Nb*KD; i++) sb[i] = sigma[i];
    for (Long i = 0; i < Ns*KD; i++) ss[i] = sigma[Nb*KD + i];
    base.WriteVTK("vis/CiliaBridge_density_base", sb, comm);
    shaft.WriteVTK("vis/CiliaBridge_density_shaft", ss, comm);
    Real bmax = 0, smax = 0;
    for (Long i = 0; i < Nb; i++) { Real m = std::sqrt(sb[i*KD]*sb[i*KD]+sb[i*KD+1]*sb[i*KD+1]+sb[i*KD+2]*sb[i*KD+2]); bmax = std::max(bmax, m); }
    for (Long i = 0; i < Ns; i++) { Real m = std::sqrt(ss[i*KD]*ss[i*KD]+ss[i*KD+1]*ss[i*KD+1]+ss[i*KD+2]*ss[i*KD+2]); smax = std::max(smax, m); }
    bmax = GlobalReduce((double)bmax, comm, CommOp::MAX); smax = GlobalReduce((double)smax, comm, CommOp::MAX);
    if (!comm.Rank()) std::cout << std::setprecision(6) << "  wrote vis/CiliaBridge_density_{base,shaft}.pvtu"
                                << "  (max|sigma| base=" << bmax << " shaft=" << smax << ")\n";
  }

  // Induced velocity u = BIO(sigma) - u_bg at arbitrary targets (both XY-periodic).
  const Real zc = (Real)0.5 * (z_bottom + z_top);
  auto eval_induced = [&](const Vector<Real>& Xt) {
    const Long Nt = Xt.Dim() / COORD_DIM;
    Op.SetTargetCoord(Xt);
    Vector<Real> U(Nt * KDIM); U = 0; BIO(&U, sigma);
    if (KDIM == COORD_DIM) {
      Vector<Real> Ub = bg_flow_2peri(Xt); Ub *= (pressure_drop / L);
      U -= Ub;
    }
    return U;
  };

  // ===== Verification 1: periodicity across the x and y faces =====
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
  if (Nvis <= 1) {
    if (!comm.Rank()) std::cout << "  flow: Nvis<=1, skipping the volume-vis slice (timing/scaling run)\n";
    return;
  }
  // Geometry-accurate finger mask: a target is inside solid iff within ~R_shaft of some cilium's (wiggled)
  // centerline. The tubes have radius R_shaft; the wall-to-wall polyline (bridge_samples) bounds them exactly.
  std::vector<Vec3<Real>> fpts;
  if (fingers) { std::vector<Vec3<Real>> tmp; for (const auto& c : curves) { bridge_samples<Real>(c, tmp, 128); for (const auto& p : tmp) fpts.push_back(p); } }
  const Real hgrid = (Real)0.9 * L / (Real)(Nvis > 1 ? Nvis - 1 : 1);
  const char* mc_env = std::getenv("QJ_VIS_MASK_CELLS");
  const Real mask_cells = mc_env ? (Real)std::atof(mc_env) : (Real)0.5;
  const char* rf_env = std::getenv("QJ_VIS_MASK_RFRAC");
  const Real mask_rfrac = rf_env ? (Real)std::atof(rf_env) : (Real)1.2;
  const Real mask_r  = std::max(mask_rfrac * R_shaft, R_shaft + mask_cells * hgrid);
  const Real mask_r2 = mask_r * mask_r;
  const char* wm_env = std::getenv("QJ_VIS_WALL_MARGIN");
  const Real wall_margin = wm_env ? (Real)std::atof(wm_env) : mask_r;
  if (!comm.Rank()) std::cout << std::setprecision(4) << "  flow: vis finger mask radius = " << mask_r
                              << " (=" << (mask_r/R_shaft) << "*R_shaft; R=" << R_shaft << ", h=" << hgrid
                              << ")   near-wall margin = " << wall_margin << "\n";

  CubeVolumeVisShifted<Real> vv(Nvis, (Real)0.9, comm);
  Vector<Real> Xv = vv.GetCoord();
  const Long Ntrg = Xv.Dim() / COORD_DIM;
  if (!comm.Rank()) std::cout << "  flow: volume grid " << Nvis << "^3, " << (Nvis*Nvis*Nvis) << " targets\n";
  Vector<Real> U = eval_induced(Xv);
  Vector<Real> Ubg = bg_flow_2peri(Xv); Ubg *= (pressure_drop / L);
  Real umax = 0, uxsum = 0, uysum = 0, uzsum = 0; Long nmid = 0;
  for (Long i = 0; i < Ntrg; i++) {
    const Real x = Xv[i*COORD_DIM], y = Xv[i*COORD_DIM+1], z = Xv[i*COORD_DIM+2];
    bool masked = (z < z_bottom + wall_margin || z > z_top - wall_margin);
    if (!masked) for (const auto& p : fpts) {
      const Real dx = x-p[0], dy = y-p[1], dz = z-p[2];
      if (dx*dx + dy*dy + dz*dz < mask_r2) { masked = true; break; }
    }
    if (masked) for (Integer c = 0; c < KDIM; c++) U[i*KDIM+c] = 0;
    Real u2 = 0; for (Integer c = 0; c < KDIM; c++) u2 += U[i*KDIM+c]*U[i*KDIM+c];
    umax = std::max(umax, std::sqrt(u2));
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
  vv.WriteVTK("vis/CiliaBridge_U", U);
  if (!comm.Rank()) std::cout << "  wrote vis/CiliaBridge_U.pvtu\n";
}

} // namespace

int main(int argc, char** argv) {
  Comm::MPI_Init(&argc, &argv);
  using Real = double;
  {
    Comm comm = Comm::World();

    // CLI: [Npatch order tol Naz pdrop Nvis fingers fourier cheb n_axial z_plate]
    const Integer Npatch   = (argc > 1)  ? std::atoi(argv[1])  : 8;
    SCTL_ASSERT_MSG(Npatch >= 1, "cilia bridge requires Npatch >= 1");
    const Integer order    = (argc > 2)  ? std::atoi(argv[2])  : 12;    // QuadElemList base order (multiple of 4)
    const Real    tol      = (argc > 3)  ? std::atof(argv[3])  : 1e-7;
    const Integer Naz      = (argc > 4)  ? std::atoi(argv[4])  : 8;
    const Real    pdrop    = (argc > 5)  ? std::atof(argv[5])  : -1.0;
    const Long    Nvis     = (argc > 6)  ? std::atol(argv[6])  : 60;
    const bool    fingers  = (argc > 7)  ? (std::atoi(argv[7]) != 0) : true;
    const Long    fourier  = (argc > 8)  ? std::atol(argv[8])  : 24;    // slender azimuthal Fourier order
    const Long    cheb     = (argc > 9)  ? std::atol(argv[9])  : 10;    // slender Chebyshev order (only q10 precomputed)
    const Integer n_axial_in = (argc > 10) ? std::atoi(argv[10]) : -1;  // slender axial panels/fiber (-1 = auto)
    const Real    z_plate  = (argc > 11) ? std::atof(argv[11]) : 0.01;  // z_bottom=z_plate, z_top=L-z_plate
    SCTL_ASSERT_MSG(z_plate > 0 && z_plate < 0.5, "z_plate must be in (0, 0.5)");

    const Real L = 1.0, S = L / (Real)(2 * Npatch);
    const Real R_shaft = (Real)0.25 * S, r_fil = (Real)0.1 * R_shaft;   // LOCKED patch-relative (scale-invariant)
    const Real z_bottom = z_plate, z_top = L - z_plate;
    const Real core_frac = 0.40, grade_exp = 1.0;
    const Integer n_straight = std::getenv("QJ_N_STRAIGHT") ? std::atoi(std::getenv("QJ_N_STRAIGHT")) : 3;
    const Integer n_trans    = std::getenv("QJ_N_TRANS")    ? std::atoi(std::getenv("QJ_N_TRANS"))    : 3;
    const char* seed_env = std::getenv("QJ_CILIA_SEED");
    const uint64_t seed = seed_env ? (uint64_t)std::strtoull(seed_env, nullptr, 10) : (uint64_t)12345;
    const char* bb_env = std::getenv("QJ_BOX_BUFFER");
    const Real box_buffer = bb_env ? (Real)std::atof(bb_env) : (Real)0.02;
    const char* wig_env = std::getenv("QJ_CILIA_WIGGLE");
    const bool cilia_wiggle = !wig_env || std::atoi(wig_env) != 0;

    // Wall-to-wall cilia: one BridgeCurve per cell with a confined random wiggle + collision verifier.
    Profile::Enable(true);
    Profile::Tic("cilia_geometry_build", &comm, true);
    std::vector<BridgeCurve<Real>> curves = generate_cilia_bridge<Real>(
        Npatch, z_bottom, z_top, L, R_shaft, r_fil, seed, comm, box_buffer, (Real)0.005, cilia_wiggle);
    const Integer n_axial = (n_axial_in >= 1) ? n_axial_in : curves[0].auto_n_axial(fourier);
    Integer Nbeta, max_depth; quad_scheme_for_tol<Real>(tol, Nbeta, max_depth);

    if (!comm.Rank())
      std::cout << "cilia-bridge Npatch=" << Npatch << " (cilia=" << Npatch*Npatch << ")"
                << " order=" << order << " tol=" << tol << " (Nbeta=" << Nbeta << " max_depth=" << max_depth << ")"
                << " Naz=" << Naz << " R_shaft=" << R_shaft << " r_fil=" << r_fil
                << " plates(z)=" << z_bottom << "/" << z_top << " span=" << curves[0].span << "\n"
                << "  slender shaft: fourier=" << fourier << " cheb=" << cheb << " n_axial/fiber=" << n_axial << std::endl;

    QuadElemList<Real> base = BuildCiliaBridgeBase<Real>(order, curves, R_shaft, r_fil, Naz, S,
        z_bottom, z_top, core_frac, grade_exp, n_straight, n_trans, comm);
    SlenderElemList<Real> shaft = BuildCiliaBridgeShafts<Real>(curves, n_axial, R_shaft, cheb, fourier, comm);
    Profile::Toc();                                    // cilia_geometry_build
    Profile::print(&comm, {"t_avg", "t_max"});

    base.SetQuadScheme(quad_junctions::QJDefaultScheme<Real>(), 10, Nbeta, max_depth);
    report_area<Real>(base, comm);
    base.WriteVTK("vis/CiliaBridge_geom", Vector<Real>(), comm);
    shaft.WriteVTK("vis/CiliaBridge_shaft", Vector<Real>(), comm);
    {
      const Long nbp = GlobalReduce((Long)base.Size(),  comm, CommOp::SUM);
      const Long nsp = GlobalReduce((Long)shaft.Size(), comm, CommOp::SUM);
      Vector<Real> Xb, Xnb, Xs, Xns; base.GetNodeCoord(&Xb,&Xnb,nullptr); shaft.GetNodeCoord(&Xs,&Xns,nullptr);
      const Long nbn = GlobalReduce((Long)(Xb.Dim()/3), comm, CommOp::SUM), nsn = GlobalReduce((Long)(Xs.Dim()/3), comm, CommOp::SUM);
      if (!comm.Rank())
        std::cout << "  wrote vis/CiliaBridge_{geom(base),shaft}.pvtu/.vtu\n"
                  << "  base panels=" << nbp << " (" << nbn << " nodes)   shaft slender-panels=" << nsp
                  << " (" << nsn << " nodes)   TOTAL nodes=" << (nbn+nsn) << "\n";
      Long pci=-1, pcj=-1; const Real gap = cilia_bridge_min_clearance<Real>(curves, pci, pcj);
      if (!comm.Rank()) std::cout << std::setprecision(6) << "  [overlap] min cilium surface clearance = " << gap
                                  << " (>0 => no overlap)  closest pair (" << pci << "," << pcj << ")\n";
    }

    // QJ_GEOM_ONLY=1: geometry + report + VTK only (no BIE solve).
    if (std::getenv("QJ_GEOM_ONLY") && std::atoi(std::getenv("QJ_GEOM_ONLY")) != 0) {
      if (!comm.Rank()) std::cout << "  QJ_GEOM_ONLY set -- geometry only, skipping the flow solve.\n";
    } else {
      run_flow<Real>(base, shaft, comm, tol, L, pdrop, Nvis, fingers, z_bottom, z_top, R_shaft, curves);
    }
  }
  Comm::MPI_Finalize();
  return 0;
}
