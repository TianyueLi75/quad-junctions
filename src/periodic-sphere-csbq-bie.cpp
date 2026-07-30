/**
 * Doubly-periodic (XY) Stokes: rigid sphere between two no-slip plates -- CSBQ-SLENDER-SPHERE variant.
 *
 * Same physical problem as periodic-sphere-bie.cpp (pressure-driven plane-Poiseuille flow past a rigid
 * no-slip sphere held between two no-slip plates in a unit XY-periodic cell), but the sphere obstacle is
 * built as a CSBQ SlenderElemList instead of a cubed-sphere QuadElemList. The two flat plates stay a
 * QuadElemList. This deliberately exercises a QuadElemList + SlenderElemList mix in the otherwise-working
 * periodic test, to see whether that mix is what breaks the hybrid drivers.
 *
 * Sphere discretization (per the stokes-periodize-numtest sphere_geom pattern): centerline along x through
 * the cell center, x = c + R cos(theta), radius eps = R sin(theta), theta in [0,pi] over Nelem panels;
 * ElemOrder(Chebyshev)=10, FourierOrder=24. The slender normal is radially outward = INTO the fluid, so the
 * sphere DOFs get NormalOrient = -1; the plates keep the out-of-fluid convention (+1). Combined-field solve
 * SL_scal=1, DL_scal=1 (jump handled per-node via NormalOrient), exactly as planes_with_loops.
 *
 *   make PVFMM=1 bin/periodic-sphere-csbq-bie
 *   PVFMM_DIR=extern/pvfmm OMP_NUM_THREADS=8 ./bin/periodic-sphere-csbq-bie [order R tol pdrop Nvis Nelem fourier]
 */
#include <csbq.hpp>
#include <stokes_bio.hpp>
#include <quad_junctions/plane_cilia_geom.hpp>   // orient_group_flat, report_area, QuadElemList::ParamNodes
#include <quad_junctions/periodic_flow_utils.hpp>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>

using namespace sctl;
using namespace quad_junctions;

namespace {

// tol -> (Nbeta, max_depth) near-singular quadrature map for the QuadElemList plates (== periodic-sphere-bie).
template <class Real> void quad_scheme_for_tol(Real tol, Integer& Nbeta, Integer& max_depth) {
  const Real    tolL[4] = {(Real)1e-5, (Real)1e-7, (Real)1e-9, (Real)1e-11};
  const Integer NbL[4]  = {48, 100, 200, 400};
  const Integer mdL[4]  = {4, 8, 12, 30};
  int idx = 3;
  for (int k = 0; k < 4; k++) if (tolL[k] <= tol * (Real)(1 + 1e-6)) { idx = k; break; }
  Nbeta = NbL[idx]; max_depth = mdL[idx];
}

// One flat order x order patch spanning [0,L]x[0,L] at z=z_plane, normal along (0,0,uz). (== periodic-sphere-bie)
template <class Real> void add_plate(Vector<Real>& Xall, Integer order, Real L, Real z_plane, Real uz) {
  const Vector<Real>& nds = QuadElemList<Real>::ParamNodes(order);
  Vector<Real> Xp;
  for (Integer i = 0; i < order; i++) { const Real yy = nds[i] * L;
    for (Integer j = 0; j < order; j++) { const Real xx = nds[j] * L;
      Xp.PushBack(xx); Xp.PushBack(yy); Xp.PushBack(z_plane); } }
  orient_group_flat<Real>(Xp, order, z_plane, uz);
  for (auto v : Xp) Xall.PushBack(v);
}

// CSBQ slender sphere of radius R centered at c: centerline x = c[0] + R cos(theta) (theta in [0,pi]),
// y=c[1], z=c[2]; tube radius eps = R sin(theta); azimuthal reference orient (0,1,0). Nelem panels,
// ElemOrder Chebyshev nodes/panel, FourierOrder azimuthal modes. MPI-partitioned by global panel index
// (the SlenderElemList ctor takes THIS rank's local elements, unlike QuadElemList which self-slices).
template <class Real> SlenderElemList<Real> build_csbq_sphere(const Real c[3], Real R, Long Nelem,
    Long ElemOrder, Long FourierOrder, const Comm& comm) {
  const Long Npr = comm.Size(), pid = comm.Rank();
  // Partition the slender panels across ranks (contiguous slice, same style as the QuadElemList ctor).
  // QJ_SPHERE_ALL_RANK0=1 forces ALL panels onto rank 0 (rank>0 gets an empty slender list) -- the control
  // that isolates "cross-rank slender split" from "presence of a (partitioned/empty) slender list".
  const bool all_rank0 = std::getenv("QJ_SPHERE_ALL_RANK0") && std::atoi(std::getenv("QJ_SPHERE_ALL_RANK0"));
  const Long k0 = all_rank0 ? (pid == 0 ? 0 : Nelem) : (Nelem * pid) / Npr;
  const Long k1 = all_rank0 ? (pid == 0 ? Nelem : Nelem) : (Nelem * (pid + 1)) / Npr;
  Vector<Long> cheb_order, forder;
  Vector<Real> coord, radius, orient;
  for (Long i = 0; i < Nelem; i++) {
    if (i < k0 || i >= k1) continue;
    cheb_order.PushBack(ElemOrder); forder.PushBack(FourierOrder);
    const Vector<Real>& nds = SlenderElemList<Real>::CenterlineNodes(ElemOrder);
    for (Long j = 0; j < ElemOrder; j++) {
      const Real theta = const_pi<Real>() * ((Real)i + nds[j]) / (Real)Nelem;
      coord.PushBack(c[0] + R * cos<Real>(theta));
      coord.PushBack(c[1]);
      coord.PushBack(c[2]);
      radius.PushBack(R * sin<Real>(theta));
      orient.PushBack((Real)0); orient.PushBack((Real)1); orient.PushBack((Real)0);
    }
  }
  return SlenderElemList<Real>(cheb_order, forder, coord, radius, orient);
}

// Concatenate plate (quad) + sphere (slender) node coords / normals in name-sorted order ("0_plate"<"1_sphere").
template <class Real> void combined_nodes(const QuadElemList<Real>& plate, const SlenderElemList<Real>& sphere,
                                          Vector<Real>& X, Vector<Real>& Xn, Long& Np, Long& Ns) {
  Vector<Real> Xp, Xnp, Xs, Xns;
  plate.GetNodeCoord(&Xp, &Xnp, nullptr);
  sphere.GetNodeCoord(&Xs, &Xns, nullptr);
  Np = Xp.Dim()/3; Ns = Xs.Dim()/3;
  X.ReInit(0); Xn.ReInit(0);
  for (auto v : Xp)  X.PushBack(v);
  for (auto v : Xs)  X.PushBack(v);
  for (auto v : Xnp) Xn.PushBack(v);
  for (auto v : Xns) Xn.PushBack(v);
}

// Combined surface mean of a KDIM-per-node density over plates + sphere (each list mapped to its far-field
// nodes then weight-integrated). (== cilia_carpet-bie's combined_surface_mean, generalized names.)
template <class Real> void combined_surface_mean(const QuadElemList<Real>& plate, const SlenderElemList<Real>& sphere,
    const Vector<Real>& sigma, const Long Np, const Long Ns, const Integer KDIM, const Real tol,
    const Comm& comm, Vector<Real>& mean_out, Real& total_area) {
  Vector<Real> sm(KDIM); sm = 0; Real area = 0;
  auto accum = [&](const auto& lst, const Vector<Real>& sig_lst) {
    Vector<Real> Xff, Xnff, wts, dist, Fff; Vector<Long> cnt;
    lst.GetFarFieldNodes(Xff, Xnff, wts, dist, cnt, tol);
    lst.GetFarFieldDensity(Fff, sig_lst);
    const Vector<Real>& dens = (Fff.Dim() ? Fff : sig_lst);
    const Long Nq = wts.Dim();
    SCTL_ASSERT_MSG(dens.Dim() == Nq * KDIM, "combined_surface_mean: far-field density/weight size mismatch");
    for (Long i = 0; i < Nq; i++) { area += wts[i]; for (Integer k = 0; k < KDIM; k++) sm[k] += wts[i] * dens[i*KDIM+k]; }
  };
  Vector<Real> sp(Np*KDIM), ss(Ns*KDIM);
  for (Long i = 0; i < Np*KDIM; i++) sp[i] = sigma[i];
  for (Long i = 0; i < Ns*KDIM; i++) ss[i] = sigma[Np*KDIM + i];
  accum(plate, sp); accum(sphere, ss);
  total_area = GlobalReduce((double)area, comm, CommOp::SUM);
  mean_out.ReInit(KDIM);
  for (Integer k = 0; k < KDIM; k++) mean_out[k] = GlobalReduce((double)sm[k], comm, CommOp::SUM) / total_area;
}

// Pressure-driven doubly-periodic Stokes solve on the hybrid (QuadElemList plates + SlenderElemList sphere)
// + periodicity check + volume-flow VTK. Combined-field BIE with surface-mean projection. Per-node
// NormalOrient: +1 on the out-of-fluid plates, -1 on the into-fluid (radial-outward) slender sphere.
template <class Real>
void run_flow(const QuadElemList<Real>& plate, const SlenderElemList<Real>& sphere, const Comm& comm,
              const Real tol, const Real L, const Real pdrop, const Long Nvis, const Real R, const Real c[3],
              const Real z_bottom, const Real z_top) {
  const Real SL_scal = 1.0, DL_scal = 1.0;
  const Real gmres_tol = std::max(tol, (Real)1e-8);
  const Long gmres_max_iter = 400;
  constexpr Integer KDIM = 3;

  Vector<Real> X0, Xn0; Long Np = 0, Ns = 0;
  combined_nodes(plate, sphere, X0, Xn0, Np, Ns);
  const Long Nnode = Np + Ns;

  StokesBIO<Real> Op(SL_scal, DL_scal, comm);
  Op.SetAccuracy(tol);
  Op.AddElemList(plate,  "0_plate");
  Op.AddElemList(sphere, "1_sphere");
  Op.SetPeriodicity(sctl::Periodicity::XY, L);
  Op.SetTargetCoord(X0);

  // +1 on the plates (normals out of fluid, -I/2 jump), -1 on the slender sphere (radial-outward normal =
  // into fluid, +I/2 jump). Same per-node mix as planes_with_loops (walls vs make_normal_orient particles).
  Vector<Real> NormalOrient(3 * Nnode);
  for (Long i = 0; i < 3*Np; i++)      NormalOrient[i] = (Real)+1;
  for (Long i = 3*Np; i < 3*Nnode; i++) NormalOrient[i] = (Real)-1;

  const auto BIO = [&](Vector<Real>* U, const Vector<Real>& sigma) {
    Vector<Real> sm; Real total_area;
    combined_surface_mean<Real>(plate, sphere, sigma, Np, Ns, KDIM, tol, comm, sm, total_area);
    Vector<Real> sigma0 = sigma; AddConstVec(sigma0, sm * (Real)-1);
    U->SetZero();
    Op.ComputePotential(*U, sigma0);
    if (DL_scal && U->Dim() == sigma0.Dim()) (*U) -= sigma0 * (Real)0.5 * NormalOrient * DL_scal;  // surface only
    AddConstVec(*U, sm);
  };

  GMRES<Real> solver(comm);
  KrylovPrecond<Real> krylov;
  Vector<Real> sigma;
  Long niter = 0;
  Vector<Real> rhs = bg_flow_2peri(X0); rhs *= (pdrop / L);
  Profile::Enable(true); Profile::reset();
  Profile::Tic("periodic_sphere_csbq_gmres_solve", &comm, true);
  solver(&sigma, BIO, rhs, gmres_tol, gmres_max_iter, false, &niter, &krylov);
  Profile::Toc();
  if (!comm.Rank()) std::cout << "  flow: GMRES converged in " << niter << " iters\n";
  Profile::print(&comm, {"t_avg", "t_max", "f/s_avg", "f/s_max"});

  const Real zc = (Real)0.5 * (z_bottom + z_top);
  auto eval_induced = [&](const Vector<Real>& Xt) {
    Op.SetTargetCoord(Xt);
    Vector<Real> U(Xt.Dim()); U = 0; BIO(&U, sigma);
    Vector<Real> Ub = bg_flow_2peri(Xt); Ub *= (pdrop / L);
    U -= Ub; return U;
  };

  // ===== Verification 1: periodicity across the x and y faces (mid-gap) =====
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

  // ===== Verification 2: mid-gap probe points (self-convergence / symmetry) =====
  {
    const Real P[][3] = { {0.25*L,0.25*L,zc},{0.75*L,0.25*L,zc},{0.25*L,0.75*L,zc},
                          {0.75*L,0.75*L,zc},{0.50*L,0.15*L,zc},{0.50*L,0.85*L,zc} };
    const Long np = sizeof(P)/sizeof(P[0]);
    Vector<Real> Xp(3*np); for(Long i=0;i<np;i++) for(int cc=0;cc<3;cc++) Xp[i*3+cc]=P[i][cc];
    Vector<Real> Up = eval_induced(Xp);
    if(!comm.Rank()){ std::cout << std::setprecision(12) << "  [verify-probe] induced u at mid-gap probe points:\n";
      for(Long i=0;i<np;i++) std::cout << "    P"<<i<<" ("<<P[i][0]<<","<<P[i][1]<<","<<P[i][2]<<") u = "
        << Up[i*3] << ", " << Up[i*3+1] << ", " << Up[i*3+2] << "\n"; }
  }

  // ---- Volume-flow VTU (outside slab or inside sphere masked to 0) ----
  CubeVolumeVisShifted<Real> vv(Nvis, (Real)0.9, comm);
  Vector<Real> Xv = vv.GetCoord();
  const Long Ntrg = Xv.Dim() / 3;
  if (!comm.Rank()) std::cout << "  flow: volume grid " << Nvis << "^3, " << (Nvis*Nvis*Nvis) << " targets\n";
  Vector<Real> U = eval_induced(Xv);
  Vector<Real> Ubg = bg_flow_2peri(Xv); Ubg *= (pdrop / L);
  Real umax_ind = 0, umax_tot = 0, uxsum = 0, uysum = 0, uzsum = 0; Long nmid = 0;
  for (Long i = 0; i < Ntrg; i++) {
    const Real x = Xv[i*3], y = Xv[i*3+1], z = Xv[i*3+2];
    const Real rr = std::sqrt((x-c[0])*(x-c[0]) + (y-c[1])*(y-c[1]) + (z-c[2])*(z-c[2]));
    const bool masked = (z < z_bottom || z > z_top || rr < R * (Real)1.02);
    const Real ui0 = U[i*3], ui1 = U[i*3+1], ui2 = U[i*3+2];   // induced (background already subtracted)
    umax_ind = std::max(umax_ind, std::sqrt(ui0*ui0 + ui1*ui1 + ui2*ui2));
    if (masked) { U[i*3] = U[i*3+1] = U[i*3+2] = 0; }          // solid / outside slab -> zero
    else { U[i*3] = ui0 + Ubg[i*3]; U[i*3+1] = ui1 + Ubg[i*3+1]; U[i*3+2] = ui2 + Ubg[i*3+2]; }  // TOTAL = induced + bg
    if (!masked) {
      umax_tot = std::max(umax_tot, std::sqrt(U[i*3]*U[i*3] + U[i*3+1]*U[i*3+1] + U[i*3+2]*U[i*3+2]));
      if (std::fabs((double)(z - zc)) < 0.03) { uxsum += U[i*3]; uysum += U[i*3+1]; uzsum += U[i*3+2]; nmid++; }
    }
  }
  umax_ind = GlobalReduce((double)umax_ind, comm, CommOp::MAX);
  umax_tot = GlobalReduce((double)umax_tot, comm, CommOp::MAX);
  uxsum = GlobalReduce((double)uxsum, comm, CommOp::SUM); uysum = GlobalReduce((double)uysum, comm, CommOp::SUM);
  uzsum = GlobalReduce((double)uzsum, comm, CommOp::SUM); nmid = GlobalReduce((Long)nmid, comm, CommOp::SUM);
  if (!comm.Rank())
    std::cout << std::setprecision(6) << "  flow: max|U_induced| = " << umax_ind << "  max|U_total| = " << umax_tot
              << "   mid-gap mean total u = (" << (nmid?uxsum/nmid:0) << ", " << (nmid?uysum/nmid:0) << ", " << (nmid?uzsum/nmid:0) << ")\n";
  vv.WriteVTK("vis/PeriodicSphereCSBQ_U", U);   // TOTAL velocity (induced + background), masked to the fluid
  plate.WriteVTK("vis/PeriodicSphereCSBQ_plate", Vector<Real>(), comm);
  sphere.WriteVTK("vis/PeriodicSphereCSBQ_sphere", Vector<Real>(), comm);
  if (!comm.Rank()) std::cout << "  wrote vis/PeriodicSphereCSBQ_{U,plate,sphere}.pvtu/.vtu\n";

  // ---- 2D TOTAL-velocity slice CSVs (rank 0 builds the grid; others pass empty targets for the collective eval) ----
  auto dump_slice = [&](const std::string& fname, int axis, Real fixed, Long Nslc) {
    Vector<Real> Xs;
    if (!comm.Rank()) {
      for (Long i = 0; i < Nslc; i++) for (Long j = 0; j < Nslc; j++) {
        const Real u = (i + (Real)0.5) / Nslc * L;
        const Real w = (axis == 2) ? ((j + (Real)0.5) / Nslc * L)
                                    : (z_bottom + (j + (Real)0.5) / Nslc * (z_top - z_bottom));
        const Real x = u, y = (axis == 2) ? w : fixed, z = (axis == 2) ? fixed : w;
        Xs.PushBack(x); Xs.PushBack(y); Xs.PushBack(z);
      }
    }
    Vector<Real> Ui = eval_induced(Xs);
    Vector<Real> Ub = bg_flow_2peri(Xs); Ub *= (pdrop / L);
    if (comm.Rank()) return;
    std::ofstream f(fname);
    f << "# a1 a2 ux uy uz mask   (axis=" << (axis==2?"z-plane a1=x a2=y":"y-plane a1=x a2=z")
      << " fixed=" << fixed << " Ns=" << Nslc << " L=" << L << ")\n";
    f << std::setprecision(8);
    for (Long k = 0; k < Xs.Dim()/3; k++) {
      const Real x = Xs[k*3], y = Xs[k*3+1], z = Xs[k*3+2];
      const Real rr = std::sqrt((x-c[0])*(x-c[0]) + (y-c[1])*(y-c[1]) + (z-c[2])*(z-c[2]));
      const int mask = (rr < R) ? 1 : 0;
      const Real ux = Ui[k*3]+Ub[k*3], uy = Ui[k*3+1]+Ub[k*3+1], uz = Ui[k*3+2]+Ub[k*3+2];
      const Real a1 = x, a2 = (axis==2) ? y : z;
      f << a1 << " " << a2 << " " << ux << " " << uy << " " << uz << " " << mask << "\n";
    }
    std::cout << "  wrote " << fname << "\n";
  };
  dump_slice("vis/PeriodicSphereCSBQ_slice_z.csv", 2, (Real)0.5*(z_bottom+z_top), 128);
  dump_slice("vis/PeriodicSphereCSBQ_slice_xz.csv", 1, (Real)0.5*L, 128);
}

} // namespace

int main(int argc, char** argv) {
  Comm::MPI_Init(&argc, &argv);
  using Real = double;
  {
    Comm comm = Comm::World();

    // CLI: [order R tol pdrop Nvis Nelem fourier]
    const Integer order    = (argc > 1) ? std::atoi(argv[1]) : 12;    // QuadElemList plate order (multiple of 4)
    const Real    R        = (argc > 2) ? std::atof(argv[2]) : 0.25;  // sphere radius
    const Real    tol      = (argc > 3) ? std::atof(argv[3]) : 1e-7;
    const Real    pdrop    = (argc > 4) ? std::atof(argv[4]) : -1.0;
    const Long    Nvis     = (argc > 5) ? std::atol(argv[5]) : 60;
    const Long    Nelem    = (argc > 6) ? std::atol(argv[6]) : 2;     // slender-sphere panels
    const Long    fourier  = (argc > 7) ? std::atol(argv[7]) : 24;    // slender azimuthal Fourier order
    const Long    ElemOrder = 10;                                     // slender Chebyshev order (special_quad_q10)

    const Real L = 1.0, z_bottom = 0.01, z_top = 0.99;
    const Real c[3] = {(Real)0.5 * L, (Real)0.5 * L, (Real)0.5 * (z_bottom + z_top)};
    SCTL_ASSERT_MSG(c[2] - R > z_bottom && c[2] + R < z_top, "sphere must fit between the plates in z");
    SCTL_ASSERT_MSG(2 * R < L && R < (Real)0.5 * L, "sphere must not touch the periodic cell boundary/image");

    Integer Nbeta, max_depth; quad_scheme_for_tol<Real>(tol, Nbeta, max_depth);

    // Plates (QuadElemList), out-of-fluid normals (bottom -z / top +z).
    Vector<Real> Xall;
    add_plate<Real>(Xall, order, L, z_bottom, /*uz=*/-1);
    add_plate<Real>(Xall, order, L, z_top,    /*uz=*/+1);
    // QJ_PLATE_ONE_RANK=<r>: keep ALL plate panels on rank r (built with Comm::Self, empty on other ranks)
    // -- the stokes-periodize-numtest pattern (the plane is added on a single rank). Default (unset / <0)
    // distributes the plates across ranks via the comm-taking ctor (PartitionRange).
    const char* pr_env = std::getenv("QJ_PLATE_ONE_RANK");
    const int plate_rank = pr_env ? std::atoi(pr_env) : -1;
    QuadElemList<Real> plate = [&]() {
      if (plate_rank < 0) return QuadElemList<Real>(order, Xall, comm);
      const int R = std::min(plate_rank, (int)comm.Size() - 1);
      if (!comm.Rank()) std::cout << "  [plate placement] all plate panels on rank " << R << " (Comm::Self)\n";
      return QuadElemList<Real>(order, (comm.Rank() == R ? Xall : Vector<Real>()), Comm::Self());
    }();
    plate.SetQuadScheme(QuadElemList<Real>::QuadScheme::Hybrid, 10, Nbeta, max_depth);

    // Sphere (CSBQ SlenderElemList), radial-outward normal = into fluid.
    SlenderElemList<Real> sphere = build_csbq_sphere<Real>(c, R, Nelem, ElemOrder, fourier, comm);

    if (!comm.Rank()) {
      const Long nplate = 2 * order * order;
      std::cout << "periodic-sphere-CSBQ  order=" << order << " tol=" << tol
                << " (Nbeta=" << Nbeta << " max_depth=" << max_depth << ")"
                << " sphere R=" << R << " Nelem=" << Nelem << " ElemOrder=" << ElemOrder << " fourier=" << fourier
                << " center=(" << c[0] << "," << c[1] << "," << c[2] << ")\n"
                << "  plates z=" << z_bottom << "/" << z_top << " (each 1 patch, " << nplate << " nodes)\n";
    }
    report_area<Real>(plate, comm);
    {
      const Long nsp = GlobalReduce((Long)sphere.Size(), comm, CommOp::SUM);
      Vector<Real> Xs; sphere.GetNodeCoord(&Xs, nullptr, nullptr);
      const Long nsn = GlobalReduce((Long)(Xs.Dim()/3), comm, CommOp::SUM);
      if (!comm.Rank()) std::cout << "  sphere slender-panels=" << nsp << " (" << nsn << " nodes)\n";
    }

    run_flow<Real>(plate, sphere, comm, tol, L, pdrop, Nvis, R, c, z_bottom, z_top);
  }
  Comm::MPI_Finalize();
  return 0;
}
