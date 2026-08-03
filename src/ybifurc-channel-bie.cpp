/**
 * Bent-connector hybrid BIE demo + accuracy check (turning slender arms).
 *
 * Exercises HybridAssembly::add_bent_arm (include/quad_junctions/ybifurc_assembly.hpp): a RACETRACK
 * slender arm. It leaves each junction STRAIGHT along that junction's arm axis (coaxial, zero curvature at
 * the seam), bends smoothly through a smootherstep partition-of-unity "shoulder", runs straight across,
 * bends again, and enters the far seam straight -- the stadium shape. Both terminal rings stay
 * perpendicular to their seam axis (watertight); all curvature sits at the two shoulders, far from the
 * junctions/caps. The centerline is piecewise a single line (leads/run) or a degree-6 line-blend, with the
 * shoulder windows snapped to CSBQ panel boundaries so each panel is represented exactly.
 *
 * Two modes, both fed through the SAME coupled QuadElemList + SlenderElemList identity tests
 * (hybrid_bie_tests.hpp) so their accuracy is directly comparable to the coaxial (sine-bump) baseline:
 *
 *   tilt : two junctions joined by ONE bent arm (arm0<->arm0) that turns by `tiltDeg` degrees, the
 *          other four arms free/capped. tiltDeg=0 reproduces the straight coaxial connection (the M6
 *          two-junction baseline) exactly; tiltDeg~15-30 is the gentle bend. Confirms the bent arm keeps
 *          floor precision (DL const-density -> -1/2 and interior Green's identity, Laplace + Stokes).
 *
 *   lens : the DIVERGING-CONVERGING CHANNEL. Two Y-junctions face each other; each stem (arm0) is a
 *          free/capped inlet/outlet, and the two BRANCH pairs are joined by bent arms:
 *              A.arm2 (upper-right) <-> B.arm1 (upper-left)   [top wall]
 *              A.arm1 (lower-right) <-> B.arm2 (lower-left)   [bottom wall]
 *          The walls bow apart (diverge) mid-span and converge back at each junction -- an eye/lens. A
 *          two-junction channel needs turning connectors: two coaxial connections between two rigid
 *          Y-junctions is geometrically impossible (both connector lines would pass through both junction
 *          centres). Verified by the same DL + Green identities to floor precision.
 *
 *   mfg  : MANUFACTURED-SOLUTION exterior Stokes accuracy test on the SAME lens racetrack surface. A
 *          single Stokeslet is placed INSIDE the tube (junction A's centre -- the excluded region for an
 *          exterior problem), so its velocity field is exact throughout the exterior fluid (the lens
 *          "eye" + far field). Its Dirichlet trace on the surface is the RHS; the combined-field CFIE
 *              ( +1/2 I + SL_scal*S + DL_scal*D ) sigma = u_e|surface
 *          is solved by GMRES (residual printed per iteration) and the represented velocity is compared to
 *          the exact field at exterior probe targets (rel-L2 + max-abs). A single Stokeslet has NONZERO
 *          net force, so the SL term is essential -- this is why it is a combined SL+DL solve, and why the
 *          SL weight matters. If GMRES stalls, tune SL_scal per the CSBQ convention (both SL_scal/DL_scal
 *          are CLI args, no recompile). Writes the solved density + z=0 field/error slices (masked to the
 *          exterior fluid, the OPPOSITE of the interior flow drivers) as VTU + a slice CSV.
 *
 *   flow : PHYSICAL interior Stokes inflow/outflow (pressure-drop, no-slip) BVP on the SAME lens racetrack,
 *          with exactly ONE inlet (junction A's stem cap, -x) and ONE outlet (junction B's stem cap, +x).
 *          Parabolic cap profiles flux-normalized to +-p (p_in / p_out), no-slip on the walls/junctions;
 *          p_in==p_out => net flux 0 (the interior-Stokes compatibility int u.n dA=0, asserted). Solved by
 *          the combined-field ( -1/2 I + SL_scal*S + DL_scal*D ) sigma = u_bc via GMRES; the CLI
 *          gmres_max_iter lets you CAP the iteration count to watch the per-iter residual for stalling
 *          before solving through. The genus-1 lumen (closed loop) gives GMRES a slow circulation-mode
 *          phase; raise the SL weight (SL_scal~30, CSBQ's closed-loop scaling) to condition it. Samples
 *          the velocity on a z=0 slice masked to the fluid interior and
 *          writes density + prescribed-BC + flow-slice VTU + a flow-slice CSV (x,y,ux,uy,uz,umag).
 *
 *   make bin/ybifurc-channel-bie
 *   OMP_NUM_THREADS=8 ./bin/ybifurc-channel-bie [mode] \
 *       [level ord nref eta_join Ns_trans s_cap nAxFree fourier tol Nbeta max_depth cov_q \
 *        sep tiltDeg nBent lead_panels corner_panels geomOnly SL_scal DL_scal Ngrid p_in p_out gmres_max_iter Nvis]
 *   (Nvis is flow-only: junction-box per-axis sample count for the 3D interior point cloud; 0 => cbrt(Ngrid).)
 *   (SL_scal/DL_scal are the combined-field weights for mfg AND flow, default 1/1; Ngrid is mfg+flow;
 *    p_in/p_out/gmres_max_iter are flow-only, defaults 10/10/400. All ignored by lens/tilt. For the flow
 *    genus-1 LOOP, try SL_scal=30 -- CSBQ's slender-loop SL scaling -- to condition the circulation mode.)
 *
 *   Floor defaults (order12/nref2/fourier36 -> ~1e-10, per the accuracy-limiters recipe):
 *       level 1.5  ord 12  nref 2  eta_join 0.4  Ns_trans 3  s_cap 0.88  nAxFree 3  fourier 36
 *       tol 1e-11  Nbeta 400  max_depth 30  cov_q 6  sep(lens 9.6/tilt 18)  tiltDeg 25  nBent(auto)
 *       lead_panels 2  corner_panels 6  geomOnly 0
 *   Shape controls: shorter run <- smaller sep; smaller/tighter corners <- smaller corner_panels;
 *   shorter coaxial junction-arm lead <- smaller lead_panels.
 *   Coarse/fast iterate: ... 1.5 8 1 0.4 3 0.88 3 12  1e-6 100 4 6 ...
 *   Nbeta must be in {48,100,200,300,400,512}; cov_q in {6,10}.
 */

#include <csbq.hpp>                                  // CSBQ SlenderElemList
#include <quad_junctions/ybifurc_assembly.hpp>       // composable component API (add_bent_arm)
#include <quad_junctions/interior_viz.hpp>           // build_arm_panel_targets / build_box_targets (interior viz)
#include <quad_junctions/hybrid_bie_tests.hpp>       // shared BIE identity / watertightness tests
#include <array>
#include <cctype>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <string>
#include <vector>

using namespace sctl;
using namespace quad_junctions;

namespace {

// Distance from point P to the segment [A,B].
template <class Real>
Real seg_dist(const Vec3<Real>& P, const Vec3<Real>& A, const Vec3<Real>& B) {
  const Vec3<Real> ab{B[0]-A[0], B[1]-A[1], B[2]-A[2]}, ap{P[0]-A[0], P[1]-A[1], P[2]-A[2]};
  const Real L2 = ab[0]*ab[0]+ab[1]*ab[1]+ab[2]*ab[2];
  Real tp = (L2 > 0) ? (ap[0]*ab[0]+ap[1]*ab[1]+ap[2]*ab[2])/L2 : (Real)0;
  tp = std::max<Real>(0, std::min<Real>(1, tp));
  const Real dx=P[0]-(A[0]+tp*ab[0]), dy=P[1]-(A[1]+tp*ab[1]), dz=P[2]-(A[2]+tp*ab[2]);
  return std::sqrt(dx*dx+dy*dy+dz*dz);
}

// A shared/bent arm centerline sampled as a polyline (for exterior-source and collision checks).
template <class Real> struct ArmPoly { std::vector<Vec3<Real>> pts; Real rtube; };

template <class Real>
Real poly_dist(const Vec3<Real>& P, const ArmPoly<Real>& a) {
  Real d = std::numeric_limits<Real>::max();
  for (size_t i = 0; i + 1 < a.pts.size(); i++) d = std::min(d, seg_dist<Real>(P, a.pts[i], a.pts[i+1]));
  return d;
}
// Min distance between two sampled polylines (segment endpoints; dense enough for a clearance guard).
template <class Real>
Real poly_poly_dist(const ArmPoly<Real>& a, const ArmPoly<Real>& b) {
  Real d = std::numeric_limits<Real>::max();
  for (const auto& p : a.pts) d = std::min(d, poly_dist<Real>(p, b));
  return d;
}

// Single custom near-eval level: divergence + DL const-density (-> -1/2) + interior Green's identity,
// Laplace and Stokes, with a quad-vs-slender region-max breakdown and error-colored VTU dumps.
template <class Real>
void run_verify(QuadElemList<Real>& junc, SlenderElemList<Real>& arms, const Comm& comm,
                const Vector<Real>& X0, const std::string& tag, const std::string& label,
                const Real tol, const Integer Nbeta, const Integer md, const Integer cov_q) {
  const RegionReport<Real> region_report = [](const Vector<Real>& err, Long Nj, Long Na) {
    Real mj = 0, ma = 0;
    for (Long i = 0; i < Nj; i++) mj = std::max(mj, err[i]);
    for (Long i = 0; i < Na; i++) ma = std::max(ma, err[Nj+i]);
    std::cout << "    [region max] quad(junctions+transitions+caps)=" << mj << " slender(bent+free arms)=" << ma << "\n";
  };
  const Long njp = GlobalReduce((Long)junc.Size(), comm, CommOp::SUM), nap = GlobalReduce((Long)arms.Size(), comm, CommOp::SUM);
  Vector<Real> Xj, Xa; junc.GetNodeCoord(&Xj, nullptr, nullptr); arms.GetNodeCoord(&Xa, nullptr, nullptr);
  const Long njn = GlobalReduce((Long)(Xj.Dim()/3), comm, CommOp::SUM), nan = GlobalReduce((Long)(Xa.Dim()/3), comm, CommOp::SUM);
  if (!comm.Rank())
    std::cout << "\n---- BIE verify [" << label << "]: quad panels=" << njp << " nodes=" << njn
              << " | slender panels=" << nap << " nodes=" << nan << " ----\n"
              << "  [near-eval] Hybrid(cov_q=" << cov_q << ", Nbeta=" << Nbeta << ", max_depth=" << md
              << ")  tol=" << std::setprecision(1) << tol << "\n" << std::setprecision(6);
  junc.SetQuadScheme(QuadElemList<Real>::QuadScheme::Hybrid, cov_q, Nbeta, md);
  divergence_check<Real>(junc, arms, tol, comm);
  if (!comm.Rank()) std::cout << "    [Laplace] "; test_DLIdentity<Real, Laplace3D_DxU>(junc, arms, comm, tol, tag+"-dl-laplace", region_report);
  if (!comm.Rank()) std::cout << "    [Stokes]  "; test_DLIdentity<Real, Stokes3D_DxU>(junc, arms, comm, tol, tag+"-dl-stokes", region_report);
  if (!comm.Rank()) std::cout << "    [Laplace] "; test_greens_identity<Real, Laplace3D_FxU, Laplace3D_DxU, Laplace3D_FxdU>(junc, arms, comm, tol, X0, tag+"-green-laplace");
  if (!comm.Rank()) std::cout << "    [Stokes]  "; test_greens_identity<Real, Stokes3D_FxU, Stokes3D_DxU, Stokes3D_FxT>(junc, arms, comm, tol, X0, tag+"-green-stokes");
}

// Write an Nx x Ny z=const grid (coords Xg, values Ug both AoS, Ug a 3-vector/point) as a VTU of VTK_QUAD
// cells. Rank-0-only data, so written with Comm::Self(). (Copied from src/ybifurc-flow-bie.cpp.)
template <class Real>
void write_plane_vtu(const std::string& fname, const Vector<Real>& Xg, const Vector<Real>& Ug,
                     const Long Nx, const Long Ny) {
  VTUData vtu;
  const Long Np = Nx*Ny;
  for (Long i = 0; i < Np; i++)
    for (Integer k = 0; k < 3; k++) vtu.coord.PushBack((VTUData::VTKReal)Xg[3*i+k]);
  for (Long i = 0; i < Np; i++)
    for (Integer k = 0; k < 3; k++) vtu.value.PushBack((VTUData::VTKReal)Ug[3*i+k]);
  auto idx = [Ny](Long ix, Long iy) -> int32_t { return (int32_t)(ix*Ny + iy); };
  int32_t off = 0;
  for (Long ix = 0; ix < Nx-1; ix++)
    for (Long iy = 0; iy < Ny-1; iy++) {
      vtu.connect.PushBack(idx(ix, iy));   vtu.connect.PushBack(idx(ix+1, iy));
      vtu.connect.PushBack(idx(ix+1, iy+1)); vtu.connect.PushBack(idx(ix, iy+1));
      off += 4; vtu.offset.PushBack(off); vtu.types.PushBack((uint8_t)9);  // 9 = VTK_QUAD
    }
  vtu.WriteVTK(fname, Comm::Self());
}

// ---- Manufactured-solution EXTERIOR Stokes accuracy test on the lens racetrack. A single Stokeslet
//      placed inside the tube (junction A's centre -- the excluded region) has an exact velocity field
//      throughout the exterior fluid; its Dirichlet trace is the RHS. Solve the combined-field CFIE
//      ( +1/2 I + SL_scal*S + DL_scal*D ) sigma = u_e|surf via verbose GMRES (residual per iter), then
//      compare the represented velocity to the exact field at exterior probe targets. A single Stokeslet
//      has NONZERO net force -> the SL term is essential (the DL alone cannot represent net force). ----
template <class Real>
void run_manufactured(QuadElemList<Real>& junc, SlenderElemList<Real>& arms, const Comm& comm,
                      const Real level, const Placement<Real>& PA, const Placement<Real>& PB,
                      const Real sep, const HybridJunction<Real>& JA,
                      const std::vector<ArmPoly<Real>>& polys,
                      const std::vector<std::pair<Vec3<Real>,Vec3<Real>>>& freeseg,
                      const std::string& tag, const Real tol, const Integer Nbeta, const Integer md,
                      const Integer cov_q, const Real SL_scal, const Real DL_scal, const Long Ngrid) {
  const YField<Real> fld;
  const Real R0 = JA.seam(0).R0;
  Stokes3D_FxU ker_sl;

  // Exterior test: P is exterior iff below the iso in BOTH junction frames AND clear (with a small
  // standoff) of every wall/free-arm axis. Reuses the driver's polyline / free-arm-segment guards.
  auto is_exterior = [&](const Vec3<Real>& P) -> bool {
    const Real fA = fld.f(PA.apply_inverse_point(P)), fB = fld.f(PB.apply_inverse_point(P));
    if (fA >= level || fB >= level) return false;
    Real dclr = std::numeric_limits<Real>::max();
    for (const auto& pl : polys) dclr = std::min(dclr, poly_dist<Real>(P, pl) - pl.rtube);
    for (const auto& fs : freeseg) dclr = std::min(dclr, seg_dist<Real>(P, fs.first, fs.second) - R0);
    return dclr > (Real)0.15*R0;   // small standoff: exercise near-singular quadrature, but not singular
  };

  // Interior Stokeslet at junction A's centre (local origin -> world PA translation, deep in the enclosed
  // region). Nonzero net force. Assert it is inside (the exterior problem needs an interior source).
  const Vec3<Real> Xs_v = PA.apply_point(Vec3<Real>{0,0,0});
  const Vector<Real> Xsrc{Xs_v[0], Xs_v[1], Xs_v[2]};
  const Vector<Real> Fsrc{(Real)1, (Real)0.5, (Real)0};
  {
    const Real fsrc = fld.f(PA.apply_inverse_point(Xs_v));
    if (!comm.Rank())
      std::cout << "\n---- MANUFACTURED exterior Stokes (single Stokeslet inside the tube) ----\n"
                << "  [source] Stokeslet at junction A centre (" << std::setprecision(4)
                << Xs_v[0] << "," << Xs_v[1] << "," << Xs_v[2] << ") force=(1,0.5,0)  f=" << fsrc
                << " (>=" << level << " => INSIDE)\n" << std::setprecision(6)
                << "  [near-eval] Hybrid(cov_q=" << cov_q << ", Nbeta=" << Nbeta << ", max_depth=" << md
                << ")  tol=" << std::setprecision(1) << tol << " SL_scal=" << SL_scal
                << " DL_scal=" << DL_scal << std::setprecision(6) << "\n";
    SCTL_ASSERT_MSG(fsrc >= level, "manufactured source not inside the tube (exterior problem needs an interior source).");
  }

  // Viz grid bbox = global node bbox (+15% margin). GetNodeCoord is a local slice -> reduce.
  Real xlo = std::numeric_limits<Real>::max(), xhi = -xlo, ylo = xlo, yhi = -xlo;
  {
    Vector<Real> Xj, Xa; junc.GetNodeCoord(&Xj, nullptr, nullptr); arms.GetNodeCoord(&Xa, nullptr, nullptr);
    auto upd = [&](const Vector<Real>& V) {
      for (Long i = 0; i < V.Dim()/3; i++) {
        xlo = std::min(xlo, V[3*i]); xhi = std::max(xhi, V[3*i]);
        ylo = std::min(ylo, V[3*i+1]); yhi = std::max(yhi, V[3*i+1]);
      }
    };
    upd(Xj); upd(Xa);
    xlo = GlobalReduce((double)xlo, comm, CommOp::MIN); xhi = GlobalReduce((double)xhi, comm, CommOp::MAX);
    ylo = GlobalReduce((double)ylo, comm, CommOp::MIN); yhi = GlobalReduce((double)yhi, comm, CommOp::MAX);
    const Real mx = (Real)0.15*(xhi-xlo), my = (Real)0.15*(yhi-ylo);
    xlo -= mx; xhi += mx; ylo -= my; yhi += my;
  }
  const Long Nx = Ngrid;
  const Long Ny = std::max<Long>(2, (Long)std::lround((double)Ngrid*(yhi-ylo)/(xhi-xlo)));

  // Targets (rank 0 only, so each is evaluated/counted once): exterior probe points (error metric) THEN
  // the z=0 viz grid. Concatenated into Xall for a single evaluation inside solve_dirichlet_bvp.
  Vector<Real> Xprobe, Xgrid, Xall;
  Long Nprobe = 0, Ngrid_pts = 0;
  if (!comm.Rank()) {
    std::vector<Vec3<Real>> cand;
    cand.push_back({0,0,0});                                          // eye centre
    for (int i = -3; i <= 3; i++) cand.push_back({(Real)i, 0, 0});    // along the eye axis
    for (int i = -2; i <= 2; i++) { cand.push_back({(Real)i*(Real)1.5, (Real)2, 0});
                                    cand.push_back({(Real)i*(Real)1.5, (Real)-2, 0}); }
    const Integer Nring = 16;                                         // a ring well outside the whole lens
    for (Integer i = 0; i < Nring; i++) { const Real th = 2*const_pi<Real>()*i/Nring;
      cand.push_back({sep*cos<Real>(th), sep*sin<Real>(th), 0}); }
    Long nkept = 0;
    for (auto& P : cand) if (is_exterior(P)) { Xprobe.PushBack(P[0]); Xprobe.PushBack(P[1]); Xprobe.PushBack(P[2]); nkept++; }
    std::cout << "  [probes] exterior probe targets kept: " << nkept << " / " << (Long)cand.size() << "\n";
    SCTL_ASSERT_MSG(Xprobe.Dim() >= 3, "no exterior probe target survived the exterior check.");
    Nprobe = Xprobe.Dim()/3;
    for (Long ix = 0; ix < Nx; ix++) { const Real xx = xlo + (xhi-xlo)*ix/(Nx-1);
      for (Long iy = 0; iy < Ny; iy++) { const Real yy = ylo + (yhi-ylo)*iy/(Ny-1);
        Xgrid.PushBack(xx); Xgrid.PushBack(yy); (Xgrid.PushBack((Real)0)); } }
    Ngrid_pts = Nx*Ny;
    for (Long i = 0; i < Xprobe.Dim(); i++) Xall.PushBack(Xprobe[i]);
    for (Long i = 0; i < Xgrid.Dim();  i++) Xall.PushBack(Xgrid[i]);
    std::cout << "  [grid] z=0 slice " << Nx << " x " << Ny << " = " << Ngrid_pts << " points over x["
              << std::setprecision(4) << xlo << "," << xhi << "] y[" << ylo << "," << yhi << "]\n" << std::setprecision(6);
  }

  // Dirichlet data: exact Stokeslet velocity at the combined surface nodes (local slice).
  Vector<Real> X, Xn; Long Nj, Na; combined_nodes(junc, arms, X, Xn, Nj, Na);
  Vector<Real> bc; ker_sl.Eval(bc, X, Xsrc, Xsrc, Fsrc);

  // Solve the EXTERIOR combined-field CFIE (interior=false -> jump=+1/2*DL_scal; no SL-sign flip) and
  // evaluate the represented velocity at Xall. Verbose GMRES prints the residual per iteration.
  junc.SetQuadScheme(QuadElemList<Real>::QuadScheme::Hybrid, cov_q, Nbeta, md);
  Vector<Real> Uall;
  const Vector<Real> sigma = solve_dirichlet_bvp<Real, Stokes3D_FxU, Stokes3D_DxU>(
      junc, arms, comm, tol, bc, /*interior=*/false, SL_scal, DL_scal, Xall, &Uall,
      "stokes manufactured (exterior)", /*gmres_max_iter=*/400);

  // Error at the exterior probes (rank 0 holds all targets; no off-surface jump term).
  Vector<Real> Ugrid, Uerr;
  if (!comm.Rank()) {
    Vector<Real> Xpr(Nprobe*3), Uref;
    for (Long i = 0; i < Nprobe*3; i++) Xpr[i] = Xall[i];
    ker_sl.Eval(Uref, Xpr, Xsrc, Xsrc, Fsrc);
    Real err2 = 0, ref2 = 0, emax = 0;
    for (Long i = 0; i < Nprobe*3; i++) { const Real e = Uall[i]-Uref[i]; err2 += e*e; ref2 += Uref[i]*Uref[i]; emax = std::max(emax, std::fabs((double)e)); }
    std::cout << std::setprecision(6) << "  stokes manufactured (exterior): probes=" << Nprobe
              << "  rel-L2 error = " << std::sqrt(err2/ref2) << "  max-abs = " << emax << "\n";
    // Grid field + error field (exact evaluated directly).
    Ugrid.ReInit(Ngrid_pts*3); for (Long i = 0; i < Ngrid_pts*3; i++) Ugrid[i] = Uall[Nprobe*3+i];
    Vector<Real> Ugex; ker_sl.Eval(Ugex, Xgrid, Xsrc, Xsrc, Fsrc);
    Uerr.ReInit(Ngrid_pts*3); for (Long i = 0; i < Ngrid_pts*3; i++) Uerr[i] = Ugrid[i]-Ugex[i];
  }

  // Exterior mask: Laplace-DL constant-density indicator (~ -1 inside the tube, ~0 exterior). KEEP the
  // exterior fluid (incl. the eye); zero the tube interior -- OPPOSITE of the interior flow drivers.
  {
    BoundaryIntegralOp<Real, Laplace3D_DxU> IndOp((Laplace3D_DxU()), false, comm);
    SetPVFMMKer(IndOp);
    IndOp.SetAccuracy(tol);
    IndOp.AddElemList(junc, "0_junc"); IndOp.AddElemList(arms, "1_arms");
    Vector<Real> ones(Nj+Na); ones = 1;
    IndOp.SetTargetCoord(Xgrid);   // rank-0 grid; empty elsewhere (collective eval below)
    Vector<Real> ind; IndOp.ComputePotential(ind, ones);
    if (!comm.Rank()) {
      Long n_ext = 0;
      for (Long i = 0; i < Ngrid_pts; i++) {
        if (std::fabs((double)ind[i]) < 0.5) n_ext++;                 // exterior fluid -> keep
        else { for (int k = 0; k < 3; k++) { Ugrid[3*i+k] = 0; Uerr[3*i+k] = 0; } }  // tube interior -> zero
      }
      std::cout << "  [grid] exterior points (|DL indicator|<0.5): " << n_ext << " / " << Ngrid_pts << "\n";
    }
  }

  // Output: solved surface density (3-vec, collective) + masked z=0 field/error slices + a slice CSV.
  {
    Vector<Real> sj(Nj*3), sa_(Na*3);
    for (Long i = 0; i < Nj*3; i++) sj[i] = sigma[i];
    for (Long i = 0; i < Na*3; i++) sa_[i] = sigma[Nj*3+i];
    junc.WriteVTK(tag + "-sigma-junc", sj,  comm);   // collective
    arms.WriteVTK(tag + "-sigma-arms", sa_, comm);
  }
  if (!comm.Rank()) {
    write_plane_vtu<Real>(tag + "-stokes-slice",     Xgrid, Ugrid, Nx, Ny);
    write_plane_vtu<Real>(tag + "-stokes-err-slice", Xgrid, Uerr,  Nx, Ny);
    std::ofstream csv(tag + "-slice.csv");
    csv << "x,y,umag,errmag\n";
    for (Long i = 0; i < Ngrid_pts; i++) {
      const Real um = std::sqrt(Ugrid[3*i]*Ugrid[3*i]+Ugrid[3*i+1]*Ugrid[3*i+1]+Ugrid[3*i+2]*Ugrid[3*i+2]);
      const Real em = std::sqrt(Uerr[3*i]*Uerr[3*i]+Uerr[3*i+1]*Uerr[3*i+1]+Uerr[3*i+2]*Uerr[3*i+2]);
      csv << Xgrid[3*i] << "," << Xgrid[3*i+1] << "," << um << "," << em << "\n";
    }
    std::cout << "  [viz] wrote " << tag << "-sigma-{junc,arms}.pvtu, -stokes-{,err-}slice.vtu, -slice.csv\n";
  }
}

// ---- Physical INTERIOR Stokes inflow/outflow (pressure-drop, no-slip) BVP on the lens racetrack, with
//      exactly ONE inlet (junction A's stem cap) and ONE outlet (junction B's stem cap). Ported from
//      src/ybifurc-flow-bie.cpp (the 2-bifurcation flow driver): parabolic cap profile flux-normalized to
//      +-p, no-slip everywhere else, net flux 0 (p_in==p_out) = the interior-Stokes compatibility. Solved
//      by the SAME combined-field ( -1/2 I - S + D ) sigma = u_bc (SL_scal=-1, DL_scal=+1) via GMRES. ----
template <class Real> struct FlowCap { Vec3<Real> C, u; Real R0 = 0, amp = 0, p = 0; int sgn = 0; };
template <class Real> inline Real dot3(const Vec3<Real>& a, const Vec3<Real>& b) { return a[0]*b[0]+a[1]*b[1]+a[2]*b[2]; }

// Parabolic dome profile prof = 1-(r/R0)^2 (r = transverse dist from the cap axis) for nodes on that cap's
// hemisphere (|X-C| ~ R0, +u side); 0 for any other point (no-slip walls/junctions/arms).
template <class Real>
bool cap_profile(const Vec3<Real>& X, const FlowCap<Real>& c, Real& prof) {
  const Vec3<Real> d{X[0]-c.C[0], X[1]-c.C[1], X[2]-c.C[2]};
  const Real ax = dot3(d, c.u), dist2 = dot3(d, d), dist = sqrt<Real>(dist2);
  if (std::fabs((double)(dist - c.R0)) < (double)((Real)0.05*c.R0) && ax > (Real)-0.05*c.R0) {
    Real r2 = dist2 - ax*ax; if (r2 < 0) r2 = 0;
    prof = (Real)1 - r2/(c.R0*c.R0); if (prof < 0) prof = 0; return true;
  }
  prof = 0; return false;
}
template <class Real>
Vec3<Real> flow_bc_vel(const Vec3<Real>& X, const std::vector<FlowCap<Real>>& caps) {
  for (const auto& c : caps) { Real prof;
    if (cap_profile<Real>(X, c, prof)) { const Real s = c.amp*prof; return Vec3<Real>{s*c.u[0], s*c.u[1], s*c.u[2]}; } }
  return Vec3<Real>{(Real)0, (Real)0, (Real)0};
}

template <class Real>
void run_flow(QuadElemList<Real>& junc, SlenderElemList<Real>& arms, const Comm& comm,
              const HybridJunction<Real>& JA, const HybridJunction<Real>& JB, const Real s_cap,
              const std::string& tag, const Real tol, const Integer Nbeta, const Integer md,
              const Integer cov_q, const Real p_in, const Real p_out, const Long gmres_max_iter,
              const Long Ngrid, const Real SL_scal, const Real DL_scal, const Long Nvis) {
  // The only two open (capped) seams are the stems: A.arm0 (-x, inlet) and B.arm0 (+x, outlet).
  std::vector<FlowCap<Real>> caps;
  {
    const ArmSeam<Real>* stems[2] = {&JA.seam(0), &JB.seam(0)};
    for (const ArmSeam<Real>* s : stems) {
      FlowCap<Real> c;
      c.C = Vec3<Real>{s->C[0]+(s_cap-s->a0)*s->u[0], s->C[1]+(s_cap-s->a0)*s->u[1], s->C[2]+(s_cap-s->a0)*s->u[2]};
      c.u = s->u; c.R0 = s->R0;
      if (c.C[0] < 0) { c.sgn = -1; c.p = p_in;  }   // negative-x stem = INFLOW
      else            { c.sgn = +1; c.p = p_out; }   // positive-x stem = OUTFLOW
      caps.push_back(c);
    }
    if (!comm.Rank()) {
      std::cout << "\n---- PRESSURE-DROP / no-slip INTERIOR Stokes flow (1 inlet, 1 outlet) ----\n"
                << "  [formulation] combined-field (-1/2 I + SL_scal*S + DL_scal*D), interior; SL_scal="
                << SL_scal << " DL_scal=" << DL_scal << "  (CSBQ uses SL~30 for a closed LOOP)\n"
                << "  [caps] R0=" << std::setprecision(6) << caps[0].R0 << "  (2 stem hemisphere caps)\n";
      for (size_t i = 0; i < caps.size(); i++)
        std::cout << "    cap " << i << ": center=(" << std::setprecision(4) << caps[i].C[0] << "," << caps[i].C[1] << "," << caps[i].C[2]
                  << ")  axis=(" << caps[i].u[0] << "," << caps[i].u[1] << "," << caps[i].u[2] << ")  "
                  << (caps[i].sgn < 0 ? "INFLOW" : "OUTFLOW") << " p=" << caps[i].p << "\n" << std::setprecision(6);
    }
  }

  // Geometric flux factor g_c = int prof*(u.n) dA (caps live in the quad list) -> amp_c = sgn_c*p_c/g_c.
  {
    Vector<Real> Xf, Xnf, wts, dist; Vector<Long> cnt;
    junc.GetFarFieldNodes(Xf, Xnf, wts, dist, cnt, tol);
    const Long Nf = wts.Dim();
    Vector<Real> g((Long)caps.size()); g = 0;
    for (Long i = 0; i < Nf; i++) {
      const Vec3<Real> X{Xf[3*i], Xf[3*i+1], Xf[3*i+2]}, n{Xnf[3*i], Xnf[3*i+1], Xnf[3*i+2]};
      for (size_t c = 0; c < caps.size(); c++) { Real prof;
        if (cap_profile<Real>(X, caps[c], prof)) { g[(Long)c] += wts[i]*prof*dot3(caps[c].u, n); break; } }
    }
    for (size_t c = 0; c < caps.size(); c++) g[(Long)c] = GlobalReduce((double)g[(Long)c], comm, CommOp::SUM);
    for (size_t c = 0; c < caps.size(); c++) {
      SCTL_ASSERT_MSG(std::fabs((double)g[(Long)c]) > 1e-30, "degenerate cap flux factor");
      caps[c].amp = (Real)caps[c].sgn * caps[c].p / g[(Long)c];
    }
  }

  // Boundary velocity RHS over the combined nodes, then VERIFY flux per cap (+-p) and total (~0).
  Vector<Real> X, Xn; Long Nj, Na; combined_nodes(junc, arms, X, Xn, Nj, Na);
  const Long Nnode = Nj + Na;
  Vector<Real> bc(Nnode*3);
  for (Long i = 0; i < Nnode; i++) {
    const Vec3<Real> v = flow_bc_vel<Real>(Vec3<Real>{X[3*i], X[3*i+1], X[3*i+2]}, caps);
    bc[3*i] = v[0]; bc[3*i+1] = v[1]; bc[3*i+2] = v[2];
  }
  {
    Vector<Real> Xf, Xnf, wts, dist; Vector<Long> cnt;
    junc.GetFarFieldNodes(Xf, Xnf, wts, dist, cnt, tol);
    const Long Nf = wts.Dim();
    Vector<Real> flux((Long)caps.size()); flux = 0;
    for (Long i = 0; i < Nf; i++) {
      const Vec3<Real> X0{Xf[3*i], Xf[3*i+1], Xf[3*i+2]}, n{Xnf[3*i], Xnf[3*i+1], Xnf[3*i+2]};
      const Vec3<Real> v = flow_bc_vel<Real>(X0, caps);
      for (size_t c = 0; c < caps.size(); c++) { Real prof;
        if (cap_profile<Real>(X0, caps[c], prof)) { flux[(Long)c] += wts[i]*dot3(v, n); break; } }
    }
    Real total = 0;
    for (size_t c = 0; c < caps.size(); c++) { flux[(Long)c] = GlobalReduce((double)flux[(Long)c], comm, CommOp::SUM); total += flux[(Long)c]; }
    if (!comm.Rank()) {
      std::cout << "  [flux check] int v.n dA per cap (target +-p):\n";
      for (size_t c = 0; c < caps.size(); c++)
        std::cout << "    cap " << c << " (" << (caps[c].sgn < 0 ? "in " : "out") << "): flux="
                  << std::setprecision(6) << flux[(Long)c] << "  (target " << (Real)caps[c].sgn*caps[c].p << ")\n";
      std::cout << "    TOTAL net flux = " << total << "  (compatibility condition int u.n dA = 0)\n";
    }
    SCTL_ASSERT_MSG(std::fabs((double)total) < 1e-6, "net flux not zero -- interior Stokes BVP incompatible");
  }

  // z=0 viz grid = global node bbox (+15% margin); rank-0 only.
  Real xlo = std::numeric_limits<Real>::max(), xhi = -xlo, ylo = xlo, yhi = -xlo;
  {
    Vector<Real> Xj, Xa; junc.GetNodeCoord(&Xj, nullptr, nullptr); arms.GetNodeCoord(&Xa, nullptr, nullptr);
    auto upd = [&](const Vector<Real>& V) { for (Long i = 0; i < V.Dim()/3; i++) {
      xlo = std::min(xlo, V[3*i]); xhi = std::max(xhi, V[3*i]); ylo = std::min(ylo, V[3*i+1]); yhi = std::max(yhi, V[3*i+1]); } };
    upd(Xj); upd(Xa);
    xlo = GlobalReduce((double)xlo, comm, CommOp::MIN); xhi = GlobalReduce((double)xhi, comm, CommOp::MAX);
    ylo = GlobalReduce((double)ylo, comm, CommOp::MIN); yhi = GlobalReduce((double)yhi, comm, CommOp::MAX);
    const Real mx = (Real)0.15*(xhi-xlo), my = (Real)0.15*(yhi-ylo); xlo -= mx; xhi += mx; ylo -= my; yhi += my;
  }
  const Long Nx = Ngrid, Ny = std::max<Long>(2, (Long)std::lround((double)Ngrid*(yhi-ylo)/(xhi-xlo)));
  // Interior point cloud (in addition to the z=0 slice): arm cross-section stars at each CSBQ panel's first
  // Chebyshev node + junction-body boxes (interior_viz.hpp). build_arm_panel_targets uses per-rank GetGeom,
  // so it is COLLECTIVE and must run on every rank; junction boxes are pure rank-0 geometry.
  Vector<Real> Xarm; build_arm_panel_targets<Real>(arms, comm, /*cheb=*/10, Xarm);
  Vector<Real> Xgrid; Long Ngrid_pts = 0, Nslice = 0, Narm = 0, Njunc = 0, Nax = 0;
  if (!comm.Rank()) {
    for (Long ix = 0; ix < Nx; ix++) { const Real xx = xlo + (xhi-xlo)*ix/(Nx-1);
      for (Long iy = 0; iy < Ny; iy++) { const Real yy = ylo + (yhi-ylo)*iy/(Ny-1);
        Xgrid.PushBack(xx); Xgrid.PushBack(yy); Xgrid.PushBack((Real)0); } }
    Nslice = Nx*Ny;
    for (Long i = 0; i < Xarm.Dim(); i++) Xgrid.PushBack(Xarm[i]);
    Narm = Xgrid.Dim()/3 - Nslice;
    // Junction cubes: center = mean of the junction's 3 seam-ring centers; half = farthest seam ring + 15%.
    Vector<Real> jc(6), jh(2);
    const HybridJunction<Real>* JJ[2] = {&JA, &JB};
    for (int j = 0; j < 2; j++) {
      Real cx = 0, cy = 0, cz = 0;
      for (int k = 0; k < 3; k++) { const ArmSeam<Real>& s = JJ[j]->seam(k); cx += s.C[0]; cy += s.C[1]; cz += s.C[2]; }
      cx /= 3; cy /= 3; cz /= 3;
      Real h = 0;
      for (int k = 0; k < 3; k++) { const ArmSeam<Real>& s = JJ[j]->seam(k);
        const Real dx = s.C[0]-cx, dy = s.C[1]-cy, dz = s.C[2]-cz; h = std::max(h, std::sqrt(dx*dx+dy*dy+dz*dz)); }
      jc[3*j] = cx; jc[3*j+1] = cy; jc[3*j+2] = cz; jh[j] = (Real)1.15*h;
    }
    Nax = (Nvis > 0) ? Nvis : std::max<Long>(3, (Long)std::lround(std::cbrt((double)Ngrid)));
    Vector<Real> Xjb; build_box_targets<Real>(jc, jh, Nax, Xjb);
    for (Long i = 0; i < Xjb.Dim(); i++) Xgrid.PushBack(Xjb[i]);
    Njunc = Xgrid.Dim()/3 - Nslice - Narm;
    Ngrid_pts = Xgrid.Dim()/3;
    std::cout << "  [grid] z=0 slice " << Nx << " x " << Ny << " = " << Nslice << " + " << Narm
              << " arm stars (" << (Narm/16) << " panels x16) + " << Njunc << " junction-box (2 x " << Nax
              << "^3) over x[" << std::setprecision(4) << xlo << "," << xhi << "] y[" << ylo << "," << yhi
              << "]\n" << std::setprecision(6);
  }

  // Solve the INTERIOR Stokes Dirichlet BVP (jump=-1/2*DL_scal). SL_scal is the SINGLE-LAYER weight (the
  // "slender coefficient"): the genus-1 lumen gives GMRES a slow circulation-mode phase, and CSBQ scales
  // the SL up (~30 for a closed loop) to condition it -- try SL_scal=30 vs the default 1. (interior +
  // same-sign SL/DL -> solve_dirichlet_bvp flips SL to negative internally, so |SL_scal| is what matters.)
  // gmres_max_iter is the CLI knob: run a SMALL value first to watch the residual, then the full value.
  junc.SetQuadScheme(QuadElemList<Real>::QuadScheme::Hybrid, cov_q, Nbeta, md);
  Vector<Real> Ugrid;
  const Vector<Real> sigma = solve_dirichlet_bvp<Real, Stokes3D_FxU, Stokes3D_DxU>(
      junc, arms, comm, tol, bc, /*interior=*/true, SL_scal, DL_scal,
      Xgrid, &Ugrid, "stokes pressure-drop (interior)", gmres_max_iter);

  // Interior mask/filter: Laplace-DL const-density indicator (~ -1 interior, ~0 exterior). On the z=0
  // slice, KEEP the fluid interior and zero the exterior. For the 3D cloud, arm stars are interior by
  // construction (always kept) and junction-box points are kept only where |ind|>0.5 (the "check inside").
  Vector<Real> Xvis, Uvis;
  {
    BoundaryIntegralOp<Real, Laplace3D_DxU> IndOp((Laplace3D_DxU()), false, comm);
    SetPVFMMKer(IndOp);
    IndOp.SetAccuracy(tol);
    IndOp.AddElemList(junc, "0_junc"); IndOp.AddElemList(arms, "1_arms");
    Vector<Real> ones(Nnode); ones = 1;
    IndOp.SetTargetCoord(Xgrid);
    Vector<Real> ind; IndOp.ComputePotential(ind, ones);
    if (!comm.Rank()) {
      Long n_in = 0;
      for (Long i = 0; i < Nslice; i++) {
        if (std::fabs((double)ind[i]) > 0.5) n_in++;
        else { Ugrid[3*i] = 0; Ugrid[3*i+1] = 0; Ugrid[3*i+2] = 0; }
      }
      Long n_junc_in = 0;
      for (Long i = Nslice; i < Nslice + Narm; i++)                 // arm stars: interior by construction
        for (Integer k = 0; k < 3; k++) { Xvis.PushBack(Xgrid[3*i+k]); Uvis.PushBack(Ugrid[3*i+k]); }
      for (Long i = Nslice + Narm; i < Ngrid_pts; i++)              // junction box: keep interior only
        if (std::fabs((double)ind[i]) > 0.5) {
          for (Integer k = 0; k < 3; k++) { Xvis.PushBack(Xgrid[3*i+k]); Uvis.PushBack(Ugrid[3*i+k]); }
          n_junc_in++;
        }
      std::cout << "  [grid] slice interior " << n_in << " / " << Nslice << "; cloud = " << Narm
                << " arm + " << n_junc_in << " / " << Njunc << " junction = " << (Xvis.Dim()/3) << " pts\n";
    }
  }

  // Output: surface density + prescribed BC velocity + z=0 flow slice + 3D interior cloud (VTU) + slice CSV.
  {
    Vector<Real> sj(Nj*3), sa_(Na*3), bcj(Nj*3);
    for (Long i = 0; i < Nj*3; i++) { sj[i] = sigma[i]; bcj[i] = bc[i]; }
    for (Long i = 0; i < Na*3; i++) sa_[i] = sigma[Nj*3+i];
    junc.WriteVTK(tag + "-sigma-junc", sj,  comm);
    arms.WriteVTK(tag + "-sigma-arms", sa_, comm);
    junc.WriteVTK(tag + "-bc-junc",    bcj, comm);
  }
  if (!comm.Rank()) {
    write_plane_vtu<Real>(tag + "-flow-slice", Xgrid, Ugrid, Nx, Ny);
    write_points_vtu<Real>(tag + "-flow-box", Xvis, Uvis, Xvis.Dim()/3);
    std::ofstream csv(tag + "-flow-slice.csv");
    csv << "x,y,ux,uy,uz,umag\n";
    for (Long i = 0; i < Nslice; i++) {
      const Real ux = Ugrid[3*i], uy = Ugrid[3*i+1], uz = Ugrid[3*i+2];
      csv << Xgrid[3*i] << "," << Xgrid[3*i+1] << "," << ux << "," << uy << "," << uz << ","
          << std::sqrt(ux*ux+uy*uy+uz*uz) << "\n";
    }
    std::cout << "  [viz] wrote " << tag << "-sigma-{junc,arms}.pvtu, -bc-junc.pvtu, -flow-slice.vtu, -flow-box.vtu, -flow-slice.csv\n";
  }
}

} // anonymous namespace

int main(int argc, char** argv) {
  Comm::MPI_Init(&argc, &argv);
  using Real = double;
  {
    const Comm comm = Comm::World();
    int a = 1;
    const std::string mode = (argc > a && argv[a][0] && !std::isdigit((unsigned char)argv[a][0])) ? std::string(argv[a++]) : std::string("lens");
    auto argf = [&](Real d) -> Real { return (argc > a) ? (Real)atof(argv[a++]) : d; };
    auto argi = [&](Integer d) -> Integer { return (argc > a) ? (Integer)atoi(argv[a++]) : d; };
    const Real    level   = argf((Real)1.5);
    const Integer ord     = argi(12);
    const Integer nref    = argi(2);
    const Real    etajoin = argf((Real)0.4);
    const Integer NsTrans = argi(3);
    const Real    s_cap   = argf((Real)0.88);
    const Integer nAxFree = argi(3);
    const Long    fourier = (Long)argi(36);
    const Real    tol     = argf((Real)1e-11);
    const Integer Nbeta   = argi(400);
    const Integer md      = argi(30);
    const Integer cov_q   = argi(6);
    const bool    mfg     = (mode == "mfg");
    const bool    flow    = (mode == "flow");
    const bool    lens    = (mode == "lens" || mfg || flow);     // mfg/flow build the SAME lens racetrack surface
    const Real    sep     = argf(lens ? (Real)9.6 : (Real)18);  // junction-center separation (lens: shorter run)
    const Real    tiltDeg = argf((Real)25);                      // tilt-mode bend angle (deg)
    const Integer nBentA  = argi(-1);                            // bent-arm axial panels (auto if <=0)
    const Integer leadP   = argi(2);                             // straight coaxial lead panels each seam
    const Integer cornerP = argi(6);                             // panels per shoulder corner
    const Integer geomOnly= argi(0);
    const Real    SL_scal = argf((Real)1);                       // mfg: SL weight (CSBQ-tunable if GMRES stalls)
    const Real    DL_scal = argf((Real)1);                       // mfg: DL weight
    const Long    Ngrid   = (Long)argi(200);                     // mfg/flow: z=0 viz grid resolution
    const Real    p_in    = argf((Real)10);                      // flow: inlet (A stem) volumetric flux
    const Real    p_out   = argf((Real)10);                      // flow: outlet (B stem) volumetric flux
    const Long    gmresMax= (Long)argi(400);                     // flow: GMRES max iters (cap it to check stalling)
    const Long    Nvis    = (Long)argi(0);                       // flow: junction-box per-axis samples (0 = cbrt(Ngrid))
    const Integer Ncap    = (Integer)(YSwept::Ncap0 * nref);
    const Long    cheb    = 10;
    pou_kind() = 1;   // smootherstep POU (order-exact)

    SCTL_ASSERT_MSG(ord >= 4 && ord <= 48 && ord % 4 == 0, "order must be a multiple of 4 in {4,...,48}.");
    SCTL_ASSERT_MSG(mode == "lens" || mode == "tilt" || mode == "mfg" || mode == "flow", "mode must be 'lens', 'tilt', 'mfg', or 'flow'.");

    if (!comm.Rank()) {
      std::cout << "\n=== BENT-CONNECTOR hybrid Y-bifurcation [" << mode << "] ===\n";
      std::cout << "  order=" << ord << " level=" << level << " nref=" << nref << " eta_join=" << etajoin
                << " Ns_trans=" << NsTrans << " s_cap=" << s_cap << " fourier=" << fourier
                << " sep=" << sep << (lens ? "" : (" tiltDeg=" + std::to_string((long)tiltDeg)))
                << " lead_panels=" << leadP << " corner_panels=" << cornerP
                << (geomOnly ? "  [GEOM-ONLY]" : "") << "\n";
    }

    HybridAssembly<Real> A(ord);
    const Real pi = const_pi<Real>();
    const Vec3<Real> up{0,0,1};

    // Place the two junctions (both rotated ONLY about z, so every seam's orient e1 stays +z and the whole
    // assembly is planar in z=0 -> add_bent_arm's planar-turn requirement holds by construction).
    Placement<Real> PA, PB;
    if (lens) {
      // A stem -> -x, B stem -> +x (facing). Branches: A.arm1 lower-right, A.arm2 upper-right;
      // B.arm1 upper-left, B.arm2 lower-left.
      PA = Placement<Real>::AlignArm(0, Vec3<Real>{-1,0,0}, up, Vec3<Real>{-sep/2,0,0});
      PB = Placement<Real>::AlignArm(0, Vec3<Real>{ 1,0,0}, up, Vec3<Real>{ sep/2,0,0});
    } else {
      // A.arm0 -> +x tilted UP by tiltDeg/2, B.arm0 -> -x tilted UP by tiltDeg/2, so the arm0<->arm0
      // connector (a SINGLE-CORNER lead|corner|run arm; testbed for the vessels-tree intra-junction link)
      // turns by exactly tiltDeg, the run plunging straight into B's stem; other arms splay outward, capped.
      const Real h = (tiltDeg*pi/180)/2;
      PA = Placement<Real>::AlignArm(0, Vec3<Real>{ cos<Real>(h), sin<Real>(h), 0}, up, Vec3<Real>{-sep/2,0,0});
      PB = Placement<Real>::AlignArm(0, Vec3<Real>{-cos<Real>(h), sin<Real>(h), 0}, up, Vec3<Real>{ sep/2,0,0});
    }
    const HybridJunction<Real> JA = A.add_junction(PA, level, nref, etajoin, NsTrans);
    const HybridJunction<Real> JB = A.add_junction(PB, level, nref, etajoin, NsTrans);

    // Bent-arm seam pairs (a from A, b from B) and the free/capped seams.
    std::vector<std::pair<ArmSeam<Real>, ArmSeam<Real>>> bent;
    std::vector<ArmSeam<Real>> freearm;
    if (lens) {
      bent.push_back({JA.seam(2), JB.seam(1)});   // top wall
      bent.push_back({JA.seam(1), JB.seam(2)});   // bottom wall
      freearm = {JA.seam(0), JB.seam(0)};         // inlet, outlet
    } else {
      bent.push_back({JA.seam(0), JB.seam(0)});   // the single tilted connection
      freearm = {JA.seam(1), JA.seam(2), JB.seam(1), JB.seam(2)};
    }

    // Emit bent arms; collect a sampled polyline of each for the exterior-source / collision checks.
    std::vector<ArmPoly<Real>> polys;
    for (auto& pr : bent) {
      const ArmSeam<Real>& sa = pr.first; const ArmSeam<Real>& sb = pr.second;
      const Real dx=sb.C[0]-sa.C[0], dy=sb.C[1]-sa.C[1], dz=sb.C[2]-sa.C[2];
      const Real len = std::sqrt(dx*dx+dy*dy+dz*dz);
      // Panel spacing ~1.5 R0 (finer than unit aspect) so the middle turn is well resolved; the straight
      // coaxial leads share the same spacing (cheap, and keeps the seam region straight+resolved).
      // ~1.5 R0 panel spacing (finer than unit aspect) so the shoulders resolve; the straight leads/run
      // share the same spacing. bent_centerline aligns the lead/corner/run panel boundaries by construction.
      const bool sc = !lens;   // tilt exercises the single-corner (lead|corner|run) arm; lens = 2-corner racetrack
      const Real pspac = (Real)1.5*std::max(sa.R0, sb.R0);
      const Integer nmin = 2*(leadP+cornerP) + 4;   // >=4 run panels (fits single 2*lead+corner or racetrack)
      const Integer ns = (nBentA > 0) ? nBentA : std::max<Integer>(nmin, (Integer)std::lround((double)len/pspac));
      A.add_bent_arm(sa, sb, ns, cheb, fourier, leadP, cornerP, sc);
      ArmPoly<Real> pl; pl.rtube = std::max(sa.R0, sb.R0);
      const Integer Nsamp = 400;
      for (Integer i = 0; i <= Nsamp; i++) pl.pts.push_back(HybridAssembly<Real>::bent_centerline(sa, sb, (Real)i/Nsamp, leadP, cornerP, ns, sc));
      polys.push_back(pl);
      if (!comm.Rank()) {
        auto ang = [](const Vec3<Real>& p, const Vec3<Real>& q){ Real d=p[0]*q[0]+p[1]*q[1]+p[2]*q[2]; d=std::max<Real>(-1,std::min<Real>(1,d)); return acos<Real>(d)*180/const_pi<Real>(); };
        const Vec3<Real> mb{-sb.u[0],-sb.u[1],-sb.u[2]};
        if (sc) {
          std::cout << "  [bent arm/single] chord=" << std::setprecision(4) << len << " panels=" << ns
                    << " (corner window " << cornerP << " panels auto-placed at Q; lead>=" << leadP
                    << ")  total-turn=" << ang(sa.u, mb) << "deg\n" << std::setprecision(6);
        } else {
          const Integer nrun = ns - 2*(leadP+cornerP);
          // run length = chord between the two shoulder vertices along the run line
          const Vec3<Real> r0 = HybridAssembly<Real>::bent_centerline(sa, sb, (Real)(leadP+cornerP)/ns, leadP, cornerP, ns);
          const Vec3<Real> r1 = HybridAssembly<Real>::bent_centerline(sa, sb, (Real)(ns-leadP-cornerP)/ns, leadP, cornerP, ns);
          const Real runlen = std::sqrt((r1[0]-r0[0])*(r1[0]-r0[0])+(r1[1]-r0[1])*(r1[1]-r0[1])+(r1[2]-r0[2])*(r1[2]-r0[2]));
          std::cout << "  [bent arm] chord=" << std::setprecision(4) << len << " panels=" << ns
                    << " (lead " << leadP << " | corner " << cornerP << " | run " << nrun << " | corner " << cornerP
                    << " | lead " << leadP << ")  run-len=" << runlen << " total-turn=" << ang(sa.u, mb) << "deg\n"
                    << std::setprecision(6);
        }
      }
    }
    // Free/capped arms (s_cap is the cap arc-station; scale==1 so it matches every seam's a0 span).
    for (const ArmSeam<Real>& s : freearm) A.add_free_arm(s, s_cap, nAxFree, Ncap, cheb, fourier);

    QuadElemList<Real> junc = A.quad(comm);
    SlenderElemList<Real> arms = A.slender(comm);
    const std::string tag = "vis/ybifurc-channel-" + mode + "-ord" + std::to_string((long)ord) + "-nref" + std::to_string((long)nref);

    // Free-arm axis segments for the collision / source checks.
    std::vector<std::pair<Vec3<Real>,Vec3<Real>>> freeseg;
    for (const ArmSeam<Real>& s : freearm) {
      const Real L = s_cap; // arc station of the cap ~ scale*s_cap; s.a0 already scaled(=1 here)
      freeseg.push_back({s.C, Vec3<Real>{s.C[0]+ (L-s.a0)*s.u[0], s.C[1]+(L-s.a0)*s.u[1], s.C[2]+(L-s.a0)*s.u[2]}});
    }

    // Collision guard (pre-BIE): min clearance between the two walls / between walls and free arms.
    if (!comm.Rank()) {
      Real cmin = std::numeric_limits<Real>::max();
      for (size_t i = 0; i < polys.size(); i++)
        for (size_t j = i+1; j < polys.size(); j++)
          cmin = std::min(cmin, poly_poly_dist<Real>(polys[i], polys[j]) - polys[i].rtube - polys[j].rtube);
      Real cfree = std::numeric_limits<Real>::max();
      for (const auto& pl : polys)
        for (const auto& fs : freeseg)
          for (const auto& p : pl.pts) cfree = std::min(cfree, seg_dist<Real>(p, fs.first, fs.second) - pl.rtube - JA.seam(0).R0);
      std::cout << "  [collision] min wall-wall clearance=" << std::setprecision(4) << cmin
                << "  min wall-free-arm clearance=" << cfree << std::setprecision(6) << "\n";
      if (cmin <= 0 || cfree <= 0) std::cout << "  [collision] WARNING: geometry may self-intersect!\n";
    }

    junc.WriteVTK(tag + "-junc", Vector<Real>(), comm);
    arms.WriteVTK(tag + "-arms", Vector<Real>(), comm);

    // Exterior Green sources (identity path only): one candidate per junction (canonical near-exterior
    // point), kept only if exterior to the WHOLE assembly -- f<level in BOTH junction frames AND clear of
    // every arm. The mfg path instead places an INTERIOR source (inside the tube) in run_manufactured.
    Vector<Real> X0;
    if (!mfg && !flow) {
      const YField<Real> fld;
      const Vec3<Real> cand[2] = {PA.apply_point(Vec3<Real>{1.6,1.4,0.9}), PB.apply_point(Vec3<Real>{1.6,1.4,0.9})};
      const Real rtube = JA.seam(0).R0 * (Real)1.3;
      for (int s = 0; s < 2; s++) {
        const Vec3<Real>& Xs = cand[s];
        const Real fA = fld.f(PA.apply_inverse_point(Xs)), fB = fld.f(PB.apply_inverse_point(Xs));
        Real darm = std::numeric_limits<Real>::max();
        for (const auto& pl : polys) darm = std::min(darm, poly_dist<Real>(Xs, pl));
        for (const auto& fs : freeseg) darm = std::min(darm, seg_dist<Real>(Xs, fs.first, fs.second));
        const bool inside = (fA >= level) || (fB >= level) || (darm <= rtube);
        if (!comm.Rank())
          std::cout << "  [source] junction " << s << " -> (" << std::setprecision(4) << Xs[0] << "," << Xs[1] << "," << Xs[2]
                    << ")  f_A=" << fA << " f_B=" << fB << " (<" << level << "?)  d_arm=" << darm << " (>" << rtube << "?)  => "
                    << (inside ? "INSIDE (dropped)" : "exterior OK (kept)") << std::setprecision(6) << "\n";
        if (!inside) { X0.PushBack(Xs[0]); X0.PushBack(Xs[1]); X0.PushBack(Xs[2]); }
      }
      SCTL_ASSERT_MSG(X0.Dim() >= 3, "no exterior Green source survived the interior/arm checks.");
    }

    if (geomOnly) {
      if (!comm.Rank()) std::cout << "  [geom-only] wrote " << tag << "-{junc,arms}.vtu; skipping BIE.\n";
      Comm::MPI_Finalize();
      return 0;
    }

    if (mfg) {
      run_manufactured<Real>(junc, arms, comm, level, PA, PB, sep, JA, polys, freeseg, tag, tol,
                             Nbeta, md, cov_q, SL_scal, DL_scal, Ngrid);
    } else if (flow) {
      run_flow<Real>(junc, arms, comm, JA, JB, s_cap, tag, tol, Nbeta, md, cov_q, p_in, p_out, gmresMax, Ngrid, SL_scal, DL_scal, Nvis);
    } else {
      const std::string label = lens ? "diverging-converging channel (2 junctions, 2 bent walls)"
                                      : ("two junctions + one bent arm (turn " + std::to_string((long)tiltDeg) + " deg)");
      run_verify<Real>(junc, arms, comm, X0, tag, label, tol, Nbeta, md, cov_q);
    }
  }
  Comm::MPI_Finalize();
  return 0;
}
