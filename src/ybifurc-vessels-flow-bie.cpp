/**
 * Physical Stokes inflow/outflow BVP on the 20-junction arterial/venous vascular network.
 *
 * Builds the SAME closed watertight network as src/ybifurc-vessels-bie.cpp (via the shared
 * build_vessels_network in quad_junctions/vessels_build.hpp) -- an arterial binary tree (root far left)
 * + mirror venous tree (root far right), 11 leaves joined across the middle, all planar in z=0 -- then
 * solves an INTERIOR Stokes velocity Dirichlet BVP on its single connected fluid interior:
 *
 *   - Parabolic INFLOW on the arterial-tree root cap (volumetric flux p_in).
 *   - Parabolic OUTFLOW on the venous-tree root cap (volumetric flux p_out).
 *   - No-slip (u = 0) everywhere else (all tube walls, junction bodies, leaf connectors).
 *
 * The two tree-root stems are the ONLY uncapped seams, so they are the only inflow/outflow ports. Each
 * cap's parabolic profile is normalized so its volumetric flux is exactly -+p; with p_in == p_out the net
 * flux is -p_in + p_out = 0, the compatibility condition int u.n dA = 0 required for the interior
 * incompressible Stokes Dirichlet problem. The driver computes each cap's flux from the far-field
 * quadrature weights and asserts the total is ~0. ("pressure-in/out" is the driving flux magnitude p --
 * the BC is a flux-normalized parabolic velocity, exactly as in src/ybifurc-flow-bie.cpp.)
 *
 * Formulation: combined-field ( -1/2 I - S + D ) sigma = u_bc via GMRES (Stokes SL/DL kernels), solved
 * by the shared solve_dirichlet_bvp() in hybrid_bie_tests.hpp. Outputs (VTU only, open in ParaView):
 *   - surface density sigma (3-component vector) on the quad list and the slender list;
 *   - the prescribed boundary velocity on the quad list (to see the inflow/outflow);
 *   - a z=0 planar slice of the interior velocity field, masked to the fluid interior via the Laplace
 *     DL constant-density indicator.
 *
 *   make bin/ybifurc-vessels-flow-bie                 # or: make MPI=1 bin/ybifurc-vessels-flow-bie
 *   OMP_NUM_THREADS=8 ./bin/ybifurc-vessels-flow-bie \
 *       [level] [order(mult4)] [nref] [eta_join] [Ns_trans] [fourier] [lead] [corner] \
 *       [tol] [Nbeta] [max_depth] [cov_q] [svg_scale] [p_in] [p_out] [Ngrid] [gmres_max_iter]
 */

#include <csbq.hpp>                                  // CSBQ SlenderElemList + CubeVolumeVis
#include <quad_junctions/vessels_build.hpp>          // shared build_vessels_network (+ dot3 etc.)
#include <quad_junctions/hybrid_bie_tests.hpp>       // combined_nodes + solve_dirichlet_bvp
#include <quad_junctions/vessels_tree_data.hpp>      // arterial/venous topology tables
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <limits>
#include <string>
#include <vector>

using namespace sctl;
using namespace quad_junctions;

namespace {

// One inflow/outflow cap: dome-equator center C, outward axis u, radius R0, signed amplitude `amp`
// (= sgn * p / g, where g is the geometric flux factor so that flux through the cap = sgn*p), and the
// prescribed flux magnitude p (sgn = -1 inflow / +1 outflow). (dot3 comes from vessels_build.hpp.)
template <class Real> struct FlowCap {
  Vec3<Real> C, u;
  Real R0 = 0, amp = 0, p = 0;
  int sgn = 0;
};

// Parabolic axial profile on a cap dome: prof(X) = 1 - (r/R0)^2 with r the transverse distance from the
// cap axis, for nodes lying on that cap's hemisphere (|X-C| ~ R0, on the +u side). Returns prof and the
// outward-axis component; 0 for any point not on a cap (no-slip walls/junctions/connectors).
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

// Prescribed boundary velocity at a point X: v = amp*prof*u on the owning cap, else 0.
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

// Write an Nx x Ny z=const grid (coords Xg, values Ug both AoS, Ug is a 3-vector/point) as a VTU of
// VTK_QUAD cells. Rank-0-only data, so written with Comm::Self().
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

} // anonymous namespace

int main(int argc, char** argv) {
  Comm::MPI_Init(&argc, &argv);
  using Real = double;
  {
    const Comm comm = Comm::World();
    const Real    level   = (argc > 1)  ? (Real)atof(argv[1])  : (Real)1.5;
    const Integer ord     = (argc > 2)  ? (Integer)atoi(argv[2])  : 12;
    const Integer nref    = (argc > 3)  ? (Integer)atoi(argv[3])  : 2;
    const Real    etajoin = (argc > 4)  ? (Real)atof(argv[4])  : (Real)0.4;
    const Integer NsTrans = (argc > 5)  ? (Integer)atoi(argv[5])  : 3;
    const Long    fourier = (argc > 6)  ? (Long)atoi(argv[6])  : 48;
    const Integer leadP   = (argc > 7)  ? (Integer)atoi(argv[7])  : 1;
    const Integer cornerP = (argc > 8)  ? (Integer)atoi(argv[8])  : 12;
    const Real    tol     = (argc > 9)  ? (Real)atof(argv[9])  : (Real)1e-9;
    const Integer Nbeta   = (argc > 10) ? (Integer)atoi(argv[10]) : 200;
    const Integer maxdep  = (argc > 11) ? (Integer)atoi(argv[11]) : 12;
    const Integer cov_q   = (argc > 12) ? (Integer)atoi(argv[12]) : 6;
    const Real    svgs    = (argc > 13) ? (Real)atof(argv[13]) : (Real)0.06;   // model units per SVG pixel
    const Real    p_in    = (argc > 14) ? (Real)atof(argv[14]) : (Real)10;     // arterial-root inflow flux
    const Real    p_out   = (argc > 15) ? (Real)atof(argv[15]) : (Real)10;     // venous-root outflow flux
    const Long    Ngrid   = (argc > 16) ? (Long)atoi(argv[16]) : 200;
    // GMRES iteration cap. NOT a cost knob -- the 400 default was reached in 12% of a 2 h allocation at
    // 2.19 s/iter. It matters because iteration count tracks the LOOP COUNT of the lumen: this network is
    // genus 10 (20 junctions, 18 intra-tree arms + 11 leaf connectors => E-V+1 = 10 independent cycles),
    // whereas the genus-1 racetrack lens of ybifurc-channel-bie needs 58 iterations with the same
    // slow-phase-then-converges shape. 400 sits below where a genus-10 problem starts to break through.
    const Long    gmaxit  = (argc > 17) ? (Long)atoi(argv[17]) : 400;
    const Integer Ncap    = (Integer)(YSwept::Ncap0 * std::max<Integer>(1, nref));
    const Long    cheb     = 10;
    const Integer nAxFree  = 3;
    const Real    tipLen   = (Real)3.0;   // root-cap free-arm length (x junction scale)
    pou_kind() = 1;                       // smootherstep POU (what the assembly transitions expect)

    SCTL_ASSERT_MSG(ord >= 4 && ord <= 48 && ord % 4 == 0, "order must be a multiple of 4 in {4,...,48}.");

    if (!comm.Rank()) {
      std::cout << "\n=== Stokes inflow/outflow BVP on the 20-junction vessels network ===\n";
      std::cout << "  order=" << ord << " level=" << level << " nref=" << nref << " eta_join=" << etajoin
                << " Ns_trans=" << NsTrans << " fourier=" << fourier << " lead=" << leadP << " corner=" << cornerP
                << " svg_scale=" << svgs << "\n";
      std::cout << "  near-eval: Hybrid(cov_q=" << cov_q << ", Nbeta=" << Nbeta << ", max_depth=" << maxdep
                << ") tol=" << std::setprecision(1) << tol << "  gmres_max_iter=" << gmaxit << "\n";
      std::cout << "  prescribed flux: arterial-root inflow p_in=" << std::setprecision(4) << p_in
                << "  venous-root outflow p_out=" << p_out << "  net=" << (-p_in + p_out) << "\n";
    }

    // ----------------------------------------------------------------------------------------------
    // (1) Build the whole network (same geometry as ybifurc-vessels-bie.cpp).
    // ----------------------------------------------------------------------------------------------
    HybridAssembly<Real> A(ord);
    const VesselsBuild<Real> vb = build_vessels_network<Real>(A, level, nref, etajoin, NsTrans, fourier,
        leadP, cornerP, svgs, Ncap, cheb, nAxFree, tipLen, comm);

    QuadElemList<Real> junc = A.quad(comm);
    SlenderElemList<Real> arms = A.slender(comm);
    junc.SetQuadScheme(QuadElemList<Real>::QuadScheme::Hybrid, cov_q, Nbeta, maxdep);
    const std::string tag = "vis/ybifurc-vessels-flow-ord" + std::to_string((long)ord) + "-nref" + std::to_string((long)nref);

    if (!comm.Rank()) {
      Vector<Real> Xj, Xa; junc.GetNodeCoord(&Xj, nullptr, nullptr); arms.GetNodeCoord(&Xa, nullptr, nullptr);
      std::cout << "\n[geometry] junctions=" << vessels_data::n_junc << " connectors=" << vessels_data::n_conn
                << " capped roots=" << vb.n_caps << "\n  quad panels=" << junc.Size() << " nodes=" << Xj.Dim()/3
                << " | slender panels=" << arms.Size() << " nodes=" << Xa.Dim()/3 << "\n";
    }

    // ----------------------------------------------------------------------------------------------
    // (2) The two capped root stems are the inflow/outflow ports. Arterial root (owner id < 10) = INFLOW,
    //     venous root (owner id >= 10) = OUTFLOW. Cap dome-equator center = seam.C + L*seam.u.
    // ----------------------------------------------------------------------------------------------
    SCTL_ASSERT_MSG(vb.cap_seams.size() == 2, "vessels network must have exactly two root caps (inlet/outlet).");
    std::vector<FlowCap<Real>> caps;
    for (size_t i = 0; i < vb.cap_seams.size(); i++) {
      const ArmSeam<Real>& s = vb.cap_seams[i];
      const Real L = vb.cap_len[i];
      FlowCap<Real> c;
      c.C  = Vec3<Real>{s.C[0] + L*s.u[0], s.C[1] + L*s.u[1], s.C[2] + L*s.u[2]};
      c.u  = s.u; c.R0 = s.R0;
      if (vb.cap_owner[i] < 10) { c.sgn = -1; c.p = p_in;  }   // arterial-tree root = inflow
      else                      { c.sgn = +1; c.p = p_out; }   // venous-tree   root = outflow
      caps.push_back(c);
    }
    if (!comm.Rank()) {
      std::cout << "\n  [caps] " << caps.size() << " root-stem ports (R0=" << std::setprecision(6) << caps[0].R0 << ")\n";
      for (size_t i = 0; i < caps.size(); i++)
        std::cout << "    cap " << i << " (junc " << vb.cap_owner[i] << "): center=(" << std::setprecision(4)
                  << caps[i].C[0] << "," << caps[i].C[1] << "," << caps[i].C[2] << ")  axis=(" << caps[i].u[0]
                  << "," << caps[i].u[1] << "," << caps[i].u[2] << ")  "
                  << (caps[i].sgn < 0 ? "INFLOW" : "OUTFLOW") << " p=" << caps[i].p << "\n";
    }

    // ----------------------------------------------------------------------------------------------
    // (3) Geometric flux factor g_c = int prof*(u.n) dA per cap (far-field quadrature), then amplitude
    //     amp_c = sgn_c * p_c / g_c so the signed flux through cap c is exactly sgn_c * p_c.
    // ----------------------------------------------------------------------------------------------
    {
      Vector<Real> Xf, Xnf, wts, dist; Vector<Long> cnt;
      junc.GetFarFieldNodes(Xf, Xnf, wts, dist, cnt, tol);   // caps live in the quad list
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
    // (4) Assemble the boundary velocity RHS over the combined "0_junc"+"1_arms" node ordering, and
    //     VERIFY the flux: int v.n dA per cap (should be -+p) and the total (should be ~0).
    // ----------------------------------------------------------------------------------------------
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
        std::cout << "\n  [flux check] int v.n dA per cap (target -+p):\n";
        for (size_t c = 0; c < caps.size(); c++)
          std::cout << "    cap " << c << " (" << (caps[c].sgn < 0 ? "in " : "out") << "): flux="
                    << std::setprecision(6) << flux[(Long)c] << "  (target " << (Real)caps[c].sgn*caps[c].p << ")\n";
        std::cout << "    TOTAL net flux = " << total << "  (compatibility condition int u.n dA = 0)\n";
      }
      SCTL_ASSERT_MSG(std::fabs((double)total) < 1e-6, "net flux not zero -- interior Stokes BVP incompatible");
    }

    // ----------------------------------------------------------------------------------------------
    // (5) Interior sampling grid on the z=0 plane (rank 0 only). Bounds are the xy bounding box of the
    //     whole surface (both lists) plus a small margin, so the slice adapts to svg_scale.
    //     GetNodeCoord returns this rank's LOCAL element slice, so the bbox MUST be reduced across the
    //     comm -- computing it under `if (!comm.Rank())` framed the slice on rank 0's 1/nranks of the
    //     panels and silently cropped the network (at 8 ranks: x[-16.4,-9.8] out of x[-19.4,20.9]).
    //     Hence this block is UNGUARDED and collective; only the grid construction below is rank-0.
    //     Same pattern (and same comment) as ybifurc-channel-bie.cpp.
    // ----------------------------------------------------------------------------------------------
    Real xlo = std::numeric_limits<Real>::max(), xhi = -xlo, ylo = xlo, yhi = -xlo;
    Long Nx = 0, Ny = 0;
    Vector<Real> Xgrid;
    {
      Vector<Real> Xj, Xa; junc.GetNodeCoord(&Xj, nullptr, nullptr); arms.GetNodeCoord(&Xa, nullptr, nullptr);
      auto acc = [&](const Vector<Real>& XX) {
        for (Long i = 0; i < XX.Dim()/3; i++) {
          xlo = std::min(xlo, XX[3*i]); xhi = std::max(xhi, XX[3*i]);
          ylo = std::min(ylo, XX[3*i+1]); yhi = std::max(yhi, XX[3*i+1]);
        }
      };
      acc(Xj); acc(Xa);
      xlo = GlobalReduce((double)xlo, comm, CommOp::MIN); xhi = GlobalReduce((double)xhi, comm, CommOp::MAX);
      ylo = GlobalReduce((double)ylo, comm, CommOp::MIN); yhi = GlobalReduce((double)yhi, comm, CommOp::MAX);
      // Per-axis margin: one shared x-derived margin would inflate y by 5% of the (much wider) x extent.
      const Real mx = (Real)0.05*(xhi-xlo), my = (Real)0.05*(yhi-ylo);
      xlo -= mx; xhi += mx; ylo -= my; yhi += my;
    }
    Nx = Ngrid;
    Ny = std::max<Long>(2, (Long)std::lround((double)Ngrid*(yhi-ylo)/(xhi-xlo)));
    if (!comm.Rank()) {
      for (Long ix = 0; ix < Nx; ix++) {
        const Real xx = xlo + (xhi-xlo)*ix/(Nx-1);
        for (Long iy = 0; iy < Ny; iy++) {
          const Real yy = ylo + (yhi-ylo)*iy/(Ny-1);
          Xgrid.PushBack(xx); Xgrid.PushBack(yy); Xgrid.PushBack((Real)0);
        }
      }
      std::cout << "\n  [grid] z=0 slice " << Nx << " x " << Ny << " = " << Nx*Ny << " points over x["
                << std::setprecision(4) << xlo << "," << xhi << "] y[" << ylo << "," << yhi << "]\n";
    }
    const Long Ngrid_pts = Xgrid.Dim()/3;

    // ----------------------------------------------------------------------------------------------
    // (6) Solve the interior Stokes Dirichlet BVP (SL_scal=-1, DL_scal=+1 -> jump=-1/2) and evaluate the
    //     represented velocity field at the grid points.
    // ----------------------------------------------------------------------------------------------
    // DISABLED 2026-07-29: block-diagonal left preconditioner on the junction rows. The machinery is
    // still there and inert (quad_junctions/junction_precond.hpp; solve_dirichlet_bvp's `precond`
    // argument defaults to nullptr), so re-enabling is just uncommenting this block and passing
    // &pspec again. Left off because the whole-junction block is 3*npj*order^2 DOF = 82,944 at
    // order 12 / nref 1 / Ns_trans 2 -> 55 GB dense and a ~3 h SVD; QJ_PRECOND_BLOCK selects
    // junction / panel / off when it is back in play.
    // JunctionPrecondSpec<Real> pspec;
    // pspec.kind     = precond_block_kind();
    // pspec.order    = ord;
    // pspec.level    = level;
    // pspec.nref     = nref;
    // pspec.eta_join = etajoin;
    // pspec.Ns_trans = NsTrans;
    // pspec.njunc    = (Long)vb.P.size();

    Vector<Real> Ugrid;
    const Vector<Real> sigma = solve_dirichlet_bvp<Real, Stokes3D_FxU, Stokes3D_DxU>(
        junc, arms, comm, tol, bc, /*interior=*/true, /*SL_scal=*/(Real)1., /*DL_scal=*/(Real)1.,
        Xgrid, &Ugrid, "stokes inflow/outflow", /*gmres_max_iter=*/gmaxit);

    // ----------------------------------------------------------------------------------------------
    // (7) Interior mask: Laplace DL constant-density indicator (~ -1 interior, ~0 exterior) at the grid.
    // ----------------------------------------------------------------------------------------------
    {
      BoundaryIntegralOp<Real, Laplace3D_DxU> IndOp((Laplace3D_DxU()), false, comm);
      SetPVFMMKer(IndOp);
      IndOp.SetAccuracy(tol);
      IndOp.AddElemList(junc, "0_junc"); IndOp.AddElemList(arms, "1_arms");
      Vector<Real> ones(Nnode); ones = 1;
      IndOp.SetTargetCoord(Xgrid);
      Vector<Real> ind;
      IndOp.ComputePotential(ind, ones);
      if (!comm.Rank()) {
        Long n_in = 0;
        for (Long i = 0; i < Ngrid_pts; i++) {
          if (std::fabs((double)ind[i]) > 0.5) n_in++;
          else { Ugrid[3*i] = 0; Ugrid[3*i+1] = 0; Ugrid[3*i+2] = 0; }   // zero out exterior
        }
        std::cout << "  [grid] interior points (|DL indicator|>0.5): " << n_in << " / " << Ngrid_pts << "\n";
      }
    }

    // ----------------------------------------------------------------------------------------------
    // (8) Output (VTU only). Surface density (3-vec) + prescribed BC velocity + interior flow slice.
    // ----------------------------------------------------------------------------------------------
    {
      Vector<Real> sj(Nj*3), sa_(Na*3), bcj(Nj*3);
      for (Long i = 0; i < Nj*3; i++) { sj[i] = sigma[i]; bcj[i] = bc[i]; }
      for (Long i = 0; i < Na*3; i++) sa_[i] = sigma[Nj*3 + i];
      junc.WriteVTK(tag + "-sigma-junc", sj,  comm);   // collective
      arms.WriteVTK(tag + "-sigma-arms", sa_, comm);
      junc.WriteVTK(tag + "-bc-junc",    bcj, comm);   // prescribed inflow/outflow velocity
      if (!comm.Rank()) {
        write_plane_vtu<Real>(tag + "-flow-slice", Xgrid, Ugrid, Nx, Ny);
        std::cout << "\n  [dump] " << tag << "-sigma-{junc,arms}.pvtu (density), " << tag
                  << "-bc-junc.pvtu (BC), " << tag << "-flow-slice.vtu (z=0 interior velocity)\n";
      }
    }
  }
  Comm::MPI_Finalize();
  return 0;
}
