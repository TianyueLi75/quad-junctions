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
 *       [tol] [Nbeta] [max_depth] [cov_q] [svg_scale] [p_in] [p_out] [Ngrid] [gmres_max_iter] [Nvis] \
 *       [gscale] [sphere_deg] [sphere_tilt]
 *
 * The last three (argv 19-21, default 1/0/0 = planar) match ybifurc-vessels-bie.cpp: gscale is a global
 * similarity scale, sphere_deg>0 drapes the whole network rigidly onto a sphere spanning that many arc
 * degrees (0 = flat), sphere_tilt nudges the two middle connectors apart. The inflow/outflow solve, cap
 * BCs, Poiseuille network, DL mask, and 3D point-cloud viz are all geometry-agnostic, so the SAME pin/pout
 * problem runs unchanged on the draped surface (the old z=0 slice is gone; the 3D cloud handles non-planar).
 *
 * Interior-flow visualization (quad_junctions/interior_viz.hpp): a VTU point cloud sampled where the fluid
 * is -- arm cross-section stars at each CSBQ panel's first Chebyshev node (3 radial x 5 azimuthal + the
 * centerline) + an Nvis^3 uniform box per junction body kept to the interior. Nvis (argv 18) is the
 * junction-box per-axis sample count; 0 => cbrt(Ngrid). For the full 20-junction network, Nvis ~ 10-12 is
 * a good balance (10^3/junction ~= 1e3 candidates, ~1/3 interior -> ~6-10k junction points on top of the
 * arm stars); the plain cbrt(80)=4 the submit script implies is too sparse for a useful junction cloud.
 *
 * Verification (after the GMRES solve): a reduced Hagen-Poiseuille RESISTANCE-NETWORK solve
 * (quad_junctions/vessels_network_solve.hpp) predicts the pressure at every junction and the flux of
 * each connector arm in terms of p_in; the driver then probes the BIE velocity across the mid-section of
 * each of the 11 arterial<->venous connector arms and reports (a) the fully-developed-Poiseuille profile
 * shape error vs the BIE-MEASURED cross-section flux (an exact Stokes check) and (b) the BIE-measured
 * flux vs the network-predicted flux (validating the reduced model + global conservation).
 */

#include <csbq.hpp>                                  // CSBQ SlenderElemList + CubeVolumeVis
#include <quad_junctions/vessels_build.hpp>          // shared build_vessels_network (+ dot3 etc.)
#include <quad_junctions/vessels_network_solve.hpp>  // reduced Hagen-Poiseuille resistance-network solve
#include <quad_junctions/interior_viz.hpp>           // build_arm_panel_targets / build_box_targets (interior viz)
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

// Write an Nx x Ny x Nz Cartesian box (coords Xg, values Ug both AoS, Ug a 3-vector/point, index order
// (ix*Ny+iy)*Nz+iz) as a VTU of VTK_HEXAHEDRON cells. Rank-0-only data, written with Comm::Self(). Cell
// connectivity mirrors CubeVolumeVisShifted::GetVTUData in periodic_flow_utils.hpp (the unit-cell volume
// visualizer); we roll our own here because that class is hardcoded to a cube centered at (0.5,0.5,0.5),
// wrong for this off-origin elongated slab.
template <class Real>
void write_box_vtu(const std::string& fname, const Vector<Real>& Xg, const Vector<Real>& Ug,
                   const Long Nx, const Long Ny, const Long Nz) {
  VTUData vtu;
  const Long Np = Nx*Ny*Nz;
  for (Long i = 0; i < Np; i++)
    for (Integer k = 0; k < 3; k++) vtu.coord.PushBack((VTUData::VTKReal)Xg[3*i+k]);
  for (Long i = 0; i < Np; i++)
    for (Integer k = 0; k < 3; k++) vtu.value.PushBack((VTUData::VTKReal)Ug[3*i+k]);
  auto idx = [Ny, Nz](Long ix, Long iy, Long iz) -> int32_t { return (int32_t)((ix*Ny + iy)*Nz + iz); };
  int32_t off = 0;
  for (Long ix = 0; ix < Nx-1; ix++)
    for (Long iy = 0; iy < Ny-1; iy++)
      for (Long iz = 0; iz < Nz-1; iz++) {
        vtu.connect.PushBack(idx(ix,   iy,   iz  )); vtu.connect.PushBack(idx(ix,   iy,   iz+1));
        vtu.connect.PushBack(idx(ix,   iy+1, iz+1)); vtu.connect.PushBack(idx(ix,   iy+1, iz  ));
        vtu.connect.PushBack(idx(ix+1, iy,   iz  )); vtu.connect.PushBack(idx(ix+1, iy,   iz+1));
        vtu.connect.PushBack(idx(ix+1, iy+1, iz+1)); vtu.connect.PushBack(idx(ix+1, iy+1, iz  ));
        off += 8; vtu.offset.PushBack(off); vtu.types.PushBack((uint8_t)12);  // 12 = VTK_HEXAHEDRON
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
    const Long    Nvis    = (argc > 18) ? (Long)atoi(argv[18]) : 0;   // junction-box per-axis samples; 0 = cbrt(Ngrid)
    const Real    gscale    = (argc > 19) ? (Real)atof(argv[19]) : (Real)1;   // global similarity scale
    const Real    sphereDeg = (argc > 20) ? (Real)atof(argv[20]) : (Real)0;   // 0=planar; >0 drapes onto a sphere (arc deg)
    const Real    sphereTilt= (argc > 21) ? (Real)atof(argv[21]) : (Real)0;   // tilt the 2 middle connectors +/- deg
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
                << " svg_scale=" << svgs << " gscale=" << gscale
                << (sphereDeg > 0 ? "  [SPHERE-DRAPED " + std::to_string((long)sphereDeg) + " deg, tilt "
                                    + std::to_string((long)sphereTilt) + "]" : "  [planar]") << "\n";
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
        leadP, cornerP, svgs, Ncap, cheb, nAxFree, tipLen, comm, gscale, sphereDeg, sphereTilt);

    QuadElemList<Real> junc = A.quad(comm);
    SlenderElemList<Real> arms = A.slender(comm);
    junc.SetQuadScheme(QuadElemList<Real>::QuadScheme::Hybrid, cov_q, Nbeta, maxdep);
    const std::string tag = "vis/ybifurc-vessels-flow-ord" + std::to_string((long)ord) + "-nref" + std::to_string((long)nref)
        + (sphereDeg > 0 ? "-sph" + std::to_string((long)sphereDeg) : "");   // distinct tag so sph90 doesn't clobber planar

    {
      // GLOBAL panel/node counts: GetNodeCoord/Size() return this rank's LOCAL slice, so sum across the comm
      // (GlobalReduce is collective -- called on every rank, printed on rank 0). Without this the counts were
      // per-rank (rank 0's ~1/nranks share), misleading under MPI.
      Vector<Real> Xj, Xa; junc.GetNodeCoord(&Xj, nullptr, nullptr); arms.GetNodeCoord(&Xa, nullptr, nullptr);
      const Long njp = GlobalReduce((Long)junc.Size(),   comm, CommOp::SUM);
      const Long nap = GlobalReduce((Long)arms.Size(),   comm, CommOp::SUM);
      const Long njn = GlobalReduce((Long)(Xj.Dim()/3),  comm, CommOp::SUM);
      const Long nan = GlobalReduce((Long)(Xa.Dim()/3),  comm, CommOp::SUM);
      if (!comm.Rank())
        std::cout << "\n[geometry] junctions=" << vessels_data::n_junc << " connectors=" << vessels_data::n_conn
                  << " capped roots=" << vb.n_caps << " (global counts, summed over " << comm.Size() << " ranks)"
                  << "\n  quad panels=" << njp << " nodes=" << njn
                  << " | slender panels=" << nap << " nodes=" << nan
                  << " | TOTAL panels=" << (njp+nap) << " nodes=" << (njn+nan) << "\n";
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
    // (5) Interior visualization targets, sampled where the fluid actually is (a coarse uniform box over
    //     the whole ~44-unit domain never lands inside the r~0.15 tubes). Two geometry-aware groups
    //     (quad_junctions/interior_viz.hpp):
    //       ARMS -- at the FIRST Chebyshev node of every CSBQ slender panel, a cross-section star:
    //               centerline point + 3 radial x 5 azimuthal interior points at radii {1,2,3}*r/5 (the
    //               outermost shell at 3/5 r, clear of the wall). GetGeom sees each rank's LOCAL panels,
    //               so build_arm_panel_targets is COLLECTIVE (Allgatherv) -- it MUST run on every rank.
    //       JUNCTIONS -- an Nax^3 uniform grid in a cube around each junction body (bulky enough for a box
    //               to resolve); kept only where the Laplace-DL indicator says interior (the "check inside"
    //               step 7). Nax = cbrt(Ngrid) by default, or the argv-18 override.
    //     Arm points are interior by construction and are always kept.
    // ----------------------------------------------------------------------------------------------
    Vector<Real> Xarm;
    build_arm_panel_targets<Real>(arms, comm, cheb, Xarm);   // collective: per-rank panels gathered to all
    Long Narm = 0, Njunc = 0, Nax = 0;
    Vector<Real> Xgrid;
    if (!comm.Rank()) {
      for (Long i = 0; i < Xarm.Dim(); i++) Xgrid.PushBack(Xarm[i]);
      Narm = Xgrid.Dim()/3;
      // Per-junction cube: center = placement origin; half-width = farthest attached arm-seam ring + 15%.
      const int NJ = vessels_data::n_junc;
      Vector<Real> jc(3*NJ), jh(NJ); jh = 0;
      for (int i = 0; i < NJ; i++) {
        const Vec3<Real> c = vb.P[(size_t)i].apply_point(Vec3<Real>{0,0,0});
        jc[3*i] = c[0]; jc[3*i+1] = c[1]; jc[3*i+2] = c[2];
      }
      for (const ArmSeg<Real>& seg : vb.segs) {
        if (seg.cl.empty()) continue;
        if (seg.j0 >= 0 && seg.j0 < NJ) {
          const Vec3<Real> c{jc[3*seg.j0], jc[3*seg.j0+1], jc[3*seg.j0+2]};
          jh[seg.j0] = std::max(jh[seg.j0], nrm3(sub3(seg.cl.front(), c)));
        }
        if (seg.j1 >= 0 && seg.j1 < NJ && seg.j1 != seg.j0) {
          const Vec3<Real> c{jc[3*seg.j1], jc[3*seg.j1+1], jc[3*seg.j1+2]};
          jh[seg.j1] = std::max(jh[seg.j1], nrm3(sub3(seg.cl.back(), c)));
        }
      }
      for (int i = 0; i < NJ; i++) jh[i] = (jh[i] > 0) ? (Real)1.15*jh[i] : (Real)3*caps[0].R0;
      Nax = (Nvis > 0) ? Nvis : std::max<Long>(3, (Long)std::lround(std::cbrt((double)Ngrid)));
      Vector<Real> Xjb;
      build_box_targets<Real>(jc, jh, Nax, Xjb);
      for (Long i = 0; i < Xjb.Dim(); i++) Xgrid.PushBack(Xjb[i]);
      Njunc = Xgrid.Dim()/3 - Narm;
      std::cout << "\n  [viz] interior targets: " << Narm << " arm cross-section (" << (Narm/16)
                << " panels x 16) + " << Njunc << " junction-box (" << NJ << " x " << Nax
                << "^3) = " << (Narm+Njunc) << " (junctions filtered to interior in step 7)\n";
    }
    const Long Ngrid_pts = Xgrid.Dim()/3;

    // ----------------------------------------------------------------------------------------------
    // (5b) Reduced Hagen-Poiseuille RESISTANCE-NETWORK solve: predict the pressure at every junction and
    //      the flux through each connector arm in terms of p_in (an analytic reference for the BIE flow).
    //      Then, on rank 0, APPEND mid-arm probe targets (a cross-section disk to integrate the BIE flux,
    //      + a radial line for the profile) to Xgrid so they ride the SAME operator eval as the box.
    // ----------------------------------------------------------------------------------------------
    int root_in = -1, root_out = -1;
    for (size_t i = 0; i < vb.cap_owner.size(); i++)
      (vb.cap_owner[i] < 10 ? root_in : root_out) = vb.cap_owner[i];
    const NetworkSolution<Real> net =
        solve_vessels_pressure_network<Real>(vb.segs, vessels_data::n_junc, root_in, root_out, p_in);

    // Per-connector probe metadata (rank 0 only; carried to the post-solve check in step 9).
    struct Probe {
      int cidx; Real r, Qpred; Vec3<Real> mid, t;
      Long disk_off, disk_n; std::vector<Real> disk_w;    // disk points: flux = sum (u.t)*w
      Long line_off, line_n; std::vector<Real> line_rho;  // radial line points (signed rho along e1)
    };
    std::vector<Probe> probes;
    const Long Nring = 8, Nang = 16, Nrad = 21;

    if (!comm.Rank()) {
      Real Qsum = 0; int nconn = 0;
      std::cout << "\n  [network] reduced Hagen-Poiseuille resistance solve (mu=1; pressures ~ p_in=" << p_in << "):\n";
      std::cout << "    junction pressures P[0.." << (vessels_data::n_junc-1) << "] (mean-zero):\n     ";
      for (int i = 0; i < vessels_data::n_junc; i++)
        std::cout << " " << std::setprecision(4) << net.P[(size_t)i] << (i%10==9 ? "\n     " : "");
      std::cout << "\n    arterial<->venous connectors (arm i:  jA->jV   r      L       dP        Q_pred):\n";
      for (const NetEdge<Real>& e : net.edges) {
        if (!e.is_conn) continue;
        Qsum += e.Q; nconn++;
        std::cout << "      conn " << std::setw(2) << nconn-1 << ":  " << std::setw(2) << e.j0 << "->"
                  << std::setw(2) << e.j1 << "  " << std::setprecision(4) << e.r << "  " << std::setw(7) << e.L
                  << "  " << std::setw(9) << e.dP << "  " << std::setw(9) << e.Q << "\n";

        // Build the mid-arm cross-section frame from the connector's centerline polyline.
        const std::vector<Vec3<Real>>& cl = vb.segs[(size_t)e.seg].cl;
        if ((Long)cl.size() < 3) continue;
        const Long m = (Long)cl.size()/2;
        const Vec3<Real> mid = cl[m];
        Vec3<Real> t = unit3(sub3(cl[m+1], cl[m-1]));
        Vec3<Real> ref = (std::fabs((double)t[2]) < 0.9) ? Vec3<Real>{0,0,1} : Vec3<Real>{1,0,0};
        const Vec3<Real> e1 = unit3(sub3(ref, mul3(dot3(ref,t), t)));
        const Vec3<Real> e2 = Vec3<Real>{t[1]*e1[2]-t[2]*e1[1], t[2]*e1[0]-t[0]*e1[2], t[0]*e1[1]-t[1]*e1[0]};
        Probe pr; pr.cidx = nconn-1; pr.r = e.r; pr.Qpred = e.Q; pr.mid = mid; pr.t = t;

        // Cross-section disk (polar midpoint rule): rho_i=(i+0.5)*r/Nr, weight = rho_i*(r/Nr)*(2pi/Na).
        pr.disk_off = Xgrid.Dim()/3; pr.disk_n = Nring*Nang;
        const Real dr = e.r/(Real)Nring, dth = (Real)2*const_pi<Real>()/(Real)Nang;
        for (Long ir = 0; ir < Nring; ir++) {
          const Real rho = ((Real)ir + (Real)0.5)*dr;
          for (Long ia = 0; ia < Nang; ia++) {
            const Real th = (Real)ia*dth, cph = std::cos((double)th), sph = std::sin((double)th);
            const Vec3<Real> X = add3(mid, add3(mul3(rho*cph, e1), mul3(rho*sph, e2)));
            Xgrid.PushBack(X[0]); Xgrid.PushBack(X[1]); Xgrid.PushBack(X[2]);
            pr.disk_w.push_back(rho*dr*dth);
          }
        }
        // Radial line along e1 across the diameter (for the CSV profile plot).
        pr.line_off = Xgrid.Dim()/3; pr.line_n = Nrad;
        for (Long j = 0; j < Nrad; j++) {
          const Real rho = ((Real)-0.9 + (Real)1.8*j/(Nrad-1))*e.r;
          const Vec3<Real> X = add3(mid, mul3(rho, e1));
          Xgrid.PushBack(X[0]); Xgrid.PushBack(X[1]); Xgrid.PushBack(X[2]);
          pr.line_rho.push_back(rho);
        }
        probes.push_back(pr);
      }
      std::cout << "    sum of connector fluxes Q = " << std::setprecision(6) << Qsum
                << "  (target p_in=" << p_in << ", global conservation)\n";
      std::cout << "    mid-arm probes: " << nconn << " connectors x (" << Nring << "x" << Nang
                << " disk + " << Nrad << " line) = " << (Long)(Xgrid.Dim()/3 - Ngrid_pts) << " points appended\n";
    }

    // ----------------------------------------------------------------------------------------------
    // (6) Solve the interior Stokes Dirichlet BVP (SL_scal=-1, DL_scal=+1 -> jump=-1/2) and evaluate the
    //     represented velocity field at the grid + mid-arm probe points.
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
        junc, arms, comm, tol, bc, /*interior=*/true, /*SL_scal=*/(Real)-1., /*DL_scal=*/(Real)1.,
        Xgrid, &Ugrid, "stokes inflow/outflow", /*gmres_max_iter=*/gmaxit);

    // ----------------------------------------------------------------------------------------------
    // (7) Interior filter: Laplace DL constant-density indicator (~ -1 interior, ~0 exterior) at the viz
    //     targets. Arm cross-section points are interior by construction and always kept; junction-box
    //     candidates are kept only where |indicator|>0.5. Build the interior-only cloud (Xvis,Uvis).
    // ----------------------------------------------------------------------------------------------
    Vector<Real> Xvis, Uvis;
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
        Long n_junc_in = 0;
        for (Long i = 0; i < Narm; i++)                          // arms: interior by construction
          for (Integer k = 0; k < 3; k++) { Xvis.PushBack(Xgrid[3*i+k]); Uvis.PushBack(Ugrid[3*i+k]); }
        for (Long i = Narm; i < Ngrid_pts; i++)                  // junction box: keep interior only
          if (std::fabs((double)ind[i]) > 0.5) {
            for (Integer k = 0; k < 3; k++) { Xvis.PushBack(Xgrid[3*i+k]); Uvis.PushBack(Ugrid[3*i+k]); }
            n_junc_in++;
          }
        std::cout << "  [grid] kept " << Narm << " arm + " << n_junc_in << " / " << Njunc
                  << " junction-box interior points = " << (Xvis.Dim()/3) << " total\n";
      }
    }

    // ----------------------------------------------------------------------------------------------
    // (8) Output (VTU only). Surface density (3-vec) + prescribed BC velocity + 3D interior velocity box.
    // ----------------------------------------------------------------------------------------------
    {
      Vector<Real> sj(Nj*3), sa_(Na*3), bcj(Nj*3);
      for (Long i = 0; i < Nj*3; i++) { sj[i] = sigma[i]; bcj[i] = bc[i]; }
      for (Long i = 0; i < Na*3; i++) sa_[i] = sigma[Nj*3 + i];
      junc.WriteVTK(tag + "-sigma-junc", sj,  comm);   // collective
      arms.WriteVTK(tag + "-sigma-arms", sa_, comm);
      junc.WriteVTK(tag + "-bc-junc",    bcj, comm);   // prescribed inflow/outflow velocity
      if (!comm.Rank()) {
        // Interior-only cloud built in step 7: arm cross-section stars + junction-box points inside the
        // surface. The mid-arm Poiseuille probes (appended after Ngrid_pts) are never added to Xvis.
        write_points_vtu<Real>(tag + "-flow-box", Xvis, Uvis, Xvis.Dim()/3);
        std::cout << "\n  [dump] " << tag << "-sigma-{junc,arms}.pvtu (density), " << tag
                  << "-bc-junc.pvtu (BC), " << tag << "-flow-box.vtu (interior velocity: arm stars + junction boxes)\n";
      }
    }

    // ----------------------------------------------------------------------------------------------
    // (9) Mid-arm Poiseuille verification (rank 0). For each connector the BIE probe velocities are the
    //     tail of Ugrid. Integrate the cross-section flux Q_bie = sum (u.t)*w over the disk, then compare
    //     the BIE axial profile to the fully-developed Poiseuille of that MEASURED flux
    //     u_axial(rho)=2*Q_bie/(pi r^2)*(1-(rho/r)^2) (an exact Stokes check, independent of the reduced
    //     model), and separately report Q_bie vs the network-PREDICTED flux Q_pred (validates the model).
    //     Also dump the radial profile per connector to a CSV.
    // ----------------------------------------------------------------------------------------------
    if (!comm.Rank() && !probes.empty()) {
      const std::string csv = tag + "-poiseuille.csv";
      FILE* fp = fopen(csv.c_str(), "w");
      if (fp) fprintf(fp, "conn,rho,u_axial_bie,u_transverse_bie,u_axial_measured,u_axial_predicted\n");
      Real worst_relL2 = 0, worst_flux_rel = 0;
      std::cout << "\n  [poiseuille] mid-arm BIE velocity vs fully-developed Hagen-Poiseuille (per connector):\n"
                << "    conn   Q_bie      Q_pred    |dQ|/Q_pred   relL2(shape)  maxTransv/umax\n";
      for (const Probe& pr : probes) {
        // (a) measured flux over the disk.
        Real Qbie = 0;
        for (Long k = 0; k < pr.disk_n; k++) {
          const Long i = pr.disk_off + k;
          const Real ut = Ugrid[3*i]*pr.t[0] + Ugrid[3*i+1]*pr.t[1] + Ugrid[3*i+2]*pr.t[2];
          Qbie += ut*pr.disk_w[(size_t)k];
        }
        const Real umax_m = (Real)2*Qbie/(const_pi<Real>()*pr.r*pr.r);   // measured-flux Poiseuille peak
        // (b) shape error over the disk vs measured-flux Poiseuille + max transverse speed.
        Real e2 = 0, ref2 = 0, tmax = 0;
        for (Long k = 0; k < pr.disk_n; k++) {
          const Long i = pr.disk_off + k;
          const Real ux = Ugrid[3*i], uy = Ugrid[3*i+1], uz = Ugrid[3*i+2];
          const Real ut = ux*pr.t[0] + uy*pr.t[1] + uz*pr.t[2];
          const Real rho2 = ux*ux+uy*uy+uz*uz - ut*ut;               // transverse speed^2
          tmax = std::max<Real>(tmax, sqrt<Real>(rho2 < 0 ? 0 : rho2));
          // radius of this disk point from ring index: rho = ((k/Nang)+0.5)*r/Nring
          const Real rho = (((Real)(k/Nang)) + (Real)0.5)*pr.r/(Real)Nring;
          const Real uex = umax_m*((Real)1 - (rho*rho)/(pr.r*pr.r));
          const Real d = ut - uex; e2 += d*d; ref2 += uex*uex;
        }
        const Real relL2 = (ref2 > 0) ? sqrt<Real>(e2/ref2) : (Real)0;
        const Real fluxrel = (std::fabs((double)pr.Qpred) > 0) ? std::fabs((double)(Qbie-pr.Qpred))/std::fabs((double)pr.Qpred) : (Real)0;
        const Real umaxabs = std::fabs((double)umax_m) > 0 ? std::fabs((double)umax_m) : (Real)1;
        worst_relL2 = std::max<Real>(worst_relL2, relL2);
        worst_flux_rel = std::max<Real>(worst_flux_rel, fluxrel);
        std::cout << "    " << std::setw(3) << pr.cidx << "  " << std::setprecision(4) << std::setw(9) << Qbie
                  << "  " << std::setw(9) << pr.Qpred << "  " << std::setw(9) << fluxrel
                  << "     " << std::setw(9) << relL2 << "    " << std::setw(9) << tmax/umaxabs << "\n";
        // (c) radial-line profile to CSV: measured- and predicted-flux Poiseuille.
        const Real umax_p = (Real)2*pr.Qpred/(const_pi<Real>()*pr.r*pr.r);
        for (Long j = 0; j < pr.line_n; j++) {
          const Long i = pr.line_off + j;
          const Real ux = Ugrid[3*i], uy = Ugrid[3*i+1], uz = Ugrid[3*i+2];
          const Real ut = ux*pr.t[0] + uy*pr.t[1] + uz*pr.t[2];
          const Real rho = pr.line_rho[(size_t)j];
          const Real u_meas = umax_m*((Real)1 - (rho*rho)/(pr.r*pr.r));
          const Real u_pred = umax_p*((Real)1 - (rho*rho)/(pr.r*pr.r));
          const Real utrans = sqrt<Real>(std::max<Real>(0, ux*ux+uy*uy+uz*uz - ut*ut));
          if (fp) fprintf(fp, "%d,%.6f,%.9e,%.9e,%.9e,%.9e\n", pr.cidx, (double)rho, (double)ut,
                          (double)utrans, (double)u_meas, (double)u_pred);
        }
      }
      if (fp) fclose(fp);
      std::cout << "    worst-case over connectors: relL2(shape)=" << std::setprecision(4) << worst_relL2
                << "  |dQ|/Q_pred=" << worst_flux_rel << "  (wrote " << csv << ")\n";
    }
  }
  Comm::MPI_Finalize();
  return 0;
}
