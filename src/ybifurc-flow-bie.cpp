/**
 * Physical Stokes inflow-outflow BVP on the 2-bifurcation (two-junction) hybrid geometry.
 *
 * Builds the SAME two-junction assembly as ybifurc-multi-bie.cpp case 2 -- junction A at (-10,0,0)
 * (arm0->+x), junction B at (5,0,0) (arm0->-x), joined by ONE shared slender arm on the x-axis, the
 * other four arms free and closed by hemisphere-dome caps -- then solves an INTERIOR Stokes velocity
 * Dirichlet BVP on it:
 *
 *   - Parabolic INFLOW on junction A's two caps (negative-x side), volumetric flux p_in1, p_in2.
 *   - Parabolic OUTFLOW on junction B's two caps (positive-x side), volumetric flux p_out1, p_out2.
 *   - No-slip (u = 0) everywhere else (tube walls, junction bodies, shared arm).
 *
 * The profile amplitude on each cap is normalized so that the prescribed volumetric flux through that
 * cap is exactly +-p. The net flux is -p_in1 - p_in2 + p_out1 + p_out2 = -10 -10 +5 +15 = 0, which is
 * the compatibility condition int u.n dA = 0 required for solvability of the interior incompressible
 * Stokes Dirichlet problem -- the driver computes each cap's flux from the far-field quadrature weights
 * and asserts the total is ~0.
 *
 * Formulation: combined-field ( -1/2 I - S + D ) sigma = u_bc via GMRES (Stokes SL/DL kernels), solved
 * by the shared solve_dirichlet_bvp() in hybrid_bie_tests.hpp. Outputs (VTU only, open in ParaView):
 *   - surface density sigma (3-component vector) on the quad list and the slender list;
 *   - the prescribed boundary velocity on the quad list (to see the inflow/outflow);
 *   - a z=0 planar slice of the interior velocity field, masked to the fluid interior via the Laplace
 *     DL constant-density indicator.
 *
 *   make bin/ybifurc-flow-bie                    # or: make MPI=1 bin/ybifurc-flow-bie
 *   OMP_NUM_THREADS=8 ./bin/ybifurc-flow-bie \
 *       [level] [order(mult4)] [nref] [eta_join] [Ns_trans] [s_cap] [n_axial_free] [fourier] \
 *       [tol] [Nbeta] [max_depth] [cov_q] [p_in1] [p_in2] [p_out1] [p_out2] [Ngrid]
 */

#include <csbq.hpp>                                  // CSBQ SlenderElemList + CubeVolumeVis
#include <quad_junctions/ybifurc_assembly.hpp>       // composable component API
#include <quad_junctions/quad_scheme.hpp>            // QJDefaultScheme (Duffy default, SCTL_SELF_SCHEME=hybrid opt-out)
#include <quad_junctions/hybrid_bie_tests.hpp>       // combined_nodes + solve_dirichlet_bvp
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

using namespace sctl;
using namespace quad_junctions;

namespace {

// One inflow/outflow cap: dome-equator center C, outward axis u, radius R0, signed amplitude `amp`
// (= sgn * p / g, where g is the geometric flux factor so that flux through the cap = sgn*p), and the
// prescribed flux magnitude p (sgn = -1 inflow / +1 outflow).
template <class Real> struct FlowCap {
  Vec3<Real> C, u;
  Real R0 = 0, amp = 0, p = 0;
  int sgn = 0;
};

template <class Real> inline Real dot3(const Vec3<Real>& a, const Vec3<Real>& b) {
  return a[0]*b[0] + a[1]*b[1] + a[2]*b[2];
}

// Parabolic axial profile on a cap dome: prof(X) = 1 - (r/R0)^2 with r the transverse distance from the
// cap axis, for nodes lying on that cap's hemisphere (|X-C| ~ R0, on the +u side). Returns prof and the
// outward-axis component; 0 for any point not on a cap (no-slip walls/junctions/shared arm).
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
    const Real    p_in1   = (argc > 13) ? (Real)atof(argv[13]) : (Real)10;
    const Real    p_in2   = (argc > 14) ? (Real)atof(argv[14]) : (Real)10;
    const Real    p_out1  = (argc > 15) ? (Real)atof(argv[15]) : (Real)5;
    const Real    p_out2  = (argc > 16) ? (Real)atof(argv[16]) : (Real)15;
    const Long    Ngrid   = (argc > 17) ? (Long)atoi(argv[17]) : 200;
    const Integer Ncap    = (Integer)(YSwept::Ncap0 * nref);
    pou_kind() = 1;   // smootherstep POU (order-exact) -- what the assembly's transitions expect

    SCTL_ASSERT_MSG(ord >= 4 && ord <= 48 && ord % 4 == 0, "order must be a multiple of 4 in {4,...,48}.");

    if (!comm.Rank()) {
      std::cout << "\n=== Stokes inflow-outflow BVP on the 2-bifurcation ===\n";
      std::cout << "  order=" << ord << " level=" << level << " nref=" << nref << " eta_join=" << etajoin
                << " Ns_trans=" << NsTrans << " s_cap=" << s_cap << " n_axial(free)=" << nAxial
                << " fourier=" << fourier << "\n";
      std::cout << "  near-eval: Hybrid(cov_q=" << cov_q << ", Nbeta=" << Nbeta << ", max_depth=" << maxdep
                << ") tol=" << std::setprecision(1) << tol << "\n";
      std::cout << "  prescribed flux: inflow p_in={" << std::setprecision(4) << p_in1 << "," << p_in2
                << "} outflow p_out={" << p_out1 << "," << p_out2 << "}  net="
                << (-p_in1 - p_in2 + p_out1 + p_out2) << "\n";
    }

    // ----------------------------------------------------------------------------------------------
    // (1) Build the two-junction assembly (identical to ybifurc-multi-bie.cpp case 2, straight shared arm).
    // ----------------------------------------------------------------------------------------------
    HybridAssembly<Real> A(ord);
    const Placement<Real> PA = Placement<Real>::AlignArm(0, Vec3<Real>{1,0,0},  Vec3<Real>{0,0,1}, Vec3<Real>{-10,0,0});
    const Placement<Real> PB = Placement<Real>::AlignArm(0, Vec3<Real>{-1,0,0}, Vec3<Real>{0,0,1}, Vec3<Real>{5,0,0});
    const HybridJunction<Real> JA = A.add_junction(PA, level, nref, etajoin, NsTrans);
    const HybridJunction<Real> JB = A.add_junction(PB, level, nref, etajoin, NsTrans);

    const ArmSeam<Real>& sa = JA.seam(0); const ArmSeam<Real>& sb = JB.seam(0);
    const Real dx = sb.C[0]-sa.C[0], dy = sb.C[1]-sa.C[1], dz = sb.C[2]-sa.C[2];
    const Real len = sqrt<Real>(dx*dx+dy*dy+dz*dz);
    const Integer ns = std::max<Integer>(4, (Integer)std::lround((double)len/(double)(2*sa.R0)));
    A.add_shared_arm(sa, sb, ns, 10, fourier, (Real)0, (Real)0);   // straight (no sine wiggle)

    // the other two arms of each junction are free/capped
    for (int k = 1; k < 3; k++) {
      A.add_free_arm(JA.seam(k), s_cap, nAxial, Ncap, 10, fourier);
      A.add_free_arm(JB.seam(k), s_cap, nAxial, Ncap, 10, fourier);
    }

    QuadElemList<Real> junc = A.quad(comm);
    SlenderElemList<Real> arms = A.slender(comm);
    junc.SetQuadScheme(quad_junctions::QJDefaultScheme<Real>(), cov_q, Nbeta, maxdep);
    const std::string tag = "vis/ybifurc-flow-ord" + std::to_string((long)ord) + "-nref" + std::to_string((long)nref);

    // ----------------------------------------------------------------------------------------------
    // (2) Identify the four free-arm caps and assign inflow/outflow flux by the sign of the cap x-coord.
    //     Cap dome-equator center = seam.C + (s_cap - seam.a0)*seam.u (see add_free_arm).
    // ----------------------------------------------------------------------------------------------
    std::vector<FlowCap<Real>> caps;
    {
      const ArmSeam<Real>* freeseams[4] = {&JA.seam(1), &JA.seam(2), &JB.seam(1), &JB.seam(2)};
      int in_cnt = 0, out_cnt = 0;
      for (const ArmSeam<Real>* s : freeseams) {
        FlowCap<Real> c;
        c.C  = Vec3<Real>{s->C[0] + (s_cap - s->a0)*s->u[0], s->C[1] + (s_cap - s->a0)*s->u[1], s->C[2] + (s_cap - s->a0)*s->u[2]};
        c.u  = s->u; c.R0 = s->R0;
        if (c.C[0] < 0) { c.sgn = -1; c.p = (in_cnt++  == 0) ? p_in1  : p_in2;  }  // negative-x = inflow
        else            { c.sgn = +1; c.p = (out_cnt++ == 0) ? p_out1 : p_out2; }  // positive-x = outflow
        caps.push_back(c);
      }
      if (!comm.Rank()) {
        std::cout << "\n  [caps] R0=" << std::setprecision(6) << caps[0].R0 << "  (4 free-arm hemisphere caps)\n";
        for (size_t i = 0; i < caps.size(); i++)
          std::cout << "    cap " << i << ": center=(" << std::setprecision(4) << caps[i].C[0] << "," << caps[i].C[1] << "," << caps[i].C[2]
                    << ")  axis=(" << caps[i].u[0] << "," << caps[i].u[1] << "," << caps[i].u[2] << ")  "
                    << (caps[i].sgn < 0 ? "INFLOW" : "OUTFLOW") << " p=" << caps[i].p << "\n";
      }
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
    //     VERIFY the flux: int v.n dA per cap (should be +-p) and the total (should be ~0).
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
        std::cout << "\n  [flux check] int v.n dA per cap (target +-p):\n";
        for (size_t c = 0; c < caps.size(); c++)
          std::cout << "    cap " << c << " (" << (caps[c].sgn < 0 ? "in " : "out") << "): flux="
                    << std::setprecision(6) << flux[(Long)c] << "  (target " << (Real)caps[c].sgn*caps[c].p << ")\n";
        std::cout << "    TOTAL net flux = " << total << "  (compatibility condition int u.n dA = 0)\n";
      }
      SCTL_ASSERT_MSG(std::fabs((double)total) < 1e-6, "net flux not zero -- interior Stokes BVP incompatible");
    }

    // ----------------------------------------------------------------------------------------------
    // (5) Interior sampling grid on the z=0 plane (rank 0 only, so eval/reductions count each once).
    // ----------------------------------------------------------------------------------------------
    const Real xlo = -11.5, xhi = 6.5, ylo = -2.5, yhi = 2.5;
    const Long Nx = Ngrid;
    const Long Ny = std::max<Long>(2, (Long)std::lround((double)Ngrid*(yhi-ylo)/(xhi-xlo)));
    Vector<Real> Xgrid;
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

    // ----------------------------------------------------------------------------------------------
    // (5b) Analytical-check probes: cross-section lines in the shared-arm fully-developed region,
    //      APPENDED after the grid targets (evaluated in the same solve). The shared arm is a straight
    //      cylinder of radius R0 on the x-axis carrying the full inflow Q = p_in1+p_in2 toward B (+x);
    //      far from both junctions the flow is Hagen-Poiseuille u_x(rho)=2Q/(pi R0^2)*(1-(rho/R0)^2).
    // ----------------------------------------------------------------------------------------------
    Long Ngrid_pts = 0;
    Real umax = 0;
    std::vector<Real> pb_x, pb_rho, pb_uxex;   // rank-0 probe metadata
    {
      const Real Rp = sa.R0;
      const Real Qshared = p_in1 + p_in2;                                  // flux through the shared arm (+x)
      umax = 2*Qshared/(const_pi<Real>()*Rp*Rp);
      const Real xm = (Real)0.5*(sa.C[0] + sb.C[0]);
      if (!comm.Rank()) {
        Ngrid_pts = Xgrid.Dim()/3;
        const Real x0s[5] = {xm-2, xm-1, xm, xm+1, xm+2};
        const Long Nrad = 19;
        for (const Real x0 : x0s)
          for (Long j = 0; j < Nrad; j++) {
            const Real yy = ((Real)-0.9 + (Real)1.8*j/(Nrad-1))*Rp;        // y across the diameter, z=0
            Xgrid.PushBack(x0); Xgrid.PushBack(yy); Xgrid.PushBack((Real)0);
            pb_x.push_back(x0); pb_rho.push_back(yy);
            pb_uxex.push_back(umax*((Real)1 - (yy*yy)/(Rp*Rp)));
          }
        std::cout << "  [poiseuille] shared-arm axis x in [" << std::setprecision(4) << sa.C[0] << "," << sb.C[0]
                  << "] R0=" << Rp << " Q=" << Qshared << " -> umax=" << umax
                  << " ; probes: 5 x-stations x " << Nrad << " radial = " << (Long)pb_x.size() << " (x in ["
                  << xm-2 << "," << xm+2 << "])\n";
      }
    }

    // ----------------------------------------------------------------------------------------------
    // (6) Solve the interior Stokes Dirichlet BVP (SL_scal=-1, DL_scal=+1 -> jump=-1/2) and evaluate the
    //     represented velocity field at the grid + probe points.
    // ----------------------------------------------------------------------------------------------
    Vector<Real> Ugrid;
    const Vector<Real> sigma = solve_dirichlet_bvp<Real, Stokes3D_FxU, Stokes3D_DxU>(
        junc, arms, comm, tol, bc, /*interior=*/true, /*SL_scal=*/(Real)-1, /*DL_scal=*/(Real)1,
        Xgrid, &Ugrid, "stokes inflow-outflow", /*gmres_max_iter=*/400);

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
        for (Long i = 0; i < Ngrid_pts; i++) {   // grid points only (probes appended after are left as-is)
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

    // ----------------------------------------------------------------------------------------------
    // (9) Analytical accuracy check: BIE velocity vs Hagen-Poiseuille in the shared-arm mid-region.
    //     The probe velocities are the tail of Ugrid (indices Ngrid_pts ...). Report rel-L2 of u_x,
    //     max axial error / umax, and the max transverse speed / umax (should be ~0 by symmetry).
    // ----------------------------------------------------------------------------------------------
    if (!comm.Rank()) {
      const Long Np = (Long)pb_x.size();
      Real e2 = 0, ref2 = 0, emax = 0, tmax = 0;
      const std::string csv = tag + "-poiseuille.csv";
      FILE* fp = fopen(csv.c_str(), "w");
      if (fp) fprintf(fp, "x,y,rho,ux_bie,uy_bie,uz_bie,ux_exact\n");
      for (Long k = 0; k < Np; k++) {
        const Long i = Ngrid_pts + k;
        const Real ux = Ugrid[3*i], uy = Ugrid[3*i+1], uz = Ugrid[3*i+2], uxe = pb_uxex[k];
        const Real e = ux - uxe;
        e2 += e*e; ref2 += uxe*uxe;
        emax = std::max<Real>(emax, std::fabs((double)e));
        tmax = std::max<Real>(tmax, sqrt<Real>(uy*uy + uz*uz));
        if (fp) fprintf(fp, "%.6f,%.6f,%.6f,%.9e,%.9e,%.9e,%.9e\n", (double)pb_x[k], (double)pb_rho[k],
                        std::fabs((double)pb_rho[k]), (double)ux, (double)uy, (double)uz, (double)uxe);
      }
      if (fp) fclose(fp);
      std::cout << "\n  [poiseuille] BIE velocity vs analytical Hagen-Poiseuille (shared-arm mid-region):\n"
                << std::setprecision(6)
                << "    rel-L2(u_x) = " << (ref2 > 0 ? sqrt<Real>(e2/ref2) : (Real)0)
                << "   max|u_x-exact|/umax = " << emax/umax
                << "   max transverse |u_t|/umax = " << tmax/umax << "\n"
                << "    (wrote " << csv << " : " << Np << " probe points)\n";
    }
  }
  Comm::MPI_Finalize();
  return 0;
}
