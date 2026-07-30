/**
 * M2 HYBRID Y-bifurcation BIE driver.
 *
 * Builds the QuadElemList junction (+ POU transition tubes + R0-hemisphere caps) and the
 * SlenderElemList arms, adds BOTH to one BoundaryIntegralOp, and runs the coupled identity sweep
 * (DL constant-density identity -> -1/2; interior Green's identity) for Laplace and Stokes. This is
 * the M2 counterpart of ybifurc-bie.cpp; the acceptance bar is the M1 full-quad accuracy.
 *
 * Global node/density ordering is the operator's NAME-SORTED list concatenation: the lists are added
 * as "0_junc" then "1_arms", so every global array here is [junction nodes ; arm nodes] in that order.
 *
 * arm_kind selects the arm element type -- 0=SlenderElemList (CSBQ, default), 1=a full-quad QuadElemList
 * tube over the SAME [a0,s_cap] R0 cylinder. Everything else (junction+transition+cap) is identical, so
 * the accuracy/timing delta between the two runs is attributable to the shaft representation alone. The
 * quad tube defaults to a node-CONFORMING azimuthal count (Na_shaft = 16*nref = the junction/cap count),
 * which makes the quad<->quad seam machine-exact; override argv[14] to sweep the azimuthal resolution.
 *
 *   make bin/ybifurc-hybrid-bie                       # or: make MPI=1 bin/ybifurc-hybrid-bie
 *   OMP_NUM_THREADS=8 ./bin/ybifurc-hybrid-bie \
 *       [level] [tol(unused,swept)] [order(mult4)] [nref] [eta_join] [Ns_trans] [s_cap] [n_axial] \
 *       [fourier] [nlev] [pou_kind] [arm_kind(0=slender/1=quad-tube)] [Ns_shaft] [Na_shaft] [Ncap]
 *
 * Ncap (argv[15], -1=default 2*nref) sets the butterfly-cap panels/direction of the junction caps,
 * refining the cap independently of nref. It enters the mesh-cache key, so distinct Ncap values
 * build/cache distinct junction meshes.
 */

#include <csbq.hpp>                                 // CSBQ SlenderElemList
#include <quad_junctions/ybifurc_hybrid_geom.hpp>   // hybrid builders (junction+transition+cap, slender arms)
#include <quad_junctions/hybrid_bie_tests.hpp>      // shared BIE identity/watertightness tests
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>

using namespace sctl;
using namespace quad_junctions;

namespace {

inline void assert_allowed_params(Integer order, Integer cov_q, Integer Nbeta, Integer max_depth, const Comm& comm = Comm::Self()) {
  (void)comm;   // no output here; comm accepted for a uniform call signature with the other drivers
  auto in = [](Integer v, std::initializer_list<Integer> s){ for (Integer x : s) if (x == v) return true; return false; };
  SCTL_ASSERT_MSG(order >= 4 && order <= 48 && order % 4 == 0, "ElemOrder must be a multiple of 4 in {4,8,...,48}.");
  SCTL_ASSERT_MSG(in(cov_q, {6, 10}), "cov_q must be one of {6,10}.");
  SCTL_ASSERT_MSG(in(Nbeta, {48, 100, 200, 300, 400, 512}), "Nbeta must be one of {48,100,200,300,400,512}.");
  SCTL_ASSERT_MSG(in(max_depth, {4, 8, 12, 30}), "max_depth must be one of {4,8,12,30}.");
}

// Quad-list node-index -> region label (junction sphere / POU transition tube / hemisphere cap), per
// arm, matching the emit order in BuildYJunctionWithTransitions: arm0[J,T,C], arm1[...], arm2[...].
struct QuadRegions {
  Long per_arm = 0, nJ = 0, nT = 0;
  QuadRegions(Integer order, Integer nref, Integer Ns_trans, Integer Ncap) {
    const Long o2 = (Long)order*order, Na = 16*nref, Nr = 2*nref, nc = Ncap;
    nJ = Nr*Na*o2; nT = (Long)Ns_trans*Na*o2; const Long nC = 5*nc*nc*o2; per_arm = nJ+nT+nC;
  }
  std::string label(Long i) const {
    const Long arm = i/per_arm, w = i%per_arm;
    const char* r = (w < nJ) ? "junction" : (w < nJ+nT) ? "transition" : "cap";
    return std::string(r) + "/arm" + std::to_string((long)arm);
  }
};

// Set the near-quadrature scheme on the arm list: no-op for a slender list (it schemes itself off the
// operator tol), but a QuadElemList arm tube needs the same Hybrid scheme the junction gets.
template <class Real> void set_arm_scheme(SlenderElemList<Real>&, Integer, Integer, Integer) {}
template <class Real> void set_arm_scheme(QuadElemList<Real>& a, Integer cov_q, Integer Nb, Integer md) {
  a.SetQuadScheme(QuadElemList<Real>::QuadScheme::Hybrid, cov_q, Nb, md);
}

// Panel squareness: physical edge lengths in the two param directions, aspect = max/min.
template <class Real> void verify_aspect(const QuadElemList<Real>& el, Integer order, const Comm& comm) {
  Vector<Real> X; el.GetNodeCoord(&X, nullptr, nullptr);
  const Long ne = el.Size(), nn = (Long)order*order;
  auto nd = [&](Long e, Integer i, Integer j, int c){ return X[(e*nn + (Long)i*order + j)*3 + c]; };
  Real amax = 0, amin = 1e30, asum = 0; Long n2 = 0, n3 = 0;
  for (Long e = 0; e < ne; e++) {
    Real Lu = 0, Lv = 0;
    for (Integer j = 0; j < order; j++) { Real L = 0; for (Integer i = 1; i < order; i++) { Real d = 0; for (int c=0;c<3;c++){ const Real t=nd(e,i,j,c)-nd(e,i-1,j,c); d+=t*t; } L += sqrt<Real>(d); } Lu += L; }
    for (Integer i = 0; i < order; i++) { Real L = 0; for (Integer j = 1; j < order; j++) { Real d = 0; for (int c=0;c<3;c++){ const Real t=nd(e,i,j,c)-nd(e,i,j-1,c); d+=t*t; } L += sqrt<Real>(d); } Lv += L; }
    Lu /= order; Lv /= order;
    const Real a = std::max(Lu,Lv) / std::max<Real>(std::min(Lu,Lv), (Real)1e-30);
    amax = std::max(amax,a); amin = std::min(amin,a); asum += a; if (a>2) n2++; if (a>3) n3++;
  }
  // Global aspect statistics over all ranks' panels.
  amax = GlobalReduce((double)amax, comm, CommOp::MAX);
  amin = GlobalReduce((double)amin, comm, CommOp::MIN);
  asum = GlobalReduce((double)asum, comm, CommOp::SUM);
  const Long ne_tot = GlobalReduce((Long)ne, comm, CommOp::SUM);
  n2 = GlobalReduce((Long)n2, comm, CommOp::SUM);
  n3 = GlobalReduce((Long)n3, comm, CommOp::SUM);
  if (!comm.Rank())
    std::cout << "  [aspect] panels=" << ne_tot << std::setprecision(3) << "  min=" << amin
              << " mean=" << asum/ne_tot << " max=" << amax << "  (#>2x=" << n2 << ", #>3x=" << n3 << ")\n";
}

// The generic coupled-BIE tests (combined_nodes, divergence_check, test_DLIdentity,
// test_greens_identity) now live in <quad_junctions/hybrid_bie_tests.hpp>; the single-junction
// per-region breakdown is injected below via a RegionReport closure keyed on QuadRegions.

// Run the divergence check + coupled BIE identity/timing sweep for a given arm list (slender OR
// quad-tube). Everything above the arm list -- junction+transition+cap -- is identical between the two;
// this is the single-variable A/B. `arm_tag` labels the arm type in the printout.
template <class Real, class ArmList>
void run_comparison(QuadElemList<Real>& junc, ArmList& arms, const Comm& comm, const Integer ord,
                    const Integer nref, const Integer NsTrans, const Integer NcapEff, const std::string& tag,
                    const std::string& arm_tag, const Integer nlev, const Integer cov_q) {
  const Real    tolL[4] = {(Real)1e-5, (Real)1e-7, (Real)1e-9, (Real)1e-11};
  const Integer NbL[4]  = {48, 100, 200, 400};
  const Integer mdL[4]  = {4, 8, 12, 30};
  const Vector<Real> X0{1.6, 1.4, 0.9};
  const QuadRegions reg(ord, nref, NsTrans, NcapEff);
  // per-region max breakdown (junction / transition / cap / slender-arm), serial-only (see header).
  const RegionReport<Real> region_report = [&reg](const Vector<Real>& err, Long Nj, Long Na) {
    Real mj=0,mt=0,mc=0,ma=0; const Long Nnode = Nj+Na;
    for (Long i=0;i<Nnode;i++){ if(i>=Nj){ma=std::max(ma,err[i]);continue;}
      const Long w=i%reg.per_arm; if(w<reg.nJ)mj=std::max(mj,err[i]); else if(w<reg.nJ+reg.nT)mt=std::max(mt,err[i]); else mc=std::max(mc,err[i]); }
    std::cout << "    [region max] junction=" << mj << " transition=" << mt << " cap=" << mc << " arm=" << ma << "\n"; };
  const Long njp = GlobalReduce((Long)junc.Size(), comm, CommOp::SUM), nap = GlobalReduce((Long)arms.Size(), comm, CommOp::SUM);
  Vector<Real> Xj, Xa; junc.GetNodeCoord(&Xj, nullptr, nullptr); arms.GetNodeCoord(&Xa, nullptr, nullptr);
  const Long njn = GlobalReduce((Long)(Xj.Dim()/3), comm, CommOp::SUM), nan = GlobalReduce((Long)(Xa.Dim()/3), comm, CommOp::SUM);
  if (!comm.Rank())
    std::cout << "\n---- BIE sweep [arm=" << arm_tag << "]: junction panels=" << njp << " nodes=" << njn
              << " | arm panels=" << nap << " nodes=" << nan << " ----\n";
  // watertightness / orientation diagnostic (cheap; needs the quad scheme set on both quad lists)
  junc.SetQuadScheme(QuadElemList<Real>::QuadScheme::Hybrid, cov_q, NbL[0], mdL[0]);
  set_arm_scheme<Real>(arms, cov_q, NbL[0], mdL[0]);
  divergence_check<Real>(junc, arms, tolL[0], comm);
  for (int idx = 0; idx < nlev; idx++) {
    junc.SetQuadScheme(QuadElemList<Real>::QuadScheme::Hybrid, cov_q, NbL[idx], mdL[idx]);
    set_arm_scheme<Real>(arms, cov_q, NbL[idx], mdL[idx]);
    if (!comm.Rank()) std::cout << "  [tol=" << tolL[idx] << " Nbeta=" << NbL[idx] << " max_depth=" << mdL[idx] << "]\n";
    const bool dump = (idx == nlev-1);
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
    const Real    tol_in  = (argc > 2) ? (Real)atof(argv[2]) : (Real)1e-6; (void)tol_in;  // quad tol swept below
    const Integer ord     = (argc > 3) ? (Integer)atoi(argv[3]) : 12;
    const Integer nref    = (argc > 4) ? (Integer)atoi(argv[4]) : 1;
    const Real    etajoin = (argc > 5) ? (Real)atof(argv[5]) : (Real)0.4;
    const Integer NsTrans = (argc > 6) ? (Integer)atoi(argv[6]) : 3;
    const Real    s_cap   = (argc > 7) ? (Real)atof(argv[7]) : (Real)0.88;
    const Integer nAxial  = (argc > 8) ? (Integer)atoi(argv[8]) : 3;
    const Long    fourier = (argc > 9) ? (Long)atoi(argv[9]) : 12;
    const Integer nlev    = (argc > 10) ? (Integer)atoi(argv[10]) : 4;   // BIE sweep levels (1..4); use <4 for fast debug
    const Integer poukind = (argc > 11) ? (Integer)atoi(argv[11]) : 1;   // 0=C-inf bump, 1=smootherstep (default)
    const Integer armkind = (argc > 12) ? (Integer)atoi(argv[12]) : 0;   // 0=slender arms (default) / 1=full-quad tube arms
    // Quad-tube shaft discretization (only used when arm_kind=1):
    //   Ns_shaft (argv[13]): axial panels; default matches the slender axial node count n_axial*cheb(10).
    //   Na_shaft (argv[14]): azimuthal panels; default = the junction/transition/cap azimuthal count
    //     (YSwept::Na0*nref = 16*nref), which makes the base/tip seam rings NODE-CONFORMING to the quad
    //     neighbour -> the quad<->quad cross-list seam is machine-exact. (Coarser, non-conforming Na
    //     floors the seam; see m3 memo.) Overridable to sweep the azimuthal resolution.
    const Integer NsShaft = (argc > 13) ? (Integer)atoi(argv[13]) : std::max<Integer>(1, (Integer)std::lround((double)nAxial*10.0/ord));
    const Integer NaShaft = (argc > 14) ? (Integer)atoi(argv[14]) : (Integer)(YSwept::Na0 * nref);
    // Junction butterfly-cap panels/direction. -1 (default) => YSwept::Ncap0*nref (=2*nref), matching
    // BuildYJunctionWithTransitions' internal default; override (argv[15]) to refine the cap alone.
    const Integer Ncap    = (argc > 15) ? (Integer)atoi(argv[15]) : -1;
    const Integer NcapEff = (Ncap > 0) ? Ncap : (Integer)(YSwept::Ncap0 * nref);
    pou_kind() = poukind;
    const Integer cov_q = 6;

    if (!comm.Rank()) {
      std::cout << "\n=== HYBRID Y-bifurcation (QuadElemList junction+transitions+caps  +  "
                << (armkind==0 ? "SlenderElemList" : "QuadElemList-tube") << " arms) ===\n";
      std::cout << "  order=" << ord << " level=" << level << " nref=" << nref << " eta_join=" << etajoin
                << " Ns_trans=" << NsTrans << " Ncap=" << NcapEff << " s_cap=" << s_cap << " n_axial=" << nAxial << " fourier=" << fourier
                << " pou_kind=" << poukind << (poukind==0?"(bump)":"(smootherstep)")
                << " arm_kind=" << armkind << (armkind==0?"(slender)":"(quad-tube)");
      if (armkind == 1) std::cout << " Ns_shaft=" << NsShaft << " Na_shaft=" << NaShaft
                                  << (NaShaft==YSwept::Na0*nref ? "(conforming)" : "");
      std::cout << "\n";
    }
    assert_allowed_params(ord, cov_q, 48, 4, comm);
    if (armkind == 1) SCTL_ASSERT_MSG(NaShaft >= 2, "quad-tube arms: Na_shaft must be >= 2 (a closed tube needs >=2 azimuthal panels).");

    Real R0[3], a0[3], sc[3], max_res = 0;
    QuadElemList<Real> junc;
    // Junction mesh depends on (order, level, nref, eta_join, Ns_trans, s_cap, pou_kind) but NOT on
    // the arm Fourier order -> cache per-order and reuse across fourier runs.
    std::ostringstream key;
    key << "data/mesh-cache/junc-L" << level << "-ord" << ord << "-nref" << nref
        << "-eta" << etajoin << "-Ns" << NsTrans << "-Ncap" << NcapEff << "-scap" << s_cap << "-pou" << poukind;
    const std::string mesh_file  = key.str() + ".mesh";
    const std::string frame_file = key.str() + ".frame";
    auto exists = [](const std::string& f){ std::ifstream s(f); return s.good(); };
    if (exists(mesh_file) && exists(frame_file)) {
      junc.template Read<Real>(mesh_file, comm);
      std::ifstream s(frame_file);
      for (int k=0;k<3;k++) { s>>R0[k]; }
      for (int k=0;k<3;k++) { s>>a0[k]; }
      for (int k=0;k<3;k++) { s>>sc[k]; }
      if (!comm.Rank()) std::cout << "  [mesh] read cached junction from " << mesh_file << "\n";
    } else {
      junc = BuildYJunctionWithTransitions<Real>(ord, level, nref, etajoin, NsTrans, s_cap,
                                                 R0, a0, sc, Ncap, &max_res, comm);
      max_res = GlobalReduce((double)max_res, comm, CommOp::MAX);
      junc.Write(mesh_file, comm);                 // collective: rank 0 writes one shared file
      if (!comm.Rank()) {
        std::ofstream s(frame_file); s << std::setprecision(17);
        for (int k=0;k<3;k++) { s<<R0[k]<<"\n"; }
        for (int k=0;k<3;k++) { s<<a0[k]<<"\n"; }
        for (int k=0;k<3;k++) { s<<sc[k]<<"\n"; }
        std::cout << "  [mesh] built + cached junction to " << mesh_file
                  << "  (junction ray projection max|f-level| = " << std::setprecision(3) << max_res << ")\n";
      }
    }
    if (!comm.Rank()) {
      for (int k = 0; k < 3; k++)
        std::cout << "  arm " << k << ": R0=" << std::setprecision(6) << R0[k] << "  axial [" << a0[k] << ", " << sc[k] << "]\n";
    }
    verify_aspect<Real>(junc, ord, comm);   // junction panel squareness (junction+transition+cap quads)

    // Tag encodes the arm type so the A (slender) and B (quad-tube) VTK dumps don't clobber each other.
    const std::string tag = "vis/ybifurc-hybrid-L" + std::to_string((double)level) + "-ord" + std::to_string((long)ord)
                          + "-nref" + std::to_string((long)nref) + (armkind==0 ? "-slender" : "-quadtube");

    // Build the arm list (only piece that differs between A/B) and run the identical sweep on it.
    if (armkind == 0) {
      SlenderElemList<Real> arms = BuildYArmsSlender<Real>(R0, a0, sc, nAxial, 10, fourier, comm);
      junc.WriteVTK(tag + "-junc", Vector<Real>(), comm);
      arms.WriteVTK(tag + "-arms", Vector<Real>(), comm);
      if (!comm.Rank()) std::cout << "  wrote " << tag << "-{junc,arms}.vtu\n";
      run_comparison<Real>(junc, arms, comm, ord, nref, NsTrans, NcapEff, tag, "slender", nlev, cov_q);
    } else {
      QuadElemList<Real> arms = BuildYArmsQuadTube<Real>(ord, R0, a0, sc, NsShaft, NaShaft, comm);
      if (!comm.Rank()) {
        const double h_ax = (double)(sc[0]-a0[0])/NsShaft, w_az = 2*M_PI*(double)R0[0]/NaShaft;
        std::cout << "  quad-tube: Ns_shaft=" << NsShaft << " Na_shaft=" << NaShaft
                  << "  panel h_axial=" << std::setprecision(3) << h_ax << " w_azim=" << w_az
                  << " aspect=" << (w_az>h_ax ? w_az/h_ax : h_ax/w_az) << " (want ~1)\n";
      }
      junc.WriteVTK(tag + "-junc", Vector<Real>(), comm);
      arms.WriteVTK(tag + "-arms", Vector<Real>(), comm);
      if (!comm.Rank()) std::cout << "  wrote " << tag << "-{junc,arms}.vtu\n";
      run_comparison<Real>(junc, arms, comm, ord, nref, NsTrans, NcapEff, tag, "quad-tube", nlev, cov_q);
    }
  }
  Comm::MPI_Finalize();
  return 0;
}
