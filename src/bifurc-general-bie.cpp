/**
 * GENERALIZED bifurcation BIE driver: an N-arm junction for ARBITRARY branch angles, an arbitrary (incl.
 * EVEN) number of branches, and NON-COPLANAR (3D) arm directions, joined to CSBQ SlenderElemList arms in
 * ONE BoundaryIntegralOp, verified by the shared watertightness + DL-identity + Green's-identity tests
 * (Laplace and Stokes) in hybrid_bie_tests.hpp -- the same acceptance bar as ybifurc-hybrid-bie.
 *
 * The geometry kernel is include/quad_junctions/gen_junction_geom.hpp (GenSpec + NField + JuncGeom +
 * BuildGenJunctionWithTransitions/BuildGenArmsSlender). Node/density ordering is the operator's
 * name-sorted "0_junc" then "1_arms", identical to ybifurc-hybrid-bie, so hybrid_bie_tests.hpp applies
 * unchanged.
 *
 *   make bin/bifurc-general-bie
 *   OMP_NUM_THREADS=8 ./bin/bifurc-general-bie \
 *       [spec] [level] [order(mult4)] [nref] [eta_join] [Ns_trans] [s_cap] [n_axial] [fourier] \
 *       [nlev] [pou_kind]
 *
 * spec grammar (argv[1]):
 *   gaps:g1,g2,...,gN   coplanar arms; g_i = angular GAPS in degrees around +z (must sum to 360).
 *                       e.g. gaps:150,150,60  or  gaps:90,90,90,90 (planar 4-way +).
 *   dirs:x,y,z;x,y,z;.. explicit unit-ish arm directions (any N, non-coplanar allowed).
 *   y120  cross4  tetra4  tri3d      named presets (see parse_spec).
 * Default spec = y120 (symmetric 120 coplanar Y, i.e. the ybifurc shape rebuilt through this kernel).
 */

#include <csbq.hpp>                                 // CSBQ SlenderElemList
#include <quad_junctions/gen_junction_geom.hpp>     // generalized junction kernel
#include <quad_junctions/gen_geom_io.hpp>           // exact reloadable geometry-bundle export
#include <quad_junctions/hybrid_bie_tests.hpp>      // shared BIE identity/watertightness tests
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

using namespace sctl;
using namespace quad_junctions;

namespace {

using Real = double;

// ---- Parse the geometry spec token into a list of arm directions. ----
std::vector<Vec3<Real>> parse_spec(const std::string& spec) {
  auto from_gaps = [](const std::vector<double>& gaps) {
    std::vector<Vec3<Real>> d; double a = 0, sum = 0;
    for (double g : gaps) sum += g;
    SCTL_ASSERT_MSG(std::fabs(sum - 360.0) < 1e-6, "gaps: the angular gaps must sum to 360 degrees.");
    for (size_t i = 0; i < gaps.size(); i++) { const double th = a * M_PI/180; d.push_back(Vec3<Real>{(Real)std::cos(th), (Real)std::sin(th), 0}); a += gaps[i]; }
    return d;
  };
  auto split = [](const std::string& s, char c) { std::vector<std::string> out; std::stringstream ss(s); std::string t; while (std::getline(ss, t, c)) if (!t.empty()) out.push_back(t); return out; };

  if (spec == "y120"   || spec.empty()) return from_gaps({120,120,120});
  if (spec == "cross4")  return from_gaps({90,90,90,90});
  if (spec == "tri3d") {   // 3 arms lifted out of plane (non-coplanar)
    std::vector<Vec3<Real>> d; const double lift = 0.5;
    for (int k = 0; k < 3; k++) { const double th = k*2*M_PI/3; d.push_back(gv_unit(Vec3<Real>{(Real)std::cos(th), (Real)std::sin(th), (Real)lift})); }
    return d;
  }
  if (spec == "tetra4") {  // regular tetrahedron directions (non-coplanar, even count)
    std::vector<Vec3<Real>> d = { {1,1,1},{1,-1,-1},{-1,1,-1},{-1,-1,1} };
    for (auto& v : d) v = gv_unit(v); return d;
  }
  if (spec.rfind("gaps:", 0) == 0) {
    std::vector<double> g; for (auto& t : split(spec.substr(5), ',')) g.push_back(std::atof(t.c_str()));
    SCTL_ASSERT_MSG(g.size() >= 2, "gaps: need >= 2 arms.");
    return from_gaps(g);
  }
  if (spec.rfind("dirs:", 0) == 0) {
    std::vector<Vec3<Real>> d;
    for (auto& tok : split(spec.substr(5), ';')) { auto c = split(tok, ','); SCTL_ASSERT_MSG(c.size()==3, "dirs: each direction needs x,y,z."); d.push_back(gv_unit(Vec3<Real>{(Real)std::atof(c[0].c_str()), (Real)std::atof(c[1].c_str()), (Real)std::atof(c[2].c_str())})); }
    SCTL_ASSERT_MSG(d.size() >= 2, "dirs: need >= 2 arms.");
    return d;
  }
  SCTL_ASSERT_MSG(false, "unrecognized spec (use gaps:.. / dirs:.. / y120 / cross4 / tetra4 / tri3d).");
  return {};
}

// Quad-list node-index -> region (junction / transition / cap) per arm, matching the emit order in
// BuildGenJunctionWithTransitions: arm0[J,T,C], arm1[...], ... (identical layout to the ybifurc driver).
struct QuadRegions {
  Long per_arm = 0, nJ = 0, nT = 0;
  QuadRegions(Integer order, Integer nref, Integer Ns_trans, Integer Ncap, Integer Na0, Integer Nr0) {
    const Long o2 = (Long)order*order, Na = (Long)Na0*nref, Nr = (Long)Nr0*nref, nc = Ncap;
    nJ = Nr*Na*o2; nT = (Long)Ns_trans*Na*o2; const Long nC = 5*nc*nc*o2; per_arm = nJ+nT+nC;
  }
};

// Panel squareness (edge-length aspect max/min), identical to the ybifurc driver's verify_aspect.
void verify_aspect(const QuadElemList<Real>& el, Integer order, const Comm& comm) {
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
  amax = GlobalReduce((double)amax, comm, CommOp::MAX); amin = GlobalReduce((double)amin, comm, CommOp::MIN);
  asum = GlobalReduce((double)asum, comm, CommOp::SUM); const Long ne_tot = GlobalReduce((Long)ne, comm, CommOp::SUM);
  n2 = GlobalReduce((Long)n2, comm, CommOp::SUM); n3 = GlobalReduce((Long)n3, comm, CommOp::SUM);
  if (!comm.Rank()) std::cout << "  [aspect] panels=" << ne_tot << std::setprecision(3) << "  min=" << amin
              << " mean=" << asum/ne_tot << " max=" << amax << "  (#>2x=" << n2 << ", #>3x=" << n3 << ")\n";
}

} // anonymous namespace

int main(int argc, char** argv) {
  Comm::MPI_Init(&argc, &argv);
  {
    const Comm comm = Comm::World();
    const std::string spec_str = (argc > 1) ? std::string(argv[1]) : std::string("y120");
    const Real    level   = (argc > 2) ? (Real)atof(argv[2]) : (Real)1.5;
    const Integer ord     = (argc > 3) ? (Integer)atoi(argv[3]) : 12;
    const Integer nref    = (argc > 4) ? (Integer)atoi(argv[4]) : 1;
    const Real    etajoin = (argc > 5) ? (Real)atof(argv[5]) : (Real)0.4;
    const Integer NsTrans = (argc > 6) ? (Integer)atoi(argv[6]) : 3;
    const Real    s_cap   = (argc > 7) ? (Real)atof(argv[7]) : (Real)0.88;
    const Integer nAxial  = (argc > 8) ? (Integer)atoi(argv[8]) : 3;
    const Long    fourier = (argc > 9) ? (Long)atoi(argv[9]) : 12;
    const Integer nlev    = (argc > 10) ? (Integer)atoi(argv[10]) : 4;
    const Integer poukind = (argc > 11) ? (Integer)atoi(argv[11]) : 1;
    const Integer Ncap    = (argc > 12) ? (Integer)atoi(argv[12]) : -1;
    const Real    alphaD  = (argc > 13) ? (Real)atof(argv[13]) : (Real)38.0;   // nominal hole half-angle (auto-clamped for tight gaps)
    const Real    sigmaIn = (argc > 14) ? (Real)atof(argv[14]) : (Real)-1;      // Gaussian width; <0 => auto-thin for tight gaps
    const Real    clampf  = (argc > 15) ? (Real)atof(argv[15]) : (Real)0.8;     // hole clamp fraction of min half-gap
    pou_kind() = poukind;
    const Integer cov_q = 6;
    // Junction singular/near scheme: default Hybrid (RectPolar self + graded near); SCTL_SELF_SCHEME=duffy
    // opts into the ported Duffy edge-collapsed self + upstream split-at-foot near (runtime-digits driven).
    // Under Duffy the RectPolar self is not used, so the Nbeta (cov_order) ladder below is ignored; the
    // tol + max_depth ladder still drives it. Matches the SCTL_SELF_SCHEME switch in ybifurc-hybrid-bie.
    using QScheme = QuadElemList<Real>::QuadScheme;
    QScheme jsch = QScheme::Hybrid;
    if (const char* sch_env = std::getenv("SCTL_SELF_SCHEME"))
      if (std::string(sch_env) == "duffy") jsch = QScheme::Duffy;
    SCTL_ASSERT_MSG(ord >= 4 && ord <= 48 && ord % 4 == 0, "order must be a multiple of 4 in {4,...,48}.");

    // ---- Geometry spec + junction geometry ----
    GenSpec<Real> spec; spec.arm_dir = parse_spec(spec_str); spec.alpha_deg = alphaD; spec.alpha_clamp_frac = clampf;
    const Integer N = (Integer)spec.arm_dir.size();
    // Auto-thin the tubes for tight junctions: fat tubes (sigma 0.15, tuned for the symmetric 120 Y)
    // nearly merge in a tight gap and wreck the mesh; scale sigma with the min pairwise arm angle so a
    // 60-deg gap gets ~0.08 while wide (>=110 deg) junctions keep the nominal 0.15. Overridable (argv14>0).
    Real min_ang = M_PI;
    for (Integer i = 0; i < N; i++) for (Integer j = i+1; j < N; j++) {
      double d = 0; for (int c=0;c<3;c++) d += (double)spec.arm_dir[i][c]*(double)spec.arm_dir[j][c];
      min_ang = std::min(min_ang, (Real)std::acos(std::max(-1.0,std::min(1.0,d))));
    }
    const Real min_gap_deg = min_ang*180/M_PI;
    spec.sigma = (sigmaIn > 0) ? sigmaIn : std::max<Real>((Real)0.075, (Real)0.15*std::min<Real>((Real)1, min_gap_deg/(Real)110));
    JuncGeom<Real> jg = build_junc_geom<Real>(spec, nref);
    const Integer NcapEff = (Ncap > 0) ? Ncap : (Integer)(spec.Ncap0 * nref);

    if (!comm.Rank()) {
      std::cout << "\n=== GENERALIZED bifurcation (N=" << N << " arms, "
                << (jg.coplanar ? "coplanar" : "non-coplanar") << ") + CSBQ slender arms ===\n";
      std::cout << "  spec=\"" << spec_str << "\"  order=" << ord << " level=" << level << " nref=" << nref
                << " eta_join=" << etajoin << " Ns_trans=" << NsTrans << " Ncap=" << NcapEff << " s_cap=" << s_cap
                << " n_axial=" << nAxial << " fourier=" << fourier << " pou_kind=" << poukind
                << (poukind==0?"(bump)":"(smootherstep)")
                << " sigma=" << std::setprecision(4) << spec.sigma << " min_gap=" << min_gap_deg << "deg"
                << " alpha_eff=" << jg.alpha*180/M_PI << "deg (nominal " << alphaD << ")\n";
      for (Integer k = 0; k < N; k++)
        std::cout << "    arm " << k << " dir=(" << std::setprecision(4) << jg.u[k][0] << "," << jg.u[k][1] << "," << jg.u[k][2] << ")\n";
    }

    // ---- Full mesh-determining parameter set = the reloadable-bundle cache key beyond spec/order/nref. ----
    GenGeomParams<Real> prm;
    prm.order = ord; prm.nref = nref; prm.Ns_trans = NsTrans; prm.Ncap = Ncap; prm.n_axial = nAxial;
    prm.pou_kind = poukind; prm.cheb_order = 10; prm.fourier_order = fourier;
    prm.level = level; prm.eta_join = etajoin; prm.s_cap = s_cap; prm.alpha_deg = alphaD; prm.sigma = spec.sigma; prm.clampf = clampf;

    // ---- Build the coupled surface: junction (quad) + slender arms -- OR reload it from a matching bundle.
    //      When making a general bifurcation we ALWAYS check GenGeomDir() (geom/) first: the validated
    //      presets are committed there, so a preset run loads the exact surface instead of re-meshing. A
    //      miss (or QJ_GEOM_CACHE=0, or a parameter mismatch) rebuilds and then writes the bundle back. The
    //      loaded (junc, arms) pair is bit-for-bit the mesher's output -- the same product ReadGenGeom
    //      hands a downstream BIE/flow driver. ----
    const std::string prefix = GenGeomPath(spec_str, ord, nref);           // GenGeomDir() + spec-name + ord + nref
    std::vector<Real> R0, a0, sc;
    QuadElemList<Real> junc; SlenderElemList<Real> arms; GenArmTable<Real> tab;
    if (TryLoadGenGeom<Real>(prefix, prm, junc, arms, tab, comm)) {
      for (Integer k = 0; k < N; k++) { R0.push_back(tab.R0[k]); a0.push_back(tab.a0[k]); sc.push_back(tab.s_cap[k]); }
      if (!comm.Rank()) std::cout << "  [mesh] LOADED cached bundle " << prefix << ".{mesh,arms} (skipped field/Voronoi/mesh build)\n";
    } else {
      Real max_res = 0;
      junc = BuildGenJunctionWithTransitions<Real>(spec, jg, ord, level, nref, etajoin, NsTrans, s_cap, R0, a0, sc, Ncap, &max_res, comm);
      max_res = GlobalReduce((double)max_res, comm, CommOp::MAX);
      arms = BuildGenArmsSlender<Real>(jg, R0, a0, sc, nAxial, 10, fourier, comm);
      WriteGenGeom<Real>(prefix, junc, jg, R0, a0, sc, prm, comm);
      if (!comm.Rank()) std::cout << "  [mesh] built junction (ray projection max|f-level| = " << std::setprecision(3) << max_res
                                  << ") + wrote bundle " << prefix << ".{mesh,arms}\n";
    }
    if (!comm.Rank())
      for (Integer k = 0; k < N; k++)
        std::cout << "  arm " << k << ": R0=" << std::setprecision(6) << R0[k] << "  axial [" << a0[k] << ", " << sc[k] << "]\n";
    verify_aspect(junc, ord, comm);

    // ---- Exterior Green's sources: one beyond each arm cap tip (assert exterior via f < level). The
    //      interior third-identity form needs sources OUTSIDE the solid; placing them near each tip
    //      localizes the Green error near every arm. ----
    const NField<Real> fld(spec);
    Vector<Real> X0;
    for (Integer k = 0; k < N; k++) {
      const Real d = sc[k] + (Real)0.6;
      const Vec3<Real> p{d*jg.u[k][0], d*jg.u[k][1], d*jg.u[k][2]};
      SCTL_ASSERT_MSG(fld.f(p) < (Real)0.95*level, "Green source not exterior; increase the tip margin.");
      X0.PushBack(p[0]); X0.PushBack(p[1]); X0.PushBack(p[2]);
    }

    // ---- Region breakdown (serial only; coplanar only -- the per-arm node layout is uniform there.
    //      For the Voronoi path cell node counts vary, so the fixed per_arm stride would mislabel; skip. ----
    const QuadRegions reg(ord, nref, NsTrans, NcapEff, spec.Na0, spec.Nr0);
    const RegionReport<Real> region_report = !jg.coplanar ? RegionReport<Real>{} :
      [&reg](const Vector<Real>& err, Long Nj, Long Na) {
      Real mj=0,mt=0,mc=0,ma=0; const Long Nnode = Nj+Na;
      for (Long i=0;i<Nnode;i++){ if(i>=Nj){ma=std::max(ma,err[i]);continue;}
        const Long w=i%reg.per_arm; if(w<reg.nJ)mj=std::max(mj,err[i]); else if(w<reg.nJ+reg.nT)mt=std::max(mt,err[i]); else mc=std::max(mc,err[i]); }
      std::cout << "    [region max] junction=" << mj << " transition=" << mt << " cap=" << mc << " arm=" << ma << "\n"; };

    // ---- Sweep: watertightness + DL/Green identities (Laplace + Stokes) over the tol/Nbeta/max_depth ladder ----
    const Real    tolL[4] = {(Real)1e-5, (Real)1e-7, (Real)1e-9, (Real)1e-11};
    const Integer NbL[4]  = {48, 100, 200, 400};
    const Integer mdL[4]  = {4, 8, 12, 30};
    const Long njp = GlobalReduce((Long)junc.Size(), comm, CommOp::SUM), nap = GlobalReduce((Long)arms.Size(), comm, CommOp::SUM);
    Vector<Real> Xj, Xa; junc.GetNodeCoord(&Xj, nullptr, nullptr); arms.GetNodeCoord(&Xa, nullptr, nullptr);
    const Long njn = GlobalReduce((Long)(Xj.Dim()/3), comm, CommOp::SUM), nan = GlobalReduce((Long)(Xa.Dim()/3), comm, CommOp::SUM);
    if (!comm.Rank())
      std::cout << "\n---- BIE sweep [scheme=" << (jsch==QScheme::Duffy ? "Duffy" : "Hybrid") << "]: junction panels=" << njp << " nodes=" << njn << " | arm panels=" << nap << " nodes=" << nan << " ----\n";

    junc.SetQuadScheme(jsch, cov_q, NbL[0], mdL[0]);
    divergence_check<Real>(junc, arms, tolL[0], comm);
    for (int idx = 0; idx < nlev; idx++) {
      junc.SetQuadScheme(jsch, cov_q, NbL[idx], mdL[idx]);
      if (!comm.Rank()) std::cout << "  [tol=" << tolL[idx] << " Nbeta=" << NbL[idx] << " max_depth=" << mdL[idx] << "]\n";
      if (!comm.Rank()) std::cout << "    [Laplace] "; test_DLIdentity<Real, Laplace3D_DxU>(junc, arms, comm, tolL[idx], "", region_report);
      if (!comm.Rank()) std::cout << "    [Stokes]  "; test_DLIdentity<Real, Stokes3D_DxU>(junc, arms, comm, tolL[idx], "", region_report);
      if (!comm.Rank()) std::cout << "    [Laplace] "; test_greens_identity<Real, Laplace3D_FxU, Laplace3D_DxU, Laplace3D_FxdU>(junc, arms, comm, tolL[idx], X0, "");
      if (!comm.Rank()) std::cout << "    [Stokes]  "; test_greens_identity<Real, Stokes3D_FxU, Stokes3D_DxU, Stokes3D_FxT>(junc, arms, comm, tolL[idx], X0, "");
    }
  }
  Comm::MPI_Finalize();
  return 0;
}
