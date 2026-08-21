/**
 * gen_network_geom.hpp -- assemble a vessel NETWORK from a vmtk-derived graph, using the generalized
 * N-arm junction kernel (gen_junction_geom.hpp) for the junction BODIES and bent, tapered,
 * centerline-following CSBQ slender tubes for the ARMS.
 *
 * The graph (python/vessels_vmtk_graph.py -> <prefix>.graph) gives: representative CLUSTERS (degree +
 * a canonical `dirs:` arm-direction spec), NODES (world position, degree, junction/cap, radius,
 * cluster), and EDGES (n0,n1, radius endpoints, a resampled world centerline polyline). This header:
 *
 *   1. builds ONE canonical junction body per cluster (memoized) via the gen kernel with
 *      emit_caps=false (open seam rings) -- the "few representative junctions" approximation;
 *   2. PLACES each junction node by a least-squares rigid fit (Horn quaternion Kabsch) of the
 *      canonical arm directions to the node's true incident edge directions + a uniform radius scale;
 *   3. reconciles the approximation to the true geometry by BENDING each arm along its edge's vmtk
 *      centerline (Catmull-Rom through the polyline, straight panel-aligned leads at both seam rings,
 *      linear radius taper, rotation-minimizing orientation frame) -- exactly the seam-conforming,
 *      watertight join the ybifurc-hybrid / vessels arms use, generalized to an arbitrary centerline;
 *   4. caps leaf (degree-1) tips with a hemisphere butterfly;
 *   5. writes a PER-JUNCTION bundle (<prefix>-jNN.mesh + .arms) whose node/density ordering is the
 *      operator's name-sorted "0_junc" then "1_arms", so any ybifurc-hybrid-layout solve driver
 *      consumes it unchanged. The top-level <prefix>.graph carries the connectivity.
 *
 * Reuses (does not fork): gen_junction_geom.hpp (GenSpec/build_junc_geom/BuildGenJunctionWithTransitions),
 * ybifurc_assembly.hpp (Placement/ArmSeam/transform_nodes/transform_seam/add_cap_hemisphere_frame/
 * pou_ramp), quad_element (QuadElemList::Init/GetNodeCoord/Write), SlenderElemList ctor.
 */
#pragma once

#include <quad_junctions/gen_junction_geom.hpp>
#include <quad_junctions/ybifurc_assembly.hpp>
#include <algorithm>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <map>
#include <numeric>
#include <sstream>
#include <string>
#include <vector>

namespace quad_junctions {
using namespace sctl;

// ===================================================================================================
// Graph data (mirrors python/vessels_vmtk_graph.py's <prefix>.graph plain-text format).
// ===================================================================================================
template <class Real> struct GraphCluster {
  Integer degree = 0;
  std::vector<Vec3<Real>> arm_dir;                 // canonical (medoid) incident unit directions
};
template <class Real> struct GraphNode {
  Vec3<Real> pos{(Real)0,(Real)0,(Real)0};
  Integer degree = 0;
  bool is_junc = false;
  Real radius = 0;
  Integer cluster = -1;
};
template <class Real> struct GraphEdge {
  Integer n0 = -1, n1 = -1;
  Real r0 = 0, r1 = 0;
  std::vector<Vec3<Real>> cl;                       // world centerline polyline (n0 -> n1)
  std::vector<Real> rad;                            // radius profile along cl
};
template <class Real> struct NetworkGraph {
  std::vector<GraphCluster<Real>> clusters;
  std::vector<GraphNode<Real>> nodes;
  std::vector<GraphEdge<Real>> edges;
};

// Parse a "dirs:x,y,z;x,y,z;..." spec into unit directions (same grammar as bifurc-general-*).
template <class Real> std::vector<Vec3<Real>> parse_dirs_spec(const std::string& spec) {
  std::vector<Vec3<Real>> d;
  SCTL_ASSERT_MSG(spec.rfind("dirs:", 0) == 0, "gen_network: cluster spec must be 'dirs:...'");
  std::stringstream ss(spec.substr(5)); std::string tok;
  while (std::getline(ss, tok, ';')) {
    if (tok.empty()) continue;
    std::stringstream cs(tok); std::string c; Real v[3]; int i = 0;
    while (std::getline(cs, c, ',') && i < 3) v[i++] = (Real)std::atof(c.c_str());
    SCTL_ASSERT_MSG(i == 3, "gen_network: each dir needs 3 comps");
    d.push_back(gv_unit(Vec3<Real>{v[0], v[1], v[2]}));
  }
  return d;
}

template <class Real> NetworkGraph<Real> ReadNetworkGraph(const std::string& path) {
  std::ifstream f(path); SCTL_ASSERT_MSG(f.good(), "ReadNetworkGraph: cannot open " + path);
  NetworkGraph<Real> g; std::string line;
  auto next = [&]() -> std::string {
    std::string l; while (std::getline(f, l)) { size_t p = l.find_first_not_of(" \t\r\n");
      if (p != std::string::npos && l[p] != '#') return l; } return std::string();
  };
  auto head = [&](const std::string& key) -> long {
    std::istringstream ss(next()); std::string k; long n; ss >> k >> n;
    SCTL_ASSERT_MSG(k == key, "ReadNetworkGraph: expected section " + key + " got " + k); return n;
  };
  const long ncl = head("NCLUSTER");
  g.clusters.resize(ncl);
  for (long c = 0; c < ncl; c++) {
    std::istringstream ss(next()); Integer cid, deg; std::string spec; ss >> cid >> deg >> spec;
    g.clusters[cid].degree = deg; g.clusters[cid].arm_dir = parse_dirs_spec<Real>(spec);
    SCTL_ASSERT_MSG((Integer)g.clusters[cid].arm_dir.size() == deg, "ReadNetworkGraph: cluster degree mismatch");
  }
  const long nn = head("NNODE");
  g.nodes.resize(nn);
  for (long i = 0; i < nn; i++) {
    std::istringstream ss(next()); Integer id, deg, typ, cl; Real x, y, z, r;
    ss >> id >> x >> y >> z >> deg >> typ >> r >> cl;
    g.nodes[id].pos = Vec3<Real>{x, y, z}; g.nodes[id].degree = deg;
    g.nodes[id].is_junc = (typ == 0); g.nodes[id].radius = r; g.nodes[id].cluster = cl;
  }
  const long ne = head("NEDGE");
  g.edges.resize(ne);
  for (long e = 0; e < ne; e++) {
    std::istringstream ss(next()); Integer id, n0, n1, np; Real r0, r1;
    ss >> id >> n0 >> n1 >> r0 >> r1 >> np;
    g.edges[id].n0 = n0; g.edges[id].n1 = n1; g.edges[id].r0 = r0; g.edges[id].r1 = r1;
    g.edges[id].cl.resize(np); g.edges[id].rad.resize(np);
    for (Integer k = 0; k < np; k++) {
      std::istringstream ps(next()); Real x, y, z, r; ps >> x >> y >> z >> r;
      g.edges[id].cl[k] = Vec3<Real>{x, y, z}; g.edges[id].rad[k] = r;
    }
  }
  return g;
}

// ===================================================================================================
// Small vector helpers (self-contained).
// ===================================================================================================
namespace gnet {
template <class Real> inline Real dot(const Vec3<Real>& a, const Vec3<Real>& b) { return a[0]*b[0]+a[1]*b[1]+a[2]*b[2]; }
template <class Real> inline Vec3<Real> sub(const Vec3<Real>& a, const Vec3<Real>& b) { return Vec3<Real>{a[0]-b[0],a[1]-b[1],a[2]-b[2]}; }
template <class Real> inline Vec3<Real> add(const Vec3<Real>& a, const Vec3<Real>& b) { return Vec3<Real>{a[0]+b[0],a[1]+b[1],a[2]+b[2]}; }
template <class Real> inline Vec3<Real> scal(Real s, const Vec3<Real>& a) { return Vec3<Real>{s*a[0],s*a[1],s*a[2]}; }
template <class Real> inline Vec3<Real> cross(const Vec3<Real>& a, const Vec3<Real>& b) { return Vec3<Real>{a[1]*b[2]-a[2]*b[1],a[2]*b[0]-a[0]*b[2],a[0]*b[1]-a[1]*b[0]}; }
template <class Real> inline Real nrm(const Vec3<Real>& a) { return sqrt<Real>(dot(a,a)); }
template <class Real> inline Vec3<Real> unit(const Vec3<Real>& a) { const Real n = nrm(a); return (double)n>1e-30 ? scal((Real)1/n,a) : a; }
}  // namespace gnet

// ===================================================================================================
// Horn (1987) quaternion least-squares rotation: R minimizing sum |R*src[k] - dst[k]|^2 (unit vecs).
// Solved as the largest-eigenvector of the 4x4 key matrix via cyclic Jacobi (self-contained, no LAPACK).
// Returns the 9-entry row-major rotation matrix (proper, det=+1).
// ===================================================================================================
template <class Real> void horn_rotation(const std::vector<Vec3<Real>>& src, const std::vector<Vec3<Real>>& dst, Real R[9]) {
  using namespace gnet;
  // covariance H = sum src[k] dst[k]^T
  Real H[3][3] = {{0,0,0},{0,0,0},{0,0,0}};
  for (size_t k = 0; k < src.size(); k++)
    for (int i = 0; i < 3; i++) for (int j = 0; j < 3; j++) H[i][j] += src[k][i]*dst[k][j];
  // 4x4 symmetric key matrix N (Horn)
  const Real Sxx=H[0][0],Sxy=H[0][1],Sxz=H[0][2],Syx=H[1][0],Syy=H[1][1],Syz=H[1][2],Szx=H[2][0],Szy=H[2][1],Szz=H[2][2];
  Real N[4][4] = {
    { Sxx+Syy+Szz, Syz-Szy,      Szx-Sxz,      Sxy-Syx      },
    { Syz-Szy,     Sxx-Syy-Szz,  Sxy+Syx,      Szx+Sxz      },
    { Szx-Sxz,     Sxy+Syx,      -Sxx+Syy-Szz, Syz+Szy      },
    { Sxy-Syx,     Szx+Sxz,      Syz+Szy,      -Sxx-Syy+Szz }};
  // Jacobi eigen-decomposition of symmetric 4x4.
  Real V[4][4] = {{1,0,0,0},{0,1,0,0},{0,0,1,0},{0,0,0,1}};
  for (int sweep = 0; sweep < 50; sweep++) {
    Real off = 0; for (int p=0;p<4;p++) for (int q=p+1;q<4;q++) off += N[p][q]*N[p][q];
    if ((double)off < 1e-30) break;
    for (int p=0;p<4;p++) for (int q=p+1;q<4;q++) {
      if ((double)std::fabs((double)N[p][q]) < 1e-300) continue;
      const Real theta = (N[q][q]-N[p][p])/(2*N[p][q]);
      const Real t = (theta>=0?(Real)1:(Real)-1)/(std::fabs((double)theta)+sqrt<Real>(theta*theta+(Real)1));
      const Real c = (Real)1/sqrt<Real>(t*t+(Real)1), s = t*c;
      for (int i=0;i<4;i++){ const Real nip=N[i][p], niq=N[i][q]; N[i][p]=c*nip-s*niq; N[i][q]=s*nip+c*niq; }
      for (int i=0;i<4;i++){ const Real npi=N[p][i], nqi=N[q][i]; N[p][i]=c*npi-s*nqi; N[q][i]=s*npi+c*nqi; }
      for (int i=0;i<4;i++){ const Real vip=V[i][p], viq=V[i][q]; V[i][p]=c*vip-s*viq; V[i][q]=s*vip+c*viq; }
    }
  }
  int best = 0; for (int i=1;i<4;i++) if ((double)N[i][i] > (double)N[best][best]) best = i;
  Real qq[4] = {V[0][best],V[1][best],V[2][best],V[3][best]};
  const Real qn = sqrt<Real>(qq[0]*qq[0]+qq[1]*qq[1]+qq[2]*qq[2]+qq[3]*qq[3]);
  for (int i=0;i<4;i++) qq[i] /= qn;
  const Real w=qq[0],x=qq[1],y=qq[2],z=qq[3];
  R[0]=1-2*(y*y+z*z); R[1]=2*(x*y-w*z);   R[2]=2*(x*z+w*y);
  R[3]=2*(x*y+w*z);   R[4]=1-2*(x*x+z*z); R[5]=2*(y*z-w*x);
  R[6]=2*(x*z-w*y);   R[7]=2*(y*z+w*x);   R[8]=1-2*(x*x+y*y);
}

// ===================================================================================================
// Canonical junction body: built ONCE per cluster (emit_caps=false -> open seam rings). Carries the
// junction quad nodes in canonical/local coords + per-arm seam frame (canonical) so a placement is a
// rigid transform of both.
// ===================================================================================================
template <class Real> struct CanonicalJunction {
  Integer order = 0, N = 0;
  Vector<Real> Xcanon;                              // junction quad nodes (canonical), AoS
  std::vector<Vec3<Real>> u, e1, e2;                // per-arm frame (canonical)
  std::vector<Real> R0, a0;                         // per-arm seam radius + axial station (canonical)
  Real s_cap = 0;
  // Canonical seam ring of arm k (local coords): center a0*u, axis u, frame e1/e2, radius R0.
  ArmSeam<Real> seam(Integer k) const {
    ArmSeam<Real> s; s.u = u[k]; s.e1 = e1[k]; s.e2 = e2[k]; s.R0 = R0[k]; s.a0 = a0[k];
    s.C = Vec3<Real>{a0[k]*u[k][0], a0[k]*u[k][1], a0[k]*u[k][2]}; return s;
  }
};

// Build the canonical junction for one cluster. Params mirror the bifurc-general playbook defaults; the
// sigma auto-thin rule (tight-gap safety) is applied here exactly as the drivers do.
template <class Real> CanonicalJunction<Real> build_canonical(const GraphCluster<Real>& cl, Integer order,
    Real level, Integer nref, Real eta_join, Integer Ns_trans, Real s_cap_arc, Integer Ncap,
    Real alpha_deg, Real clampf, Real sigma_in) {
  GenSpec<Real> spec;
  spec.arm_dir = cl.arm_dir;
  spec.alpha_deg = alpha_deg; spec.alpha_clamp_frac = clampf;
  // auto-thin sigma with the min pairwise arm angle (fat tubes merge in tight gaps).
  Real min_gap = 180;
  for (size_t a = 0; a < cl.arm_dir.size(); a++) for (size_t b = a+1; b < cl.arm_dir.size(); b++) {
    Real c = gnet::dot(cl.arm_dir[a], cl.arm_dir[b]); c = std::max<Real>((Real)-1, std::min<Real>((Real)1, c));
    min_gap = std::min<Real>(min_gap, (Real)(std::acos((double)c)*180.0/M_PI));
  }
  // sigma floor lowered 0.075 -> 0.05 (2026-08-10): the 0.075 floor clamped tight-gap junctions (min_gap
  // < ~55 deg, where 0.15*gap/110 < 0.075) FATTER than the arm tubes can be without merging, so their
  // transition-tube rims degrade and the watertightness floors ~2e-3 relative -- amplified up to ~400x by
  // the network's large placement scale (a radius-2 junction built from a meanR0~0.1 canonical is scaled
  // ~20x, area ~400x). The dominant network leak (j298 = cluster 150, 42 deg gap, |f|=0.702 == the whole
  // combined residual) sat exactly on this floor. Letting the gap-adaptive rule reach ~0.057 for a 42 deg
  // gap drops that junction 540x (1.8e-3 -> 3.3e-6). Wide junctions (gap > 55 deg) have 0.15*gap/110 >
  // 0.05 already, so their sigma -- and geometry -- is byte-identical. A small 0.05 safety floor remains
  // to avoid degenerate over-thinning at extreme gaps.
  spec.sigma = (sigma_in > 0) ? sigma_in : std::max<Real>((Real)0.05, (Real)0.15*std::min<Real>((Real)1, min_gap/(Real)110));
  JuncGeom<Real> jg = build_junc_geom<Real>(spec, nref);
  CanonicalJunction<Real> C; C.order = order; C.N = jg.N; C.s_cap = s_cap_arc;
  std::vector<Real> R0, a0, sc;
  QuadElemList<Real> junc = BuildGenJunctionWithTransitions<Real>(spec, jg, order, level, nref, eta_join,
      Ns_trans, s_cap_arc, R0, a0, sc, Ncap, nullptr, Comm::Self(), /*emit_caps*/false);
  junc.GetNodeCoord(&C.Xcanon, nullptr, nullptr);
  C.R0 = R0; C.a0 = a0;
  C.u.resize(jg.N); C.e1.resize(jg.N); C.e2.resize(jg.N);
  for (Integer k = 0; k < jg.N; k++) gen_arm_frame<Real>(jg, k, C.u[k], C.e1[k], C.e2[k]);
  return C;
}

// ===================================================================================================
// Placement fit: best rigid transform sending the canonical arm dirs onto the node's incident edge
// dirs. Brute-forces the arm<->edge matching permutation (degree <= ~6), Horn-fits each, keeps the min
// angular-residual one. Returns the Placement + the per-edge canonical-arm index (edge_dirs order).
// ===================================================================================================
template <class Real> Placement<Real> fit_placement(const CanonicalJunction<Real>& C,
    const std::vector<Vec3<Real>>& edge_dirs, const Vec3<Real>& center, Real scale,
    std::vector<Integer>& arm_of_edge, Real* worst_deg = nullptr) {
  using namespace gnet;
  const Integer n = (Integer)edge_dirs.size();
  SCTL_ASSERT_MSG(n == C.N, "fit_placement: node degree != canonical arm count");
  std::vector<Integer> perm(n); std::iota(perm.begin(), perm.end(), 0);
  Real bestR[9] = {1,0,0,0,1,0,0,0,1}; Real best_res = 1e30; std::vector<Integer> best_perm = perm;
  do {
    std::vector<Vec3<Real>> src(n), dst(n);
    for (Integer k = 0; k < n; k++) { src[k] = C.u[k]; dst[k] = edge_dirs[perm[k]]; }
    Real R[9]; horn_rotation<Real>(src, dst, R);
    Real res = 0;
    for (Integer k = 0; k < n; k++) {
      Vec3<Real> Ru{R[0]*src[k][0]+R[1]*src[k][1]+R[2]*src[k][2],
                    R[3]*src[k][0]+R[4]*src[k][1]+R[5]*src[k][2],
                    R[6]*src[k][0]+R[7]*src[k][1]+R[8]*src[k][2]};
      Real c = dot(Ru, dst[k]); c = std::max<Real>((Real)-1, std::min<Real>((Real)1, c));
      res += (Real)std::acos((double)c);
    }
    if ((double)res < (double)best_res) { best_res = res; for (int i=0;i<9;i++) bestR[i]=R[i]; best_perm = perm; }
  } while (std::next_permutation(perm.begin(), perm.end()));
  Placement<Real> P; P.t = center; P.scale = scale; for (int i=0;i<9;i++) P.R[i] = bestR[i];
  // arm_of_edge[e] = canonical arm index assigned to edge e.
  arm_of_edge.assign(n, -1);
  for (Integer k = 0; k < n; k++) arm_of_edge[best_perm[k]] = k;
  if (worst_deg) {
    Real w = 0;
    for (Integer k = 0; k < n; k++) {
      Vec3<Real> Ru{bestR[0]*C.u[k][0]+bestR[1]*C.u[k][1]+bestR[2]*C.u[k][2],
                    bestR[3]*C.u[k][0]+bestR[4]*C.u[k][1]+bestR[5]*C.u[k][2],
                    bestR[6]*C.u[k][0]+bestR[7]*C.u[k][1]+bestR[8]*C.u[k][2]};
      Real c = dot(Ru, edge_dirs[best_perm[k]]); c = std::max<Real>((Real)-1, std::min<Real>((Real)1, c));
      w = std::max<Real>(w, (Real)(std::acos((double)c)*180.0/M_PI));
    }
    *worst_deg = w;
  }
  return P;
}

// ===================================================================================================
// Catmull-Rom evaluation of a world polyline, parameterized by NORMALIZED ARC LENGTH tau in [0,1].
// ===================================================================================================
template <class Real> struct ArcPolyline {
  std::vector<Vec3<Real>> P; std::vector<Real> s;    // cumulative arc length, s[0]=0, s.back()=total
  Real total = 0;
  void init(const std::vector<Vec3<Real>>& pts) {
    P = pts; s.assign(pts.size(), 0);
    for (size_t k = 1; k < pts.size(); k++) s[k] = s[k-1] + gnet::nrm(gnet::sub(pts[k], pts[k-1]));
    total = s.empty() ? 0 : s.back();
  }
  Vec3<Real> eval(Real tau) const {
    if (P.size() == 1) return P[0];
    const Real target = tau*total;
    size_t i = 0; while (i+1 < s.size() && s[i+1] < target) i++;
    if (i+1 >= P.size()) return P.back();
    const Real seg = s[i+1]-s[i]; const Real w = (double)seg>1e-30 ? (target-s[i])/seg : (Real)0;
    // Catmull-Rom with clamped neighbors
    const Vec3<Real>& p1 = P[i]; const Vec3<Real>& p2 = P[i+1];
    const Vec3<Real>& p0 = P[i>0 ? i-1 : i]; const Vec3<Real>& p3 = P[i+2<P.size() ? i+2 : i+1];
    const Real w2 = w*w, w3 = w2*w;
    auto cr = [&](Real a, Real b, Real c, Real d) -> Real {
      return (Real)0.5*((2*b) + (-a+c)*w + (2*a-5*b+4*c-d)*w2 + (-a+3*b-3*c+d)*w3);
    };
    return Vec3<Real>{cr(p0[0],p1[0],p2[0],p3[0]), cr(p0[1],p1[1],p2[1],p3[1]), cr(p0[2],p1[2],p2[2],p3[2])};
  }
};

// Forward declaration (defined below): straight tapered fiber, used as the fold-safe fallback.
template <class Real> void append_straight_fiber(const ArmSeam<Real>& seamA, const Vec3<Real>& endC,
    Real rA, Real rB, Integer n_axial, Long cheb, Long fourier,
    Vector<Long>& elem_order, Vector<Long>& forder, Vector<Real>& coord, Vector<Real>& radius, Vector<Real>& orient);

// ===================================================================================================
// One bent, tapered, centerline-following slender arm appended to (elem_order,forder,coord,radius,orient).
// The centerline r(t): straight panel-aligned lead off seamA along seamA.u, Catmull-Rom middle following
// `cl` (endpoint-pinned to the lead ends), straight lead into `endC` along the travel direction endTang.
// Radius tapers rA->rB. Orientation = double-reflection rotation-minimizing frame seeded at seamA.e1.
// t=0 -> seamA.C (tangent seamA.u), t=1 -> endC (tangent endTang): terminal rings conform (watertight).
// ===================================================================================================
template <class Real> void append_centerline_fiber(const ArmSeam<Real>& seamA, const Vec3<Real>& endC,
    const Vec3<Real>& endTang, const std::vector<Vec3<Real>>& cl_in, Real rA, Real rB,
    Integer n_axial, Integer lead_panels, Long cheb, Long fourier,
    Vector<Long>& elem_order, Vector<Long>& forder, Vector<Real>& coord, Vector<Real>& radius, Vector<Real>& orient,
    bool constant_orient = false, Real lead_frac = (Real)0.18, Integer corner_panels = -1,
    Real turn_thresh_deg = (Real)20) {
  using namespace gnet;
  Integer n = std::max<Integer>(6, n_axial);
  Integer lp = std::max<Integer>(1, lead_panels);
  Integer cp = (corner_panels > 0) ? corner_panels : lp;
  const Vec3<Real> A0 = seamA.C;
  const Vec3<Real> chord = sub(endC, A0); const Real L = nrm(chord);
  const Vec3<Real> uA = unit(seamA.u);
  const Vec3<Real> uB = scal((Real)-1, unit(endTang));                   // junction B's arm-exit axis

  // TURN-ADAPTIVE corner: how far each mouth axis must swing to reach the straight chord. A gentle arm
  // (turn <= 35 deg) keeps the short lead + base corner; a sharp arm gets (a) MORE corner panels to
  // resolve the tighter bend and (b) a LONGER lead == larger corner radius == more legroom to swing out
  // and hit the graph centerline. Extra corner panels are ADDED to n (2 per corner) so the straight run
  // is unchanged. Lead grows with the turn, capped so the run never vanishes.
  Real lead_frac_eff = std::max<Real>((Real)0, lead_frac);
  if ((double)L > 0) {
    const Vec3<Real> cd = scal((Real)1/L, chord);
    auto ang = [](const Vec3<Real>& p, const Vec3<Real>& q){ double d=(double)(p[0]*q[0]+p[1]*q[1]+p[2]*q[2]); d=std::max(-1.0,std::min(1.0,d)); return std::acos(d)*180.0/M_PI; };
    const double turn = std::max(ang(uA, cd), ang(uB, scal((Real)-1, cd)));
    const double thr = std::max(1.0, (double)turn_thresh_deg);
    if (turn > thr) {
      const double f = std::min(turn/thr, 3.0);                          // scale factor, capped 3x
      const Integer extra = std::min<Integer>((Integer)std::lround((f-1.0)*cp), 3*cp);   // added corner panels/side
      cp += extra; n += 2*extra;                                         // more corner resolution, run unchanged
      lead_frac_eff = std::min<Real>((Real)(lead_frac*f), (Real)0.40);   // more legroom, capped at 0.40*L
    }
  }
  if (2*(lp+cp) >= n) { lp = std::max<Integer>(1, n/6); cp = lp; }        // keep a straight run in the middle

  // LEAD-CORNER-STRAIGHT racetrack, robust at ANY seam-axis/chord angle -- EVERY interior arm bends (no
  // straight fallback).  It leaves junction A along A's arm-exit axis seamA.u for a SHORT lead, corners
  // onto the straight run, then corners into junction B's arm-exit axis (-endTang) for a short lead, so
  // both terminal rings conform to the junctions' dictated arm directions (the bifurcation bend).
  //
  //   * Lead LENGTH = lead_frac * (junction-to-junction distance L)  -- so it SCALES with the arm and the
  //     turn stays LOCALIZED near the junction instead of stretching across the whole arm.
  //   * Lead length is DECOUPLED from the panel count: raising lead_panels only refines the (still short)
  //     lead to RESOLVE the turn, it does NOT elongate it.  (The old code tied the lead reach to the panel
  //     fraction, L_reach = t1*L/(1-t1) ~ L/3, which both over-stretched the turn and, via a steep-arm
  //     guard, dropped most arms to a plain straight tube -- the "most arms look straight" symptom.)
  const Real Llead = lead_frac_eff * L;
  const Vec3<Real> QA = add(A0,   scal(Llead, uA));                      // end of lead A
  const Vec3<Real> QB = add(endC, scal(Llead, uB));                      // end of lead B
  // Stable lead-corner-straight base: straight lead A0->QA coaxial with seamA.u, straight run QA->QB,
  // straight lead QB->endC coaxial with -endTang, smootherstep corners. Geometrically clean (watertight
  // ~228, no near-folds). A gentle single-midpoint sin^2 tilt toward the vmtk arc midpoint gives a light
  // centerline lean without the overshoot/fold of a full spline. (Full vmtk-tracing splines -- windowed
  // offset and clamped Catmull-Rom -- were tried and BOTH worsened the geometry: kinks -> 669, CR
  // overshoot -> 2597. Left out pending targeted per-junction feedback.)
  const Real t1 = (Real)(lp + cp*(Real)0.5)/n, t2 = (Real)1 - t1;
  const Real e1L=(Real)lp/n, e1R=(Real)(lp+cp)/n, e2L=(Real)(n-lp-cp)/n, e2R=(Real)(n-lp)/n;
  auto lerp = [&](const Vec3<Real>& p, const Vec3<Real>& q, Real w){ return add(scal((Real)1-w, p), scal(w, q)); };
  auto lineA = [&](Real x){ return lerp(A0, QA, x/t1); };
  auto lineR = [&](Real x){ return lerp(QA, QB, (x-t1)/(t2-t1)); };
  auto lineB = [&](Real x){ return lerp(QB, endC, (x-t2)/((Real)1-t2)); };
  auto base = [&](Real t) -> Vec3<Real> {
    if (t <= e1L) return lineA(t);
    if (t <  e1R) return lerp(lineA(t), lineR(t), HybridAssembly<Real>::pou_ramp((t-e1L)/(e1R-e1L)));
    if (t <= e2L) return lineR(t);
    if (t <  e2R) return lerp(lineR(t), lineB(t), HybridAssembly<Real>::pou_ramp((t-e2L)/(e2R-e2L)));
    return lineB(t);
  };
  // Clean lead-corner-straight-corner-lead: the base is a single monotonic bend (smootherstep corners do
  // not overshoot), so the arm no longer wiggles up-down-up. The turn-adaptive corner (above) already
  // gives sharp arms the legroom to reach the graph centerline near the junction, so the previous sin^2
  // midpoint tilt toward the vmtk arc-midpoint -- which added the redundant mid-arm hump -- is dropped.
  (void)cl_in;
  auto curve = [&](Real t) -> Vec3<Real> { return base(t); };
  // Rotation-minimizing frame (double reflection, Wang et al.) tabulated on a fine grid.
  const Integer Ng = std::max<Integer>((Integer)200, (Integer)20*n);
  auto tang = [&](Integer i) -> Vec3<Real> {
    const Real h = (Real)1/Ng;
    if (i <= 0)  return unit(sub(curve(h), curve((Real)0)));
    if (i >= Ng) return unit(sub(curve((Real)1), curve((Real)1-h)));
    return unit(sub(curve((i+1)*h), curve((i-1)*h)));
  };
  std::vector<Vec3<Real>> Rtab((size_t)Ng+1);
  Vec3<Real> ti = tang(0);
  Vec3<Real> r = unit(sub(seamA.e1, scal(dot(seamA.e1, ti), ti)));
  if ((double)nrm(r) < 1e-9) { Vec3<Real> a{1,0,0}; if (std::fabs((double)ti[0])>0.9) a = Vec3<Real>{0,1,0}; r = unit(sub(a, scal(dot(a,ti), ti))); }
  Rtab[0] = r;
  for (Integer i = 0; i < Ng; i++) {
    const Vec3<Real> xi = curve(i/(Real)Ng), xi1 = curve((i+1)/(Real)Ng);
    const Vec3<Real> ti1 = tang(i+1);
    const Vec3<Real> v1 = sub(xi1, xi); const Real c1 = dot(v1, v1);
    const Vec3<Real> rL = (double)c1>0 ? sub(r,  scal(2*dot(v1,r )/c1, v1)) : r;
    const Vec3<Real> tL = (double)c1>0 ? sub(ti, scal(2*dot(v1,ti)/c1, v1)) : ti;
    const Vec3<Real> v2 = sub(ti1, tL); const Real c2 = dot(v2, v2);
    Vec3<Real> r1 = (double)c2>0 ? sub(rL, scal(2*dot(v2,rL)/c2, v2)) : rL;
    r1 = unit(r1); Rtab[(size_t)i+1] = r1; r = r1; ti = ti1;
  }
  const Vec3<Real> r0const = Rtab[0];
  auto orient_of = [&](Real t) -> Vec3<Real> {
    if (constant_orient) return r0const;
    Real g = t*Ng; if ((double)g < 0) g = 0; if (g > (Real)Ng) g = (Real)Ng;
    Integer i = (Integer)g; if (i >= Ng) i = Ng-1; const Real f = g - i;
    return unit(add(scal((1-f), Rtab[(size_t)i]), scal(f, Rtab[(size_t)i+1])));
  };
  // Sample Chebyshev-per-panel.
  const Vector<Real>& cn = SlenderElemList<Real>::CenterlineNodes(cheb);
  for (Integer p = 0; p < n; p++) {
    elem_order.PushBack(cheb); forder.PushBack(fourier);
    for (Long j = 0; j < cheb; j++) {
      const Real t = (p + cn[j]) / (Real)n;
      const Vec3<Real> Pc = curve(t); const Vec3<Real> o = orient_of(t);
      coord.PushBack(Pc[0]); coord.PushBack(Pc[1]); coord.PushBack(Pc[2]);
      radius.PushBack(rA + t*(rB - rA));
      orient.PushBack(o[0]); orient.PushBack(o[1]); orient.PushBack(o[2]);
    }
  }
}

// A STRAIGHT constant/tapered fiber seamA.C -> endC (debug A/B vs the bent one; mirrors add_free_arm's
// straight fiber exactly: axis = seamA.u for a leaf, orient = constant seamA.e1). No RMF, no blend.
template <class Real> void append_straight_fiber(const ArmSeam<Real>& seamA, const Vec3<Real>& endC,
    Real rA, Real rB, Integer n_axial, Long cheb, Long fourier,
    Vector<Long>& elem_order, Vector<Long>& forder, Vector<Real>& coord, Vector<Real>& radius, Vector<Real>& orient) {
  using namespace gnet;
  const Vec3<Real> d = sub(endC, seamA.C); const Real L = nrm(d); const Vec3<Real> u = unit(d);
  // orient perp to u (project seamA.e1)
  Vec3<Real> e1 = unit(sub(seamA.e1, scal(dot(seamA.e1, u), u)));
  if ((double)nrm(e1) < 1e-9) { Vec3<Real> a{1,0,0}; if (std::fabs((double)u[0])>0.9) a=Vec3<Real>{0,1,0}; e1 = unit(sub(a, scal(dot(a,u),u))); }
  const Vector<Real>& cn = SlenderElemList<Real>::CenterlineNodes(cheb);
  for (Integer p = 0; p < n_axial; p++) {
    elem_order.PushBack(cheb); forder.PushBack(fourier);
    for (Long j = 0; j < cheb; j++) {
      const Real t = (p + cn[j]) / (Real)n_axial; const Real s = t*L;
      coord.PushBack(seamA.C[0]+s*u[0]); coord.PushBack(seamA.C[1]+s*u[1]); coord.PushBack(seamA.C[2]+s*u[2]);
      radius.PushBack(rA + t*(rB - rA));
      orient.PushBack(e1[0]); orient.PushBack(e1[1]); orient.PushBack(e1[2]);
    }
  }
}

// ===================================================================================================
// Per-junction bundle I/O.  <prefix>-jNN.mesh = placed junction QuadElemList (body + owned leaf caps);
// <prefix>-jNN.arms = the owned bent arms, stored as raw CSBQ slender arrays (exact reload) + per-arm
// connectivity metadata (the neighbor node id, and whether the far end is a cap).
// ===================================================================================================
template <class Real> struct NetworkArmBundle {
  Integer order = 0;
  Long cheb = 10, fourier = 12;
  std::vector<Integer> other_node;                  // per owned arm: the node at the far end
  std::vector<Integer> is_cap;                      // per owned arm: 1 if far end is a leaf cap
  std::vector<Long> npanel;                          // per owned arm: panel count
  Vector<Long> elem_order, forder;                   // concatenated across arms (CSBQ ctor input)
  Vector<Real> coord, radius, orient;
};

template <class Real> void WriteNetworkBundle(const std::string& prefix, Integer jid,
    const QuadElemList<Real>& junc, const NetworkArmBundle<Real>& arms, const Comm& comm = Comm::Self()) {
  std::ostringstream nm; nm << prefix << "-j" << std::setw(3) << std::setfill('0') << jid;
  junc.Write(nm.str() + ".mesh", comm);
  if (comm.Rank()) return;
  std::ofstream f(nm.str() + ".arms"); SCTL_ASSERT_MSG(f.good(), "WriteNetworkBundle: cannot open .arms");
  f << std::setprecision(17);
  f << "# gen-network arm bundle v1\n";
  f << "# junction " << jid << " ; order narm cheb fourier\n";
  const Integer narm = (Integer)arms.npanel.size();
  f << arms.order << " " << narm << " " << arms.cheb << " " << arms.fourier << "\n";
  f << "# per arm: other_node is_cap npanel\n";
  Long node_off = 0, panel_off = 0;
  for (Integer a = 0; a < narm; a++) {
    f << arms.other_node[a] << " " << arms.is_cap[a] << " " << arms.npanel[a] << "\n";
    for (Long p = 0; p < arms.npanel[a]; p++, panel_off++) {
      f << arms.elem_order[panel_off] << " " << arms.forder[panel_off] << "\n";
      for (Long j = 0; j < arms.elem_order[panel_off]; j++, node_off++)
        f << arms.coord[node_off*3] << " " << arms.coord[node_off*3+1] << " " << arms.coord[node_off*3+2] << " "
          << arms.radius[node_off] << " "
          << arms.orient[node_off*3] << " " << arms.orient[node_off*3+1] << " " << arms.orient[node_off*3+2] << "\n";
    }
  }
}

// Exact inverse of WriteNetworkBundle: reload a per-junction bundle WITHOUT re-forming any mesh.
//   - the junction BODY comes back through the MPI-aware QuadElemList reader; its global coord0
//     (element-major, node, xyz -- exactly what QuadElemList::Init consumes) is returned in `junc_coord`;
//   - the ARMS come back as the raw CSBQ slender arrays stored by WriteNetworkBundle, ready to hand
//     straight to the SlenderElemList ctor (no BuildGenArmsSlender / BuildArmsSlenderFromTable rebuild --
//     these are the NETWORK's bent centerline arms, not the geom-test straight limbs).
// Returns false if <prefix>-jNN.mesh does not exist (so a caller can scan a range of ids).
template <class Real> bool ReadNetworkBundle(const std::string& prefix, Integer jid,
    Vector<Real>& junc_coord, Integer& order_out, NetworkArmBundle<Real>& arms,
    const Comm& comm = Comm::Self()) {
  std::ostringstream nm; nm << prefix << "-j" << std::setw(3) << std::setfill('0') << jid;
  { std::ifstream probe(nm.str() + ".mesh"); if (!probe.good()) return false; }

  QuadElemList<Real> q;
  q.template Read<Real>(nm.str() + ".mesh", comm);      // MPI-aware loader (rank-sliced if comm is distributed)
  order_out = q.Order();
  q.GetNodeCoord(&junc_coord, nullptr, nullptr);        // coord0 layout, exactly what the mpi-aware builder wants

  std::ifstream f(nm.str() + ".arms");
  SCTL_ASSERT_MSG(f.good(), "ReadNetworkBundle: cannot open " + nm.str() + ".arms");
  auto next = [&](std::string& l) -> bool {
    while (std::getline(f, l)) { size_t p = l.find_first_not_of(" \t\r\n");
      if (p != std::string::npos && l[p] != '#') return true; } return false; };

  arms = NetworkArmBundle<Real>{};
  std::string line; SCTL_ASSERT_MSG(next(line), "ReadNetworkBundle: truncated header");
  Integer aorder = 0, narm = 0; { std::istringstream s(line); s >> aorder >> narm >> arms.cheb >> arms.fourier; }
  arms.order = aorder;
  for (Integer a = 0; a < narm; a++) {
    SCTL_ASSERT_MSG(next(line), "ReadNetworkBundle: truncated arm header");
    Integer other = 0, iscap = 0; Long npan = 0; { std::istringstream s(line); s >> other >> iscap >> npan; }
    arms.other_node.push_back(other); arms.is_cap.push_back(iscap); arms.npanel.push_back(npan);
    for (Long p = 0; p < npan; p++) {
      SCTL_ASSERT_MSG(next(line), "ReadNetworkBundle: truncated panel header");
      Long eo = 0, fo = 0; { std::istringstream s(line); s >> eo >> fo; }
      arms.elem_order.PushBack(eo); arms.forder.PushBack(fo);
      for (Long j = 0; j < eo; j++) {
        SCTL_ASSERT_MSG(next(line), "ReadNetworkBundle: truncated node line");
        Real x, y, z, r, ox, oy, oz; { std::istringstream s(line); s >> x >> y >> z >> r >> ox >> oy >> oz; }
        arms.coord.PushBack(x); arms.coord.PushBack(y); arms.coord.PushBack(z);
        arms.radius.PushBack(r);
        arms.orient.PushBack(ox); arms.orient.PushBack(oy); arms.orient.PushBack(oz);
      }
    }
  }
  return true;
}

}  // namespace quad_junctions
