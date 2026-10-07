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
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
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

// Build the canonical junction for one cluster at an EXPLICIT sigma (no auto rule). Returns the emit_caps
// =false body + per-arm seam frame. Split out of build_canonical so the sigma auto-selector can rebuild
// at several sigmas and keep the one that closes best.
template <class Real> CanonicalJunction<Real> build_canonical_sigma(const GraphCluster<Real>& cl, Integer order,
    Real level, Integer nref, Real eta_join, Integer Ns_trans, Real s_cap_arc, Integer Ncap,
    Real alpha_deg, Real clampf, Real sigma) {
  GenSpec<Real> spec;
  spec.arm_dir = cl.arm_dir;
  spec.alpha_deg = alpha_deg; spec.alpha_clamp_frac = clampf; spec.sigma = sigma;
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

// RELATIVE watertightness |int n dA| / area of a canonical body closed at its seams by hemisphere caps
// (scale-invariant: same as the placed-junction relative closure, which the network's absolute leak is
// this times placed-area). This is the sigma-selection objective.
template <class Real> double canon_rel_closure(const CanonicalJunction<Real>& C, Integer order) {
  Vector<Real> Xb = C.Xcanon;
  for (Integer k = 0; k < C.N; k++) add_cap_hemisphere_frame<Real>(Xb, C.seam(k), order, 2, (Real)0.40);
  QuadElemList<Real> el(order, Xb, Comm::Self());
  Vector<Real> X, Xn, wts, dist; Vector<Long> cnt; el.GetFarFieldNodes(X, Xn, wts, dist, cnt, (Real)1e-10);
  double A = 0, f[3] = {0,0,0};
  for (Long i = 0; i < wts.Dim(); i++) { A += (double)wts[i]; for (int k=0;k<3;k++) f[k] += (double)wts[i]*(double)Xn[i*3+k]; }
  return (A > 0) ? std::sqrt(f[0]*f[0]+f[1]*f[1]+f[2]*f[2]) / A : 1e30;
}

// Build the canonical junction for one cluster with ADAPTIVE sigma selection. The per-junction absolute
// watertightness leak is (relative closure) x (placed area ~ radius^2/meanR0^2 x area_canon); with the
// uniform radius the placed area is ~fixed, so MINIMIZING THE RELATIVE CLOSURE of the capped canonical
// minimizes the network leak. The gap-adaptive rule sigma=0.15*min_gap/110 is only a proxy for the min
// pairwise gap and MISSES the near-antiparallel case (a >~165 deg pair squeezes the Voronoi cells into a
// fold that sits on a sharp sigma cliff -- e.g. cluster 150 / j298, 173.7 deg pair: its auto sigma 0.057
// lands right at the cliff -> 2.8e-3 relative, while 0.050 gives 4e-8). So: start at the gap-adaptive
// sigma, measure the capped-body relative closure, and while it exceeds REL_TOL, thin sigma geometrically
// and rebuild (bounded tries), keeping the best. Wide well-formed junctions pass on the first sigma and
// are byte-identical to before; only the few pathological ones retry. QJ_CANON_REL_TOL / QJ_CANON_TRIES
// tune it; sigma_in>0 (explicit QJ_JSIGMA) skips the search entirely.
template <class Real> CanonicalJunction<Real> build_canonical(const GraphCluster<Real>& cl, Integer order,
    Real level, Integer nref, Real eta_join, Integer Ns_trans, Real s_cap_arc, Integer Ncap,
    Real alpha_deg, Real clampf, Real sigma_in) {
  Real min_gap = 180;
  for (size_t a = 0; a < cl.arm_dir.size(); a++) for (size_t b = a+1; b < cl.arm_dir.size(); b++) {
    Real c = gnet::dot(cl.arm_dir[a], cl.arm_dir[b]); c = std::max<Real>((Real)-1, std::min<Real>((Real)1, c));
    min_gap = std::min<Real>(min_gap, (Real)(std::acos((double)c)*180.0/M_PI));
  }
  const Real sig0 = std::max<Real>((Real)0.05, (Real)0.15*std::min<Real>((Real)1, min_gap/(Real)110));
  // Explicit override: build once, no search.
  if (sigma_in > 0) return build_canonical_sigma<Real>(cl, order, level, nref, eta_join, Ns_trans, s_cap_arc, Ncap, alpha_deg, clampf, sigma_in);

  const double REL_TOL = std::getenv("QJ_CANON_REL_TOL") ? atof(std::getenv("QJ_CANON_REL_TOL")) : 1e-7;
  const Integer TRIES  = std::getenv("QJ_CANON_TRIES")   ? (Integer)atoi(std::getenv("QJ_CANON_TRIES")) : 6;
  const Real    THIN   = (Real)0.9;                                  // geometric thinning per retry
  const Real    SIG_FLOOR = (Real)0.02;                             // don't over-thin into a degenerate mesh
  Real sig = sig0;
  CanonicalJunction<Real> best = build_canonical_sigma<Real>(cl, order, level, nref, eta_join, Ns_trans, s_cap_arc, Ncap, alpha_deg, clampf, sig);
  double best_rel = canon_rel_closure<Real>(best, order); Real best_sig = sig;
  for (Integer t = 0; t < TRIES && best_rel > REL_TOL; t++) {
    sig *= THIN; if ((double)sig < (double)SIG_FLOOR) break;
    CanonicalJunction<Real> cand = build_canonical_sigma<Real>(cl, order, level, nref, eta_join, Ns_trans, s_cap_arc, Ncap, alpha_deg, clampf, sig);
    double rel = canon_rel_closure<Real>(cand, order);
    if (rel < best_rel) { best_rel = rel; best = cand; best_sig = sig; }
  }
  if (std::getenv("QJ_CLUDIAG"))
    std::cout << "    [canon] deg=" << cl.degree << " min_gap=" << std::setprecision(4) << (double)min_gap
              << " sig0=" << (double)sig0 << " -> sig=" << (double)best_sig << " rel_closure=" << best_rel << "\n";
  return best;
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
  Real total = 0; Real alpha = (Real)0.5;            // knot exponent: 0=uniform CR, 0.5=CENTRIPETAL (no
                                                     // cusps/overshoot -- the fix for the prior full-trace blowup)
  // C2 mode (opt-in): the centripetal Catmull-Rom above is only C1 -- curvature JUMPS at every control knot.
  // Each degree-10 CSBQ panel then straddles those knots (they don't align to panel boundaries), so on a
  // high-tortuosity arm (a curved cap) the per-panel polynomial reconstructs a wrong curvature => the DL
  // const-density identity fails on the mid-body (~0.5 free-edge, curvature-magnitude-INDEPENDENT). The M8
  // hard rule. Fix: interpolate the SAME knots with a globally-C2 natural cubic spline (2nd-derivative-
  // continuous everywhere, zero curvature at the ends so the straight leads/dome conform). Curvature-limited
  // smoothing (QJ_ARM_CL_SMOOTH) still runs first on the control points, so the C2 curve stays rho>r AND curved.
  bool c2 = false; std::vector<Vec3<Real>> M;         // per-knot 2nd derivative wrt arclength (empty => C1 CR)
  void init(const std::vector<Vec3<Real>>& pts) {
    P = pts; s.assign(pts.size(), 0);
    for (size_t k = 1; k < pts.size(); k++) s[k] = s[k-1] + gnet::nrm(gnet::sub(pts[k], pts[k-1]));
    total = s.empty() ? 0 : s.back();
    M.clear();
    if (c2 && P.size() >= 3) {                        // natural cubic spline: solve tridiagonal for M (per component)
      const size_t n = P.size();
      std::vector<Real> h(n-1); for (size_t i=0;i+1<n;i++) h[i]=std::max((Real)1e-30, s[i+1]-s[i]);
      M.assign(n, Vec3<Real>{(Real)0,(Real)0,(Real)0});
      for (int d=0; d<3; d++) {                        // Thomas solve for interior M[1..n-2], natural M0=M_{n-1}=0
        std::vector<Real> a(n,0), b(n,0), c(n,0), rhs(n,0), m(n,0);
        b[0]=1; b[n-1]=1;                              // natural BC
        for (size_t i=1;i+1<n;i++) {
          a[i]=h[i-1]; b[i]=2*(h[i-1]+h[i]); c[i]=h[i];
          rhs[i]=6*(((double)P[i+1][d]-(double)P[i][d])/(double)h[i]
                    -((double)P[i][d]-(double)P[i-1][d])/(double)h[i-1]);
        }
        for (size_t i=1;i<n;i++) { const Real w=(double)b[i-1]>1e-30?a[i]/b[i-1]:(Real)0;
          b[i]-=w*c[i-1]; rhs[i]-=w*rhs[i-1]; }
        m[n-1]=(double)b[n-1]>1e-30?rhs[n-1]/b[n-1]:(Real)0;
        for (size_t i=n-1;i-->0;) m[i]=(double)b[i]>1e-30?(rhs[i]-c[i]*m[i+1])/b[i]:(Real)0;
        for (size_t i=0;i<n;i++) M[i][d]=m[i];
      }
    }
  }
  Vec3<Real> eval(Real tau) const {
    if (P.size() == 1) return P[0];
    if (c2 && !M.empty()) {                            // C2 natural cubic spline
      const Real target = tau*total;
      size_t i = 0; while (i+1 < s.size() && s[i+1] < target) i++;
      if (i+1 >= P.size()) return P.back();
      const Real h = std::max((Real)1e-30, s[i+1]-s[i]);
      const Real a = (s[i+1]-target)/h, b = (target-s[i])/h;
      const Real ca = (a*a*a-a)*h*h/(Real)6, cb = (b*b*b-b)*h*h/(Real)6;
      return gnet::add(gnet::add(gnet::scal(a, P[i]), gnet::scal(b, P[i+1])),
                       gnet::add(gnet::scal(ca, M[i]), gnet::scal(cb, M[i+1])));
    }
    const Real target = tau*total;
    size_t i = 0; while (i+1 < s.size() && s[i+1] < target) i++;
    if (i+1 >= P.size()) return P.back();
    const Real seg = s[i+1]-s[i]; const Real w = (double)seg>1e-30 ? (target-s[i])/seg : (Real)0;
    // Non-uniform (Barry-Goldman) Catmull-Rom over the 4-point neighborhood, knot spacing |dP|^alpha.
    // alpha=0 reduces to the classic uniform CR; alpha=0.5 (default) is the centripetal spline, which is
    // provably free of self-intersections/overshoot -- so a curved vmtk arm cannot loop back on itself.
    const Vec3<Real>& p1 = P[i]; const Vec3<Real>& p2 = P[i+1];
    const Vec3<Real>& p0 = P[i>0 ? i-1 : i]; const Vec3<Real>& p3 = P[i+2<P.size() ? i+2 : i+1];
    auto knot = [&](const Vec3<Real>& a, const Vec3<Real>& b, Real t0) -> Real {
      const Real d = gnet::nrm(gnet::sub(b, a));
      const Real dt = (Real)std::pow(std::max(1e-300, (double)d), (double)alpha);
      return t0 + std::max((Real)1e-9, dt);                       // guard coincident points
    };
    const Real t0 = 0, t1 = knot(p0,p1,t0), t2 = knot(p1,p2,t1), t3 = knot(p2,p3,t2);
    const Real t = t1 + w*(t2 - t1);
    auto mix = [&](const Vec3<Real>& a, const Vec3<Real>& b, Real ta, Real tb) -> Vec3<Real> {
      const Real d = (double)(tb-ta)>1e-30 ? (tb - t)/(tb - ta) : (Real)0.5;
      return gnet::add(gnet::scal(d, a), gnet::scal((Real)1 - d, b));
    };
    const Vec3<Real> A1 = mix(p0,p1,t0,t1), A2 = mix(p1,p2,t1,t2), A3 = mix(p2,p3,t2,t3);
    const Vec3<Real> B1 = mix(A1,A2,t0,t2), B2 = mix(A2,A3,t1,t3);
    return mix(B1,B2,t1,t2);
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
    Real turn_thresh_deg = (Real)20, std::vector<Vec3<Real>>* sampled_out = nullptr,
    std::vector<Vec3<Real>>* sampled_ref = nullptr, Vec3<Real> endE1 = Vec3<Real>{(Real)0,(Real)0,(Real)0},
    bool is_cap = false, Vec3<Real>* out_termC = nullptr, Vec3<Real>* out_termU = nullptr) {
  using namespace gnet;
  Integer n = std::max<Integer>(6, n_axial);
  Integer lp = std::max<Integer>(1, lead_panels);
  Integer cp = (corner_panels > 0) ? corner_panels : lp;
  const Vec3<Real> A0 = seamA.C;
  const Vec3<Real> chord = sub(endC, A0); const Real L = nrm(chord);
  const Vec3<Real> uA = unit(seamA.u);
  const Vec3<Real> uB = scal((Real)-1, unit(endTang));                   // junction B's arm-exit axis

  // ---- DISPLACEMENT-BLEND arm (QJ_ARM_BLEND, the production centerline scheme). KEEP the raw vmtk
  // centerline v(l) everywhere and correct only near the two seams: a SHORT coaxial lead along the junction
  // axis u (axis-conforming / watertight mouth ring), a quintic-Hermite corner rotating u onto the vmtk
  // tangent, then the RAW vmtk run v(l) (zero displacement in the middle), mirrored at the far mouth. The
  // corner is kept GENTLE by the turn guard below (arclength lengthening + per-panel tangent cap). When
  // QJ_ARM_BLEND is unset the function falls back to the plain straight lead-corner-straight racetrack.
  //   (Earlier experimental schemes -- QJ_ARM_FOLLOW canonical-Hermite/vmtk-warp, the two-sided combine, and
  //   the pure position ramp -- were removed 2026-09-23; the coaxial-lead blend is the sole followed path.)
  // QJ_ARM_BLEND: fixed to production ON (was env; baked 2026-10-07 as the seamsettle_tr6 baseline)
  const bool blend = true && (double)L > 1e-12 && cl_in.size() >= 2;

  // TURN-ADAPTIVE corner: how far each mouth axis must swing to reach the straight chord. A gentle arm
  // (turn <= 35 deg) keeps the short lead + base corner; a sharp arm gets (a) MORE corner panels to
  // resolve the tighter bend and (b) a LONGER lead == larger corner radius == more legroom to swing out
  // and hit the graph centerline. Extra corner panels are ADDED to n (2 per corner) so the straight run
  // is unchanged. Lead grows with the turn, capped so the run never vanishes.
  Real lead_frac_eff = std::max<Real>((Real)0, lead_frac);
  if (!blend && (double)L > 0) {
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
  //
  // NB the breakpoints are recomputed from the LIVE lp/cp/n each call, so the corner-curvature guard below
  // can widen the corner window (cp) in place and have `base` reflect it immediately.
  //
  // ---- Corner Hermite handle for the blend coaxial-lead corners (QJ_ARM_CORNER_KR). ----
  const char* ck = std::getenv("QJ_ARM_CORNER_KR");
  const Real corner_kr = ck ? (Real)std::atof(ck) : (Real)0.4;   // blend-corner Hermite handle = kr*|corner chord|
  // corner scheme: fixed to the seam-settle piecewise lead+Hermite path (straight coaxial lead + separate hermite5
  // corner + centripetal-CR run); merge/smooth alternatives and the QJ_ARM_LC_HANDLE knob removed 2026-10-07 --
  // backfired, see gen_network_geom.hpp network notes / network-corner-scheme-comparison memory.
  // ---- FERGUSON tangent-magnitude clamp (QJ_ARM_FERGUSON, default off) ----
  // The corner Hermite rotates the fixed junction axis onto the vmtk tangent over the chord Q<->P. With a
  // FIXED handle h=corner_kr*|chord| for BOTH ends (the default), an endpoint tangent that points AWAY from
  // that chord (dot<0, the sharp-swing case) drives the quintic to bulge backward -> the meshed centerline
  // doubles back and the tube contacts itself (the mid-arm HAIRPIN self-contact, e.g. edge 271-277, which is
  // the largest DL-identity error and is INVISIBLE to rho/r since the centerline stays gently curved). Ferguson's
  // rule scales each end's tangent magnitude to the chord AND to that end's forward projection on the chord, so
  // a backward-pointing tangent gets a SHORT handle (no backward overshoot) while an aligned tangent keeps the
  // full corner_kr*|chord|. Per-end: h_i = corner_kr*|chord|*clamp(t_i.chordhat, floor, cap). floor keeps the
  // handle from collapsing to a kink; cap (Ferguson =1) keeps it from exceeding the chord. Default-off => the
  // two h's stay = corner_kr*|chord| (byte-identical to before).
  //   MEASURED 2026-09-28 (default OFF): NEUTRAL on the reported "hairpin" (edge 271-277 dist/r -0.18 unchanged;
  //   rho/r 2.60 unchanged). corner_kr=0.4 already keeps the handle well under the chord, so the overshoot the
  //   clamp targets was not the cause of that self-contact. Harmless; kept for the genuine sharp-swing corners.
  const bool ferguson = (std::getenv("QJ_ARM_FERGUSON") != nullptr);
  const Real ferg_floor = std::getenv("QJ_ARM_FERGUSON_FLOOR") ? (Real)std::atof(std::getenv("QJ_ARM_FERGUSON_FLOOR")) : (Real)0.10;
  const Real ferg_cap   = std::getenv("QJ_ARM_FERGUSON_CAP")   ? (Real)std::atof(std::getenv("QJ_ARM_FERGUSON_CAP"))   : (Real)1.00;
  // ---- DISPLACEMENT-BLEND precompute (QJ_ARM_BLEND): keep the raw vmtk centerline v(l) everywhere and
  // correct only near the two seams via a SHORT coaxial lead + Hermite corner onto v. A short straight lead
  // A0->QA along the junction axis u makes the mouth ring axis == u (AXIS-conforming / watertight seam); a
  // quintic-Hermite corner rotates u onto the vmtk tangent; then the RAW vmtk run v(l) (zero displacement in
  // the middle), mirrored at the far mouth. Lead length QJ_ARM_BLEND_LEAD (default 1 arm radius, floored
  // positive), capped to a fraction of the chord; entry/exit pinned to the nearest vmtk arclength inside QA/QB.
  ArcPolyline<Real> bcl;
  const char* blk = std::getenv("QJ_ARM_BLEND_LEAD");   // coaxial-lead length; default 1 arm radius (scale-inv stub)
  const Real blead_len = std::max<Real>((Real)1e-6*L, blk ? (Real)std::atof(blk) : std::max(rA, rB));
  Vec3<Real> bQA{0,0,0}, bQB{0,0,0}; Real bse = 0, bsx = 1, bLl = (Real)0;   // lead ends + vmtk entry/exit arclength + lead length
  // ---- Curvature-limited centerline SMOOTHING (QJ_ARM_CL_SMOOTH, default off => bcl.init(cl_in) unchanged).
  // Where the raw vmtk polyline kinks tighter than the tube can wrap (local curvature radius rho < r*target),
  // the meshed tube self-folds (inner wall crosses its own axis => DL near-quad garbage). A lumen of radius r
  // physically cannot bend at rho<r, so a deep kink is a centerline artifact; relax ONLY those points toward
  // their neighbour midpoint, by an amount that vanishes as rho -> target (a curvature limiter, not a global
  // smoother: any point already at rho>=target does not move, so the vmtk path is kept everywhere except the
  // kinks). Endpoints are fixed (the seam pins bse/bsx are taken from the smoothed curve just below, so they
  // stay consistent). Radius is untouched => closure-neutral. Residual folds (if any) are cleaned by the
  // seam-safe radius taper QJ_ARM_RADIUS_DIP downstream ("smooth then taper").
  //
  // CURVATURE METRIC (2026-09-25 fix): the limiter must measure the curvature of the CURVE CSBQ ACTUALLY
  // MESHES -- the centripetal Catmull-Rom spline through the control points (bcl, below), NOT the control
  // polygon. The 3-point circumradius of consecutive CONTROL points over-estimates rho: the centripetal
  // spline bends ~2x tighter BETWEEN knots, so "control rho>=1.6r" left the meshed spline at rho~0.88r --
  // still self-folding (the j169 leaf-cap bend near (-504,243,-187): control 1.6 vs spline 0.88). We instead
  // estimate rho at node k from the SPLINE, sampled at three knot-bracketing pairs of the two adjacent
  // segments (min over them = the tightest osculating radius the tube must survive). Search-free (segment
  // known) => O(npt)/sweep, same cost as before. Same lesson as diag2.py's "resample, don't trust cheb-node
  // spacing", now applied to the smoother's own convergence test. See [[network-greens-arm-selfintersect]].
  std::vector<Vec3<Real>> cl_s = cl_in;
  // ---- Shared centerline-curvature model for the two smoothers below (QJ_ARM_CL_SMOOTH curvature limiter and
  // QJ_ARM_CL_BRIDGE near-seam bend bypass). Both must measure the curvature of the CENTRIPETAL Catmull-Rom
  // spline CSBQ actually meshes (bcl), NOT the control polygon. cl_rho3 = 3-point circumradius; cl_seg_pt = CR
  // spline point in segment [i,i+1] at fractional w (mirrors ArcPolyline::eval EXACTLY -- same Barry-Goldman
  // knot/mix -- but for a KNOWN segment, no arclength search); cl_rho_spline = min osculating radius at node k
  // over three knot-bracketing sample pairs of the two adjacent segments (samples pulled toward the knot where
  // a control turn bends the spline tightest; the MIN is the conservative radius the tube must clear).
  auto cl_rho3 = [&](const Vec3<Real>& a, const Vec3<Real>& b, const Vec3<Real>& c) -> Real {
    const Vec3<Real> cr = cross(sub(b, a), sub(b, c)); const Real a2 = nrm(cr);
    if ((double)a2 < 1e-30) return (Real)1e30;
    return nrm(sub(b, a)) * nrm(sub(b, c)) * nrm(sub(c, a)) / (2 * a2);
  };
  auto cl_seg_pt = [&](const std::vector<Vec3<Real>>& P, size_t i, Real w) -> Vec3<Real> {
    const size_t np = P.size();
    const Vec3<Real>& p1 = P[i]; const Vec3<Real>& p2 = P[i+1];
    const Vec3<Real>& p0 = P[i > 0 ? i-1 : i]; const Vec3<Real>& p3 = P[i+2 < np ? i+2 : i+1];
    auto knot = [&](const Vec3<Real>& a, const Vec3<Real>& b, Real t0) -> Real {
      const Real dt = (Real)std::pow(std::max(1e-300, (double)nrm(sub(b, a))), 0.5);   // alpha=0.5
      return t0 + std::max((Real)1e-9, dt);
    };
    const Real t0 = 0, t1 = knot(p0,p1,t0), t2 = knot(p1,p2,t1), t3 = knot(p2,p3,t2);
    const Real t = t1 + w*(t2 - t1);
    auto mix = [&](const Vec3<Real>& a, const Vec3<Real>& b, Real ta, Real tb) -> Vec3<Real> {
      const Real d = (double)(tb-ta) > 1e-30 ? (tb - t)/(tb - ta) : (Real)0.5;
      return add(scal(d, a), scal((Real)1 - d, b));
    };
    const Vec3<Real> A1 = mix(p0,p1,t0,t1), A2 = mix(p1,p2,t1,t2), A3 = mix(p2,p3,t2,t3);
    const Vec3<Real> B1 = mix(A1,A2,t0,t2), B2 = mix(A2,A3,t1,t3);
    return mix(B1,B2,t1,t2);
  };
  auto cl_rho_spline = [&](const std::vector<Vec3<Real>>& P, size_t k) -> Real {
    static const Real wl[3] = {(Real)0.5, (Real)(2.0/3.0), (Real)(5.0/6.0)};
    static const Real wr[3] = {(Real)0.5, (Real)(1.0/3.0), (Real)(1.0/6.0)};
    Real best = (Real)1e30;
    for (int m = 0; m < 3; m++)
      best = std::min(best, cl_rho3(cl_seg_pt(P, k-1, wl[m]), P[k], cl_seg_pt(P, k, wr[m])));
    return best;
  };
  {
    // QJ_ARM_CL_SMOOTH: fixed to production 2.5 (was env; baked 2026-10-07 as the seamsettle_tr6 baseline)
    const Real cse_val = (Real)2.5;
    if (blend && cl_s.size() >= 3) {
      const Real r_arm = std::max(rA, rB);
      const Real m_loc = std::getenv("QJ_CORNER_MARGIN") ? (Real)std::atof(std::getenv("QJ_CORNER_MARGIN")) : (Real)0.25;
      const Real smf = cse_val;
      const Real target = (smf > (Real)1 ? smf : ((Real)1 + m_loc)) * r_arm;   // want rho >= target everywhere
      // QJ_ARM_CL_SMOOTH_ITERS: fixed to production 1500 (was env; baked 2026-10-07 as the seamsettle_tr6 baseline)
      const Integer iters = (Integer)1500;
      const Real wmax = std::getenv("QJ_ARM_CL_SMOOTH_W") ? (Real)std::atof(std::getenv("QJ_ARM_CL_SMOOTH_W")) : (Real)0.6;
      // END BOOST: the sharp APPROACH BENDS live in the outer ~15% of the arm (the vessel's final curve into the
      // neighbour junction), right where the fixed endpoint stiffens the last CR segment so the uniform target
      // under-relaxes. Raise the target toward the ends (target*(1+endboost) at the very end, ramping to plain
      // target by end_span) so those points relax harder -- closure-neutral (radius untouched), and consistent
      // with the mouth already deviating from vmtk via the coaxial lead. QJ_ARM_CL_SMOOTH_ENDBOOST(0=off).
      // QJ_ARM_CL_SMOOTH_ENDBOOST: fixed to production 1.5 (was env; baked 2026-10-07 as the seamsettle_tr6 baseline)
      const Real endboost = (Real)1.5;
      // QJ_ARM_CL_SMOOTH_ENDSPAN: fixed to production 0.15 (was env; baked 2026-10-07 as the seamsettle_tr6 baseline)
      const Real end_span = (Real)0.15;
      const size_t npt = cl_s.size();
      Integer moved_total = 0;
      for (Integer it = 0; it < iters; it++) {
        Real worst = (Real)1e30; Integer moved = 0;
        std::vector<Vec3<Real>> up = cl_s;
        for (size_t k = 1; k + 1 < cl_s.size(); k++) {
          const Real rho = cl_rho_spline(cl_s, k);
          Real tgt_k = target;
          if ((double)endboost > 0) {
            const Real fend = (Real)std::min(k, npt-1-k) / (Real)std::max<size_t>(1, npt-1);   // 0 at ends -> ~0.5 mid
            const Real ew = (double)fend < (double)end_span ? ((Real)1 - fend/end_span) : (Real)0; // 1 at end -> 0 by end_span
            tgt_k = target * ((Real)1 + endboost * ew);
          }
          if ((double)rho < (double)tgt_k) {
            const Vec3<Real> mid = scal((Real)0.5, add(cl_s[k-1], cl_s[k+1]));
            Real def = (Real)1 - rho / tgt_k; if ((double)def < 0) def = 0; if ((double)def > 1) def = 1;
            const Real w = wmax * def;                                    // drive vanishes at rho==target
            up[k] = add(scal((Real)1 - w, cl_s[k]), scal(w, mid));
            worst = std::min(worst, rho); moved++;
          }
        }
        cl_s.swap(up); moved_total += moved;
        (void)worst;
        if (moved == 0) break;                                            // every point at/above its (boosted) target
      }
      if (std::getenv("QJ_CORNER"))
        std::cout << "  [cl-smooth] target rho>=" << (double)target << " (r_arm=" << (double)r_arm
                  << ") : " << (long)moved_total << " point-relaxations over <=" << (long)iters << " iters\n";
    }
  }
  // ---- NEAR-SEAM BEND BYPASS (QJ_ARM_CL_BRIDGE, default off). The curvature limiter above relaxes each
  // kinked node toward its neighbour midpoint, but near the SEAM the fixed endpoint stiffens the last CR
  // segments and a tight APPROACH BEND that sits in the first ~2 panels (right where the vessel dives into
  // the junction) cannot be relaxed enough -- it is too close to the coaxial-lead/corner region for either
  // the limiter or the PIN_CURV run-pin to reach. Remedy (per request): EXCISE the high-curvature section
  // near each end and REPLACE it with a smooth low-curvature interpolation that BYPASSES those panels, then
  // continue with the ORIGINAL vmtk centerline. MEASURED HARMFUL / INERT 2026-09-26 (default OFF): on n298 the
  // tight far bends are NOT in the vmtk run at all -- they sit in the synthesized far-CORNER panels (the u->v
  // swing onto the junction axis), which coax() builds from the junction axis, not from cl_s. So this bridge
  // left the far-half rho/r distribution byte-identical (min 0.869, 5 arms <1.0) AND raised closure 1.19->1.60.
  // Kept env-gated for the record; the actual tight bends are a CORNER property (see QJ_ARM_CORNER_RELAX note).
  // Concretely: locate the tightest spline node within a search
  // window of each end; grow the contiguous sub-target cluster; pad it outward by ~QJ_ARM_CL_BRIDGE_PAN
  // axial-panel-lengths to two ANCHOR nodes (kept, one on each side, in the gentle vessel); rebuild every
  // control node strictly between the anchors onto a quintic Hermite bridge whose endpoint tangents are the
  // KEPT vmtk tangents just outside the window. The bridge spreads the same net turn over the full
  // window chord => curvature ~ turn/chord, far below the vmtk spike, and is C1 with the retained vmtk on
  // both sides (no slit). Endpoints (seam centers) are NEVER moved => watertightness-neutral. Runs on cl_s
  // BEFORE bcl.init, so the corner Hermite anchor (bcl.eval(bse)), the raw run, and the RMF frame all see
  // the de-kinked centerline. Only the two ends are touched (search window), never the mid-arm.
  {
    const char* bre = std::getenv("QJ_ARM_CL_BRIDGE");
    if (blend && bre && cl_s.size() >= 5) {
      const Real r_arm = std::max(rA, rB);
      const Real m_loc = std::getenv("QJ_CORNER_MARGIN") ? (Real)std::atof(std::getenv("QJ_CORNER_MARGIN")) : (Real)0.25;
      const Real brf = (Real)std::atof(bre);
      const Real target = (brf > (Real)1 ? brf : ((Real)1 + m_loc)) * r_arm;    // bridge any node with rho < target
      const Real bpan = std::getenv("QJ_ARM_CL_BRIDGE_PAN") ? std::max<Real>((Real)0.25, (Real)std::atof(std::getenv("QJ_ARM_CL_BRIDGE_PAN"))) : (Real)2;
      const Real bkr  = std::getenv("QJ_ARM_CL_BRIDGE_KR")  ? (Real)std::atof(std::getenv("QJ_ARM_CL_BRIDGE_KR"))  : corner_kr;
      const Real search_frac = std::getenv("QJ_ARM_CL_BRIDGE_SEARCH") ? (Real)std::atof(std::getenv("QJ_ARM_CL_BRIDGE_SEARCH")) : (Real)0.40;
      auto arclen_of = [&](const std::vector<Vec3<Real>>& P) {                  // cumulative chord arclength
        std::vector<Real> s(P.size(), (Real)0);
        for (size_t k = 1; k < P.size(); k++) s[k] = s[k-1] + nrm(sub(P[k], P[k-1]));
        return s;
      };
      // Bridge the tightest sub-target node found in the search window at ONE end. `front` picks which end.
      auto bridge_end = [&](bool front) -> bool {
        const size_t npt = cl_s.size();
        std::vector<Real> sarr = arclen_of(cl_s);
        const Real Ltot = sarr.back();
        if ((double)Ltot < 1e-12) return false;
        const Real Whalf = bpan * (Ltot / (Real)std::max<Integer>(1, n)) * (Real)0.5;   // pad each side (panels/2)
        const Real searchL = search_frac * Ltot;
        // tightest interior node within searchL of the chosen end
        size_t kstar = 0; Real rmin = (Real)1e30;
        for (size_t k = 1; k + 1 < npt; k++) {
          const Real send = front ? sarr[k] : (Ltot - sarr[k]);
          if ((double)send > (double)searchL) continue;
          const Real rho = cl_rho_spline(cl_s, k);
          if ((double)rho < (double)rmin) { rmin = rho; kstar = k; }
        }
        if (kstar == 0 || (double)rmin >= (double)target) return false;           // no tight approach bend this end
        // grow the contiguous sub-target cluster around kstar
        size_t klo = kstar, khi = kstar;
        while (klo > 1 && (double)cl_rho_spline(cl_s, klo-1) < (double)target) klo--;
        while (khi + 2 < npt && (double)cl_rho_spline(cl_s, khi+1) < (double)target) khi++;
        // pad outward to the two kept anchor nodes (>=1 and <=npt-2 so a kept neighbour exists for the tangent)
        size_t ka = klo, kb = khi;
        while (ka > 1 && (double)(sarr[klo] - sarr[ka-1]) < (double)Whalf) ka--;
        while (kb + 2 < npt && (double)(sarr[kb+1] - sarr[khi]) < (double)Whalf) kb++;
        if (kb < ka + 2) return false;                                            // no interior node to replace
        // anchor tangents from KEPT vmtk just OUTSIDE the window (one-sided; unaffected by the rebuild)
        const Vec3<Real> tA = unit(sub(cl_s[ka], cl_s[ka-1]));
        const Vec3<Real> tB = unit(sub(cl_s[kb+1], cl_s[kb]));
        const Real h = bkr * nrm(sub(cl_s[kb], cl_s[ka]));
        const Real sden = sarr[kb] - sarr[ka];
        if ((double)sden < 1e-30) return false;
        for (size_t k = ka + 1; k < kb; k++) {
          const Real u = (sarr[k] - sarr[ka]) / sden;                             // chord-arclength fraction in window
          cl_s[k] = HybridAssembly<Real>::hermite5(cl_s[ka], tA, h, cl_s[kb], tB, h, u);
        }
        if (std::getenv("QJ_CORNER")) {
          Real rnew = (Real)1e30;
          for (size_t k = ka + 1; k < kb; k++) rnew = std::min(rnew, cl_rho_spline(cl_s, k));
          std::cout << "  [cl-bridge] end=" << (front ? "front" : "back ")
                    << " kstar=" << (long)kstar << " rho/r=" << (double)(rmin/r_arm)
                    << " -> bridged nodes [" << (long)(ka+1) << ".." << (long)(kb-1) << "]"
                    << " span=" << (double)((sarr[kb]-sarr[ka])/(Ltot/(Real)std::max<Integer>(1,n))) << " panels"
                    << "  new min rho/r=" << (double)(rnew/r_arm) << "\n";
        }
        return true;
      };
      bridge_end(true);
      bridge_end(false);   // recomputes arclength internally, so the far end sees the de-kinked front
    }
  }
  if (blend) {
    // ---- Curvature-noise LOW-PASS (QJ_ARM_CL_LOWPASS, default off => cl_s unchanged). The rho-FLOOR limiter
    // (QJ_ARM_CL_SMOOTH) only pulls points where rho<target, so it cannot touch the high-frequency curvature
    // WIGGLE the raw vmtk knots carry: after the C2 spline the curvature is continuous but dkappa/ds still
    // oscillates (rho/r swinging 3<->200 along a cap), and CSBQ's per-panel TOROIDAL special-quad rule -- built
    // for near-CONSTANT panel curvature -- cannot resolve that, giving a tol-INVARIANT DL residual concentrated
    // on the most-curved caps (verified: the smooth analytic spiral at the SAME tortuosity hits the ~3e-8 floor;
    // raising the rho-floor target instead STRAIGHTENS/kinks the tortuous caps). A Taubin lambda|mu filter is a
    // genuine low-pass: a lambda>0 shrink step then a mu<-lambda inflation step removes only the high-frequency
    // band, so it denoises curvature while PRESERVING the overall bend (tortuosity) -- the "keep arms curved"
    // directive. Endpoints (seam center + cap tip) fixed; radius untouched => closure-neutral. Scoped to caps
    // (is_cap) unless =all. Runs BEFORE the C2 fit so the spline interpolates the denoised knots.
    // QJ_ARM_CL_LOWPASS: fixed to production "all" (was env; baked 2026-10-07 as the seamsettle_tr6 baseline)
    {
      const char* lpe = "all";
      const std::string lps(lpe);
      const bool allarms = (lps == "all");
      const bool on = allarms ? true : is_cap;
      Integer passes = allarms ? (Integer)12 : (Integer)std::atoi(lpe);
      passes = (Integer)64;   // QJ_ARM_CL_LOWPASS_PASSES: fixed to production 64 (was env; baked 2026-10-07 as the seamsettle_tr6 baseline)
      if (on && passes > 0 && cl_s.size() >= 5) {
        const Real lam = std::getenv("QJ_ARM_CL_LOWPASS_LAMBDA") ? (Real)std::atof(std::getenv("QJ_ARM_CL_LOWPASS_LAMBDA")) : (Real)0.5;
        const Real mu  = std::getenv("QJ_ARM_CL_LOWPASS_MU")     ? (Real)std::atof(std::getenv("QJ_ARM_CL_LOWPASS_MU"))     : (Real)-0.53;
        const size_t n = cl_s.size();
        auto lap = [&](const std::vector<Vec3<Real>>& X, size_t k) {   // uniform-weight discrete Laplacian
          return sub(scal((Real)0.5, add(X[k-1], X[k+1])), X[k]); };
        for (Integer it = 0; it < passes; it++) {
          std::vector<Vec3<Real>> up = cl_s;                           // lambda (shrink) step
          for (size_t k = 1; k + 1 < n; k++) up[k] = add(cl_s[k], scal(lam, lap(cl_s, k)));
          cl_s.swap(up);
          up = cl_s;                                                   // mu (anti-shrink) step
          for (size_t k = 1; k + 1 < n; k++) up[k] = add(cl_s[k], scal(mu, lap(cl_s, k)));
          cl_s.swap(up);
        }
        if (std::getenv("QJ_CORNER"))
          std::cout << "  [cl-lowpass] Taubin lambda=" << (double)lam << " mu=" << (double)mu
                    << " x" << (long)passes << " passes on " << (long)n << " knots\n";
      }
    }
    // C2 centerline (QJ_ARM_C2_SPLINE, default off => byte-identical C1 centripetal CR). For caps the flag is
    // scoped to CAP arms only unless QJ_ARM_C2_SPLINE=all (interior arms are near-straight; their C1 knots are
    // absorbed). Removes the per-panel curvature-jump straddle on the curved cap body (the M8 hard rule).
    // QJ_ARM_C2_SPLINE: fixed to production "all" (was env; baked 2026-10-07 as the seamsettle_tr6 baseline)
    bcl.c2 = true;
    bcl.init(cl_s);
    const Real S = bcl.total;
    if ((double)S > 1e-12) {
      auto eval_s = [&](Real s) { return bcl.eval(s/S); };
      auto nearest_s = [&](const Vec3<Real>& Q, Real lo, Real hi) -> Real {   // arclength of v closest to Q
        const Integer NS = 800; Real best = lo, bd = (Real)1e30;
        for (Integer k = 0; k <= NS; k++) { const Real s = lo + (hi-lo)*(Real)k/NS;
          const Real d = nrm(sub(eval_s(s), Q)); if ((double)d < (double)bd) { bd = d; best = s; } }
        return best;
      };
      // Coaxial lead: keep it SHORT (few arm radii, capped to a fraction of the chord). QA/QB are the lead
      // ends on the junction axes; entry/exit pin the raw vmtk run to the nearest arclength just inside.
      {
        bLl = std::min<Real>(blead_len, (Real)0.20*L);
        const Real Ll = bLl;
        bQA = add(A0,   scal(Ll, uA));
        bQB = add(endC, scal(Ll, uB));
        bse = nearest_s(bQA, (Real)0,      (Real)0.45*S);
        bsx = nearest_s(bQB, (Real)0.55*S, S);
        if ((double)bsx <= (double)bse) { bse = (Real)0.05*S; bsx = (Real)0.95*S; }
        // GENTLE-CORNER guard (QJ_ARM_BLEND_TURN, degrees of tangent turn allowed per corner panel; default
        // 12). After the straight lead the corner rotates the junction seam axis u onto the vmtk tangent v at
        // the projected mouth. If u is far off v the base corner (short chord Q0->Pen, few panels) is a SHARP
        // turn. Two coupled remedies, both keyed to the turn angle so the arm is only a SMALL deviation off the
        // centerline per step yet still fully realigns to v (watertight interior):
        //   (a) LENGTHEN the transition -- push the vmtk entry/exit points DOWN the vessel by an arclength
        //       proportional to the turn angle, so the Hermite chord is long and the bend gentle;
        //   (b) add corner panels so the per-panel tangent change stays <= the cap (run panels preserved by
        //       inflating n, exactly like the non-blend turn-adaptive branch: cp+=extra, n+=2*extra).
        auto vtan = [&](Real s, Real toward) -> Vec3<Real> {              // unit vmtk tangent at s, pointing toward `toward`
          const Real d = std::max<Real>((Real)1e-3*S, (Real)1e-6);
          const Vec3<Real> tg = unit(sub(eval_s(std::min<Real>(S,s+d)), eval_s(std::max<Real>((Real)0,s-d))));
          return (double)toward >= (double)s ? tg : scal((Real)-1, tg);
        };
        auto angd = [&](const Vec3<Real>& p, const Vec3<Real>& q){ double c=(double)dot(p,q);
          c=std::max(-1.0,std::min(1.0,c)); return std::acos(c)*180.0/M_PI; };
        const double thA = angd(uA, vtan(bse, S));                        // u_A vs vmtk tangent leaving mouth A
        const double thB = angd(uB, vtan(bsx, (Real)0));                  // u_B vs vmtk tangent leaving mouth B
        // QJ_ARM_BLEND_TURN: fixed to production 6 (was env; baked 2026-10-07 as the seamsettle_tr6 baseline)
        const double turn_cap = 6.0; // target tangent turn per panel (deg)
        // (a) LENGTHEN: push the vmtk entry/exit down the vessel by an arclength proportional to the turn
        // angle, so the corner Hermite chord is long and the bend gentle (a genuine u->v swing turns slowly).
        const Real hpan = S/(Real)std::max<Integer>(1, n);                // vmtk arclength per panel
        const Real LtA  = std::min<Real>((Real)(thA/turn_cap)*hpan, (Real)0.35*S);
        const Real LtB  = std::min<Real>((Real)(thB/turn_cap)*hpan, (Real)0.35*S);
        bse = std::min<Real>(bse + LtA, (Real)0.45*S);                    // push vmtk entry down the vessel
        bsx = std::max<Real>(bsx - LtB, (Real)0.55*S);                    // push vmtk exit  up   the vessel
        // (a') CURVATURE pin push (QJ_ARM_PIN_CURV, default off). The turn-lengthen keys on the tangent ANGLE
        // u vs v, which misses a sharp APPROACH BEND right at the mouth: the vmtk can enter the junction with
        // an aligned tangent yet high curvature (rho<r) in the last ~2% of arclength -> the raw run just inside
        // the pin self-folds, and neither the mouth-windowed radius dip nor the endpoint-pinned centerline
        // smoothing can touch it. Remedy: walk each run pin FURTHER into the vessel until the CR-spline
        // curvature radius there clears target*r, so the smooth corner Hermite (Pex..Q1) spans the bend and
        // the raw run starts/ends only where the vessel is gentle. Capped at 0.45S/0.55S (the run never
        // vanishes; the reset below still guards bsx<=bse).
        if (const char* pce = std::getenv("QJ_ARM_PIN_CURV")) {
          const Real r_arm = std::max(rA, rB);
          const Real pin_tgt = ((Real)std::atof(pce) > (Real)1 ? (Real)std::atof(pce) : (Real)1.25) * r_arm;
          auto rho_s = [&](Real s) -> Real {                              // CR-spline curvature radius at arclength s
            const Real d = std::max<Real>((Real)2e-3*S, (Real)1e-6);
            const Vec3<Real> xm = eval_s(std::max<Real>((Real)0, s-d)), x0 = eval_s(s), xp = eval_s(std::min<Real>(S, s+d));
            const Vec3<Real> d1 = scal((Real)1/(2*d), sub(xp, xm));
            const Vec3<Real> d2 = scal((Real)1/(d*d), sub(add(xp, xm), scal((Real)2, x0)));
            const Real sp = nrm(d1), cn2 = nrm(cross(d1, d2));
            return ((double)sp > 1e-30 && (double)cn2 > 1e-30) ? (Real)(sp*sp*sp/cn2) : (Real)1e30;
          };
          const Real step = (Real)0.01*S; Integer guard = 0;
          while ((double)rho_s(bse) < (double)pin_tgt && (double)bse < (double)((Real)0.45*S) && guard++ < 60) bse += step;
          guard = 0;
          while ((double)rho_s(bsx) < (double)pin_tgt && (double)bsx > (double)((Real)0.55*S) && guard++ < 60) bsx -= step;
        }
        if ((double)bsx <= (double)bse) { bse = (Real)0.05*S; bsx = (Real)0.95*S; }
        if (std::getenv("QJ_CORNER"))
          std::cout << "  [blend-turn] thA=" << thA << " thB=" << thB << " cap=" << turn_cap
                    << "  Ltrans " << (double)LtA << "/" << (double)LtB << " (panel-count guard follows)\n";
      }
    }
  }
  // DEGENERATE-CORNER MERGE (QJ_ARM_CORNER_MERGE=lead|vmtk). When the coaxial-lead endpoint ~= the vmtk
  // entry (the common well-aligned case), the Hermite corner OVERSHOOTS (bulges out and back) between two
  // near-coincident points => the meshed centerline reverses inside the corner panel, the RMF flips ~180
  // deg, and CSBQ meshes a self-crossing tube (DL near-quad garbage). Fix in TWO coupled steps, both gated
  // on the per-corner flags decided just after the corner guard (once the corner arclength is known):
  //   (1) base() straightens the degenerate corner (linear between its endpoints -- monotone, since the
  //       vmtk entry is pushed downstream of the lead end, so the straight join never doubles back); the
  //       flags are captured by REFERENCE so the RMF built below sees the straightened curve (no flip);
  //   (2) the panel knot vector drops the (now short, straight) corner panel by merging it into the LEAD
  //       (=lead) or the first vmtk-RUN (=vmtk) panel, so its aspect is healthy (CSBQ in-range).
  // QJ_ARM_CORNER_MERGE: fixed to production "lead" (was env; baked 2026-10-07 as the seamsettle_tr6 baseline)
  const char* cmerge_env = "lead";
  const bool merge_lead = cmerge_env && std::string(cmerge_env) == "lead";
  const bool merge_vmtk = cmerge_env && std::string(cmerge_env) == "vmtk";
  const bool do_merge = blend && (merge_lead || merge_vmtk);
  const Real merge_frac = std::getenv("QJ_ARM_CORNER_MERGE_FRAC") ? (Real)std::atof(std::getenv("QJ_ARM_CORNER_MERGE_FRAC")) : (Real)0.5;
  bool cmerge0 = false, cmerge1 = false;                                  // set after the corner guard (below)
  // QJ_CAP_SETTLE (default off => byte-identical). Curved-arm-preserving alternative to QJ_CAP_STRAIGHT for
  // the far-corner MERGE case on CAP arms: QJ_CAP_STRAIGHT forces the WHOLE arm onto the single-Hermite
  // rescue (interior discarded); QJ_CAP_SETTLE only replaces the already-straight merged terminal
  // [e2L,1] (corner1+lead1 collapsed by cmerge1 into one chord Pex->C1 tilted off the cap axis, which is
  // the root cause of the cap free-edge hotspot: the meshed terminal ring lands tilted under the dome) with
  // ONE smooth quintic Hermite that still starts at Pex along the vmtk exit tangent but ARRIVES at the cap
  // tip C1 coaxial with the cap axis (tangent -U1), so the terminal ring sits flush under the dome equator
  // (QJ_CAP_TERM_REG then has nothing to correct). The raw vmtk run [e1R,e2L] and corner0 are UNTOUCHED, so
  // the arm stays genuinely curved. Only takes effect when is_cap && cmerge1; captured here (before the
  // coax lambda below) so it can be read by reference, same pattern as cmerge0/cmerge1.
  const bool cap_settle = is_cap && (std::getenv("QJ_CAP_SETTLE") != nullptr);
  // QJ_SEAM_SETTLE (default off => byte-identical). Generalizes QJ_CAP_SETTLE to BOTH merged terminals
  // (cmerge0 home-end AND cmerge1 far-end) and to EVERY arm (cap + interior junction seams), not just caps.
  // Diagnosis: the junction-seam ~0.5 cross-list free-edge defect at interior arms is the SAME mechanism as
  // the cap hotspot -- a straight merged terminal chord lands tilted off the seam axis -- but on the HOME
  // (cmerge0) end too. Replacing the straight chord with a quintic Hermite that starts (resp. ends) coaxial
  // with the owner's seam axis U0 (resp. U1) re-seats the terminal ring flush under the seam, exactly as
  // QJ_CAP_SETTLE already does for the far/cap end; it is the curved-arm-preserving fix (the raw vmtk run and
  // the opposite corner are untouched, so interior arms stay genuinely curved) and is closure-neutral (the
  // terminal ring stays exactly on the seam circle C/axis/R0). QJ_CAP_SETTLE remains a subset (cap-only, far-
  // end-only) of this flag and keeps working unchanged; when QJ_SEAM_SETTLE is unset, behavior is byte-identical.
  // QJ_SEAM_SETTLE: fixed to production ON (was env; baked 2026-10-07 as the seamsettle_tr6 baseline)
  const bool seam_settle = true;
  // ---- CURVATURE-BOUNDED SINGLE-HERMITE RESCUE (QJ_ARM_SMOOTH_CORNER, default off). Decided far below
  // (after the corner guard + relax + merge, once lp/cp/n/bse/bsx are final) but the flag+handle are
  // declared here so `base()` can override to it by reference. When `rescue` is set, the arm's interior is
  // ONE gentle quintic Hermite between the two coaxial-lead ends Q0/Q1 (tangents = the junction seam axes
  // uA/-uB), replacing the corner->vmtk-run->corner middle that folds on short arms. The mouth rings and
  // the 1-panel coaxial leads are UNCHANGED (tangent(0)=uA, tangent(1)=-uB) => the seam stays
  // axis-conforming / watertight (closure-neutral); the raw vmtk interior shape is discarded (fit relaxed,
  // per the user directive to value a closed well-formed surface over representational accuracy). This is
  // the root-cause fix for the far-/near-mouth corner folds (rho/r<1) and self-approach overlaps (dist/r<0)
  // that the census localizes to loc~0.85 on short arms, which QJ_ARM_CL_SMOOTH cannot touch (it relaxes
  // only the vmtk run, not the synthesized corner). See [[network-hotspot-fix-todo]].
  bool rescue = false; Real rescue_h = 0;
  auto base = [&](Real t) -> Vec3<Real> {
    if (rescue) {
      const Real e1L = (Real)lp/n, e2R = (Real)(n-lp)/n;
      const Vec3<Real> Q0 = add(A0, scal(bLl, uA)), Q1 = add(endC, scal(bLl, uB));
      if (t <= e1L) return add(A0, scal(t/e1L, sub(Q0, A0)));                       // lead A (coaxial, uA)
      if (t <  e2R) { const Real s = (t-e1L)/(e2R-e1L);
        const Vec3<Real> tB = unit(sub(endC, Q1));                                  // exit toward C1 (== -uB)
        return HybridAssembly<Real>::hermite5(Q0, uA, rescue_h, Q1, tB, rescue_h, s); }
      return add(Q1, scal((t-e2R)/((Real)1-e2R), sub(endC, Q1)));                   // lead B (coaxial, -uB)
    }
    if (blend && (double)bcl.total > 1e-12) {
      const Real S = bcl.total;
      auto btan = [&](Real s) -> Vec3<Real> {                          // unit vmtk tangent at arclength s
        const Real d = std::max<Real>((Real)1e-4*S, (Real)1e-6);
        const Real lo = std::max<Real>((Real)0, s-d), hi = std::min<Real>(S, s+d);
        return unit(sub(bcl.eval(hi/S), bcl.eval(lo/S)));
      };
      // Coaxial lead + Hermite corner + raw vmtk run (axis-conforming, watertight seam), as a curve LED
      // from mouth (C0,U0,s0): lead C0->Q0 along U0, corner U0->vmtk-tangent, raw vmtk run s0->s1, corner
      // vmtk-tangent->-U1, lead Q1->C1. The run tangent is signed by traversal direction (s1</>s0) so the
      // corner joins stay C1 whether arclength increases or decreases.
      {
        const bool corner_c2 = (std::getenv("QJ_ARM_CORNER_C2") != nullptr);
        // Run 2nd-derivative wrt ARCLENGTH at bcl arclength s (== curvature vector; traversal-direction
        // independent). Scaled by the corner-end handle^2 below to convert to the corner's own parameter,
        // so the C2 quintic corner meets the vmtk RUN with MATCHING curvature (no M8 curvature-jump straddle).
        auto kvec = [&](Real s) -> Vec3<Real> {
          const Real S2 = bcl.total; const Real d = std::max<Real>((Real)2e-3*S2, (Real)1e-6);
          const Vec3<Real> xm = bcl.eval(std::max<Real>((Real)0,s-d)/S2), x0 = bcl.eval(s/S2), xp = bcl.eval(std::min<Real>(S2,s+d)/S2);
          return scal((Real)1/(d*d), sub(add(xm, xp), scal((Real)2, x0)));
        };
        auto coax = [&](Real tt, const Vec3<Real>& C0, const Vec3<Real>& U0, Real s0,
                                 const Vec3<Real>& C1, const Vec3<Real>& U1, Real s1) -> Vec3<Real> {
          const Real e1L=(Real)lp/n, e1R=(Real)(lp+cp)/n, e2L=(Real)(n-lp-cp)/n, e2R=(Real)(n-lp)/n;
          const Vec3<Real> Q0 = add(C0, scal(bLl, U0)), Q1 = add(C1, scal(bLl, U1));
          const Vec3<Real> Pen = bcl.eval(s0/S), Pex = bcl.eval(s1/S);
          const Real dir = (double)s1 >= (double)s0 ? (Real)1 : (Real)-1;   // vmtk tangent aligned with traversal
          // DEGENERATE corner0 (cmerge0): the coaxial lead OVERSHOOTS the vmtk entry (Q0 lands past Pen along
          // U0), so both the straight and the Hermite u->v corner double back -> the meshed centerline reverses
          // (and, once clamped to a point, leaves COINCIDENT Chebyshev nodes). Route the whole lead+corner
          // interval [0,e1R] STRAIGHT from the mouth C0 to the vmtk entry Pen: monotone, nodes well spread (no
          // coincidence), and base(e1R)=Pen exactly matches the run start (no slit). tang(0)=(Pen-C0)/|.| is
          // within a few deg of U0 for every merged corner (they are flagged BECAUSE Pen ~= C0+bLl*U0), so the
          // mouth-ring conformity / watertightness is preserved (measured mouth tilt <=5 deg, ~0 for almost all).
          if (cmerge0 && tt < e1R) {
            if (seam_settle) {                                  // SETTLE: smooth Hermite, starts coaxial with U0
              const Real s = tt/e1R;
              const Vec3<Real> te0 = scal(dir, btan(s0));                    // end tangent: vmtk entry (as corner0 uses)
              Real h0 = corner_kr*nrm(sub(Pen, C0)), h1 = h0;                // default: equal Ferguson-chord handles
              if (ferguson) { const Real Lc = nrm(sub(Pen, C0)); const Vec3<Real> ch = unit(sub(Pen, C0));
                h0 = corner_kr*Lc*std::min(ferg_cap, std::max(ferg_floor, dot(U0, ch)));
                h1 = corner_kr*Lc*std::min(ferg_cap, std::max(ferg_floor, dot(te0, ch))); }
              return HybridAssembly<Real>::hermite5(C0, U0, h0, Pen, te0, h1, s);
            }
            return add(C0, scal(tt/e1R, sub(Pen, C0)));                    // merged: C0 -> Pen (straight)
          }
          if (tt <= e1L) return add(C0, scal(tt/e1L, sub(Q0, C0)));                              // lead 0, coaxial U0
          if (tt <  e1R) { const Real s=(tt-e1L)/(e1R-e1L);
            const Vec3<Real> te0 = scal(dir, btan(s0));                    // vmtk tangent at the corner-0 exit
            Real h0=corner_kr*nrm(sub(Pen, Q0)), h1=h0;                    // default: equal Ferguson-chord handles
            if (ferguson) { const Real Lc=nrm(sub(Pen,Q0)); const Vec3<Real> ch=unit(sub(Pen,Q0));
              h0 = corner_kr*Lc*std::min(ferg_cap, std::max(ferg_floor, dot(U0, ch)));
              h1 = corner_kr*Lc*std::min(ferg_cap, std::max(ferg_floor, dot(te0, ch))); }
            if (corner_c2)                                                          // C2: match run curvature at Pen, 0 at lead Q0
              return HybridAssembly<Real>::hermite5_c2(Q0, scal(h0,U0), Vec3<Real>{0,0,0},
                                                       Pen, scal(h1,te0), scal(h1*h1, kvec(s0)), s);
            return HybridAssembly<Real>::hermite5(Q0, U0, h0, Pen, te0, h1, s); }  // corner 0: U0 -> vmtk
          if (tt <= e2L) { const Real q=(tt-e1R)/(e2L-e1R); return bcl.eval((s0+q*(s1-s0))/S); } // raw vmtk run s0->s1
          if (cmerge1 && tt >= e2L) {
            if (cap_settle || seam_settle) {                       // SETTLE: smooth Hermite, arrives coaxial with U1
              const Real s = (tt-e2L)/((Real)1-e2L);
              const Vec3<Real> ts1 = scal(dir, btan(s1));                      // start tangent: vmtk exit (as corner1 uses)
              const Vec3<Real> te1 = scal((Real)-1, U1);                       // end tangent: cap axis, pointing out of the dome
              Real h0 = corner_kr*nrm(sub(C1, Pex)), h1 = h0;                  // default: equal Ferguson-chord handles
              if (ferguson) { const Real Lc = nrm(sub(C1, Pex)); const Vec3<Real> ch = unit(sub(C1, Pex));
                h0 = corner_kr*Lc*std::min(ferg_cap, std::max(ferg_floor, dot(ts1, ch)));
                h1 = corner_kr*Lc*std::min(ferg_cap, std::max(ferg_floor, dot(te1, ch))); }
              return HybridAssembly<Real>::hermite5(Pex, ts1, h0, C1, te1, h1, s);
            }
            return add(Pex, scal((tt-e2L)/((Real)1-e2L), sub(C1, Pex)));  // merged: Pex -> C1 (straight)
          }
          if (tt <  e2R) { const Real s=(tt-e2L)/(e2R-e2L);
            const Vec3<Real> ts1 = scal(dir, btan(s1)), te1 = scal((Real)-1, U1);  // vmtk tangent in / -axis out
            Real h0=corner_kr*nrm(sub(Q1, Pex)), h1=h0;
            if (ferguson) { const Real Lc=nrm(sub(Q1,Pex)); const Vec3<Real> ch=unit(sub(Q1,Pex));
              h0 = corner_kr*Lc*std::min(ferg_cap, std::max(ferg_floor, dot(ts1, ch)));
              h1 = corner_kr*Lc*std::min(ferg_cap, std::max(ferg_floor, dot(te1, ch))); }
            if (corner_c2)                                                          // C2: match run curvature at Pex, 0 at lead Q1
              return HybridAssembly<Real>::hermite5_c2(Pex, scal(h0,ts1), scal(h0*h0, kvec(s1)),
                                                       Q1, scal(h1,te1), Vec3<Real>{0,0,0}, s);
            return HybridAssembly<Real>::hermite5(Pex, ts1, h0, Q1, te1, h1, s); } // corner 1
          return add(Q1, scal((tt-e2R)/((Real)1-e2R), sub(C1, Q1)));                             // lead 1, coaxial -U1
        };
        return coax(t, A0, uA, bse, endC, uB, bsx);                   // one-sided coaxial-lead blend
      }
    }
    const Real e1L=(Real)lp/n, e1R=(Real)(lp+cp)/n, e2L=(Real)(n-lp-cp)/n, e2R=(Real)(n-lp)/n;
    // Non-blend fallback (QJ_ARM_BLEND unset): straight lead-corner-straight racetrack, smootherstep corners.
    const Real t1 = (Real)(lp + cp*(Real)0.5)/n, t2 = (Real)1 - t1;
    auto lerp = [&](const Vec3<Real>& p, const Vec3<Real>& q, Real w){ return add(scal((Real)1-w, p), scal(w, q)); };
    auto lineA = [&](Real x){ return lerp(A0, QA, x/t1); };
    auto lineR = [&](Real x){ return lerp(QA, QB, (x-t1)/(t2-t1)); };
    auto lineB = [&](Real x){ return lerp(QB, endC, (x-t2)/((Real)1-t2)); };
    if (t <= e1L) return lineA(t);
    if (t <  e1R) return lerp(lineA(t), lineR(t), HybridAssembly<Real>::pou_ramp((t-e1L)/(e1R-e1L)));
    if (t <= e2L) return lineR(t);
    if (t <  e2R) return lerp(lineR(t), lineB(t), HybridAssembly<Real>::pou_ramp((t-e2L)/(e2R-e2L)));
    return lineB(t);
  };

  // ---- Corner-curvature / self-overlap guard (mirrors HybridAssembly::min_corner_radius, ybifurc_assembly
  // .hpp). A tube of radius R_arm bent to radius of curvature rho self-overlaps on its inner wall once
  // rho <= R_arm; we aim for rho_min > R_arm*(1+margin) at every corner. REMEDY ORDER (per request):
  //   1. STRETCH the corner -- trade straight RUN panels into the corner window (cp += 1 at FIXED n) so the
  //      bend gets gentler / rho_min rises, without inflating DOF, while a >=4-panel straight run remains.
  //      (Needs spare run panels: raise n_axial for arms that report STILL TIGHT.)
  //   2. Reducing the arm radius is DEFAULT-OFF: rA is the arm MOUTH radius and MUST equal the owner
  //      junction's seam-hole radius (and rB the neighbour's) for the coupled surface to stay watertight.
  //      Silently shrinking it opens a mouth<->hole gap (the seam-mismatch leak). It is therefore opt-in
  //      via QJ_CORNER_SHRINK=1 and only correct when the caller ALSO rescales the matching junction hole
  //      (the "few junctions need their arm holes rescaled" path). Otherwise we keep the seam-conforming
  //      radius and just WARN -- exactly the pre-guard behaviour, which was watertight despite tight bends.
  const Real margin = std::getenv("QJ_CORNER_MARGIN") ? (Real)std::atof(std::getenv("QJ_CORNER_MARGIN")) : (Real)0.25;
  const bool allow_shrink = (std::getenv("QJ_CORNER_SHRINK") != nullptr);
  auto rho_min_of = [&]() -> Real {
    const Integer M = std::max<Integer>((Integer)2000, (Integer)40*n);
    const Real h = (Real)1/M; Real kmax = 0;
    for (Integer i = 1; i < M; i++) {
      const Vec3<Real> xm = base((i-1)*h), x0 = base(i*h), xp = base((i+1)*h);
      const Vec3<Real> d1 = scal((Real)1/(2*h), sub(xp, xm));
      const Vec3<Real> d2 = scal((Real)1/(h*h), sub(add(xp, xm), scal((Real)2, x0)));
      const Real sp = nrm(d1), cn = nrm(cross(d1, d2));
      if ((double)sp > 1e-30) { const Real kappa = cn/(sp*sp*sp); if ((double)kappa > (double)kmax) kmax = kappa; }
    }
    return (double)kmax > 1e-30 ? (Real)1/kmax : (Real)1e30;
  };
  {
    const Integer cp0 = cp; const Real rA0 = rA;
    Real rho = rho_min_of();
    if (blend) {
      // BLEND coaxial-lead corner: guard the DISCRETE per-panel tangent turn against QJ_ARM_BLEND_TURN. The
      // meshed centerline is n panels; measure the largest turn between consecutive panel-chord tangents and,
      // while it exceeds the cap and a >=6-panel straight run remains, add a corner panel each side (cp+=1,
      // n+=2 so the run is preserved). More panels resolve the same lead+corner bridge into gentler steps --
      // this is what removes the "sharp turn" on the meshed arm (the continuous rho_min, a radius-vs-curvature
      // property, is handled separately by the mid-arm radius dip; here we only bound the tangent change).
      // QJ_ARM_BLEND_TURN: fixed to production 6 (was env; baked 2026-10-07 as the seamsettle_tr6 baseline)
      const double turn_cap = 6.0;
      auto max_panel_turn = [&]() -> double {                            // max deg between consecutive panel tangents
        // Precompute panel-chord vectors + lengths; IGNORE degenerate (near-zero) panels -- the coaxial lead /
        // Hermite corner can momentarily stall (two mesh nodes ~coincident), and the roundoff direction of a
        // ~0-length chord reads as a spurious ~180 deg reversal. Skip any chord below 5% of the mean length.
        std::vector<Vec3<Real>> ch((size_t)n); std::vector<Real> ln((size_t)n); Real mean = 0;
        Vec3<Real> pprev = base((Real)0);
        for (Integer i = 0; i < n; i++) { const Vec3<Real> p = base((Real)(i+1)/n);
          ch[(size_t)i] = sub(p, pprev); ln[(size_t)i] = nrm(ch[(size_t)i]); mean += ln[(size_t)i]; pprev = p; }
        mean /= (Real)std::max<Integer>(1, n); const Real thr = (Real)0.05*mean;
        double mx = 0; Vec3<Real> tprev{0,0,0}; bool have = false;
        for (Integer i = 0; i < n; i++) {
          if ((double)ln[(size_t)i] <= (double)thr) continue;            // skip degenerate stall panels
          const Vec3<Real> t = scal((Real)1/ln[(size_t)i], ch[(size_t)i]);
          if (have) { double c = std::max(-1.0, std::min(1.0, (double)dot(t, tprev)));
            mx = std::max(mx, std::acos(c)*180.0/M_PI); }
          tprev = t; have = true;
        }
        return mx;
      };
      // Bound the extra corner panels (QJ_ARM_BLEND_MAXCP, default 6 added per side): on a well-placed arm a
      // few panels + the arclength lengthening bring the turn under the cap; a j298-class arm whose seam CENTER
      // sits far off the vessel is an intrinsically sharp jut (junction PLACEMENT, not the corner) that no sane
      // panel budget can smooth -- widening there only inflates DOF, so we cap it and WARN rather than chase it.
      // QJ_ARM_BLEND_MAXCP: fixed to production 16 (was env; baked 2026-10-07 as the seamsettle_tr6 baseline)
      const Integer maxadd = (Integer)16;
      const Integer cp_b0 = cp; double mpt = max_panel_turn(); const double mpt0 = mpt;
      while ((double)mpt > turn_cap && (n - 2*(lp+cp)) >= 6 && (cp - cp_b0) < maxadd) {
        cp += 1; n += 2; const double m2 = max_panel_turn();           // widen corner, run preserved
        if (m2 >= mpt - 0.5) { cp -= 1; n -= 2; break; }               // no progress (turn is in the RUN, not the
        mpt = m2;                                                       // corner) -> revert this add and stop
      }
      if (std::getenv("QJ_CORNER"))
        std::cout << "  [blend-guard] max panel turn " << mpt0 << " -> " << mpt << " deg (cap " << turn_cap
                  << ")  cp " << (long)cp_b0 << "->" << (long)cp << " n=" << (long)n
                  << ((double)mpt > turn_cap ? "  STILL OVER (seam-center off vessel = junction placement)" : "  OK") << "\n";
    } else {
      // Non-blend fallback: widen the corner window (cp) into the straight run until the continuous rho_min
      // clears R_arm*(1+margin), or the run bottoms out at 6 panels.
      while ((double)rho <= (double)(std::max(rA,rB)*((Real)1+margin)) && (n - 2*(lp+cp)) >= 6) {
        cp += 1; rho = rho_min_of();                                     // widen corner, run -> corner
      }
    }
    if (allow_shrink && (double)rho <= (double)(std::max(rA,rB)*((Real)1+margin))) {  // opt-in only (breaks seams)
      const Real fac = std::max<Real>((Real)0.4, std::min<Real>((Real)1, (rho/((Real)1+margin))/std::max(rA,rB)));
      rA *= fac; rB *= fac;
    }
    if (std::getenv("QJ_CORNER") && (cp != cp0 || (double)rA != (double)rA0 ||
                                     (double)rho <= (double)(std::max(rA,rB)*((Real)1+margin))))
      std::cout << "  [corner] rho_min=" << (double)rho << " vs R_arm=" << (double)std::max(rA,rB)
                << (blend ? "  mode BLEND (vmtk + localized mouth correction)"
                          : ("  corner " + std::to_string((long)cp0) + "->" + std::to_string((long)cp) + " panels"))
                << ((double)rA != (double)rA0 ? ("  radius x" + std::to_string((double)(rA/rA0))) : "")
                << (((double)rho > (double)(std::max(rA,rB)*((Real)1+margin))) ? "  OK" : "  STILL TIGHT (raise n_axial)") << "\n";
  }
  // ---- CONTINUOUS CORNER RELAX (QJ_ARM_CORNER_RELAX, default off). The blend panel-turn guard above only
  // bounds the DISCRETE per-panel tangent step; the far/near corner Hermite can still be CONTINUOUSLY tight
  // (rho<r => tube self-touch) because it swings the FIXED junction axis u onto the vmtk tangent over a SHORT
  // chord, and the mid-arm radius dip that used to absorb that is off in the production geometry. MEASURED
  // 2026-09-26: on n298 every tight far bend (rho/r->0.87, self-touch at scale 0.5) sits in the corner panels,
  // NOT the vmtk run -- so a centerline-run bypass (QJ_ARM_CL_BRIDGE) cannot touch it. Remedy = the requested
  // "swap the high-curvature section for a longer smooth bypass", applied where the tightness actually is: the
  // corner. While a corner's continuous rho_min is below target*r_arm, pull THAT corner's vmtk pin further
  // inside the vessel (bse up / bsx down). Q1/Q0 (mouth + coaxial lead) are fixed, so a pin moved inward
  // lengthens the corner-Hermite chord |Q-Pex|, and its handle h=corner_kr*|Q-Pex| grows with it => the same
  // u->v turn spreads over more arclength => lower curvature. Radius is untouched and the mouth ring is
  // unchanged (still C1 + lead along u), so this is closure-neutral BY CONSTRUCTION; it only makes the arm
  // deviate a little further from the raw vmtk path near the seam (the run shrinks). Capped so a >=~4-panel
  // straight run always remains.
  //   MEASURED HARMFUL 2026-09-26 (default OFF): the "closure-neutral" claim is WRONG in practice. Relaxing the
  //   corner DID clear the self-touch far bends (far-half min rho/r 0.869->1.297, #<1.0 5->0 at tgt=1.4) but
  //   monotonically WORSENED watertightness: tgt1.4 1.19->2.59, clean-cap tgt1.25 1.19->3.18. The tight far
  //   corner is the geometric COST of a watertight axis-conforming mouth: any deviation of the mouth approach
  //   from the vmtk path (to spread the u->v turn) opens the arm-mouth/junction-hole flux match. Same failure
  //   class as the radius dip and turn-refine (all near-seam geometry perturbations trade closure). The
  //   self-touch bends are better cleared by the RADIUS route (QJ_RADIUS_SCALE=0.1, rho/r rises 5x, mouth
  //   geometry untouched => closure stays ~1e-6) -- see [[network-arm-frame-closure]] / [[network-greens-arm-selfintersect]].
  // QJ_ARM_CORNER_RELAX: fixed to production 1.6 (was env; baked 2026-10-07 as the seamsettle_tr6 baseline)
  {
    const char* cr = "1.6";
    if (blend && (double)bcl.total > 1e-12) {
      const Real S = bcl.total;
      const Real r_arm = std::max(rA, rB);
      const Real tgt = ((Real)std::atof(cr) > (Real)1 ? (Real)std::atof(cr) : (Real)1.25) * r_arm;
      auto corner_rho = [&](Real lo, Real hi) -> Real {                    // min curvature radius of base() over [lo,hi]
        const Integer M = 200; const Real h = (hi - lo) / (Real)M; Real kmax = 0;
        for (Integer i = 1; i < M; i++) {
          const Real t0 = lo + (Real)i*h;
          const Vec3<Real> xm = base(t0-h), x0 = base(t0), xp = base(t0+h);
          const Vec3<Real> d1 = scal((Real)1/(2*h), sub(xp, xm));
          const Vec3<Real> d2 = scal((Real)1/(h*h), sub(add(xp, xm), scal((Real)2, x0)));
          const Real sp = nrm(d1), cn2 = nrm(cross(d1, d2));
          if ((double)sp > 1e-30) { const Real k = cn2/(sp*sp*sp); if ((double)k > (double)kmax) kmax = k; }
        }
        return (double)kmax > 1e-30 ? (Real)1/kmax : (Real)1e30;
      };
      // HARD caps that CANNOT meet (bse<=0.40S < 0.60S<=bsx), so the two pins never cross => the run never
      // collapses to the near-raw-vmtk reset (which re-introduces bends AND spoils mouth conformity). A >=0.2S
      // central vmtk run is always retained. The cap fraction is env-tunable (QJ_ARM_CORNER_RELAX_CAP, default
      // 0.40) so a stubborn sharp-mouth SHORT arm -- whose u->v turn cannot reach rho>=tgt within a 0.40S corner
      // chord -- can spread its corner over more arclength while KEEPING the arm curved (no straightening); the
      // symmetric exit cap is 1-frac so the >= (1-2*frac)*S central run is retained (frac<0.5). gmax scales with
      // frac so the 0.01S step can actually reach the wider cap.
      // QJ_ARM_CORNER_RELAX_CAP: fixed to production 0.49 (was env; baked 2026-10-07 as the seamsettle_tr6 baseline)
      const Real cap_frac = (Real)0.49;
      // MUTUAL-BORROW cap (2026-09-30). The old caps were SYMMETRIC and INDEPENDENT (bse<=frac*S, bsx>=(1-frac)*S),
      // so a jut arm whose NEAR corner needs no room (straight/merged, rho>>tgt) still could not spread its FAR
      // corner past (1-frac)*S -- e.g. j004->6 stalled at far rho/r=0.824 with bsx pinned at 0.5S while its near
      // corner sat slack at bse=0.21S. Replace the two fixed caps with ONE mutual constraint: each pin may push
      // toward mid-arm as long as >= min_run of raw vmtk run remains between the pins (bse + min_run <= bsx). A
      // tight corner therefore BORROWS the opposite corner's slack; a >= min_run central run is always retained
      // (never the near-raw reset), the mouth rings/leads are untouched (closure-neutral), and the arm stays
      // curved (the u->v turn just spreads over more arclength). min_run reuses the RELAX_CAP knob: (1-2*frac)*S,
      // so CAP=0.49 => min_run 0.02S (max borrowing), CAP=0.40 => 0.20S (as before, but now reallocatable).
      const Real min_run = std::max<Real>((Real)0.02, (Real)1 - (Real)2*cap_frac) * S;
      const Real step = (Real)0.01*S; const Integer gmax = (Integer)110;   // 110*0.01S covers a full-S traverse
      auto e1Lw = [&](){ return (Real)lp/n; };      auto e1Rw = [&](){ return (Real)(lp+cp)/n; };
      auto e2Lw = [&](){ return (Real)(n-lp-cp)/n; }; auto e2Rw = [&](){ return (Real)(n-lp)/n; };
      // NEAR corner first: settle bse (push toward mid while it folds), so the far corner sees the final run floor.
      { Integer g=0;
        while ((double)corner_rho(e1Lw(), e1Rw()) < (double)tgt && (double)(bse + min_run) < (double)bsx && g++ < gmax) bse += step; }
      // FAR corner: push bsx toward mid, floored only by keeping min_run past the (now settled) near pin.
      { Integer g=0;
        while ((double)corner_rho(e2Lw(), e2Rw()) < (double)tgt && (double)(bsx - min_run) > (double)bse && g++ < gmax) bsx -= step; }
      // NEAR corner again: if the near corner is the tight one, let it now borrow whatever the far corner left.
      { Integer g=0;
        while ((double)corner_rho(e1Lw(), e1Rw()) < (double)tgt && (double)(bse + min_run) < (double)bsx && g++ < gmax) bse += step; }
      if (std::getenv("QJ_CORNER"))
        std::cout << "  [corner-relax] far rho/r=" << (double)(corner_rho(e2Lw(),e2Rw())/r_arm)
                  << " near rho/r=" << (double)(corner_rho(e1Lw(),e1Rw())/r_arm)
                  << "  bse/S=" << (double)(bse/S) << " bsx/S=" << (double)(bsx/S)
                  << " (tgt " << (double)(tgt/r_arm) << ")\n";
    }
  }
  // Decide the degenerate-corner flags NOW (corner arclengths are final), BEFORE the RMF is tabulated, so
  // the straightened base() feeds the frame (no flip). Measured with the flags still false => the reversing
  // corner's true (short) arclength is what triggers the merge.
  if (do_merge) {
    auto seg_len = [&](Real a, Real b) -> Real {
      const Integer M = 24; Real s = 0; Vec3<Real> pp = base(a);
      for (Integer i = 1; i <= M; i++) { const Vec3<Real> p = base(a + (b - a) * (Real)i / M); s += nrm(sub(p, pp)); pp = p; }
      return s;
    };
    const Real e1L=(Real)lp/n, e1R=(Real)(lp+cp)/n, e2L=(Real)(n-lp-cp)/n, e2R=(Real)(n-lp)/n;
    const Integer nrun = std::max<Integer>(1, n - 2*(lp+cp));
    const Real mean_run = seg_len(e1R, e2L) / (Real)nrun;                 // arclength per RUN panel
    const Real thr = merge_frac * mean_run;
    cmerge0 = ((double)seg_len(e1L, e1R) < (double)thr);                  // corner arclength below half a run panel
    cmerge1 = ((double)seg_len(e2L, e2R) < (double)thr);
    if (std::getenv("QJ_CORNER"))
      std::cout << "  [corner-merge] mode=" << cmerge_env << "  corner0 len=" << (double)seg_len(e1L,e1R)
                << (cmerge0 ? " MERGE" : " keep") << "  corner1 len=" << (double)seg_len(e2L,e2R)
                << (cmerge1 ? " MERGE" : " keep") << "  (thr=" << (double)thr << ")\n";
  }
  // ---- CURVATURE-BOUNDED SINGLE-HERMITE RESCUE decision (QJ_ARM_SMOOTH_CORNER). Now that lp/cp/n/bse/bsx
  // and the merge flags are final, measure the CONTINUOUS curvature radius of the pre-rescue base() in the
  // two corner windows. If either corner folds (rho < target*r_arm) -- the short-arm far-/near-mouth fold
  // the census pins at loc~0.85 -- switch this arm to the single-Hermite rescue and grow its handle until
  // the whole-arm rho clears the target (or the handle cap is hit; then WARN). Leaves healthy arms on the
  // blend path (rescue stays false => base() byte-identical). Default-off => byte-identical when unset.
  // QJ_CAP_STRAIGHT (default off): for CAP arms (far end = free hemisphere tip, NOT a shared seam), force the
  // single-Hermite rescue UNCONDITIONALLY -- skip the fold/self-gap trigger and the far-end revert guard. The
  // rescued arm ends with a straight coaxial lead Q1->tip along uB=-endTang, so its terminal mouth ring is
  // EXACTLY perpendicular to endTang == the dome equator plane (add_cap_hemisphere_frame uses capring.u=endTang),
  // reproducing the straight-capsule seam (8e-15 closure / 3.4e-6 DL) that a bent cap breaks. Without this, on a
  // sharply-bent cap the far-corner MERGE routes the terminal straight along (tip-Pex) -- tilted a few deg off
  // endTang -- so the last ring sits tilted under the dome (partial slit => DL free-edge ~0.5). The far end is
  // fidelity-free (user directive: closed well-formed surface over vmtk fidelity), so forcing straight is safe.
  const bool cap_straight = is_cap && std::getenv("QJ_CAP_STRAIGHT") != nullptr;
  if (blend && (std::getenv("QJ_ARM_SMOOTH_CORNER") || cap_straight) && (double)bcl.total > 1e-12) {
    const Real r_arm = std::max(rA, rB);
    const Real smf = std::getenv("QJ_ARM_SMOOTH_MARGIN") ? (Real)std::atof(std::getenv("QJ_ARM_SMOOTH_MARGIN")) : margin;
    const Real target = ((Real)1 + smf) * r_arm;
    auto win_rho = [&](Real lo, Real hi) -> Real {                        // min curvature radius of base() over [lo,hi]
      const Integer M = 200; const Real h = (hi - lo) / (Real)M; Real kmax = 0;
      for (Integer i = 1; i < M; i++) { const Real t0 = lo + (Real)i*h;
        const Vec3<Real> xm = base(t0-h), x0 = base(t0), xp = base(t0+h);
        const Vec3<Real> d1 = scal((Real)1/(2*h), sub(xp, xm));
        const Vec3<Real> d2 = scal((Real)1/(h*h), sub(add(xp, xm), scal((Real)2, x0)));
        const Real sp = nrm(d1), cn2 = nrm(cross(d1, d2));
        if ((double)sp > 1e-30) { const Real k = cn2/(sp*sp*sp); if ((double)k > (double)kmax) kmax = k; } }
      return (double)kmax > 1e-30 ? (Real)1/kmax : (Real)1e30;
    };
    const Real e1L=(Real)lp/n, e1R=(Real)(lp+cp)/n, e2L=(Real)(n-lp-cp)/n, e2R=(Real)(n-lp)/n;
    const Real rho_near = win_rho(e1L, e1R), rho_far = win_rho(e2L, e2R);
    // Coarse SELF-APPROACH scan of base(): sample the centerline and find the min surface gap
    // (|Xi-Xj|-2r)/r over station pairs separated by more than a tube DIAMETER in arclength (so straight-tube
    // adjacency is excluded -- same mask as census_arms.py). A negative gap is a real self-crossing that rho
    // is blind to (the near-/far-mouth hairpin, e.g. edge 004-8). Cheap: ~50^2 point pairs. Callable pre- and
    // post-rescue (reads the live `rescue` flag through base()) so the accept/revert guard can compare.
    auto scan_self_gap = [&]() -> Real {
      const Integer MS = 200; std::vector<Vec3<Real>> Xs((size_t)MS+1); std::vector<Real> ss((size_t)MS+1);
      Vec3<Real> pp = base((Real)0); ss[0] = 0; Real g = (Real)1e30;
      for (Integer i = 0; i <= MS; i++) { Xs[(size_t)i] = base((Real)i/MS);
        if (i>0) ss[(size_t)i] = ss[(size_t)i-1] + nrm(sub(Xs[(size_t)i], pp)); pp = Xs[(size_t)i]; }
      const Real maskL = (Real)2.2*r_arm;
      for (Integer i = 0; i <= MS; i++) for (Integer j = i+1; j <= MS; j++)
        if ((double)(ss[(size_t)j]-ss[(size_t)i]) > (double)maskL)
          g = std::min(g, (nrm(sub(Xs[(size_t)i], Xs[(size_t)j])) - (Real)2*r_arm)/r_arm);
      return g;
    };
    const Real self_gap = scan_self_gap();                                 // pre-rescue (rescue==false)
    if (cap_straight || std::min((double)rho_near, (double)rho_far) < (double)target || (double)self_gap < 0) {
      rescue = true;                                                       // base() now = single Hermite Q0->Q1
      const Real kr0 = std::getenv("QJ_ARM_SMOOTH_KR") ? (Real)std::atof(std::getenv("QJ_ARM_SMOOTH_KR")) : (Real)1.0;
      const Vec3<Real> Q0 = add(A0, scal(bLl, uA)), Q1 = add(endC, scal(bLl, uB));
      const Real chord = std::max<Real>((Real)1e-12, nrm(sub(Q1, Q0)));
      Real kr = kr0, rho_r = 0;
      for (Integer it = 0; it < 8; it++) {                                 // grow the handle until rho clears target
        rescue_h = kr * chord; rho_r = rho_min_of() / r_arm;
        if ((double)rho_r >= (double)((Real)1 + smf)) break;
        kr *= (Real)1.3; if ((double)kr > 4.0) break;
      }
      // ACCEPT/REVERT guard: the single Hermite must not WORSEN self-approach. On a near-antiparallel seam
      // (a junction-PLACEMENT pathology, e.g. edge 004-8) the direct Hermite bulges its start toward the
      // opposite mouth and the near-mouth gap gets worse; keeping the blend path is then the lesser evil.
      // Accept iff the rescued curve clears the fold AND its self_gap is no worse than baseline (small tol).
      const Real post_gap = scan_self_gap();                              // post-rescue (rescue==true)
      const bool fold_fixed = ((double)rho_r >= (double)((Real)1 + smf));
      const bool gap_ok = ((double)post_gap >= 0) || ((double)post_gap >= (double)self_gap - 1e-6);
      // For a forced cap-straight arm accept unconditionally: the terminal-straight seam is the goal regardless
      // of the interior fold/gap (tip is free; the near-mouth ring stays coaxial with the owner junction hole).
      rescue = cap_straight || (fold_fixed && gap_ok);                    // else leave the arm on the blend path
      if (std::getenv("QJ_CORNER"))
        std::cout << "  [smooth-corner]" << (cap_straight ? "[cap-straight]" : "")
                  << " pre rho/r near=" << (double)(rho_near/r_arm) << " far=" << (double)(rho_far/r_arm)
                  << " self_gap/r=" << (double)self_gap
                  << " -> " << (rescue ? "RESCUE" : "revert(placement)") << " kr=" << (double)kr
                  << " rho/r=" << (double)rho_r << " post_gap/r=" << (double)post_gap
                  << ((double)rho_r < (double)((Real)1+smf) ? "  STILL TIGHT" : "") << "\n";
    }
  }
  // ---- SELF-CONTACT loop-collapse (QJ_ARM_MONOTONE, default off) ----
  // Detect where the tube contacts ITSELF: two interior stations more than QJ_ARM_MONOTONE_ARCSEP panel-lengths
  // apart in arclength (default 1.5, so genuinely non-adjacent) whose CENTER distance |Xi-Xj| < (2+tol)*r_mid
  // (tol = QJ_ARM_MONOTONE_TOL, default 0.5) -- i.e. their surfaces are within tol*r. Such a pair brackets a
  // doubled-back excursion (a hairpin) OR a hard inner-wall fold. Replace the bracketed segment [ti,tj] with the
  // STRAIGHT chord Xi->Xj (a single short-arclength, single-directional segment): the artificial excursion is
  // excised. Restricted to the interior [e1L,e2R] so the terminal coaxial LEADS -- and the axis-conforming mouth
  // rings -- are untouched. NB chord-PROJECTION monotonicity is deliberately NOT used: it both misses parallel
  // self-contacts (projection stays monotone through the loop) and false-flags legitimate >90 deg axis-conforming
  // corners (measured 2026-09-28: it collapsed 80 innocent arms and drove closure 1.19->8.96 while leaving the
  // real self-contacts untouched). Default-off => curve==base (byte-identical).
  //   MEASURED HARMFUL 2026-09-28 (this SELF-CONTACT version, default OFF): it DOES straighten the genuine
  //   curvature folds (edge 117-130 rho/r 0.87->1.41, cap28 1.48->2.19, 271-277 2.60->3.80) but WORSENS closure
  //   1.19->5.92 (32 arms collapsed; arm |int n dA| 81.52->75.8 no longer cancels junc 81.56). Same watertightness
  //   trade as every near-seam perturbation (bridge/corner-relax/radius-dip/turn-refine): straightening an interior
  //   segment pulls the arm off the vmtk path so the mouth/frame flux match opens. It also does NOT move the
  //   1-panel-separation dist/r "self-contact" metric, which reflects tube-DIAMETER proximity of adjacent sections
  //   at a bend, not a collapsible loop. The radius route (QJ_RADIUS_SCALE) stays the only closure-neutral fold fix.
  struct MonoLoop { Real ta, tb; Vec3<Real> Pa, Pb; };
  std::vector<MonoLoop> mono_loops;
  if (std::getenv("QJ_ARM_MONOTONE") && blend && (double)bcl.total > 1e-12) {
    const Real e1L=(Real)lp/n, e2R=(Real)(n-lp)/n;                          // keep terminal leads intact
    const Real rmid = (Real)0.5*(rA+rB);
    const Real ctol = std::getenv("QJ_ARM_MONOTONE_TOL")    ? (Real)std::atof(std::getenv("QJ_ARM_MONOTONE_TOL"))    : (Real)0.5;
    const Real asep = std::getenv("QJ_ARM_MONOTONE_ARCSEP") ? (Real)std::atof(std::getenv("QJ_ARM_MONOTONE_ARCSEP")) : (Real)1.5;
    const Integer N = std::max<Integer>((Integer)400, (Integer)30*n);
    std::vector<Real> tt((size_t)N+1), ss((size_t)N+1); std::vector<Vec3<Real>> XX((size_t)N+1);
    for (Integer i=0;i<=N;i++){ tt[(size_t)i]=e1L+(e2R-e1L)*(Real)i/(Real)N; XX[(size_t)i]=base(tt[(size_t)i]); }
    ss[0]=0; for (Integer i=1;i<=N;i++) ss[(size_t)i]=ss[(size_t)i-1]+nrm(sub(XX[(size_t)i],XX[(size_t)i-1]));
    const Real gap_arc = asep*(ss[(size_t)N]/(Real)std::max<Integer>(1,(n-2*lp)));  // >asep interior-panel-lengths
    const Real contact = ((Real)2+ctol)*rmid;
    std::vector<char> mark((size_t)N+1, 0);
    for (Integer i=0;i<=N;i++) for (Integer j=i+1;j<=N;j++) {
      if ((double)(ss[(size_t)j]-ss[(size_t)i]) <= (double)gap_arc) continue;
      if ((double)nrm(sub(XX[(size_t)i],XX[(size_t)j])) < (double)contact) { for (Integer k=i;k<=j;k++) mark[(size_t)k]=1; }
    }
    Integer i=0;
    while (i<=N){
      if (mark[(size_t)i]) { Integer j=i; while (j<N && mark[(size_t)j+1]) j++;   // maximal marked run [i,j]
        mono_loops.push_back(MonoLoop{ tt[(size_t)i], tt[(size_t)j], XX[(size_t)i], XX[(size_t)j] }); i=j+1; }
      else i++;
    }
    if (std::getenv("QJ_CORNER") && !mono_loops.empty())
      std::cout << "  [monotone] collapsed " << mono_loops.size() << " self-contact loop(s)"
                << " (tol " << (double)ctol << " arcsep " << (double)asep << ")\n";
  }
  auto curve = [&](Real t) -> Vec3<Real> {
    for (const auto& lp_ : mono_loops)
      if ((double)t >= (double)lp_.ta && (double)t <= (double)lp_.tb) {
        const Real u = ((double)lp_.tb > (double)lp_.ta) ? (t-lp_.ta)/(lp_.tb-lp_.ta) : (Real)0;
        return add(lp_.Pa, scal(u, sub(lp_.Pb, lp_.Pa)));
      }
    return base(t);
  };
  // Dump the FINAL (post-guard) meshed centerline for the approximation-graph verification artifact.
  if (sampled_out) {
    const Integer Ns = std::max<Integer>((Integer)4*n, (Integer)32);
    sampled_out->clear(); sampled_out->reserve((size_t)Ns + 1);
    for (Integer k = 0; k <= Ns; k++) sampled_out->push_back(curve((Real)k/Ns));
    // For BLEND mode also emit the REFERENCE vmtk centerline v(l) at the same stations, so the exact
    // displacement |c(l)-v(l)| = |a_A*d_A + a_B*d_B| can be measured directly (not via nearest-to-polyline,
    // which conflates the centripetal-CR-vs-chord interpolation gap on curved edges).
    if (sampled_ref) {
      sampled_ref->clear();
      if (blend && (double)bcl.total > 1e-12) {
        for (Integer k = 0; k <= Ns; k++) {                            // NEAREST point on the vmtk curve to
          const Vec3<Real> P = curve((Real)k/Ns);                      // each meshed node = true deviation
          Real best = 0, bd = (Real)1e30; const Integer NS = 2000;
          for (Integer m = 0; m <= NS; m++) { const Real s = (Real)m/NS;
            const Real d = nrm(sub(bcl.eval(s), P)); if ((double)d < (double)bd) { bd = d; best = s; } }
          Real lo = std::max<Real>(0, best - (Real)1/NS), hi = std::min<Real>(1, best + (Real)1/NS);
          for (Integer ref = 0; ref < 3; ref++) {                      // parabola-free local refine (removes the
            const Integer M = 40; Real b2 = lo, d2 = (Real)1e30;       // ~S/NS sampling floor -> true ~0 on the run)
            for (Integer m = 0; m <= M; m++) { const Real s = lo + (hi-lo)*(Real)m/M;
              const Real d = nrm(sub(bcl.eval(s), P)); if ((double)d < (double)d2) { d2 = d; b2 = s; } }
            const Real w = (hi-lo)/M; lo = std::max<Real>(0, b2-w); hi = std::min<Real>(1, b2+w); best = b2;
          }
          sampled_ref->push_back(bcl.eval(best));
        }
      }
    }
  }
  // ---- Mid-arm radius DIP (opt-in, seam-safe). Where the centerline bends tighter than the
  // seam-conforming radius allows (local curvature radius rho(t) < r*(1+margin) => inner-wall self-fold),
  // locally THIN the tube so rho(t) > r*(1+margin), leaving BOTH mouth rings exact (t=0->rA, t=1->rB) via
  // an endpoint-vanishing window -- the seam-safe cousin of QJ_CORNER_SHRINK (which scales rA/rB and so
  // breaks watertightness). Bounded below by QJ_RADIUS_DIP_FLOOR * seam-taper radius so the arm can never
  // become a needle. QJ_ARM_RADIUS_DIP=1 enables; the fold-clip target reuses QJ_CORNER_MARGIN (`margin`).
  const bool radius_dip = (std::getenv("QJ_ARM_RADIUS_DIP") != nullptr);
  const Real dip_floor = std::getenv("QJ_RADIUS_DIP_FLOOR") ? (Real)std::atof(std::getenv("QJ_RADIUS_DIP_FLOOR")) : (Real)0.5;
  // Mouth POU half-window (fraction of arclength over which the dip ramps from 0 at the seam to full in the
  // interior). Smaller => the taper reaches CLOSER to the mouth (clears near-junction folds) while the ring
  // AT t=0/1 stays exactly rA/rB (watertight). Default 0.08; QJ_RADIUS_DIP_WIN overrides.
  const Real dip_win = std::getenv("QJ_RADIUS_DIP_WIN") ? std::max<Real>((Real)1e-3, (Real)std::atof(std::getenv("QJ_RADIUS_DIP_WIN"))) : (Real)0.08;
  const Integer Nrho = std::max<Integer>((Integer)400, (Integer)20*n);
  std::vector<Real> rho_tab((size_t)Nrho+1, (Real)1e30);
  if (radius_dip) {
    const Real h = (Real)1/Nrho;
    for (Integer i = 1; i < Nrho; i++) {
      const Vec3<Real> xm = base((i-1)*h), x0 = base(i*h), xp = base((i+1)*h);
      const Vec3<Real> d1 = scal((Real)1/(2*h), sub(xp, xm));
      const Vec3<Real> d2 = scal((Real)1/(h*h), sub(add(xp, xm), scal((Real)2, x0)));
      const Real sp = nrm(d1), cn2 = nrm(cross(d1, d2));
      rho_tab[(size_t)i] = ((double)sp > 1e-30 && (double)cn2 > 1e-30) ? (Real)(sp*sp*sp/cn2) : (Real)1e30;
    }
    rho_tab[0] = rho_tab[1]; rho_tab[(size_t)Nrho] = rho_tab[(size_t)Nrho-1];
  }
  auto radius_at = [&](Real t) -> Real {
    const Real rlin = rA + t*(rB - rA);                                    // seam-conforming linear taper
    if (!radius_dip) return rlin;
    Real g = t*Nrho; if ((double)g < 0) g = 0; if (g > (Real)Nrho) g = (Real)Nrho;
    Integer i = (Integer)g; if (i >= Nrho) i = Nrho-1; const Real f = g - i;
    const Real rho_t = rho_tab[(size_t)i]*((Real)1-f) + rho_tab[(size_t)i+1]*f;
    const Real r_allow = rho_t/((Real)1+margin);                           // radius keeping rho > r*(1+margin)
    Real rr = std::min(rlin, r_allow);
    const Real floor_r = dip_floor * rlin;                                 // never a needle
    if ((double)rr < (double)floor_r) rr = floor_r;
    const Real w = std::min<Real>((Real)1, std::min<Real>(t, (Real)1-t)/dip_win);  // 0 at mouths
    const Real wr = HybridAssembly<Real>::pou_ramp(w);
    return rlin*((Real)1-wr) + rr*wr;                                      // mouths exact, dip in interior
  };
  // Rotation-minimizing frame (double reflection, Wang et al.) tabulated on a fine grid.
  const Integer Ng = std::max<Integer>((Integer)200, (Integer)20*n);
  auto tang = [&](Integer i) -> Vec3<Real> {
    const Real h = (Real)1/Ng;
    if (i <= 0)  return unit(sub(curve(h), curve((Real)0)));
    if (i >= Ng) return unit(sub(curve((Real)1), curve((Real)1-h)));
    return unit(sub(curve((i+1)*h), curve((i-1)*h)));
  };
  // Report the ACTUAL meshed terminal cross-section (center = curve(1), outward axis = terminal tangent) so a
  // leaf-cap dome can be registered to the tube's real terminal ring instead of the idealized endTang -- the
  // flagella "cap conforms to the slender's tip ring" recipe. The idealized endTang can tilt a few tenths of
  // a degree off the meshed tangent (corner-merge/blend at the far mouth), tilting the dome equator plane so
  // part of the ring dips ~1e-4 INSIDE the tube surface => the cross-list near-singular cap blow-up.
  if (out_termC) *out_termC = curve((Real)1);
  if (out_termU) *out_termU = tang(Ng);
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
  // ---- CLOSURE TWIST (owner<->neighbour frame conformity). The RMF above is seeded to match the OWNER seam
  // frame at t=0 (r = seamA.e1) but drifts freely to t=1, so the far ring lands azimuthally MISMATCHED to the
  // neighbour junction's seam frame (measured mean ~91 deg, up to 180 across the n298 network). The tube
  // surface set is frame-invariant, but the PARAMETRISATION is not: e1(t) enters dX/dt (an r*e1'(t) term) and
  // the CSBQ self/near toroidal quadrature, so a frame that jumps ~90 deg between the arm's far ring and the
  // adjoining junction hole is a NONSMOOTH junction->arm transition => high DL/near-eval error at the far seam
  // (the "far end always worse than home end" asymmetry). Remedy: when the caller supplies the far target
  // frame endE1 (neighbour seam e1 for interior arms, cap-ring e1 for leaves), distribute a UNIFORM twist
  // theta(t)=t*phi about the local tangent so the frame matches endE1 at t=1 too (phi = minimal residual angle
  // from the propagated end frame to endE1, about the end tangent). Uniform twist keeps de1/dt bounded/smooth
  // (a constant added twist-rate phi/L) -- strictly better than the current infinite jump at the seam. Both
  // seams now conform, exactly like the home end. endE1==0 (unset) preserves the old free-drift behaviour.
  //
  const Vec3<Real> r0const = Rtab[0];
  auto orient_of = [&](Real t) -> Vec3<Real> {
    if (constant_orient) return r0const;
    Real g = t*Ng; if ((double)g < 0) g = 0; if (g > (Real)Ng) g = (Real)Ng;
    Integer i = (Integer)g; if (i >= Ng) i = Ng-1; const Real f = g - i;
    return unit(add(scal((1-f), Rtab[(size_t)i]), scal(f, Rtab[(size_t)i+1])));
  };
  // ---- Panel knot vector. UNIFORM by default => t = tk[p] + cn[j]*(tk[p+1]-tk[p]) = (p+cn[j])/n, i.e.
  // byte-identical to the pre-merge sampling (regression-safe: merge OFF changes nothing). When a corner is
  // flagged degenerate (cmerge0/cmerge1, decided above) its now-short straight panel(s) are absorbed into
  // the neighbour by deleting the shared knot: for corner0 the lead is on the LEFT of the run, for corner1
  // on the RIGHT, so merge=lead deletes the knot on the lead side and merge=vmtk the knot on the run side.
  // Terminal knots (0,1) are never removed (rings stay conforming to the junction holes).
  std::vector<Real> tk; tk.reserve((size_t)n + 1);
  for (Integer k = 0; k <= n; k++) tk.push_back((Real)k / (Real)n);
  if (do_merge && (cmerge0 || cmerge1)) {
    std::vector<bool> keepk((size_t)n + 1, true);
    auto rm = [&](Integer k){ if (k > 0 && k < n) keepk[(size_t)k] = false; };
    if (cmerge0) for (Integer p = lp;         p < lp + cp; p++) rm(merge_lead ? p     : p + 1);  // corner0: lead left / run right
    if (cmerge1) for (Integer p = n - lp - cp; p < n - lp;  p++) rm(merge_lead ? p + 1 : p);      // corner1: lead right / run left
    std::vector<Real> tk2; tk2.reserve(tk.size());
    for (Integer k = 0; k <= n; k++) if (keepk[(size_t)k]) tk2.push_back(tk[(size_t)k]);
    tk.swap(tk2);
    if (std::getenv("QJ_CORNER"))
      std::cout << "  [corner-merge] " << (long)(n + 1 - (Integer)tk.size()) << " knot(s) removed => "
                << (long)(tk.size() - 1) << " panels (was " << (long)n << ")\n";
  }
  // Panel-ROLE transition t-values (lead0|corner0, corner0|run, run|corner1, corner1|lead1). All are integer
  // multiples of 1/n (uniform panels), so each sits ON a knot -- UNLESS its corner was merged, in which case the
  // transition is now interior to a merged panel (no inter-panel break there, correctly excluded below).
  const double role_tb[4] = { (double)lp/n, (double)(lp+cp)/n, (double)(n-lp-cp)/n, (double)(n-lp)/n };
  // ---- Local n_axial REFINEMENT at panel-ROLE transitions (QJ_ARM_ROLE_REFINE, default off => tk unchanged,
  // bit-identical). The surface "open spaces" are inter-panel gaps where the CSBQ centerline turns too fast
  // BETWEEN panels: two order-`cheb` Chebyshev panels meet at a shared knot, and across a lead|corner or
  // corner|run role boundary the tangent turns sharply, so the per-panel tube reconstructions mismatch (a
  // gap/overlap). Subdividing the panels TOUCHING those role knots into REFINE sub-panels resolves the same
  // bend in smaller steps (per-panel turn ~1/REFINE) WITHOUT moving the curve or the radius (closure-neutral).
  // QJ_ARM_ROLE_REFINE_SPAN(=1) panels each side of each role knot qualify.
  const Integer role_refine = std::getenv("QJ_ARM_ROLE_REFINE") ? std::max<Integer>(1, (Integer)std::atoi(std::getenv("QJ_ARM_ROLE_REFINE"))) : (Integer)1;
  const Integer role_span   = std::getenv("QJ_ARM_ROLE_REFINE_SPAN") ? std::max<Integer>(0, (Integer)std::atoi(std::getenv("QJ_ARM_ROLE_REFINE_SPAN"))) : (Integer)1;
  // ---- Curvature-adaptive n_axial refinement (QJ_ARM_TURN_REFINE=<deg>, default 6 = on). The diagnostic
  // showed the inter-panel direction jumps are NOT at the role transitions (those are gentle, ~4 deg/panel)
  // but in the interior raw-vmtk RUN, which bends ~26 deg/panel (up to 72). Subdivide ANY panel whose own
  // tangent bend across it exceeds the cap into ceil(bend/cap) sub-panels (capped by QJ_ARM_TURN_REFINE_MAX,
  // default 24), so every sub-panel turns <= cap. Closure-neutral (same curve, same radius); resolves the
  // under-resolution "open spaces" (does NOT fix true rho<r self-folds -- those need smoothing/taper/scale).
  // Now DEFAULT-ON (the "tr6" baseline, 2026-10-07): refine angle 6 deg, per-panel subdivision cap 24. Both remain
  // env-overridable knobs (QJ_ARM_TURN_REFINE=<deg>, QJ_ARM_TURN_REFINE_MAX=<n>).
  const char* trc = std::getenv("QJ_ARM_TURN_REFINE") ? std::getenv("QJ_ARM_TURN_REFINE") : "6";
  if (trc && std::atof(trc) <= 0.0) trc = nullptr;                          // QJ_ARM_TURN_REFINE=0 turns it off
  const Integer turn_refine_max = std::getenv("QJ_ARM_TURN_REFINE_MAX") ? std::max<Integer>(1, (Integer)std::atoi(std::getenv("QJ_ARM_TURN_REFINE_MAX"))) : (Integer)24;
  if (trc) {
    const double cap = std::max(1.0, std::atof(trc));
    auto tang_at = [&](Real t, int side) -> Vec3<Real> {                   // side<0: forward from t; side>0: backward to t
      const Real d = std::max<Real>((Real)1e-6, (Real)1e-3);
      return side < 0 ? unit(sub(curve(std::min<Real>((Real)1, t + d)), curve(t)))
                      : unit(sub(curve(t), curve(std::max<Real>((Real)0, t - d))));
    };
    std::vector<Real> tk3; tk3.reserve(tk.size() * 2);
    for (Integer p = 0; p + 1 < (Integer)tk.size(); p++) {
      const Real ta = tk[(size_t)p], tb = tk[(size_t)p + 1];
      const Vec3<Real> t0 = tang_at(ta, -1), t1 = tang_at(tb, +1);
      double d = (double)dot(t0, t1); d = std::max(-1.0, std::min(1.0, d));
      const double bend = std::acos(d) * 180.0 / M_PI;
      Integer k = (Integer)std::ceil(bend / cap); if (k < 1) k = 1; if (k > turn_refine_max) k = turn_refine_max;
      tk3.push_back(ta);
      for (Integer s = 1; s < k; s++) tk3.push_back((Real)(ta + (tb - ta) * (Real)s / (Real)k));
    }
    tk3.push_back(tk.back());
    tk.swap(tk3);
    if (std::getenv("QJ_CORNER"))
      std::cout << "  [turn-refine] cap " << cap << " deg (max x" << (long)turn_refine_max
                << ") => " << (long)(tk.size() - 1) << " panels (was " << (long)n << ")\n";
  } else if (std::getenv("QJ_ARM_ROLE_REFINE") && role_refine > 1) {
    const double htol = (double)(role_span + (Integer)0.5) / (double)n;    // panel center within this of a role knot qualifies
    std::vector<Real> tk3; tk3.reserve(tk.size() * (size_t)role_refine);
    for (Integer p = 0; p + 1 < (Integer)tk.size(); p++) {
      const double ta = (double)tk[(size_t)p], tb = (double)tk[(size_t)p + 1], tcen = 0.5 * (ta + tb);
      bool near_role = false;
      for (int b = 0; b < 4; b++) if (std::fabs(tcen - role_tb[b]) < htol) near_role = true;
      tk3.push_back(tk[(size_t)p]);
      if (near_role) for (Integer s = 1; s < role_refine; s++)
        tk3.push_back((Real)(ta + (tb - ta) * (double)s / (double)role_refine));
    }
    tk3.push_back(tk.back());
    tk.swap(tk3);
    if (std::getenv("QJ_CORNER"))
      std::cout << "  [role-refine] x" << (long)role_refine << " span " << (long)role_span
                << " => " << (long)(tk.size() - 1) << " panels (was " << (long)n << ")\n";
  }
  // ---- Panel-boundary DIRECTION-JUMP diagnostic (QJ_ARM_PANEL_DIAG): confirms whether the breaks sit at role
  // transitions. Reports the max inter-panel chord-tangent turn at ROLE knots vs at interior-RUN knots.
  if (std::getenv("QJ_ARM_PANEL_DIAG") && (Integer)tk.size() >= 3) {
    const double dtol = 0.25 / (double)n;
    auto chord = [&](Integer p) { return unit(sub(curve(tk[(size_t)p + 1]), curve(tk[(size_t)p]))); };
    double role_max = 0, run_max = 0;
    for (Integer p = 1; p + 1 < (Integer)tk.size(); p++) {
      const Vec3<Real> c0 = chord(p - 1), c1 = chord(p);
      double d = (double)dot(c0, c1); d = std::max(-1.0, std::min(1.0, d));
      const double turn = std::acos(d) * 180.0 / M_PI;
      bool role = false; for (int b = 0; b < 4; b++) if (std::fabs((double)tk[(size_t)p] - role_tb[b]) < dtol) role = true;
      if (role) role_max = std::max(role_max, turn); else run_max = std::max(run_max, turn);
    }
    std::cout << "  [panel-diag] npan=" << (long)((Integer)tk.size() - 1)
              << " roleMaxTurn=" << role_max << " runMaxTurn=" << run_max << "\n";
  }
  const Integer npanel = (Integer)tk.size() - 1;
  const bool merged = (npanel != n);                                      // any corner panel merged?
  // Sample Chebyshev-per-panel.
  const Vector<Real>& cn = SlenderElemList<Real>::CenterlineNodes(cheb);
  Real dip_deepest = (Real)1;                                             // min applied radius / seam-taper radius
  // Fourier BUMP on tapered panels (QJ_ARM_DIP_FOURIER, default 0 = off). Where the radius taper thins the
  // tube (a waist), raise the azimuthal order on THAT panel proportional to the taper depth, so the tighter
  // near-quadrature aspect/toroidal ranges stay resolved. Capped at 52 (the densest cached toroidal rule).
  const Integer dipf_extra = std::getenv("QJ_ARM_DIP_FOURIER") ? (Integer)std::atoi(std::getenv("QJ_ARM_DIP_FOURIER")) : (Integer)0;
  for (Integer p = 0; p < npanel; p++) {
    const Real ta = tk[(size_t)p], tb = tk[(size_t)p + 1];
    Long fp = fourier;
    if (radius_dip && dipf_extra > 0) {
      Real rmin_ratio = (Real)1;
      for (Long j = 0; j < cheb; j++) {
        const Real t = merged ? (ta + cn[j] * (tb - ta)) : ((p + cn[j]) / (Real)n);
        const Real rl = rA + t*(rB - rA); if ((double)rl > 0) rmin_ratio = std::min(rmin_ratio, radius_at(t)/rl);
      }
      if ((double)rmin_ratio < 0.98) {
        Real def = (Real)1 - rmin_ratio; if ((double)def > 1) def = 1;
        fp = std::min<Long>((Long)52, fourier + (Long)std::lround((double)dipf_extra * (double)def));
      }
    }
    elem_order.PushBack(cheb); forder.PushBack(fp);
    for (Long j = 0; j < cheb; j++) {
      // No merge => reproduce the pre-merge expression EXACTLY (bit-identical bundles); merged panels are
      // non-uniform so must use the knot interval.
      const Real t = merged ? (ta + cn[j] * (tb - ta)) : ((p + cn[j]) / (Real)n);
      const Vec3<Real> Pc = curve(t); const Vec3<Real> o = orient_of(t);
      const Real rlin = rA + t*(rB - rA); const Real rr = radius_at(t);
      if ((double)rlin > 0 && (double)(rr/rlin) < (double)dip_deepest) dip_deepest = rr/rlin;
      coord.PushBack(Pc[0]); coord.PushBack(Pc[1]); coord.PushBack(Pc[2]);
      radius.PushBack(rr);
      orient.PushBack(o[0]); orient.PushBack(o[1]); orient.PushBack(o[2]);
    }
  }
  if (radius_dip && std::getenv("QJ_CORNER") && (double)dip_deepest < 0.999)
    std::cout << "  [dip] mid-arm radius thinned to x" << (double)dip_deepest
              << " of seam taper (floor x" << (double)dip_floor << ", mouths exact)\n";
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
