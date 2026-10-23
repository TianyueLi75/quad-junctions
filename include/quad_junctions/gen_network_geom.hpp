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
#include <quad_junctions/polyfit_segfit.hpp>
#include <fstream>
#include <iomanip>
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
  // CLAMPED mode (opt-in, used by the downsample-interpolate arm path): instead of the natural end BC
  // (M0=M_{n-1}=0, i.e. free end curvature) pin the end TANGENTS dP/ds to T0 at s=0 and T1 at s=total.
  // This lets the clamped C2 spline arrive/leave along the two junction axes exactly, so the short straight
  // coaxial leads join it C1 on a panel boundary (watertight mouth). Needs >=2 control points. clamped=false
  // => the natural path above is byte-identical.
  bool clamped = false; Vec3<Real> T0{(Real)0,(Real)0,(Real)0}, T1{(Real)0,(Real)0,(Real)0};
  void init(const std::vector<Vec3<Real>>& pts) {
    P = pts; s.assign(pts.size(), 0);
    for (size_t k = 1; k < pts.size(); k++) s[k] = s[k-1] + gnet::nrm(gnet::sub(pts[k], pts[k-1]));
    total = s.empty() ? 0 : s.back();
    M.clear();
    if (c2 && ((clamped && P.size() >= 2) || (!clamped && P.size() >= 3))) {  // cubic spline: solve tridiagonal for M (per component)
      const size_t n = P.size();
      std::vector<Real> h(n-1); for (size_t i=0;i+1<n;i++) h[i]=std::max((Real)1e-30, s[i+1]-s[i]);
      M.assign(n, Vec3<Real>{(Real)0,(Real)0,(Real)0});
      for (int d=0; d<3; d++) {                        // Thomas solve; natural (M0=M_{n-1}=0) or clamped end BC
        std::vector<Real> a(n,0), b(n,0), c(n,0), rhs(n,0), m(n,0);
        if (clamped) {                                 // first-derivative (tangent) BC at both ends
          b[0]=2*h[0]; c[0]=h[0];
          rhs[0]=6*(((double)P[1][d]-(double)P[0][d])/(double)h[0] - (double)T0[d]);
          a[n-1]=h[n-2]; b[n-1]=2*h[n-2];
          rhs[n-1]=6*((double)T1[d] - ((double)P[n-1][d]-(double)P[n-2][d])/(double)h[n-2]);
        } else {
          b[0]=1; b[n-1]=1;                            // natural BC
        }
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
// INTER-PANEL SEAM SMOOTHNESS CHECK for a just-built CSBQ slender arm (QJ_ARM_SEAM_CHECK, default ON).
//
// WHY this exists: CSBQ represents each arm panel as an independent order-`cheb` Lagrange polynomial
// through its own Chebyshev centerline nodes (slender_element.cpp: Derivative(dx,coord,nds),
// Derivative(d2x,dx,nds) -- EXACTLY the reconstruction mirrored below). Two neighbouring panels share a
// knot (the "seam"), but each reconstructs position / tangent / curvature there from ITS OWN nodes only.
// A smooth world curve does NOT guarantee the two per-panel polynomials agree at that shared knot: if the
// centerline bends too fast across a panel, the degree-(cheb-1) fit over-/under-shoots and the two sides
// disagree in tangent (C1) or curvature (C2). That disagreement is precisely what the per-panel toroidal
// special-quad rule cannot resolve, and it is the geometric origin of the DL "open space" / cross-list
// free-edge residuals. Measuring it from the CHEBYSHEV NODES ONLY (never curve()) reports the smoothness
// of the surface CSBQ actually sees -- a cheap intermediate gate BEFORE the BIE verification step.
//
// `coord` is the global element-major node accumulator; `coord0` is its Dim() just before this arm's
// panels were appended, `npanel`/`cheb` the panels/order of this arm. Reports, over all npanel-1 interior
// seams: max |Delta position| (watertight continuity), max tangent turn (deg, C1), max |Delta curvature|
// and its relative value (C2). Position/tangent/curvature are parametrization-invariant geometric
// quantities, so the differing local panel-interval widths across a seam cancel and the two sides are
// directly comparable. Pure diagnostic: reads `coord`, mutates nothing, byte-identical mesh.
//   QJ_ARM_SEAM_CHECK=0 silences; =2 (or QJ_CORNER) prints every seam, not just the per-arm max.
template <class Real> void report_arm_seam_smoothness(const Vector<Real>& coord, Long coord0,
    Integer npanel, Long cheb, const char* tag, std::ostream& os = std::cout) {
  using namespace gnet;
  const char* env = std::getenv("QJ_ARM_SEAM_CHECK");
  if (env && std::atoi(env) == 0) return;                                   // opt-out
  if (npanel < 2 || cheb < 2) return;                                       // no interior seam to check
  const bool verbose = (env && std::atoi(env) >= 2) || std::getenv("QJ_CORNER");

  const Vector<Real>& nds = SlenderElemList<Real>::CenterlineNodes(cheb);   // local Chebyshev nodes in [0,1]
  Vector<Real> tL(1), tR(1); tL[0] = (Real)0; tR[0] = (Real)1;              // panel endpoints (the shared knot)
  Vector<Real> wL, wR;                                                      // cheb x 1 Lagrange eval weights
  LagrangeInterp<Real>::Interpolate(wL, nds, tL);
  LagrangeInterp<Real>::Interpolate(wR, nds, tR);

  // Reconstruct position P, unit tangent T and curvature kappa at the panel endpoint selected by `w`
  // (wL=left/t=0, wR=right/t=1), from this panel's `cheb` nodes ONLY -- the same Lagrange-derivative chain
  // CSBQ uses internally (first derivative at nodes, then its derivative => second derivative at nodes).
  auto panel_end = [&](Integer p, const Vector<Real>& w, Vec3<Real>& P, Vec3<Real>& T, Real& kappa) {
    Vector<Real> c(3 * cheb), dx(3 * cheb), d2x(3 * cheb);                  // component-major (dof=3), like CSBQ
    for (Long j = 0; j < cheb; j++)
      for (int k = 0; k < 3; k++) c[k * cheb + j] = coord[coord0 + ((Long)p * cheb + j) * 3 + k];
    LagrangeInterp<Real>::Derivative(dx, c, nds);                          // dP/ds at nodes
    LagrangeInterp<Real>::Derivative(d2x, dx, nds);                        // d2P/ds2 at nodes
    Vec3<Real> d1{(Real)0,(Real)0,(Real)0}, d2{(Real)0,(Real)0,(Real)0};
    for (int k = 0; k < 3; k++) { P[k] = (Real)0; }
    for (Long j = 0; j < cheb; j++)
      for (int k = 0; k < 3; k++) {
        P[k]  += w[j] * c  [k * cheb + j];
        d1[k] += w[j] * dx [k * cheb + j];
        d2[k] += w[j] * d2x[k * cheb + j];
      }
    T = unit(d1);
    const Real sp = nrm(d1);                                               // |dP/ds|
    kappa = ((double)sp > 1e-30) ? nrm(cross(d1, d2)) / (sp * sp * sp) : (Real)0;  // |r' x r''|/|r'|^3, invariant
  };

  double max_dpos = 0, max_dang = 0, max_dkap = 0, max_rkap = 0;
  Integer worst_pos = -1, worst_ang = -1, worst_kap = -1;
  for (Integer p = 0; p + 1 < npanel; p++) {
    Vec3<Real> PR, TR, PL, TL2; Real kR, kL;
    panel_end(p,     wR, PR, TR, kR);                                      // right end of panel p
    panel_end(p + 1, wL, PL, TL2, kL);                                     // left end of panel p+1 (shared knot)
    const double dpos = (double)nrm(sub(PR, PL));
    double cd = (double)dot(TR, TL2); cd = std::max(-1.0, std::min(1.0, cd));
    const double dang = std::acos(cd) * 180.0 / M_PI;
    const double dkap = std::fabs((double)kR - (double)kL);
    const double rkap = dkap / std::max(1e-30, std::max((double)kR, (double)kL));
    if (dpos > max_dpos) { max_dpos = dpos; worst_pos = p; }
    if (dang > max_dang) { max_dang = dang; worst_ang = p; }
    if (dkap > max_dkap) { max_dkap = dkap; worst_kap = p; }
    max_rkap = std::max(max_rkap, rkap);
    if (verbose)
      os << "  [seam-check" << (tag ? std::string(" ") + tag : "") << "] seam " << (long)p << "|" << (long)(p + 1)
         << "  dpos=" << dpos << "  dtang=" << dang << " deg  dkappa=" << dkap << " (rel " << rkap << ")\n";
  }
  os << "  [seam-check" << (tag ? std::string(" ") + tag : "") << "] " << (long)(npanel - 1)
     << " interior seams: max dpos=" << max_dpos << " (seam " << (long)worst_pos << ")"
     << "  max dtang=" << max_dang << " deg (seam " << (long)worst_ang << ")"
     << "  max dkappa=" << max_dkap << " (rel " << max_rkap << ", seam " << (long)worst_kap << ")\n";
}


// ===================================================================================================
// DOWNSAMPLE-INTERPOLATE arm centerline -- INTERNAL short-arm FALLBACK for append_polyfit_fiber only.
// (Historically the user-selectable QJ_ARM_DOWNSAMPLE "attempt 2"; retained solely as polyfit's fallback
//  for arms too short to carry a poly-fit body window. Not reachable from the assembler dispatch.)
// ---------------------------------------------------------------------------------------------------
// The vmtk centerline is already smooth (vis/network_pipeline_pair*), so this path SMOOTHS the discrete-
// curvature noise with a plain moving average, DOWNSAMPLES to a sparse set of UNIFORM-arclength
// representative knots, and interpolates them with ONE globally-C2 NATURAL cubic spline (zero end
// curvature). Uniform knot spacing (not data-adaptive) means the natural cubic cannot Runge-overshoot, so
// no curvature limiter / radius dip / corner relax / monotone collapse is needed.
//
// Watertight mouths & the C2 lead join: short STRAIGHT coaxial leads along uA (home) and endTang (far),
// each pinned to EXACTLY ONE panel, so the panel polynomial is an exact line whose extrapolation to t=0/1
// lands on the seam center with axis uA/-uB -- correct even though the Chebyshev nodes never sample the
// panel ENDPOINTS (the key pitfall: a panel's last node is NOT its interval end, so the mouth ring must
// come from a genuinely straight lead, never "a node at t=0"). The spline spans QA->QB and carries COLLINEAR
// ghost knots just inside each lead end; a natural cubic through collinear knots is EXACTLY straight (all
// 2nd-moments zero), so it leaves QA along uA / arrives QB along uB with tangent exact AND curvature zero.
// The glued straight lead therefore joins the spline fully C2 (position, tangent, curvature=0 all match) --
// the lead is a zero-curvature line and C2 matches it with nothing clamped, hence no overshoot. Caps: the
// far lead is coaxial with the cap axis and out_termC/out_termU report the ACTUAL meshed terminal ring so
// QJ_CAP_TERM_REG seats the dome flush (same cap-care recipe).
template <class Real> void append_downsampled_fiber(const ArmSeam<Real>& seamA, const Vec3<Real>& endC,
    const Vec3<Real>& endTang, const std::vector<Vec3<Real>>& cl_in, Real rA, Real rB,
    Integer n_axial, Integer lead_panels, Long cheb, Long fourier,
    Vector<Long>& elem_order, Vector<Long>& forder, Vector<Real>& coord, Vector<Real>& radius, Vector<Real>& orient,
    bool constant_orient, Vec3<Real> endE1, bool is_cap,
    Vec3<Real>* out_termC, Vec3<Real>* out_termU, const char* dump_tag) {
  using namespace gnet;
  const bool diag = (std::getenv("QJ_CORNER") != nullptr);
  const Vec3<Real> A0 = seamA.C;
  const Vec3<Real> uA = unit(seamA.u);
  const Vec3<Real> fwdB = unit(endTang);                 // forward travel direction at the far end
  const Vec3<Real> uB = scal((Real)-1, fwdB);            // junction B's arm-exit axis (= -endTang)
  const Vec3<Real> chord = sub(endC, A0); const Real L = nrm(chord);
  const Real r_arm = std::max(rA, rB);

  // Degenerate arm => exact straight tapered tube (handled by the dedicated helper).
  if ((double)L < 1e-12 || cl_in.size() < 2) {
    append_straight_fiber<Real>(seamA, endC, rA, rB, std::max<Integer>(6, n_axial), cheb, fourier,
                                elem_order, forder, coord, radius, orient);
    if (out_termC) *out_termC = endC;
    if (out_termU) *out_termU = uA;
    return;
  }

  // ---- SMOOTH the FULL vmtk centerline up front (simple moving average, global ends fixed). The vmtk
  // skeleton is already smooth; only its *discrete* curvature is noisy, and a few averaging passes open the
  // tightest bends (here raw rho 0.73r -> 2.1r) without moving the path at figure scale. Doing it on the WHOLE
  // curve -- not on the later extracted window -- is essential: a tight bend often sits right next to a mouth,
  // and a window-local average that pins its endpoints would leave exactly that bend unsmoothed. ----
  const Integer sm_passes = std::getenv("QJ_DS_SMOOTH") ? std::max<Integer>(0,(Integer)std::atoi(std::getenv("QJ_DS_SMOOTH"))) : (Integer)8;
  std::vector<Vec3<Real>> clS = cl_in;
  for (Integer it = 0; it < sm_passes && clS.size() >= 3; it++) {
    std::vector<Vec3<Real>> sm = clS;
    for (size_t k = 1; k+1 < clS.size(); k++)
      sm[k] = scal((Real)0.25, add(add(clS[k-1], clS[k+1]), scal((Real)2, clS[k])));
    clS.swap(sm);
  }
  // ---- Taubin lambda|mu low-pass (DEFAULT ON, QJ_DS_LOWPASS passes, default 64; QJ_DS_LOWPASS=0 opts out =>
  // byte-identical to the moving-average-only path). Applied to INTERIOR arms only (NOT caps -- see below), on
  // the full SMOOTHED centerline BEFORE the window extraction / uniform downsample, so the sparse knots are
  // sampled from the denoised curve. The 3-point average above is a pure Laplacian SHRINK step (equivalent to
  // one Taubin lambda=0.5 half-pass with no anti-shrink): a weak low-pass that, on a tortuous interior arm,
  // leaves residual dkappa/ds curvature noise the toroidal special quadrature cannot resolve -> a ~0.5 DL
  // free-edge MID-ARM (see [[network-interior-arm-lowpass]]; measured on interior j120->133: DL arm 0.545 ->
  // 0.00424, 128x, with 64 Taubin passes). Taubin interleaves an anti-shrink mu step (mu<-lambda<0) so it
  // removes the high-frequency curvature band WITHOUT shrinking the bend (tortuosity preserved) -- the
  // production CL_LOWPASS fix (lambda 0.5 / mu -0.53). Global ends held fixed each half-step => mouth rings
  // untouched, watertightness-neutral.
  //   !is_cap: CAP (leaf) arms are EXCLUDED. A cap's far end is a free tip with a dome seated on the tube's
  // terminal; caps are already clean without the low-pass (DL ~0.006), and applying Taubin reshapes the near-
  // tip body curvature (rho/r 1.77->1.41 under the dome) and drives a ~0.9 DL free-edge there (measured on cap
  // j286->289: 0.006 -> 0.915). Re-deriving the cap axis from the smoothed centerline does NOT help (Taubin
  // barely moves the near-fixed-endpoint tangent); the only noisy arms that NEED the low-pass are the long
  // tortuous interior ones. So the low-pass is interior-only. (QJ_CAP_LOWPASS, which opted caps back in after a
  // tip trim, was reverted 2026-10-15: at full-net scale Taubin un-pins the cap's JUNCTION-SEAM endpoint and
  // spawns worse cross-list blowups -- the dominant cap error is the seam, not the tip. See s11 memory.) ----
  const Integer lp_passes = std::getenv("QJ_DS_LOWPASS") ? std::max<Integer>(0,(Integer)std::atoi(std::getenv("QJ_DS_LOWPASS"))) : (Integer)64;
  if (lp_passes > 0 && !is_cap && clS.size() >= 3) {
    const Real lp_lam = std::getenv("QJ_DS_LOWPASS_LAMBDA") ? (Real)std::atof(std::getenv("QJ_DS_LOWPASS_LAMBDA")) : (Real)0.5;
    const Real lp_mu  = std::getenv("QJ_DS_LOWPASS_MU")     ? (Real)std::atof(std::getenv("QJ_DS_LOWPASS_MU"))     : (Real)-0.53;
    auto lap_step = [&](Real w){                              // one uniform-weight Laplacian smoothing half-step, ends fixed
      std::vector<Vec3<Real>> sm = clS;
      for (size_t k = 1; k+1 < clS.size(); k++) {
        const Vec3<Real> lapk = sub(scal((Real)0.5, add(clS[k-1], clS[k+1])), clS[k]);
        sm[k] = add(clS[k], scal(w, lapk));
      }
      clS.swap(sm);
    };
    for (Integer it = 0; it < lp_passes; it++) { lap_step(lp_lam); lap_step(lp_mu); }
  }
  std::vector<Real> cs(clS.size(), (Real)0);
  for (size_t k = 1; k < clS.size(); k++) cs[k] = cs[k-1] + nrm(sub(clS[k], clS[k-1]));
  const Real total = cs.back();

  // ---- SEAM-CORNER mode (DEFAULT; the production seam-settle recipe). The
  // clamp_mode below forced the body-spline END knots QA/QB exactly onto the junction axis AND clamped the
  // end tangent -- a double constraint that bent the ~0.1r-off-axis vessel back onto the axis over the one
  // segment next to the clamp, the residual ~12x gap to the production seamsettle_tr6 baseline. Production
  // keeps the body on the TRUE centerline and bridges each mouth with ONE gentle quintic Hermite "seam
  // corner" that leaves the on-axis mouth ring coaxial (tangent = seam axis, zero endpoint acceleration) and
  // ARRIVES on the vessel with the vessel's own tangent -- re-seating the terminal ring flush under the seam
  // without straightening the vessel. Reproduced here: 1-panel straight coaxial lead (watertight mouth ring)
  // -> cp-panel quintic Hermite corner QA->P_hA -> downsampled C2 body spline over the real (un-forced)
  // window knots -> mirror corner -> lead. Opt out with QJ_DS_NO_SEAM_CORNER (=> old clamp/natural bridge).
  const bool seam_corner = (std::getenv("QJ_DS_NO_SEAM_CORNER") == nullptr);
  // ---- coaxial lead length: one scale-invariant stub, capped to a fraction of the arm. Seam-corner and the
  // legacy CLAMPED mode both use a SHORTER half-radius stub: the stub end QA/QB carries the on-axis mouth
  // ring, and the quintic corner (resp. clamped spline) curves onto the off-axis vessel over its own run; a
  // long stub would instead cram the turn into the one segment next to the stub and fold. The glued stub is
  // straight+coaxial so the mouth ring conforms either way. ----
  const bool clamp_mode = std::getenv("QJ_DS_CLAMP") != nullptr;
  Real Llead = (clamp_mode || seam_corner) ? (Real)0.5 * std::max(rA, rB) : std::max(rA, rB);
  if (const char* e = std::getenv("QJ_DS_LEAD")) Llead = std::max((Real)1e-9, (Real)std::atof(e));
  Llead = std::min(Llead, (Real)0.35 * total);           // leave room for a non-degenerate interior
  const Vec3<Real> QA = add(A0,   scal(Llead, uA));
  const Vec3<Real> QB = add(endC, scal(Llead, uB));      // QB = endC - Llead*endTang (so endC = QB + Llead*fwdB)

  // ---- point on the SMOOTHED centerline at arclength s (linear-interp) ----
  auto cl_at = [&](Real s) -> Vec3<Real> {
    if ((double)s <= 0) return clS.front();
    if ((double)s >= (double)total) return clS.back();
    size_t i = 0; while (i+1 < cs.size() && (double)cs[i+1] < (double)s) i++;
    const Real h = std::max((Real)1e-30, cs[i+1]-cs[i]); const Real w = (s-cs[i])/h;
    return add(scal((Real)1-w, clS[i]), scal(w, clS[i+1]));
  };
  // ---- interior knots: the raw vmtk centerline between the two lead ends [Llead, total-Llead], SMOOTHED by
  // a simple moving average and then DOWNSAMPLED to a sparse UNIFORM-arclength set of representative knots.
  // The vmtk centerline is already smooth (see vis/network_pipeline_pair*): only its *discrete* curvature
  // estimate is noisy, and that noise is removed by plain averaging -- no curvature limiter needed. Uniform
  // (not data-adaptive/RDP) knot spacing is what lets a global NATURAL C2 cubic interpolate them WITHOUT the
  // non-uniform-knot Runge overshoot that previously forced a fallback to the mere-C1 centripetal curve. ----
  // The coaxial straight entry occupies two collinear stretches at each mouth: the glued lead A0->QA (length
  // Llead) plus a ghost run QA->QA+gspan*uA that the spline reproduces straight. The interior window must
  // therefore begin at axial depth (Llead+gspan) past A0 and end that far back from endC. Pin it by
  // PROJECTION onto the junction axis, NOT by arclength-from-cl[0]: the vmtk centerline can start/end a short
  // way BEHIND the seam center (it runs into the junction body -- here cl[0] is ~2r behind A0), so arclength
  // Llead from cl[0] lands behind QA and the collinear lead then doubles back over it (a 180 deg control
  // reversal => the tube self-folds). bse = first arclength whose forward projection past A0 reaches the ghost
  // depth; bsx = last arclength whose projection back from endC reaches it. ----
  Real gspan = std::getenv("QJ_DS_GHOST_LEN") ? (Real)std::atof(std::getenv("QJ_DS_GHOST_LEN")) : (Real)2 * r_arm;
  gspan = std::max((Real)0, std::min(gspan, (Real)0.25 * total));
  // TURN-ADAPTIVE corner room: the mouth RING axis is dictated by the junction (uA / uB) and generally differs
  // from the centerline's local tangent by some swing angle theta (here the far junction's arc-window axis is
  // ~64 deg off the centerline). A radius-r tube physically cannot turn theta in less than r*theta of
  // arclength, so the spline MUST be given >= target*r*theta of run to swing the axis onto the tangent or it
  // folds (rho = run/theta < r). Begin the interior that much deeper at the misaligned mouth: the spline then
  // rounds the swing as ONE gentle C2 corner (rho ~ run/theta ~ Kturn*r). theta is measured from each mouth
  // axis to the arm's general direction over the near third (robust to the tight vmtk kink at the very seam).
  auto ang_rad = [](const Vec3<Real>& p, const Vec3<Real>& q){ double d=(double)dot(unit(p),unit(q)); d=std::max(-1.0,std::min(1.0,d)); return (Real)std::acos(d); };
  // Mouth swing = the TOTAL angle the spline must bend from the junction-dictated mouth axis onto the
  // centerline. The centerline keeps turning through the near-mouth zone (its tangent can match the axis right
  // at the seam yet swing away a few radii in -- exactly the far mouth of this interior arm, where the raw
  // rho dips to 0.73r before the smoothing opens it), so measure the MAX angular deviation of the (lightly
  // smoothed) tangent from the mouth axis over the near-mouth ~45%% -- not a single end tangent. The corner
  // room below then grants that bend r*theta of run so rho stays >= Kturn*r.
  auto cl_tan = [&](Real s)->Vec3<Real>{ const Real hh=(Real)0.03*total+(Real)1e-9; return unit(sub(cl_at(std::min(total,s+hh)), cl_at(std::max((Real)0,s-hh)))); };
  Real thA = 0, thB = 0; { const Integer NS = 40; const Real span = (Real)0.45*total;
    for (Integer i = 0; i <= NS; i++) { const Real s=(Real)i/(Real)NS*span;        thA = std::max(thA, ang_rad(uA, cl_tan(s))); }
    for (Integer i = 0; i <= NS; i++) { const Real s=total-(Real)i/(Real)NS*span;  thB = std::max(thB, ang_rad(uB, scal((Real)-1, cl_tan(s)))); }
  }
  thA = std::min(thA, (Real)(0.5*M_PI)); thB = std::min(thB, (Real)(0.5*M_PI));     // cap the room at a right angle
  const Real Kturn = std::getenv("QJ_DS_KTURN") ? (Real)std::atof(std::getenv("QJ_DS_KTURN")) : (Real)1.6;
  Real depthA = Llead + gspan + Kturn * r_arm * thA;     // axial depth (past A0 along uA) where the interior begins
  Real depthB = Llead + gspan + Kturn * r_arm * thB;     // axial depth (back from endC along uB) where it ends
  depthA = std::min(depthA, (Real)0.45*total);           // keep a non-degenerate interior window
  depthB = std::min(depthB, (Real)0.45*total);
  // CLAMPED mode (QJ_DS_CLAMP): decouple ring-conformance from inward straightness. The natural-cubic-through-
  // collinear-lead design pins the mouth tangent to uA/uB by forcing the spline STRAIGHT over Llead+gspan+room
  // -- but where the vessel bends within a few radii of a mouth (the interior far mouth here) that forced-
  // straight run OVERRUNS the bend and compresses the vessel's own gentle ~2.9r turn into a ~0.48r fold. A
  // clamped C2 spline pins the END TANGENT to the junction axis (ring still conforms, straight coaxial stub
  // still joins C1 on the lead panel boundary) while leaving end CURVATURE free, so the spline curves onto the
  // centerline immediately instead of straightening it. No ghost run: the body starts at the coaxial-stub
  // depth Llead itself.
  if (clamp_mode && !seam_corner) { depthA = Llead; depthB = Llead; }
  // arclength at a given projection depth past A0 (forward) / back from endC (backward).
  auto sdepth_fwd = [&](Real depth)->Real{ const Integer NS=2000;
    for (Integer i=0;i<=NS;i++){ const Real s=(Real)i/(Real)NS*total; if ((double)dot(sub(cl_at(s),A0),uA)>=(double)depth) return s; } return (Real)0.45*total; };
  auto sdepth_bwd = [&](Real depth)->Real{ const Integer NS=2000;
    for (Integer i=NS;i>=0;i--){ const Real s=(Real)i/(Real)NS*total; if ((double)dot(sub(cl_at(s),endC),uB)>=(double)depth) return s; } return (Real)0.55*total; };

  Real bse = 0, bsx = total;
  if (seam_corner) {
    // ADAPTIVE per-corner chord extension -- the downsample analog of the production CORNER_RELAX + mutual-
    // borrow. Instead of a fixed room, hand the body off to each corner as SHALLOW
    // as possible and deepen ONLY the corner that folds: while a corner's continuous quintic rho_min < tgt*r,
    // push its handoff pin deeper (bse up / bsx down), which lengthens the Hermite chord |Q-Ph| and its handle
    // h=kr*|chord|, so the same axis->vessel swing spreads over more arclength => rho rises. MUTUAL-BORROW cap:
    // a pin may push toward mid-arm as long as >= min_run of body remains between the pins (bse+min_run<=bsx),
    // so a tight corner borrows the slack corner's room. This extends the chord ONLY at tight turns, keeping
    // every other corner shallow (minimal deviation from the real vessel => best fidelity + closure), exactly
    // the production lever. QJ_DS_CORNER_TGT (default 1.6*r) is the fold-clear target; QJ_DS_CORNER_ROOM the
    // shallow START depth (units of r); QJ_DS_CORNER_RUN the min body run kept between pins (fraction of total).
    const Vec3<Real>& fwd = fwdB;
    auto h5min_rho = [&](const Vec3<Real>& P0,const Vec3<Real>& T0,Real h0,const Vec3<Real>& P1,const Vec3<Real>& T1,Real h1)->Real{
      const Integer M=100; const Real hh=(Real)1/(M+2); Real km=0;
      for (Integer i=1;i<M;i++){ const Real s=(Real)i/M;
        const Vec3<Real> xm=HybridAssembly<Real>::hermite5(P0,T0,h0,P1,T1,h1,s-hh);
        const Vec3<Real> x0=HybridAssembly<Real>::hermite5(P0,T0,h0,P1,T1,h1,s);
        const Vec3<Real> xp=HybridAssembly<Real>::hermite5(P0,T0,h0,P1,T1,h1,s+hh);
        const Vec3<Real> d1=scal((Real)1/(2*hh),sub(xp,xm)); const Vec3<Real> d2=scal((Real)1/(hh*hh),sub(add(xp,xm),scal((Real)2,x0)));
        const Real sp=nrm(d1),cn2=nrm(cross(d1,d2)); if ((double)sp>1e-30){ const Real k=cn2/(sp*sp*sp); if ((double)k>(double)km) km=k; } }
      return (double)km>1e-30 ? (Real)1/km : (Real)1e30; };
    const Real ckr = std::getenv("QJ_DS_CORNER_KR") ? (Real)std::atof(std::getenv("QJ_DS_CORNER_KR")) : (Real)0.8;
    const Real cfl = std::getenv("QJ_DS_CORNER_FLOOR") ? (Real)std::atof(std::getenv("QJ_DS_CORNER_FLOOR")) : (Real)1.00;
    const Real cca = std::getenv("QJ_DS_CORNER_CAP")   ? (Real)std::atof(std::getenv("QJ_DS_CORNER_CAP"))   : (Real)1.00;
    auto fh = [&](const Vec3<Real>& T,const Vec3<Real>& chord)->Real{ const Real L=nrm(chord); if ((double)L<1e-30) return (Real)0;
      return ckr*L*std::min(cca,std::max(cfl,dot(T,unit(chord)))); };
    auto rhoA_at = [&](Real s)->Real{ const Vec3<Real> Ph=cl_at(s),tn=cl_tan(s),ch=sub(Ph,QA);
      return h5min_rho(QA,uA,fh(uA,ch),Ph,tn,fh(tn,ch)); };
    auto rhoB_at = [&](Real s)->Real{ const Vec3<Real> Ph=cl_at(s),tn=cl_tan(s),ch=sub(QB,Ph);
      return h5min_rho(Ph,tn,fh(tn,ch),QB,fwd,fh(fwd,ch)); };
    // room0 = the START corner depth (units of r) before adaptive deepening. 2*r is the measured full-n298
    // closure sweet spot: the near-seam swing is already gentle there, and the adaptive loop then deepens ONLY
    // the still-folding corners further. A shallow start (0.75r) clears folds too but PUTS the swing in a short
    // near-seam chord -> more curved near-seam panels -> worse whole-net closure (0.171 vs 0.085 at 2r), the
    // opposite of what the sub8 subset suggests. See [[network-downsample-arm-pipeline]].
    const Real room0 = (std::getenv("QJ_DS_CORNER_ROOM") ? (Real)std::atof(std::getenv("QJ_DS_CORNER_ROOM")) : (Real)2) * r_arm;
    const Real tgt   = (std::getenv("QJ_DS_CORNER_TGT")  ? (Real)std::atof(std::getenv("QJ_DS_CORNER_TGT"))  : (Real)1.6) * r_arm;
    const Real min_run = (std::getenv("QJ_DS_CORNER_RUN") ? (Real)std::atof(std::getenv("QJ_DS_CORNER_RUN")) : (Real)0.12) * total;
    const Real step = (Real)0.01*total; const Integer gmax = (Integer)130;
    bse = sdepth_fwd(Llead + room0); bsx = sdepth_bwd(Llead + room0);
    const Real bse0 = bse, bsx0 = bsx;
    { Integer g=0; while ((double)rhoA_at(bse)<(double)tgt && (double)(bse+min_run)<(double)bsx && g++<gmax) bse+=step; }  // near first
    { Integer g=0; while ((double)rhoB_at(bsx)<(double)tgt && (double)(bsx-min_run)>(double)bse && g++<gmax) bsx-=step; }  // far
    { Integer g=0; while ((double)rhoA_at(bse)<(double)tgt && (double)(bse+min_run)<(double)bsx && g++<gmax) bse+=step; }  // near borrows
    if (diag)
      std::cout << "  [ds-corner-relax] " << (dump_tag?dump_tag:"") << " near rho/r=" << (double)(rhoA_at(bse)/r_arm)
                << " far rho/r=" << (double)(rhoB_at(bsx)/r_arm) << "  bse/tot " << (double)(bse0/total) << "->" << (double)(bse/total)
                << "  bsx/tot " << (double)(bsx0/total) << "->" << (double)(bsx/total) << " (tgt " << (double)(tgt/r_arm) << ")\n";
  } else {
    bse = sdepth_fwd(depthA); bsx = sdepth_bwd(depthB);
  }
  if ((double)bsx <= (double)bse + 1e-9) { bse = (Real)0.4*total; bsx = (Real)0.6*total; }  // degenerate guard
  std::vector<Vec3<Real>> dense; dense.push_back(cl_at(bse));   // already smoothed (clS); window endpoints opened too
  for (size_t k = 0; k < clS.size(); k++)
    if ((double)cs[k] > (double)bse + 1e-9 && (double)cs[k] < (double)bsx - 1e-9) dense.push_back(clS[k]);
  dense.push_back(cl_at(bsx));
  std::vector<Real> dcs(dense.size(), (Real)0);
  for (size_t k = 1; k < dense.size(); k++) dcs[k] = dcs[k-1] + nrm(sub(dense[k], dense[k-1]));
  const Real dtot = std::max((Real)1e-30, dcs.back());
  auto dense_at = [&](Real s) -> Vec3<Real> {
    if ((double)s <= 0) return dense.front();
    if ((double)s >= (double)dtot) return dense.back();
    size_t i = 0; while (i+1 < dcs.size() && (double)dcs[i+1] < (double)s) i++;
    const Real h = std::max((Real)1e-30, dcs[i+1]-dcs[i]); const Real w = (s-dcs[i])/h;
    return add(scal((Real)1-w, dense[i]), scal(w, dense[i+1]));
  };

  // ---- control points at ONE UNIFORM arclength spacing hstep (the user's prescription): a mixed grid of
  // coarse lead knots and fine body knots makes the natural cubic Runge-overshoot at the spacing jump (the
  // 0.167 fold), so every knot -- on-axis lead runs AND the smoothed centerline body -- sits on the same
  // spacing. Three pieces, each collinear/smooth: (1) home lead run ON THE uA AXIS from QA (depth Llead) out
  // to depth depthA; (2) the smoothed centerline body over [bse,bsx]; (3) far lead run on the uB axis from
  // depth depthB back to QB. The two lead runs are collinear, so the NATURAL cubic (zero end curvature)
  // reproduces them EXACTLY straight (all 2nd-moments zero) and leaves QA along uA / arrives QB along uB with
  // tangent exact and curvature zero -- the glued straight lead then joins fully C2. The lead runs reach the
  // interior depth, so when the mouth axis is misaligned the spline rounds that swing over Kturn*r*theta of
  // run (rho ~ Kturn*r), and when it is aligned the body simply continues straight. QJ_DS_H sets hstep. ----
  const Real hstep = std::getenv("QJ_DS_H") ? std::max((Real)1e-6,(Real)std::atof(std::getenv("QJ_DS_H"))) : (Real)1.0 * r_arm;
  const Integer Kmax = std::getenv("QJ_DS_NKNOT_MAX") ? std::max<Integer>(2,(Integer)std::atoi(std::getenv("QJ_DS_NKNOT_MAX"))) : (Integer)200;
  auto n_of = [&](Real len)->Integer{ return std::max<Integer>(1, (Integer)std::llround((double)(len/hstep))); };
  // QJ_DS_NO_PANEL_SNAP opts OUT of panel-snap (DEFAULT ON): align the body degree-10 panels to the
  // C2 spline knots so NO panel straddles a knot's 3rd-derivative jump -- the [cheb interp err] body spike
  // (~1e-4 vs ~1e-12 at the seam) that was the CSBQ formation error behind the full-net downsample DL hotspot.
  // It coarsens the body spline to exactly nmid segments (one per body panel) and later SHIFTS the body panel
  // boundaries onto those knots -- shift, not split, so panel count/size are unchanged (unlike knot-snap's
  // slivers that broke the toroidal aspect range). Leads/corners untouched. Full-net gate (job 7250799): DL
  // const-density max rel err 0.227 -> 9.63e-4, closure 0.1476 -> 5.995e-3, no aspect warnings.
  const bool panel_snap = std::getenv("QJ_DS_NO_PANEL_SNAP") == nullptr;
  // Vessel tangents at the body-window ends (used by the seam corner AND the clamped body-end tangents so the
  // quintic corner joins the body C1). cl_tan is the lightly-smoothed centerline tangent.
  const Vec3<Real> tanA = cl_tan(bse), tanB = cl_tan(bsx);
  std::vector<Vec3<Real>> ctrl;
  if (seam_corner) {
    // Body = the REAL (un-forced, off-axis) smoothed-centerline window knots, uniform arclength. No axis
    // overwrite and no on-axis lead run: the quintic Hermite corners (in curve() below) bridge the on-axis
    // mouth stubs QA/QB to the true window ends P_hA=ctrl.front()/P_hB=ctrl.back(), matching the vessel
    // tangent there. The body is a CLAMPED C2 cubic with end tangents = tanA/tanB so each corner meets it C1.
    // panel_snap: coarsen to exactly nmid (= body-panel count) uniform-arclength knots so each body panel is one
    // spline segment (boundaries land on knots below); otherwise the fine knot grid (~dtot/r) that the panels
    // straddle. The sparse segments are smoother (low fidelity is intended here) and NEVER straddled.
    // snap: K = nmid exactly (one segment per body panel) with NO Kmax cap -- Kmax bounds the FINE-knot spline
    // (non-snap), but here K is just the body-panel count (already bounded by the assembler's n_arm), and
    // capping it below nmid would break the knot<->panel 1:1 map so the shift guard skips the arm (the six
    // ctrl=201 straddle holdouts). Floor 2 keeps >=3 ctrl points for the cubic.
    const Integer K = panel_snap ? std::max<Integer>(2, n_axial)
                                 : std::min(Kmax, n_of(dtot));
    for (Integer k = 0; k <= K; k++) ctrl.push_back(dense_at((Real)k/(Real)K * dtot));
  } else if (clamp_mode) {
    // One UNIFORM knot list over the body window [depth Llead .. depth Llead] (depthA=depthB=Llead above), with
    // the two END knots OVERWRITTEN by the exact on-axis QA/QB (they coincide with the window ends to within
    // the coaxial-stub tolerance, so this only snaps them onto the axis -- it does not perturb spacing). End
    // tangents are clamped to the junction axes below. NOTE: appending QA/QB on TOP of the window-end knots
    // would place a near-coincident control point next to the clamped end => a curvature spike (Runge at the
    // clamp); overwriting keeps the spacing uniform and the spline fold-free.
    const Integer K = std::min(Kmax, n_of(dtot));
    for (Integer k = 0; k <= K; k++) ctrl.push_back(dense_at((Real)k/(Real)K * dtot));
    ctrl.front() = QA; ctrl.back() = QB;
  } else {
    { const Real runA = std::max((Real)0, depthA - Llead); const Integer nA = n_of(runA);        // home lead run on axis
      for (Integer j = 0; j <= nA; j++) ctrl.push_back(add(A0, scal(Llead + runA*(Real)j/(Real)nA, uA))); }
    { const Integer K = std::min(Kmax, n_of(dtot));                                              // centerline body
      for (Integer k = 1; k <= K-1; k++) ctrl.push_back(dense_at((Real)k/(Real)K * dtot)); }     // interior only (ends added by lead runs)
    { const Real runB = std::max((Real)0, depthB - Llead); const Integer nB = n_of(runB);        // far lead run on axis
      for (Integer j = nB; j >= 0; j--) ctrl.push_back(add(endC, scal(Llead + runB*(Real)j/(Real)nB, uB))); }
  }

  // QJ_DS_NO_CORNER_C2 opts OUT (DEFAULT ON): make the body spline NATURAL (zero end curvature) instead of
  // clamped-to-vessel-tangent, and let the hermite5 corner arrive at the body's own natural start tangent. A
  // clamped body pins the end TANGENT, and the sparse cubic then develops a large end CURVATURE (a 16-25/r clamp
  // spike) to honor it; the C1 corner (zero end curvature) meets that spike with a dkappa/ds STEP at the panel
  // boundary ~s/L~0.13. A NATURAL body has zero end curvature, so the zero-curvature corner end meets it C2 (both
  // kappa=0 at the join, ramping up smoothly into the body) with NO spike and NO fold risk -- fold-free keeper.
  const bool corner_c2 = std::getenv("QJ_DS_NO_CORNER_C2") == nullptr;
  ArcPolyline<Real> bcl; bcl.c2 = true;
  if (seam_corner)       { bcl.clamped = !corner_c2; bcl.T0 = tanA; bcl.T1 = tanB; }  // clamp tangents=vessel (C1); C2=>natural
  else if (clamp_mode)   { bcl.clamped = true; bcl.T0 = uA;   bcl.T1 = fwdB; }   // pin mouth tangents; curvature free
  else bcl.clamped = false;                                            // NATURAL C2: straight leads = collinear knots
  bcl.init(ctrl);
  // Corner body-end tangents: vessel tangent (clamped body, C1 join) OR the natural body's OWN end tangent (C2
  // join -- the corner then arrives exactly along the body, both ends at zero curvature).
  Vec3<Real> teA = tanA, teB = tanB;
  if (corner_c2) { const Real eps = (Real)1e-3;
    teA = unit(sub(bcl.eval(eps), bcl.eval((Real)0)));
    teB = unit(sub(bcl.eval((Real)1), bcl.eval((Real)1-eps))); }

  // ---- panelization: one lead panel each side (per the assembler), plus (seam_corner) one quintic corner
  // panel-block each side, the rest spread over the spline body.
  const Integer lp = std::max<Integer>(1, lead_panels);
  const Integer cp = seam_corner ? (std::getenv("QJ_DS_CORNER_PANELS")
                     ? std::max<Integer>(1, (Integer)std::atoi(std::getenv("QJ_DS_CORNER_PANELS"))) : lp) : 0;
  const Integer nmid = std::max<Integer>(1, n_axial);
  const Integer n = nmid + 2*lp + 2*cp;                  // total panels before turn-refine
  // panel-fraction boundaries (seam_corner): lead A [0,wLA] | corner A [wLA,wCA] | body [wCA,wCB] | corner B
  // [wCB,wLB] | lead B [wLB,1]. For clamp/natural (cp=0) wCA=wLA and wCB=wLB collapse to the 3-piece layout.
  const Real wLA = (Real)lp/(Real)n, wCA = (Real)(lp+cp)/(Real)n;
  const Real wCB = (Real)(n-lp-cp)/(Real)n, wLB = (Real)(n-lp)/(Real)n;
  // Quintic Hermite seam-corner geometry: unit end tangents (axis uA/fwdB at the stub, vessel tanA/tanB at the
  // body) scaled by a PER-END Ferguson handle, zero endpoint acceleration (same hermite5 the production
  // seam-settle uses). P_hA/P_hB are the true (off-axis) body-window ends. The handle is
  // h = corner_kr*|chord|*clamp(dot(tangent, chordhat), floor, cap). The Ferguson projection (floor<1) SHRINKS
  // a handle whose tangent points away from the chord -- the production QJ_ARM_FERGUSON overshoot fix. But the
  // seam corner's failure is the OPPOSITE: it joins an on-axis stub to an off-axis vessel end and needs a LONG
  // handle to swing gently, so shrinking the handle tightens the fold. Hence Ferguson is OFF by default here
  // (floor=cap=1 => h=corner_kr*|chord|) and the handle is set generously (corner_kr=0.8); set QJ_DS_CORNER_FLOOR<1
  // to re-enable the projection. corner_kr sweet spot is ~0.8-1.0 (0.6 whips, >1.2 overshoots).
  const Real corner_kr = std::getenv("QJ_DS_CORNER_KR") ? (Real)std::atof(std::getenv("QJ_DS_CORNER_KR")) : (Real)0.8;
  const Real cfloor = std::getenv("QJ_DS_CORNER_FLOOR") ? (Real)std::atof(std::getenv("QJ_DS_CORNER_FLOOR")) : (Real)1.00;
  const Real ccap   = std::getenv("QJ_DS_CORNER_CAP")   ? (Real)std::atof(std::getenv("QJ_DS_CORNER_CAP"))   : (Real)1.00;
  const Vec3<Real> PhA = ctrl.front(), PhB = ctrl.back();
  auto ferg_h = [&](const Vec3<Real>& T, const Vec3<Real>& chord) -> Real {
    const Real Lc = nrm(chord); if ((double)Lc < 1e-30) return (Real)0;
    const Vec3<Real> ch = unit(chord);
    return corner_kr * Lc * std::min(ccap, std::max(cfloor, dot(T, ch)));
  };
  const Vec3<Real> chA = sub(PhA, QA), chB = sub(QB, PhB);
  const Real hA0 = ferg_h(uA, chA),  hA1 = ferg_h(teA, chA);      // corner A: QA(uA) -> PhA(teA)
  const Real hB0 = ferg_h(teB, chB), hB1 = ferg_h(fwdB, chB);     // corner B: PhB(teB) -> QB(fwdB)
  if (corner_c2 && diag) std::cout << "    [ds-c2] natural body; teA.tanA=" << (double)dot(teA,tanA)
                                   << " teB.tanB=" << (double)dot(teB,tanB) << "\n";

  // Piecewise centerline c(t). seam_corner: lead -> quintic corner -> C2 body spline -> quintic corner -> lead.
  // clamp/natural: straight lead A0->QA, C2 spline QA->QB, straight lead QB->endC.
  auto curve = [&](Real t) -> Vec3<Real> {
    if (seam_corner) {
      if ((double)t <= (double)wLA) { const Real q = (double)wLA>1e-30 ? t/wLA : (Real)0; return add(A0, scal(q*Llead, uA)); }
      if ((double)t <  (double)wCA) { const Real s = (t-wLA)/(wCA-wLA);         // corner A: stub(uA) -> body(teA)
        return HybridAssembly<Real>::hermite5(QA, uA, hA0, PhA, teA, hA1, s); }
      if ((double)t <= (double)wCB) { const Real tau = (t-wCA)/(wCB-wCA); return bcl.eval(tau); }
      if ((double)t <  (double)wLB) { const Real s = (t-wCB)/(wLB-wCB);         // corner B: body(teB) -> stub(fwdB)
        return HybridAssembly<Real>::hermite5(PhB, teB, hB0, QB, fwdB, hB1, s); }
      const Real q = (double)(1-wLB)>1e-30 ? (t-wLB)/((Real)1-wLB) : (Real)1; return add(QB, scal(q*Llead, fwdB));
    }
    if ((double)t <= (double)wLA) { const Real q = (double)wLA>1e-30 ? t/wLA : (Real)0; return add(A0, scal(q*Llead, uA)); }
    if ((double)t >= (double)wLB) { const Real q = (double)(1-wLB)>1e-30 ? (t-wLB)/((Real)1-wLB) : (Real)1; return add(QB, scal(q*Llead, fwdB)); }
    const Real tau = (t - wLA)/(wLB - wLA);
    return bcl.eval(tau);
  };
  auto radius_at = [&](Real t) -> Real { return rA + t*(rB - rA); };      // exact seam-conforming linear taper

  // Min osculating radius of the spline body (honest fold report; sparse decimation should keep rho >> r).
  // Sample the spline on its OWN parameter (bcl.eval) so the stencil never straddles the C1 lead/spline
  // joins -- doing that on curve(t) reported a spurious ~0 curvature right at the mouths (the earlier
  // false 494-FOLD spam). The joins are deliberate curvature steps on panel boundaries, not folds.
  if (diag) {
    const Integer M = 400; Real kmax = 0; const Real h = (Real)1/M;
    for (Integer i = 1; i < M; i++) {
      const Real tt = (Real)i/M;                          // spline-local parameter in (0,1)
      const Vec3<Real> xm = bcl.eval(tt-h), x0 = bcl.eval(tt), xp = bcl.eval(tt+h);
      const Vec3<Real> d1 = scal((Real)1/(2*h), sub(xp, xm));
      const Vec3<Real> d2 = scal((Real)1/(h*h), sub(add(xp, xm), scal((Real)2, x0)));
      const Real sp = nrm(d1), cn2 = nrm(cross(d1, d2));
      if ((double)sp > 1e-30) { const Real k = cn2/(sp*sp*sp); if ((double)k > (double)kmax) kmax = k; }
    }
    const Real rho_body = (double)kmax > 1e-30 ? (Real)1/kmax : (Real)1e30;
    // seam_corner: also sample each quintic corner on curve(t) WITHIN its own (wLA,wCA)/(wCB,wLB) interval so
    // the stencil never straddles a C1 panel-boundary join (same reason the body is sampled on bcl.eval).
    Real rho_cA = (Real)1e30, rho_cB = (Real)1e30;
    if (seam_corner) {
      auto corner_rho = [&](Real lo, Real hi) -> Real {
        const Integer MC = 200; const Real span = hi - lo; if ((double)span <= 1e-12) return (Real)1e30;
        const Real hh = span / (MC + 2); Real km = 0;
        for (Integer i = 1; i < MC; i++) {
          const Real tt = lo + span*(Real)i/MC;
          const Vec3<Real> xm = curve(tt-hh), x0 = curve(tt), xp = curve(tt+hh);
          const Vec3<Real> d1 = scal((Real)1/(2*hh), sub(xp, xm));
          const Vec3<Real> d2 = scal((Real)1/(hh*hh), sub(add(xp, xm), scal((Real)2, x0)));
          const Real sp = nrm(d1), cn2 = nrm(cross(d1, d2));
          if ((double)sp > 1e-30) { const Real k = cn2/(sp*sp*sp); if ((double)k > (double)km) km = k; }
        }
        return (double)km > 1e-30 ? (Real)1/km : (Real)1e30;
      };
      rho_cA = corner_rho(wLA, wCA); rho_cB = corner_rho(wCB, wLB);
      kmax = std::max(kmax, std::max((Real)1/std::max((Real)1e-30,rho_cA), (Real)1/std::max((Real)1e-30,rho_cB)));
      std::cout << "    [corner rho/r] A=" << (double)(rho_cA/r_arm) << " body=" << (double)(rho_body/r_arm)
                << " B=" << (double)(rho_cB/r_arm) << "\n";
    }
    const Real rho = (double)kmax > 1e-30 ? (Real)1/kmax : (Real)1e30;
    std::cout << "  [downsample] " << (dump_tag?dump_tag:"") << (is_cap?" (cap)":"")
              << (seam_corner?" seamcorner":"") << " ctrl=" << ctrl.size()
              << " (h=" << (double)hstep << " swing A/B=" << (double)(thA*180/M_PI) << "/" << (double)(thB*180/M_PI) << "deg)"
              << " rho_min/r=" << (double)(rho/r_arm)
              << ((double)rho < (double)r_arm ? "  FOLD" : "") << "\n";
  }

  // ---- rotation-minimizing frame (double reflection), tabulated; closure-twist onto endE1 at t=1. ----
  const Integer Ng = std::max<Integer>((Integer)200, (Integer)20*n);
  auto tang = [&](Integer i) -> Vec3<Real> {
    const Real h = (Real)1/Ng;
    if (i <= 0)  return unit(sub(curve(h), curve((Real)0)));
    if (i >= Ng) return unit(sub(curve((Real)1), curve((Real)1-h)));
    return unit(sub(curve((i+1)*h), curve((i-1)*h)));
  };
  if (out_termC) *out_termC = curve((Real)1);
  if (out_termU) *out_termU = tang(Ng);
  std::vector<Vec3<Real>> Rtab((size_t)Ng+1);
  Vec3<Real> ti = tang(0);
  Vec3<Real> rr = unit(sub(seamA.e1, scal(dot(seamA.e1, ti), ti)));
  if ((double)nrm(rr) < 1e-9) { Vec3<Real> a{1,0,0}; if (std::fabs((double)ti[0])>0.9) a = Vec3<Real>{0,1,0}; rr = unit(sub(a, scal(dot(a,ti), ti))); }
  Rtab[0] = rr;
  for (Integer i = 0; i < Ng; i++) {
    const Vec3<Real> xi = curve(i/(Real)Ng), xi1 = curve((i+1)/(Real)Ng);
    const Vec3<Real> ti1 = tang(i+1);
    const Vec3<Real> v1 = sub(xi1, xi); const Real c1 = dot(v1, v1);
    const Vec3<Real> rL = (double)c1>0 ? sub(rr, scal(2*dot(v1,rr)/c1, v1)) : rr;
    const Vec3<Real> tL = (double)c1>0 ? sub(ti, scal(2*dot(v1,ti)/c1, v1)) : ti;
    const Vec3<Real> v2 = sub(ti1, tL); const Real c2d = dot(v2, v2);
    Vec3<Real> r1 = (double)c2d>0 ? sub(rL, scal(2*dot(v2,rL)/c2d, v2)) : rL;
    r1 = unit(r1); Rtab[(size_t)i+1] = r1; rr = r1; ti = ti1;
  }
  // Uniform closure twist so the far ring matches endE1 (neighbour/cap frame); identical to the blend path.
  Real phi = 0;
  if (!constant_orient && ((double)endE1[0]!=0 || (double)endE1[1]!=0 || (double)endE1[2]!=0)) {
    const Vec3<Real> tend = tang(Ng);
    const Vec3<Real> e1t = unit(sub(endE1, scal(dot(endE1, tend), tend)));
    const Vec3<Real> rend = Rtab[(size_t)Ng];
    if ((double)nrm(e1t) > 1e-9) {
      Real cph = dot(rend, e1t); cph = std::max((Real)-1, std::min((Real)1, cph));
      phi = std::acos((double)cph);
      if ((double)dot(cross(rend, e1t), tend) < 0) phi = -phi;
    }
  }
  const Vec3<Real> r0const = Rtab[0];
  auto orient_of = [&](Real t) -> Vec3<Real> {
    if (constant_orient) return r0const;
    Real g = t*Ng; if ((double)g < 0) g = 0; if (g > (Real)Ng) g = (Real)Ng;
    Integer i = (Integer)g; if (i >= Ng) i = Ng-1; const Real f = g - i;
    Vec3<Real> e1 = unit(add(scal((1-f), Rtab[(size_t)i]), scal(f, Rtab[(size_t)i+1])));
    if ((double)phi != 0) {                              // rotate e1 by t*phi about the local tangent (Rodrigues)
      const Vec3<Real> k = tang((Integer)std::lround((double)t*Ng));
      const Real th = t*phi, ct = std::cos((double)th), st = std::sin((double)th);
      e1 = add(add(scal(ct, e1), scal(st, cross(k, e1))), scal(dot(k,e1)*((Real)1-ct), k));
      e1 = unit(e1);
    }
    return e1;
  };

  // ---- panel knot vector: uniform, lead pinned to panels [0,lp) and [n-lp,n); turn-refine the spline body.
  std::vector<Real> tk; tk.reserve((size_t)n + 1);
  for (Integer k = 0; k <= n; k++) tk.push_back((Real)k / (Real)n);
  // QJ_DS_PANEL_SNAP: shift the body panel boundaries onto the C2 spline knots (no degree-10 panel straddles a
  // knot's 3rd-derivative jump). Body = tk indices [lp+cp .. n-lp-cp] (nmid panels); with K==nmid the spline has
  // exactly nmid segments, knot k at arclength bcl.s[k], mapped to t = wCA + (bcl.s[k]/bcl.total)*(wCB-wCA).
  // Leads/corners keep their uniform boundaries. SHIFT, not split: panel count/size unchanged (runs BEFORE
  // turn-refine, which then subdivides WITHIN a knot-aligned segment -- no interior knot, no straddle).
  if (panel_snap && seam_corner && (Integer)bcl.s.size() == nmid + 1 && (double)bcl.total > 1e-30) {
    const Integer b0 = lp + cp;                                // tk[b0]==wCA ; tk[b0+nmid]==wCB
    for (Integer k = 1; k < nmid; k++)
      tk[(size_t)(b0 + k)] = wCA + (bcl.s[(size_t)k]/bcl.total) * (wCB - wCA);
  }
  // Curvature-adaptive turn refinement (same lever as the blend path, default 6 deg / cap 24): split any
  // panel whose tangent bends more than the cap into ceil(bend/cap) sub-panels (same curve, same radius).
  const char* trc = std::getenv("QJ_ARM_TURN_REFINE") ? std::getenv("QJ_ARM_TURN_REFINE") : "6";
  if (trc && std::atof(trc) <= 0.0) trc = nullptr;
  const Integer turn_refine_max = std::getenv("QJ_ARM_TURN_REFINE_MAX") ? std::max<Integer>(1, (Integer)std::atoi(std::getenv("QJ_ARM_TURN_REFINE_MAX"))) : (Integer)24;
  if (trc) {
    const double cap = std::max(1.0, std::atof(trc));
    auto tang_at = [&](Real t, int side) -> Vec3<Real> {
      const Real d = (Real)1e-3;
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
      for (Integer sgi = 1; sgi < k; sgi++) tk3.push_back((Real)(ta + (tb - ta) * (Real)sgi / (Real)k));
    }
    tk3.push_back(tk.back());
    tk.swap(tk3);
  }
  const Integer npanel = (Integer)tk.size() - 1;

  // ---- Chebyshev interpolation error of the centerline (QJ_CORNER diag). CSBQ represents each panel's
  // centerline by its values at the `cheb` Chebyshev nodes; the degree-(cheb-1) interpolant of those nodes
  // is what the solver actually sees. The error = max_t |L_panel(t) - curve(t)| over each panel, i.e. how
  // far the panel polynomial departs from the true geometry. A seam corner that is smooth within each panel
  // drives this toward zero; a knot/curvature step inside a panel (the M8 straddle) spikes it. Report the
  // overall max/r AND the max over the two SEAM panel-blocks (lead+corner on each end) separately, since the
  // seam neighbourhood is what sets the junction cross-list free-edge. ----
  if (diag) {
    const Vector<Real>& cnd = SlenderElemList<Real>::CenterlineNodes(cheb);
    auto lagrange = [&](const std::vector<Vec3<Real>>& fj, Real xi) -> Vec3<Real> {   // nodes at cnd[], sample xi in [0,1]
      Vec3<Real> acc{(Real)0,(Real)0,(Real)0};
      for (Long j = 0; j < cheb; j++) {
        Real w = 1;
        for (Long m = 0; m < cheb; m++) if (m != j) w *= (double)(xi - cnd[m]) / (double)(cnd[j] - cnd[m]);
        acc = add(acc, scal(w, fj[(size_t)j]));
      }
      return acc;
    };
    Real emax = 0, eseam = 0; const Integer MS = 50;
    for (Integer p = 0; p < npanel; p++) {
      const Real ta = tk[(size_t)p], tb = tk[(size_t)p+1];
      std::vector<Vec3<Real>> fj((size_t)cheb);
      for (Long j = 0; j < cheb; j++) fj[(size_t)j] = curve(ta + cnd[j]*(tb-ta));
      Real ep = 0;
      for (Integer sgi = 0; sgi <= MS; sgi++) { const Real xi = (Real)sgi/MS;
        const Real e = nrm(sub(lagrange(fj, xi), curve(ta + xi*(tb-ta)))); if ((double)e > (double)ep) ep = e; }
      if ((double)ep > (double)emax) emax = ep;
      const bool seam_block = seam_corner ? ((double)tb <= (double)wCA + 1e-9 || (double)ta >= (double)wCB - 1e-9)
                                          : ((double)tb <= (double)wLA + 1e-9 || (double)ta >= (double)wLB - 1e-9);
      if (seam_block && (double)ep > (double)eseam) eseam = ep;
    }
    std::cout << "    [cheb interp err/r] max=" << (double)(emax/r_arm) << " seam=" << (double)(eseam/r_arm)
              << " (npanel=" << npanel << ")\n";
  }

  // ---- optional per-stage dump (QJ_DUMP_STAGES) so the verification artifact can draw this path too. ----
  if (const char* dsf = std::getenv("QJ_DUMP_STAGES"); dsf && dsf[0] && dump_tag) {
    std::ofstream dsout(dsf, std::ios::app);
    if (dsout) {
      dsout << std::setprecision(17);
      dsout << "# ARM " << dump_tag << " is_cap=" << (is_cap ? 1 : 0) << " downsample=1\n";
      dsout << "SEAM A0 " << (double)A0[0] << " " << (double)A0[1] << " " << (double)A0[2]
            << " uA " << (double)uA[0] << " " << (double)uA[1] << " " << (double)uA[2]
            << " endC " << (double)endC[0] << " " << (double)endC[1] << " " << (double)endC[2]
            << " uB " << (double)uB[0] << " " << (double)uB[1] << " " << (double)uB[2]
            << " rA " << (double)rA << " rB " << (double)rB << "\n";
      auto dump_pts = [&](const char* nm, const std::vector<Vec3<Real>>& P) {
        dsout << "STAGE " << nm << " n=" << P.size() << "\n";
        for (const auto& p : P) dsout << (double)p[0] << " " << (double)p[1] << " " << (double)p[2] << "\n";
      };
      dump_pts("raw", cl_in);
      dump_pts("downsampled", ctrl);
      const Integer Nf = std::max<Integer>((Integer)4*npanel, (Integer)64);
      dsout << "STAGE final n=" << (Nf + 1) << "\n";
      for (Integer k = 0; k <= Nf; k++) { const Real t=(Real)k/Nf; const Vec3<Real> c=curve(t);
        dsout << (double)c[0] << " " << (double)c[1] << " " << (double)c[2] << " r=" << (double)radius_at(t) << "\n"; }
      dsout << "# END ARM " << dump_tag << "\n";
    }
  }

  // ---- emit CSBQ panels: Chebyshev-per-panel sampling of curve/radius/orient (never the panel endpoints). ----
  const Vector<Real>& cn = SlenderElemList<Real>::CenterlineNodes(cheb);
  const Long seam_c0 = coord.Dim();                             // node offset for the inter-panel seam check below
  for (Integer p = 0; p < npanel; p++) {
    const Real ta = tk[(size_t)p], tb = tk[(size_t)p + 1];
    elem_order.PushBack(cheb); forder.PushBack(fourier);
    for (Long j = 0; j < cheb; j++) {
      const Real t = ta + cn[j] * (tb - ta);
      const Vec3<Real> Pc = curve(t); const Vec3<Real> o = orient_of(t); const Real rv = radius_at(t);
      coord.PushBack(Pc[0]); coord.PushBack(Pc[1]); coord.PushBack(Pc[2]);
      radius.PushBack(rv);
      orient.PushBack(o[0]); orient.PushBack(o[1]); orient.PushBack(o[2]);
    }
  }
  // Inter-panel seam smoothness from the Chebyshev nodes CSBQ actually sees (QJ_ARM_SEAM_CHECK, default on):
  // dpos (C0), dtang (C1), dkappa (C2). With panel_snap aligning panels to the C2 knots these should fall to
  // the Chebyshev floor (dpos ~1e-10, dtang ~1e-6); a straddle keeps them at ~1e-4. Pure diagnostic.
  report_arm_seam_smoothness<Real>(coord, seam_c0, npanel, cheb, dump_tag);
}

// ===================================================================================================
// POLY-FIT arm centerline (QJ_ARM_POLYFIT) -- see docs/polyfit_centerline.md and python/polyfit_centerline.py.
// Copy of append_downsampled_fiber (same signature; smoothing, mouths, straight lead, quintic-Hermite corner,
// RMF orient, radius taper, seam-check, caps IDENTICAL) with ONLY these stages replaced:
//   window  -> pf::adaptive_body_window (fold fix: push an end's room deeper while its fitted rho/r < 1.6),
//   body    -> pf::SegPolyFit (turn+length-aware wide-knot piecewise degree-QJ_PF_DEG Chebyshev, C^QJ_PF_CONT),
//   panels  -> boundaries SNAPPED ONTO the few segment knots, n_axial distributed ~ segment arclength.
// Env knobs: QJ_PF_DEG[9] QJ_PF_CONT[4] QJ_PF_LSEG[45] QJ_PF_TURN_SEG[35] QJ_PF_SINGLE[0].
// ===================================================================================================
template <class Real> void append_polyfit_fiber(const ArmSeam<Real>& seamA, const Vec3<Real>& endC,
    const Vec3<Real>& endTang, const std::vector<Vec3<Real>>& cl_in, Real rA, Real rB,
    Integer n_axial, Integer lead_panels, Long cheb, Long fourier,
    Vector<Long>& elem_order, Vector<Long>& forder, Vector<Real>& coord, Vector<Real>& radius, Vector<Real>& orient,
    bool constant_orient, Vec3<Real> endE1, bool is_cap,
    Vec3<Real>* out_termC, Vec3<Real>* out_termU, const char* dump_tag, Real* out_termR = nullptr) {
  using namespace gnet;
  const bool diag = (std::getenv("QJ_CORNER") != nullptr);
  const Vec3<Real> A0 = seamA.C;
  const Vec3<Real> uA = unit(seamA.u);
  const Vec3<Real> fwdB = unit(endTang);                 // forward travel direction at the far end
  const Vec3<Real> uB = scal((Real)-1, fwdB);            // junction B's arm-exit axis (= -endTang)
  const Vec3<Real> chord = sub(endC, A0); const Real L = nrm(chord);
  const Real r_arm = std::max(rA, rB);

  // Degenerate arm => exact straight tapered tube (handled by the dedicated helper).
  if ((double)L < 1e-12 || cl_in.size() < 2) {
    append_straight_fiber<Real>(seamA, endC, rA, rB, std::max<Integer>(6, n_axial), cheb, fourier,
                                elem_order, forder, coord, radius, orient);
    if (out_termC) *out_termC = endC;
    if (out_termU) *out_termU = uA;
    return;
  }

  // ---- SMOOTH the FULL vmtk centerline up front (simple moving average, global ends fixed). The vmtk
  // skeleton is already smooth; only its *discrete* curvature is noisy, and a few averaging passes open the
  // tightest bends (here raw rho 0.73r -> 2.1r) without moving the path at figure scale. Doing it on the WHOLE
  // curve -- not on the later extracted window -- is essential: a tight bend often sits right next to a mouth,
  // and a window-local average that pins its endpoints would leave exactly that bend unsmoothed. ----
  const Integer sm_passes = std::getenv("QJ_DS_SMOOTH") ? std::max<Integer>(0,(Integer)std::atoi(std::getenv("QJ_DS_SMOOTH"))) : (Integer)8;
  std::vector<Vec3<Real>> clS = cl_in;
  for (Integer it = 0; it < sm_passes && clS.size() >= 3; it++) {
    std::vector<Vec3<Real>> sm = clS;
    for (size_t k = 1; k+1 < clS.size(); k++)
      sm[k] = scal((Real)0.25, add(add(clS[k-1], clS[k+1]), scal((Real)2, clS[k])));
    clS.swap(sm);
  }
  // ---- Taubin lambda|mu low-pass (DEFAULT ON, QJ_DS_LOWPASS passes, default 64; QJ_DS_LOWPASS=0 opts out =>
  // byte-identical to the moving-average-only path). Applied to INTERIOR arms only (NOT caps -- see below), on
  // the full SMOOTHED centerline BEFORE the window extraction / uniform downsample, so the sparse knots are
  // sampled from the denoised curve. The 3-point average above is a pure Laplacian SHRINK step (equivalent to
  // one Taubin lambda=0.5 half-pass with no anti-shrink): a weak low-pass that, on a tortuous interior arm,
  // leaves residual dkappa/ds curvature noise the toroidal special quadrature cannot resolve -> a ~0.5 DL
  // free-edge MID-ARM (see [[network-interior-arm-lowpass]]; measured on interior j120->133: DL arm 0.545 ->
  // 0.00424, 128x, with 64 Taubin passes). Taubin interleaves an anti-shrink mu step (mu<-lambda<0) so it
  // removes the high-frequency curvature band WITHOUT shrinking the bend (tortuosity preserved) -- the
  // production CL_LOWPASS fix (lambda 0.5 / mu -0.53). Global ends held fixed each half-step => mouth rings
  // untouched, watertightness-neutral.
  //   !is_cap: CAP (leaf) arms are EXCLUDED. A cap's far end is a free tip with a dome seated on the tube's
  // terminal; caps are already clean without the low-pass (DL ~0.006), and applying Taubin reshapes the near-
  // tip body curvature (rho/r 1.77->1.41 under the dome) and drives a ~0.9 DL free-edge there (measured on cap
  // j286->289: 0.006 -> 0.915). Re-deriving the cap axis from the smoothed centerline does NOT help (Taubin
  // barely moves the near-fixed-endpoint tangent); the only noisy arms that NEED the low-pass are the long
  // tortuous interior ones. So the low-pass is interior-only. (QJ_CAP_LOWPASS, which opted caps back in after a
  // tip trim, was reverted 2026-10-15: at full-net scale Taubin un-pins the cap's JUNCTION-SEAM endpoint and
  // spawns worse cross-list blowups -- the dominant cap error is the seam, not the tip. See s11 memory.) ----
  const Integer lp_passes = std::getenv("QJ_DS_LOWPASS") ? std::max<Integer>(0,(Integer)std::atoi(std::getenv("QJ_DS_LOWPASS"))) : (Integer)64;
  if (lp_passes > 0 && !is_cap && clS.size() >= 3) {
    const Real lp_lam = std::getenv("QJ_DS_LOWPASS_LAMBDA") ? (Real)std::atof(std::getenv("QJ_DS_LOWPASS_LAMBDA")) : (Real)0.5;
    const Real lp_mu  = std::getenv("QJ_DS_LOWPASS_MU")     ? (Real)std::atof(std::getenv("QJ_DS_LOWPASS_MU"))     : (Real)-0.53;
    auto lap_step = [&](Real w){                              // one uniform-weight Laplacian smoothing half-step, ends fixed
      std::vector<Vec3<Real>> sm = clS;
      for (size_t k = 1; k+1 < clS.size(); k++) {
        const Vec3<Real> lapk = sub(scal((Real)0.5, add(clS[k-1], clS[k+1])), clS[k]);
        sm[k] = add(clS[k], scal(w, lapk));
      }
      clS.swap(sm);
    };
    for (Integer it = 0; it < lp_passes; it++) { lap_step(lp_lam); lap_step(lp_mu); }
  }
  std::vector<Real> cs(clS.size(), (Real)0);
  for (size_t k = 1; k < clS.size(); k++) cs[k] = cs[k-1] + nrm(sub(clS[k], clS[k-1]));
  const Real total = cs.back();

  // ---- SEAM-CORNER mode (DEFAULT; the production seam-settle recipe). The
  // clamp_mode below forced the body-spline END knots QA/QB exactly onto the junction axis AND clamped the
  // end tangent -- a double constraint that bent the ~0.1r-off-axis vessel back onto the axis over the one
  // segment next to the clamp, the residual ~12x gap to the production seamsettle_tr6 baseline. Production
  // keeps the body on the TRUE centerline and bridges each mouth with ONE gentle quintic Hermite "seam
  // corner" that leaves the on-axis mouth ring coaxial (tangent = seam axis, zero endpoint acceleration) and
  // ARRIVES on the vessel with the vessel's own tangent -- re-seating the terminal ring flush under the seam
  // without straightening the vessel. Reproduced here: 1-panel straight coaxial lead (watertight mouth ring)
  // -> cp-panel quintic Hermite corner QA->P_hA -> downsampled C2 body spline over the real (un-forced)
  // window knots -> mirror corner -> lead. Opt out with QJ_DS_NO_SEAM_CORNER (=> old clamp/natural bridge).
  const bool seam_corner = true;                          // polyfit is always seam-corner (lead|quintic corner|body|corner|lead)
  // ---- coaxial lead length: one scale-invariant stub, capped to a fraction of the arm. Seam-corner and the
  // legacy CLAMPED mode both use a SHORTER half-radius stub: the stub end QA/QB carries the on-axis mouth
  // ring, and the quintic corner (resp. clamped spline) curves onto the off-axis vessel over its own run; a
  // long stub would instead cram the turn into the one segment next to the stub and fold. The glued stub is
  // straight+coaxial so the mouth ring conforms either way. ----
  const bool clamp_mode = false;
  Real Llead = (clamp_mode || seam_corner) ? (Real)0.5 * std::max(rA, rB) : std::max(rA, rB);
  if (const char* e = std::getenv("QJ_DS_LEAD")) Llead = std::max((Real)1e-9, (Real)std::atof(e));
  Llead = std::min(Llead, (Real)0.35 * total);           // leave room for a non-degenerate interior
  const Vec3<Real> QA = add(A0,   scal(Llead, uA));
  const Vec3<Real> QB = add(endC, scal(Llead, uB));      // QB = endC - Llead*endTang (so endC = QB + Llead*fwdB)

  // ---- point on the SMOOTHED centerline at arclength s (linear-interp) ----
  auto cl_at = [&](Real s) -> Vec3<Real> {
    if ((double)s <= 0) return clS.front();
    if ((double)s >= (double)total) return clS.back();
    size_t i = 0; while (i+1 < cs.size() && (double)cs[i+1] < (double)s) i++;
    const Real h = std::max((Real)1e-30, cs[i+1]-cs[i]); const Real w = (s-cs[i])/h;
    return add(scal((Real)1-w, clS[i]), scal(w, clS[i+1]));
  };

  // ---- POLYFIT stages 3-5: adaptive body window + segmented constrained fit on the SMOOTHED centerline clS ----
  auto envd = [](const char* k, double d){ const char* e = std::getenv(k); return e ? std::atof(e) : d; };
  const int pf_deg  = (int)envd("QJ_PF_DEG", 9),  pf_cont = (int)envd("QJ_PF_CONT", 4);
  const double pf_Lseg = envd("QJ_PF_LSEG", 45.0), pf_turn = envd("QJ_PF_TURN_SEG", 35.0);
  const bool pf_single = envd("QJ_PF_SINGLE", 0.0) != 0.0;
  // QJ_PF_END_KAPPA0 (DEFAULT ON): pin the body poly's end curvature to zero so the corner->body join is C2
  // (matches DOWNSAMPLE's natural-C2 ends; the quintic corner is already zero-accel there). Set =0 to opt out.
  const bool pf_endk0 = envd("QJ_PF_END_KAPPA0", 1.0) != 0.0;
  // LOCAL curvature relax (QJ_PF_RELAX_RHO, default 0 = OFF). When >0, any arm whose FITTED body min rho/r falls
  // below this target gets EXTRA moving-average passes applied to its smoothed centerline clS (ends fixed, so the
  // mouth rings and watertightness are untouched -- the same safe shrink step already run `sm_passes` times), and
  // the body window + segmented fit are redone, until the fitted body rho/r clears the target or QJ_PF_RELAX_MAX
  // extra passes are spent. This OPENS the tightest bends toward the CSBQ clean toroidal threshold (~1.5): the DL
  // const-density floor on the n298 net is set by the only two arms below 1.5 (caps j141->155 rho/r~1.27,
  // j152->170 ~1.24), and that floor is a near-pinch QUADRATURE limit, not a centerline-representation or
  // continuity deficit (raising Fourier made it worse). Arms already above the target fit on the first pass and
  // get ZERO extra passes => byte-identical. A cap's far end is a free tip, so opening its bend moves the path
  // <0.02 r (well inside the ~0.036 r baseline fit noise). The corner guard (pf_extra) re-runs each refit, so the
  // seam corner stays >= pf_ctgt. Tunables: QJ_PF_RELAX_PASSES (batch per iter, default 4), QJ_PF_RELAX_MAX (cap).
  const double pf_relax_rho = envd("QJ_PF_RELAX_RHO", 0.0);
  const Integer pf_relax_batch = std::max<Integer>(1, (Integer)envd("QJ_PF_RELAX_PASSES", 4.0));
  const Integer pf_relax_max = std::max<Integer>(0, (Integer)envd("QJ_PF_RELAX_MAX", 128.0));
  // ---- CAP radius-shrink (QJ_PF_CAP_SHRINK=<target rho/r>, default 0=off; CAP arms only). Where the relax
  // above still leaves a near-pinch cap below the clean toroidal threshold, instead of smoothing the path
  // further, SHRINK the cap-END radius rB so rho/r_local clears the target at the tightest body point, and
  // keep tapering DOWN to that smaller rB -- no flare back to the vessel radius. The owner MOUTH rA is
  // unchanged (== junction seam => watertight); only the free tip shrinks. The hemisphere cap is then sized
  // to the shrunk terminal radius (reported via out_termR), so the arm tip ring == cap equator (watertight).
  // Floored at QJ_PF_CAP_SHRINK_FLOOR*rA (default 0.3, "never a needle"). Fires only when is_cap && the fitted
  // body rho/r < target => clean caps and all interior arms are byte-identical.
  const double pf_cap_shrink = envd("QJ_PF_CAP_SHRINK", 0.0);
  const Real   pf_cap_floor  = (Real)envd("QJ_PF_CAP_SHRINK_FLOOR", 0.3);
  auto cl_tan = [&](Real s)->Vec3<Real>{ const Real hh=(Real)0.03*total+(Real)1e-9; return unit(sub(cl_at(std::min(total,s+hh)), cl_at(std::max((Real)0,s-hh)))); };
  (void)cl_tan;
  std::vector<quad_junctions::pf::V3> smd(clS.size());
  auto rebuild_smd = [&]{ smd.resize(clS.size()); for (size_t k = 0; k < clS.size(); k++) smd[k] = {(double)clS[k][0], (double)clS[k][1], (double)clS[k][2]}; };
  rebuild_smd();
  quad_junctions::pf::SegPolyFit pfit; quad_junctions::pf::PfWindow pwin;
  // Window room is measured by AXIAL DEPTH past each mouth (projection on uA / uB), like the DOWNSAMPLE path's
  // sdepth_fwd/bwd -- NOT arclength from clS[0]: the vmtk centerline starts ~2-3.5 r BEHIND the seam centre, so
  // arclength Llead+2r from clS[0] lands behind the lead stub QA and the quintic corner hooks back (rho/r ~ 1e-4).
  const std::function<int(double)> pf_idxA = [&](double depth)->int{
    for (size_t k = 0; k < clS.size(); k++) if ((double)dot(sub(clS[k],A0),uA) >= depth) return (int)k;
    return (int)clS.size() - 3; };
  const std::function<int(double)> pf_idxB = [&](double depth)->int{
    for (size_t k = clS.size(); k-- > 0;) if ((double)dot(sub(clS[k],endC),uB) >= depth) return (int)k;
    return 2; };
  // Corner guard (QJ_PF_CORNER_TGT, default 1.6 r; 0 disables): the python reference only tests the BODY rho/r, but with a
  // real seam the on-axis stub must also swing onto the body start through ONE quintic Hermite corner; a body that starts
  // too close to the mouth makes that corner fold (rho/r<1). Same push-deeper reaction as the body test (analogue of
  // DOWNSAMPLE's QJ_DS_CORNER_TGT loop).
  const double pf_ctgt = envd("QJ_PF_CORNER_TGT", 1.6);
  const Real pf_ckr = std::getenv("QJ_DS_CORNER_KR") ? (Real)std::atof(std::getenv("QJ_DS_CORNER_KR")) : (Real)0.8;
  auto pf_h5min = [&](const Vec3<Real>& P0,const Vec3<Real>& T0,Real h0,const Vec3<Real>& P1,const Vec3<Real>& T1,Real h1)->Real{
    const Integer M=100; const Real hh=(Real)1/(M+2); Real km=0;
    for (Integer i=1;i<M;i++){ const Real s=(Real)i/M;
      const Vec3<Real> xm=HybridAssembly<Real>::hermite5(P0,T0,h0,P1,T1,h1,s-hh);
      const Vec3<Real> x0=HybridAssembly<Real>::hermite5(P0,T0,h0,P1,T1,h1,s);
      const Vec3<Real> xp=HybridAssembly<Real>::hermite5(P0,T0,h0,P1,T1,h1,s+hh);
      const Vec3<Real> d1=scal((Real)1/(2*hh),sub(xp,xm)); const Vec3<Real> d2=scal((Real)1/(hh*hh),sub(add(xp,xm),scal((Real)2,x0)));
      const Real sp=nrm(d1),cn2=nrm(cross(d1,d2)); if ((double)sp>1e-30){ const Real k=cn2/(sp*sp*sp); if ((double)k>(double)km) km=k; } }
    return (double)km>1e-30 ? (Real)1/km : (Real)1e30; };
  const std::function<std::pair<bool,bool>(const quad_junctions::pf::SegPolyFit&, const quad_junctions::pf::PfWindow&)> pf_extra =
    [&](const quad_junctions::pf::SegPolyFit& F, const quad_junctions::pf::PfWindow&)->std::pair<bool,bool>{
      if (pf_ctgt <= 0) return {true,true};
      auto v3 = [](const quad_junctions::pf::V3& a){ return Vec3<Real>{(Real)a[0],(Real)a[1],(Real)a[2]}; };
      const Vec3<Real> PA = v3(F.eval(0.0)), PB = v3(F.eval(1.0));
      const Vec3<Real> tA = unit(v3(F.deriv(0.0,1))), tB = unit(v3(F.deriv(1.0,1)));
      const Vec3<Real> chA = sub(PA, QA), chB = sub(QB, PB);
      const Real LA = nrm(chA), LB = nrm(chB);
      const Real rA_ = pf_h5min(QA, uA, pf_ckr*LA, PA, tA, pf_ckr*LA);
      const Real rB_ = pf_h5min(PB, tB, pf_ckr*LB, QB, fwdB, pf_ckr*LB);
      return {(double)rA_ >= pf_ctgt*(double)r_arm, (double)rB_ >= pf_ctgt*(double)r_arm}; };
  auto do_fit = [&]() -> bool {                                  // base window, then shrink the room before giving up
    bool ok = quad_junctions::pf::adaptive_body_window(smd, (double)r_arm, (double)Llead, pfit, pwin,
                  2.0, 2.0, 12.0, 1.6, 0.12, pf_deg, pf_cont, pf_Lseg, pf_turn, pf_single, pf_endk0, pf_idxA, pf_idxB, pf_extra);
    if (!ok) for (double kb : {1.0, 0.5, 0.0}) {
      ok = quad_junctions::pf::adaptive_body_window(smd, (double)r_arm, (double)Llead, pfit, pwin,
                  kb, 2.0, 12.0, 1.6, 0.12, pf_deg, pf_cont, pf_Lseg, pf_turn, pf_single, pf_endk0, pf_idxA, pf_idxB, pf_extra);
      if (ok) break; }
    return ok; };
  auto body_min_rho_r = [&]() -> double { const auto rr = pfit.rho_over_r((double)r_arm, 1200);
    return rr.empty() ? 1e30 : *std::min_element(rr.begin(), rr.end()); };
  bool pf_ok = do_fit();
  // LOCAL curvature relax: open the body bend with extra ends-fixed moving-average passes until the fitted body
  // min rho/r clears pf_relax_rho (see the knob comment above). Fires only on arms below target; reverts to the
  // last good fit if a refit ever fails (a smoother curve almost never does).
  if (pf_ok && pf_relax_rho > 0 && (double)body_min_rho_r() < pf_relax_rho) {
    Integer nrelax = 0;
    while (nrelax < pf_relax_max && (double)body_min_rho_r() < pf_relax_rho) {
      std::vector<Vec3<Real>> clS_prev = clS;
      for (Integer it = 0; it < pf_relax_batch && clS.size() >= 3; it++) {
        std::vector<Vec3<Real>> sm = clS;                        // one 0.25*(prev+next+2*cur) half-step, ends fixed
        for (size_t k = 1; k+1 < clS.size(); k++)
          sm[k] = scal((Real)0.25, add(add(clS[k-1], clS[k+1]), scal((Real)2, clS[k])));
        clS.swap(sm);
      }
      rebuild_smd();
      if (!do_fit()) { clS.swap(clS_prev); rebuild_smd(); do_fit(); break; }   // revert to last good, stop
      nrelax += pf_relax_batch;
    }
    if (diag) std::cout << "    [pf-relax] " << (dump_tag?dump_tag:"") << " +" << (long)nrelax
                        << " mov-avg passes -> body rho/r=" << body_min_rho_r() << " (target " << pf_relax_rho << ")\n";
  }
  // CAP radius-shrink (see knob comment): taper the free cap-end radius DOWN so the tightest fitted-body bend
  // clears target rho/r_local; the cap dome is sized to this terminal radius via out_termR below. radius_at()
  // captures rB by reference, so modifying it here (before panelization) retargets the linear taper's far end.
  if (pf_ok && is_cap && pf_cap_shrink > 0) {
    const double rr = body_min_rho_r();                        // min rho / r_arm over the fitted body
    if (rr < pf_cap_shrink && rr < 1e29) {
      const Real min_rho = (Real)rr * r_arm;                   // absolute min curvature radius of the body
      const Real rB_new  = std::max<Real>(pf_cap_floor * rA, (Real)(min_rho / pf_cap_shrink));
      if ((double)rB_new < (double)rB) {
        if (diag) std::cout << "    [pf-capshrink] " << (dump_tag?dump_tag:"") << " rB " << (double)rB
                            << " -> " << (double)rB_new << "  (body rho/r " << rr << " -> tip rho/r "
                            << (double)(min_rho/rB_new) << ", target " << pf_cap_shrink << ")\n";
        rB = rB_new;
      }
    }
  }
  if (!pf_ok) {   // still no valid window: fall back to the DOWNSAMPLE builder for this (very short) arm
    if (diag) std::cout << "  [polyfit] " << (dump_tag?dump_tag:"") << " window/fit failed -> DOWNSAMPLE fallback\n";
    append_downsampled_fiber<Real>(seamA, endC, endTang, cl_in, rA, rB, n_axial, lead_panels, cheb, fourier,
                                   elem_order, forder, coord, radius, orient, constant_orient, endE1, is_cap,
                                   out_termC, out_termU, dump_tag);
    return;
  }
  if (out_termR) *out_termR = rB;                                // terminal (cap-end) tube radius = rB (possibly cap-shrunk)
  const std::vector<double> pf_f = pfit.f;                       // segment knot fractions (tau in [0,1])
  const Integer pf_S = (Integer)pfit.S;
  auto pf_eval = [&](Real tau) -> Vec3<Real> { const auto v = pfit.eval((double)tau); return Vec3<Real>{(Real)v[0],(Real)v[1],(Real)v[2]}; };
  const Integer nmid_pf = std::max<Integer>(std::max<Integer>(1, n_axial), pf_S);   // >=1 panel per segment
  std::vector<Vec3<Real>> ctrl;                                  // body points (for dump / PhA,PhB)
  for (int k = pwin.iA; k <= pwin.iB; k++) ctrl.push_back(clS[(size_t)k]);
  // Body end tangents for the corners: the fit's own analytic end tangent (corner arrives along the body).
  const Vec3<Real> tanA = Vec3<Real>{(Real)pfit.deriv(0.0,1)[0],(Real)pfit.deriv(0.0,1)[1],(Real)pfit.deriv(0.0,1)[2]};
  const Vec3<Real> tanB = Vec3<Real>{(Real)pfit.deriv(1.0,1)[0],(Real)pfit.deriv(1.0,1)[1],(Real)pfit.deriv(1.0,1)[2]};
  const Vec3<Real> teA = unit(tanA), teB = unit(tanB);
  if (diag) std::cout << "    [pf-window] " << (dump_tag?dump_tag:"") << " cl0-behind-A0/r=" << (double)(dot(sub(clS.front(),A0),uA)/r_arm)
      << " PhA axial depth/r=" << (double)(dot(sub(ctrl.front(),A0),uA)/r_arm) << " (Llead/r=" << (double)(Llead/r_arm)
      << ")  PhB axial depth/r=" << (double)(dot(sub(ctrl.back(),endC),uB)/r_arm) << " tanA.uA=" << (double)dot(teA,uA) << " tanB.fwdB=" << (double)dot(teB,fwdB) << "\n";
  const Real thA = 0, thB = 0; (void)thA; (void)thB;

  // ---- panelization: one lead panel each side (per the assembler), plus (seam_corner) one quintic corner
  // panel-block each side, the rest spread over the spline body.
  const Integer lp = std::max<Integer>(1, lead_panels);
  const Integer cp = seam_corner ? (std::getenv("QJ_DS_CORNER_PANELS")
                     ? std::max<Integer>(1, (Integer)std::atoi(std::getenv("QJ_DS_CORNER_PANELS"))) : lp) : 0;
  const Integer nmid = nmid_pf;
  const Integer n = nmid + 2*lp + 2*cp;                  // total panels before turn-refine
  // panel-fraction boundaries (seam_corner): lead A [0,wLA] | corner A [wLA,wCA] | body [wCA,wCB] | corner B
  // [wCB,wLB] | lead B [wLB,1]. For clamp/natural (cp=0) wCA=wLA and wCB=wLB collapse to the 3-piece layout.
  const Real wLA = (Real)lp/(Real)n, wCA = (Real)(lp+cp)/(Real)n;
  const Real wCB = (Real)(n-lp-cp)/(Real)n, wLB = (Real)(n-lp)/(Real)n;
  // Quintic Hermite seam-corner geometry: unit end tangents (axis uA/fwdB at the stub, vessel tanA/tanB at the
  // body) scaled by a PER-END Ferguson handle, zero endpoint acceleration (same hermite5 the production
  // seam-settle uses). P_hA/P_hB are the true (off-axis) body-window ends. The handle is
  // h = corner_kr*|chord|*clamp(dot(tangent, chordhat), floor, cap). The Ferguson projection (floor<1) SHRINKS
  // a handle whose tangent points away from the chord -- the production QJ_ARM_FERGUSON overshoot fix. But the
  // seam corner's failure is the OPPOSITE: it joins an on-axis stub to an off-axis vessel end and needs a LONG
  // handle to swing gently, so shrinking the handle tightens the fold. Hence Ferguson is OFF by default here
  // (floor=cap=1 => h=corner_kr*|chord|) and the handle is set generously (corner_kr=0.8); set QJ_DS_CORNER_FLOOR<1
  // to re-enable the projection. corner_kr sweet spot is ~0.8-1.0 (0.6 whips, >1.2 overshoots).
  const Real corner_kr = std::getenv("QJ_DS_CORNER_KR") ? (Real)std::atof(std::getenv("QJ_DS_CORNER_KR")) : (Real)0.8;
  const Real cfloor = std::getenv("QJ_DS_CORNER_FLOOR") ? (Real)std::atof(std::getenv("QJ_DS_CORNER_FLOOR")) : (Real)1.00;
  const Real ccap   = std::getenv("QJ_DS_CORNER_CAP")   ? (Real)std::atof(std::getenv("QJ_DS_CORNER_CAP"))   : (Real)1.00;
  const Vec3<Real> PhA = ctrl.front(), PhB = ctrl.back();   // true (off-axis) smoothed-centerline body-window ends
  auto ferg_h = [&](const Vec3<Real>& T, const Vec3<Real>& chord) -> Real {
    const Real Lc = nrm(chord); if ((double)Lc < 1e-30) return (Real)0;
    const Vec3<Real> ch = unit(chord);
    return corner_kr * Lc * std::min(ccap, std::max(cfloor, dot(T, ch)));
  };
  const Vec3<Real> chA = sub(PhA, QA), chB = sub(QB, PhB);
  const Real hA0 = ferg_h(uA, chA),  hA1 = ferg_h(teA, chA);      // corner A: QA(uA) -> PhA(teA)
  const Real hB0 = ferg_h(teB, chB), hB1 = ferg_h(fwdB, chB);     // corner B: PhB(teB) -> QB(fwdB)

  // Piecewise centerline c(t). seam_corner: lead -> quintic corner -> C2 body spline -> quintic corner -> lead.
  // clamp/natural: straight lead A0->QA, C2 spline QA->QB, straight lead QB->endC.
  auto curve = [&](Real t) -> Vec3<Real> {
    if (seam_corner) {
      if ((double)t <= (double)wLA) { const Real q = (double)wLA>1e-30 ? t/wLA : (Real)0; return add(A0, scal(q*Llead, uA)); }
      if ((double)t <  (double)wCA) { const Real s = (t-wLA)/(wCA-wLA);         // corner A: stub(uA) -> body(teA)
        return HybridAssembly<Real>::hermite5(QA, uA, hA0, PhA, teA, hA1, s); }
      if ((double)t <= (double)wCB) { const Real tau = (t-wCA)/(wCB-wCA); return pf_eval(tau); }
      if ((double)t <  (double)wLB) { const Real s = (t-wCB)/(wLB-wCB);         // corner B: body(teB) -> stub(fwdB)
        return HybridAssembly<Real>::hermite5(PhB, teB, hB0, QB, fwdB, hB1, s); }
      const Real q = (double)(1-wLB)>1e-30 ? (t-wLB)/((Real)1-wLB) : (Real)1; return add(QB, scal(q*Llead, fwdB));
    }
    if ((double)t <= (double)wLA) { const Real q = (double)wLA>1e-30 ? t/wLA : (Real)0; return add(A0, scal(q*Llead, uA)); }
    if ((double)t >= (double)wLB) { const Real q = (double)(1-wLB)>1e-30 ? (t-wLB)/((Real)1-wLB) : (Real)1; return add(QB, scal(q*Llead, fwdB)); }
    const Real tau = (t - wLA)/(wLB - wLA);
    return pf_eval(tau);
  };
  auto radius_at = [&](Real t) -> Real { return rA + t*(rB - rA); };      // exact seam-conforming linear taper

  // Honest fold report: min rho/r of the fitted body (SegPolyFit analytic derivs) + each quintic corner.
  if (diag) {
    const std::vector<double> rr = pfit.rho_over_r((double)r_arm, 1200);
    const Real rho_body = (Real)(*std::min_element(rr.begin(), rr.end())) * r_arm;
    auto corner_rho = [&](Real lo, Real hi) -> Real {
      const Integer MC = 200; const Real span = hi - lo; if ((double)span <= 1e-12) return (Real)1e30;
      const Real hh = span / (MC + 2); Real km = 0;
      for (Integer i = 1; i < MC; i++) {
        const Real tt = lo + span*(Real)i/MC;
        const Vec3<Real> xm = curve(tt-hh), x0 = curve(tt), xp = curve(tt+hh);
        const Vec3<Real> d1 = scal((Real)1/(2*hh), sub(xp, xm));
        const Vec3<Real> d2 = scal((Real)1/(hh*hh), sub(add(xp, xm), scal((Real)2, x0)));
        const Real sp = nrm(d1), cn2 = nrm(cross(d1, d2));
        if ((double)sp > 1e-30) { const Real k = cn2/(sp*sp*sp); if ((double)k > (double)km) km = k; }
      }
      return (double)km > 1e-30 ? (Real)1/km : (Real)1e30;
    };
    const Real rho_cA = corner_rho(wLA, wCA), rho_cB = corner_rho(wCB, wLB);
    std::cout << "    [corner rho/r] A=" << (double)(rho_cA/r_arm) << " body=" << (double)(rho_body/r_arm)
              << " B=" << (double)(rho_cB/r_arm) << "\n";
    const Real rho = std::min(rho_body, std::min(rho_cA, rho_cB));
    std::cout << "  [downsample] " << (dump_tag?dump_tag:"") << (is_cap?" (cap)":"") << " polyfit S=" << pf_S
              << " deg=" << pf_deg << " C" << pf_cont << " kA/kB=" << pwin.kA << "/" << pwin.kB
              << " turn=" << pfit.total_turn_deg << "deg nbody=" << ctrl.size()
              << " rho_min/r=" << (double)(rho/r_arm) << ((double)rho < (double)r_arm ? "  FOLD" : "") << "\n";
  }

  // ---- rotation-minimizing frame (double reflection), tabulated; closure-twist onto endE1 at t=1. ----
  const Integer Ng = std::max<Integer>((Integer)200, (Integer)20*n);
  auto tang = [&](Integer i) -> Vec3<Real> {
    const Real h = (Real)1/Ng;
    if (i <= 0)  return unit(sub(curve(h), curve((Real)0)));
    if (i >= Ng) return unit(sub(curve((Real)1), curve((Real)1-h)));
    return unit(sub(curve((i+1)*h), curve((i-1)*h)));
  };
  if (out_termC) *out_termC = curve((Real)1);
  if (out_termU) *out_termU = tang(Ng);
  std::vector<Vec3<Real>> Rtab((size_t)Ng+1);
  Vec3<Real> ti = tang(0);
  Vec3<Real> rr = unit(sub(seamA.e1, scal(dot(seamA.e1, ti), ti)));
  if ((double)nrm(rr) < 1e-9) { Vec3<Real> a{1,0,0}; if (std::fabs((double)ti[0])>0.9) a = Vec3<Real>{0,1,0}; rr = unit(sub(a, scal(dot(a,ti), ti))); }
  Rtab[0] = rr;
  for (Integer i = 0; i < Ng; i++) {
    const Vec3<Real> xi = curve(i/(Real)Ng), xi1 = curve((i+1)/(Real)Ng);
    const Vec3<Real> ti1 = tang(i+1);
    const Vec3<Real> v1 = sub(xi1, xi); const Real c1 = dot(v1, v1);
    const Vec3<Real> rL = (double)c1>0 ? sub(rr, scal(2*dot(v1,rr)/c1, v1)) : rr;
    const Vec3<Real> tL = (double)c1>0 ? sub(ti, scal(2*dot(v1,ti)/c1, v1)) : ti;
    const Vec3<Real> v2 = sub(ti1, tL); const Real c2d = dot(v2, v2);
    Vec3<Real> r1 = (double)c2d>0 ? sub(rL, scal(2*dot(v2,rL)/c2d, v2)) : rL;
    r1 = unit(r1); Rtab[(size_t)i+1] = r1; rr = r1; ti = ti1;
  }
  // Uniform closure twist so the far ring matches endE1 (neighbour/cap frame); identical to the blend path.
  Real phi = 0;
  if (!constant_orient && ((double)endE1[0]!=0 || (double)endE1[1]!=0 || (double)endE1[2]!=0)) {
    const Vec3<Real> tend = tang(Ng);
    const Vec3<Real> e1t = unit(sub(endE1, scal(dot(endE1, tend), tend)));
    const Vec3<Real> rend = Rtab[(size_t)Ng];
    if ((double)nrm(e1t) > 1e-9) {
      Real cph = dot(rend, e1t); cph = std::max((Real)-1, std::min((Real)1, cph));
      phi = std::acos((double)cph);
      if ((double)dot(cross(rend, e1t), tend) < 0) phi = -phi;
    }
  }
  const Vec3<Real> r0const = Rtab[0];
  auto orient_of = [&](Real t) -> Vec3<Real> {
    if (constant_orient) return r0const;
    Real g = t*Ng; if ((double)g < 0) g = 0; if (g > (Real)Ng) g = (Real)Ng;
    Integer i = (Integer)g; if (i >= Ng) i = Ng-1; const Real f = g - i;
    Vec3<Real> e1 = unit(add(scal((1-f), Rtab[(size_t)i]), scal(f, Rtab[(size_t)i+1])));
    if ((double)phi != 0) {                              // rotate e1 by t*phi about the local tangent (Rodrigues)
      const Vec3<Real> k = tang((Integer)std::lround((double)t*Ng));
      const Real th = t*phi, ct = std::cos((double)th), st = std::sin((double)th);
      e1 = add(add(scal(ct, e1), scal(st, cross(k, e1))), scal(dot(k,e1)*((Real)1-ct), k));
      e1 = unit(e1);
    }
    return e1;
  };

  // ---- panel knot vector: uniform, lead pinned to panels [0,lp) and [n-lp,n); turn-refine the spline body.
  std::vector<Real> tk; tk.reserve((size_t)n + 1);
  for (Integer k = 0; k <= n; k++) tk.push_back((Real)k / (Real)n);
  // POLYFIT panel layout: body = tk indices [lp+cp .. n-lp-cp] (nmid panels). Distribute them over the S wide
  // segments proportional to segment width (>=1 each, largest-remainder), uniform within a segment, so EVERY
  // segment knot is a panel boundary (each panel interior to one analytic degree-deg polynomial => exact).
  // tau (body parameter) maps to t = wCA + tau*(wCB-wCA). Turn-refine below only subdivides WITHIN panels.
  {
    const Integer b0 = lp + cp; const Integer S_ = pf_S;
    std::vector<Integer> nper((size_t)S_, 1); std::vector<std::pair<double,Integer>> frac;
    Integer used = S_; const Integer extra = nmid - S_;
    for (Integer i = 0; i < S_; i++) { const double x = (double)extra * (pf_f[(size_t)i+1]-pf_f[(size_t)i]);
      const Integer fl = (Integer)std::floor(x); nper[(size_t)i] += fl; used += fl; frac.push_back({x-fl, i}); }
    std::sort(frac.begin(), frac.end(), [](const std::pair<double,Integer>& a, const std::pair<double,Integer>& b){ return a.first > b.first; });
    for (size_t q = 0; used < nmid && q < frac.size(); q++, used++) nper[(size_t)frac[q].second]++;
    Integer k = 0;
    for (Integer i = 0; i < S_; i++)
      for (Integer m = 0; m < nper[(size_t)i]; m++, k++) {
        if (k == 0) continue;                                   // tk[b0] stays wCA
        const double tau = pf_f[(size_t)i] + (pf_f[(size_t)i+1]-pf_f[(size_t)i]) * (double)m / (double)nper[(size_t)i];
        tk[(size_t)(b0 + k)] = wCA + (Real)tau * (wCB - wCA);
      }
  }
  // Curvature-adaptive turn refinement (same lever as the blend path, default 6 deg / cap 24): split any
  // panel whose tangent bends more than the cap into ceil(bend/cap) sub-panels (same curve, same radius).
  const char* trc = std::getenv("QJ_ARM_TURN_REFINE") ? std::getenv("QJ_ARM_TURN_REFINE") : "6";
  if (trc && std::atof(trc) <= 0.0) trc = nullptr;
  const Integer turn_refine_max = std::getenv("QJ_ARM_TURN_REFINE_MAX") ? std::max<Integer>(1, (Integer)std::atoi(std::getenv("QJ_ARM_TURN_REFINE_MAX"))) : (Integer)24;
  if (trc) {
    const double cap = std::max(1.0, std::atof(trc));
    auto tang_at = [&](Real t, int side) -> Vec3<Real> {
      const Real d = (Real)1e-3;
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
      for (Integer sgi = 1; sgi < k; sgi++) tk3.push_back((Real)(ta + (tb - ta) * (Real)sgi / (Real)k));
    }
    tk3.push_back(tk.back());
    tk.swap(tk3);
  }
  const Integer npanel = (Integer)tk.size() - 1;

  // ---- Chebyshev interpolation error of the centerline (QJ_CORNER diag). CSBQ represents each panel's
  // centerline by its values at the `cheb` Chebyshev nodes; the degree-(cheb-1) interpolant of those nodes
  // is what the solver actually sees. The error = max_t |L_panel(t) - curve(t)| over each panel, i.e. how
  // far the panel polynomial departs from the true geometry. A seam corner that is smooth within each panel
  // drives this toward zero; a knot/curvature step inside a panel (the M8 straddle) spikes it. Report the
  // overall max/r AND the max over the two SEAM panel-blocks (lead+corner on each end) separately, since the
  // seam neighbourhood is what sets the junction cross-list free-edge. ----
  if (diag) {
    const Vector<Real>& cnd = SlenderElemList<Real>::CenterlineNodes(cheb);
    auto lagrange = [&](const std::vector<Vec3<Real>>& fj, Real xi) -> Vec3<Real> {   // nodes at cnd[], sample xi in [0,1]
      Vec3<Real> acc{(Real)0,(Real)0,(Real)0};
      for (Long j = 0; j < cheb; j++) {
        Real w = 1;
        for (Long m = 0; m < cheb; m++) if (m != j) w *= (double)(xi - cnd[m]) / (double)(cnd[j] - cnd[m]);
        acc = add(acc, scal(w, fj[(size_t)j]));
      }
      return acc;
    };
    Real emax = 0, eseam = 0; const Integer MS = 50;
    for (Integer p = 0; p < npanel; p++) {
      const Real ta = tk[(size_t)p], tb = tk[(size_t)p+1];
      std::vector<Vec3<Real>> fj((size_t)cheb);
      for (Long j = 0; j < cheb; j++) fj[(size_t)j] = curve(ta + cnd[j]*(tb-ta));
      Real ep = 0;
      for (Integer sgi = 0; sgi <= MS; sgi++) { const Real xi = (Real)sgi/MS;
        const Real e = nrm(sub(lagrange(fj, xi), curve(ta + xi*(tb-ta)))); if ((double)e > (double)ep) ep = e; }
      if ((double)ep > (double)emax) emax = ep;
      const bool seam_block = seam_corner ? ((double)tb <= (double)wCA + 1e-9 || (double)ta >= (double)wCB - 1e-9)
                                          : ((double)tb <= (double)wLA + 1e-9 || (double)ta >= (double)wLB - 1e-9);
      if (seam_block && (double)ep > (double)eseam) eseam = ep;
    }
    std::cout << "    [cheb interp err/r] max=" << (double)(emax/r_arm) << " seam=" << (double)(eseam/r_arm)
              << " (npanel=" << npanel << ")\n";
  }

  // ---- optional per-stage dump (QJ_DUMP_STAGES) so the verification artifact can draw this path too. ----
  if (const char* dsf = std::getenv("QJ_DUMP_STAGES"); dsf && dsf[0] && dump_tag) {
    std::ofstream dsout(dsf, std::ios::app);
    if (dsout) {
      dsout << std::setprecision(17);
      dsout << "# ARM " << dump_tag << " is_cap=" << (is_cap ? 1 : 0) << " downsample=1 polyfit=1\n";
      dsout << "SEAM A0 " << (double)A0[0] << " " << (double)A0[1] << " " << (double)A0[2]
            << " uA " << (double)uA[0] << " " << (double)uA[1] << " " << (double)uA[2]
            << " endC " << (double)endC[0] << " " << (double)endC[1] << " " << (double)endC[2]
            << " uB " << (double)uB[0] << " " << (double)uB[1] << " " << (double)uB[2]
            << " rA " << (double)rA << " rB " << (double)rB << "\n";
      auto dump_pts = [&](const char* nm, const std::vector<Vec3<Real>>& P) {
        dsout << "STAGE " << nm << " n=" << P.size() << "\n";
        for (const auto& p : P) dsout << (double)p[0] << " " << (double)p[1] << " " << (double)p[2] << "\n";
      };
      dump_pts("raw", cl_in);
      dump_pts("downsampled", ctrl);   // polyfit body points
      const Integer Nf = std::max<Integer>((Integer)4*npanel, (Integer)64);
      dsout << "STAGE final n=" << (Nf + 1) << "\n";
      for (Integer k = 0; k <= Nf; k++) { const Real t=(Real)k/Nf; const Vec3<Real> c=curve(t);
        dsout << (double)c[0] << " " << (double)c[1] << " " << (double)c[2] << " r=" << (double)radius_at(t) << "\n"; }
      dsout << "# END ARM " << dump_tag << "\n";
    }
  }

  // ---- emit CSBQ panels: Chebyshev-per-panel sampling of curve/radius/orient (never the panel endpoints). ----
  const Vector<Real>& cn = SlenderElemList<Real>::CenterlineNodes(cheb);
  const Long seam_c0 = coord.Dim();                             // node offset for the inter-panel seam check below
  for (Integer p = 0; p < npanel; p++) {
    const Real ta = tk[(size_t)p], tb = tk[(size_t)p + 1];
    elem_order.PushBack(cheb); forder.PushBack(fourier);
    for (Long j = 0; j < cheb; j++) {
      const Real t = ta + cn[j] * (tb - ta);
      const Vec3<Real> Pc = curve(t); const Vec3<Real> o = orient_of(t); const Real rv = radius_at(t);
      coord.PushBack(Pc[0]); coord.PushBack(Pc[1]); coord.PushBack(Pc[2]);
      radius.PushBack(rv);
      orient.PushBack(o[0]); orient.PushBack(o[1]); orient.PushBack(o[2]);
    }
  }
  // Inter-panel seam smoothness from the Chebyshev nodes CSBQ actually sees (QJ_ARM_SEAM_CHECK, default on):
  // dpos (C0), dtang (C1), dkappa (C2). With panel_snap aligning panels to the C2 knots these should fall to
  // the Chebyshev floor (dpos ~1e-10, dtang ~1e-6); a straddle keeps them at ~1e-4. Pure diagnostic.
  report_arm_seam_smoothness<Real>(coord, seam_c0, npanel, cheb, dump_tag);
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
