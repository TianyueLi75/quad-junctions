/**
 * bifurc-network-assemble -- turn a vmtk-derived vessel GRAPH (python/vessels_vmtk_graph.py ->
 * <prefix>.graph) into a placed quad-junction + bent-slender-arm NETWORK, and write it out as
 * per-junction reloadable bundles (<out>-jNN.{mesh,arms}) in the gen_geom "0_junc"/"1_arms" layout.
 *
 * Pipeline (see include/quad_junctions/gen_network_geom.hpp):
 *   - build ONE canonical junction body per graph CLUSTER (few representative junctions), open seams;
 *   - PLACE every junction node by a Horn-quaternion rigid fit of the canonical arm dirs to the node's
 *     true incident edge dirs + a uniform radius scale;
 *   - BEND each arm along its edge's vmtk centerline (tapered, seam-conforming, watertight), cap leaves;
 *   - report per-junction & COMBINED area / min-Jacobian-weight / watertight-flux (the playbook's
 *     geometry acceptance), optionally dump a combined VTU, and write the per-junction bundles.
 *
 *   make bin/bifurc-network-assemble
 *   ./bin/bifurc-network-assemble <graph> <out-prefix> [order nref level eta_join Ns_trans n_axial \
 *        cheb fourier lead_panels Ncap alpha_deg clampf s_cap core_frac dumpVTU]
 */
#include <csbq.hpp>
#include <quad_junctions/gen_network_geom.hpp>
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

using namespace sctl;
using namespace quad_junctions;

namespace {
using Real = double;

// Area / min-Jacobian-weight / watertight-flux / volume for one element list (copied from
// bifurc-general-geom.cpp). Returns the flux VECTOR via fout for combining lists.
template <class EL> void report_area(const std::string& name, const EL& el, const Comm& comm, double fout[3] = nullptr) {
  Vector<Real> X, Xn, wts, dist; Vector<Long> cnt;
  el.GetFarFieldNodes(X, Xn, wts, dist, cnt, (Real)1e-10);
  const Long N = wts.Dim(); Real A = 0, minw = 1e30, vol = 0, f[3] = {0, 0, 0};
  for (Long i = 0; i < N; i++) { A += wts[i]; minw = std::min(minw, wts[i]);
    for (int k = 0; k < 3; k++) f[k] += wts[i]*Xn[i*3+k];
    vol += (X[i*3]*Xn[i*3]+X[i*3+1]*Xn[i*3+1]+X[i*3+2]*Xn[i*3+2])*wts[i]/3; }
  A = GlobalReduce((double)A, comm, CommOp::SUM); vol = GlobalReduce((double)vol, comm, CommOp::SUM);
  minw = GlobalReduce((double)minw, comm, CommOp::MIN);
  for (int k = 0; k < 3; k++) f[k] = GlobalReduce((double)f[k], comm, CommOp::SUM);
  if (fout) for (int k = 0; k < 3; k++) fout[k] = f[k];
  if (!comm.Rank()) std::cout << "  [" << name << "] area=" << std::setprecision(8) << A
    << "  minWt=" << std::setprecision(4) << minw << (minw > 0 ? " (Jac>0)" : "  ***NEG JACOBIAN (fold)***")
    << "  |int n dA|=" << std::sqrt(f[0]*f[0]+f[1]*f[1]+f[2]*f[2]) << "  vol=" << vol << "\n";
}

// Unit tangent of `edge` leaving `nodeid`, over a short arc window (robust to the raw skeleton kink).
Vec3<Real> edge_dir_at(const GraphEdge<Real>& e, Integer nodeid, Real window) {
  using namespace gnet;
  const bool fwd = (e.n0 == nodeid);
  const auto& cl = e.cl; const Long n = (Long)cl.size();
  auto pt = [&](Long k) -> Vec3<Real> { return fwd ? cl[k] : cl[n-1-k]; };
  const Vec3<Real> base = pt(0); Vec3<Real> tip = pt(n-1); Real acc = 0;
  for (Long k = 1; k < n; k++) { acc += nrm(sub(pt(k), pt(k-1))); if ((double)acc >= (double)window) { tip = pt(k); break; } }
  Vec3<Real> d = sub(tip, base); return (double)nrm(d) > 1e-12 ? unit(d) : Vec3<Real>{1, 0, 0};
}

}  // namespace

int main(int argc, char** argv) {
  Comm::MPI_Init(&argc, &argv);
  {
    const Comm comm = Comm::World();
    if (argc < 3) { if (!comm.Rank()) std::cerr << "usage: bifurc-network-assemble <graph> <out-prefix> [order nref level eta_join Ns_trans n_axial cheb fourier lead_panels Ncap alpha_deg clampf s_cap core_frac dumpVTU lead_frac corner_panels turn_thresh_deg]\n"; Comm::MPI_Finalize(); return 1; }
    const std::string graph_path = argv[1];
    const std::string out_prefix = argv[2];
    const Integer order   = (argc > 3)  ? (Integer)atoi(argv[3])  : 12;
    const Integer nref    = (argc > 4)  ? (Integer)atoi(argv[4])  : 1;
    const Real    level   = (argc > 5)  ? (Real)atof(argv[5])     : (Real)1.5;
    const Real    etaj    = (argc > 6)  ? (Real)atof(argv[6])     : (Real)0.4;
    const Integer NsTr    = (argc > 7)  ? (Integer)atoi(argv[7])  : 3;
    const Integer nAx     = (argc > 8)  ? (Integer)atoi(argv[8])  : 12;
    const Long    cheb    = (argc > 9)  ? (Long)atoi(argv[9])     : 10;
    const Long    fourier = (argc > 10) ? (Long)atoi(argv[10])    : 12;
    const Integer leadP   = (argc > 11) ? (Integer)atoi(argv[11]) : 2;
    const Integer Ncap    = (argc > 12) ? (Integer)atoi(argv[12]) : -1;
    const Real    alphaD  = (argc > 13) ? (Real)atof(argv[13])    : (Real)38.0;
    const Real    clampf  = (argc > 14) ? (Real)atof(argv[14])    : (Real)0.8;
    const Real    s_cap   = (argc > 15) ? (Real)atof(argv[15])    : (Real)0.88;
    const Real    coreF   = (argc > 16) ? (Real)atof(argv[16])    : (Real)0.40;
    const bool    dumpVTU = (argc > 17) ? (atoi(argv[17]) != 0)   : false;
    const Real    leadFrac= (argc > 18) ? (Real)atof(argv[18])    : (Real)0.18;  // arm lead length / junction distance
    const Integer cornerP = (argc > 19) ? (Integer)atoi(argv[19]) : -1;          // arm corner panels (<0 => = lead_panels)
    const Real    turnThr = (argc > 20) ? (Real)atof(argv[20])    : (Real)20;    // turn (deg) above which the corner widens/adds panels

    if (!comm.Rank()) std::cout << "\n=== bifurc-network-assemble  graph=" << graph_path
      << "  order=" << order << " nref=" << nref << " nAx=" << nAx << " fourier=" << fourier
      << " leadP=" << leadP << " leadFrac=" << leadFrac << " cornerP=" << cornerP << " ===\n";

    NetworkGraph<Real> g = ReadNetworkGraph<Real>(graph_path);
    const Integer nnode = (Integer)g.nodes.size(), nedge = (Integer)g.edges.size(), nclu = (Integer)g.clusters.size();
    Integer njunc = 0, ncap = 0; for (auto& nd : g.nodes) (nd.is_junc ? njunc : ncap)++;
    if (!comm.Rank()) std::cout << "  graph: " << nnode << " nodes (" << njunc << " junc, " << ncap << " cap), "
      << nedge << " edges, " << nclu << " clusters\n";

    // Per-junction overrides for SELECT bad junctions (tight-gap, large-scale). QJ_JSIGMA="id:sigma,..."
    // forces the canonical sigma for that junction's cluster; QJ_SHRINK="id:factor,..." uniformly scales
    // the placed body DOWN (angle-preserving) so its absolute |int n dA| leak (~ size^2) shrinks -- the
    // arms auto-taper from the smaller seam to the neighbour's radius, and body+seam scale together so the
    // arm mouth still conforms. Clusters are 1:1 with junctions here, so a per-node sigma maps to its cluster.
    std::map<Integer,Real> node_sigma, node_shrink, clu_sigma;
    auto parse_ovr = [](const char* env, std::map<Integer,Real>& m){
      const char* s = getenv(env); if (!s) return; std::string str(s); size_t p = 0;
      while (p < str.size()) { size_t c = str.find(':', p), e = str.find(',', p);
        if (c == std::string::npos) break; if (e == std::string::npos) e = str.size();
        m[(Integer)std::atoi(str.substr(p, c-p).c_str())] = (Real)std::atof(str.substr(c+1, e-c-1).c_str()); p = e + 1; } };
    parse_ovr("QJ_JSIGMA", node_sigma); parse_ovr("QJ_SHRINK", node_shrink);
    for (auto& kv : node_sigma) if (kv.first >= 0 && kv.first < nnode) clu_sigma[g.nodes[kv.first].cluster] = kv.second;
    if (!comm.Rank()) { for (auto&kv:node_sigma) std::cout<<"  [ovr] j"<<kv.first<<" sigma="<<kv.second<<"\n";
                        for (auto&kv:node_shrink) std::cout<<"  [ovr] j"<<kv.first<<" shrink="<<kv.second<<"\n"; }

    // ---- 1. canonical junction per cluster (memoized) ----
    std::map<Integer, CanonicalJunction<Real>> canon;
    for (Integer c = 0; c < nclu; c++) {
      const Real sig_in = clu_sigma.count(c) ? clu_sigma[c] : (Real)-1;
      canon[c] = build_canonical<Real>(g.clusters[c], order, level, nref, etaj, NsTr, s_cap, Ncap, alphaD, clampf, sig_in);
      Real mR = 0; for (Real r : canon[c].R0) mR += r; mR /= std::max<size_t>(1, canon[c].R0.size());
      if (!comm.Rank()) std::cout << "  cluster " << c << " (deg " << g.clusters[c].degree << ")  meanR0=" << std::setprecision(4) << mR << "\n";
    }

    // ---- 2. place every junction: incident edges, arm<->edge match, world seams ----
    std::vector<Vector<Real>> Xbody(nnode);                          // per-junction placed quad nodes (+leaf caps)
    std::vector<Long> bodyN(nnode, 0);                               // node count BEFORE caps (debug split)
    std::vector<std::vector<Integer>> inc(nnode);                     // incident edge ids per node
    std::vector<std::map<Integer, ArmSeam<Real>>> seam(nnode);        // world seam per (node,edge)
    for (Integer e = 0; e < nedge; e++) { inc[g.edges[e].n0].push_back(e); inc[g.edges[e].n1].push_back(e); }

    // AUTO size-shrink for tight junctions (DEFAULT ON). Junction size is ~uniform here, so the absolute
    // watertightness leak is governed by scale = radius/meanR0: a tight-gap junction has a thin canonical
    // (small meanR0) => large scale => its leak (~ scale^2 * relative-aspect-floor) dominates the network
    // |int n dA|. Uniformly shrinking such a junction (angle-preserving; the arms taper to match each end,
    // see push_arm) cuts its absolute leak ~ shrink^2. Linear ramp in meanR0: factor 1 above ONSET, down to
    // MIN at LO. Tune/disable via env (QJ_SHRINK_MIN=1 disables); explicit per-junction QJ_SHRINK wins.
    const Real sh_on  = getenv("QJ_SHRINK_ONSET") ? (Real)atof(getenv("QJ_SHRINK_ONSET")) : (Real)0.13;
    const Real sh_lo  = getenv("QJ_SHRINK_LO")    ? (Real)atof(getenv("QJ_SHRINK_LO"))    : (Real)0.10;
    const Real sh_min = getenv("QJ_SHRINK_MIN")   ? (Real)atof(getenv("QJ_SHRINK_MIN"))   : (Real)0.5;
    auto auto_shrink = [&](Real mR0)->Real{ if (mR0 >= sh_on) return (Real)1;
      const Real t = (mR0 - sh_lo)/std::max<Real>((Real)1e-9, sh_on - sh_lo);
      return std::max<Real>(sh_min, std::min<Real>((Real)1, sh_min + ((Real)1 - sh_min)*t)); };

    Real worst_fit = 0;
    for (Integer i = 0; i < nnode; i++) {
      if (!g.nodes[i].is_junc) continue;
      const CanonicalJunction<Real>& C = canon[g.nodes[i].cluster];
      SCTL_ASSERT_MSG((Integer)inc[i].size() == C.N, "node degree != cluster arm count");
      const Real win = std::max<Real>((Real)1.5*g.nodes[i].radius, (Real)1e-6);
      std::vector<Vec3<Real>> edge_dirs(inc[i].size());
      for (size_t a = 0; a < inc[i].size(); a++) edge_dirs[a] = edge_dir_at(g.edges[inc[i][a]], i, win);
      Real mR0 = 0; for (Real r : C.R0) mR0 += r; mR0 /= std::max<size_t>(1, C.R0.size());
      const Real scale0 = (g.nodes[i].radius > 0 && mR0 > 0) ? g.nodes[i].radius / mR0 : (Real)1;
      const Real shk = node_shrink.count(i) ? node_shrink[i] : auto_shrink(mR0);        // explicit override else auto
      const Real scale = scale0 * shk;                                                   // per-junction size shrink
      if ((double)shk < 0.999 && !comm.Rank())
        std::cout << "  [shrink] j" << i << " meanR0=" << std::setprecision(4) << (double)mR0
                  << " -> factor=" << (double)shk << "\n";
      std::vector<Integer> arm_of;                                    // arm_of[a] = canonical arm for incident edge a
      Real wdeg = 0;
      Placement<Real> P = fit_placement<Real>(C, edge_dirs, g.nodes[i].pos, scale, arm_of, &wdeg);
      worst_fit = std::max(worst_fit, wdeg);
      Xbody[i] = C.Xcanon; transform_nodes<Real>(Xbody[i], P);        // placed body (caps appended later)
      bodyN[i] = Xbody[i].Dim();
      for (size_t a = 0; a < inc[i].size(); a++)
        seam[i][inc[i][a]] = transform_seam<Real>(C.seam(arm_of[a]), P);
    }
    if (!comm.Rank()) std::cout << "  worst per-arm placement residual: " << std::setprecision(3) << worst_fit << " deg\n";

    // ---- 3. arms: interior owned by min(i,j); leaf capped. Accumulate global slender + per-junction bundles ----
    std::vector<NetworkArmBundle<Real>> bundle(nnode);
    for (Integer i = 0; i < nnode; i++) { bundle[i].order = order; bundle[i].cheb = cheb; bundle[i].fourier = fourier; }
    Vector<Long> gelem, gforder; Vector<Real> gcoord, gradius, gorient;   // combined slender

    const bool straight = (getenv("QJ_STRAIGHT") != nullptr);
    auto push_arm = [&](Integer owner, const ArmSeam<Real>& sA, const Vec3<Real>& endC, const Vec3<Real>& endTang,
                        const std::vector<Vec3<Real>>& cl, Real rA, Real rB, Integer other, bool is_cap) {
      NetworkArmBundle<Real>& B = bundle[owner];
      const Long p0 = (Long)B.elem_order.Dim();
      // The arm mouth radius at t=0 IS rA (radius=rA+t*(rB-rA)); rA=owner seam R0 (shrunk if the owner
      // junction is shrunk), rB=neighbour seam R0 (interior) or vessel/cap radius (leaf). So the arm
      // tapers to match EACH end's junction size. QJ_ARMDIAG proves it (mouth==shrunk seam, far==neighbour).
      if (getenv("QJ_ARMDIAG") && !comm.Rank())
        std::cout << "  [armdiag] owner=j" << owner << " other=j" << other << (is_cap?" (cap)":"")
                  << "  rA(mouth)=" << (double)rA << " == ownerSeamR0=" << (double)sA.R0
                  << "   rB(far)=" << (double)rB << "\n";
      // BOTH interior and leaf arms get the lead-corner-straight bend: the arm leaves the junction along
      // its arm-exit axis seamA.u (lead) so its terminal ring conforms to the junction hole (no gap), then
      // corners toward the far end -- the neighbour junction seam (interior) or the cap tip (leaf). A plain
      // straight leaf tube pointed its mouth ring at (tip - seamC) instead of seamA.u, tilting it off the
      // hole -> the junction<->cap gap. The cap tip position and centerline still come from the network.
      // QJ_STRAIGHT forces the old straight tube for A/B.
      auto emit = [&](Vector<Long>& eo, Vector<Long>& fo, Vector<Real>& co, Vector<Real>& ra, Vector<Real>& oo) {
        if (straight) append_straight_fiber<Real>(sA, endC, rA, rB, nAx, cheb, fourier, eo, fo, co, ra, oo);
        else append_centerline_fiber<Real>(sA, endC, endTang, cl, rA, rB, nAx, leadP, cheb, fourier, eo, fo, co, ra, oo,
                                           getenv("QJ_CONSTORIENT") != nullptr, leadFrac, cornerP, turnThr);
      };
      emit(B.elem_order, B.forder, B.coord, B.radius, B.orient);
      B.other_node.push_back(other); B.is_cap.push_back(is_cap ? 1 : 0);
      B.npanel.push_back((Long)B.elem_order.Dim() - p0);
      emit(gelem, gforder, gcoord, gradius, gorient);                 // mirror into combined slender list
    };

    // Optional connection diagnostic: per interior edge, the TURN each junction mouth axis makes relative
    // to the straight chord to the neighbour (how far off-chord the arm sprouts), the seam center's offset
    // from the junction node, and the seam-vs-neighbour radius match -- all tagged by junction id.
    const bool seamdiag = (getenv("QJ_SEAMDIAG") != nullptr);
    std::vector<double> turnsA, turnsB, radErr, ctrOff;
    std::vector<Integer> owA, owB, kind;   // kind: 0=interior edge, 1=leaf (junction->cap)

    for (Integer e = 0; e < nedge; e++) {
      const GraphEdge<Real>& E = g.edges[e];
      const Integer i = E.n0, j = E.n1;
      const bool ji = g.nodes[i].is_junc, jj = g.nodes[j].is_junc;
      if (ji && jj) {
        // interior: owner = min, other = max. centerline oriented owner->other.
        const Integer own = std::min(i, j), oth = std::max(i, j);
        const ArmSeam<Real>& sA = seam[own].at(e);
        const ArmSeam<Real>& sB = seam[oth].at(e);
        std::vector<Vec3<Real>> cl = E.cl; if (E.n0 != own) std::reverse(cl.begin(), cl.end());
        const Vec3<Real> endTang = gnet::scal((Real)-1, sB.u);          // travel INTO the other junction
        if (seamdiag) {
          const Vec3<Real> ch = gnet::sub(sB.C, sA.C); const Real Lc = gnet::nrm(ch);
          const Vec3<Real> cd = (Lc > 0) ? gnet::scal((Real)1/Lc, ch) : ch;
          auto ang = [&](const Vec3<Real>& p, const Vec3<Real>& q){ double d = (double)gnet::dot(p, q); d = std::max(-1.0, std::min(1.0, d)); return std::acos(d)*180.0/M_PI; };
          turnsA.push_back(ang(sA.u, cd));
          turnsB.push_back(ang(sB.u, gnet::scal((Real)-1, cd)));
          kind.push_back(0);
          radErr.push_back(std::fabs((double)(sA.R0 - sB.R0))/std::max(1e-30,(double)sA.R0));
          ctrOff.push_back((Lc>0)?(double)gnet::nrm(gnet::sub(sA.C, g.nodes[own].pos))/(double)Lc:0.0);
          owA.push_back(own); owB.push_back(oth);
        }
        push_arm(own, sA, sB.C, endTang, cl, sA.R0, sB.R0, oth, false);
      } else if (ji || jj) {
        // leaf: junction endpoint owns; far end is a cap tip.
        const Integer own = ji ? i : j, cap = ji ? j : i;
        const ArmSeam<Real>& sA = seam[own].at(e);
        std::vector<Vec3<Real>> cl = E.cl; if (E.n0 != own) std::reverse(cl.begin(), cl.end());
        const Vec3<Real> tip = g.nodes[cap].pos;
        Vec3<Real> endTang = gnet::unit(gnet::sub(tip, sA.C));
        const Real rB = (E.n0 == cap) ? E.r0 : E.r1;                    // vessel radius at the cap
        const Real rBt = (rB > 0) ? rB : sA.R0;
        if (seamdiag) {
          const Vec3<Real> ch = gnet::sub(tip, sA.C); const Real Lc = gnet::nrm(ch);
          const Vec3<Real> cd = (Lc > 0) ? gnet::scal((Real)1/Lc, ch) : ch;
          double d = (double)gnet::dot(sA.u, cd); d = std::max(-1.0, std::min(1.0, d));
          turnsA.push_back(std::acos(d)*180.0/M_PI); turnsB.push_back(0.0);
          radErr.push_back(0.0); ctrOff.push_back(0.0);
          owA.push_back(own); owB.push_back(cap); kind.push_back(1);
        }
        push_arm(own, sA, tip, endTang, cl, sA.R0, rBt, cap, true);
        // hemisphere cap butterfly at the tip (world frame): any e1 perp to endTang.
        ArmSeam<Real> capring; capring.C = tip; capring.u = endTang; capring.R0 = rBt;
        Vec3<Real> a{1, 0, 0}; if (std::fabs((double)endTang[0]) > 0.9) a = Vec3<Real>{0, 1, 0};
        capring.e1 = gnet::unit(gnet::sub(a, gnet::scal(gnet::dot(a, endTang), endTang)));
        capring.e2 = gnet::cross(capring.e1, endTang);
        add_cap_hemisphere_frame<Real>(Xbody[own], capring, order, Ncap > 0 ? Ncap : 2, coreF);
      }
    }

    // PER-JUNCTION localizer: close each junction body ALONE by capping every incident seam with a
    // hemisphere, then measure its own |int n dA| / area. Isolates whether the network watertightness
    // floor is a few tight-angle junction BODIES (gen-mesher aspect floor) vs the arm seams.
    if (getenv("QJ_PERJUNC") && !comm.Rank()) {
      std::vector<std::pair<double,Integer>> leak;   // (|f|, junction id)
      double sumf = 0, maxf = 0; Integer argmax = -1;
      for (Integer i = 0; i < nnode; i++) {
        if (!g.nodes[i].is_junc) continue;
        Vector<Real> Xb; for (Long k = 0; k < bodyN[i]; k++) Xb.PushBack(Xbody[i][k]);   // pure placed body
        for (Integer e : inc[i]) add_cap_hemisphere_frame<Real>(Xb, seam[i][e], order, 2, coreF);
        QuadElemList<Real> jc(order, Xb, Comm::Self());
        Vector<Real> X, Xn, wts, dist; Vector<Long> cnt; jc.GetFarFieldNodes(X, Xn, wts, dist, cnt, (Real)1e-10);
        double f[3] = {0,0,0}, A = 0, minw = 1e30;
        for (Long q = 0; q < wts.Dim(); q++) { A += (double)wts[q]; minw = std::min(minw,(double)wts[q]);
          for (int k=0;k<3;k++) f[k] += (double)wts[q]*(double)Xn[q*3+k]; }
        const double nf = std::sqrt(f[0]*f[0]+f[1]*f[1]+f[2]*f[2]);
        leak.push_back({nf, i}); sumf += nf; if (nf > maxf) { maxf = nf; argmax = i; }
        if (minw <= 0) std::cout << "  [perjunc] j" << i << " FOLD minWt=" << minw << "\n";
      }
      std::sort(leak.begin(), leak.end(), [](auto&a, auto&b){ return a.first > b.first; });
      std::cout << "  [perjunc] " << leak.size() << " closed junction bodies: sum|f|=" << std::setprecision(6) << sumf
                << "  max|f|=" << maxf << " (j" << argmax << ")\n";
      std::cout << "  [perjunc] worst 12 (|f| : deg gap):\n";
      for (size_t r = 0; r < std::min<size_t>(12, leak.size()); r++) {
        const Integer i = leak[r].second; const auto& cl = g.clusters[g.nodes[i].cluster];
        std::cout << "      j" << i << "  |f|=" << std::setprecision(4) << leak[r].first
                  << "  deg=" << cl.degree << "\n";
      }
    }

    if (seamdiag && !comm.Rank() && !turnsA.empty()) {
      auto stat = [&](std::vector<double>& v, const char* nm){ double mn=1e30,mx=-1e30,s=0; for (double x:v){mn=std::min(mn,x);mx=std::max(mx,x);s+=x;}
        std::cout << "  [seamdiag] " << nm << ": min=" << mn << " mean=" << s/v.size() << " max=" << mx << "\n"; };
      std::cout << "  [seamdiag] " << turnsA.size() << " arms (interior + leaf)\n";
      stat(turnsA, "turn@A (deg, mouth-axis vs chord)");
      stat(turnsB, "turn@B (deg)");
      // histogram of the max turn per arm
      const double edges[] = {0,10,20,30,35,45,60,90,180};
      const int nb = 8; int hist[8] = {0};
      for (size_t k=0;k<turnsA.size();k++){ double m=std::max(turnsA[k],turnsB[k]); for (int b=0;b<nb;b++) if (m>=edges[b]&&m<edges[b+1]){hist[b]++;break;} }
      std::cout << "  [seamdiag] max-turn histogram:";
      for (int b=0;b<nb;b++) std::cout << " [" << edges[b] << "-" << edges[b+1] << ")=" << hist[b];
      std::cout << "\n";
      // full CSV for offline analysis by junction id
      std::ofstream csv("data/premade-val/netfix_turns.csv");
      csv << "kind,jA,jB,turnA,turnB,maxturn\n";
      for (size_t k=0;k<turnsA.size();k++)
        csv << (kind[k]?"leaf":"int") << "," << owA[k] << "," << owB[k] << ","
            << turnsA[k] << "," << turnsB[k] << "," << std::max(turnsA[k],turnsB[k]) << "\n";
      std::cout << "  [seamdiag] wrote data/premade-val/netfix_turns.csv (" << turnsA.size() << " arms)\n";
      std::vector<size_t> idx(turnsA.size()); for (size_t k=0;k<idx.size();k++) idx[k]=k;
      std::sort(idx.begin(), idx.end(), [&](size_t a, size_t b){ return std::max(turnsA[a],turnsB[a]) > std::max(turnsA[b],turnsB[b]); });
      std::cout << "  [seamdiag] worst 15 turns (kind jA<->jB : turnA turnB):\n";
      for (size_t r=0; r<std::min<size_t>(15, idx.size()); r++){ size_t k=idx[r];
        std::cout << "      " << (kind[k]?"leaf":"int ") << " j" << owA[k] << " <-> j" << owB[k] << " : " << turnsA[k] << " " << turnsB[k] << "\n"; }
    }

    // ---- 4. combined validation surface ----
    // NB the whole geometry is REPLICATED on every rank (the build loops above run identically per rank;
    // this driver gets no MPI speedup). So the combined flux must be measured on ONE rank with Comm::Self:
    // SlenderElemList has no comm ctor, so arms_all is replicated -- reporting it over the World comm
    // GlobalReduce-SUMs it once per rank (arms flux 4x under -n 4 => bogus COMBINED ~906). quad_all
    // partitioned fine, but mixing a partitioned junc with a replicated-and-4x-summed arms is the bug.
    // Doing it all on rank 0 with Comm::Self is correct and identical in serial.
    double fj[3] = {0, 0, 0}, fa[3] = {0, 0, 0};
    if (!comm.Rank()) {
      Vector<Real> Xall;
      for (Integer i = 0; i < nnode; i++) if (Xbody[i].Dim()) for (Long k = 0; k < Xbody[i].Dim(); k++) Xall.PushBack(Xbody[i][k]);
      QuadElemList<Real> quad_all(order, Xall, Comm::Self());
      SlenderElemList<Real> arms_all(gelem, gforder, gcoord, gradius, gorient);
      std::cout << "\n--- combined geometry acceptance ---\n";
      report_area("junc(all)", quad_all, Comm::Self(), fj);
      report_area("arms(all)", arms_all, Comm::Self(), fa);
      const double fc[3] = {fj[0]+fa[0], fj[1]+fa[1], fj[2]+fa[2]};
      std::cout << "  [COMBINED] |int n dA| = " << std::setprecision(6)
        << std::sqrt(fc[0]*fc[0]+fc[1]*fc[1]+fc[2]*fc[2]) << "   (watertight closure)\n";
    }

    if (getenv("QJ_NETDEBUG")) {
      Vector<Real> Xbo, Xcp;
      for (Integer i = 0; i < nnode; i++) if (Xbody[i].Dim()) {
        for (Long k = 0; k < bodyN[i]; k++) Xbo.PushBack(Xbody[i][k]);
        for (Long k = bodyN[i]; k < Xbody[i].Dim(); k++) Xcp.PushBack(Xbody[i][k]);
      }
      QuadElemList<Real> body_only(order, Xbo, comm);
      if (!comm.Rank()) std::cout << "  [debug split]\n";
      double fb[3] = {0,0,0}, fcp[3] = {0,0,0};
      report_area("body-only(no caps)", body_only, comm, fb);
      if (Xcp.Dim()) { QuadElemList<Real> caps_only(order, Xcp, comm); report_area("caps-only", caps_only, comm, fcp); }
      if (!comm.Rank()) {
        auto mag = [](const double f[3]) { return std::sqrt(f[0]*f[0]+f[1]*f[1]+f[2]*f[2]); };
        const double bs[3] = {fb[0]+fa[0], fb[1]+fa[1], fb[2]+fa[2]};       // body + arms = SEAM closure
        const double cs[3] = {fcp[0]+fa[0], fcp[1]+fa[1], fcp[2]+fa[2]};    // caps + arms = TIP closure
        std::cout << "  |body+arms| (seam closure) = " << std::setprecision(4) << mag(bs)
                  << "   |caps+arms| (tip closure) = " << mag(cs) << "\n";
      }
    }

    if (dumpVTU && !comm.Rank()) {   // legacy single-piece dump (the MPI VTK is the separate bifurc-network-vtk)
      Vector<Real> Xall;
      for (Integer i = 0; i < nnode; i++) if (Xbody[i].Dim()) for (Long k = 0; k < Xbody[i].Dim(); k++) Xall.PushBack(Xbody[i][k]);
      QuadElemList<Real> quad_all(order, Xall, Comm::Self());
      SlenderElemList<Real> arms_all(gelem, gforder, gcoord, gradius, gorient);
      quad_all.WriteVTK(out_prefix + "-junc.vtu"); arms_all.WriteVTK(out_prefix + "-arms.vtu");
      std::cout << "  wrote " << out_prefix << "-{junc,arms}.vtu\n";
    }

    // ---- 5. per-junction bundles ----
    Integer nb = 0;
    for (Integer i = 0; i < nnode; i++) {
      if (!g.nodes[i].is_junc) continue;
      QuadElemList<Real> junc_i(order, Xbody[i], Comm::Self());
      WriteNetworkBundle<Real>(out_prefix, i, junc_i, bundle[i], comm);
      nb++;
    }
    if (!comm.Rank()) std::cout << "  wrote " << nb << " per-junction bundles to " << out_prefix << "-jNNN.{mesh,arms}\n";
  }
  Comm::MPI_Finalize();
  return 0;
}
