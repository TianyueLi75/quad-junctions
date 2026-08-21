/**
 * ybifurc-tree-bie: a ~10-junction vascular-network approximation.
 *
 * A rough BIE model in the spirit of a cerebral vessel tangle: ONE reused Y-junction (the canonical
 * iso-surface of ybifurc_geom / ybifurc_assembly), rigidly ROTATED + uniformly SCALED into a branching
 * tree, with neighbouring junctions joined by CSBQ slender arms whose radius (r) tapers between the two
 * junction sizes and whose centerline carries a mid-arm sine wiggle (Xc) that vanishes flat at both
 * seams -- so all curvature sits far from the junctions and caps, preserving the near-singular
 * quadrature. Terminal seams get straight hemisphere-capped free arms (pointing in varied directions
 * set by each junction's rotation, giving a curved-terminal-vessel illusion).
 *
 * This generalizes case 2 of ybifurc-multi-bie.cpp (two junctions on the x-axis) to a recursively
 * placed tree. All new logic is local to this driver; it reuses the HybridAssembly component API
 * (include/quad_junctions/ybifurc_assembly.hpp) and the shared BIE identity tests
 * (include/quad_junctions/hybrid_bie_tests.hpp) unchanged.
 *
 * Verification: constant-density DL identity (= -1/2, watertightness) + Green's identity with EXTERIOR
 * point sources (one candidate per junction, validated exterior to the whole assembly), Laplace + Stokes.
 *
 *   make bin/ybifurc-tree-bie
 *   OMP_NUM_THREADS=8 ./bin/ybifurc-tree-bie \
 *       [level] [order(mult4)] [nref] [eta_join] [Ns_trans] [tipLen] [n_axial_free] [fourier] \
 *       [nlev] [tol] [Nbeta] [max_depth] [cov_q] [geomOnly]
 *
 *   geomOnly=1 builds the geometry, runs the collision/watertightness geometry checks, writes the VTU
 *   meshes and exits WITHOUT the (expensive) BIE solve -- for fast layout iteration.
 */

#include <csbq.hpp>                                  // CSBQ SlenderElemList
#include <quad_junctions/ybifurc_assembly.hpp>       // composable component API
#include <quad_junctions/quad_scheme.hpp>            // QJDefaultScheme (Duffy default, SCTL_SELF_SCHEME=hybrid opt-out)
#include <quad_junctions/hybrid_bie_tests.hpp>       // shared BIE identity / watertightness tests
#include <array>
#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <limits>
#include <string>
#include <vector>

using namespace sctl;
using namespace quad_junctions;

namespace {

// --- small Vec3 helpers (Vec3 has operator[] but we keep component-wise math explicit) -------------
template <class Real> Real dot3(const Vec3<Real>& a, const Vec3<Real>& b) { return a[0]*b[0]+a[1]*b[1]+a[2]*b[2]; }
template <class Real> Real nrm3(const Vec3<Real>& a) { return std::sqrt((double)dot3(a,a)); }
template <class Real> Vec3<Real> cross3(const Vec3<Real>& a, const Vec3<Real>& b) {
  return Vec3<Real>{a[1]*b[2]-a[2]*b[1], a[2]*b[0]-a[0]*b[2], a[0]*b[1]-a[1]*b[0]};
}
template <class Real> Vec3<Real> add3(const Vec3<Real>& a, const Vec3<Real>& b) { return Vec3<Real>{a[0]+b[0],a[1]+b[1],a[2]+b[2]}; }
template <class Real> Vec3<Real> sub3(const Vec3<Real>& a, const Vec3<Real>& b) { return Vec3<Real>{a[0]-b[0],a[1]-b[1],a[2]-b[2]}; }
template <class Real> Vec3<Real> mul3(Real s, const Vec3<Real>& a) { return Vec3<Real>{s*a[0],s*a[1],s*a[2]}; }

// Pick an "up" reference for AlignArm that is safely non-parallel to `dir` (falls back through the axes).
template <class Real> Vec3<Real> safe_up(const Vec3<Real>& dir, Vec3<Real> hint) {
  if (nrm3(cross3(dir, hint)) > (Real)0.15) return hint;
  const Vec3<Real> ax[3] = {Vec3<Real>{1,0,0}, Vec3<Real>{0,1,0}, Vec3<Real>{0,0,1}};
  for (int i = 0; i < 3; i++) if (nrm3(cross3(dir, ax[i])) > (Real)0.5) return ax[i];
  return Vec3<Real>{0,0,1};
}

// Place a child junction so its arm 0 is anti-parallel to and collinear with the parent seam `sp` it
// hangs off, with the child seam-0 ring a gap `L` beyond `sp.C` along `sp.u`. This makes the two seams
// coaxial and facing, exactly satisfying add_shared_arm's asserts. `cseam0C` is the canonical (identity-
// frame) seam-0 center; `scale` uniformly resizes the child.
template <class Real>
Placement<Real> place_child(const ArmSeam<Real>& sp, const Vec3<Real>& up_hint, Real L, Real scale,
                            const Vec3<Real>& cseam0C) {
  const Vec3<Real> dir = mul3((Real)-1, sp.u);                 // child arm0 faces back toward the parent
  const Vec3<Real> up  = safe_up<Real>(dir, up_hint);
  Placement<Real> P0 = Placement<Real>::AlignArm(0, dir, up, Vec3<Real>{0,0,0}, scale);   // rotation+scale only
  const Vec3<Real> delta = P0.apply_point(cseam0C);            // = scale*(R*cseam0C), origin at 0
  const Vec3<Real> Cc = add3(sp.C, mul3(L, sp.u));             // target child seam-0 center
  Placement<Real> P = P0; P.t = sub3(Cc, delta);              // shift origin so seam0 lands at Cc
  return P;
}

// Closest distance between two 3D segments [p1,q1] and [p2,q2] (clamped parametric solve).
template <class Real>
Real seg_seg_dist(const Vec3<Real>& p1, const Vec3<Real>& q1, const Vec3<Real>& p2, const Vec3<Real>& q2) {
  const Vec3<Real> d1 = sub3(q1,p1), d2 = sub3(q2,p2), r = sub3(p1,p2);
  const Real a = dot3(d1,d1), e = dot3(d2,d2), f = dot3(d2,r);
  Real s, t;
  const Real eps = (Real)1e-12;
  if (a <= eps && e <= eps) return nrm3(r);
  if (a <= eps) { s = 0; t = std::min<Real>(1, std::max<Real>(0, f/e)); }
  else {
    const Real c = dot3(d1,r);
    if (e <= eps) { t = 0; s = std::min<Real>(1, std::max<Real>(0, -c/a)); }
    else {
      const Real b = dot3(d1,d2), den = a*e-b*b;
      s = (den > eps) ? std::min<Real>(1, std::max<Real>(0,(b*f-c*e)/den)) : (Real)0;
      t = (b*s+f)/e;
      if (t < 0)      { t = 0; s = std::min<Real>(1, std::max<Real>(0, -c/a)); }
      else if (t > 1) { t = 1; s = std::min<Real>(1, std::max<Real>(0, (b-c)/a)); }
    }
  }
  const Vec3<Real> c1 = add3(p1, mul3(s,d1)), c2 = add3(p2, mul3(t,d2));
  return nrm3(sub3(c1,c2));
}

// One tree node: a placed junction. parent<0 is the root. `pseam` is the parent seam this node hangs
// off; `armL` the shared-arm gap; `sine_amp` the wiggle amplitude as a MULTIPLE of the arm's mean tube
// radius; `sine_per` the wiggle periods; `up` the placement up-hint; `scale` the junction resize.
template <class Real> struct Node {
  int parent; int pseam; Real scale; Real armL; Real sine_amp; Real sine_per; Vec3<Real> up;
};

// An arm's axis segment + effective outer radius + the junction id(s) it touches (for the collision
// guard and exterior-source validation).
template <class Real> struct ArmSeg { Vec3<Real> A, B; Real rtube; int j0, j1; };

} // anonymous namespace

int main(int argc, char** argv) {
  Comm::MPI_Init(&argc, &argv);
  using Real = double;
  {
    const Comm comm = Comm::World();
    const Real    level   = (argc > 1)  ? (Real)atof(argv[1])  : (Real)1.5;
    const Integer ord     = (argc > 2)  ? (Integer)atoi(argv[2])  : 8;
    const Integer nref    = (argc > 3)  ? (Integer)atoi(argv[3])  : 1;
    const Real    etajoin = (argc > 4)  ? (Real)atof(argv[4])  : (Real)0.4;
    const Integer NsTrans = (argc > 5)  ? (Integer)atoi(argv[5])  : 3;
    const Real    tipLen  = (argc > 6)  ? (Real)atof(argv[6])  : (Real)3.0;   // free-arm length (x junction scale)
    const Integer nAxFree = (argc > 7)  ? (Integer)atoi(argv[7])  : 3;
    const Long    fourier = (argc > 8)  ? (Long)atoi(argv[8])  : 8;
    const Integer nlev    = (argc > 9)  ? (Integer)atoi(argv[9])  : 1;        // unused>1 here (single level)
    const Real    tol     = (argc > 10) ? (Real)atof(argv[10]) : (Real)1e-6;
    const Integer Nbeta   = (argc > 11) ? (Integer)atoi(argv[11]) : 100;
    const Integer maxdep  = (argc > 12) ? (Integer)atoi(argv[12]) : 4;
    const Integer cov_q   = (argc > 13) ? (Integer)atoi(argv[13]) : 6;
    const Integer geomOnly= (argc > 14) ? (Integer)atoi(argv[14]) : 0;
    (void)nlev;
    const Integer Ncap    = (Integer)(YSwept::Ncap0 * nref);
    const Long    cheb    = 10;
    pou_kind() = 1;   // smootherstep POU (order-exact) -- what the assembly's transitions expect

    SCTL_ASSERT_MSG(ord >= 4 && ord <= 48 && ord % 4 == 0, "order must be a multiple of 4 in {4,...,48}.");

    if (!comm.Rank()) {
      std::cout << "\n=== ybifurc-tree: ~10-junction vascular-network approximation ===\n";
      std::cout << "  order=" << ord << " level=" << level << " nref=" << nref << " eta_join=" << etajoin
                << " Ns_trans=" << NsTrans << " fourier=" << fourier << " tipLen=" << tipLen
                << (geomOnly ? "  [GEOM-ONLY]\n" : "\n");
    }

    // Native (unscaled) seam radius -> scale = r/R0n; and the canonical seam-0 center (identity frame)
    // needed by place_child. One throwaway reference junction supplies the canonical seams.
    const Real R0n = canonical_seam_R0<Real>(level, etajoin);
    const HybridJunction<Real> Jref(ord, level, nref, etajoin, NsTrans, Placement<Real>::Identity());
    const Vec3<Real> cseam0C = Jref.seam(0).C;

    // ----------------------------------------------------------------------------------------------
    // Static tree spec: root (3 children) + 3 (each 2 children) = 10 junctions. Parents precede
    // children (BFS order), so a child always reads its parent's already-built world seam.
    // sine_amp is a MULTIPLE of the arm's mean tube radius; armL/tipLen scale with junction size.
    // ----------------------------------------------------------------------------------------------
    std::vector<Node<Real>> nodes = {
      /* 0 root */ { -1, 0, (Real)1.4, (Real)0,   (Real)0,   (Real)0,   Vec3<Real>{1,0,0} },
      /* 1      */ {  0, 0, (Real)1.0, (Real)7.0, (Real)1.4, (Real)1.0, Vec3<Real>{0,1,0} },
      /* 2      */ {  0, 1, (Real)1.0, (Real)7.5, (Real)1.6, (Real)1.5, Vec3<Real>{0,0,1} },
      /* 3      */ {  0, 2, (Real)0.9, (Real)8.0, (Real)1.3, (Real)1.0, Vec3<Real>{0,1,0} },
      /* 4      */ {  1, 1, (Real)0.65,(Real)6.0, (Real)1.5, (Real)1.5, Vec3<Real>{1,0,0} },
      /* 5      */ {  1, 2, (Real)0.70,(Real)6.5, (Real)1.4, (Real)1.0, Vec3<Real>{0,0,1} },
      /* 6      */ {  2, 1, (Real)0.65,(Real)6.5, (Real)1.5, (Real)1.5, Vec3<Real>{1,0,0} },
      /* 7      */ {  2, 2, (Real)0.60,(Real)6.0, (Real)1.6, (Real)1.0, Vec3<Real>{0,1,0} },
      /* 8      */ {  3, 1, (Real)0.70,(Real)6.0, (Real)1.4, (Real)1.5, Vec3<Real>{0,1,0} },
      /* 9      */ {  3, 2, (Real)0.60,(Real)6.5, (Real)1.5, (Real)1.0, Vec3<Real>{0,0,1} },
    };
    const int NJ = (int)nodes.size();

    HybridAssembly<Real> A(ord);
    std::vector<HybridJunction<Real>> J; J.reserve(NJ);
    std::vector<Placement<Real>> P(NJ);
    std::vector<std::array<bool,3>> consumed(NJ, {false,false,false});
    std::vector<ArmSeg<Real>> segs;

    // --- build junctions in tree order; connect each non-root to its parent by a shared wiggly arm ---
    for (int i = 0; i < NJ; i++) {
      const Node<Real>& nd = nodes[i];
      const Real scale = nd.scale;
      if (nd.parent < 0) {
        // root: arm0 along +z, up +x, at origin
        P[i] = Placement<Real>::AlignArm(0, Vec3<Real>{0,0,1}, safe_up<Real>(Vec3<Real>{0,0,1}, nd.up),
                                         Vec3<Real>{0,0,0}, scale);
      } else {
        const ArmSeam<Real>& sp = J[nd.parent].seam(nd.pseam);
        P[i] = place_child<Real>(sp, nd.up, nd.armL, scale, cseam0C);
      }
      J.push_back(A.add_junction(P[i], level, nref, etajoin, NsTrans));

      if (nd.parent >= 0) {
        consumed[i][0] = true;                    // inbound seam of this child
        consumed[nd.parent][nd.pseam] = true;     // spawn seam of the parent
        const ArmSeam<Real>& sa = J[nd.parent].seam(nd.pseam);   // parent seam (points toward child)
        const ArmSeam<Real>& sb = J[i].seam(0);                  // child seam0 (points back at parent)
        const Real len = nrm3(sub3(sb.C, sa.C));
        const Real rmean = (Real)0.5*(sa.R0 + sb.R0);
        const Real sineAmpS = nd.sine_amp * rmean;               // amplitude as a length
        // axial panels: unit-aspect for the straight part, but resolve the wiggle to its curvature
        // (~40 panels/wavelength; the C-inf envelope makes this placement-insensitive).
        Real pspac = sa.R0 + sb.R0;
        if (sineAmpS != (Real)0 && nd.sine_per > 0) pspac = std::min(pspac, (len/nd.sine_per)/40);
        const Integer ns = std::max<Integer>(4, (Integer)std::lround((double)len/pspac));
        A.add_shared_arm(sa, sb, ns, cheb, fourier, sineAmpS, nd.sine_per);
        segs.push_back(ArmSeg<Real>{sa.C, sb.C, std::max(sa.R0,sb.R0)+std::fabs((double)sineAmpS), nd.parent, i});
      }
    }

    // --- cap every remaining (unconsumed) seam with a straight hemisphere free arm ---
    int n_caps = 0;
    for (int i = 0; i < NJ; i++)
      for (int k = 0; k < 3; k++)
        if (!consumed[i][k]) {
          const ArmSeam<Real>& s = J[i].seam(k);
          const Real L = nodes[i].scale * tipLen;
          A.add_free_arm(s, s.a0 + L, nAxFree, Ncap, cheb, fourier);
          segs.push_back(ArmSeg<Real>{s.C, add3(s.C, mul3(L, s.u)), s.R0, i, i});
          n_caps++;
        }

    QuadElemList<Real> junc = A.quad(comm);
    SlenderElemList<Real> arms = A.slender(comm);
    const std::string tag = "vis/ybifurc-tree-ord" + std::to_string((long)ord) + "-nref" + std::to_string((long)nref);

    // ---- geometry report + collision guard (min clearance between non-adjacent tubes) ----
    if (!comm.Rank()) {
      const Long njp = junc.Size(), nap = arms.Size();
      Vector<Real> Xj, Xa; junc.GetNodeCoord(&Xj, nullptr, nullptr); arms.GetNodeCoord(&Xa, nullptr, nullptr);
      std::cout << "\n[geometry] junctions=" << NJ << " shared arms=" << (NJ-1) << " capped tips=" << n_caps
                << "\n  quad panels=" << njp << " nodes=" << Xj.Dim()/3
                << " | slender panels=" << nap << " nodes=" << Xa.Dim()/3 << "\n";
      // junction body radius proxy (scaled) for junction-vs-arm clearance
      auto Jorigin = [&](int i){ return P[i].apply_point(Vec3<Real>{0,0,0}); };
      const Real Jrad_can = (Real)1.3;   // canonical junction blob outer radius proxy
      Real minclear = std::numeric_limits<Real>::max(); int mi=-1, mj=-1;
      for (size_t a = 0; a < segs.size(); a++)
        for (size_t b = a+1; b < segs.size(); b++) {
          const bool adjacent = (segs[a].j0==segs[b].j0)||(segs[a].j0==segs[b].j1)||
                                (segs[a].j1==segs[b].j0)||(segs[a].j1==segs[b].j1);
          if (adjacent) continue;   // arms sharing a junction are meant to be near it
          const Real d = seg_seg_dist<Real>(segs[a].A, segs[a].B, segs[b].A, segs[b].B);
          const Real clear = d - (segs[a].rtube + segs[b].rtube);
          if (clear < minclear) { minclear = clear; mi=(int)a; mj=(int)b; }
        }
      // junction bodies vs non-incident arm segments
      Real minJclear = std::numeric_limits<Real>::max();
      for (int i = 0; i < NJ; i++) {
        const Vec3<Real> C = Jorigin(i); const Real Rb = nodes[i].scale*Jrad_can;
        for (const auto& s : segs) {
          if (s.j0==i || s.j1==i) continue;
          const Real d = seg_seg_dist<Real>(C, C, s.A, s.B);   // point-to-segment via degenerate seg
          minJclear = std::min(minJclear, d - (Rb + s.rtube));
        }
      }
      std::cout << "  [collision] min arm-arm clearance=" << std::setprecision(4) << minclear
                << " (segs " << mi << "," << mj << ")   min junction-arm clearance=" << minJclear << "\n";
      if (minclear <= 0 || minJclear <= 0)
        std::cout << "  *** WARNING: geometry self-intersection (clearance <= 0) -- adjust the tree spec ***\n";
    }

    junc.WriteVTK(tag + "-junc", Vector<Real>(), comm);
    arms.WriteVTK(tag + "-arms", Vector<Real>(), comm);

    // ---- exterior Green sources: one candidate per junction (canonical exterior point transformed by
    //      that junction's placement), kept only if exterior to the WHOLE assembly ----
    Vector<Real> X0;
    {
      const YField<Real> fld;
      auto exterior = [&](const Vec3<Real>& Xs) -> bool {
        for (int i = 0; i < NJ; i++) if (fld.f(P[i].apply_inverse_point(Xs)) >= level) return false;
        for (const auto& s : segs) if (seg_seg_dist<Real>(Xs, Xs, s.A, s.B) <= s.rtube) return false;
        return true;
      };
      for (int i = 0; i < NJ; i++) {
        const Vec3<Real> Xs = P[i].apply_point(Vec3<Real>{1.6, 1.4, 0.9});
        const bool ext = exterior(Xs);
        if (!comm.Rank())
          std::cout << "  [source] junction " << i << " -> (" << std::setprecision(4) << Xs[0] << ","
                    << Xs[1] << "," << Xs[2] << ")  " << (ext ? "exterior OK (kept)" : "inside/near (dropped)") << "\n";
        if (ext) { X0.PushBack(Xs[0]); X0.PushBack(Xs[1]); X0.PushBack(Xs[2]); }
      }
      SCTL_ASSERT_MSG(X0.Dim() > 0, "no exterior Green source survived validation.");
    }

    if (geomOnly) {
      if (!comm.Rank()) std::cout << "\n[geom-only] meshes written to " << tag << "-{junc,arms}.vtu; skipping BIE.\n";
      Comm::MPI_Finalize();
      return 0;
    }

    // ---- BIE verification: divergence + DL(=-1/2) watertightness + Green's identity, Laplace & Stokes ----
    const RegionReport<Real> region_report = [](const Vector<Real>& err, Long Nj, Long Na) {
      Real mj = 0, ma = 0;
      for (Long i = 0; i < Nj; i++) mj = std::max(mj, err[i]);
      for (Long i = 0; i < Na; i++) ma = std::max(ma, err[Nj+i]);
      std::cout << "    [region max] quad(junctions+transitions+caps)=" << mj << " slender(arms)=" << ma << "\n";
    };

    junc.SetQuadScheme(quad_junctions::QJDefaultScheme<Real>(), cov_q, Nbeta, maxdep);
    if (!comm.Rank())
      std::cout << "\n---- BIE verification [tol=" << tol << " Nbeta=" << Nbeta << " max_depth=" << maxdep
                << " cov_q=" << cov_q << "] ----\n";
    divergence_check<Real>(junc, arms, tol, comm);

    if (!comm.Rank()) std::cout << "  [DL const  Laplace] ";
    test_DLIdentity<Real, Laplace3D_DxU>(junc, arms, comm, tol, tag+"-dl-laplace", region_report);
    if (!comm.Rank()) std::cout << "  [DL const  Stokes ] ";
    test_DLIdentity<Real, Stokes3D_DxU>(junc, arms, comm, tol, tag+"-dl-stokes", region_report);
    if (!comm.Rank()) std::cout << "  [Green     Laplace] ";
    test_greens_identity<Real, Laplace3D_FxU, Laplace3D_DxU, Laplace3D_FxdU>(junc, arms, comm, tol, X0, tag+"-green-laplace");
    if (!comm.Rank()) std::cout << "  [Green     Stokes ] ";
    test_greens_identity<Real, Stokes3D_FxU, Stokes3D_DxU, Stokes3D_FxT>(junc, arms, comm, tol, X0, tag+"-green-stokes");
  }
  Comm::MPI_Finalize();
  return 0;
}
