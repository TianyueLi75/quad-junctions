/**
 * probe-fmm-vs-direct: isolate WHERE PVFMM's far field diverges from direct summation.
 *
 * Context: on the 20-junction vessels network the DL constant-density identity is correct under
 * PVFMM at quad order 8 (2.16e-3, matching direct summation) but catastrophically wrong at order 12
 * (118) and order 16 (1.4e6) -- same geometry, same quadrature tier, scale-invariant, and unaffected
 * by the CSBQ arm resolution. This probe skips BoundaryIntegralOp entirely and drives sctl::ParticleFMM
 * directly on the SAME far-field source/target set, so we can diff EvalPVFMM against EvalDirect
 * pointwise and see which interactions are wrong rather than only the aggregate identity error.
 *
 * It answers three things the aggregate error cannot:
 *   1. Is the far field wrong everywhere, or only for a subset of targets? (-> print the error
 *      distribution, and the coordinates/bbox-cell of the worst offenders)
 *   2. Does the error correlate with source/target separation? (-> a broken translation operator
 *      shows up at LARGE separation; a near/far bookkeeping error shows up at SMALL separation)
 *   3. Does it depend on the number of nodes per panel (order) at fixed geometry?
 *
 * Sources/targets are taken from a real element list via GetFarFieldNodes, so the point
 * distribution (clustered Gauss-Legendre nodes on high-order panels) is faithful -- that clustering
 * is the prime suspect.
 *
 * GEOMETRY (argv[5]): 0 = the 20-junction vessels network (default; the geometry the bug was found
 * on), 1 = the SINGLE canonical Y-junction from ybifurc-hybrid with full-quad TUBE arms
 * (arm_kind=1). Geometry 1 is the control: like the cilia/stud_sphere case that PVFMM handles
 * correctly, it is one compact pure-quad QuadElemList surface with no CSBQ slender elements, so the
 * solver sees the same kind of object in both. If order 12 explodes here too, the trigger is quad
 * order / node clustering per se; if it stays clean, the trigger is something specific to the
 * extended vessels network (its ~40x20x0.5 planar extent, thin tubes, or 20-junction assembly).
 *
 *   make bin/probe-fmm-vs-direct
 *   OMP_NUM_THREADS=4 mpirun -n 1 ./bin/probe-fmm-vs-direct [order] [nref] [fourier] [digits] [geom]
 *       [ncopy] [layout] [pitch] [gscale] [Ns_trans]
 *
 * QJ_PROBE_STOKES=1 adds Stokes DxU/FxU rows (the kernels the flow BVP actually solves; every earlier
 * table here is Laplace-only). QJ_PROBE_TRG_STRIDE=<k> probes every k-th node as a target while keeping
 * all sources -- EvalDirect is O(Ns*Nt), so this is what makes the 9x-costlier Stokes rows affordable.
 * Keep the resulting global target count >= 40000 (asserted) or EvalPVFMM silently goes direct.
 *
 * QJ_DUMP_QUAD=<path> writes the probed surface via QuadElemList::Write (17-digit ASCII = exact for
 * double), same oracle as the assembly drivers. Geometry 1 is two element lists (junction + arms),
 * so it writes <path> and <path>.arms; geometry 0 writes the single merged <path>.
 */
#include <csbq.hpp>
#include <quad_junctions/vessels_build.hpp>
#include <quad_junctions/vessels_tree_data.hpp>
#include <quad_junctions/ybifurc_hybrid_geom.hpp>
#include <quad_junctions/fmm_kernels.hpp>
#include <quad_junctions/plane_cilia_geom.hpp>   // geoms 4/5: add_cubedsphere, orient_group_flat, flip_group
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstdio>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <vector>

using namespace sctl;
using namespace quad_junctions;

namespace {

// Pointwise FMM-vs-direct comparison for one kernel on a given source/target set.
template <class Real, class Ker> void probe(const Vector<Real>& Xs, const Vector<Real>& Xn,
                                            const Vector<Real>& Xt, const Integer digits,
                                            const std::string& name, const Comm& comm,
                                            Vector<Long>* worst_out = nullptr) {
  static constexpr Integer DIM = 3;
  const Ker ker;
  const Long Ns = Xs.Dim()/DIM, Nt = Xt.Dim()/DIM;
  const Integer SrcDim = ker.SrcDim(), TrgDim = ker.TrgDim();

  // Constant unit density -- the same input the DL identity uses.
  Vector<Real> F(Ns*SrcDim); F = 1;

  ParticleFMM<Real,DIM> fmm(comm);
  fmm.SetAccuracy(digits);
  // Same PVFMM-safe translation kernels the drivers install (fmm_kernels.hpp / StokesBIO).
  const typename FMMTransKer<Ker>::M2M m2m;
  const typename FMMTransKer<Ker>::L2L l2l;
  fmm.SetKernels(m2m, m2m, l2l);
  fmm.AddSrc("S", ker, ker);
  fmm.AddTrg("T", m2m, l2l);
  fmm.SetKernelS2T("S", "T", ker);
  fmm.SetSrcCoord("S", Xs, Xn);
  fmm.SetSrcDensity("S", F);
  fmm.SetTrgCoord("T", Xt);

  Vector<Real> Ufmm, Udir;
  fmm.Eval(Ufmm, "T");        // EvalPVFMM (>= 40000 global targets, else silently direct)
  fmm.EvalDirect(Udir, "T");

  // Error distribution + worst offenders.
  const Long N = std::min<Long>(Ufmm.Dim(), Udir.Dim());
  Real umax = 0;
  for (Long i = 0; i < N; i++) umax = std::max<Real>(umax, fabs(Udir[i]));
  std::vector<std::pair<Real,Long>> err(Nt, {0,0});
  for (Long i = 0; i < Nt; i++) {
    Real e = 0;
    for (Integer k = 0; k < TrgDim; k++) {
      const Long j = i*TrgDim + k;
      if (j < N) e = std::max<Real>(e, fabs(Ufmm[j]-Udir[j]));
    }
    err[i] = {e, i};
  }
  std::sort(err.begin(), err.end(), [](const std::pair<Real,Long>& a, const std::pair<Real,Long>& b){ return a.first > b.first; });

  // Percentiles of the relative error tell us "everywhere" vs "a few points".
  auto pct = [&](double p) { return Nt ? err[(Long)((1.0-p)*(Nt-1))].first / (umax>0?umax:1) : 0; };
  if (!comm.Rank()) {
    std::cout << std::scientific << std::setprecision(3)
              << "  [" << name << "] Ns=" << Ns << " Nt=" << Nt << " |U|max=" << umax << "\n"
              << "      rel err: max=" << pct(1.0) << "  p99=" << pct(0.99)
              << "  p50=" << pct(0.50) << "  p01=" << pct(0.01) << "\n";
  }

  if (worst_out) {   // hand the worst targets back so the caller can locate them ON the geometry
    const Long nw = std::min<Long>(10, Nt);
    worst_out->ReInit(nw);
    for (Long q = 0; q < nw; q++) (*worst_out)[q] = err[q].second;
  }

  // For the worst targets: distance to the NEAREST source. A broken translation operator hurts
  // well-separated targets; a near/far split error hurts targets sitting close to a source.
  const Long nshow = std::min<Long>(5, Nt);
  for (Long q = 0; q < nshow; q++) {
    const Long i = err[q].second;
    Real dmin = 1e30;
    for (Long j = 0; j < Ns; j++) {
      Real d2 = 0;
      for (Integer k = 0; k < DIM; k++) { const Real d = Xt[i*DIM+k]-Xs[j*DIM+k]; d2 += d*d; }
      dmin = std::min<Real>(dmin, d2);
    }
    dmin = sqrt<Real>(dmin);
    if (!comm.Rank())
      std::cout << "      #" << q << " abs_err=" << err[q].first
                << " at (" << Xt[i*DIM] << "," << Xt[i*DIM+1] << "," << Xt[i*DIM+2] << ")"
                << "  nearest-source dist=" << dmin << "\n";
  }
}

// Locate the worst targets ON the geometry: which element, where inside that element, which region
// (junction body/transition vs root cap), and how close they sit to the slender seam. Answers
// "near the connection to the slender arms, or the middle of a quad patch? junction or cap?".
//
// Element indexing: GetFarFieldNodes emits exactly order^2 nodes per element in element order, and
// build_vessels_network emits ALL njunc junctions first (pass 1: add_junction -> junction sphere +
// POU transition tubes), then the bent arms (slender only, no quad), then the root caps (pass 4:
// add_free_arm -> slender fiber + hemisphere cap into Xquad_). So quad element e < njunc*npj belongs
// to junction e/npj, and e >= njunc*npj is cap material.
template <class Real> void classify_worst(const Vector<Long>& widx, const Vector<Real>& Xprobed,
                                          const Vector<Real>& Xq, const Vector<Real>& Xsl,
                                          Long nq_probed, Integer ord, Long npj, Long njunc,
                                          const std::vector<Vec3<Real>>& jcen, const Comm& comm) {
  static constexpr Integer DIM = 3;
  if (comm.Rank()) return;
  const Long npe = (Long)ord*ord;                  // nodes per quad element
  const Long nquad_elem = Xq.Dim()/DIM/npe;
  std::cout << "      --- where on the geometry (npj=" << npj << " panels/junction, "
            << njunc << " junctions -> quad elems 0.." << njunc*npj-1 << " are junction+transition, "
            << njunc*npj << ".." << nquad_elem-1 << " are root caps) ---\n";
  for (Long q = 0; q < widx.Dim() && q < 5; q++) {
    const Long i = widx[q];
    const bool is_quad = (i*DIM < nq_probed);
    std::cout << "      #" << q << " " << (is_quad ? "QUAD" : "SLENDER");
    if (!is_quad) { std::cout << " node (slender arm)\n"; continue; }
    const Long e = i/npe, loc = i%npe, iu = loc/ord, iv = loc%ord;
    // Panel size, from the element's own node bbox.
    Real lo[DIM], hi[DIM];
    for (Integer k = 0; k < DIM; k++) { lo[k] = Xq[e*npe*DIM+k]; hi[k] = lo[k]; }
    for (Long n = 0; n < npe; n++)
      for (Integer k = 0; k < DIM; k++) {
        const Real v = Xq[(e*npe+n)*DIM+k];
        lo[k] = std::min<Real>(lo[k], v); hi[k] = std::max<Real>(hi[k], v);
      }
    Real diag = 0; for (Integer k = 0; k < DIM; k++) diag += (hi[k]-lo[k])*(hi[k]-lo[k]);
    diag = sqrt<Real>(diag);
    // Nearest node in a DIFFERENT element (detects near-touching / interpenetrating panels) and
    // nearest slender node (detects "sitting on the quad<->slender seam").
    Real d_other = 1e30, d_slen = 1e30;
    for (Long j = 0; j < Xq.Dim()/DIM; j++) {
      if (j/npe == e) continue;
      Real d2 = 0; for (Integer k = 0; k < DIM; k++) { const Real d = Xprobed[i*DIM+k]-Xq[j*DIM+k]; d2 += d*d; }
      d_other = std::min<Real>(d_other, d2);
    }
    for (Long j = 0; j < Xsl.Dim()/DIM; j++) {
      Real d2 = 0; for (Integer k = 0; k < DIM; k++) { const Real d = Xprobed[i*DIM+k]-Xsl[j*DIM+k]; d2 += d*d; }
      d_slen = std::min<Real>(d_slen, d2);
    }
    d_other = sqrt<Real>(d_other); d_slen = sqrt<Real>(d_slen);
    const bool edge = (iu == 0 || iu == ord-1 || iv == 0 || iv == ord-1);
    std::cout << " elem=" << e;
    if (e < njunc*npj) std::cout << " [JUNCTION " << e/npj << ", its panel " << e%npj << "]";
    else               std::cout << " [ROOT CAP, cap panel " << e-njunc*npj << "]";
    std::cout << " node(" << iu << "," << iv << ")/" << ord << (edge ? " PANEL-EDGE" : " panel-interior") << "\n"
              << "         panel diag=" << diag << "  dist to other-elem node=" << d_other
              << "  dist to SLENDER=" << d_slen;
    if (e < njunc*npj && (Long)jcen.size() > e/npj) {
      const Vec3<Real>& c = jcen[e/npj];
      Real d2 = 0; for (Integer k = 0; k < DIM; k++) { const Real d = Xprobed[i*DIM+k]-c[k]; d2 += d*d; }
      std::cout << "  dist to junction centre=" << sqrt<Real>(d2);
    }
    std::cout << "\n";
  }
}

// Bounding box + aspect ratio of a point set. This is the quantity under test for the "flat bbox"
// hypothesis: SCTL normalizes into [0,1]^3 with bbox_scale = 1/bbox_len, i.e. by the LONGEST side, so
// a planar geometry enters the octree as a thin slab and its occupancy is wildly imbalanced. Uniform
// rescaling of the geometry cannot change this number -- which is why the earlier gscale 1-vs-0.02
// test could not have detected it.
template <class Real> void report_bbox(const Vector<Real>& X, const Comm& comm) {
  static constexpr Integer DIM = 3;
  const Long N = X.Dim()/DIM;
  if (!N) return;
  Real lo[DIM], hi[DIM];
  for (Integer k = 0; k < DIM; k++) { lo[k] = X[k]; hi[k] = X[k]; }
  for (Long i = 0; i < N; i++)
    for (Integer k = 0; k < DIM; k++) {
      lo[k] = std::min<Real>(lo[k], X[i*DIM+k]);
      hi[k] = std::max<Real>(hi[k], X[i*DIM+k]);
    }
  Real ext[DIM], emax = 0, emin = 1e30;
  for (Integer k = 0; k < DIM; k++) { ext[k] = hi[k]-lo[k]; emax = std::max<Real>(emax, ext[k]); emin = std::min<Real>(emin, ext[k]); }
  if (!comm.Rank()) {
    std::cout << std::fixed << std::setprecision(3)
              << "  bbox extent: " << ext[0] << " x " << ext[1] << " x " << ext[2]
              << "   ASPECT (max/min) = " << std::setprecision(1) << (emin>0 ? emax/emin : 0) << "\n"
              << "  normalized into the unit cube (bbox_scale=1/" << std::setprecision(3) << emax << "): "
              << ext[0]/emax << " x " << ext[1]/emax << " x " << ext[2]/emax << "\n" << std::scientific;
  }
}

// Replicate a far-field node set on a lattice. Normals are translation-invariant, and probe() uses a
// constant unit density, so K translated copies are exactly the point set BoundaryIntegralOp would
// hand the FMM for K disjoint copies of the surface. Holding `pitch` fixed between layouts keeps all
// LOCAL structure (node spacing, nearest-neighbour distances, per-copy panel geometry) identical, so
// planar-vs-compact differs ONLY in the global bounding box -- the whole point of the experiment.
template <class Real> void replicate(Vector<Real>& X, Vector<Real>& Xn, Integer K, Integer layout,
                                     Real pitch, const Comm& comm) {
  static constexpr Integer DIM = 3;
  if (K <= 1) return;
  std::vector<std::array<Real,DIM>> off;
  if (layout == 0) {                                  // PLANAR: nx x ny x 1 -> slab bbox
    const Integer nx = (Integer)std::ceil(std::sqrt((double)K));
    for (Integer iy = 0; off.size() < (size_t)K; iy++)
      for (Integer ix = 0; ix < nx && off.size() < (size_t)K; ix++)
        off.push_back({ix*pitch, iy*pitch, (Real)0});
  } else {                                            // COMPACT: n x n x n -> ~isotropic bbox
    const Integer n = (Integer)std::ceil(std::cbrt((double)K));
    for (Integer iz = 0; off.size() < (size_t)K; iz++)
      for (Integer iy = 0; iy < n && off.size() < (size_t)K; iy++)
        for (Integer ix = 0; ix < n && off.size() < (size_t)K; ix++)
          off.push_back({ix*pitch, iy*pitch, iz*pitch});
  }
  const Long n0 = X.Dim();
  Vector<Real> X2(n0*K), Xn2(n0*K);
  for (Integer c = 0; c < K; c++)
    for (Long i = 0; i < n0/DIM; i++)
      for (Integer k = 0; k < DIM; k++) {
        X2 [c*n0 + i*DIM+k] = X [i*DIM+k] + off[c][k];
        Xn2[c*n0 + i*DIM+k] = Xn[i*DIM+k];             // translation leaves normals alone
      }
  X.Swap(X2); Xn.Swap(Xn2);
  if (!comm.Rank())
    std::cout << "  replicated x" << K << (layout==0 ? " PLANAR (nx x ny x 1)" : " COMPACT (n x n x n)")
              << " pitch=" << std::fixed << std::setprecision(2) << pitch << std::scientific << "\n";
}

// ---- geoms 4/5: the periodic-sphere test surfaces (copied from periodic-sphere{,-csbq}-bie.cpp) ----
// A single flat order x order patch spanning [0,L]x[0,L] at z=z_plane, normal along (0,0,uz).
template <class Real> void ps_add_plate(Vector<Real>& Xall, Integer order, Real L, Real z_plane, Real uz) {
  const Vector<Real>& nds = QuadElemList<Real>::ParamNodes(order);
  Vector<Real> Xp;
  for (Integer i = 0; i < order; i++) { const Real yy = nds[i] * L;
    for (Integer j = 0; j < order; j++) { const Real xx = nds[j] * L;
      Xp.PushBack(xx); Xp.PushBack(yy); Xp.PushBack(z_plane); } }
  orient_group_flat<Real>(Xp, order, z_plane, uz);
  for (auto v : Xp) Xall.PushBack(v);
}
// A cubed sphere of radius R centered at c, normals toward c (out of fluid). (== periodic-sphere-bie)
template <class Real> void ps_add_obstacle_sphere(Vector<Real>& Xall, Integer order, Long PatchPerFace,
                                                  Real R, const Real c[3]) {
  Vector<Real> Xs;
  add_cubedsphere<Real>(Xs, order, PatchPerFace, R, /*skipFace=*/-1, 0, 0);
  { QuadElemList<Real> tmp(order, Xs);
    Vector<Real> Xc, Xnc; tmp.GetNodeCoord(&Xc, &Xnc, nullptr);
    Real acc = 0;
    for (Long i = 0; i < Xc.Dim()/3; i++) for (int k = 0; k < 3; k++) acc += Xnc[i*3+k]*Xc[i*3+k];
    if (acc > 0) flip_group<Real>(Xs, order); }
  for (Long i = 0; i < Xs.Dim()/3; i++) { Xs[i*3+0] += c[0]; Xs[i*3+1] += c[1]; Xs[i*3+2] += c[2]; }
  for (auto v : Xs) Xall.PushBack(v);
}
// CSBQ slender sphere: centerline x = c[0]+R cos(theta), radius R sin(theta), theta in [0,pi] over Nelem
// panels, azimuthal orient (0,1,0). (== periodic-sphere-csbq-bie's build_csbq_sphere; serial build here.)
template <class Real> SlenderElemList<Real> ps_build_csbq_sphere(const Real c[3], Real R, Long Nelem,
    Long ElemOrder, Long FourierOrder) {
  Vector<Long> cheb_order, forder;
  Vector<Real> coord, radius, orient;
  for (Long i = 0; i < Nelem; i++) {
    cheb_order.PushBack(ElemOrder); forder.PushBack(FourierOrder);
    const Vector<Real>& nds = SlenderElemList<Real>::CenterlineNodes(ElemOrder);
    for (Long j = 0; j < ElemOrder; j++) {
      const Real theta = const_pi<Real>() * ((Real)i + nds[j]) / (Real)Nelem;
      coord.PushBack(c[0] + R*cos<Real>(theta)); coord.PushBack(c[1]); coord.PushBack(c[2]);
      radius.PushBack(R*sin<Real>(theta));
      orient.PushBack((Real)0); orient.PushBack((Real)1); orient.PushBack((Real)0);
    }
  }
  return SlenderElemList<Real>(cheb_order, forder, coord, radius, orient);
}

}  // namespace

int main(int argc, char** argv) {
  Comm::MPI_Init(&argc, &argv);
  {
    using Real = double;
    const Comm comm = Comm::World();
    const Integer ord     = (argc > 1) ? (Integer)atoi(argv[1]) : 12;
    const Integer nref    = (argc > 2) ? (Integer)atoi(argv[2]) : 1;
    const Long    fourier = (argc > 3) ? (Long)atoi(argv[3]) : 12;
    const Integer digits  = (argc > 4) ? (Integer)atoi(argv[4]) : 6;
    const Integer geom    = (argc > 5) ? (Integer)atoi(argv[5]) : 0;   // 0=vessels, 1=ybifurc, 2=ybifurc xK
    const Integer ncopy   = (argc > 6) ? (Integer)atoi(argv[6]) : 8;   // geom 2: number of copies
    const Integer layout  = (argc > 7) ? (Integer)atoi(argv[7]) : 0;   // geom 2: 0=planar slab, 1=compact cube
    const Real    pitch   = (argc > 8) ? (Real)atof(argv[8]) : (Real)8;// geom 2: lattice spacing
    // geom 0: GLOBAL SIMILARITY factor on the whole network (positions AND junction sizes). gscale=2
    // makes the gen-3 junctions (8, 9 -- where the worst FMM errors sit, at 0.8^3 = 0.512) as large in
    // ABSOLUTE terms as the gen-0 root is at gscale=1. Tests "the elements are too small / the nodes
    // are crammed together". Relative sizes within the network are untouched by construction.
    const Real    gscale  = (argc > 9) ? (Real)atof(argv[9]) : (Real)1;
    // Ns_trans (geom 0 only). Was hardcoded to 3; the flow driver's mesh-reduction runs use 2, and the
    // probe is only a faithful reproducer of the failing operator if it builds the SAME mesh.
    const Integer NsTr    = (argc > 10) ? (Integer)atoi(argv[10]) : 3;

    static const char* gname[6] = {"(vessels-network)", "(ybifurc-junction-quadtube)", "(ybifurc xK lattice)",
                                   "(lens racetrack)", "(periodic-sphere full-quad)", "(periodic-sphere hybrid CSBQ)"};
    if (!comm.Rank()) {
      std::cout << "=== FMM vs direct probe: order=" << ord << " nref=" << nref
                << " fourier=" << fourier << " digits=" << digits
                << " geom=" << geom << gname[(geom<0||geom>5) ? 0 : geom];
      if (geom == 2) std::cout << " ncopy=" << ncopy << " layout=" << (layout==0?"PLANAR":"COMPACT")
                               << " pitch=" << pitch;
      if (geom == 0) std::cout << " gscale=" << gscale;
      std::cout << " ===\n";
    }

    const char* dump = std::getenv("QJ_DUMP_QUAD");

    // FAR-FIELD nodes: exactly the point set BoundaryIntegralOp hands to the FMM, including the
    // upsampling, so the clustering of high-order GL nodes is reproduced faithfully. For geometry 1
    // the surface is two element lists, and BoundaryIntegralOp's FMM sees the UNION of their
    // far-field nodes -- so we concatenate, which is the faithful analogue of the merged geometry 0.
    Vector<Real> X, Xn, wts, dist_far;
    Vector<Long> elem_cnt;

    // Kept for classify_worst(): the quad/slender node arrays, how many quad coords lead the probed
    // set, panels-per-junction, and the junction centres.
    Vector<Real> Xq_keep, Xsl_keep; Long nq_probed = 0, npj = 0, njunc = 0;
    std::vector<Vec3<Real>> jcen;
    std::vector<Placement<Real>> jP;   // full similarity per junction (t, R, scale) for the node dump

    if (geom == 0) {  // vessels
      // Same vessels geometry the identity tests use. Like geom 3, `layout` selects which element
      // type is probed: 0=union (what the coupled BoundaryIntegralOp feeds the FMM), 1=quad only
      // (the 20 junction/transition/cap blobs -- this is what geom 0 probed exclusively before, and
      // the configuration that blows up), 2=slender only (the bent/racetrack arms).
      HybridAssembly<Real> A(ord);
      const auto vb = build_vessels_network<Real>(A, (Real)1.5, nref, (Real)0.4, NsTr, fourier, 1, 12, (Real)0.06,
                                  (Integer)(YSwept::Ncap0*std::max<Integer>(1,nref)), 10, 3, (Real)1.0, comm, gscale);
      for (const auto& p : vb.P) { jcen.push_back(p.t); jP.push_back(p); }
      njunc = (Long)vb.P.size();
      { // panels emitted by ONE add_junction (junction sphere + POU transitions, no cap): the
        // canonical mesh is memoized, so this extra build is essentially free.
        HybridAssembly<Real> A1(ord);
        A1.add_junction(vb.P[0], (Real)1.5, nref, (Real)0.4, NsTr, comm);
        npj = A1.quad(comm).Size();
      }
      QuadElemList<Real> junc = A.quad(comm);   // honors QJ_DUMP_QUAD itself
      junc.SetQuadScheme(QuadElemList<Real>::QuadScheme::Hybrid, 6, 48, 4);
      SlenderElemList<Real> arms = A.slender(comm);

      Vector<Real> Xq, Xnq, wq, dq, Xs_, Xns, ws, ds; Vector<Long> cq, cs;
      junc.GetFarFieldNodes(Xq, Xnq, wq, dq, cq, 1);
      arms.GetFarFieldNodes(Xs_, Xns, ws, ds, cs, 1);
      if (!comm.Rank())
        std::cout << "  vessels: quad panels=" << junc.Size() << " slender panels=" << arms.Size() << "\n"
                  << "  far-field nodes: quad=" << Xq.Dim()/3 << " slender=" << Xs_.Dim()/3 << "\n";
      const Long nq = Xq.Dim(), nsl = Xs_.Dim();
      Xq_keep = Xq; Xsl_keep = Xs_;
      if (layout == 1) { nq_probed = nq; X.Swap(Xq); Xn.Swap(Xnq); if (!comm.Rank()) std::cout << "  probing QUAD ONLY\n"; }
      else if (layout == 2) { nq_probed = 0; X.Swap(Xs_); Xn.Swap(Xns); if (!comm.Rank()) std::cout << "  probing SLENDER ONLY\n"; }
      else {
        nq_probed = nq;
        Vector<Real> T(nq+nsl), Tn(nq+nsl);
        for (Long i=0;i<nq; i++) { T[i]=Xq[i];     Tn[i]=Xnq[i]; }
        for (Long i=0;i<nsl;i++) { T[nq+i]=Xs_[i]; Tn[nq+i]=Xns[i]; }
        X.Swap(T); Xn.Swap(Tn);
        if (!comm.Rank()) std::cout << "  probing UNION (quad + slender)\n";
      }
    } else if (geom == 1 || geom == 2) {
      // ONE canonical Y-junction + full-quad tube arms -- the same pair ybifurc-hybrid-bie builds
      // for arm_kind=1, with its defaults (level 1.5, eta_join 0.4, Ns_trans 3, s_cap 0.88,
      // n_axial 3), so this probes the identical surface those DL/Green sweeps run on.
      Real R0[3], a0[3], sc[3];
      const Integer Ncap = (Integer)(YSwept::Ncap0*std::max<Integer>(1,nref));
      QuadElemList<Real> junc = BuildYJunctionWithTransitions<Real>(
          ord, (Real)1.5, nref, (Real)0.4, 3, (Real)0.88, R0, a0, sc, Ncap, nullptr, comm);
      const Integer NsShaft = std::max<Integer>(1, (Integer)std::lround(3*10.0/(double)ord));
      const Integer NaShaft = (Integer)(YSwept::Na0 * nref);
      QuadElemList<Real> arms = BuildYArmsQuadTube<Real>(ord, R0, a0, sc, NsShaft, NaShaft, comm);
      junc.SetQuadScheme(QuadElemList<Real>::QuadScheme::Hybrid, 6, 48, 4);
      arms.SetQuadScheme(QuadElemList<Real>::QuadScheme::Hybrid, 6, 48, 4);
      if (dump) {                               // built directly, not via HybridAssembly::quad()
        junc.Write(std::string(dump), comm);                 // collective; rank 0 writes
        arms.Write(std::string(dump) + ".arms", comm);
        if (!comm.Rank()) std::cout << "  [dump] wrote " << dump << " and " << dump << ".arms\n";
      }

      Vector<Real> Xa, Xna, wa, da; Vector<Long> ca;
      junc.GetFarFieldNodes(X,  Xn,  wts, dist_far, elem_cnt, 1);
      arms.GetFarFieldNodes(Xa, Xna, wa,  da,       ca,       1);
      const Long n0 = X.Dim(), na = Xa.Dim();
      { Vector<Real> T(n0+na); for (Long i=0;i<n0;i++) T[i]=X[i];  for (Long i=0;i<na;i++) T[n0+i]=Xa[i];  X.Swap(T); }
      { Vector<Real> T(n0+na); for (Long i=0;i<n0;i++) T[i]=Xn[i]; for (Long i=0;i<na;i++) T[n0+i]=Xna[i]; Xn.Swap(T); }
      if (!comm.Rank())
        std::cout << "  panels: junc=" << junc.Size() << " arms=" << arms.Size()
                  << " (Ns_shaft=" << NsShaft << " Na_shaft=" << NaShaft << ")\n"
                  << "  far-field nodes: junc=" << n0/3 << " arms=" << na/3 << "\n";
    }
    // geom 2: same junction, K copies on a lattice. Isolates bounding-box aspect ratio -- order,
    // panel count per copy, element type and local node spacing are all identical between layouts.
    if (geom == 2) replicate<Real>(X, Xn, ncopy, layout, pitch, comm);

    // geom 3: the ybifurc-channel-bie LENS racetrack -- two Y-junctions facing each other, their
    // branch pairs joined by two bent (racetrack) slender arms, stems free/capped. Unlike geoms 0-2
    // this probes BOTH element types: the quad junctions AND the CSBQ slender arms, which is what
    // BoundaryIntegralOp's FMM actually sees. `layout` selects which part: 0=union (default),
    // 1=quad only, 2=slender only -- so a blow-up can be attributed to one or the other.
    if (geom == 3) {
      pou_kind() = 1;                                  // smootherstep POU, as ybifurc-channel-bie sets
      const Real level = 1.5, etajoin = 0.4, s_cap = 0.88, sep = 9.6;
      const Integer NsTrans = 3, nAxFree = 3, leadP = 2, cornerP = 6;
      const Integer Ncap = (Integer)(YSwept::Ncap0 * nref);
      const Long cheb = 10;
      const Vec3<Real> up{0,0,1};
      HybridAssembly<Real> A(ord);
      // Both junctions rotated only about z -> the assembly stays planar in z=0, which is
      // add_bent_arm's planar-turn requirement.
      const auto PA = Placement<Real>::AlignArm(0, Vec3<Real>{-1,0,0}, up, Vec3<Real>{-sep/2,0,0});
      const auto PB = Placement<Real>::AlignArm(0, Vec3<Real>{ 1,0,0}, up, Vec3<Real>{ sep/2,0,0});
      const HybridJunction<Real> JA = A.add_junction(PA, level, nref, etajoin, NsTrans, comm);
      const HybridJunction<Real> JB = A.add_junction(PB, level, nref, etajoin, NsTrans, comm);
      const std::pair<ArmSeam<Real>,ArmSeam<Real>> bent[2] = {{JA.seam(2), JB.seam(1)},   // top wall
                                                              {JA.seam(1), JB.seam(2)}};  // bottom wall
      for (const auto& pr : bent) {
        const ArmSeam<Real>& sa = pr.first; const ArmSeam<Real>& sb = pr.second;
        const Real dx=sb.C[0]-sa.C[0], dy=sb.C[1]-sa.C[1], dz=sb.C[2]-sa.C[2];
        const Real len = std::sqrt(dx*dx+dy*dy+dz*dz);
        const Real pspac = (Real)1.5*std::max(sa.R0, sb.R0);         // same spacing rule as the driver
        const Integer nmin = 2*(leadP+cornerP) + 4;
        const Integer ns = std::max<Integer>(nmin, (Integer)std::lround((double)len/pspac));
        A.add_bent_arm(sa, sb, ns, cheb, fourier, leadP, cornerP, /*single_corner=*/false);
      }
      A.add_free_arm(JA.seam(0), s_cap, nAxFree, Ncap, cheb, fourier);   // inlet
      A.add_free_arm(JB.seam(0), s_cap, nAxFree, Ncap, cheb, fourier);   // outlet

      QuadElemList<Real> q = A.quad(comm);        // honors QJ_DUMP_QUAD
      SlenderElemList<Real> s = A.slender(comm);
      q.SetQuadScheme(QuadElemList<Real>::QuadScheme::Hybrid, 6, 48, 4);

      Vector<Real> Xq, Xnq, wq, dq, Xs_, Xns, ws, ds; Vector<Long> cq, cs;
      q.GetFarFieldNodes(Xq, Xnq, wq, dq, cq, 1);
      s.GetFarFieldNodes(Xs_, Xns, ws, ds, cs, 1);
      if (!comm.Rank())
        std::cout << "  lens: quad panels=" << q.Size() << " slender panels=" << s.Size() << "\n"
                  << "  far-field nodes: quad=" << Xq.Dim()/3 << " slender=" << Xs_.Dim()/3 << "\n";
      const Long nq = Xq.Dim(), nsl = Xs_.Dim();
      if (layout == 1) { X.Swap(Xq); Xn.Swap(Xnq); if (!comm.Rank()) std::cout << "  probing QUAD ONLY\n"; }
      else if (layout == 2) { X.Swap(Xs_); Xn.Swap(Xns); if (!comm.Rank()) std::cout << "  probing SLENDER ONLY\n"; }
      else {                                     // union: what the coupled BoundaryIntegralOp feeds the FMM
        Vector<Real> T(nq+nsl), Tn(nq+nsl);
        for (Long i=0;i<nq; i++) { T[i]=Xq[i];      Tn[i]=Xnq[i]; }
        for (Long i=0;i<nsl;i++) { T[nq+i]=Xs_[i];  Tn[nq+i]=Xns[i]; }
        X.Swap(T); Xn.Swap(Tn);
        if (!comm.Rank()) std::cout << "  probing UNION (quad + slender)\n";
      }
    }

    // geoms 4/5: the periodic-sphere test surfaces (the geometries the flow drivers actually run). These
    // are compact and near-isotropic (unlike the planar vessels network), so they are the control for
    // "is PVFMM's far field correct on the hybrid quad+slender mix at all". geom 4 = the all-quad cubed
    // sphere between two plates (periodic-sphere-bie); geom 5 = the SAME plates with the sphere replaced by
    // a CSBQ SlenderElemList (periodic-sphere-csbq-bie). NB these run FREE-SPACE (no periodicity) at whatever
    // node count -- pair with SCTL_FMM_FORCE_TREE=1 so EvalPVFMM forms the tree below the 40000 cutoff.
    if (geom == 4 || geom == 5) {
      const Real L = 1.0, R = 0.25, z_bottom = 0.01, z_top = 0.99;
      const Real c[3] = {(Real)0.5*L, (Real)0.5*L, (Real)0.5*(z_bottom+z_top)};
      const char* ppf_env = std::getenv("QJ_PPF");
      const Long PatchPerFace = ppf_env ? std::max<Long>(1, atol(ppf_env)) : 8;
      Vector<Real> Xall;
      ps_add_plate<Real>(Xall, ord, L, z_bottom, -1);
      ps_add_plate<Real>(Xall, ord, L, z_top,    +1);
      if (geom == 4) {
        ps_add_obstacle_sphere<Real>(Xall, ord, PatchPerFace, R, c);
        QuadElemList<Real> surf(ord, Xall, comm);
        surf.SetQuadScheme(QuadElemList<Real>::QuadScheme::Hybrid, 10, 100, 8);
        surf.GetFarFieldNodes(X, Xn, wts, dist_far, elem_cnt, 1);
        if (!comm.Rank())
          std::cout << "  periodic-sphere FULL-QUAD: order=" << ord << " PatchPerFace=" << PatchPerFace
                    << " R=" << R << "   far-field nodes=" << X.Dim()/3 << "\n";
      } else {
        QuadElemList<Real> plate(ord, Xall, comm);
        plate.SetQuadScheme(QuadElemList<Real>::QuadScheme::Hybrid, 10, 100, 8);
        const char* ne_env2 = std::getenv("QJ_SPH_NELEM");
        const Long Nelem = ne_env2 ? std::max<Long>(1, atol(ne_env2)) : 2;
        SlenderElemList<Real> sphere = ps_build_csbq_sphere<Real>(c, R, Nelem, /*ElemOrder=*/10, fourier);
        Vector<Real> Xp, Xnp, wp, dp, Xs_, Xns, ws, ds; Vector<Long> cp, cs;
        plate.GetFarFieldNodes(Xp, Xnp, wp, dp, cp, 1);
        sphere.GetFarFieldNodes(Xs_, Xns, ws, ds, cs, 1);
        const Long nq = Xp.Dim(), nsl = Xs_.Dim();
        Vector<Real> T(nq+nsl), Tn(nq+nsl);
        for (Long i=0;i<nq; i++) { T[i]=Xp[i];      Tn[i]=Xnp[i]; }
        for (Long i=0;i<nsl;i++) { T[nq+i]=Xs_[i];  Tn[nq+i]=Xns[i]; }
        X.Swap(T); Xn.Swap(Tn);
        if (!comm.Rank())
          std::cout << "  periodic-sphere HYBRID: order=" << ord << " (plates) + CSBQ sphere Nelem=" << Nelem
                    << " ElemOrder=10 fourier=" << fourier << "\n"
                    << "  far-field nodes: quad=" << nq/3 << " slender=" << nsl/3 << " total=" << X.Dim()/3 << "\n";
      }
    }

    const char* force_tree_env = std::getenv("SCTL_FMM_FORCE_TREE");
    const bool force_tree = force_tree_env && atoi(force_tree_env);
    if (!comm.Rank()) {
      std::cout << "  far-field nodes: " << X.Dim()/3 << (X.Dim()/3 >= 40000
                   ? "  (>= 40000 -> EvalPVFMM really runs)"
                   : (force_tree ? "  (< 40000 but SCTL_FMM_FORCE_TREE=1 -> EvalPVFMM forced)"
                                 : "  *** < 40000: EvalPVFMM SILENTLY FALLS BACK TO DIRECT -- set SCTL_FMM_FORCE_TREE=1 ***")) << "\n";
    }
    report_bbox<Real>(X, comm);

    // QJ_PROBE_DUMP_NODES=<path> writes the probed point set for offline inspection: <path> is the raw
    // little-endian float64 (N x 3) coordinate array, <path>.meta an ASCII sidecar with the counts and
    // the per-junction similarity (t, R, scale). This is what lets two geometries be compared node for
    // node -- e.g. mapping each junction's nodes back through apply_inverse_point to canonical
    // coordinates, so a genuine mesh difference is distinguishable from a pure change of size. Pair
    // with QJ_PROBE_NO_EVAL=1 to stop before the O(N^2) direct sum, which is the entire run cost.
    if (const char* dn = std::getenv("QJ_PROBE_DUMP_NODES")) {
      if (!comm.Rank()) {
        FILE* f = fopen(dn, "wb");
        SCTL_ASSERT_MSG(f, "QJ_PROBE_DUMP_NODES: cannot open output file.");
        fwrite(&X[0], sizeof(Real), X.Dim(), f);
        fclose(f);
        std::ofstream m(std::string(dn) + ".meta");
        m << std::setprecision(17) << std::scientific
          << "nnode " << X.Dim()/3 << "\nnquad_node " << nq_probed/3 << "\nnodes_per_elem " << ord*ord
          << "\npanels_per_junction " << npj << "\nnjunc " << njunc << "\n";
        for (Long i = 0; i < (Long)jP.size(); i++) {
          m << "junc " << i << " scale " << jP[i].scale
            << " t " << jP[i].t[0] << " " << jP[i].t[1] << " " << jP[i].t[2] << " R";
          for (int k = 0; k < 9; k++) m << " " << jP[i].R[k];
          m << "\n";
        }
        std::cout << "  [dump] wrote " << X.Dim()/3 << " nodes to " << dn << " (+ .meta)\n";
      }
    }
    const char* ne_env = std::getenv("QJ_PROBE_NO_EVAL");
    const bool no_eval = (ne_env && atoi(ne_env) != 0);
    if (no_eval && !comm.Rank())
      std::cout << "  QJ_PROBE_NO_EVAL=1: skipping the FMM/direct comparison.\n";

    // QJ_PROBE_CLASSIFY=0 suppresses the per-target geometric breakdown (element / junction / panel-edge /
    // distance to seam). Cheap to compute, but noise once the failure locus is already known.
    const char* cls_env = std::getenv("QJ_PROBE_CLASSIFY");
    const bool classify = (geom == 0) && !(cls_env && atoi(cls_env) == 0);

    // QJ_PROBE_TRG_STRIDE=<k> keeps every k-th node as a TARGET while keeping ALL sources, so the far
    // field each target sees is unchanged but EvalDirect's O(Ns*Nt) cost drops by k. That is what makes
    // the Stokes rows affordable (TrgDim 3 x SrcDim 3 = ~9x a scalar kernel). Nt must stay >= 40000 or
    // EvalPVFMM silently falls back to EvalDirect and the comparison is vacuous (fmm-wrapper.txx:861) --
    // asserted below rather than left to be discovered in the output.
    Vector<Real> Xt = X;
    Long stride = 1;
    if (const char* se = std::getenv("QJ_PROBE_TRG_STRIDE")) stride = std::max<Long>(1, atol(se));
    if (stride > 1) {
      Xt.ReInit(0);
      for (Long i = 0; i < X.Dim()/3; i += stride) { Xt.PushBack(X[3*i]); Xt.PushBack(X[3*i+1]); Xt.PushBack(X[3*i+2]); }
      if (!comm.Rank())
        std::cout << "  QJ_PROBE_TRG_STRIDE=" << stride << ": targets " << X.Dim()/3 << " -> " << Xt.Dim()/3
                  << " (all " << X.Dim()/3 << " sources kept)\n";
      SCTL_ASSERT_MSG(force_tree || GlobalReduce((Long)(Xt.Dim()/3), comm, CommOp::SUM) >= 40000,
                      "QJ_PROBE_TRG_STRIDE leaves < 40000 global targets: EvalPVFMM would fall back to "
                      "EvalDirect and the FMM-vs-direct comparison would be vacuous. Lower the stride or set "
                      "SCTL_FMM_FORCE_TREE=1.");
    }
    // classify_worst indexes into the PROBED target set; with a stride those indices no longer map onto
    // the full node array, so the geometric breakdown would be wrong rather than merely coarse.
    const bool classify_ok = classify && (stride == 1);
    if (classify && !classify_ok && !comm.Rank())
      std::cout << "  [note] classify suppressed: it requires QJ_PROBE_TRG_STRIDE=1 to index the geometry.\n";

    // QJ_PROBE_STOKES=1 adds the Stokes pair. The Laplace rows are the historical reference (every
    // earlier table is Laplace), but the flow BVP solves STOKES through StokesBIO -- so a Laplace-only
    // verdict says nothing about the operator that actually stalls. Off by default to keep the existing
    // Laplace runs the same cost.
    const char* stk_env = std::getenv("QJ_PROBE_STOKES");
    const bool do_stokes = (stk_env && atoi(stk_env) != 0);

    Vector<Long> worst;
    if (!no_eval) {
      probe<Real, Laplace3D_DxU>(X, Xn, Xt, digits, "Laplace DxU", comm, &worst);
      if (classify_ok) classify_worst<Real>(worst, X, Xq_keep, Xsl_keep, nq_probed, ord, npj, njunc, jcen, comm);
      probe<Real, Laplace3D_FxU>(X, Xn, Xt, digits, "Laplace FxU", comm, &worst);
      if (classify_ok) classify_worst<Real>(worst, X, Xq_keep, Xsl_keep, nq_probed, ord, npj, njunc, jcen, comm);
      if (do_stokes) {
        probe<Real, Stokes3D_DxU>(X, Xn, Xt, digits, "Stokes DxU", comm, &worst);
        if (classify_ok) classify_worst<Real>(worst, X, Xq_keep, Xsl_keep, nq_probed, ord, npj, njunc, jcen, comm);
        probe<Real, Stokes3D_FxU>(X, Xn, Xt, digits, "Stokes FxU", comm, &worst);
        if (classify_ok) classify_worst<Real>(worst, X, Xq_keep, Xsl_keep, nq_probed, ord, npj, njunc, jcen, comm);
      }
    }
  }
  Comm::MPI_Finalize();
  return 0;
}
