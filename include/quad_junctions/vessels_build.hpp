/**
 * Shared geometry build for the 20-junction arterial/venous vascular network
 * (`arterial_venous_smoothed_nolabels.svg`; topology in vessels_tree_data.hpp).
 *
 * Extracted verbatim (behavior-preserving) from src/ybifurc-vessels-bie.cpp so that both the identity
 * driver (ybifurc-vessels-bie.cpp) and the physical inflow/outflow flow driver
 * (ybifurc-vessels-flow-bie.cpp) assemble the SAME geometry from one source -- same precedent as
 * hybrid_bie_tests.hpp. `build_vessels_network` runs the four build passes:
 *   1. orient each junction so its two 120-deg branches STRADDLE its two targets (children/leaf-connectors)
 *   2. intra-tree parent->child: single-corner (lead|corner|run) arm, else racetrack fallback
 *   3. leaf connectors A_i<->V_i: two-corner racetrack (four pairs close into racetrack lenses)
 *   4. cap the two unconsumed root stems (the network inlet/outlet)
 * It only POPULATES the assembly `A`; the caller extracts QuadElemList/SlenderElemList via A.quad()/A.slender()
 * and owns the BIE step. Returns the per-junction placements, all tube segments, and the capped root-stem
 * seams (so the flow driver can prescribe inflow/outflow on them).
 */
#pragma once

#include <quad_junctions/ybifurc_assembly.hpp>       // HybridAssembly / Placement / ArmSeam / YField
#include <quad_junctions/vessels_tree_data.hpp>      // arterial/venous junction + connector tables
#include <array>
#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <vector>

namespace quad_junctions {
using namespace sctl;

template <class Real> inline Real dot3(const Vec3<Real>& a, const Vec3<Real>& b) { return a[0]*b[0]+a[1]*b[1]+a[2]*b[2]; }
template <class Real> inline Real nrm3(const Vec3<Real>& a) { return std::sqrt((double)dot3(a,a)); }
template <class Real> inline Vec3<Real> add3(const Vec3<Real>& a, const Vec3<Real>& b) { return Vec3<Real>{a[0]+b[0],a[1]+b[1],a[2]+b[2]}; }
template <class Real> inline Vec3<Real> sub3(const Vec3<Real>& a, const Vec3<Real>& b) { return Vec3<Real>{a[0]-b[0],a[1]-b[1],a[2]-b[2]}; }
template <class Real> inline Vec3<Real> mul3(Real s, const Vec3<Real>& a) { return Vec3<Real>{s*a[0],s*a[1],s*a[2]}; }
template <class Real> inline Vec3<Real> unit3(const Vec3<Real>& a) { const Real n=nrm3(a); return (n>(Real)1e-30)?mul3((Real)1/n,a):a; }

// A tube segment [A,B] of tube radius rtube joining junctions j0,j1 (j0==j1 for a root cap). `cl` is the
// arm's actual (bent) centerline sampled as a polyline -- the straight chord [A,B] misses the bow of a
// draped arm, so adjacent-arm clearance on a sphere must be measured against `cl`. Used by the caller for
// the collision guard and exterior-source validation.
template <class Real> struct ArmSeg { Vec3<Real> A, B; Real rtube; int j0, j1; std::vector<Vec3<Real>> cl; };

// Everything the caller needs after the geometry is populated into `A`.
template <class Real> struct VesselsBuild {
  std::vector<Placement<Real>> P;         // per-junction placement (exterior/collision use)
  std::vector<ArmSeg<Real>>    segs;       // all tube segments (intra arms + connectors + root caps)
  std::vector<ArmSeam<Real>>   cap_seams;  // the capped root-stem seams (network inlet/outlet)
  std::vector<Real>            cap_len;     // free-arm length L per cap (dome-equator center = C + L*u)
  std::vector<int>             cap_owner;   // owning junction id per cap (arterial root <10, venous >=10)
  int n_single = 0, n_race = 0, n_caps = 0;
};

// Build the whole network into `A`. Parameters mirror ybifurc-vessels-bie.cpp's CLI-derived values.
template <class Real>
VesselsBuild<Real> build_vessels_network(HybridAssembly<Real>& A, const Real level, const Integer nref,
    const Real etajoin, const Integer NsTrans, const Long fourier, const Integer leadP,
    const Integer cornerP, const Real svgs, const Integer Ncap, const Long cheb, const Integer nAxFree,
    const Real tipLen, const Comm& comm, const Real gscale = (Real)1, const Real sphere_deg = (Real)0,
    const Real sphere_tilt_deg = (Real)0,
    const bool open_roots = false, const Vec3<Real> world_off = Vec3<Real>{(Real)0,(Real)0,(Real)0},
    const bool arterial_only = false) {
  namespace vd = vessels_data;
  const int NJ = vd::n_junc, NC = vd::n_conn;
  VesselsBuild<Real> out;

  // `arterial_only`: build ONLY the arterial half of the network (junctions 0..NJ/2-1). The venous tree
  // (junctions NJ/2..NJ-1) is omitted entirely, and the leaf connectors A_i<->V_i are NOT drawn --
  // instead every arterial branch seam that would have fed a connector is left unconsumed, so Pass 4
  // caps it (a slender stub + hemisphere) exactly like a root stem. The arterial tree therefore ends in
  // capped free arms where it used to merge into the venous tree. The topology (auto-generated) places
  // arterial junctions in the first half of `juncs` and every connector's a_jct in that half / v_jct in
  // the second; assert that so the i<NJ/2 split stays valid if the SVG is ever re-emitted.
  const int NJ_art = NJ / 2;
  auto active = [&](int i) { return !arterial_only || i < NJ_art; };
  if (arterial_only)
    for (int ci = 0; ci < NC; ci++)
      SCTL_ASSERT_MSG(vd::conns[ci].a_jct < NJ_art && vd::conns[ci].v_jct >= NJ_art,
                      "arterial_only assumes arterial junctions occupy the first half of the topology table.");

  // `gscale` is a GLOBAL SIMILARITY factor: it multiplies BOTH the SVG->world position scale and every
  // junction's own size, so the network is geometrically similar -- only its absolute extent changes
  // (default 1 = unchanged). Note `svgs` alone is NOT a uniform scale: it moves fixed-size junctions
  // closer together, and shrinking it far enough makes add_bent_arm reject the crowded seams.
  // Purpose: the DL constant-density identity is exactly scale-invariant, and SCTL already renormalizes
  // into PVFMM's [0,1]^3 box (fmm-wrapper.txx: bbox_scale=1/bbox_len), so any dependence of the answer
  // on gscale is evidence of an absolute-size bug in the FMM path rather than real physics.
  const Real svgs_g = svgs * gscale;

  // SVG pixel (y-down) -> world (planar z=0), centered: X=s*(x-340), Y=s*(270-y). `world_off` is an
  // OPTIONAL global post-translation (added after the gscale-scaled position); with gscale it realizes a
  // full uniform affine (e.g. to fit the network into a prescribed box). Currently unused. Default
  // {0,0,0} leaves every existing caller byte-for-byte identical. Note the junction interiors are built
  // relative to world(...) and the arms follow the seams, so a constant offset is a rigid translation.
  auto world = [&](Real x, Real y) { return Vec3<Real>{ svgs_g*(x-(Real)340)+world_off[0],
                                                        svgs_g*((Real)270-y)+world_off[1], world_off[2] }; };

  // QJ_VESSELS_FLAT_SCALE=1 removes the generational taper: every junction is built at the SAME size
  // (the gen-0 root's) instead of 0.8^gen, positions untouched. Diagnostic only -- it is not a
  // similarity transform (arms get shorter relative to the blobs they join), so it is NOT a valid
  // physical geometry to quote identity errors from in absolute terms. Its purpose is to test whether
  // the PVFMM far-field blow-up needs a SPREAD of element sizes: gscale is normalized away by the
  // octree (bbox_scale=1/bbox_len) and so can only probe absolute size, whereas this changes the
  // ratio between the largest and smallest features at nearly fixed extent and node count.
  // QJ_VESSELS_TAPER=<r> overrides the 0.8 base, so scale[i] = r^gen: r=0.8 is production, r=1 makes
  // every junction the root's size (no spread at all), and intermediate r sweeps the spread
  // continuously. QJ_VESSELS_FLAT_SCALE=1 is the r=1 shorthand. Positions are untouched.
  const char* flat_env  = std::getenv("QJ_VESSELS_FLAT_SCALE");
  const char* taper_env = std::getenv("QJ_VESSELS_TAPER");
  const Real taper = (flat_env && atoi(flat_env) != 0) ? (Real)1
                   : (taper_env ? (Real)atof(taper_env) : (Real)0.8);
  std::vector<Vec3<Real>> center(NJ);
  std::vector<Real>       scale(NJ);
  int genmax = 0;
  for (int i = 0; i < NJ; i++) {
    center[i] = world((Real)vd::juncs[i].x, (Real)vd::juncs[i].y);
    scale[i]  = std::pow(taper, (Real)vd::juncs[i].gen) * gscale;
    genmax    = std::max(genmax, vd::juncs[i].gen);
  }
  if (taper != (Real)0.8 && !comm.Rank())
    std::cout << "  [QJ_VESSELS_TAPER] scale = " << taper << "^gen * " << gscale << " (production 0.8^gen); "
              << "max gen " << genmax << " -> size spread " << std::pow((double)taper, genmax) << ":1\n";

  // --- Sphere drape (sphere_deg > 0). The planar blueprint is kept, but every junction is placed RIGIDLY
  //     tangent to a sphere whose radius R makes the network's overall length span `sphere_deg` degrees of
  //     arc; the bent arms carry the 3D curvature (transported-frame add_bent_arm). Junctions stay exact
  //     isometries of the canonical mesh (undeformed), so DL/Green's stay at the planar floor. sphere_deg=0
  //     leaves every placement/orientation/arm byte-for-byte identical to the planar network. ---
  const bool onsphere = ((double)sphere_deg > 0);
  // DIAGNOSTIC: force the transported (RMF) arm frame even in the planar build, to isolate whether the
  // transported frame (vs the sphere placement) is responsible for any watertight/identity change.
  const char* ft_env = std::getenv("QJ_FORCE_TRANSPORTED");
  const bool armtrans = onsphere || (ft_env && atoi(ft_env) != 0);
  Vec3<Real> Gc{0,0,0}; Real Rsph = 0;
  auto matmat = [](const Real A[9], const Real B[9], Real C[9]) {
    for (int r=0;r<3;r++) for (int c=0;c<3;c++) { Real s=0; for (int k=0;k<3;k++) s+=A[3*r+k]*B[3*k+c]; C[3*r+c]=s; } };
  auto matT_vec = [](const Real R[9], const Vec3<Real>& v) {
    return Vec3<Real>{ R[0]*v[0]+R[3]*v[1]+R[6]*v[2], R[1]*v[0]+R[4]*v[1]+R[7]*v[2], R[2]*v[0]+R[5]*v[1]+R[8]*v[2] }; };
  auto mat_vec = [](const Real R[9], const Vec3<Real>& v) {
    return Vec3<Real>{ R[0]*v[0]+R[1]*v[1]+R[2]*v[2], R[3]*v[0]+R[4]*v[1]+R[5]*v[2], R[6]*v[0]+R[7]*v[1]+R[8]*v[2] }; };
  if (onsphere) {
    int nact = 0;
    for (int i=0;i<NJ;i++) if (active(i)) { Gc[0]+=center[i][0]; Gc[1]+=center[i][1]; Gc[2]+=center[i][2]; nact++; }
    Gc[0]/=nact; Gc[1]/=nact; Gc[2]/=nact;
    Real Lspan = 0;
    for (int i=0;i<NJ;i++) if (active(i)) for (int j=i+1;j<NJ;j++) if (active(j)) { const Real d=nrm3(sub3(center[i],center[j])); if (d>Lspan) Lspan=d; }
    Rsph = Lspan / (sphere_deg * (const_pi<Real>()/(Real)180));
    if (!comm.Rank())
      std::cout << "  [sphere] span L=" << Lspan << " -> R=" << Rsph << " (" << sphere_deg
                << " deg arc); junctions placed tangent, arms hug the surface (RMF frame)\n";
  }
  // Sphere centre O (pole = Gc, cap toward -z) and a pointer used to pull each arm run onto the surface.
  const Vec3<Real> Osph{Gc[0], Gc[1], Gc[2]-Rsph};
  const Vec3<Real>* hugP = onsphere ? &Osph : nullptr;
  // Sphere point for a planar world point q (pole = network centroid Gc; cap curves toward -z).
  auto sphere_map = [&](const Vec3<Real>& q) -> Vec3<Real> {
    const Real dx=q[0]-Gc[0], dy=q[1]-Gc[1], rho=std::sqrt((double)(dx*dx+dy*dy));
    if ((double)rho < 1e-30) return Vec3<Real>{Gc[0], Gc[1], Gc[2]};
    const Real cphi=dx/rho, sphi=dy/rho, th=rho/Rsph, cth=std::cos((double)th), sth=std::sin((double)th);
    return Vec3<Real>{ Gc[0]+Rsph*sth*cphi, Gc[1]+Rsph*sth*sphi, Gc[2]-Rsph+Rsph*cth };
  };
  // Rigid tangent frame Q (row-major 3x3: plane axes -> sphere tangent) + center T for a junction center q.
  auto sphere_frame = [&](const Vec3<Real>& q, Real Q[9], Vec3<Real>& T) {
    const Real dx=q[0]-Gc[0], dy=q[1]-Gc[1], rho=std::sqrt((double)(dx*dx+dy*dy));
    if ((double)rho < 1e-30) { for (int k=0;k<9;k++) Q[k]=(k%4==0)?(Real)1:(Real)0; T=Vec3<Real>{Gc[0],Gc[1],Gc[2]}; return; }
    const Real cphi=dx/rho, sphi=dy/rho, th=rho/Rsph, cth=std::cos((double)th), sth=std::sin((double)th);
    const Vec3<Real> nh{sth*cphi, sth*sphi, cth}, et{cth*cphi, cth*sphi, -sth}, ep{-sphi, cphi, (Real)0};
    T = Vec3<Real>{ Gc[0]+Rsph*nh[0], Gc[1]+Rsph*nh[1], Gc[2]-Rsph+Rsph*nh[2] };
    const Real M[9]  = { et[0],ep[0],nh[0], et[1],ep[1],nh[1], et[2],ep[2],nh[2] };  // cols (e_theta,e_phi,n_hat)
    const Real BT[9] = { cphi,sphi,(Real)0, -sphi,cphi,(Real)0, (Real)0,(Real)0,(Real)1 };  // (rho,phi,z)^T
    matmat(M, BT, Q);  // Q = [e_theta e_phi n_hat] * [rho_hat phi_hat z_hat]^T
  };

  // Per-junction branch TARGETS (each junction has exactly two): child junctions + connector leaves.
  struct Target { int kind; int a; int b; Vec3<Real> goal; };   // kind 0=child(a=child id); 1=leaf(a=conn,b=side)
  std::vector<std::vector<Target>> tgt(NJ);
  for (int i = 0; i < NJ; i++)
    if (vd::juncs[i].parent >= 0) tgt[vd::juncs[i].parent].push_back({0, i, 0, center[i]});
  for (int ci = 0; ci < NC; ci++) {
    tgt[vd::conns[ci].a_jct].push_back({1, ci, 0, world((Real)vd::conns[ci].ax, (Real)vd::conns[ci].ay)});
    tgt[vd::conns[ci].v_jct].push_back({1, ci, 1, world((Real)vd::conns[ci].vx, (Real)vd::conns[ci].vy)});
  }
  for (int i = 0; i < NJ; i++) SCTL_ASSERT_MSG(tgt[i].size() == 2, "each junction must have exactly two branch targets.");

  std::vector<HybridJunction<Real>> J; J.reserve(NJ);
  std::vector<Placement<Real>>& P = out.P; P.resize(NJ);
  std::vector<Vec3<Real>> stemdir(NJ);
  std::vector<std::array<bool,3>> consumed(NJ, {false,false,false});
  std::vector<std::array<int,2>> clink(NJ, {-1,-1});          // child -> (parent junction id, parent branch seam k)
  std::vector<std::array<int,2>> llnkJ(NC, {-1,-1}), llnkK(NC, {-1,-1});  // conn -> {A,V} junction id / seam k
  std::vector<ArmSeg<Real>>& segs = out.segs;
  const Vec3<Real> up{0,0,1};

  // --- Pass 1: orient each junction so its two 120-deg branches STRADDLE its two targets
  //     (stem = -bisector of the target directions -> the stem points back toward the parent/root side,
  //     the branches fan toward the children/leaf-connectors). Orientation is independent per junction
  //     (all target positions are known up front), so no BFS dependency. Then place it and assign its
  //     two branch seams to the two targets by best axis-direction match. ---
  for (int i = 0; i < NJ; i++) {
    if (!active(i)) continue;   // arterial_only: skip the venous half (indices >= NJ/2)
    int seamk[2];
    if (onsphere) {
      // Rigid tangent placement: compose the local (planar) alignment with the sphere tangent frame Q.
      // R = Q * R_plane, t = T (sphere point), scale unchanged -> an exact isometry (undeformed junction).
      Real Q[9]; Vec3<Real> T; sphere_frame(center[i], Q, T);
      auto localdir = [&](const Vec3<Real>& goal) {           // target dir projected into the tangent plane
        Vec3<Real> w = matT_vec(Q, sub3(sphere_map(goal), T)); w[2] = 0; return unit3(w); };
      const Vec3<Real> d0l = localdir(tgt[i][0].goal), d1l = localdir(tgt[i][1].goal);
      const Vec3<Real> sdl = mul3((Real)-1, add3(d0l, d1l));
      const Vec3<Real> stem_l = (nrm3(sdl) > (Real)0.1) ? unit3(sdl) : ((i < 10) ? Vec3<Real>{-1,0,0} : Vec3<Real>{1,0,0});
      stemdir[i] = mat_vec(Q, stem_l);
      const Placement<Real> Ppl = Placement<Real>::AlignArm(0, stem_l, up, Vec3<Real>{0,0,0}, scale[i]);
      Placement<Real> Ps; matmat(Q, Ppl.R, Ps.R); Ps.t = T; Ps.scale = scale[i];
      P[i] = Ps;
      J.push_back(A.add_junction(P[i], level, nref, etajoin, NsTrans, comm));
      // Seam assignment in the local frame (dot products are rotation-invariant), same choice as planar.
      const Vec3<Real> s1l = matT_vec(Q, J[i].seam(1).u), s2l = matT_vec(Q, J[i].seam(2).u);
      const Real sc00 = dot3(s1l,d0l) + dot3(s2l,d1l), sc01 = dot3(s1l,d1l) + dot3(s2l,d0l);
      seamk[0] = (sc00 >= sc01) ? 1 : 2; seamk[1] = (sc00 >= sc01) ? 2 : 1;
    } else {
      const Vec3<Real> d0 = unit3(sub3(tgt[i][0].goal, center[i])), d1 = unit3(sub3(tgt[i][1].goal, center[i]));
      const Vec3<Real> sd = mul3((Real)-1, add3(d0, d1));
      stemdir[i] = (nrm3(sd) > (Real)0.1) ? unit3(sd) : ((i < 10) ? Vec3<Real>{-1,0,0} : Vec3<Real>{1,0,0});
      P[i] = Placement<Real>::AlignArm(0, stemdir[i], up, center[i], scale[i]);
      J.push_back(A.add_junction(P[i], level, nref, etajoin, NsTrans, comm));

      const ArmSeam<Real> s1 = J[i].seam(1), s2 = J[i].seam(2);
      const Real sc00 = dot3(s1.u,d0) + dot3(s2.u,d1), sc01 = dot3(s1.u,d1) + dot3(s2.u,d0);
      seamk[0] = (sc00 >= sc01) ? 1 : 2; seamk[1] = (sc00 >= sc01) ? 2 : 1;
    }
    for (int k = 0; k < 2; k++) {
      const Target& t = tgt[i][k];
      if (t.kind == 0) clink[t.a] = {i, seamk[k]};
      else { llnkJ[t.a][t.b] = i; llnkK[t.a][t.b] = seamk[k]; }
    }
  }

  // Sample an arm's ACTUAL (bent) centerline as a polyline, using the same corner params add_bent_arm did
  // (skew_safe=armtrans so it matches the emitted 3D fiber). Stored per-segment for the adjacent-arm
  // clearance check, which the straight-chord [A,B] would miss on a curved (draped) network.
  auto sample_cl = [&](const ArmSeam<Real>& a, const ArmSeam<Real>& b, Integer ns, bool sc,
                       const Vec3<Real>& tilt = Vec3<Real>{0,0,0}) {
    std::vector<Vec3<Real>> cl; const int M = 48; cl.reserve(M+1);
    for (int m = 0; m <= M; m++) {
      const Real t = (Real)m/M;
      Vec3<Real> p = HybridAssembly<Real>::bent_centerline(a, b, t, leadP, cornerP, ns, sc, armtrans);
      Real e = std::sin((double)(const_pi<Real>()*t)); e *= e;             // sin^2(pi t) mid-arm bump
      p = Vec3<Real>{p[0]+tilt[0]*e, p[1]+tilt[1]*e, p[2]+tilt[2]*e};
      if (onsphere) p = HybridAssembly<Real>::hug_to_sphere(p, a.C, b.C, t, leadP, cornerP, ns, Osph);
      cl.push_back(p);
    }
    return cl;
  };

  // --- Pass 2: intra-tree connectors. Prefer the SINGLE-CORNER (lead|corner|run) arm whose run plunges
  //     straight into and becomes the child stem; if the parent branch axis and child stem axis do not
  //     admit a valid single corner (near-parallel, or the corner vertex Q would fall behind a seam --
  //     happens for near-perpendicular "steep" children directly above/below the root), fall back to the
  //     two-corner racetrack. Both taper radius parent->child. ---
  int n_single = 0, n_race = 0;
  for (int c = 0; c < NJ; c++) {
    if (vd::juncs[c].parent < 0) continue;
    if (!active(c)) continue;   // arterial_only: no venous intra-tree arms
    const int pj = clink[c][0], pk = clink[c][1];
    const ArmSeam<Real> a = J[pj].seam(pk), b = J[c].seam(0);
    const Vec3<Real> chv = sub3(b.C, a.C);
    const Real chord = nrm3(chv), cc = dot3(a.u, b.u);
    bool sc = false;
    if (std::fabs((double)cc) < 0.9999) {
      const Real den = (Real)1 - cc*cc;
      const Real r_ = (dot3(chv,a.u)*cc - dot3(chv,b.u))/den, s_ = dot3(chv,a.u) + r_*cc;
      sc = (s_ > (Real)0 && r_ > (Real)0);
    }
    const Real pspac = (Real)1.5*std::max(a.R0, b.R0);
    const Integer nmin = 2*(leadP + cornerP) + 4;   // room for either single (2*lead+corner) or racetrack
    const Integer ns = std::max<Integer>(nmin, (Integer)std::lround((double)chord/pspac));
    A.add_bent_arm(a, b, ns, cheb, fourier, leadP, cornerP, sc, /*transported*/armtrans,
                   Vec3<Real>{0,0,0}, /*hug_O*/hugP);
    consumed[pj][pk] = true; consumed[c][0] = true;
    segs.push_back(ArmSeg<Real>{a.C, b.C, std::max(a.R0,b.R0), pj, c, sample_cl(a,b,ns,sc)});
    (sc ? n_single : n_race)++;
  }
  if (!comm.Rank()) std::cout << "  [intra-tree] single-corner=" << n_single << " racetrack-fallback=" << n_race << "\n";

  // --- Pass 3: leaf connectors A_i <-> V_i (two-corner racetrack; lenses close by construction). ---
  //     Skipped entirely in arterial_only mode: the arterial branch seams that would have fed a connector
  //     stay unconsumed and get capped in Pass 4, so the arterial tree terminates in free arms + caps.
  for (int ci = 0; !arterial_only && ci < NC; ci++) {
    const ArmSeam<Real> a = J[llnkJ[ci][0]].seam(llnkK[ci][0]), b = J[llnkJ[ci][1]].seam(llnkK[ci][1]);
    const Real chord = nrm3(sub3(b.C, a.C));
    const Real pspac = (Real)1.5*std::max(a.R0, b.R0);
    const Integer nmin = 2*(leadP + cornerP) + 4;
    const Integer ns = std::max<Integer>(nmin, (Integer)std::lround((double)chord/pspac));
    // The two MIDDLE connectors (4: A4<->V4 above centre, 5: A5<->V5 below) run closest and touch when the
    // network is tightly draped. Nudge them apart out of the local tangent plane: connector 4 "up" (+radial,
    // away from the sphere centre) and connector 5 "down" (-radial), by sphere_tilt_deg. The bump vanishes
    // at the seams so watertightness is preserved. Only active on a sphere with a nonzero tilt requested.
    Vec3<Real> tilt{0,0,0};
    if (onsphere && (double)sphere_tilt_deg > 0 && (ci == 4 || ci == 5)) {
      const Vec3<Real> mid = mul3((Real)0.5, add3(a.C, b.C));
      const Vec3<Real> O{Gc[0], Gc[1], Gc[2]-Rsph};                     // sphere centre
      const Vec3<Real> radial = unit3(sub3(mid, O));                    // "up" = away from centre
      // sin^2(pi t) bump of amplitude `amp` over chord L has max slope amp*pi/L, so amp = L*tan(deg)/pi
      // makes sphere_tilt_deg the actual peak tilt angle of the centerline off the tangent plane.
      const Real amp = nrm3(sub3(b.C,a.C)) * (Real)std::tan((double)(sphere_tilt_deg*const_pi<Real>()/180)) / const_pi<Real>();
      tilt = mul3((ci == 4 ? amp : -amp), radial);
      if (!comm.Rank()) std::cout << "  [sphere] tilt connector " << ci << " by " << (ci==4?"+":"-")
                                  << sphere_tilt_deg << " deg (|offset|=" << std::fabs((double)amp) << ")\n";
    }
    A.add_bent_arm(a, b, ns, cheb, fourier, leadP, cornerP, /*single_corner*/false, /*transported*/armtrans, tilt, /*hug_O*/hugP);
    consumed[llnkJ[ci][0]][llnkK[ci][0]] = true; consumed[llnkJ[ci][1]][llnkK[ci][1]] = true;
    segs.push_back(ArmSeg<Real>{a.C, b.C, std::max(a.R0,b.R0), llnkJ[ci][0], llnkJ[ci][1], sample_cl(a,b,ns,false,tilt)});
  }

  // --- Pass 4: cap every remaining (unconsumed) seam -- the two tree-root stems (inlet/outlet). ---
  int n_caps = 0;
  for (int i = 0; i < NJ; i++) {
    if (!active(i)) continue;   // arterial_only: only cap arterial seams (venous half was never built)
    for (int k = 0; k < 3; k++)
      if (!consumed[i][k]) {
        const ArmSeam<Real> s = J[i].seam(k);
        const Real L = scale[i] * tipLen;
        // `open_roots`: emit ONLY the straight stub fiber (no hemisphere), so the root ends in an OPEN
        // circular ring -- the port for a periodic (or otherwise externally-closed) problem. Otherwise the
        // production behavior: a slender stub + a hemisphere cap (closed watertight network).
        A.add_free_arm(s, s.a0 + L, nAxFree, Ncap, cheb, fourier, (Real)0.40, /*with_cap*/!open_roots);
        segs.push_back(ArmSeg<Real>{s.C, add3(s.C, mul3(L, s.u)), s.R0, i, i, {s.C, add3(s.C, mul3(L, s.u))}});
        out.cap_seams.push_back(s); out.cap_len.push_back(L); out.cap_owner.push_back(i);
        n_caps++;
      }
  }

  out.n_single = n_single; out.n_race = n_race; out.n_caps = n_caps;
  return out;
}

} // namespace quad_junctions
