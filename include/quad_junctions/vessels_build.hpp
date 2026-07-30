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

// A tube segment [A,B] of tube radius rtube joining junctions j0,j1 (j0==j1 for a root cap). Used by the
// caller for the collision guard and exterior-source validation.
template <class Real> struct ArmSeg { Vec3<Real> A, B; Real rtube; int j0, j1; };

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
    const Real tipLen, const Comm& comm, const Real gscale = (Real)1) {
  namespace vd = vessels_data;
  const int NJ = vd::n_junc, NC = vd::n_conn;
  VesselsBuild<Real> out;

  // `gscale` is a GLOBAL SIMILARITY factor: it multiplies BOTH the SVG->world position scale and every
  // junction's own size, so the network is geometrically similar -- only its absolute extent changes
  // (default 1 = unchanged). Note `svgs` alone is NOT a uniform scale: it moves fixed-size junctions
  // closer together, and shrinking it far enough makes add_bent_arm reject the crowded seams.
  // Purpose: the DL constant-density identity is exactly scale-invariant, and SCTL already renormalizes
  // into PVFMM's [0,1]^3 box (fmm-wrapper.txx: bbox_scale=1/bbox_len), so any dependence of the answer
  // on gscale is evidence of an absolute-size bug in the FMM path rather than real physics.
  const Real svgs_g = svgs * gscale;

  // SVG pixel (y-down) -> world (planar z=0), centered: X=s*(x-340), Y=s*(270-y).
  auto world = [&](Real x, Real y) { return Vec3<Real>{ svgs_g*(x-(Real)340), svgs_g*((Real)270-y), (Real)0 }; };

  // QJ_VESSELS_FLAT_SCALE=1 removes the generational taper: every junction is built at the SAME size
  // (the gen-0 root's) instead of 0.9^gen, positions untouched. Diagnostic only -- it is not a
  // similarity transform (arms get shorter relative to the blobs they join), so it is NOT a valid
  // physical geometry to quote identity errors from in absolute terms. Its purpose is to test whether
  // the PVFMM far-field blow-up needs a SPREAD of element sizes: gscale is normalized away by the
  // octree (bbox_scale=1/bbox_len) and so can only probe absolute size, whereas this changes the
  // ratio between the largest and smallest features at nearly fixed extent and node count.
  // QJ_VESSELS_TAPER=<r> overrides the 0.9 base, so scale[i] = r^gen: r=0.9 is production, r=1 makes
  // every junction the root's size (no spread at all), and intermediate r sweeps the spread
  // continuously. QJ_VESSELS_FLAT_SCALE=1 is the r=1 shorthand. Positions are untouched.
  const char* flat_env  = std::getenv("QJ_VESSELS_FLAT_SCALE");
  const char* taper_env = std::getenv("QJ_VESSELS_TAPER");
  const Real taper = (flat_env && atoi(flat_env) != 0) ? (Real)1
                   : (taper_env ? (Real)atof(taper_env) : (Real)0.9);
  std::vector<Vec3<Real>> center(NJ);
  std::vector<Real>       scale(NJ);
  int genmax = 0;
  for (int i = 0; i < NJ; i++) {
    center[i] = world((Real)vd::juncs[i].x, (Real)vd::juncs[i].y);
    scale[i]  = std::pow(taper, (Real)vd::juncs[i].gen) * gscale;
    genmax    = std::max(genmax, vd::juncs[i].gen);
  }
  if (taper != (Real)0.9 && !comm.Rank())
    std::cout << "  [QJ_VESSELS_TAPER] scale = " << taper << "^gen * " << gscale << " (production 0.9^gen); "
              << "max gen " << genmax << " -> size spread " << std::pow((double)taper, genmax) << ":1\n";

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
    const Vec3<Real> d0 = unit3(sub3(tgt[i][0].goal, center[i])), d1 = unit3(sub3(tgt[i][1].goal, center[i]));
    const Vec3<Real> sd = mul3((Real)-1, add3(d0, d1));
    stemdir[i] = (nrm3(sd) > (Real)0.1) ? unit3(sd) : ((i < 10) ? Vec3<Real>{-1,0,0} : Vec3<Real>{1,0,0});
    P[i] = Placement<Real>::AlignArm(0, stemdir[i], up, center[i], scale[i]);
    J.push_back(A.add_junction(P[i], level, nref, etajoin, NsTrans, comm));

    const ArmSeam<Real> s1 = J[i].seam(1), s2 = J[i].seam(2);
    const Real sc00 = dot3(s1.u,d0) + dot3(s2.u,d1), sc01 = dot3(s1.u,d1) + dot3(s2.u,d0);
    const int seamk[2] = { (sc00 >= sc01) ? 1 : 2, (sc00 >= sc01) ? 2 : 1 };
    for (int k = 0; k < 2; k++) {
      const Target& t = tgt[i][k];
      if (t.kind == 0) clink[t.a] = {i, seamk[k]};
      else { llnkJ[t.a][t.b] = i; llnkK[t.a][t.b] = seamk[k]; }
    }
  }

  // --- Pass 2: intra-tree connectors. Prefer the SINGLE-CORNER (lead|corner|run) arm whose run plunges
  //     straight into and becomes the child stem; if the parent branch axis and child stem axis do not
  //     admit a valid single corner (near-parallel, or the corner vertex Q would fall behind a seam --
  //     happens for near-perpendicular "steep" children directly above/below the root), fall back to the
  //     two-corner racetrack. Both taper radius parent->child. ---
  int n_single = 0, n_race = 0;
  for (int c = 0; c < NJ; c++) {
    if (vd::juncs[c].parent < 0) continue;
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
    A.add_bent_arm(a, b, ns, cheb, fourier, leadP, cornerP, sc);
    consumed[pj][pk] = true; consumed[c][0] = true;
    segs.push_back(ArmSeg<Real>{a.C, b.C, std::max(a.R0,b.R0), pj, c});
    (sc ? n_single : n_race)++;
  }
  if (!comm.Rank()) std::cout << "  [intra-tree] single-corner=" << n_single << " racetrack-fallback=" << n_race << "\n";

  // --- Pass 3: leaf connectors A_i <-> V_i (two-corner racetrack; lenses close by construction). ---
  for (int ci = 0; ci < NC; ci++) {
    const ArmSeam<Real> a = J[llnkJ[ci][0]].seam(llnkK[ci][0]), b = J[llnkJ[ci][1]].seam(llnkK[ci][1]);
    const Real chord = nrm3(sub3(b.C, a.C));
    const Real pspac = (Real)1.5*std::max(a.R0, b.R0);
    const Integer nmin = 2*(leadP + cornerP) + 4;
    const Integer ns = std::max<Integer>(nmin, (Integer)std::lround((double)chord/pspac));
    A.add_bent_arm(a, b, ns, cheb, fourier, leadP, cornerP, /*single_corner*/false);
    consumed[llnkJ[ci][0]][llnkK[ci][0]] = true; consumed[llnkJ[ci][1]][llnkK[ci][1]] = true;
    segs.push_back(ArmSeg<Real>{a.C, b.C, std::max(a.R0,b.R0), llnkJ[ci][0], llnkJ[ci][1]});
  }

  // --- Pass 4: cap every remaining (unconsumed) seam -- the two tree-root stems (inlet/outlet). ---
  int n_caps = 0;
  for (int i = 0; i < NJ; i++)
    for (int k = 0; k < 3; k++)
      if (!consumed[i][k]) {
        const ArmSeam<Real> s = J[i].seam(k);
        const Real L = scale[i] * tipLen;
        A.add_free_arm(s, s.a0 + L, nAxFree, Ncap, cheb, fourier);
        segs.push_back(ArmSeg<Real>{s.C, add3(s.C, mul3(L, s.u)), s.R0, i, i});
        out.cap_seams.push_back(s); out.cap_len.push_back(L); out.cap_owner.push_back(i);
        n_caps++;
      }

  out.n_single = n_single; out.n_race = n_race; out.n_caps = n_caps;
  return out;
}

} // namespace quad_junctions
