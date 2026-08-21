/**
 * Rigid spherical OBSTACLES near the centerlines of the bifurcation geometries.
 *
 * Spheres are represented exactly as ../stokes-periodize-numtest does: a CSBQ SlenderElemList
 * SURFACE OF REVOLUTION about a local axis (centerline x_c(theta)=R*cos(theta) along an axis u,
 * cross-section radius R*sin(theta), theta in [0,pi]; the pole degeneracy R*sin->0 is what CSBQ
 * calls a "closed" slender body). One SlenderElemList holds ALL obstacle spheres, node-balanced
 * across MPI ranks with the same partition InitElemList uses in the reference.
 *
 * Placement (deterministic, replicated on every rank):
 *   ARMS      -- one sphere per axial panel, at the panel's MIDDLE Chebyshev node. The centerline
 *                point C and cross-section frame (e1,e2,r) come from four SlenderElemList::GetGeom
 *                probes (the same window build_arm_panel_targets uses). The center is pushed a
 *                random radial distance off the centerline, staying well inside the tube; panels
 *                lying on an excluded stem (inflow/outflow arm) are skipped.
 *   JUNCTIONS -- one sphere per junction, at the junction center pushed a random 3D direction,
 *                staying near the center.
 * Sphere radius = clamp(radfrac*r, 0.01*r, 0.45*r) of the LOCAL tube/junction radius r (radfrac
 * default 0.2). All draws use srand48 with a fixed seed so the obstacle set is reproducible and
 * INDEPENDENT of the MPI rank count (candidates are gathered + canonically sorted before drawing).
 *
 * ORIENTATION NOTE. A sphere built this way carries the SAME CSBQ normal convention as the vessel
 * tubes: "radially outward from the local centerline." For a tube that points OUT of the fluid;
 * for an obstacle sphere (fluid OUTSIDE it) that points INTO the fluid -- i.e. OPPOSITE the wall
 * convention. The combined-field solve therefore applies the DL self-jump with the OPPOSITE sign
 * on the obstacle block (see solve_dirichlet_bvp's per-node jump), exactly as the reference does
 * with its per-node NormalOrient. SlenderElemList normals are not flippable in geometry, so the
 * sign lives in the jump, not the mesh.
 */
#pragma once

#include <sctl.hpp>
#include <quad_junctions/ybifurc_geom.hpp>   // Vec3<Real> = std::array<Real,3>
#include <vector>
#include <algorithm>
#include <cstdlib>
#include <cmath>

namespace quad_junctions {
using namespace sctl;

template <class Real> struct SphereObstacle { Vec3<Real> c; Real r; };

// A finite cylinder used to EXCLUDE arm panels (e.g. the inflow/outflow stems): axis from A along
// unit u for length L, radius `rad`. A candidate is excluded if its centerline point projects onto
// [-marg, L+marg] and lies within fac*rad of the axis.
template <class Real> struct ExclCyl { Vec3<Real> A; Vec3<Real> u; Real L; Real rad; };

namespace sph_detail {
  template <class Real> inline Real dot(const Vec3<Real>& a, const Vec3<Real>& b) { return a[0]*b[0]+a[1]*b[1]+a[2]*b[2]; }
  template <class Real> inline Real len(const Vec3<Real>& a) { return (Real)std::sqrt((double)dot(a,a)); }
  template <class Real> inline Vec3<Real> axpy(Real s, const Vec3<Real>& a, const Vec3<Real>& b) { return Vec3<Real>{s*a[0]+b[0], s*a[1]+b[1], s*a[2]+b[2]}; }
  template <class Real> inline Vec3<Real> unit(const Vec3<Real>& a) { const Real n = len(a); return (n > (Real)1e-30) ? Vec3<Real>{a[0]/n,a[1]/n,a[2]/n} : a; }
  template <class Real> inline Real clampr(Real v, Real lo, Real hi) { return v < lo ? lo : (v > hi ? hi : v); }

  template <class Real> inline bool inside_excl(const Vec3<Real>& C, const std::vector<ExclCyl<Real>>& X) {
    for (const ExclCyl<Real>& c : X) {
      const Vec3<Real> v{C[0]-c.A[0], C[1]-c.A[1], C[2]-c.A[2]};
      const Real ax = dot(v, c.u);
      const Real marg = c.rad;
      if (ax < -marg || ax > c.L + marg) continue;
      const Vec3<Real> perp{v[0]-ax*c.u[0], v[1]-ax*c.u[1], v[2]-ax*c.u[2]};
      if (len(perp) < (Real)1.5 * c.rad) return true;
    }
    return false;
  }
}  // namespace sph_detail

// True if (x,y,z) is OUTSIDE every obstacle sphere (buffered by `buf`>=1). Used to drop interior
// visualization targets that fall inside/near an obstacle (cf. reference outside_sphere, buf=1.05).
template <class Real>
bool outside_all_spheres(Real x, Real y, Real z, const std::vector<SphereObstacle<Real>>& S, Real buf = (Real)1.05) {
  for (const SphereObstacle<Real>& s : S) {
    const Real dx = x-s.c[0], dy = y-s.c[1], dz = z-s.c[2];
    if (dx*dx + dy*dy + dz*dz < s.r*s.r*buf*buf) return false;
  }
  return true;
}

// ---- Build ONE SlenderElemList holding every sphere in `S` as a surface of revolution. Npanel
// ---- theta-panels of ElemOrder Chebyshev nodes each; FourierOrder azimuthal nodes. Node-balanced
// ---- MPI partition (identical to the reference InitElemList).
// ----
// ---- IMPORTANT: pass ElemOrder = 10 (the arm order). CSBQ keys its self/near "special_quad_q<ElemOrder>"
// ---- rule on the Chebyshev ORDER, and this repo's data/ cache ships ONLY special_quad_q10. Any other
// ---- ElemOrder makes CSBQ recompute the rule at RUN TIME (minutes of per-level "condition number ..."
// ---- logging) before the solve even starts. FourierOrder is free (data/ has the toroidal_quad_rule_m*
// ---- family densely). See CLAUDE.md, "CSBQ ElemOrder". ----
template <class Real>
SlenderElemList<Real> build_obstacle_elem_list(const std::vector<SphereObstacle<Real>>& S,
                                               const Long ElemOrder, const Long FourierOrder,
                                               const Long Npanel, const Comm& comm) {
  SlenderElemList<Real> lst;
  const Long Nsph = (Long)S.size();
  if (!Nsph) return lst;   // empty list (every rank returns an empty list -> harmless in the operator)

  // Sphere axis is arbitrary (isotropic); a fixed frame keeps the build deterministic.
  const Vec3<Real> axis{(Real)1,(Real)0,(Real)0};
  const Vec3<Real> e1{(Real)0,(Real)1,(Real)0};
  const Real pi = const_pi<Real>();

  // Full (global) per-panel arrays, in a fixed sphere-major order.
  Vector<Long> ElemOrderVec, FourierOrderVec;
  Vector<Real> Xc, eps, orient;
  const Vector<Real>& cn = SlenderElemList<Real>::CenterlineNodes((Integer)ElemOrder);
  for (Long p = 0; p < Nsph; p++) {
    const Real R = S[(size_t)p].r;
    const Vec3<Real>& c = S[(size_t)p].c;
    for (Long ip = 0; ip < Npanel; ip++) {
      ElemOrderVec.PushBack(ElemOrder); FourierOrderVec.PushBack(FourierOrder);
      for (Long j = 0; j < ElemOrder; j++) {
        const Real theta = pi * (ip + cn[j]) / (Real)Npanel;
        const Real xr = R * (Real)std::cos((double)theta);   // station along the axis
        const Real rr = R * (Real)std::sin((double)theta);   // cross-section radius
        Xc.PushBack(c[0] + xr*axis[0]); Xc.PushBack(c[1] + xr*axis[1]); Xc.PushBack(c[2] + xr*axis[2]);
        eps.PushBack(rr);
        orient.PushBack(e1[0]); orient.PushBack(e1[1]); orient.PushBack(e1[2]);
      }
    }
  }

  // Node-balanced partition (node_cnt[e] = ElemOrder*FourierOrder^2), same as reference InitElemList.
  const Long Nelem = ElemOrderVec.Dim();
  Vector<Long> node_cnt(Nelem), node_dsp(Nelem); node_dsp = 0;
  for (Long i = 0; i < Nelem; i++) node_cnt[i] = ElemOrderVec[i]*FourierOrderVec[i]*FourierOrderVec[i];
  omp_par::scan(node_cnt.begin(), node_dsp.begin(), Nelem);
  const Long Nnodes = node_cnt[Nelem-1] + node_dsp[Nelem-1];
  const Long Np = comm.Size(), rank = comm.Rank();
  Long a = std::lower_bound(node_dsp.begin(), node_dsp.end(), Nnodes*(rank+0)/Np) - node_dsp.begin();
  Long b = std::lower_bound(node_dsp.begin(), node_dsp.end(), Nnodes*(rank+1)/Np) - node_dsp.begin();
  if (rank == 0) a = 0;
  if (rank == Np - 1) b = Nelem;
  const Long loc_cnt = b - a, loc_dsp = a;

  Long dsp = 0, cnt = 0;
  for (Long i = 0; i < loc_dsp; i++) dsp += ElemOrderVec[i];
  for (Long i = 0; i < loc_cnt; i++) cnt += ElemOrderVec[loc_dsp+i];
  const Vector<Long> LocElemOrder(loc_cnt, ElemOrderVec.begin()+loc_dsp, false);
  const Vector<Long> LocFourierOrder(loc_cnt, FourierOrderVec.begin()+loc_dsp, false);
  const Vector<Real> X_(cnt*3, Xc.begin()+dsp*3, false);
  const Vector<Real> R_(cnt, eps.begin()+dsp, false);
  const Vector<Real> O_(cnt*3, orient.begin()+dsp*3, false);
  lst.template Init<Real>(LocElemOrder, LocFourierOrder, X_, R_, O_);
  return lst;
}

// ---- One obstacle per axial panel of `arms`, off the centerline by a random radial offset. Panels
// ---- on an excluded stem are skipped. Candidates (C,e1,e2,r) are gathered from all ranks and
// ---- canonically sorted so the srand48 draw order -- and hence the obstacle set -- is identical
// ---- for any MPI rank count. Appends to `out`. ----
template <class Real>
void place_arm_panel_obstacles(const SlenderElemList<Real>& arms, const Comm& comm, const Long cheb,
                               const std::vector<ExclCyl<Real>>& skip, const unsigned seed,
                               const Real radfrac, std::vector<SphereObstacle<Real>>& out) {
  // Probe at the MIDDLE Chebyshev node ("near the middle axially").
  const Vector<Real>& cnodes = SlenderElemList<Real>::CenterlineNodes((Integer)cheb);
  Vector<Real> s_param(1); s_param[0] = cnodes[cheb/2];
  Vector<Real> sin_th(4), cos_th(4);
  for (Integer a = 0; a < 4; a++) { const double th = 0.5*M_PI*(double)a; sin_th[a] = (Real)std::sin(th); cos_th[a] = (Real)std::cos(th); }

  // Pack 10 reals per candidate: C(3), e1(3), e2(3), r(1).
  Vector<Real> loc;
  const Long np = arms.Size();   // LOCAL panels under MPI
  for (Long e = 0; e < np; e++) {
    Vector<Real> Xs;
    arms.GetGeom(&Xs, nullptr, nullptr, nullptr, nullptr, s_param, sin_th, cos_th, e);
    Real C[3], u1[3], u2[3];
    for (Integer k = 0; k < 3; k++) C[k] = (Xs[k] + Xs[3+k] + Xs[6+k] + Xs[9+k]) * (Real)0.25;
    Real r = 0, n2 = 0;
    for (Integer k = 0; k < 3; k++) { u1[k] = Xs[k]   - C[k]; r  += u1[k]*u1[k]; }
    for (Integer k = 0; k < 3; k++) { u2[k] = Xs[3+k] - C[k]; n2 += u2[k]*u2[k]; }
    r = (Real)std::sqrt((double)r); const Real n = (Real)std::sqrt((double)n2);
    if (!(r > 0) || !(n > 0)) continue;
    for (Integer k = 0; k < 3; k++) { u1[k] /= r; u2[k] /= n; }
    loc.PushBack(C[0]); loc.PushBack(C[1]); loc.PushBack(C[2]);
    loc.PushBack(u1[0]); loc.PushBack(u1[1]); loc.PushBack(u1[2]);
    loc.PushBack(u2[0]); loc.PushBack(u2[1]); loc.PushBack(u2[2]);
    loc.PushBack(r);
  }

  // Gather every rank's candidates so the placement is computed identically everywhere.
  const Integer nranks = comm.Size();
  Vector<Long> scv(1); scv[0] = loc.Dim();
  Vector<Long> rc((Long)nranks); comm.Allgather(scv.begin(), 1, rc.begin(), 1);
  Vector<Long> rd((Long)nranks); Long tot = 0;
  for (Integer i = 0; i < nranks; i++) { rd[i] = tot; tot += rc[i]; }
  Vector<Real> all(tot); if (tot) comm.Allgatherv(loc.begin(), loc.Dim(), all.begin(), rc.begin(), rd.begin());

  // Canonical order: sort candidate records by centerline coordinate (lexicographic).
  const Long Ncand = tot / 10;
  std::vector<Long> idx((size_t)Ncand);
  for (Long i = 0; i < Ncand; i++) idx[(size_t)i] = i;
  std::sort(idx.begin(), idx.end(), [&](Long i, Long j) {
    for (Integer k = 0; k < 3; k++) { if (all[i*10+k] < all[j*10+k]) return true; if (all[i*10+k] > all[j*10+k]) return false; }
    return false;
  });

  srand48((long)seed);
  for (Long ii = 0; ii < Ncand; ii++) {
    const Long i = idx[(size_t)ii];
    const Vec3<Real> C{all[i*10+0], all[i*10+1], all[i*10+2]};
    const Vec3<Real> u1{all[i*10+3], all[i*10+4], all[i*10+5]};
    const Vec3<Real> u2{all[i*10+6], all[i*10+7], all[i*10+8]};
    const Real r = all[i*10+9];
    // Draw regardless of skip so the RNG stream (and thus the kept set) is stable under the skip list.
    const Real phi = (Real)(2.0*M_PI*drand48());
    const Real u   = (Real)drand48();
    if (sph_detail::inside_excl<Real>(C, skip)) continue;
    const Real r_sph = sph_detail::clampr<Real>(radfrac*r, (Real)0.01*r, (Real)0.45*r);
    const Real rho   = (Real)0.6 * (r - r_sph) * u;   // outer extent <= r_sph + rho well inside the tube
    const Real cp = (Real)std::cos((double)phi), sp = (Real)std::sin((double)phi);
    SphereObstacle<Real> s;
    for (Integer k = 0; k < 3; k++) s.c[k] = C[k] + rho*(cp*u1[k] + sp*u2[k]);
    s.r = r_sph;
    out.push_back(s);
  }
}

// ---- One obstacle per junction, at the junction center pushed a random 3D direction, staying near
// ---- the center. `centers`/`R0`/`halfwidth` are one per junction (replicated rank-0 geometry). ----
template <class Real>
void place_junction_obstacles(const std::vector<Vec3<Real>>& centers, const std::vector<Real>& R0,
                              const std::vector<Real>& halfwidth, const unsigned seed,
                              const Real radfrac, std::vector<SphereObstacle<Real>>& out) {
  srand48((long)seed + 1);   // independent stream from the arm placement
  for (size_t i = 0; i < centers.size(); i++) {
    const Real r0 = R0[i];
    if (!(r0 > 0)) continue;
    const Real r_sph = sph_detail::clampr<Real>(radfrac*r0, (Real)0.01*r0, (Real)0.45*r0);
    const Real box   = std::min<Real>(halfwidth[i], (Real)3*r0);
    Real rho_max = (Real)0.5 * (box - r_sph);
    if (!(rho_max > 0)) rho_max = 0;
    // Uniform direction on the sphere: cos(psi) uniform in [-1,1], azimuth uniform.
    const Real cz  = (Real)(2.0*drand48() - 1.0);
    const Real sz  = (Real)std::sqrt((double)std::max<Real>(0, 1 - cz*cz));
    const Real phi = (Real)(2.0*M_PI*drand48());
    const Real rho = rho_max * (Real)drand48();
    SphereObstacle<Real> s;
    s.c[0] = centers[i][0] + rho*sz*(Real)std::cos((double)phi);
    s.c[1] = centers[i][1] + rho*sz*(Real)std::sin((double)phi);
    s.c[2] = centers[i][2] + rho*cz;
    s.r = r_sph;
    out.push_back(s);
  }
}

}  // namespace quad_junctions
