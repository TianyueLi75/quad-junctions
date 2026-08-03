/**
 * Interior-flow visualization target builders (shared by the hybrid flow drivers).
 *
 * The interior velocity field of a thin-tube network cannot be visualized on a coarse uniform box (the
 * tubes are a ~1% sliver of the bounding box, so an affordable box never lands inside them). Instead we
 * sample where the fluid actually is, in two geometry-aware groups:
 *
 *   ARMS (slender tubes) -- build_arm_panel_targets(): walk the CSBQ SlenderElemList panel by panel and, at
 *     the FIRST Chebyshev node of each panel, drop a small cross-section star: the centerline point (once)
 *     plus Nrad x Naz interior points at radii {1..Nrad}*r/(Nrad+2) (so the outermost shell sits at
 *     Nrad/(Nrad+2) of the radius -- e.g. 3/5 r -- well clear of the wall). The centerline frame (C,e1,e2)
 *     and radius r at the node are recovered from four surface points via SlenderElemList::GetGeom (the only
 *     public window onto the private centerline data). GetGeom sees this rank's LOCAL panels only, so the
 *     per-rank points are Allgatherv'd; the full set is returned (replicated) on every rank.
 *
 *   JUNCTIONS (quad bodies) -- build_box_targets(): a junction body is bulky enough that a modest uniform
 *     box DOES resolve it. For each junction we sample an Nax^3 uniform grid in the cube center +- half;
 *     the caller then keeps the ones inside the surface via its Laplace-DL indicator (the "check whether
 *     inside" step). Pure rank-0 geometry (replicated inputs), so no communication here.
 *
 * SlenderElemList (a CSBQ type) is assumed already included by the translation unit (every driver does
 * `#include <csbq.hpp>` first), matching the convention of quad_junctions/hybrid_bie_tests.hpp.
 */
#pragma once

#include <sctl.hpp>
#include <cmath>
#include <cstdint>
#include <string>

namespace quad_junctions {
using namespace sctl;

// Write the first Np points of (coords Xg, values Ug, both AoS, Ug a 3-vector/point) as a VTU point cloud
// of VTK_VERTEX cells. Rank-0-only data, so written with Comm::Self(). Shared by the interior-flow drivers.
template <class Real>
void write_points_vtu(const std::string& fname, const Vector<Real>& Xg, const Vector<Real>& Ug, const Long Np) {
  VTUData vtu;
  for (Long i = 0; i < Np; i++)
    for (Integer k = 0; k < 3; k++) vtu.coord.PushBack((VTUData::VTKReal)Xg[3*i+k]);
  for (Long i = 0; i < Np; i++)
    for (Integer k = 0; k < 3; k++) vtu.value.PushBack((VTUData::VTKReal)Ug[3*i+k]);
  for (Long i = 0; i < Np; i++) {
    vtu.connect.PushBack((int32_t)i); vtu.offset.PushBack((int32_t)(i+1)); vtu.types.PushBack((uint8_t)1);  // 1 = VTK_VERTEX
  }
  vtu.WriteVTK(fname, Comm::Self());
}

// Cross-section star at the first Chebyshev node of every slender panel. Returns Xout (AoS, 3/pt),
// replicated on all ranks (gathered across the comm). Nrad radial shells x Naz angles + 1 centerline
// point per panel; rho_k = k*r/(Nrad+2), k=1..Nrad (defaults 3x5 -> radii {1,2,3}*r/5).
template <class Real>
void build_arm_panel_targets(const SlenderElemList<Real>& arms, const Comm& comm, const Long cheb,
                             Vector<Real>& Xout, const Long Nrad = 3, const Long Naz = 5) {
  const Real pi = const_pi<Real>();
  // First Chebyshev node in [0,1] for this panel order (all slender panels share `cheb` in these builds).
  const Vector<Real>& cnodes = SlenderElemList<Real>::CenterlineNodes((Integer)cheb);
  Vector<Real> s_param(1); s_param[0] = cnodes[0];
  // Four cross-section probes at theta = 0, pi/2, pi, 3pi/2 (GetGeom: y = x + e1*r*cos + e2*r*sin).
  Vector<Real> sin_th(4), cos_th(4);
  for (Integer a = 0; a < 4; a++) { const double th = 0.5*M_PI*(double)a; sin_th[a] = (Real)std::sin(th); cos_th[a] = (Real)std::cos(th); }

  Vector<Real> Xloc;
  const Long np = arms.Size();   // LOCAL panel count under MPI
  for (Long e = 0; e < np; e++) {
    Vector<Real> Xs;             // 4 surface points, AoS
    arms.GetGeom(&Xs, nullptr, nullptr, nullptr, nullptr, s_param, sin_th, cos_th, e);
    Real C[3], e1[3], e2[3];
    for (Integer k = 0; k < 3; k++) C[k] = (Xs[k] + Xs[3+k] + Xs[6+k] + Xs[9+k]) * (Real)0.25;
    Real r = 0, n2 = 0;
    for (Integer k = 0; k < 3; k++) { e1[k] = Xs[k]     - C[k]; r  += e1[k]*e1[k]; }
    for (Integer k = 0; k < 3; k++) { e2[k] = Xs[3+k]   - C[k]; n2 += e2[k]*e2[k]; }
    r = sqrt<Real>(r); const Real n = sqrt<Real>(n2);
    if (!(r > 0) || !(n > 0)) continue;
    for (Integer k = 0; k < 3; k++) { e1[k] /= r; e2[k] /= n; }
    // Centerline point (once).
    Xloc.PushBack(C[0]); Xloc.PushBack(C[1]); Xloc.PushBack(C[2]);
    for (Long ir = 1; ir <= Nrad; ir++) {
      const Real rho = (Real)ir * r / (Real)(Nrad + 2);
      for (Long ia = 0; ia < Naz; ia++) {
        const Real th = (Real)2 * pi * (Real)ia / (Real)Naz;
        const Real ca = (Real)std::cos((double)th), sa = (Real)std::sin((double)th);
        for (Integer k = 0; k < 3; k++) Xloc.PushBack(C[k] + rho*(ca*e1[k] + sa*e2[k]));
      }
    }
  }

  // Gather every rank's local points so the full field is available (used on rank 0 downstream).
  const Integer nranks = comm.Size();
  Vector<Long> scv(1); scv[0] = Xloc.Dim();
  Vector<Long> rcounts((Long)nranks);
  comm.Allgather(scv.begin(), 1, rcounts.begin(), 1);
  Vector<Long> rdispls((Long)nranks); Long tot = 0;
  for (Integer i = 0; i < nranks; i++) { rdispls[i] = tot; tot += rcounts[i]; }
  Xout.ReInit(tot);
  comm.Allgatherv(Xloc.begin(), Xloc.Dim(), Xout.begin(), rcounts.begin(), rdispls.begin());
}

// Uniform Nax^3 samples in each cube (center +- half). `centers` is AoS (3 per junction), `halves` one
// per junction. Appends to Xout (AoS, 3/pt). Caller filters to the interior afterwards. Rank-0 geometry.
template <class Real>
void build_box_targets(const Vector<Real>& centers, const Vector<Real>& halves, const Long Nax,
                       Vector<Real>& Xout) {
  const Long nj = halves.Dim();
  for (Long j = 0; j < nj; j++) {
    const Real cx = centers[3*j], cy = centers[3*j+1], cz = centers[3*j+2], h = halves[j];
    if (!(h > 0)) continue;
    for (Long ix = 0; ix < Nax; ix++) {
      const Real fx = (Nax > 1) ? (Real)ix/(Real)(Nax-1) : (Real)0.5;
      for (Long iy = 0; iy < Nax; iy++) {
        const Real fy = (Nax > 1) ? (Real)iy/(Real)(Nax-1) : (Real)0.5;
        for (Long iz = 0; iz < Nax; iz++) {
          const Real fz = (Nax > 1) ? (Real)iz/(Real)(Nax-1) : (Real)0.5;
          Xout.PushBack(cx - h + 2*h*fx);
          Xout.PushBack(cy - h + 2*h*fy);
          Xout.PushBack(cz - h + 2*h*fz);
        }
      }
    }
  }
}

}  // namespace quad_junctions
