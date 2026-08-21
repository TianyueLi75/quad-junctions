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

// ---- DENSE UNIFORM interior fill of the slender arms (mirrors ../stokes-periodize-numtest VolumeVis).
// ---- Per panel: s_order axial stations x t_order azimuth x r_order radial layers, interpolated from the
// ---- cross-section CENTROID (k=0) out to ~the surface (k=r_order-1, at (1-1e-3) of the radius). The
// ---- point order per panel is [s][theta][r] (r fastest, theta wraps), so a hex writer reconstructs the
// ---- (s,theta,r) cell lattice purely from the global panel count. Coords are Allgatherv'd so the full
// ---- set is replicated on every rank (used on rank 0 downstream). Npanel_out = global panel count. ----
template <class Real>
void build_arm_volume_targets(const SlenderElemList<Real>& arms, const Comm& comm, Vector<Real>& Xout,
                              Long& Npanel_out, const Long s_order = 6, const Long t_order = 16,
                              const Long r_order = 5) {
  const Real pi = const_pi<Real>();
  Vector<Real> s_param(s_order), sin_th(t_order), cos_th(t_order);
  for (Long i = 0; i < s_order; i++) s_param[i] = (s_order > 1) ? (Real)i/(Real)(s_order-1) : (Real)0.5;
  for (Long j = 0; j < t_order; j++) { const Real t = (Real)j/(Real)t_order;
    sin_th[j] = (Real)std::sin((double)(2*pi*t)); cos_th[j] = (Real)std::cos((double)(2*pi*t)); }
  const Real r_inv = (Real)(1 - 1e-3)/(Real)(r_order-1);

  Vector<Real> loc;
  const Long np = arms.Size();   // LOCAL panels under MPI
  for (Long e = 0; e < np; e++) {
    Vector<Real> X_;             // s_order*t_order surface points, AoS, order [s][theta]
    arms.GetGeom(&X_, nullptr, nullptr, nullptr, nullptr, s_param, sin_th, cos_th, e);
    for (Long i = 0; i < s_order; i++) {
      Real Xc[3] = {0,0,0};      // cross-section centroid = azimuthal mean at this s-station
      for (Long j = 0; j < t_order; j++) for (Integer l = 0; l < 3; l++) Xc[l] += X_[(i*t_order+j)*3+l]/(Real)t_order;
      for (Long j = 0; j < t_order; j++)
        for (Long k = 0; k < r_order; k++)
          for (Integer l = 0; l < 3; l++)
            loc.PushBack((X_[(i*t_order+j)*3+l] - Xc[l]) * (Real)k * r_inv + Xc[l]);
    }
  }

  const Integer nranks = comm.Size();
  Vector<Long> scv(1); scv[0] = loc.Dim();
  Vector<Long> rc((Long)nranks); comm.Allgather(scv.begin(), 1, rc.begin(), 1);
  Vector<Long> rd((Long)nranks); Long tot = 0;
  for (Integer i = 0; i < nranks; i++) { rd[i] = tot; tot += rc[i]; }
  Xout.ReInit(tot);
  if (tot) comm.Allgatherv(loc.begin(), loc.Dim(), Xout.begin(), rc.begin(), rd.begin());
  const Long blk = s_order*t_order*r_order;
  Npanel_out = blk ? (Xout.Dim()/3)/blk : 0;
}

// Write the arm-volume lattice (from build_arm_volume_targets) as VTK_HEXAHEDRON cells over each panel's
// (s,theta,r) grid, colored by the 3-vector field Ug. Rank-0-only data (Comm::Self()). Points are KEPT
// (never deleted) -- the caller zeroes Ug inside obstacles / outside the fluid so the solid volume stays
// intact and the obstacles read as U=0 holes in the flow.
template <class Real>
void write_arm_volume_vtu(const std::string& fname, const Vector<Real>& Xg, const Vector<Real>& Ug,
                          const Long Npanel, const Long s_order, const Long t_order, const Long r_order) {
  VTUData vtu;
  const Long Npts = Xg.Dim()/3;
  for (Long i = 0; i < Npts; i++) for (Integer k = 0; k < 3; k++) vtu.coord.PushBack((VTUData::VTKReal)Xg[3*i+k]);
  for (Long i = 0; i < Npts; i++) for (Integer k = 0; k < 3; k++) vtu.value.PushBack((VTUData::VTKReal)Ug[3*i+k]);
  for (Long l = 0; l < Npanel; l++) {
    const Long off = l * s_order*t_order*r_order;
    auto idx = [&](Long i, Long j, Long k) { return (int32_t)(off + (i*t_order + (j % t_order))*r_order + k); };
    for (Long i = 0; i < s_order-1; i++)
      for (Long j = 0; j < t_order; j++)         // theta wraps (j+1)%t_order -> closed tube
        for (Long k = 0; k < r_order-1; k++) {
          vtu.connect.PushBack(idx(i+0,j+0,k+0)); vtu.connect.PushBack(idx(i+0,j+0,k+1));
          vtu.connect.PushBack(idx(i+0,j+1,k+1)); vtu.connect.PushBack(idx(i+0,j+1,k+0));
          vtu.connect.PushBack(idx(i+1,j+0,k+0)); vtu.connect.PushBack(idx(i+1,j+0,k+1));
          vtu.connect.PushBack(idx(i+1,j+1,k+1)); vtu.connect.PushBack(idx(i+1,j+1,k+0));
          vtu.offset.PushBack((int32_t)vtu.connect.Dim()); vtu.types.PushBack((uint8_t)12);  // 12 = VTK_HEXAHEDRON
        }
  }
  vtu.WriteVTK(fname, Comm::Self());
}

// Write the junction-box lattice (from build_box_targets: Njunc blocks of an Nax^3 grid, point order
// [ix][iy][iz], iz fastest) as VTK_HEXAHEDRON cells colored by Ug. Rank-0-only. Same keep-and-zero
// convention as write_arm_volume_vtu.
template <class Real>
void write_box_hex_vtu(const std::string& fname, const Vector<Real>& Xg, const Vector<Real>& Ug,
                       const Long Njunc, const Long Nax) {
  VTUData vtu;
  const Long Npts = Xg.Dim()/3;
  for (Long i = 0; i < Npts; i++) for (Integer k = 0; k < 3; k++) vtu.coord.PushBack((VTUData::VTKReal)Xg[3*i+k]);
  for (Long i = 0; i < Npts; i++) for (Integer k = 0; k < 3; k++) vtu.value.PushBack((VTUData::VTKReal)Ug[3*i+k]);
  for (Long b = 0; b < Njunc; b++) {
    const Long off = b * Nax*Nax*Nax;
    auto idx = [&](Long i, Long j, Long k) { return (int32_t)(off + (i*Nax + j)*Nax + k); };
    for (Long i = 0; i < Nax-1; i++)
      for (Long j = 0; j < Nax-1; j++)
        for (Long k = 0; k < Nax-1; k++) {
          vtu.connect.PushBack(idx(i+0,j+0,k+0)); vtu.connect.PushBack(idx(i+0,j+0,k+1));
          vtu.connect.PushBack(idx(i+0,j+1,k+1)); vtu.connect.PushBack(idx(i+0,j+1,k+0));
          vtu.connect.PushBack(idx(i+1,j+0,k+0)); vtu.connect.PushBack(idx(i+1,j+0,k+1));
          vtu.connect.PushBack(idx(i+1,j+1,k+1)); vtu.connect.PushBack(idx(i+1,j+1,k+0));
          vtu.offset.PushBack((int32_t)vtu.connect.Dim()); vtu.types.PushBack((uint8_t)12);
        }
  }
  vtu.WriteVTK(fname, Comm::Self());
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
