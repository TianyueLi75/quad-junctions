#pragma once
// Default singular/near quadrature scheme for quad-junctions drivers.
//
// As of 2026-08-04 the standard scheme for parameter sweeps ACROSS THE BOARD is the ported Duffy
// scheme (edge-collapsed self + upstream split-at-foot near, runtime-digits driven). On the Y120
// junction sweep it dominates the RectPolar-self/graded-near "Hybrid" path on the Stokes identities
// (~4x on the DL const-density identity, ~10x on Green's) and is on-par-to-slightly-worse on Laplace
// on the distorted junction panels -- so it does NOT shrink the mesh a Laplace target needs, but it is
// the uniform default. Set env SCTL_SELF_SCHEME=hybrid to opt any driver back to the old Hybrid path.
//
// Drivers pass the returned scheme straight into QuadElemList::SetQuadScheme(scheme, q, Nbeta, max_depth);
// the Duffy path ignores q/Nbeta (its self/near orders come from the operator tol / runtime digits) but
// still honours max_depth, so existing (q, Nbeta, max_depth) call arguments remain valid.
#include <sctl/experimental/quad_element.hpp>
#include <cstdlib>
#include <cstring>

namespace quad_junctions {

template <class Real>
inline typename sctl::QuadElemList<Real>::QuadScheme QJDefaultScheme() {
  using QS = typename sctl::QuadElemList<Real>::QuadScheme;
  const char* e = std::getenv("SCTL_SELF_SCHEME");
  if (e && !std::strcmp(e, "hybrid")) return QS::Hybrid;
  return QS::Duffy;
}

}  // namespace quad_junctions
