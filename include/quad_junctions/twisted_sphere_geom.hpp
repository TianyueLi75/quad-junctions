#ifndef QUAD_JUNCTIONS_TWISTED_SPHERE_GEOM_HPP
#define QUAD_JUNCTIONS_TWISTED_SPHERE_GEOM_HPP

/**
 * Twisted cubed-sphere geometry (pure QuadElemList).
 *
 * A cubed sphere of radius `Radius` -- PatchPerFace^2 quad patches per cube face, `ElemOrder`
 * GL nodes per parametric direction -- with a height-dependent twist about the z axis: at height
 * z, the {x,y} coordinates are rotated by an angle theta(z) = theta_twist * z. theta_twist = 0
 * recovers the plain cubed sphere.
 *
 * Ported verbatim from the SCTL fork's SCTL_quad_element/src/test-quad-elem.cpp (BuildTwistedSphere);
 * FacePoint (gnomonic cube-face -> sphere map) is reused from collar_mount_geom.hpp so there is one
 * source of truth for the cubed-sphere map.
 *
 * Every rank builds the full node array X, then the QuadElemList ctor keeps only this rank's
 * contiguous element slice (replicate-then-slice partitioning) -- same convention as the other
 * BuildY and add_cubedsphere geometry builders.
 */

#include <sctl.hpp>
#include <quad_junctions/collar_mount_geom.hpp>   // FacePoint (cubed-sphere gnomonic map)

namespace quad_junctions {
using namespace sctl;

template <class Real>
QuadElemList<Real> BuildTwistedSphere(Long ElemOrder, Long PatchPerFace, Real Radius,
                                      Real theta_twist = 0., const Comm& comm = Comm::Self()) {
  Vector<Real> X;
  const Vector<Real>& nds = QuadElemList<Real>::ParamNodes(ElemOrder);
  for (Integer face = 0; face < 6; face++) {
    for (Long iu = 0; iu < PatchPerFace; iu++) {
      for (Long iv = 0; iv < PatchPerFace; iv++) {
        for (Long i = 0; i < ElemOrder; i++) {
          const Real u = (iu + nds[i]) / (Real)PatchPerFace;
          const Real a = 2 * u - 1;
          for (Long j = 0; j < ElemOrder; j++) {
            const Real v = (iv + nds[j]) / (Real)PatchPerFace;
            const Real b = 2 * v - 1;

            Real x, y, z;
            FacePoint<Real>(x, y, z, face, a, b, Radius);
            const Real sin_theta = sin<Real>(theta_twist * z);
            const Real cos_theta = cos<Real>(theta_twist * z);
            X.PushBack(x * cos_theta + y * sin_theta);
            X.PushBack(-x * sin_theta + y * cos_theta);
            X.PushBack(z);
          }
        }
      }
    }
  }
  return QuadElemList<Real>(ElemOrder, X, comm);
}

}  // namespace quad_junctions

#endif  // QUAD_JUNCTIONS_TWISTED_SPHERE_GEOM_HPP
