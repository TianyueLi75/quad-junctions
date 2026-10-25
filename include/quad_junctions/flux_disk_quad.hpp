#ifndef QUAD_JUNCTIONS_FLUX_DISK_QUAD_HPP
#define QUAD_JUNCTIONS_FLUX_DISK_QUAD_HPP
/*
 * flux_disk_quad.hpp -- high-order quadrature for the cross-section flux Q = int_disk (u.t) rho drho dtheta.
 *
 * Rule: Gauss-Legendre (order Nrad, mapped from [0,1] to [0,r]) in the radial direction x periodic trapezoid
 * (Nang equispaced points) in the angular direction.  For node (i,a):
 *     rho_i = r*x_i,  theta_a = 2*pi*a/Nang,  X = C + rho_i cos(theta_a) e1 + rho_i sin(theta_a) e2,
 *     W     = rho_i * (r*omega_i) * (2*pi/Nang)        (rho_i = polar Jacobian).
 * Node ordering is ir-outer, ia-inner: j = ir*Nang + ia.
 *
 * Exactness: a Poiseuille profile gives integrand rho*(1-(rho/r)^2), a degree-3 polynomial in rho, so GL with
 * Nrad >= 2 is exact; the angular integrand is constant so Nang >= 1 is exact.  Real fields deviate smoothly,
 * so convergence is spectral in both directions.
 */
#include <sctl.hpp>
#include <cmath>
#include <vector>

namespace quad_junctions {

using sctl::Long;
using sctl::Vector;

// V3 is any 3-vector with operator[] (quad_junctions::Vec3 = std::array<Real,3>, sctl::Vec3, ...).
template <class Real, class V3>
inline void build_flux_disk(const V3& C, const V3& e1, const V3& e2,
                            Real r, sctl::Long Nrad, sctl::Long Nang,
                            sctl::Vector<Real>& X,        // APPENDS Nrad*Nang xyz triples
                            std::vector<Real>& w,         // APPENDS Nrad*Nang quadrature weights
                            std::vector<Real>& rho) {     // APPENDS Nrad*Nang rho values
  sctl::Vector<Real> nds, wts;
  sctl::LegQuadRule<Real>::ComputeNdsWts(&nds, &wts, Nrad);
  const Real dth = (Real)2 * (Real)M_PI / (Real)Nang;
  for (Long ir = 0; ir < Nrad; ir++) {
    const Real rh = r * nds[ir];
    const Real wr = rh * (r * wts[ir]) * dth;
    for (Long ia = 0; ia < Nang; ia++) {
      const double th = (double)((Real)ia * dth), cp = std::cos(th), sp = std::sin(th);
      for (int q = 0; q < 3; q++) X.PushBack(C[q] + rh * (Real)cp * e1[q] + rh * (Real)sp * e2[q]);
      w.push_back(wr);
      rho.push_back(rh);
    }
  }
}

}  // namespace quad_junctions

#endif
