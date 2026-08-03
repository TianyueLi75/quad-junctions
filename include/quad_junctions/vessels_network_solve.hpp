/**
 * Reduced Hagen-Poiseuille RESISTANCE-NETWORK solve for the 20-junction arterial/venous vessels network.
 *
 * The network is planar and made of long thin tubes, so its flux split is well-approximated by a linear
 * resistance network: model each arm as a Hagen-Poiseuille resistor R = 8*mu*L/(pi*r^4) between its two
 * junction nodes, inject the total volumetric flux `p_in` at the arterial-tree root node and withdraw it
 * at the venous-tree root node, and solve Kirchhoff's laws (flow conservation at every junction) for the
 * nodal pressures. This yields the pressure at each junction and the flux/pressure-drop of each arm in
 * terms of `p_in` -- a cheap analytic reference against which the full boundary-integral Stokes solve
 * (src/ybifurc-vessels-flow-bie.cpp) can be checked at the mid-point of each connector arm.
 *
 * Kirchhoff -> a graph-Laplacian system  L p = b,  with  L_ii = sum_j g_ij,  L_ij = -g_ij (g = 1/R the
 * conductance), and RHS b = +p_in at the inflow root node, -p_in at the outflow root node, 0 elsewhere.
 * The Laplacian is singular (constant null space); since sum(b)=0, the minimum-norm (mean-zero-pressure)
 * solution p = L^+ b via the pseudo-inverse is exact. No node needs pinning.
 *
 * Pure serial linear algebra on tiny (n_junc x n_junc) matrices -- no MPI, no BIE. The geometry (`segs`)
 * is replicated on every rank, so callers may run this identically on all ranks and print on rank 0.
 */
#pragma once

#include <quad_junctions/vessels_build.hpp>   // ArmSeg, dot3/nrm3/sub3, sctl (Matrix, const_pi, Vec3)
#include <vector>

namespace quad_junctions {
using namespace sctl;

// One arm modelled as a resistor: seg index, junction endpoints j0->j1, tube radius r, centerline length
// L, conductance g = 1/R, solved pressure drop dP = p[j0]-p[j1] and flux Q = g*dP. `is_conn` marks the
// arterial(<10)<->venous(>=10) connector arms (the 11 mid arms the flow test probes).
template <class Real> struct NetEdge {
  int seg = -1, j0 = -1, j1 = -1;
  Real r = 0, L = 0, g = 0, dP = 0, Q = 0;
  bool is_conn = false;
};

template <class Real> struct NetworkSolution {
  std::vector<Real>          P;      // per-junction pressure (size n_junc), mean zero, proportional to p_in
  std::vector<NetEdge<Real>> edges;  // one per resistive arm (root-cap stubs excluded)
};

// Solve the resistance network. `segs` are all tube segments from build_vessels_network (intra-tree arms,
// connectors, and the two root-cap stubs which have j0==j1 and are skipped). `root_in`/`root_out` are the
// junction ids of the arterial/venous root stems (the inflow/outflow ports). `mu` is the dynamic
// viscosity (1 for Stokes; the result scales linearly with p_in/mu regardless).
template <class Real>
NetworkSolution<Real> solve_vessels_pressure_network(
    const std::vector<ArmSeg<Real>>& segs, const int n_junc,
    const int root_in, const int root_out, const Real p_in, const Real mu = (Real)1) {
  NetworkSolution<Real> out;
  out.P.assign((size_t)n_junc, (Real)0);

  Matrix<Real> Lap(n_junc, n_junc); Lap.SetZero();
  for (size_t s = 0; s < segs.size(); s++) {
    const ArmSeg<Real>& e = segs[s];
    if (e.j0 == e.j1) continue;                        // root-cap stub: a port, not a resistor
    Real Llen = 0;                                     // true bent centerline length
    for (size_t i = 0; i + 1 < e.cl.size(); i++) Llen += nrm3(sub3(e.cl[i+1], e.cl[i]));
    if (Llen <= (Real)0) Llen = nrm3(sub3(e.B, e.A));  // fall back to the chord
    const Real r = e.rtube;
    const Real R = (Real)8 * mu * Llen / (const_pi<Real>() * r * r * r * r);
    const Real g = (Real)1 / R;
    Lap[e.j0][e.j0] += g; Lap[e.j1][e.j1] += g;
    Lap[e.j0][e.j1] -= g; Lap[e.j1][e.j0] -= g;
    NetEdge<Real> ne;
    ne.seg = (int)s; ne.j0 = e.j0; ne.j1 = e.j1; ne.r = r; ne.L = Llen; ne.g = g;
    ne.is_conn = ((e.j0 < 10) != (e.j1 < 10));         // crosses the arterial(<10)/venous(>=10) split
    out.edges.push_back(ne);
  }

  Matrix<Real> b(n_junc, 1); b.SetZero();
  b[root_in][0]  =  p_in;                              // inject total flux at the arterial root
  b[root_out][0] = -p_in;                              // withdraw it at the venous root
  const Matrix<Real> P = Lap.pinv() * b;               // min-norm (mean-zero) pressures
  for (int i = 0; i < n_junc; i++) out.P[(size_t)i] = P[i][0];

  for (NetEdge<Real>& ne : out.edges) {
    ne.dP = out.P[(size_t)ne.j0] - out.P[(size_t)ne.j1];
    ne.Q  = ne.g * ne.dP;
  }
  return out;
}

}  // namespace quad_junctions
