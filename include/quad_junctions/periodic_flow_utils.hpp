#pragma once
/**
 * Doubly-periodic Stokes-flow support routines, ported from ../stokes-periodize-numtest
 * (include/utils_tests.cpp and include/utils_vis.cpp). Copied here rather than cross-included to avoid
 * ODR clashes with that repo's own stokes_bio (this repo has a byte-identical copy). Only the pieces the
 * cilia-carpet driver needs are ported:
 *   - bg_flow_2peri : plane-Poiseuille background flow between two z-walls.
 *   - SurfaceIntegral / AddConstVec : surface-mean density projection during the GMRES solve.
 *   - CubeVolumeVisShifted : uniform Cartesian grid over a sub-cube of the unit cell, for volume-flow VTK.
 */

#include <sctl.hpp>
#include <string>

// Singly-periodic (x) pressure-driven background flow: PIPE Poiseuille, a radial parabola about the axis
// (y,z)=(0.5,0.5) driving flow in +x. Ported from ../stokes-periodize-numtest/include/utils_tests.cpp
// (bg_flow_1peri). Its axial Laplacian is a constant (-1), i.e. a uniform x-pressure-gradient particular
// solution; the (0.5,0.5) centering is a harmonic shift absorbed by the layer potential, so it drives the
// same pressure drop regardless of where the geometry sits. Used by the singly-periodic vessels driver.
template <class Real> sctl::Vector<Real> bg_flow_1peri(const sctl::Vector<Real>& X) {
  const sctl::Long N = X.Dim() / 3;
  sctl::Vector<Real> U(N * 3);
  for (sctl::Long i = 0; i < N; i++) {
    const auto x = X.begin() + i * 3;
    U[i*3+0] = -(((x[1] - (Real)0.5) * (x[1] - (Real)0.5) + (x[2] - (Real)0.5) * (x[2] - (Real)0.5)) / (Real)4);
    U[i*3+1] = 0;
    U[i*3+2] = 0;
  }
  return U;
}

// Doubly-periodic pressure-driven background flow (plane Poiseuille between walls at z=0 and z=1).
template <class Real> sctl::Vector<Real> bg_flow_2peri(const sctl::Vector<Real>& X) {
  const sctl::Long N = X.Dim() / 3;
  sctl::Vector<Real> U(N * 3);
  for (sctl::Long i = 0; i < N; i++) {
    const auto x = X.begin() + i * 3;
    U[i*3+0] = -(Real)0.5 * ((x[2] - (Real)0.5) * (x[2] - (Real)0.5));
    U[i*3+1] = 0;
    U[i*3+2] = 0;
  }
  return U;
}

// Quadrature-weighted surface integral of a dof-fast/node-slow field into I (used for surface-mean density).
template <class Real> void SurfaceIntegral(sctl::Vector<Real>& I, const sctl::Vector<Real>& vals, const sctl::Vector<Real>& wts) {
  const sctl::Long dof = vals.Dim() / wts.Dim();
  SCTL_ASSERT(vals.Dim() == wts.Dim() * dof);
  if (I.Dim() != dof) I.ReInit(dof);
  I = 0;
  for (sctl::Long i = 0; i < wts.Dim(); i++)
    for (sctl::Long j = 0; j < dof; j++)
      I[j] += vals[i*dof + j] * wts[i];
}

// Add the per-component constant c0 to every node of a dof-fast/node-slow field (restore the surface mean).
template <class Real> void AddConstVec(sctl::Vector<Real>& vals, const sctl::Vector<Real>& c0) {
  const sctl::Long dof = c0.Dim();
  const sctl::Long N = vals.Dim() / dof;
  SCTL_ASSERT(vals.Dim() == N * dof);
  for (sctl::Long i = 0; i < N; i++)
    for (sctl::Long j = 0; j < dof; j++)
      vals[i*dof + j] += c0[j];
}

// Uniform Cartesian grid of N x N x N points on a sub-cube of edge L centered at (0.5,0.5,0.5). Partitioned
// across MPI ranks along the slowest index; emits hexahedral VTK cells. (CubeVolumeVisShifted from utils_vis.)
template <class Real> class CubeVolumeVisShifted {
  static constexpr sctl::Integer COORD_DIM = 3;
 public:
  CubeVolumeVisShifted() = default;
  CubeVolumeVisShifted(const sctl::Long N_, Real L, const sctl::Comm& comm_ = sctl::Comm::Self()) : N(N_), comm(comm_) {
    const sctl::Long pid = comm.Rank(), Np = comm.Size();
    const sctl::Long NN = sctl::pow<COORD_DIM-1, sctl::Long>(N);
    const sctl::Long a = (N-1)*(pid+0)/Np, b = (N-1)*(pid+1)/Np;
    N0 = b - a + 1;
    if (N0 < 2) return;
    coord.ReInit(N0 * NN * COORD_DIM);
    for (sctl::Long i = 0; i < N0; i++)
      for (sctl::Long j = 0; j < NN; j++)
        for (sctl::Long k = 0; k < COORD_DIM; k++) {
          const sctl::Long idx = ((i+a)*NN + j);
          coord[(i*NN+j)*COORD_DIM+k] = (((idx/sctl::pow<sctl::Long>(N,k)) % N)/(Real)(N-1) - (Real)0.5) * L + (Real)0.5;
        }
  }
  const sctl::Vector<Real>& GetCoord() const { return coord; }
  void GetVTUData(sctl::VTUData& vtu_data, const sctl::Vector<Real>& F) const {
    for (const auto& x : coord) vtu_data.coord.PushBack((float)x);
    for (const auto& x : F)     vtu_data.value.PushBack((float)x);
    for (sctl::Long i = 0; i < N0-1; i++)
      for (sctl::Long j = 0; j < N-1; j++)
        for (sctl::Long k = 0; k < N-1; k++) {
          auto idx = [this](sctl::Long i, sctl::Long j, sctl::Long k) { return (i*N+j)*N+k; };
          vtu_data.connect.PushBack(idx(i+0,j+0,k+0)); vtu_data.connect.PushBack(idx(i+0,j+0,k+1));
          vtu_data.connect.PushBack(idx(i+0,j+1,k+1)); vtu_data.connect.PushBack(idx(i+0,j+1,k+0));
          vtu_data.connect.PushBack(idx(i+1,j+0,k+0)); vtu_data.connect.PushBack(idx(i+1,j+0,k+1));
          vtu_data.connect.PushBack(idx(i+1,j+1,k+1)); vtu_data.connect.PushBack(idx(i+1,j+1,k+0));
          vtu_data.offset.PushBack(vtu_data.connect.Dim());
          vtu_data.types.PushBack(12);
        }
  }
  void WriteVTK(const std::string& fname, const sctl::Vector<Real>& F) const {
    sctl::VTUData vtu_data; GetVTUData(vtu_data, F); vtu_data.WriteVTK(fname, comm);
  }
 private:
  sctl::Long N = 0, N0 = 0;
  sctl::Comm comm;
  sctl::Vector<Real> coord;
};
