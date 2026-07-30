// =============================================================================
// fmm_kernels.hpp -- PVFMM translation-kernel selection for BoundaryIntegralOp.
//
// WHY THIS EXISTS
// BoundaryIntegralOp's default FMM setup (sctl/boundary_integral.txx:556) reuses
// the operator's own kernel for the M2M/M2L/L2L translations:
//
//     fmm.SetKernels(ker, ker, ker);
//
// That is only valid for a kernel with no normal vector. For a double-layer
// kernel (NormalDim=3) ParticleFMM::SetKernels registers it as
// PVFMMKernelFn<Ker> with use_dummy_normal=false, so the functor reads
// SrcDim+NormalDim values while the pvfmm::Kernel is *declared* with
// ker_dim=(SrcDim,TrgDim). pvfmm then sizes its interaction matrix for the
// smaller dim and GenericKernel::BuildMatrix writes past it -- a heap buffer
// overflow (ASan: write at pvfmm/include/kernel.txx:1108). It is invisible
// without PVFMM because EvalDirect never touches the translation kernels.
//
// The fix is to pass normal-free translation kernels explicitly via SetFMMKer.
// The choices below are exactly the ones used by:
//   - StokesBIO   (include/stokes_bio.cpp:30-31), and
//   - CSBQ bvp_solve (extern/CSBQ/test/bvp-solve.cpp:369,373).
//
// Calling this is FREE when PVFMM is off: EvalDirect never reads the M2M/M2L/
// L2L kernels, so a non-PVFMM build is bit-for-bit unchanged. It is a no-op
// unless SCTL_HAVE_PVFMM is defined.
//
// NOTE: StokesBIO additionally passes a volume-potential correction on its
// single-layer operator (stokes_sl_volpot), which only matters for XYZ-periodic
// runs. Nothing in this repo is periodic, and CSBQ's bvp_solve likewise passes
// none, so SetPVFMMKer omits it.
// =============================================================================
#ifndef _QUAD_JUNCTIONS_FMM_KERNELS_HPP_
#define _QUAD_JUNCTIONS_FMM_KERNELS_HPP_

#include <sctl.hpp>

namespace quad_junctions {
using namespace sctl;

// Translation kernels for a given layer kernel: M2M/M2L/M2T use `M2M`, L2L/L2T use `L2L`.
template <class Ker> struct FMMTransKer;

// Laplace: the multipole/local equivalent density is a plain scalar charge, so every
// translation is the single-layer kernel (CSBQ bvp-solve.cpp:373 passes Laplace3D_FxU x5).
template <> struct FMMTransKer<Laplace3D_FxU> { using M2M = Laplace3D_FxU; using L2L = Laplace3D_FxU; };
template <> struct FMMTransKer<Laplace3D_DxU> { using M2M = Laplace3D_FxU; using L2L = Laplace3D_FxU; };

// Stokes: the double-layer multipole needs the single-layer + source/sink kernel FSxU
// ("required for FMM translations involving double-layer - M2M, M2L, M2T", per the
// kernel_functions.hpp comment); the local expansion is plain FxU.
template <> struct FMMTransKer<Stokes3D_FxU> { using M2M = Stokes3D_FxU;  using L2L = Stokes3D_FxU; };
template <> struct FMMTransKer<Stokes3D_DxU> { using M2M = Stokes3D_FSxU; using L2L = Stokes3D_FxU; };

/**
 * Give `op` PVFMM-safe translation kernels. No-op without SCTL_HAVE_PVFMM.
 * Must be called before the first ComputePotential (SetFMMKer rebuilds the FMM kernels).
 */
template <class Real, class Ker> inline void SetPVFMMKer(BoundaryIntegralOp<Real, Ker>& op) {
  #ifdef SCTL_HAVE_PVFMM
  const Ker ker;
  const typename FMMTransKer<Ker>::M2M m2m;
  const typename FMMTransKer<Ker>::L2L l2l;
  op.SetFMMKer(ker, ker, ker, m2m, m2m, m2m, l2l, l2l);
  #else
  (void)op;
  #endif
}

}  // namespace quad_junctions

#endif
