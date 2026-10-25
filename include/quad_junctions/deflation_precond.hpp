#pragma once
// ---------------------------------------------------------------------------
// deflation_precond.hpp -- deflated GMRES for the capsule near-null (standing-
// wave pressure) modes of the combined-field interior Stokes operator.
//
// Krylov engine is sctl::GMRES (same solver the production BVP solves use, so
// the M=0 baseline IS the production stall and the result ports to the
// matrix-free path). PETSc supplies the DEFLATION linear algebra: the operator
// A and deflation space W as dense matrices, AW = A W, the coarse matrix
// Ac = W^T A W and its (direct) coarse solve, and all projection mat-vecs.
//
// Deflation projector (Frank & Vuik 2001):
//     P   = I - A W Ac^{-1} W^T          (Ac = W^T A W,  W orthonormal columns)
//   solve  P A x~ = P b   with sctl::GMRES,
//   recover  x = W Ac^{-1} W^T b + ( x~ - W Ac^{-1} W^T A x~ ).
// Since span(W) is spanned by right eigenvectors of A it is A-invariant, so P A
// annihilates exactly that subspace: the near-null eigenvalues are removed from
// the spectrum GMRES sees. Exact (A x = b) by construction -- see the identity
//     A x = A W Ac^{-1} W^T b + P A x~ = A W Ac^{-1} W^T b + P b = b.
//
// Orthonormalization of the raw eigenvector columns is done with LAPACK QR
// (dgeqrf/dorgqr, MKL -- robust for the possibly non-orthogonal eigenvectors of
// a non-normal A); everything else is PETSc. Compiles to nothing without
// QJ_HAVE_PETSC so the TU links in a plain (non-PETSc) build.
// ---------------------------------------------------------------------------

#include <sctl.hpp>
#include <vector>
#include <string>
#include <cmath>

#ifdef QJ_HAVE_PETSC
#include <petscksp.h>

// LAPACK QR (MKL, LP64). Column-major.
extern "C" {
  void dgeqrf_(const int* m, const int* n, double* a, const int* lda, double* tau,
               double* work, const int* lwork, int* info);
  void dtrtri_(const char* uplo, const char* diag, const int* n, double* a, const int* lda, int* info);
  void dorgqr_(const int* m, const int* n, const int* k, double* a, const int* lda,
               const double* tau, double* work, const int* lwork, int* info);
}

namespace quad_junctions {

struct DeflateResult {
  long   iters       = 0;      // sctl::GMRES iterations
  double rel_resid   = 0;      // ||b - A x|| / ||b||  (true residual)
  double rel_err     = -1;     // ||x - x_true|| / ||x_true||  (<0 if x_true not given)
  long   Mdefl       = 0;      // number of (orthonormal) deflation columns actually used
};

namespace defl_detail {

// Orthonormalize the columns of Wcm (N x Mraw, COLUMN-MAJOR) in place via LAPACK
// QR. Returns the numerical rank r (<= Mraw); the first r columns of Wcm hold an
// orthonormal basis of the original column space. Columns beyond r are left as
// Q's extra orthonormal columns (harmless, but we only USE the first r).
inline long qr_orthonormalize(std::vector<double>& Wcm, long N, long Mraw,
                                std::vector<double>* Rout = nullptr) {
  if (Mraw == 0) return 0;
  const int m = (int)N, n = (int)Mraw, lda = (int)N;
  std::vector<double> tau(Mraw);
  int info = 0, lwork = -1;
  double wq = 0;
  dgeqrf_(&m, &n, Wcm.data(), &lda, tau.data(), &wq, &lwork, &info);
  lwork = (int)wq;
  std::vector<double> work(std::max(1, lwork));
  dgeqrf_(&m, &n, Wcm.data(), &lda, tau.data(), work.data(), &lwork, &info);
  SCTL_ASSERT(info == 0);
  // Numerical rank from |R_ii| (R is in the upper triangle of Wcm, col-major).
  double rmax = 0;
  for (long j = 0; j < Mraw; j++) rmax = std::max(rmax, std::fabs(Wcm[j + j * N]));
  long r = 0;
  const double rtol = 1e-10 * (rmax > 0 ? rmax : 1.0);
  for (long j = 0; j < Mraw; j++) if (std::fabs(Wcm[j + j * N]) > rtol) r++;
  if (Rout) {  // capture r x r upper-tri R (col-major) before dorgqr overwrites it
    Rout->assign((size_t)r * r, 0.0);
    for (long j = 0; j < r; j++) for (long i = 0; i <= j; i++) (*Rout)[i + j * r] = Wcm[i + j * N];
  }
  // Form explicit Q (N x Mraw) overwriting Wcm.
  const int k = (int)Mraw;
  lwork = -1;
  dorgqr_(&m, &n, &k, Wcm.data(), &lda, tau.data(), &wq, &lwork, &info);
  lwork = (int)wq;
  work.assign(std::max(1, lwork), 0.0);
  dorgqr_(&m, &n, &k, Wcm.data(), &lda, tau.data(), work.data(), &lwork, &info);
  SCTL_ASSERT(info == 0);
  return r;
}

inline void sctl_to_vec(const sctl::Vector<double>& x, Vec v) {
  PetscScalar* a;
  PetscCallAbort(PETSC_COMM_SELF, VecGetArray(v, &a));
  const long n = x.Dim();
  for (long i = 0; i < n; i++) a[i] = x[i];
  PetscCallAbort(PETSC_COMM_SELF, VecRestoreArray(v, &a));
}
inline void vec_to_sctl(Vec v, sctl::Vector<double>& x) {
  const PetscScalar* a;
  PetscCallAbort(PETSC_COMM_SELF, VecGetArrayRead(v, &a));
  const long n = x.Dim();
  for (long i = 0; i < n; i++) x[i] = PetscRealPart(a[i]);
  PetscCallAbort(PETSC_COMM_SELF, VecRestoreArrayRead(v, &a));
}

} // namespace defl_detail

// A: dense N x N, ROW-MAJOR (A[i*N+j]).  b, x: length N.
// Wcols: N x Mraw COLUMN-MAJOR (column c = raw mode c); Mraw==0 -> plain baseline.
// Returns iteration count / residual / error. `comm` is the (serial) sctl comm.
// Optional Eang/AWang (col-major M x M, N x M): precomputed coarse matrix and A*W (path 2, no matvec);
// when given, Wcols must already be orthonormal with Mraw columns (no QR).
inline DeflateResult deflated_gmres_impl(
    const std::vector<double>& A, long N,
    const std::vector<double>& b,
    const std::vector<double>& Wcols, long Mraw,
    std::vector<double>& x,
    double tol, long max_iter,
    const std::vector<double>* x_true,
    const sctl::Comm& comm,
    const std::vector<double>* Eang, const std::vector<double>* AWang) {
  using namespace defl_detail;
  DeflateResult R;

  // ---- A as a column-major dense PETSc Mat. A is row-major: A[i*N+j]=A_{ij};
  // storing Acm[i+j*N]=A_{ij} is PETSc column-major for A itself (NOT A^T). ----
  // NOTE: MatCreateSeqDense with a user array does NOT copy -- it wraps the
  // pointer, so every backing std::vector MUST outlive its Mat. Acm and Wcm are
  // therefore both at function scope; a block-scoped Wcm is a use-after-free that
  // feeds Wpet garbage during the solve (baseline M=0 never touches Wpet, so it
  // survives -- which is exactly why only the deflated rows blew up).
  std::vector<double> Acm((size_t)N * N);
  for (long i = 0; i < N; i++)
    for (long j = 0; j < N; j++) Acm[i + j * N] = A[i * N + j];
  Mat Apet;
  PetscCallAbort(PETSC_COMM_SELF,
      MatCreateSeqDense(PETSC_COMM_SELF, N, N, Acm.data(), &Apet));
  PetscCallAbort(PETSC_COMM_SELF, MatAssemblyBegin(Apet, MAT_FINAL_ASSEMBLY));
  PetscCallAbort(PETSC_COMM_SELF, MatAssemblyEnd(Apet, MAT_FINAL_ASSEMBLY));

  Vec vin, vt, vb;
  PetscCallAbort(PETSC_COMM_SELF, MatCreateVecs(Apet, &vin, &vt));
  PetscCallAbort(PETSC_COMM_SELF, VecDuplicate(vin, &vb));
  sctl::Vector<double> bs(N);
  for (long i = 0; i < N; i++) bs[i] = b[i];
  sctl_to_vec(bs, vb);                 // vb holds b for the whole routine

  // ---- The ParallelOp handed to sctl::GMRES. ----
  Mat Wpet = nullptr, AWpet = nullptr, Ac = nullptr, AcAIJ = nullptr;
  std::vector<double> AWbk, Ebk;       // back AWpet/Ac in path 2
  std::vector<double> Wcm;             // backs Wpet -- MUST outlive it (no-copy wrap)
  KSP cksp = nullptr;
  Vec vs = nullptr, vu = nullptr, vc = nullptr;  // M-length coarse work + N-length tmp
  long M = 0;

  sctl::GMRES<double> solver(comm, /*verbose=*/false);
  using Op = std::function<void(sctl::Vector<double>*, const sctl::Vector<double>&)>;
  Op Aop, PAop;

  // Plain operator:  out = A in.
  Aop = [&](sctl::Vector<double>* out, const sctl::Vector<double>& in) {
    sctl_to_vec(in, vin);
    PetscCallAbort(PETSC_COMM_SELF, MatMult(Apet, vin, vt));
    if (out->Dim() != N) out->ReInit(N);
    vec_to_sctl(vt, *out);
  };

  if (Mraw > 0) {
    // ---- orthonormalize W (LAPACK QR), keep first r columns ----
    Wcm = Wcols;                                     // copy (QR overwrites)
    long r = AWang ? Mraw : qr_orthonormalize(Wcm, N, Mraw);
    M = r;
    // trim to M columns (col-major: first M*N entries are the first M columns)
    Wcm.resize((size_t)M * N);
    PetscCallAbort(PETSC_COMM_SELF,
        MatCreateSeqDense(PETSC_COMM_SELF, N, M, Wcm.data(), &Wpet));
    PetscCallAbort(PETSC_COMM_SELF, MatAssemblyBegin(Wpet, MAT_FINAL_ASSEMBLY));
    PetscCallAbort(PETSC_COMM_SELF, MatAssemblyEnd(Wpet, MAT_FINAL_ASSEMBLY));
    if (AWang) {   // path 2: analytic AW and E (backings AWbk/Ebk outlive the Mats)
      AWbk = *AWang; Ebk = *Eang;
      PetscCallAbort(PETSC_COMM_SELF, MatCreateSeqDense(PETSC_COMM_SELF, N, M, AWbk.data(), &AWpet));
      PetscCallAbort(PETSC_COMM_SELF, MatAssemblyBegin(AWpet, MAT_FINAL_ASSEMBLY));
      PetscCallAbort(PETSC_COMM_SELF, MatAssemblyEnd(AWpet, MAT_FINAL_ASSEMBLY));
      PetscCallAbort(PETSC_COMM_SELF, MatCreateSeqDense(PETSC_COMM_SELF, M, M, Ebk.data(), &Ac));
      PetscCallAbort(PETSC_COMM_SELF, MatAssemblyBegin(Ac, MAT_FINAL_ASSEMBLY));
      PetscCallAbort(PETSC_COMM_SELF, MatAssemblyEnd(Ac, MAT_FINAL_ASSEMBLY));
    } else {
    // AW = A W
    PetscCallAbort(PETSC_COMM_SELF,
        MatMatMult(Apet, Wpet, MAT_INITIAL_MATRIX, PETSC_DEFAULT, &AWpet));
    // Ac = W^T (A W)
    PetscCallAbort(PETSC_COMM_SELF,
        MatTransposeMatMult(Wpet, AWpet, MAT_INITIAL_MATRIX, PETSC_DEFAULT, &Ac));
    }
    // coarse solve: direct LU (convert tiny dense Ac -> AIJ so PCLU is bulletproof)
    PetscCallAbort(PETSC_COMM_SELF, MatConvert(Ac, MATSEQAIJ, MAT_INITIAL_MATRIX, &AcAIJ));
    PetscCallAbort(PETSC_COMM_SELF, KSPCreate(PETSC_COMM_SELF, &cksp));
    PetscCallAbort(PETSC_COMM_SELF, KSPSetOperators(cksp, AcAIJ, AcAIJ));
    PetscCallAbort(PETSC_COMM_SELF, KSPSetType(cksp, KSPPREONLY));
    { PC pc; PetscCallAbort(PETSC_COMM_SELF, KSPGetPC(cksp, &pc));
      PetscCallAbort(PETSC_COMM_SELF, PCSetType(pc, PCLU)); }
    PetscCallAbort(PETSC_COMM_SELF, KSPSetUp(cksp));
    PetscCallAbort(PETSC_COMM_SELF, MatCreateVecs(Ac, &vs, &vu)); // length M
    PetscCallAbort(PETSC_COMM_SELF, VecDuplicate(vin, &vc));      // length N

    // out = P A in = A in - A W Ac^{-1} W^T (A in)
    PAop = [&](sctl::Vector<double>* out, const sctl::Vector<double>& in) {
      sctl_to_vec(in, vin);
      PetscCallAbort(PETSC_COMM_SELF, MatMult(Apet, vin, vt));         // vt = A in
      PetscCallAbort(PETSC_COMM_SELF, MatMultTranspose(Wpet, vt, vs)); // vs = W^T vt
      PetscCallAbort(PETSC_COMM_SELF, KSPSolve(cksp, vs, vu));         // vu = Ac^{-1} vs
      PetscCallAbort(PETSC_COMM_SELF, MatMult(AWpet, vu, vc));         // vc = AW vu
      PetscCallAbort(PETSC_COMM_SELF, VecAXPY(vt, -1.0, vc));          // vt = P A in
      if (out->Dim() != N) out->ReInit(N);
      vec_to_sctl(vt, *out);
    };
  }
  R.Mdefl = M;

  // ---- RHS, solve, recover ----  (bs already holds b)
  sctl::Vector<double> xs(N);
  xs.SetZero();
  sctl::Long iters = 0;

  if (M == 0) {
    solver(&xs, Aop, bs, tol, (sctl::Integer)max_iter, false, &iters, nullptr);
  } else {
    // u0 = Ac^{-1} W^T b ;  Pb = b - AW u0
    sctl_to_vec(bs, vb);
    PetscCallAbort(PETSC_COMM_SELF, MatMultTranspose(Wpet, vb, vs));
    PetscCallAbort(PETSC_COMM_SELF, KSPSolve(cksp, vs, vu));   // vu = u0
    PetscCallAbort(PETSC_COMM_SELF, MatMult(AWpet, vu, vc));
    PetscCallAbort(PETSC_COMM_SELF, VecWAXPY(vt, -1.0, vc, vb)); // vt = Pb
    sctl::Vector<double> Pb(N); vec_to_sctl(vt, Pb);
    // xc = W u0
    PetscCallAbort(PETSC_COMM_SELF, MatMult(Wpet, vu, vc));
    sctl::Vector<double> xc(N); vec_to_sctl(vc, xc);
    // solve P A x~ = Pb
    sctl::Vector<double> xt(N); xt.SetZero();
    solver(&xt, PAop, Pb, tol, (sctl::Integer)max_iter, false, &iters, nullptr);
    // correction: x = xc + x~ - W Ac^{-1} W^T (A x~)
    sctl_to_vec(xt, vin);
    PetscCallAbort(PETSC_COMM_SELF, MatMult(Apet, vin, vt));          // A x~
    PetscCallAbort(PETSC_COMM_SELF, MatMultTranspose(Wpet, vt, vs));
    PetscCallAbort(PETSC_COMM_SELF, KSPSolve(cksp, vs, vu));
    PetscCallAbort(PETSC_COMM_SELF, MatMult(Wpet, vu, vc));           // W Ac^{-1} W^T A x~
    sctl::Vector<double> corr(N); vec_to_sctl(vc, corr);
    for (long i = 0; i < N; i++) xs[i] = xc[i] + xt[i] - corr[i];
  }
  R.iters = (long)iters;

  // ---- diagnostics: true relative residual, and error vs x_true ----
  x.assign(N, 0.0);
  for (long i = 0; i < N; i++) x[i] = xs[i];
  { sctl_to_vec(xs, vin);
    PetscCallAbort(PETSC_COMM_SELF, MatMult(Apet, vin, vt));  // A x
    PetscCallAbort(PETSC_COMM_SELF, VecAXPY(vt, -1.0, vb));   // A x - b
    PetscReal rn = 0, bn = 0;
    PetscCallAbort(PETSC_COMM_SELF, VecNorm(vt, NORM_2, &rn));
    PetscCallAbort(PETSC_COMM_SELF, VecNorm(vb, NORM_2, &bn));
    R.rel_resid = (bn > 0) ? (double)(rn / bn) : (double)rn;
  }
  if (x_true) {
    double en = 0, tn = 0;
    for (long i = 0; i < N; i++) { double d = x[i] - (*x_true)[i];
      en += d * d; tn += (*x_true)[i] * (*x_true)[i]; }
    R.rel_err = (tn > 0) ? std::sqrt(en / tn) : std::sqrt(en);
  }

  // ---- teardown ----
  VecDestroy(&vin); VecDestroy(&vt); VecDestroy(&vb);
  MatDestroy(&Apet);
  if (M > 0) {
    VecDestroy(&vs); VecDestroy(&vu); VecDestroy(&vc);
    KSPDestroy(&cksp);
    MatDestroy(&Wpet); MatDestroy(&AWpet); MatDestroy(&Ac); MatDestroy(&AcAIJ);
  }
  return R;
}

// Path 1 / baseline: true AW = A W computed by PETSc.
inline DeflateResult deflated_gmres_petsc(
    const std::vector<double>& A, long N, const std::vector<double>& b,
    const std::vector<double>& Wcols, long Mraw, std::vector<double>& x,
    double tol, long max_iter, const std::vector<double>* x_true, const sctl::Comm& comm) {
  return deflated_gmres_impl(A, N, b, Wcols, Mraw, x, tol, max_iter, x_true, comm, nullptr, nullptr);
}

// Path 2: analytic AW, no matvec with A for the coarse space. W_raw = W' R (QR); model "raw columns are
// eigenvectors with eigenvalues D": E = R D R^{-1}, AW = W' E.
inline DeflateResult deflated_gmres_petsc_analyticAW(
    const std::vector<double>& A, long N, const std::vector<double>& b,
    const std::vector<double>& Wraw, long Mraw, const std::vector<double>& lambda_raw,
    std::vector<double>& x, double tol, long max_iter,
    const std::vector<double>* x_true, const sctl::Comm& comm) {
  using namespace defl_detail;
  std::vector<double> Wq = Wraw, Rf;
  const long r = qr_orthonormalize(Wq, N, Mraw, &Rf);
  Wq.resize((size_t)r * N);
  // Rinv via dtrtri (upper, non-unit)
  std::vector<double> Ri = Rf;
  { const int rr = (int)r, info_dummy = 0; int info = info_dummy;
    dtrtri_("U", "N", &rr, Ri.data(), &rr, &info); SCTL_ASSERT(info == 0); }
  // E = R D Rinv : (R D)[i,j] = R[i,j]*lam[j]
  std::vector<double> RD((size_t)r * r), E((size_t)r * r, 0.0);
  for (long j = 0; j < r; j++) for (long i = 0; i < r; i++) RD[i + j * r] = Rf[i + j * r] * lambda_raw[j];
  for (long j = 0; j < r; j++) for (long k = 0; k < r; k++) { const double v = Ri[k + j * r];
    if (v != 0) for (long i = 0; i < r; i++) E[i + j * r] += RD[i + k * r] * v; }
  // AW = W' E
  std::vector<double> AW((size_t)N * r, 0.0);
  for (long j = 0; j < r; j++) for (long k = 0; k < r; k++) { const double e = E[k + j * r];
    const double* wk = &Wq[(size_t)k * N]; double* aj = &AW[(size_t)j * N];
    for (long i = 0; i < N; i++) aj[i] += wk[i] * e; }
  return deflated_gmres_impl(A, N, b, Wq, r, x, tol, max_iter, x_true, comm, &E, &AW);
}

} // namespace quad_junctions
#endif // QJ_HAVE_PETSC
