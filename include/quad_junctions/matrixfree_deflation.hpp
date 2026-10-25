#pragma once
// ---------------------------------------------------------------------------
// matrixfree_deflation.hpp -- MPI-distributed, MATRIX-FREE deflated GMRES
// (Frank & Vuik 2001, DEF1) for the production interior-Stokes solve.
//
// deflation_precond.hpp solves a DENSE operator dumped to disk, serially, with
// PETSc -- fine for the small racetrack study but hopeless at the vessels
// production resolution (tens of millions of DOF, no dense A, PVFMM far field,
// MPI). This header instead takes the operator A as the solve's OWN matrix-free
// matvec (a std::function, `ApplyK` with completion and all), and the deflation
// basis W as THIS RANK's local column slice, so deflation rides inside the
// distributed production GMRES. No dense A, no PETSc -- the only dense objects
// are W^T W (M x M) and the coarse matrix Ac = W^T A W (M x M), both tiny (M ~
// tens), handled by MKL LAPACK which every build already links.
//
//   P      = I - A W Ac^{-1} W^T            (Ac = W^T A W,  W orthonormal cols)
//   solve    P A x~ = P b                   with sctl::GMRES (the production engine)
//   recover  x = W Ac^{-1} W^T b + ( x~ - W Ac^{-1} W^T A x~ ).
// Exact (A x = b) by construction for any full-column-rank W; span(W) spanning
// the near-null right eigenvectors removes those eigenvalues from the spectrum
// GMRES sees. Each coarse apply is one M-length Allreduce (W^T v) + one local
// M x M LU solve + one local W/AW gemv -- cheap next to a single A matvec.
//
// W is orthonormalized GLOBALLY by a Gram-matrix eigen-whitening, a
// distributed-friendly surrogate for the dense QR in deflation_precond.hpp:
//   G = W^T W  (M x M, Allreduce);  G = V L V^T (dsyev);  keep L > rtol*Lmax;
//   T = V_keep L_keep^{-1/2};  W <- W T   (local gemv, same T on every rank).
// After this W^T W = I globally and rank-deficient raw columns are dropped
// consistently across ranks (G is global).
//
// Everything is double (production Real = double); the N-length work vectors are
// sctl::Vector<Real> to interface with the operator and GMRES, the small dense
// pieces std::vector<double> for LAPACK.
// ---------------------------------------------------------------------------

#include <sctl.hpp>
#include <quad_junctions/mpi_utils.hpp>   // GlobalReduce(Vector<double>&, comm, op)
#include <functional>
#include <vector>
#include <cmath>
#include <algorithm>

// MKL LAPACK (LP64, column-major) -- same linkage deflation_precond.hpp relies on.
extern "C" {
  void dsyev_(const char* jobz, const char* uplo, const int* n, double* a, const int* lda,
              double* w, double* work, const int* lwork, int* info);
  void dgetrf_(const int* m, const int* n, double* a, const int* lda, int* ipiv, int* info);
  void dgetrs_(const char* trans, const int* n, const int* nrhs, const double* a, const int* lda,
               const int* ipiv, double* b, const int* ldb, int* info);
}

namespace quad_junctions {

struct MatfreeDeflateResult {
  long   iters     = 0;   // sctl::GMRES iterations on the deflated (or plain) system
  double rel_resid = 0;   // ||b - A x|| / ||b||  (true residual, globally reduced)
  long   Mdefl     = 0;   // orthonormal deflation columns actually used (<= Mraw)
};

// Column-block partition of W, supplied to exploit the LAYERED structure of the basis during whitening.
// The layered per-arm basis is a set of blocks with MUTUALLY DISJOINT dof supports (each arm's columns are
// zero off that arm's nodes), optionally preceded by ONE global flux column that overlaps every arm. So
// G = W^T W is block-diagonal among the arm blocks (plus the flux row/col). With this partition the whitening
// decouples into one tiny per-arm dsyev over that arm's dofs, instead of a single dense Mraw x Mraw dsyev
// over all Nloc rows -- the dense setup cost (the "eigendecomposition" region) drops ~by the arm count.
//   flux_first : column 0 of W is the global coupling (flux) column; the arm blocks follow it contiguously.
//   cols[b]    : number of columns in arm block b (flux excluded); sum(cols) + flux_first == Mraw.
//   dofs[b]    : this rank's LOCAL row indices (0..Nloc) where arm block b's columns are supported.
//   uniform_blocks : every arm block is congruent (same aspect L/R -> same n_max -> same column count and
//                    the same mode shapes in normalized arclength), so its Gram is block-independent. When
//                    set, block_whiten computes the whitening transform ONCE from block 0 (one Gram + one
//                    dsyev + one Allreduce) and reuses it for every arm, instead of repeating that per block.
struct DeflateBlocks {
  bool                            flux_first = false;
  bool                            uniform_blocks = false;
  std::vector<long>               cols;
  std::vector<std::vector<long>>  dofs;
};

namespace mfdefl_detail {

// m[j] = sum_i W[:,j] . v  (local partial), then Allreduce -> replicated length-M vector.
template <class Real>
inline std::vector<double> Wt_apply(const std::vector<double>& Wcm, long Nloc, long M,
                                    const sctl::Vector<Real>& v, const sctl::Comm& comm) {
  sctl::Vector<double> m((sctl::Long)M); m = 0;
  for (long j = 0; j < M; j++) {
    const double* wj = &Wcm[(size_t)j * Nloc];
    double s = 0;
    for (long i = 0; i < Nloc; i++) s += wj[i] * (double)v[i];
    m[(sctl::Long)j] = s;
  }
  GlobalReduce(m, comm, sctl::CommOp::SUM);
  std::vector<double> out((size_t)M);
  for (long j = 0; j < M; j++) out[(size_t)j] = m[(sctl::Long)j];
  return out;
}

// out = C y  where C is a local Nloc x M col-major matrix (W or AW) and y is length M.
template <class Real>
inline void mat_apply(const std::vector<double>& Ccm, long Nloc, long M,
                      const std::vector<double>& y, sctl::Vector<Real>& out) {
  if (out.Dim() != Nloc) out.ReInit((sctl::Long)Nloc);
  out = 0;
  for (long j = 0; j < M; j++) {
    const double* cj = &Ccm[(size_t)j * Nloc];
    const double yj = y[(size_t)j];
    if (yj == 0) continue;
    for (long i = 0; i < Nloc; i++) out[i] += (Real)(cj[i] * yj);
  }
}

// Block-structured Gram whitening (exploits the DeflateBlocks partition). Whitens each arm block with its
// own small dsyev over that arm's dofs only -- the arm columns stay mutually orthonormal by disjoint support
// -- then Gram-Schmidts the single flux column against them. Produces the SAME span / deflation projector as
// the monolithic whitening, but the dense work drops from O(Nloc*Mraw^2) to O(sum_b nnz_b*cb^2): each arm's
// dofs are a fraction of Nloc and each block size cb << Mraw. W is overwritten in place with the Nloc x Mkeep
// orthonormal basis (arm columns first, flux last). Returns Mkeep. Rank-deficient columns are dropped per
// block at 1e-10 relative, the flux column dropped if it lies (numerically) in the arm span.
// Form the whitening transform for ONE block: Gram Gb = Wb^T Wb over this block's dofs (reduced across ranks),
// eigendecompose, and return the kept eigenvectors scaled by 1/sqrt(lambda) as a cb x keptn col-major matrix T,
// so a whitened column over the block's dofs is W_block * T(:,q). Largest eigenvalue first, dropping columns
// below rtol_rel*lmax. Returns keptn via the out param.
inline void build_block_transform(const std::vector<double>& Wcm, long Nloc, long off, long cb,
                                   const std::vector<long>& d, const sctl::Comm& comm, double rtol_rel,
                                   std::vector<double>& T, long& keptn) {
  sctl::Vector<double> Gv((sctl::Long)(cb * cb)); Gv = 0;
  for (long a = 0; a < cb; a++) {
    const double* wa = &Wcm[(size_t)(off + a) * Nloc];
    for (long c = a; c < cb; c++) {
      const double* wc = &Wcm[(size_t)(off + c) * Nloc];
      double s = 0; for (long idx : d) s += wa[(size_t)idx] * wc[(size_t)idx];
      Gv[(sctl::Long)(a + c * cb)] = s; Gv[(sctl::Long)(c + a * cb)] = s;
    }
  }
  GlobalReduce(Gv, comm, sctl::CommOp::SUM);
  std::vector<double> G((size_t)cb * cb);
  for (long i = 0; i < cb * cb; i++) G[(size_t)i] = Gv[(sctl::Long)i];
  std::vector<double> lam((size_t)cb);
  { const int n = (int)cb; int info = 0, lwork = -1; double wq = 0;
    dsyev_("V", "U", &n, G.data(), &n, lam.data(), &wq, &lwork, &info);
    lwork = (int)wq; std::vector<double> work((size_t)std::max(1, lwork));
    dsyev_("V", "U", &n, G.data(), &n, lam.data(), work.data(), &lwork, &info);
    SCTL_ASSERT_MSG(info == 0, "block_whiten: dsyev failed"); }
  const double lmax = lam[(size_t)cb - 1];
  const double rtol = rtol_rel * (lmax > 0 ? lmax : 1.0);
  T.clear(); keptn = 0;
  for (long kk = cb - 1; kk >= 0; kk--) {          // ascending eigenvalues: largest first
    if (lam[(size_t)kk] <= rtol) break;            // and all remaining are smaller -> stop
    const double inv = 1.0 / std::sqrt(lam[(size_t)kk]);
    const double* vk = &G[(size_t)kk * cb];        // eigenvector (col-major)
    T.resize((size_t)(keptn + 1) * cb);
    double* tq = &T[(size_t)keptn * cb];
    for (long a = 0; a < cb; a++) tq[(size_t)a] = vk[(size_t)a] * inv;
    keptn++;
  }
}

template <class Real>
inline long block_whiten(std::vector<double>& Wcm, long Nloc, long Mraw,
                         const DeflateBlocks& B, const sctl::Comm& comm) {
  (void)Mraw;
  const double rtol_rel = 1e-10;
  std::vector<double> Worth((size_t)Nloc * Mraw, 0.0);
  long Mkeep = 0;
  long off = B.flux_first ? 1 : 0;

  // Congruent arm blocks (uniform L/R): the Gram is block-independent, so build the whitening transform ONCE
  // from the first block and reuse it for every arm -- one Gram/dsyev/Allreduce instead of one per block. The
  // deflation span is unchanged (each block still spans the same per-arm mode subspace); only the in-span O.N.
  // basis of blocks != block0 is approximate, which deflation recovery is insensitive to.
  std::vector<double> shared_T; long shared_kept = 0, shared_cb = 0; bool have_shared = false;
  if (B.uniform_blocks) {
    size_t b0 = 0; while (b0 < B.cols.size() && B.cols[b0] <= 0) b0++;
    if (b0 < B.cols.size()) {
      shared_cb = B.cols[b0];
      for (size_t b = 0; b < B.cols.size(); b++)
        SCTL_ASSERT_MSG(B.cols[b] <= 0 || B.cols[b] == shared_cb,
                        "block_whiten: uniform_blocks set but arm blocks differ in column count");
      long off0 = B.flux_first ? 1 : 0;
      for (size_t b = 0; b < b0; b++) off0 += B.cols[b];
      build_block_transform(Wcm, Nloc, off0, shared_cb, B.dofs[b0], comm, rtol_rel, shared_T, shared_kept);
      have_shared = true;
    }
  }

  for (size_t b = 0; b < B.cols.size(); b++) {
    const long cb = B.cols[b];
    if (cb <= 0) continue;
    const std::vector<long>& d = B.dofs[b];
    // Whitening transform for this block: the shared one (uniform blocks) or its own small Gram + dsyev.
    const double* T; long keptn; std::vector<double> Tlocal;
    if (have_shared) { T = shared_T.data(); keptn = shared_kept; }   // cb == shared_cb asserted above
    else { build_block_transform(Wcm, Nloc, off, cb, d, comm, rtol_rel, Tlocal, keptn); T = Tlocal.data(); }
    // Apply it over THIS block's dofs: whitened column q = W_block * T(:,q) (inv already folded into T).
    for (long q = 0; q < keptn; q++) {
      const double* tq = &T[(size_t)q * cb];
      double* ok = &Worth[(size_t)Mkeep * Nloc];   // write only this block's dofs; rest stay zero
      for (long idx : d) {
        double acc = 0;
        for (long a = 0; a < cb; a++) acc += Wcm[(size_t)(off + a) * Nloc + (size_t)idx] * tq[(size_t)a];
        ok[(size_t)idx] = acc;
      }
      Mkeep++;
    }
    off += cb;
  }
  // Flux column: project against the (now orthonormal) arm columns, normalize, keep unless ~0 (i.e. unless it
  // is already spanned by the arm blocks). One Gram-Schmidt sweep suffices because the arm columns are O.N.
  if (B.flux_first) {
    const double* f0 = &Wcm[0];
    std::vector<double> f((size_t)Nloc); double nf0 = 0;
    for (long i = 0; i < Nloc; i++) { f[(size_t)i] = f0[i]; nf0 += f0[i] * f0[i]; }
    nf0 = GlobalReduce(nf0, comm, sctl::CommOp::SUM);
    sctl::Vector<double> al((sctl::Long)std::max<long>(Mkeep, 1)); al = 0;
    for (long q = 0; q < Mkeep; q++) {
      const double* wq = &Worth[(size_t)q * Nloc];
      double s = 0; for (long i = 0; i < Nloc; i++) s += wq[i] * f[(size_t)i];
      al[(sctl::Long)q] = s;
    }
    if (Mkeep > 0) GlobalReduce(al, comm, sctl::CommOp::SUM);
    for (long q = 0; q < Mkeep; q++) {
      const double a = al[(sctl::Long)q]; if (a == 0) continue;
      const double* wq = &Worth[(size_t)q * Nloc];
      for (long i = 0; i < Nloc; i++) f[(size_t)i] -= a * wq[i];
    }
    double nf = 0; for (long i = 0; i < Nloc; i++) nf += f[(size_t)i] * f[(size_t)i];
    nf = GlobalReduce(nf, comm, sctl::CommOp::SUM);
    if (nf0 > 0 && nf > rtol_rel * nf0) {          // flux retains content outside the arm span
      const double inv = 1.0 / std::sqrt(nf);
      double* ok = &Worth[(size_t)Mkeep * Nloc];
      for (long i = 0; i < Nloc; i++) ok[i] = f[(size_t)i] * inv;
      Mkeep++;
    }
  }
  Worth.resize((size_t)Nloc * Mkeep);
  Wcm.swap(Worth);
  return Mkeep;
}

} // namespace mfdefl_detail

// Distributed matrix-free deflated GMRES.
//   Aop    : the matrix-free operator (local in/out, MPI-collective inside, same on every rank).
//   b, x   : local RHS / solution slices (length Nloc = b.Dim()).
//   Wcm    : THIS rank's deflation basis, COLUMN-MAJOR Nloc x Mraw (col c at Wcm[c*Nloc + i]);
//            orthonormalized IN PLACE (shrunk to Nloc x M, M <= Mraw). Mraw == 0 -> plain baseline.
//   comm   : the distributed comm (every rank calls with identical Mraw / M).
template <class Real>
inline MatfreeDeflateResult deflated_gmres_matfree(
    const std::function<void(sctl::Vector<Real>*, const sctl::Vector<Real>&)>& Aop,
    const sctl::Vector<Real>& b, std::vector<double>& Wcm, long Mraw,
    sctl::Vector<Real>& x, double tol, long max_iter, const sctl::Comm& comm, bool verbose,
    const DeflateBlocks* blocks = nullptr) {
  using namespace mfdefl_detail;
  MatfreeDeflateResult R;
  const long Nloc = b.Dim();
  sctl::GMRES<Real> solver(comm, verbose);

  auto true_resid = [&](const sctl::Vector<Real>& xx) {
    sctl::Vector<Real> Ax; Aop(&Ax, xx);
    double rn = 0, bn = 0;
    for (long i = 0; i < Nloc; i++) { const double d = (double)(b[i] - Ax[i]); rn += d*d; bn += (double)b[i]*(double)b[i]; }
    rn = GlobalReduce(rn, comm, sctl::CommOp::SUM);
    bn = GlobalReduce(bn, comm, sctl::CommOp::SUM);
    return bn > 0 ? std::sqrt(rn / bn) : std::sqrt(rn);
  };

  // ---- Baseline (no deflation): the production path, so M=0 reproduces the stall exactly. ----
  if (Mraw <= 0) {
    x.ReInit((sctl::Long)Nloc); x = 0;
    sctl::Long it = 0;
    solver(&x, Aop, b, (Real)tol, (sctl::Integer)max_iter, false, &it, nullptr);
    R.iters = (long)it; R.Mdefl = 0; R.rel_resid = true_resid(x);
    return R;
  }

  // ---- 1-2. Orthonormalize W by Gram eigen-whitening.  Two paths, SAME resulting span / deflation
  //      projector, differing only in the O.N. basis within that span (and possibly a few columns in how
  //      rank-deficiency is cut):
  //        * block path  -- a DeflateBlocks partition supplied: one tiny per-arm dsyev over that arm's dofs,
  //          then one Gram-Schmidt of the flux column; cheap setup, exploits the layered disjoint supports;
  //        * global path -- default: one dense Mraw x Mraw dsyev over all rows (bit-identical to before). ----
  long M = 0;
  sctl::Profile::Tic("deflate: (a) whiten", &comm);   // Gram + eigendecomposition + W <- W T  (one-time)
  if (blocks && !blocks->cols.empty()) {
    M = mfdefl_detail::block_whiten<Real>(Wcm, Nloc, Mraw, *blocks, comm);
    SCTL_ASSERT_MSG(M > 0, "matrixfree_deflation: block whitening produced numerical rank 0");
  } else {
  // ---- 1. Global Gram G = W^T W (M x M), reduced across ranks. ----
  sctl::Vector<double> Gv((sctl::Long)(Mraw * Mraw)); Gv = 0;
  for (long a = 0; a < Mraw; a++) {
    const double* wa = &Wcm[(size_t)a * Nloc];
    for (long c = a; c < Mraw; c++) {
      const double* wc = &Wcm[(size_t)c * Nloc];
      double s = 0; for (long i = 0; i < Nloc; i++) s += wa[i] * wc[i];
      Gv[(sctl::Long)(a + c * Mraw)] = s; Gv[(sctl::Long)(c + a * Mraw)] = s;
    }
  }
  GlobalReduce(Gv, comm, sctl::CommOp::SUM);

  // ---- 2. Symmetric eigendecomposition G = V L V^T (dsyev), ascending eigenvalues. ----
  std::vector<double> G((size_t)Mraw * Mraw);
  for (long i = 0; i < Mraw * Mraw; i++) G[(size_t)i] = Gv[(sctl::Long)i];
  std::vector<double> lam((size_t)Mraw);
  {
    const int n = (int)Mraw; int info = 0, lwork = -1; double wq = 0;
    dsyev_("V", "U", &n, G.data(), &n, lam.data(), &wq, &lwork, &info);
    lwork = (int)wq; std::vector<double> work((size_t)std::max(1, lwork));
    dsyev_("V", "U", &n, G.data(), &n, lam.data(), work.data(), &lwork, &info);
    SCTL_ASSERT_MSG(info == 0, "matrixfree_deflation: dsyev failed");
  }
  const double lmax = lam.empty() ? 0.0 : lam[(size_t)Mraw - 1];
  const double rtol = 1e-10 * (lmax > 0 ? lmax : 1.0);
  M = 0; for (long j = 0; j < Mraw; j++) if (lam[(size_t)j] > rtol) M++;
  SCTL_ASSERT_MSG(M > 0, "matrixfree_deflation: deflation basis has numerical rank 0");

  // Whitening T (Mraw x M): keep the M largest eigenpairs, column k = V[:,idx]/sqrt(lam[idx]).
  std::vector<double> T((size_t)Mraw * M);
  for (long k = 0; k < M; k++) {
    const long idx = Mraw - 1 - k;                 // kth largest eigenvalue
    const double inv = 1.0 / std::sqrt(lam[(size_t)idx]);
    const double* vk = &G[(size_t)idx * Mraw];     // eigenvector (col-major)
    for (long a = 0; a < Mraw; a++) T[(size_t)a + (size_t)k * Mraw] = vk[(size_t)a] * inv;
  }
  // W <- W T  (local gemv; same T on every rank -> globally orthonormal columns).
  std::vector<double> Worth((size_t)Nloc * M, 0.0);
  for (long k = 0; k < M; k++) {
    double* ok = &Worth[(size_t)k * Nloc];
    for (long a = 0; a < Mraw; a++) {
      const double t = T[(size_t)a + (size_t)k * Mraw];
      if (t == 0) continue;
      const double* wa = &Wcm[(size_t)a * Nloc];
      for (long i = 0; i < Nloc; i++) ok[i] += wa[i] * t;
    }
  }
  Wcm.swap(Worth);
  } // end global whitening path
  sctl::Profile::Toc();                               // deflate: (a) whiten
  R.Mdefl = M;

  // ---- 3. AW = A W  (M production matvecs; one-time setup). ----
  sctl::Profile::Tic("deflate: (b) AW matvecs", &comm);   // M operator applies (ApplyK/ComputePotential)
  std::vector<double> AWcm((size_t)Nloc * M);
  {
    sctl::Vector<Real> wcol((sctl::Long)Nloc), awcol;
    for (long j = 0; j < M; j++) {
      const double* wj = &Wcm[(size_t)j * Nloc];
      for (long i = 0; i < Nloc; i++) wcol[i] = (Real)wj[i];
      Aop(&awcol, wcol);
      SCTL_ASSERT(awcol.Dim() == Nloc);
      double* aj = &AWcm[(size_t)j * Nloc];
      for (long i = 0; i < Nloc; i++) aj[i] = (double)awcol[i];
    }
  }
  sctl::Profile::Toc();                               // deflate: (b) AW matvecs

  // ---- 4. Coarse matrix Ac = W^T (A W)  (M x M, reduced), then LU. ----
  sctl::Profile::Tic("deflate: (c) coarse Ac+LU", &comm); // form Galerkin Ac = W^T A W and factorize (one-time)
  std::vector<double> Ac((size_t)M * M);
  {
    sctl::Vector<double> Av((sctl::Long)(M * M)); Av = 0;
    for (long a = 0; a < M; a++) {
      const double* wa = &Wcm[(size_t)a * Nloc];
      for (long c = 0; c < M; c++) {
        const double* awc = &AWcm[(size_t)c * Nloc];
        double s = 0; for (long i = 0; i < Nloc; i++) s += wa[i] * awc[i];
        Av[(sctl::Long)(a + c * M)] = s;             // Ac[a][c] = W_a . AW_c  (col-major)
      }
    }
    GlobalReduce(Av, comm, sctl::CommOp::SUM);
    for (long i = 0; i < M * M; i++) Ac[(size_t)i] = Av[(sctl::Long)i];
  }
  std::vector<int> ipiv((size_t)M);
  { const int n = (int)M; int info = 0; dgetrf_(&n, &n, Ac.data(), &n, ipiv.data(), &info);
    SCTL_ASSERT_MSG(info == 0, "matrixfree_deflation: coarse matrix Ac is singular (dgetrf)"); }
  sctl::Profile::Toc();                               // deflate: (c) coarse Ac+LU
  auto coarse_solve = [&](std::vector<double> rhs) {           // y = Ac^{-1} rhs (replicated)
    const int n = (int)M, nrhs = 1; int info = 0;
    dgetrs_("N", &n, &nrhs, Ac.data(), &n, ipiv.data(), rhs.data(), &n, &info);
    SCTL_ASSERT(info == 0);
    return rhs;
  };

  // ---- 5. Deflated operator  P A in = A in - A W Ac^{-1} W^T (A in). ----
  auto PAop = [&](sctl::Vector<Real>* out, const sctl::Vector<Real>& in) {
    sctl::Vector<Real> Ain; Aop(&Ain, in);
    std::vector<double> y = coarse_solve(Wt_apply(Wcm, Nloc, M, Ain, comm));
    sctl::Vector<Real> AWy; mat_apply(AWcm, Nloc, M, y, AWy);
    if (out->Dim() != Nloc) out->ReInit((sctl::Long)Nloc);
    for (long i = 0; i < Nloc; i++) (*out)[i] = Ain[i] - AWy[i];
  };

  // ---- 6. Pb = b - A W Ac^{-1} W^T b ,  xc = W Ac^{-1} W^T b ,  solve P A x~ = Pb. ----
  std::vector<double> u0 = coarse_solve(Wt_apply(Wcm, Nloc, M, b, comm));
  sctl::Vector<Real> AWu0; mat_apply(AWcm, Nloc, M, u0, AWu0);
  sctl::Vector<Real> Pb((sctl::Long)Nloc); for (long i = 0; i < Nloc; i++) Pb[i] = b[i] - AWu0[i];
  sctl::Vector<Real> xc; mat_apply(Wcm, Nloc, M, u0, xc);

  sctl::Vector<Real> xt((sctl::Long)Nloc); xt = 0;
  sctl::Long it = 0;
  solver(&xt, PAop, Pb, (Real)tol, (sctl::Integer)max_iter, false, &it, nullptr);
  R.iters = (long)it;

  // ---- 7. Recover x = xc + x~ - W Ac^{-1} W^T (A x~). ----
  sctl::Vector<Real> Axt; Aop(&Axt, xt);
  std::vector<double> y2 = coarse_solve(Wt_apply(Wcm, Nloc, M, Axt, comm));
  sctl::Vector<Real> corr; mat_apply(Wcm, Nloc, M, y2, corr);
  x.ReInit((sctl::Long)Nloc);
  for (long i = 0; i < Nloc; i++) x[i] = xc[i] + xt[i] - corr[i];

  R.rel_resid = true_resid(x);
  return R;
}

} // namespace quad_junctions
