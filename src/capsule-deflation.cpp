// ---------------------------------------------------------------------------
// capsule-deflation.cpp -- deflated-GMRES correction for the capsule combined-
// field interior Stokes operator's near-null (standing-wave pressure) modes.
//
// Reads a dense operator A dumped by `capsule-robin-bie pspectrum` (env
// QJ_DUMP_A, which writes the interior combined-field "none" operator via
// sctl::Matrix::Write), computes its right eigenvectors, builds a deflation
// space W from the nev smallest-|lambda| modes (= the standing-wave modes),
// orthonormalizes, and solves A x = b with sctl::GMRES while deflating span(W).
// The M=0 row is the undeflated baseline (the production stall). See
// deflation_precond.hpp for the projector algebra.
//
// Build:   make bin/capsule-deflation PETSC=1 MPI=0 PVFMM=0
// Run:     ./bin/capsule-deflation <A.bin> [tol=1e-8] [max_iter=2000]
//                                          [nev_max=32] [csv=vis/capsule-deflation.csv]
// ---------------------------------------------------------------------------

#include <sctl.hpp>
#include <quad_junctions/deflation_precond.hpp>
#include <quad_junctions/analytic_deflation.hpp>
#include <quad_junctions/nearnull_heuristic.hpp>   // nearnull_count(L/R): nev from capsule aspect
#include <cstdlib>
#include <cstdio>
#include <cmath>
#include <vector>
#include <algorithm>
#include <string>

#ifdef QJ_HAVE_PETSC
#include <petscsys.h>

// LAPACK dgeev (MKL, LP64, column-major).
extern "C" void dgeev_(const char*, const char*, const int*, double*, const int*,
                       double*, double*, double*, const int*, double*, const int*,
                       double*, const int*, int*);

// Right eigenvectors of the row-major N x N matrix A. Our row-major buffer, read
// column-major by dgeev, IS A^T; its LEFT eigenvectors (jobvl='V') are the RIGHT
// eigenvectors of A (eigenvalues of A and A^T coincide). vl is column-major
// N x N: for a real lambda_j, column j is the real eigenvector; for a complex
// pair (wi[j]>0, wi[j+1]<0) columns j and j+1 hold Re and Im of eigenvector j.
static void right_eig(const std::vector<double>& A, int N,
                      std::vector<double>& wr, std::vector<double>& wi,
                      std::vector<double>& vl) {
  wr.assign(N, 0); wi.assign(N, 0); vl.assign((size_t)N * N, 0);
  int lda = N, ldvl = N, ldvr = 1, info = 0, lwork = -1;
  double wq = 0, vrdummy = 0;
  std::vector<double> buf = A;  // workspace query (buf untouched by query)
  dgeev_("V", "N", &N, buf.data(), &lda, wr.data(), wi.data(), vl.data(), &ldvl,
         &vrdummy, &ldvr, &wq, &lwork, &info);
  lwork = (int)wq;
  std::vector<double> work(std::max(1, lwork));
  buf = A;                      // real call overwrites buf -> fresh copy
  dgeev_("V", "N", &N, buf.data(), &lda, wr.data(), wi.data(), vl.data(), &ldvl,
         &vrdummy, &ldvr, work.data(), &lwork, &info);
  SCTL_ASSERT_MSG(info == 0, "dgeev failed");
}

// Collect the nev smallest-|lambda| eigen-modes into real deflation columns
// (col-major N x Mraw). A real eigenvalue contributes 1 column (its eigenvector);
// a complex-conjugate pair contributes 2 columns (Re, Im) -- span is A-invariant.
static long build_modes(int N, const std::vector<double>& wr, const std::vector<double>& wi,
                        const std::vector<double>& vl, long nev, std::vector<double>& Wcols) {
  std::vector<std::pair<double,int>> modes;   // (|lambda|, lead original index)
  std::vector<char> consumed(N, 0);
  for (int j = 0; j < N; j++) {
    if (consumed[j]) continue;
    const double lam = std::hypot(wr[j], wi[j]);
    consumed[j] = 1;
    if (wi[j] != 0.0 && j + 1 < N) consumed[j + 1] = 1;   // conjugate partner
    modes.push_back({lam, j});
  }
  std::sort(modes.begin(), modes.end());
  Wcols.clear();
  long covered = 0;
  for (const auto& m : modes) {
    if (covered >= nev) break;
    const int j = m.second;
    for (int i = 0; i < N; i++) Wcols.push_back(vl[i + (size_t)j * N]);   // Re / real vec
    covered += 1;
    if (wi[j] != 0.0) {
      for (int i = 0; i < N; i++) Wcols.push_back(vl[i + (size_t)(j + 1) * N]); // Im
      covered += 1;   // the pair covers two eigenvalues
    }
  }
  return (long)(Wcols.size() / N);
}

int main(int argc, char** argv) {
  PetscInitialize(&argc, &argv, nullptr, nullptr);
  if (argc < 2) {
    std::printf("usage: %s <A.bin> [tol=1e-8] [max_iter=2000] [nev_max=32] [csv]\n", argv[0]);
    PetscFinalize(); return 1;
  }
  const std::string Afile = argv[1];
  const double tol      = (argc > 2) ? std::atof(argv[2]) : 1e-8;
  const long   max_iter = (argc > 3) ? std::atol(argv[3]) : 2000;
  const long   nev_max  = (argc > 4) ? std::atol(argv[4]) : 32;
  const std::string csv = (argc > 5) ? argv[5] : "vis/capsule-deflation.csv";

  // ---- read A (row-major copy of the sctl::Matrix) ----
  sctl::Matrix<double> Am;
  Am.Read(Afile.c_str());
  const int N = (int)Am.Dim(0);
  SCTL_ASSERT_MSG(Am.Dim(0) == Am.Dim(1) && N > 0, "A must be square and non-empty");
  std::vector<double> A((size_t)N * N);
  for (int i = 0; i < N; i++)
    for (int j = 0; j < N; j++) A[(size_t)i * N + j] = Am[i][j];
  std::printf("[deflate] read A: %d x %d  from %s\n", N, N, Afile.c_str());

  const char* ea = std::getenv("QJ_DEFLATE_ANALYTIC");
  const int analytic = ea ? std::atoi(ea) : 0;
  std::vector<double> wr, wi, vl;
  if (!analytic) {
  // ---- eigen-decomposition (once) ----
  right_eig(A, N, wr, wi, vl);
  // sorted |lambda| for the near-null tail
  std::vector<double> lam(N);
  for (int j = 0; j < N; j++) lam[j] = std::hypot(wr[j], wi[j]);
  std::sort(lam.begin(), lam.end());
  std::printf("[deflate] smallest |lambda|: ");
  for (int j = 0; j < std::min(16, N); j++) std::printf("%.3e ", lam[j]);
  std::printf("\n");
  std::printf("[deflate] lambda_min (4 smallest |lambda|): %.6e %.6e %.6e %.6e\n", lam[0], lam[1], lam[2], lam[3]);
  long n_null = 0; for (int j = 0; j < N; j++) if (lam[j] < 1e-2) n_null++;
  std::printf("[deflate] near-null count (|lambda|<1e-2) = %ld\n", n_null);

  }
  // ---- RHS b = A x_true  (known solution -> iters AND accuracy) ----
  std::srand(1);
  std::vector<double> x_true(N), b(N, 0.0);
  double xn = 0; for (int i = 0; i < N; i++) { x_true[i] = (double)std::rand()/RAND_MAX - 0.5; xn += x_true[i]*x_true[i]; }
  xn = std::sqrt(xn); for (int i = 0; i < N; i++) x_true[i] /= xn;
  for (int i = 0; i < N; i++) { double s = 0; for (int j = 0; j < N; j++) s += A[(size_t)i*N+j]*x_true[j]; b[i] = s; }

  sctl::Comm comm = sctl::Comm::Self();

  if (analytic) {
    const char* e;
    const int cols_per_n = (e = std::getenv("QJ_DEFLATE_COLS")) ? std::atoi(e) : 4;
    const int nmax_cap   = (e = std::getenv("QJ_DEFLATE_NMAX")) ? std::atoi(e) : 16;
    const std::string x0f  = (e = std::getenv("QJ_DEFLATE_X0"))  ? e : Afile + ".X0";
    const std::string xn0f = (e = std::getenv("QJ_DEFLATE_XN0")) ? e : Afile + ".Xn0";
    sctl::Matrix<double> Xm, Xnm; Xm.Read(x0f.c_str()); Xnm.Read(xn0f.c_str());
    SCTL_ASSERT_MSG(Xm.Dim(0)*Xm.Dim(1) == N && Xnm.Dim(0)*Xnm.Dim(1) == N, "X0/Xn0 length != N (dof count)");
    std::vector<double> X0(N), Xn0(N);
    for (int i = 0; i < N; i++) { X0[i] = Xm[0][i]; Xn0[i] = Xnm[0][i]; }
    const double axis[3] = {1, 0, 0};
    const bool noflux = (e = std::getenv("QJ_DEFLATE_NOFLUX")) && std::atoi(e) != 0;
    const std::string acsv = (argc > 5) ? argv[5] : "vis/capsule-deflation-analytic.csv";

    // Rayleigh-quotient diagnostic at n_max = nmax_cap
    { std::vector<double> W, lm;
      long M = quad_junctions::build_analytic_modes(X0, Xn0, N, nmax_cap, cols_per_n, axis, W, &lm, 0.125, noflux);
      std::printf("\n[analytic] Rayleigh diagnostic (col, model lambda, measured w'Aw/w'w):\n");
      for (long c = 0; c < M; c++) {
        const double* w = &W[(size_t)c * N]; double ww = 0, wAw = 0;
        for (int i = 0; i < N; i++) { double s2 = 0; const double* Ai = &A[(size_t)i*N];
          for (int j = 0; j < N; j++) s2 += Ai[j]*w[j]; wAw += w[i]*s2; ww += w[i]*w[i]; }
        std::printf("  col %3ld  model %.3e  measured %.3e\n", c, lm[c], wAw/ww);
      }
    }
    std::vector<int> nms;
    for (int v : {0,1,2,3,4,6,8,12,16}) if (v <= nmax_cap) nms.push_back(v);
    FILE* f = std::fopen(acsv.c_str(), "w");
    if (f) std::fprintf(f, "mode,n_max,Mraw,Mdefl,iters,rel_resid,rel_err\n");
    std::printf("\n  mode n_max  Mraw  Mdefl   iters   rel_resid     rel_err\n");
    for (int nm : nms) {
      if (noflux && nm == 0) continue;   // empty basis
      std::vector<double> W, lm, x;
      long Mraw = quad_junctions::build_analytic_modes(X0, Xn0, N, nm, cols_per_n, axis, W, &lm, 0.125, noflux);
      quad_junctions::DeflateResult R = (analytic == 2)
        ? quad_junctions::deflated_gmres_petsc_analyticAW(A, N, b, W, Mraw, lm, x, tol, max_iter, &x_true, comm)
        : quad_junctions::deflated_gmres_petsc(A, N, b, W, Mraw, x, tol, max_iter, &x_true, comm);
      std::printf("  %4d %5d %5ld %6ld %7ld   %.3e   %.3e\n", analytic, nm, Mraw, R.Mdefl, R.iters, R.rel_resid, R.rel_err);
      if (f) std::fprintf(f, "%d,%d,%ld,%ld,%ld,%.6e,%.6e\n", analytic, nm, Mraw, R.Mdefl, R.iters, R.rel_resid, R.rel_err);
      std::fflush(stdout);
    }
    if (f) std::fclose(f);
    PetscFinalize(); return 0;
  }

  // ---- heuristic nev from the capsule aspect L/R (nearnull_heuristic.hpp). Read the companion
  //      <A>.X0 (same file the analytic path uses) to get the node coords, estimate aspect, and
  //      size the deflation space nev = flux mode + axial standing-wave rungs below theta. ----
  const double theta = std::getenv("QJ_NEARNULL_THETA") ? std::atof(std::getenv("QJ_NEARNULL_THETA")) : 0.05;
  long nev_heur = -1;
  { const std::string x0f = (std::getenv("QJ_DEFLATE_X0")) ? std::getenv("QJ_DEFLATE_X0") : Afile + ".X0";
    sctl::Matrix<double> Xm;
    if (FILE* xf = std::fopen(x0f.c_str(), "rb")) { std::fclose(xf);
      Xm.Read(x0f.c_str());
      if (Xm.Dim(0)*Xm.Dim(1) == N) {
        std::vector<double> X0v(N); for (int i = 0; i < N; i++) X0v[i] = Xm[0][i];
        const double axis[3] = {1, 0, 0};
        const double aspect = quad_junctions::capsule_aspect_from_nodes(X0v, axis);
        nev_heur = quad_junctions::nearnull_count(aspect, theta);
        std::printf("[deflate] capsule aspect L/R = %.1f -> heuristic nev (|lambda|<%.0e) = %ld\n",
                    aspect, theta, nev_heur);
      }
    }
    if (nev_heur < 0) std::printf("[deflate] no %s -> cannot size heuristic nev (pass QJ_DEFLATE_X0)\n", x0f.c_str());
  }

  // ---- nev sweep (includes the heuristic nev so its row is directly comparable) ----
  std::vector<long> nevs = {0};
  for (long v = 1; v <= nev_max; v = (v < 8 ? v + 1 : v * 2)) nevs.push_back(v);
  if (nevs.back() != nev_max) nevs.push_back(nev_max);
  if (nev_heur >= 0 && std::find(nevs.begin(), nevs.end(), nev_heur) == nevs.end()) nevs.push_back(nev_heur);
  std::sort(nevs.begin(), nevs.end());

  FILE* f = std::fopen(csv.c_str(), "w");
  if (f) std::fprintf(f, "nev,Mdefl,iters,rel_resid,rel_err\n");
  std::printf("\n  nev  Mdefl   iters   rel_resid     rel_err\n");
  std::printf(  "  ------------------------------------------------\n");
  for (long nev : nevs) {
    std::vector<double> Wcols; long Mraw = build_modes(N, wr, wi, vl, nev, Wcols);
    std::vector<double> x;
    quad_junctions::DeflateResult R =
        quad_junctions::deflated_gmres_petsc(A, N, b, Wcols, Mraw, x, tol, max_iter, &x_true, comm);
    std::printf("  %3ld  %5ld  %6ld   %.3e   %.3e%s\n", nev, R.Mdefl, R.iters, R.rel_resid, R.rel_err,
                (nev == nev_heur) ? "   <-- heuristic" : "");
    if (f) std::fprintf(f, "%ld,%ld,%ld,%.6e,%.6e\n", nev, R.Mdefl, R.iters, R.rel_resid, R.rel_err);
  }
  if (f) std::fclose(f);
  std::printf("\n[deflate] wrote %s\n", csv.c_str());
  PetscFinalize();
  return 0;
}

#else  // no PETSc
int main() {
  std::printf("capsule-deflation was built without PETSc. Rebuild: make bin/capsule-deflation PETSC=1 MPI=0 PVFMM=0\n");
  return 0;
}
#endif
