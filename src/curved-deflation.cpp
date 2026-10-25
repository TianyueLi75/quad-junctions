// ---------------------------------------------------------------------------
// curved-deflation.cpp -- standing-wave deflation with s = arclength along the ACTUAL curved centerline.
// Pure linear algebra (no CSBQ/StokesBIO). Reads A + .X0 + .Xn0 + .CL (all dumped by
// `QJ_DUMP_A=<f> QJ_DUMP_ONLY=1 capsule-robin-bie pspectrum ...`), builds the curved basis, and
// (QJ_CURVED_COMPARE=1, default) also the straight-axis {1,0,0} basis, running deflated GMRES for both.
//
// Build: make bin/curved-deflation PETSC=1 MPI=0 PVFMM=0
// Run:   [QJ_DEFLATE_COLS=20|2|4] ./bin/curved-deflation <A.bin> [tol=1e-8] [max_iter=2000] [nmax=16]
// ---------------------------------------------------------------------------
#include <sctl.hpp>
#include <quad_junctions/deflation_precond.hpp>
#include <quad_junctions/curved_deflation.hpp>
#include <cstdlib>
#include <cstdio>
#include <vector>
#include <string>

#ifdef QJ_HAVE_PETSC
#include <petscsys.h>

int main(int argc, char** argv) {
  PetscInitialize(&argc, &argv, nullptr, nullptr);
  if (argc < 2) { std::printf("usage: %s <A.bin> [tol] [max_iter] [nmax]\n", argv[0]); PetscFinalize(); return 1; }
  const std::string Afile = argv[1];
  const double tol = (argc > 2) ? std::atof(argv[2]) : 1e-8;
  const long max_iter = (argc > 3) ? std::atol(argv[3]) : 2000;
  const int nmax_cap = (argc > 4) ? std::atoi(argv[4]) : 16;
  const char* e;
  const int cols_per_n = (e = std::getenv("QJ_DEFLATE_COLS")) ? std::atoi(e) : 20;
  const int compare = (e = std::getenv("QJ_CURVED_COMPARE")) ? std::atoi(e) : 1;

  sctl::Matrix<double> Am; Am.Read(Afile.c_str());
  const int N = (int)Am.Dim(0);
  SCTL_ASSERT_MSG(Am.Dim(0) == Am.Dim(1) && N > 0, "A must be square and non-empty");
  std::vector<double> A((size_t)N * N);
  for (int i = 0; i < N; i++) for (int j = 0; j < N; j++) A[(size_t)i*N+j] = Am[i][j];
  std::printf("[curved] read A: %d x %d from %s\n", N, N, Afile.c_str());

  sctl::Matrix<double> Xm, Xnm, Cm;
  Xm.Read((Afile + ".X0").c_str()); Xnm.Read((Afile + ".Xn0").c_str());
  const std::string clf = (e = std::getenv("QJ_DEFLATE_CL")) ? e : Afile + ".CL";
  Cm.Read(clf.c_str());
  SCTL_ASSERT_MSG(Xm.Dim(0)*Xm.Dim(1) == N && Xnm.Dim(0)*Xnm.Dim(1) == N, "X0/Xn0 length != N");
  std::vector<double> X0(N), Xn0(N), CL((size_t)Cm.Dim(0)*Cm.Dim(1));
  for (int i = 0; i < N; i++) { X0[i] = Xm[0][i]; Xn0[i] = Xnm[0][i]; }
  for (size_t i = 0; i < CL.size(); i++) CL[i] = Cm[0][i];
  const double axis[3] = {1, 0, 0};
  const std::vector<double> noCL;

  { auto nc = quad_junctions::curved_node_axial(X0, N, CL, axis);
    auto ns = quad_junctions::curved_node_axial(X0, N, noCL, axis);
    std::printf("[curved] centerline pts=%zu  L_curved=%.4f R_curved=%.4f | L_straight=%.4f R_straight=%.4f\n",
                CL.size()/3, nc.L, nc.R, ns.L, ns.R); }

  std::srand(1);
  std::vector<double> x_true(N), b(N, 0.0);
  double xn = 0; for (int i = 0; i < N; i++) { x_true[i] = (double)std::rand()/RAND_MAX - 0.5; xn += x_true[i]*x_true[i]; }
  xn = std::sqrt(xn); for (int i = 0; i < N; i++) x_true[i] /= xn;
  for (int i = 0; i < N; i++) { double s = 0; for (int j = 0; j < N; j++) s += A[(size_t)i*N+j]*x_true[j]; b[i] = s; }
  sctl::Comm comm = sctl::Comm::Self();

  // Rayleigh diagnostic (n<=4) for both bases
  for (int cv = 0; cv < (compare ? 2 : 1); cv++) {
    std::vector<double> W;
    long M = quad_junctions::build_curved_modes(X0, Xn0, N, 4, cols_per_n, cv == 0 ? CL : noCL, axis, W);
    std::printf("\n[%s] Rayleigh w'Aw/w'w (n<=4):", cv == 0 ? "curved" : "straight");
    for (long c = 0; c < M; c++) {
      const double* w = &W[(size_t)c*N]; double ww = 0, wAw = 0;
      for (int i = 0; i < N; i++) { double s2 = 0; const double* Ai = &A[(size_t)i*N];
        for (int j = 0; j < N; j++) s2 += Ai[j]*w[j]; wAw += w[i]*s2; ww += w[i]*w[i]; }
      std::printf(" %.2e", wAw/ww);
    }
    std::printf("\n");
  }

  std::vector<int> nms; for (int v : {0,1,2,3,4,5,6,7,8,10,12,16}) if (v <= nmax_cap) nms.push_back(v);
  std::printf("\n  basis    n_max  Mraw  Mdefl  iters   rel_resid     rel_err\n");
  for (int nm : nms) for (int cv = 0; cv < (compare ? 2 : 1); cv++) {
    std::vector<double> W, x;
    long Mraw = quad_junctions::build_curved_modes(X0, Xn0, N, nm, cols_per_n, cv == 0 ? CL : noCL, axis, W);
    quad_junctions::DeflateResult R = quad_junctions::deflated_gmres_petsc(A, N, b, W, Mraw, x, tol, max_iter, &x_true, comm);
    std::printf("  %-8s %4d %5ld %6ld %6ld   %.3e   %.3e\n", cv == 0 ? "curved" : "straight", nm, Mraw, R.Mdefl, R.iters, R.rel_resid, R.rel_err);
    std::fflush(stdout);
  }
  PetscFinalize();
  return 0;
}
#else
int main() { std::printf("curved-deflation needs PETSc: make bin/curved-deflation PETSC=1 MPI=0 PVFMM=0\n"); return 0; }
#endif
