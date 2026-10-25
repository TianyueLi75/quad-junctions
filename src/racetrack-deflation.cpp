// racetrack-deflation.cpp -- layered-block multi-arm analytic deflation of a dumped racetrack interior
// Stokes operator (see `ybifurc-channel-bie dump`). Pure linear algebra: reads <A>, <A>.X0, <A>.Xn0,
// <A>.arm, <A>.s, <A>.armL (rows: L,R,npanel per long arm), builds per-arm blocks + one flux column and
// runs deflated GMRES (true AW) for a ladder of configurations.
// Build: make bin/racetrack-deflation PETSC=1 MPI=0 PVFMM=0
// Run:   ./bin/racetrack-deflation <A> [tol=1e-8] [max_iter=2000] [n_max=8] [csv]
#include <sctl.hpp>
#include <quad_junctions/deflation_precond.hpp>
#include <quad_junctions/multiarm_deflation.hpp>
#include <quad_junctions/nearnull_heuristic.hpp>   // nearnull_nmax(L/R): per-arm n_max from aspect
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <string>
#include <vector>
#include <algorithm>

#ifdef QJ_HAVE_PETSC
#include <petscsys.h>
using namespace quad_junctions;

static void readvec(const std::string& f, std::vector<double>& v) {
  sctl::Matrix<double> M; M.Read(f.c_str());
  v.resize((size_t)M.Dim(0) * M.Dim(1));
  for (long i = 0; i < M.Dim(0); i++) for (long j = 0; j < M.Dim(1); j++) v[(size_t)i*M.Dim(1)+j] = M[i][j];
}

int main(int argc, char** argv) {
  PetscInitialize(&argc, &argv, nullptr, nullptr);
  if (argc < 2) { std::printf("usage: %s <A> [tol] [max_iter] [n_max] [csv]\n", argv[0]); PetscFinalize(); return 1; }
  const std::string Af = argv[1];
  const double tol = argc > 2 ? std::atof(argv[2]) : 1e-8;
  const long max_iter = argc > 3 ? std::atol(argv[3]) : 2000;
  const int n_max = argc > 4 ? std::atoi(argv[4]) : 8;
  const std::string csv = argc > 5 ? argv[5] : "racetrack-deflation.csv";

  std::vector<double> A, X0, Xn0, armd, sd, armL;
  readvec(Af, A); readvec(Af + ".X0", X0); readvec(Af + ".Xn0", Xn0);
  readvec(Af + ".arm", armd); readvec(Af + ".s", sd); readvec(Af + ".armL", armL);
  const long N = (long)X0.size(), np = N / 3;
  SCTL_ASSERT_MSG((long)A.size() == N*N && (long)Xn0.size() == N && (long)armd.size() == np && (long)sd.size() == np, "size mismatch");
  const int narm = (int)(armL.size() / 3);
  std::vector<int> armid(np); std::vector<double> s = sd;
  std::vector<long> cnt(narm + 1, 0); long nctl = 0;
  for (long p = 0; p < np; p++) { armid[p] = (int)std::lround(armd[p]); if (armid[p] >= 0) cnt[armid[p]]++; else nctl++; }
  std::printf("[racetrack] N=%ld nodes=%ld narm=%d  non-arm nodes=%ld\n", N, np, narm, nctl);
  for (int k = 0; k < narm; k++) std::printf("  arm %d: L=%.4f R=%.4f L/R=%.1f panels=%.0f nodes=%ld\n", k, armL[3*k], armL[3*k+1], armL[3*k]/armL[3*k+1], armL[3*k+2], cnt[k]);

  // Negative control: standing waves on the non-arm (junction/cap/stem) nodes, id = narm, s = x - xmin.
  std::vector<int> armid_c = armid; std::vector<double> s_c = s;
  double xmin = 1e300, xmax = -1e300;
  for (long p = 0; p < np; p++) if (armid[p] < 0) { xmin = std::min(xmin, X0[3*p]); xmax = std::max(xmax, X0[3*p]); }
  for (long p = 0; p < np; p++) if (armid[p] < 0) { armid_c[p] = narm; s_c[p] = X0[3*p] - xmin; }
  const double Lctl = std::max(xmax - xmin, 1e-12);

  // Physical RHS b (p_in/p_out), interior-velocity eval matrix Bint (3*nt x N) and target coords.
  std::vector<double> b; readvec(Af + ".b", b); SCTL_ASSERT((long)b.size() == N);
  sctl::Matrix<double> Bm; Bm.Read((Af + ".Bint").c_str());
  const long nt3 = Bm.Dim(0); SCTL_ASSERT(Bm.Dim(1) == N);
  std::vector<double> Bint((size_t)nt3*N);
  for (long i = 0; i < nt3; i++) for (long j = 0; j < N; j++) Bint[(size_t)i*N+j] = Bm[i][j];
  std::vector<double> tgt; readvec(Af + ".tgt", tgt); SCTL_ASSERT((long)tgt.size() == nt3);
  auto eval_u = [&](const std::vector<double>& x) {
    std::vector<double> u(nt3, 0.0);
    for (long i = 0; i < nt3; i++) { double t = 0; const double* Bi = &Bint[(size_t)i*N]; for (long j = 0; j < N; j++) t += Bi[j]*x[j]; u[i] = t; }
    return u; };
  std::vector<double> u_ref;
  auto vel_err = [&](const std::vector<double>& u) {
    double en = 0, tn = 0; for (long i = 0; i < nt3; i++) { const double d = u[i]-u_ref[i]; en += d*d; tn += u_ref[i]*u_ref[i]; }
    return std::sqrt(en / std::max(tn, 1e-300)); };
  sctl::Comm comm = sctl::Comm::Self();

  struct Cfg { std::string label; std::vector<int> arms; bool flux; bool ctl; int nm; };
  std::vector<Cfg> cfgs;
  cfgs.push_back({"baseline", {}, false, false, 0});
  cfgs.push_back({"flux-only", {}, true, false, 0});
  if (narm > 0) cfgs.push_back({"arm0", {0}, false, false, n_max});
  if (narm > 1) { cfgs.push_back({"arm1", {1}, false, false, n_max});
                  cfgs.push_back({"arm0+arm1", {0,1}, false, false, n_max}); }
  { std::vector<int> all; for (int k = 0; k < narm; k++) all.push_back(k);
    cfgs.push_back({"arms+flux", all, true, false, n_max});
    for (int nm : {2, 4, 6}) if (nm < n_max) cfgs.push_back({"arms+flux nm=" + std::to_string(nm), all, true, false, nm});
    cfgs.push_back({"ctl-only+flux", {}, true, true, n_max});
    cfgs.push_back({"arms+flux+ctl", all, true, true, n_max}); }

  FILE* f = std::fopen(csv.c_str(), "w");
  if (f) std::fprintf(f, "label,Mraw,Mdefl,iters,rel_resid,rel_err_vel\n");
  for (const auto& c : cfgs) {
    std::vector<ArmBlockSpec> bl;
    for (int k : c.arms) bl.push_back({k, armL[3*k], c.nm});
    std::vector<int>* ai = &armid; std::vector<double>* sp = &s;
    if (c.ctl) { bl.push_back({narm, Lctl, c.nm}); ai = &armid_c; sp = &s_c; }
    std::vector<double> W, x; std::vector<long> bc;
    long Mraw = build_multiarm_modes(Xn0, N, *ai, *sp, bl, c.flux, W, &bc);
    // zero-padding check: block columns are zero outside their own arm
    { long col = c.flux ? 1 : 0;
      for (size_t q = 0; q < bl.size(); q++) { for (long j = 0; j < bc[q]; j++, col++)
        for (long p = 0; p < np; p++) if ((*ai)[p] != bl[q].id) for (int d = 0; d < 3; d++) SCTL_ASSERT(W[(size_t)col*N+3*p+d] == 0.0); } }
    DeflateResult R = deflated_gmres_petsc(A, N, b, W, Mraw, x, tol, max_iter, nullptr, comm);
    const std::vector<double> u = eval_u(x);
    if (u_ref.empty()) {   // first config = baseline (no preconditioner): record the reference interior velocities
      u_ref = u;
      std::printf("\n  [baseline reference interior velocity, %ld targets]\n    %-3s %-12s %-12s %-12s | %-13s %-13s %-13s\n", nt3/3, "k", "x", "y", "z", "vx", "vy", "vz");
      for (long k = 0; k < nt3/3; k++) std::printf("    %-3ld %-12.6f %-12.6f %-12.6f | % .6e % .6e % .6e\n", k, tgt[3*k], tgt[3*k+1], tgt[3*k+2], u_ref[3*k], u_ref[3*k+1], u_ref[3*k+2]);
      std::printf("\n  %-18s %5s %6s %6s  %-10s %-12s\n", "config", "Mraw", "Mdefl", "iters", "rel_resid", "rel_err_vel");
    }
    const double ev = vel_err(u);
    std::printf("  %-18s %5ld %6ld %6ld  %.3e  %.3e\n", c.label.c_str(), Mraw, R.Mdefl, R.iters, R.rel_resid, ev);
    if (f) std::fprintf(f, "%s,%ld,%ld,%ld,%.6e,%.6e\n", c.label.c_str(), Mraw, R.Mdefl, R.iters, R.rel_resid, ev);
    std::fflush(stdout);
  }

  // ---- HEURISTIC-SIZED run: per-arm block n_max set from THAT arm's aspect L/R via the
  //      aspect-ratio near-null heuristic (nearnull_heuristic.hpp), plus one global flux column.
  //      This is the "arm only, no junction" deflation the heuristic prescribes: no CLI n_max, no
  //      negative-control junction block -- the basis size is read straight off the geometry. ----
  {
    const double theta = std::getenv("QJ_NEARNULL_THETA") ? std::atof(std::getenv("QJ_NEARNULL_THETA")) : 0.05;
    std::vector<ArmBlockSpec> bl;
    std::printf("\n  [heuristic] theta=%.0e  per-arm n_max from aspect L/R:\n", theta);
    long total = 1;  // the global flux column
    for (int k = 0; k < narm; k++) {
      const double aspect = armL[3*k] / armL[3*k+1];
      const long nm = quad_junctions::nearnull_nmax(aspect, theta);
      bl.push_back({k, armL[3*k], (int)nm});
      std::printf("    arm %d: aspect=%.1f -> n_max=%ld (%ld cols)\n", k, aspect, nm, 2*nm);
      total += 2*nm;
    }
    std::printf("    predicted deflation dim = 1 flux + sum 2*n_max = %ld\n", total);
    std::vector<double> W, x; std::vector<long> bc;
    long Mraw = build_multiarm_modes(Xn0, N, armid, s, bl, /*include_flux=*/true, W, &bc);
    DeflateResult R = deflated_gmres_petsc(A, N, b, W, Mraw, x, tol, max_iter, nullptr, comm);
    const double ev = vel_err(eval_u(x));
    std::printf("  %-18s %5ld %6ld %6ld  %.3e  %.3e\n", "heuristic", Mraw, R.Mdefl, R.iters, R.rel_resid, ev);
    if (f) std::fprintf(f, "%s,%ld,%ld,%ld,%.6e,%.6e\n", "heuristic", Mraw, R.Mdefl, R.iters, R.rel_resid, ev);
  }

  if (f) std::fclose(f);
  PetscFinalize();
  return 0;
}
#else
int main() { std::printf("build with PETSC=1\n"); return 0; }
#endif
