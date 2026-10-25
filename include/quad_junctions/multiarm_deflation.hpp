#pragma once
// Multi-arm / layered-block analytic standing-wave deflation basis (no PETSc dependency).
// Self-contained generalization of analytic_deflation.hpp to several long arms: each requested block
// (arm id) contributes columns cos(k_n s)*nhat and sin(k_n s)*nhat, k_n = n*pi/L_arm, with s the node's
// arclength along THAT arm's centerline and nhat the node's outward normal; entries are ZERO on every
// node not belonging to the arm. An optional single global flux column (nhat on all nodes) is appended.
// Node data are per-node (np = N/3); vectors are AoS xyz like Xn0.
#include <vector>
#include <cmath>

namespace quad_junctions {

struct ArmBlockSpec { int id; double L; int n_max; };

// Column order: [flux (if include_flux)] then per block: for n=1..n_max {cos, sin}.
// Returns Mraw; Wcols is column-major N x Mraw. block_cols (optional) receives the column count per block.
inline long build_multiarm_modes(const std::vector<double>& Xn0, long N,
                                 const std::vector<int>& armid, const std::vector<double>& s,
                                 const std::vector<ArmBlockSpec>& blocks, bool include_flux,
                                 std::vector<double>& Wcols, std::vector<long>* block_cols = nullptr) {
  const double PI = 3.14159265358979323846;
  const long np = N / 3;
  long Mraw = include_flux ? 1 : 0;
  for (const auto& b : blocks) Mraw += 2L * b.n_max;
  Wcols.assign((size_t)N * Mraw, 0.0);
  if (block_cols) block_cols->clear();
  long col = 0;
  if (include_flux) {
    double* w = &Wcols[0];
    for (long i = 0; i < N; i++) w[i] = Xn0[i];
    col++;
  }
  for (const auto& b : blocks) {
    for (int n = 1; n <= b.n_max; n++) {
      const double k = n * PI / b.L;
      double* wc = &Wcols[(size_t)col * N];
      double* ws = &Wcols[(size_t)(col + 1) * N];
      for (long p = 0; p < np; p++) {
        if (armid[p] != b.id) continue;
        const double c = std::cos(k * s[p]), sn = std::sin(k * s[p]);
        for (int d = 0; d < 3; d++) { wc[3*p+d] = c * Xn0[3*p+d]; ws[3*p+d] = sn * Xn0[3*p+d]; }
      }
      col += 2;
    }
    if (block_cols) block_cols->push_back(2L * b.n_max);
  }
  return Mraw;
}

} // namespace quad_junctions
