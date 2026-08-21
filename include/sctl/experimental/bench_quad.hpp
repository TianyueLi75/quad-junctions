#ifndef _SCTL_BENCH_QUAD_HPP_
#define _SCTL_BENCH_QUAD_HPP_

// Lightweight, opt-in phase timers for the QuadElemList self/near hot loops.
//
// All instrumentation is gated on the BENCH_QUAD macro. When BENCH_QUAD is NOT
// defined the BENCH_TIC/BENCH_TOC/BENCH_FLOPS/BENCH_NEAR macros expand to nothing,
// so the instrumented translation units compile byte-identical to the
// uninstrumented build (zero runtime cost). The Reset()/Report() helpers always
// exist so the benchmark driver links in both builds.
//
// Build instrumented:   make BENCH=1 bin/bench-quad-interac
// Run single-threaded:  OMP_NUM_THREADS=1 ./bin/bench-quad-interac

#include <array>
#include <cmath>
#include <cstdio>
#include <string>

#ifdef _OPENMP
#include <omp.h>
#else
#include <chrono>
#endif

namespace sctl {
namespace bench {

// Phases of the self/near setup. Each phase includes the allocation of the
// temporaries it consumes (per-call malloc/free churn shows up inside its owning
// phase rather than as a separate line). NumPhases must stay last.
enum class Phase {
  InterpBuild = 0, // adaptive Mu_local/Mv_local (LagrangeInterp + GEMM + Transpose)
  GeomTensor,      // coord_shift + 3 geometry EvalTensorProduct calls (6 GEMMs)
  Assembly,        // Xsrc/Xnsrc/wq alloc + per-node normal/area/weight loop
  KernelEval,      // ker.KernelMatrix
  KernelWeight,    // KWc alloc + weighting loop
  Projection,      // projection EvalTensorProduct + scatter (2 GEMMs)
  QuadtreeBuild,   // BuildNearLeaves (adaptive near only)
  ClosestPoint,    // GetClosestPoint Newton/grid search (RectPolar near only)
  ClosestNode,     // GetClosestNode brute-force seed (adaptive near only)
  NearClosestPt,   // Duffy near: GetClosestPoint + EvalPoint foot solve (NearInteracBlockSplitDuffy)
  NearSubpanelInterp, // Duffy near: split-operator (Sf/St) build + Xsub subpanel GEMMs
  NearCellIntegrate,  // Duffy near: refinement/emit loop (all IntegrateNearCM cells for one target)
  NumPhases
};

inline constexpr int kNumPhases = static_cast<int>(Phase::NumPhases);
inline constexpr int kMaxThreads = 256;

inline const char* PhaseName(Phase p) {
  switch (p) {
    case Phase::InterpBuild:   return "InterpBuild";
    case Phase::GeomTensor:    return "GeomTensor";
    case Phase::Assembly:      return "Assembly";
    case Phase::KernelEval:    return "KernelEval";
    case Phase::KernelWeight:  return "KernelWeight";
    case Phase::Projection:    return "Projection";
    case Phase::QuadtreeBuild: return "QuadtreeBuild";
    case Phase::ClosestPoint:  return "ClosestPoint";
    case Phase::ClosestNode:   return "ClosestNode";
    case Phase::NearClosestPt:      return "NearClosestPt";
    case Phase::NearSubpanelInterp: return "NearSubpanelInterp";
    case Phase::NearCellIntegrate:  return "NearCellIntegrate";
    default:                   return "?";
  }
}

// Per-thread row of [time, count] keyed by phase, plus a running count of the
// "real" GEMM flops actually issued (interpolation/projection tensor products);
// padded to avoid false sharing.
struct PhaseRow {
  double t[kNumPhases];
  long   n[kNumPhases];
  double gemm_flops;
  long   near_leaves;    // total quadtree leaves summed over adaptive-near targets
  long   near_targets;   // number of adaptive-near NearInteracBlockGraded calls
  long   near_max_depth; // deepest subdivision reached across those targets
  // Duffy near (NearInteracBlockSplitDuffy) attribution counters -- the weak-scaling probes.
  long   duffy_targets;       // # NearInteracBlockSplitDuffy calls (the near work unit)
  long   duffy_elevated;      // # calls where the angle-adjusted order q_chosen > q_iso
  long   duffy_order_sum;     // sum of chosen near GL order  (mean = /duffy_targets)
  long   duffy_order_iso_sum; // sum of the isotropic baseline order q_iso (mean elevation factor)
  long   duffy_cells;         // total IntegrateNearCM cells emitted over all targets
  long   duffy_cells_elevated;// of those, how many ran at an elevated order (the cost of the bump)
  long   cp_iter_sum;         // closest-point Gauss-Newton iterations summed over targets
  long   cp_fallback;         // # targets that fell back to the shrinking-box grid search
  // Fallback-cause split (why did the Newton not certify convergence?):
  long   cp_fb_stall;         //   line-search stall (iters < kClosestMaxIter): break before max_iter
  long   cp_fb_maxiter;       //   iteration exhaustion (iters >= kClosestMaxIter)
  double cp_fb_dist_sum;      //   sum of foot distance over fallback targets (mean dist of the hard cases)
  double cp_all_dist_sum;     //   sum of foot distance over ALL near targets (baseline mean dist)
  char   pad[64];
};

// Must match `max_iter` in QuadElemList::GetClosestPoint (quad_element.cpp). Used only to classify a
// fallback as an early line-search stall vs iteration exhaustion; a stale value only mislabels, never
// miscounts, so it is a diagnostic aid, not a correctness dependency.
inline constexpr long kClosestMaxIter = 30;

// Single global table; index by OpenMP thread id. Defined inline (C++17) so the
// header can be included by multiple TUs without a separate .cpp.
inline std::array<PhaseRow, kMaxThreads>& Table() {
  static std::array<PhaseRow, kMaxThreads> table{};
  return table;
}

inline double Wtime() {
#ifdef _OPENMP
  return omp_get_wtime();
#else
  return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
#endif
}

inline int ThreadId() {
#ifdef _OPENMP
  const int t = omp_get_thread_num();
  return (t < kMaxThreads ? t : 0);
#else
  return 0;
#endif
}

inline void Accum(Phase p, double dt) {
  PhaseRow& row = Table()[ThreadId()];
  row.t[static_cast<int>(p)] += dt;
  row.n[static_cast<int>(p)] += 1;
}

inline void AccumFlops(double f) {
  Table()[ThreadId()].gemm_flops += f;
}

inline void AccumNear(long nleaf, long depth) {
  PhaseRow& row = Table()[ThreadId()];
  row.near_leaves += nleaf;
  row.near_targets += 1;
  if (depth > row.near_max_depth) row.near_max_depth = depth;
}

// One call per NearInteracBlockSplitDuffy target: records the chosen vs isotropic near GL order
// (angle-adjusted-order attribution) and the closest-point solve cost.
inline void AccumDuffyTarget(long q_chosen, long q_iso, long cp_iters, bool cp_fallback, double dist) {
  PhaseRow& row = Table()[ThreadId()];
  row.duffy_targets       += 1;
  row.duffy_order_sum     += q_chosen;
  row.duffy_order_iso_sum += q_iso;
  if (q_chosen > q_iso) row.duffy_elevated += 1;
  row.cp_iter_sum += cp_iters;
  row.cp_all_dist_sum += dist;
  if (cp_fallback) {
    row.cp_fallback += 1;
    if (cp_iters >= kClosestMaxIter) row.cp_fb_maxiter += 1; else row.cp_fb_stall += 1;
    row.cp_fb_dist_sum += dist;
  }
}

// One call per emitted IntegrateNearCM cell; `elevated` = this target used q_chosen > q_iso.
inline void AccumDuffyCell(bool elevated) {
  PhaseRow& row = Table()[ThreadId()];
  row.duffy_cells += 1;
  if (elevated) row.duffy_cells_elevated += 1;
}

inline double TotalFlops() {
  double f = 0;
  for (int th = 0; th < kMaxThreads; th++) f += Table()[th].gemm_flops;
  return f;
}

inline void Reset() {
  Table() = std::array<PhaseRow, kMaxThreads>{};
}

// Reduce across threads and print a per-phase breakdown. `label` tags the block;
// `outer_seconds` is the wall time of the whole timed region (for a "measured
// vs. total" coverage check).
// Aggregate of the per-thread table, reduced across threads. Split out so both the
// single-process Report() and the MPI ReportMPI() can share the reduction/printing.
struct Aggregate {
  double t[kNumPhases] = {};
  long   n[kNumPhases] = {};
  long   near_leaves = 0, near_targets = 0, near_max_depth = 0;
  long   duffy_targets = 0, duffy_elevated = 0, duffy_order_sum = 0, duffy_order_iso_sum = 0;
  long   duffy_cells = 0, duffy_cells_elevated = 0, cp_iter_sum = 0, cp_fallback = 0;
  long   cp_fb_stall = 0, cp_fb_maxiter = 0;
  double cp_fb_dist_sum = 0, cp_all_dist_sum = 0;
  double gemm_flops = 0;
};

inline Aggregate Collect() {
  Aggregate a;
  for (int th = 0; th < kMaxThreads; th++) {
    const PhaseRow& row = Table()[th];
    for (int p = 0; p < kNumPhases; p++) { a.t[p] += row.t[p]; a.n[p] += row.n[p]; }
    a.near_leaves += row.near_leaves;
    a.near_targets += row.near_targets;
    if (row.near_max_depth > a.near_max_depth) a.near_max_depth = row.near_max_depth;
    a.duffy_targets += row.duffy_targets;
    a.duffy_elevated += row.duffy_elevated;
    a.duffy_order_sum += row.duffy_order_sum;
    a.duffy_order_iso_sum += row.duffy_order_iso_sum;
    a.duffy_cells += row.duffy_cells;
    a.duffy_cells_elevated += row.duffy_cells_elevated;
    a.cp_iter_sum += row.cp_iter_sum;
    a.cp_fallback += row.cp_fallback;
    a.cp_fb_stall += row.cp_fb_stall;
    a.cp_fb_maxiter += row.cp_fb_maxiter;
    a.cp_fb_dist_sum += row.cp_fb_dist_sum;
    a.cp_all_dist_sum += row.cp_all_dist_sum;
    a.gemm_flops += row.gemm_flops;
  }
  return a;
}

// Print a per-phase breakdown + the grep-friendly [nearbench] key=value summary line from a
// (possibly cross-rank-reduced) Aggregate. `label` tags the block; `outer_seconds` is the wall
// time of the whole timed region (for a "measured vs. total" coverage check).
inline void PrintAggregate(const Aggregate& a, const std::string& label, double outer_seconds) {
#ifdef BENCH_QUAD
  const double* t = a.t;
  const long*   n = a.n;
  double sum = 0;
  for (int p = 0; p < kNumPhases; p++) sum += t[p];
  std::printf("  [bench] %s  (summed over threads/ranks; phase total = %.4g s)\n", label.c_str(), sum);
  std::printf("    %-18s %12s %10s %14s\n", "phase", "time[s]", "calls", "%phase");
  for (int p = 0; p < kNumPhases; p++) {
    const double pct = (sum > 0 ? 100.0 * t[p] / sum : 0.0);
    std::printf("    %-18s %12.5e %10ld %13.1f%%\n",
                PhaseName(static_cast<Phase>(p)), t[p], n[p], pct);
  }
  const double gflops = a.gemm_flops * 1e-9;
  std::printf("    %-18s %12.5e GFLOP issued in tensor GEMMs\n", "gemm_flops", gflops);
  if (a.near_targets > 0) {
    std::printf("    %-18s %12.3f leaves/target  (max_depth=%ld over %ld adaptive-near targets)\n",
                "near_quadtree", (double)a.near_leaves / a.near_targets, a.near_max_depth, a.near_targets);
  }
  if (outer_seconds > 0) {
    std::printf("    %-18s %12.5e   (phase sum is %.1f%% of outer wall time)\n",
                "outer", outer_seconds, 100.0 * sum / outer_seconds);
  }
  // Duffy-near attribution: the weak-scaling probes. Per-target means so the numbers are
  // comparable across problem sizes (rising cost-per-target = the weak-scaling driver).
  if (a.duffy_targets > 0) {
    const double inv = 1.0 / (double)a.duffy_targets;
    const double mean_order = a.duffy_order_sum * inv;
    const double mean_iso   = a.duffy_order_iso_sum * inv;
    const double elev_frac  = a.duffy_elevated * inv;
    const double cp_iters   = a.cp_iter_sum * inv;
    const double fb_frac    = a.cp_fallback * inv;
    const double cells_per  = a.duffy_cells * inv;
    const double elev_cell  = (a.duffy_cells > 0 ? (double)a.duffy_cells_elevated / a.duffy_cells : 0.0);
    const double t_closest  = t[static_cast<int>(Phase::NearClosestPt)];
    const double t_subpanel = t[static_cast<int>(Phase::NearSubpanelInterp)];
    const double t_cellint  = t[static_cast<int>(Phase::NearCellIntegrate)];
    // Fallback-cause split: of the fallbacks, what fraction were early stalls vs iteration-exhaustion,
    // and how deep-near are the fallback targets (mean dist) vs all near targets. A stall-dominated,
    // dist<<mean split is the signature of a SPURIOUS fallback (foot found, convergence not certified).
    const double fb_stall_frac = a.cp_fallback > 0 ? (double)a.cp_fb_stall   / a.cp_fallback : 0.0;
    const double fb_max_frac   = a.cp_fallback > 0 ? (double)a.cp_fb_maxiter / a.cp_fallback : 0.0;
    const double mean_all_dist = a.cp_all_dist_sum * inv;
    const double mean_fb_dist  = a.cp_fallback > 0 ? a.cp_fb_dist_sum / a.cp_fallback : 0.0;
    std::printf("    %-18s targets=%ld cp_iters=%.2f fallback=%.4f (stall=%.3f maxiter=%.3f) "
                "fbdist/alldist=%.3e/%.3e elev_frac=%.4f "
                "mean_order=%.2f mean_iso=%.2f cells_per_tgt=%.2f elev_cell_frac=%.4f\n",
                "duffy_near", a.duffy_targets, cp_iters, fb_frac, fb_stall_frac, fb_max_frac,
                mean_fb_dist, mean_all_dist, elev_frac,
                mean_order, mean_iso, cells_per, elev_cell);
    // Single machine-parseable line (parse_cilia_scaling.sh keys off this prefix).
    std::printf("[nearbench] label=%s targets=%ld t_closest=%.6e t_subpanel=%.6e t_cellint=%.6e "
                "cp_iters=%.4f fallback=%.6f fb_stall_frac=%.6f fb_max_frac=%.6f "
                "mean_fb_dist=%.6e mean_all_dist=%.6e elev_frac=%.6f mean_order=%.4f mean_iso=%.4f "
                "cells_per_tgt=%.4f elev_cell_frac=%.6f\n",
                label.c_str(), a.duffy_targets, t_closest, t_subpanel, t_cellint,
                cp_iters, fb_frac, fb_stall_frac, fb_max_frac, mean_fb_dist, mean_all_dist,
                elev_frac, mean_order, mean_iso, cells_per, elev_cell);
  }
  if (outer_seconds > 0) {
    std::printf("    %-18s %12.4f GFLOP/s (true GEMM throughput vs outer wall)\n",
                "gemm_f/s", gflops / outer_seconds);
  }
#else
  (void)a; (void)label; (void)outer_seconds;
  std::printf("  [bench] %s: instrumentation disabled (rebuild with BENCH=1 / -DBENCH_QUAD)\n", label.c_str());
#endif
}

inline void Report(const std::string& label, double outer_seconds = 0) {
  const Aggregate a = Collect();
  PrintAggregate(a, label, outer_seconds);
}

// MPI variant: the driver supplies a `reduce` callback that SUM-reduces a double buffer in place
// across ranks (e.g. wrapping quad_junctions::GlobalReduce(Vector<double>, comm, SUM)), and
// `is_root` so only rank 0 prints. The header stays free of any sctl::Comm dependency this way.
// All aggregates are additive, so a single SUM reduction of one flat buffer suffices.
template <class ReduceFn>
inline void ReportMPI(ReduceFn reduce, bool is_root, const std::string& label, double outer_seconds = 0) {
  Aggregate a = Collect();
  // Pack every scalar into one buffer, SUM across ranks, unpack.
  double buf[kNumPhases /*t*/ + kNumPhases /*n*/ + 16 + 1 /*outer*/];
  int m = 0;
  for (int p = 0; p < kNumPhases; p++) buf[m++] = a.t[p];
  for (int p = 0; p < kNumPhases; p++) buf[m++] = (double)a.n[p];
  buf[m++] = (double)a.near_leaves;
  buf[m++] = (double)a.near_targets;
  buf[m++] = (double)a.near_max_depth; // SUM of a max is not a max, but near_max_depth is only for the adaptive path; unused here.
  buf[m++] = (double)a.duffy_targets;
  buf[m++] = (double)a.duffy_elevated;
  buf[m++] = (double)a.duffy_order_sum;
  buf[m++] = (double)a.duffy_order_iso_sum;
  buf[m++] = (double)a.duffy_cells;
  buf[m++] = (double)a.duffy_cells_elevated;
  buf[m++] = (double)a.cp_iter_sum;
  buf[m++] = (double)a.cp_fallback;
  buf[m++] = (double)a.cp_fb_stall;
  buf[m++] = (double)a.cp_fb_maxiter;
  buf[m++] = a.cp_fb_dist_sum;
  buf[m++] = a.cp_all_dist_sum;
  buf[m++] = a.gemm_flops;
  buf[m++] = outer_seconds;
  reduce(buf, m);
  m = 0;
  Aggregate g;
  for (int p = 0; p < kNumPhases; p++) g.t[p] = buf[m++];
  for (int p = 0; p < kNumPhases; p++) g.n[p] = (long)llround(buf[m++]);
  g.near_leaves = (long)llround(buf[m++]);
  g.near_targets = (long)llround(buf[m++]);
  g.near_max_depth = (long)llround(buf[m++]);
  g.duffy_targets = (long)llround(buf[m++]);
  g.duffy_elevated = (long)llround(buf[m++]);
  g.duffy_order_sum = (long)llround(buf[m++]);
  g.duffy_order_iso_sum = (long)llround(buf[m++]);
  g.duffy_cells = (long)llround(buf[m++]);
  g.duffy_cells_elevated = (long)llround(buf[m++]);
  g.cp_iter_sum = (long)llround(buf[m++]);
  g.cp_fallback = (long)llround(buf[m++]);
  g.cp_fb_stall = (long)llround(buf[m++]);
  g.cp_fb_maxiter = (long)llround(buf[m++]);
  g.cp_fb_dist_sum = buf[m++];
  g.cp_all_dist_sum = buf[m++];
  g.gemm_flops = buf[m++];
  const double outer_sum = buf[m++];
  if (is_root) PrintAggregate(g, label, outer_sum);
}

} // namespace bench
} // namespace sctl

// Hot-path macros: real code only under BENCH_QUAD, otherwise nothing.
#ifdef BENCH_QUAD
#define BENCH_TIC(phase) const double _bench_t0_##phase = ::sctl::bench::Wtime()
#define BENCH_TOC(phase) ::sctl::bench::Accum(::sctl::bench::Phase::phase, ::sctl::bench::Wtime() - _bench_t0_##phase)
#define BENCH_FLOPS(n) ::sctl::bench::AccumFlops((double)(n))
#define BENCH_NEAR(nleaf, depth) ::sctl::bench::AccumNear((long)(nleaf), (long)(depth))
#define BENCH_DUFFY_TARGET(qc, qi, it, fb, d) ::sctl::bench::AccumDuffyTarget((long)(qc), (long)(qi), (long)(it), (fb), (double)(d))
#define BENCH_DUFFY_CELL(elevated) ::sctl::bench::AccumDuffyCell((elevated))
#else
#define BENCH_TIC(phase) ((void)0)
#define BENCH_TOC(phase) ((void)0)
#define BENCH_FLOPS(n) ((void)0)
#define BENCH_NEAR(nleaf, depth) ((void)0)
#define BENCH_DUFFY_TARGET(qc, qi, it, fb, d) ((void)0)
#define BENCH_DUFFY_CELL(elevated) ((void)0)
#endif

#endif // _SCTL_BENCH_QUAD_HPP_
