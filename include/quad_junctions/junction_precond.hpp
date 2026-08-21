// =============================================================================
// junction_precond.hpp
//
// Block-diagonal LEFT preconditioner for the hybrid vessels/bifurcation solves.
//
// Motivation: the interior Stokes inflow/outflow solve on the 20-junction vessels
// network stalls in GMRES (flat at ~1.3e-2 relative for 390 of 400 iterations).
// The far field was measured clean against EvalDirect (Laplace AND Stokes, 1 and 2
// ranks, max 7.4e-05) and the formulation is validated on single racetrack loops, so
// the residual plateau is a CONDITIONING problem, not a wrong operator.
//
// Method (follows ../stokes-periodize-numtest: include/utils_tests.cpp
// `precond_channel` / `precond_ptcl`, applied in test/examples.cpp):
//   1. Build the dense self-interaction matrix A11 of ONE canonical junction by
//      applying the same combined-field operator the solve uses to each unit density
//      e_col (i.e. "apply BIO to the junction with the identity as density").
//   2. SVD it and pseudo-invert: A11^+ = V S^+ U^T, stored FACTORED as
//      P0 = V and P1 = S^+ U^T, so the apply is P0 * (P1 * v) -- exactly the
//      reference's `PrecondMat0 * (PrecondMat1 * vecMat)`.
//   3. Cache both factors to disk keyed by every parameter the mesh and the operator
//      depend on, since the build is the expensive part.
//   4. Apply block-diagonally to the JUNCTION rows only (caps, arms and any junction
//      split across an MPI rank boundary get the identity), then left-precondition
//      both the operator and the RHS:  A11^+ A sigma = A11^+ b.
//
// The block is the CANONICAL (pre-placement) junction, which is why one matrix serves
// all 20: placements differ by a similarity transform (scale 0.9^gen + rotation), and
// a preconditioner only has to cluster the spectrum, not be exact. The reference does
// the same -- one `PrecondMat_ptcl` for particles of differing size and orientation.
//
// SIZE WARNING. The block is 3*npj*order^2 DOF with npj = 3*Na*(Nr+Ns_trans):
//   order 12, nref 1, Ns_trans 2 -> 192 panels -> 82,944 DOF -> 55 GB dense, ~3 h SVD
//   order  8, nref 1, Ns_trans 2 -> 192 panels -> 36,864 DOF -> 10.9 GB, ~17 min
// So `QJ_PRECOND_BLOCK` selects the granularity:
//   junction (default) -- as above; guarded by QJ_PRECOND_MAXGB (default 8 GB) so it
//                         fails fast with the numbers instead of OOMing after hours.
//   panel               -- one order^2 patch, 3*order^2 DOF (432 at order 12). This is
//                         the reference's `precond_channel` granularity: much weaker
//                         (it only captures the panel-diagonal) but essentially free.
//   off                 -- no preconditioning; the pre-existing behaviour.
// =============================================================================
#ifndef QUAD_JUNCTIONS_JUNCTION_PRECOND_HPP
#define QUAD_JUNCTIONS_JUNCTION_PRECOND_HPP

#include <csbq.hpp>
#include <stokes_bio.hpp>
#include <quad_junctions/ybifurc_assembly.hpp>
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <string>
#include <sys/stat.h>

namespace quad_junctions {

using namespace sctl;

enum class PrecondBlockKind { Off, Panel, Junction };

inline PrecondBlockKind precond_block_kind() {
  const char* e = std::getenv("QJ_PRECOND_BLOCK");
  if (!e) return PrecondBlockKind::Junction;
  const std::string s(e);
  if (s == "off"  || s == "0")    return PrecondBlockKind::Off;
  if (s == "panel")              return PrecondBlockKind::Panel;
  if (s == "junction" || s == "1") return PrecondBlockKind::Junction;
  SCTL_ASSERT_MSG(false, "QJ_PRECOND_BLOCK must be one of: off | panel | junction");
  return PrecondBlockKind::Off;
}

inline const char* precond_kind_name(PrecondBlockKind k) {
  switch (k) {
    case PrecondBlockKind::Off:      return "off";
    case PrecondBlockKind::Panel:    return "panel";
    case PrecondBlockKind::Junction: return "junction";
  }
  return "?";
}

// Everything solve_dirichlet_bvp needs to build the block itself. The operator scalings and
// the jump are deliberately NOT here: the solve passes its own (post-SL-sign-flip) values, so
// the preconditioned operator and the solved operator cannot drift apart.
template <class Real> struct JunctionPrecondSpec {
  PrecondBlockKind kind = PrecondBlockKind::Off;
  Integer order = 0;
  Real    level = 0;
  Integer nref = 0;
  Real    eta_join = 0;
  Integer Ns_trans = 0;
  Long    njunc = 0;      // number of junctions emitted before the caps in the junc elem list
};

// Factored pseudo-inverse of one block, plus the block->row bookkeeping.
template <class Real> struct BlockPrecond {
  Matrix<Real> P0, P1;      // A11^+ = P0 * P1   (P0 = V, P1 = S^+ U^T); size blk
  Long blk = 0;             // DOF per block -- the DOF the Apply slices/writes
  Long nblk_local = 0;      // number of whole blocks this rank applies it to
  Long dof_offset = 0;      // local DOF index where those blocks start
  PrecondBlockKind kind = PrecondBlockKind::Off;

  bool active() const {
    if (kind == PrecondBlockKind::Off || blk <= 0 || nblk_local <= 0) return false;
    if (!(P0.Dim(0) == blk && P0.Dim(1) == blk && P1.Dim(0) == blk && P1.Dim(1) == blk)) return false;
    return true;
  }

  // out = A11^+ applied to each whole block in [dof_offset, dof_offset + nblk_local*blk);
  // IDENTITY everywhere else (caps, arms, rank-split junctions). A block-diagonal
  // preconditioner with identity blocks is still a valid preconditioner -- it just does
  // no work on those rows -- so this is safe rather than merely convenient. The block
  // apply is the factored w = P0*(P1*v).
  void Apply(Vector<Real>& out, const Vector<Real>& in) const {
    if (out.Dim() != in.Dim()) out.ReInit(in.Dim());
    out = in;                                   // identity default
    if (!active()) return;
    for (Long i = 0; i < nblk_local; i++) {
      const Long o = dof_offset + i*blk;
      SCTL_ASSERT(o + blk <= in.Dim());
      const Matrix<Real> v(blk, 1, (Iterator<Real>)in.begin() + o, false);
      const Matrix<Real> w = P0 * (P1 * v);
      for (Long j = 0; j < blk; j++) out[o + j] = w(j, 0);
    }
  }
};

namespace precond_detail {

inline bool dir_exists(const std::string& p) {
  struct stat st;
  return !stat(p.c_str(), &st) && S_ISDIR(st.st_mode);
}

// mkdir -p, so a fresh checkout does not need the cache dir committed.
inline void make_dirs(const std::string& p) {
  std::string acc;
  for (size_t i = 0; i <= p.size(); i++) {
    if (i == p.size() || p[i] == '/') {
      if (!acc.empty() && !dir_exists(acc)) ::mkdir(acc.c_str(), 0775);
    }
    if (i < p.size()) acc.push_back(p[i]);
  }
}

inline std::string cache_dir() {
  const char* e = std::getenv("QJ_PRECOND_CACHE_DIR");
  return e ? std::string(e) : std::string("data/precond-cache");
}

// A number formatted so it is stable across runs and safe in a filename.
inline std::string tag(double v) {
  char b[64];
  std::snprintf(b, sizeof b, "%.10g", v);
  std::string s(b);
  for (auto& c : s) if (c == '.' || c == '+') c = 'p'; else if (c == '-') c = 'm';
  return s;
}

}  // namespace precond_detail

// -----------------------------------------------------------------------------
// Build (or load) the factored pseudo-inverse of one block.
//
// `jump` MUST be the same jump term the solve applies (solve_dirichlet_bvp uses
// jump = (interior ? -1/2 : +1/2)*DL_scal). Getting it wrong preconditions a
// different operator than the one being solved, which is worse than no
// preconditioner -- hence it is a required argument, not a default.
// -----------------------------------------------------------------------------
template <class Real>
BlockPrecond<Real> build_block_precond(const PrecondBlockKind kind, const Integer order,
                                       const Real level, const Integer nref, const Real eta_join,
                                       const Integer Ns_trans, const Real SL_scal,
                                       const Real DL_scal, const Real jump,
                                       const Real quad_tol, const Comm& comm) {
  BlockPrecond<Real> P;
  P.kind = kind;
  if (kind == PrecondBlockKind::Off) return P;

  const Long npj = (Long)3 * (YSwept::Na0*nref) * (YSwept::Nr0*nref + Ns_trans);
  const Long npe = (Long)order * order * 3;                                 // DOF per panel
  const Long blk = (kind == PrecondBlockKind::Junction) ? npj*npe : npe;    // block DOF (SVD + Apply)
  P.blk = blk;

  const double gb = (double)blk * (double)blk * (double)sizeof(Real) / (1024.0*1024.0*1024.0);
  if (!comm.Rank()) {
    std::cout << "  [precond] block=" << precond_kind_name(kind) << "  npj=" << npj
              << "  DOF/block=" << blk << "  dense A11 = " << std::fixed << std::setprecision(2)
              << gb << " GB (x2 cached factors)" << std::defaultfloat << "\n";
  }
  { // Fail fast with the arithmetic rather than OOM after hours of column builds.
    const char* mg = std::getenv("QJ_PRECOND_MAXGB");
    const double maxgb = mg ? atof(mg) : 8.0;
    char msg[512];
    std::snprintf(msg, sizeof msg,
        "junction_precond: block of %ld DOF needs %.2f GB dense (x2 cached, plus SVD workspace), "
        "over the %.2f GB limit. The SVD alone is ~20*n^3 = %.1e flop. Either set "
        "QJ_PRECOND_BLOCK=panel, lower `order`, or raise "
        "QJ_PRECOND_MAXGB if you really mean it.",
        (long)blk, gb, maxgb, 20.0*(double)blk*(double)blk*(double)blk);
    SCTL_ASSERT_MSG(gb <= maxgb, msg);
  }

  // Cache key: everything the block depends on -- mesh params (via the canonical junction
  // key), the operator scalings, the jump, the quadrature tol, and the block kind. A stale hit
  // here would silently precondition the wrong operator.
  using precond_detail::tag;
  const std::string dir = precond_detail::cache_dir();
  const std::string key = std::string(precond_kind_name(kind))
      + "-ord" + std::to_string(order) + "-L" + tag((double)level)
      + "-nref" + std::to_string(nref) + "-eta" + tag((double)eta_join)
      + "-Ns" + std::to_string(Ns_trans) + "-pou" + std::to_string(pou_kind())
      + "-sl" + tag((double)SL_scal) + "-dl" + tag((double)DL_scal)
      + "-jmp" + tag((double)jump) + "-qtol" + tag((double)quad_tol)
      + "-r" + std::to_string((int)sizeof(Real));
  const std::string f0 = dir + "/jprecond0-" + key + ".mat";
  const std::string f1 = dir + "/jprecond1-" + key + ".mat";

  P.P0.template Read<Real>(f0.c_str());
  if (P.P0.Dim(0) == blk && P.P0.Dim(1) == blk) {
    P.P1.template Read<Real>(f1.c_str());
    if (P.P1.Dim(0) == blk && P.P1.Dim(1) == blk) {
      if (!comm.Rank()) std::cout << "  [precond] loaded " << f0 << "\n";
      return P;
    }
    if (!comm.Rank()) std::cout << "  [precond] WARNING: " << f1 << " missing/mismatched -- rebuilding\n";
  } else if (P.P0.Dim(0) || P.P0.Dim(1)) {
    if (!comm.Rank()) std::cout << "  [precond] WARNING: " << f0 << " has wrong size ("
                                << P.P0.Dim(0) << "x" << P.P0.Dim(1) << " != " << blk
                                << ") -- rebuilding\n";
  }

  // ---- Build the dense block at `order`. Serial on every rank (comm.Self()): the matrix is
  // ---- replicated, so no communication and no rank-dependent branching.
  if (!comm.Rank()) std::cout << "  [precond] building " << blk << " columns (order " << order
                              << ") ...\n" << std::flush;

  const CanonMesh<Real>& canon = canonical_junction<Real>(order, level, nref, eta_join, Ns_trans, Comm::Self());
  Vector<Real> Xblk;
  if (kind == PrecondBlockKind::Junction) {
    Xblk = canon.X;
  } else {
    // One representative panel: the FIRST panel of the canonical junction (same choice of
    // "a representative element" the reference makes for the channel wall).
    Xblk.ReInit(npe);
    for (Long i = 0; i < npe; i++) Xblk[i] = canon.X[i];
  }
  QuadElemList<Real> blk_lst(order, Xblk, Comm::Self());
  SCTL_ASSERT(blk_lst.Size()*npe == blk);

  Vector<Real> Xt;
  blk_lst.GetNodeCoord(&Xt, nullptr, nullptr);

  StokesBIO<Real> Op(SL_scal, DL_scal, Comm::Self());
  Op.SetAccuracy(quad_tol);
  Op.AddElemList(blk_lst, "0_blk");
  Op.SetTargetCoord(Xt);

  Matrix<Real> A11(blk, blk);
  {
    Vector<Real> e(blk), col;
    for (Long c = 0; c < blk; c++) {
      e = 0; e[c] = 1;
      Op.ComputePotential(col, e);
      SCTL_ASSERT(col.Dim() == blk);
      for (Long r = 0; r < blk; r++) A11(r, c) = col[r] + (r == c ? jump : (Real)0);
      if (!comm.Rank() && blk >= 20 && (c+1) % (blk/20) == 0)
        std::cout << "  [precond]   " << (100*(c+1)/blk) << "%\r" << std::flush;
    }
    if (!comm.Rank()) std::cout << "  [precond]   columns done            \n" << std::flush;
  }

  // ---- SVD pseudo-inverse, stored factored (same as the reference).
  {
    Matrix<Real> U, S, VT;
    Matrix<Real> A = A11;              // SVD overwrites its input
    A.SVD(U, S, VT);
    Matrix<Real> Sc = S;
    const Matrix<Real> Sinv = Sc.pinv(quad_tol);
    P.P0 = VT.Transpose();
    P.P1 = Sinv * U.Transpose();
  }

  if (!comm.Rank()) {
    precond_detail::make_dirs(dir);
    P.P0.template Write<Real>(f0.c_str());
    P.P1.template Write<Real>(f1.c_str());
    std::cout << "  [precond] wrote " << f0 << "\n";
  }
  return P;
}

// -----------------------------------------------------------------------------
// Decide which LOCAL rows the block applies to.
//
// The assembled surface is AddElemList(junc,"0_junc") then AddElemList(arms,"1_arms"),
// and build_vessels_network emits all `njunc` junctions first (npj panels each), then
// the root caps. So the junction rows are the leading njunc*npj panels of the junc list.
//
// Under MPI the junc list is sliced by panel with no regard for junction boundaries
// (3880 panels / 2 ranks = 1940, and npj=192 does not divide 1940), so a junction can
// straddle a rank boundary. Those get the identity: the reference simply ASSUMES no
// particle is split, which silently misapplies the block if that assumption breaks.
// -----------------------------------------------------------------------------
template <class Real>
void precond_set_local_rows(BlockPrecond<Real>& P, const Long local_junc_panels,
                            const Long njunc, const Integer order, const Integer nref,
                            const Integer Ns_trans, const Comm& comm) {
  if (P.kind == PrecondBlockKind::Off || P.blk <= 0) { P.nblk_local = 0; return; }

  const Long npj = (Long)3 * (YSwept::Na0*nref) * (YSwept::Nr0*nref + Ns_trans);
  const Long npe = (Long)order * order * 3;

  // Exclusive prefix sum of the per-rank junc panel counts -> this rank's global panel dsp.
  Long dsp = 0;
  {
    const Long np = comm.Size();
    Vector<Long> all(np); all = 0;
    Vector<Long> mine(1); mine[0] = local_junc_panels;
    comm.Allgather(mine.begin(), 1, all.begin(), 1);
    for (Long i = 0; i < comm.Rank(); i++) dsp += all[i];
  }
  const Long lo = dsp, hi = dsp + local_junc_panels;   // global panel range owned here

  if (P.kind == PrecondBlockKind::Panel) {
    // Every locally owned junction panel (not caps) gets the block.
    const Long jlo = std::max<Long>(lo, 0), jhi = std::min<Long>(hi, njunc*npj);
    P.nblk_local = std::max<Long>(0, jhi - jlo);
    P.dof_offset = (jlo - lo) * npe;
  } else {
    // Whole junctions fully contained in [lo,hi). Contiguity: junction j occupies global
    // panels [j*npj,(j+1)*npj), so the fully-owned ones form one contiguous run.
    Long first = -1, count = 0;
    for (Long j = 0; j < njunc; j++) {
      const Long a = j*npj, b = a + npj;
      if (a >= lo && b <= hi) { if (first < 0) first = j; count++; }
    }
    P.nblk_local = count;
    P.dof_offset = (first < 0) ? 0 : (first*npj - lo) * npe;
  }

  { // Report coverage -- a preconditioner that silently covers nothing looks like a
    // preconditioner that does not help.
    const Long tot = GlobalReduce(P.nblk_local, comm, CommOp::SUM);
    if (!comm.Rank())
      std::cout << "  [precond] applying " << precond_kind_name(P.kind) << " block ("
                << P.blk << " DOF) to " << tot << " block(s) globally"
                << (P.kind == PrecondBlockKind::Junction
                      ? std::string(" of ") + std::to_string((long)njunc) + " junctions"
                      : std::string())
                << "; caps/arms/rank-split get identity\n";
  }
}

}  // namespace quad_junctions

#endif
