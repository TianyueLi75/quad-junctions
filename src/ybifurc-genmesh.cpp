/**
 * Pre-generate canonical Y-junction meshes into the on-disk cache (data/mesh-cache/ycanon-*.bin).
 *
 * The canonical junction+transition mesh (emit_junction_transitions) is a pure function of
 * (order, level, nref, eta_join, Ns_trans) + pou_kind() + the compile-time YCfg/YSwept constants, and it
 * is the dominant SERIAL cost of forming any multi-junction geometry: the 20-junction vessels network at
 * order 12 / nref 2 used to spend 35+ minutes on it. canonical_junction() caches it in memory and on
 * disk, so this tool just warms the disk half up front -- production runs then read the file instead of
 * solving the Gaussian iso-surface again.
 *
 * Nothing DEPENDS on running this: any driver that misses builds the mesh and writes the file itself.
 * It exists so the first production job doesn't pay for it, and so a new mesh resolution can be prepared
 * on a workstation instead of inside a Slurm allocation.
 *
 * Usage:
 *   ./bin/ybifurc-genmesh                 # --all: order {8,12,16} x nref {1,2} at L1.5 / eta 0.4 / Ns 3 / pou 1
 *   ./bin/ybifurc-genmesh 16 2            # one combination (level/eta/Ns/pou default as above)
 *   ./bin/ybifurc-genmesh 16 2 1.5 0.4 3 1
 *
 * Args: order nref [level] [eta_join] [Ns_trans] [pou_kind]
 * Env:  QJ_MESH_CACHE=0 disables the disk cache (this tool then does nothing useful),
 *       QJ_MESH_CACHE_DIR overrides the directory.
 */

#include <quad_junctions/ybifurc_assembly.hpp>

#include <chrono>
#include <iomanip>
#include <iostream>
#include <string>

using namespace sctl;
using namespace quad_junctions;

template <class Real> static void gen_one(Integer order, Integer nref, Real level, Real eta_join,
                                          Integer Ns_trans, int pou, const Comm& comm) {
  SCTL_ASSERT_MSG(order >= 4 && order <= 48, "order must be in [4,48].");
  SCTL_ASSERT_MSG(nref >= 1, "nref must be >= 1.");
  pou_kind() = pou;                            // part of the cache key; set before the build

  if (!comm.Rank())
    std::cout << "\n[genmesh] order=" << order << " nref=" << nref << " level=" << level
              << " eta_join=" << eta_join << " Ns_trans=" << Ns_trans << " pou=" << pou << "\n";

  const auto t0 = std::chrono::steady_clock::now();
  const CanonMesh<Real>& C = canonical_junction<Real>(order, level, nref, eta_join, Ns_trans, comm);
  const double dt = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();

  if (!comm.Rank()) {
    std::cout << "  nodes=" << C.X.Dim()/3 << "  seam R0=" << std::setprecision(10) << C.seams[0].R0
              << "  a0=" << C.seams[0].a0 << "  ray max|f-level|=" << std::setprecision(3) << C.max_res
              << "  (" << std::setprecision(4) << dt << " s)\n";
  }
}

int main(int argc, char** argv) {
  Comm::MPI_Init(&argc, &argv);
  {
    using Real = double;
    const Comm comm = Comm::World();           // normally serial; under MPI only rank 0 writes the file

    if (!canon_cache_enabled() && !comm.Rank())
      std::cout << "\n[genmesh] WARNING: QJ_MESH_CACHE=0 -- the disk cache is disabled, nothing will be"
                   " written.\n";

    const std::string first = (argc > 1) ? std::string(argv[1]) : std::string("--all");
    if (first == "--all" || first == "-a") {
      // The combinations the ybifurc-* drivers actually use. order 12 / nref 2 is the production mesh
      // (submit-vessels-flow.sh, submit-vessels-tol.sh); order 8 / nref 1 is the cheap sweep setting.
      const Integer orders[] = {8, 12, 16};
      const Integer nrefs[]  = {1, 2};
      if (!comm.Rank())
        std::cout << "\n=== genmesh --all: order {8,12,16} x nref {1,2} at level 1.5 / eta_join 0.4 /"
                     " Ns_trans 3 / pou 1 ===\n";
      for (Integer o : orders)
        for (Integer n : nrefs)
          gen_one<Real>(o, n, (Real)1.5, (Real)0.4, 3, 1, comm);
    } else {
      const Integer order   = (Integer)std::atoi(argv[1]);
      const Integer nref    = (argc > 2) ? (Integer)std::atoi(argv[2]) : 2;
      const Real    level   = (argc > 3) ? (Real)std::atof(argv[3]) : (Real)1.5;
      const Real    etajoin = (argc > 4) ? (Real)std::atof(argv[4]) : (Real)0.4;
      const Integer NsTrans = (argc > 5) ? (Integer)std::atoi(argv[5]) : 3;
      const int     pou     = (argc > 6) ? std::atoi(argv[6]) : 1;
      gen_one<Real>(order, nref, level, etajoin, NsTrans, pou, comm);
    }
    if (!comm.Rank()) std::cout << "\n[genmesh] done.\n";
  }
  Comm::MPI_Finalize();
  return 0;
}
