/**
 * bifurc-network-vtk -- dump the geometry of an ALREADY-ASSEMBLED vessel network to VTK, WITHOUT
 * re-forming any mesh.  Reads the per-junction bundles written by bifurc-network-assemble
 * (<prefix>-jNNN.{mesh,arms}) through ReadNetworkBundle (gen_network_geom.hpp): the junction bodies
 * via the MPI-aware QuadElemList reader, the bent centerline arms via their raw stored CSBQ slender
 * arrays.  The junctions are partitioned round-robin across MPI ranks, each rank builds its local
 * QuadElemList + SlenderElemList, and WriteVTK(World) emits one parallel VTU piece per rank.
 *
 *   make MPI=1 bin/bifurc-network-vtk
 *   OMP_NUM_THREADS=2 mpirun -n 4 ./bin/bifurc-network-vtk <bundle-prefix> [out-prefix]
 *
 * e.g.  OMP_NUM_THREADS=2 mpirun -n 4 ./bin/bifurc-network-vtk vis/netfix vis/netfix-geom
 * -> vis/netfix-geom-junc.vtu.pvtu (+ NNNNNN.vtu pieces) and vis/netfix-geom-arms.vtu.pvtu.
 */
#include <csbq.hpp>
#include <quad_junctions/gen_network_geom.hpp>
#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>

using namespace sctl;
using namespace quad_junctions;

namespace { using Real = double; }

int main(int argc, char** argv) {
  Comm::MPI_Init(&argc, &argv);
  {
    const Comm comm = Comm::World();
    const Integer rank = comm.Rank(), np = comm.Size();
    if (argc < 2) {
      if (!rank) std::cerr << "usage: bifurc-network-vtk <bundle-prefix> [out-prefix]\n";
      Comm::MPI_Finalize(); return 1;
    }
    const std::string prefix = argv[1];
    const std::string out    = (argc > 2) ? std::string(argv[2]) : prefix + "-geom";

    // ---- enumerate junction bundles by file existence (all ranks scan identically) ----
    std::vector<Integer> jids;
    for (Integer i = 0; i < 100000; i++) {
      std::ostringstream nm; nm << prefix << "-j" << std::setw(3) << std::setfill('0') << i;
      std::ifstream probe(nm.str() + ".mesh");
      if (probe.good()) jids.push_back(i);
    }
    if (jids.empty()) {
      if (!rank) std::cerr << "bifurc-network-vtk: no bundles found at '" << prefix << "-jNNN.mesh'\n";
      Comm::MPI_Finalize(); return 1;
    }
    if (!rank) std::cout << "bifurc-network-vtk: " << jids.size() << " junction bundles, "
                         << np << " ranks x " << (getenv("OMP_NUM_THREADS") ? getenv("OMP_NUM_THREADS") : "?")
                         << " threads; loading (no re-forming) ...\n";

    // ---- this rank's round-robin share ----
    Vector<Real> Xloc;                                       // concatenated junction-body coord0 for this rank
    Vector<Long> a_elem, a_forder; Vector<Real> a_coord, a_radius, a_orient;   // concatenated arm arrays
    // Per-node "junction_id" fields so each junction is NUMBERED in the VTK (color by it, or threshold to
    // isolate a problematic junction). Junc field: one value per body node. Arm field: SlenderElemList F is
    // per surface node = sum_panel(cheb*fourier), constant = owner junction id across each arm bundle.
    Vector<Real> Fjunc, Farms;                               // parallel to Xloc-nodes and arm surface-nodes
    Vector<Real> id_ctr;                                     // local (jid, cx, cy, cz) rows for the label CSV
    Integer order = 0, nloc = 0;
    for (size_t k = 0; k < jids.size(); k++) {
      if ((Integer)(k % np) != rank) continue;
      Vector<Real> Xj; Integer ord_j = 0; NetworkArmBundle<Real> arms;
      const bool ok = ReadNetworkBundle<Real>(prefix, jids[k], Xj, ord_j, arms, Comm::Self());
      SCTL_ASSERT(ok);
      if (!order) order = ord_j; else SCTL_ASSERT_MSG(order == ord_j, "mixed element orders across bundles");
      const Real jid = (Real)jids[k];
      // body nodes + centroid
      Real cx = 0, cy = 0, cz = 0; const Long nnode_j = Xj.Dim()/3;
      for (Long m = 0; m < Xj.Dim(); m++) Xloc.PushBack(Xj[m]);
      for (Long p = 0; p < nnode_j; p++) { Fjunc.PushBack(jid); cx += Xj[p*3]; cy += Xj[p*3+1]; cz += Xj[p*3+2]; }
      if (nnode_j) { id_ctr.PushBack(jid); id_ctr.PushBack(cx/nnode_j); id_ctr.PushBack(cy/nnode_j); id_ctr.PushBack(cz/nnode_j); }
      // arm surface nodes: cheb*fourier per panel, all tagged with this owner id
      for (Long q = 0; q < arms.elem_order.Dim(); q++) {
        const Long ns = arms.elem_order[q] * arms.forder[q];
        for (Long s = 0; s < ns; s++) Farms.PushBack(jid);
      }
      for (Long m = 0; m < arms.elem_order.Dim(); m++) a_elem.PushBack(arms.elem_order[m]);
      for (Long m = 0; m < arms.forder.Dim();     m++) a_forder.PushBack(arms.forder[m]);
      for (Long m = 0; m < arms.coord.Dim();      m++) a_coord.PushBack(arms.coord[m]);
      for (Long m = 0; m < arms.radius.Dim();     m++) a_radius.PushBack(arms.radius[m]);
      for (Long m = 0; m < arms.orient.Dim();     m++) a_orient.PushBack(arms.orient[m]);
      nloc++;
    }

    // ---- build this rank's local lists (Self: keep all local elements) and write parallel VTU pieces ----
    QuadElemList<Real> junc(order, Xloc, Comm::Self());
    SlenderElemList<Real> arms(a_elem, a_forder, a_coord, a_radius, a_orient);

    const Long njg = GlobalReduce((Long)junc.Size(), comm, CommOp::SUM);
    const Long nag = GlobalReduce((Long)arms.Size(), comm, CommOp::SUM);
    if (!rank) std::cout << "  total junction panels=" << njg << "  arm panels=" << nag << "\n";

    // NOTE: VTUData::WriteVTK appends its own ".pvtu" and "NNNNNN.vtu" to `fname`, so pass the BARE
    // base (no ".vtu") -- otherwise you get the doubled "...-junc.vtu.pvtu" / "...-junc.vtu000000.vtu".
    // The F field paints a per-node "junction_id" scalar so each junction is numbered in ParaView.
    junc.WriteVTK(out + "-junc", Fjunc, comm);   // World -> np pieces, union = whole network
    arms.WriteVTK(out + "-arms", Farms, comm);

    // ---- gather (jid, centroid) to rank 0 and write a label CSV (Table-to-Points + Point Labels) ----
    {
      StaticArray<Long,1> len{id_ctr.Dim()};
      Vector<Long> cnt(np), dsp(np);
      comm.Allgather(len + 0, 1, cnt.begin(), 1);
      dsp = 0; for (Integer r = 1; r < np; r++) dsp[r] = dsp[r-1] + cnt[r-1];
      Vector<Real> all(dsp[np-1] + cnt[np-1]);
      comm.Allgatherv(id_ctr.begin(), id_ctr.Dim(), all.begin(), cnt.begin(), dsp.begin());
      if (!rank) {
        std::ofstream csv(out + "-junction-ids.csv");
        csv << "junction_id,x,y,z\n" << std::setprecision(10);
        for (Long i = 0; i + 3 < all.Dim(); i += 4)
          csv << (long)(all[i] + 0.5) << "," << all[i+1] << "," << all[i+2] << "," << all[i+3] << "\n";
        std::cout << "  wrote " << out << "-junction-ids.csv (" << all.Dim()/4 << " junctions)\n";
      }
    }
    if (!rank) std::cout << "  wrote " << out << "-{junc,arms}.pvtu (+ " << np
                         << " NNNNNN.vtu pieces each; junction_id scalar field)\n";
  }
  Comm::MPI_Finalize();
  return 0;
}
