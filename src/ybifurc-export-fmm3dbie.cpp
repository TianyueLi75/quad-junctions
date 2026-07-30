/**
 * Export the closed all-quad Y-bifurcation surface to fmm3dbie triangular-patch format.
 *
 * Mirrors src/ybifurc-hybrid-bie.cpp's geometry construction with arm_kind=1 (full-quad tube
 * arms), so the exported surface is the SAME closed, watertight, all-quad surface that driver
 * validates via Green's identity: junction body + POU transition tubes + hemisphere caps
 * (BuildYJunctionWithTransitions) + quad-tube arms (BuildYArmsQuadTube), concatenated into one
 * QuadElemList.
 *
 * Each order-12 quad element is split along a diagonal into 2 triangles; the surface is sampled at
 * the reference order-11 Vioreanu-Rokhlin nodes (paper p=12, n_p=78) via QuadElemList::GetGeom,
 * with the quad->triangle affine chain rule giving the reference-triangle tangents. Output is the
 * fmm3dbie srcvals(12,npts) array (xyz, dX/dxi, dX/deta, unit normal), iptype=1.
 *
 *   make bin/ybifurc-export-fmm3dbie
 *   ./bin/ybifurc-export-fmm3dbie [rvnodes_file] [out_file] [order] [nref]
 *      defaults: rvnodes_o11.txt  ybifurc_p12.srcvals  12  2
 */
#include <quad_junctions/ybifurc_hybrid_geom.hpp>   // BuildYJunctionWithTransitions, BuildYArmsQuadTube, YSwept
#include <quad_junctions/stud_sphere_geom.hpp>       // report_area
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

using namespace sctl;
using namespace quad_junctions;

int main(int argc, char** argv) {
  Comm::MPI_Init(&argc, &argv);
  using Real = double;
  {
    const Comm comm = Comm::Self();   // serial export

    const std::string rvfile  = (argc > 1) ? argv[1] : "rvnodes_o11.txt";
    const std::string outfile = (argc > 2) ? argv[2] : "ybifurc_p12.srcvals";
    const Integer order = (argc > 3) ? (Integer)atoi(argv[3]) : 12;
    const Integer nref  = (argc > 4) ? (Integer)atoi(argv[4]) : 2;

    // ---- production Y-bifurcation params (ybifurc-hybrid-bie defaults, arm_kind=1) ----
    const Real    level   = 1.5;
    const Real    etajoin = 0.4;
    const Integer NsTrans = 3;
    const Real    s_cap   = 0.88;
    const Integer nAxial  = 3;
    const Integer Ncap    = -1;   // -> YSwept::Ncap0*nref
    const Integer NsShaft = std::max<Integer>(1, (Integer)std::lround((double)nAxial * 10.0 / order));
    const Integer NaShaft = (Integer)(YSwept::Na0 * nref);   // node-conforming azimuthal count

    // ---- read reference RV nodes (fmm3dbie order) ----
    std::ifstream rv(rvfile);
    if (!rv.good()) { std::cerr << "ERROR: cannot open " << rvfile << " (run dump_rvnodes first)\n"; return 1; }
    long norder_rv = 0, npols = 0;
    rv >> norder_rv >> npols;
    std::vector<Real> uvs(2 * npols);
    for (long k = 0; k < npols; k++) { Real w; rv >> uvs[2*k] >> uvs[2*k+1] >> w; }
    if (!rv.good()) { std::cerr << "ERROR: failed reading " << npols << " RV nodes from " << rvfile << "\n"; return 1; }
    std::cout << "RV nodes: norder=" << norder_rv << " npols=" << npols << "\n";

    // ---- build closed all-quad surface: junction(+transitions+caps) + quad-tube arms ----
    Real R0[3], a0[3], sc[3], max_res = 0;
    QuadElemList<Real> junc = BuildYJunctionWithTransitions<Real>(order, level, nref, etajoin, NsTrans, s_cap,
                                                                  R0, a0, sc, Ncap, &max_res, comm);
    QuadElemList<Real> arms = BuildYArmsQuadTube<Real>(order, R0, a0, sc, NsShaft, NaShaft, comm);

    // concatenate node coords (AoS) -> one combined closed list
    Vector<Real> Xj, Xa;
    junc.GetNodeCoord(&Xj, nullptr, nullptr);
    arms.GetNodeCoord(&Xa, nullptr, nullptr);
    Vector<Real> Xall(Xj.Dim() + Xa.Dim());
    for (Long i = 0; i < Xj.Dim(); i++) Xall[i] = Xj[i];
    for (Long i = 0; i < Xa.Dim(); i++) Xall[Xj.Dim() + i] = Xa[i];
    QuadElemList<Real> full(order, Xall, comm);

    std::cout << "geometry: order=" << order << " nref=" << nref
              << "  junc_elems=" << junc.Size() << " arm_elems=" << arms.Size()
              << " total_elems=" << full.Size() << "  (ray max|f-level|=" << max_res << ")\n";
    std::cout << "--- report_area (combined closed surface) ---\n";
    report_area<Real>(full, comm);   // area, min Jacobian weight, closure |int n dA| (watertightness)

    // ---- split each quad into 2 triangles; sample srcvals at RV nodes ----
    // triangles in quad param (a,b) space; both CCW (det[B-A,C-A]>0) -> outward normal matches GetGeom.
    const Real tri[2][3][2] = { {{0,0},{1,0},{1,1}}, {{0,0},{1,1},{0,1}} };
    const Long Nelem = full.Size();
    const Long npatches = 2 * Nelem;
    const Long npts = npatches * (Long)npols;

    std::ofstream out(outfile);
    if (!out.good()) { std::cerr << "ERROR: cannot open " << outfile << " for writing\n"; return 1; }
    out << npatches << " " << norder_rv << " " << npts << "\n";
    out << std::scientific << std::setprecision(16);

    Real min_dot = 1e30, min_jac = 1e30;
    char buf[512];
    for (Long e = 0; e < Nelem; e++) {
      for (int t = 0; t < 2; t++) {
        const Real* A = tri[t][0]; const Real* B = tri[t][1]; const Real* C = tri[t][2];
        const Real Ba[2] = {B[0]-A[0], B[1]-A[1]}, Ca[2] = {C[0]-A[0], C[1]-A[1]};
        for (long k = 0; k < npols; k++) {
          const Real xi = uvs[2*k], eta = uvs[2*k+1];
          Vector<Real> up(1), vp(1);
          up[0] = A[0] + xi*Ba[0] + eta*Ca[0];
          vp[0] = A[1] + xi*Ba[1] + eta*Ca[1];
          Vector<Real> X, Xn, Xar, dXda, dXdb;
          full.GetGeom(&X, &Xn, &Xar, &dXda, &dXdb, up, vp, e);
          Real dxi[3], deta[3];
          for (int c = 0; c < 3; c++) {
            dxi[c]  = dXda[c]*Ba[0] + dXdb[c]*Ba[1];
            deta[c] = dXda[c]*Ca[0] + dXdb[c]*Ca[1];
          }
          Real n[3] = { dxi[1]*deta[2] - dxi[2]*deta[1],
                        dxi[2]*deta[0] - dxi[0]*deta[2],
                        dxi[0]*deta[1] - dxi[1]*deta[0] };
          const Real nn = std::sqrt(n[0]*n[0] + n[1]*n[1] + n[2]*n[2]);
          if (nn < min_jac) min_jac = nn;
          for (int c = 0; c < 3; c++) n[c] /= nn;
          const Real dotv = n[0]*Xn[0] + n[1]*Xn[1] + n[2]*Xn[2];   // vs GetGeom outward unit normal
          if (dotv < min_dot) min_dot = dotv;
          std::snprintf(buf, sizeof(buf),
            "%.16e %.16e %.16e %.16e %.16e %.16e %.16e %.16e %.16e %.16e %.16e %.16e\n",
            (double)X[0],(double)X[1],(double)X[2],
            (double)dxi[0],(double)dxi[1],(double)dxi[2],
            (double)deta[0],(double)deta[1],(double)deta[2],
            (double)n[0],(double)n[1],(double)n[2]);
          out << buf;
        }
      }
    }
    out.close();

    std::cout << "exported: npatches=" << npatches << " (2 x " << Nelem << ")  norder=" << norder_rv
              << " npts=" << npts << "  -> " << outfile << "\n";
    std::cout << "orientation min(n . n_getgeom)=" << std::setprecision(6) << min_dot
              << "  (want ~+1)   min triangle |dxi x deta|=" << min_jac << "  (want >0, no folds)\n";
    if (min_dot < 0) std::cerr << "*** WARNING: some triangle normal flips relative to GetGeom outward normal ***\n";
  }
  Comm::MPI_Finalize();
  return 0;
}
