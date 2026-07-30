/**
 * Stud/collar-on-sphere GEOMETRY driver (cilia mount migrated from SCTL_quad_element/src/test-gmsh-geom.cpp).
 * Geometry-only checks (no BIE): watertightness/area/Jacobian (report_area), per-region tangent/normal
 * alignment with the sphere (tangent), and per-region cell squareness aspect/orthogonality/scaled-Jac (meshq).
 *
 *   make bin/stud_sphere-geom
 *   OMP_NUM_THREADS=8 ./bin/stud_sphere-geom [mode] [order] [Naz] [r_fil] [R_shaft]
 * modes: (none)=build studded finger + report_area + VTK | tangent | meshq
 */
#include <quad_junctions/stud_sphere_geom.hpp>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <string>

using namespace sctl;
using namespace quad_junctions;

namespace {
template <class Real> void diagnose_tangent_alignment(const Comm& comm, Integer order, Long PatchPerFace, Real R, Integer Naz, Real r_fil, Real grade_exp, Real R_shaft) {
  (void)comm;
  QuadElemList<Real> full = BuildCiliumStuddedSphere<Real>(order, PatchPerFace, R, Naz, r_fil, -1, -1, -1, -1, grade_exp, R_shaft);
  const Long n_cs_elem = 6*PatchPerFace*PatchPerFace - 1;
  // Replicate add_cilium_stud's auto-derived per-region panel counts (H_shaft=0.05 hard-coded there).
  const Real pi = const_pi<Real>(), H_shaft = 0.05, R_foot = R_shaft + r_fil, S = R/(Real)PatchPerFace, az = 2*pi*R_shaft/Naz;
  const Integer Ns   = std::max<Integer>(1, (Integer)std::llround((double)((H_shaft - r_fil)/az)));
  const Integer Nf   = std::max<Integer>(1, (Integer)std::llround((double)((pi/2*r_fil)/az)));
  const Integer cap_pan = 5 * (Naz/4) * (Naz/4);  // butterfly dome (add_cap_butterfly): core + 4 caps
  const Integer Nc   = collar_Nc<Real>(R_foot, S, Naz);
  const Long per_sector = Ns + Nf + Nc, stud_rev = Naz*per_sector; // shaft/fillet/collar blocks precede the cap
  auto region = [&](Long e) -> int { // 0 cubed-sphere, 1 shaft, 2 fillet, 3 collar, 4 cap
    if (e < n_cs_elem) return 0;
    const Long es = e - n_cs_elem;
    if (es >= stud_rev) return 4;
    const Long w = es % per_sector;
    return (w < Ns) ? 1 : (w < Ns + Nf) ? 2 : 3;
  };
  const Vector<Real>& nds = QuadElemList<Real>::ParamNodes(order);
  const Long nne = (Long)order*order;
  Vector<Real> Xall; full.GetNodeCoord(&Xall, nullptr, nullptr);
  const Long nelem = (Xall.Dim()/3)/nne;
  const char* name[5] = {"cubed-sphere", "shaft", "fillet", "collar (flat, on sphere)", "cap"};
  Real tu[5] = {0}, tv[5] = {0}, nd[5] = {0}; Long cnt[5] = {0};
  Vector<Real> Xe, Xne, Xa, dU, dV;
  for (Long e = 0; e < nelem; e++) {
    const int cls = region(e);
    full.GetGeom(&Xe, &Xne, &Xa, &dU, &dV, nds, nds, e);
    for (Long p = 0; p < nne; p++) {
      const Real x = Xe[p*3], y = Xe[p*3+1], z = Xe[p*3+2], r = std::sqrt(x*x+y*y+z*z);
      const Real rhx = x/r, rhy = y/r, rhz = z/r;
      const Real ux = dU[p*3], uy = dU[p*3+1], uz = dU[p*3+2], un = std::sqrt(ux*ux+uy*uy+uz*uz);
      const Real vx = dV[p*3], vy = dV[p*3+1], vz = dV[p*3+2], vn = std::sqrt(vx*vx+vy*vy+vz*vz);
      const Real turad = std::fabs(ux*rhx+uy*rhy+uz*rhz)/un;
      const Real tvrad = std::fabs(vx*rhx+vy*rhy+vz*rhz)/vn;
      const Real nx = Xne[p*3], ny = Xne[p*3+1], nz = Xne[p*3+2];
      const Real ndev = std::fabs((Real)1 - std::fabs(nx*rhx+ny*rhy+nz*rhz));
      tu[cls] = std::max(tu[cls], turad); tv[cls] = std::max(tv[cls], tvrad); nd[cls] = std::max(nd[cls], ndev); cnt[cls]++;
    }
  }
  std::cout << "\n=== diagnose_tangent_alignment (order=" << order << " Naz=" << Naz << " r_fil=" << r_fil
            << "  Ns=" << Ns << " Nf=" << Nf << " Nc=" << Nc << " cap(butterfly)=" << cap_pan << ") ===\n";
  std::cout << "  per region: max radial component of tangents (=tilt off the sphere tangent plane); collar+cubed-sphere SHOULD be ~0 (on sphere)\n";
  for (int c = 0; c < 5; c++)
    std::cout << "  " << name[c] << " (" << cnt[c] << " nodes): max|du.rhat|/|du|=" << tu[c]
              << "  max|dv.rhat|/|dv|=" << tv[c] << "  max(1-|n.rhat|)=" << nd[c] << "\n";
}

// Squareness / cell-quality metric (geometry-only, NO BIE) -- the sphere-independent anti-overfit
// signal. Per panel node, from the surface Jacobian J=[dX/du, dX/dv] (u,v in [0,1] per panel, so J
// encodes the physical cell shape): reports max cell ASPECT sigma1/sigma2 (1=square), max
// NON-ORTHOGONALITY |du.dv|/(|du||dv|) (0=orthogonal), and min SCALED-JACOBIAN |du x dv|/(|du||dv|)
// (1=ideal, <=0=fold). Must improve as the collar map is squared, WITHOUT evaluating the BIE.
template <class Real> void diagnose_mesh_quality(const Comm& comm, Integer order, Long PatchPerFace, Real R, Integer Naz, Real r_fil, Real grade_exp, Real R_shaft) {
  (void)comm;
  QuadElemList<Real> full = BuildCiliumStuddedSphere<Real>(order, PatchPerFace, R, Naz, r_fil, -1, -1, -1, -1, grade_exp, R_shaft);
  const Long n_cs_elem = 6*PatchPerFace*PatchPerFace - 1;
  const Real pi = const_pi<Real>(), H_shaft = 0.05, R_foot = R_shaft + r_fil, S = R/(Real)PatchPerFace, az = 2*pi*R_shaft/Naz;
  const Integer Ns = std::max<Integer>(1, (Integer)std::llround((double)((H_shaft - r_fil)/az)));
  const Integer Nf = std::max<Integer>(1, (Integer)std::llround((double)((pi/2*r_fil)/az)));
  const Integer Nc = collar_Nc<Real>(R_foot, S, Naz);
  const Long per_sector = Ns + Nf + Nc, stud_rev = Naz*per_sector;
  auto region = [&](Long e) -> int { if (e < n_cs_elem) return 0; const Long es = e - n_cs_elem; if (es >= stud_rev) return 4; const Long w = es % per_sector; return (w < Ns) ? 1 : (w < Ns + Nf) ? 2 : 3; };
  const Vector<Real>& nds = QuadElemList<Real>::ParamNodes(order);
  const Long nne = (Long)order*order;
  Vector<Real> Xall; full.GetNodeCoord(&Xall, nullptr, nullptr);
  const Long nelem = (Xall.Dim()/3)/nne;
  const char* name[5] = {"cubed-sphere", "shaft", "fillet", "collar (flat, on sphere)", "cap"};
  Real ar[5] = {0}, no[5] = {0}, sj[5]; for (int c = 0; c < 5; c++) sj[c] = 1; Long cnt[5] = {0};
  Vector<Real> Xe, Xa, dU, dV;
  for (Long e = 0; e < nelem; e++) {
    const int cls = region(e);
    full.GetGeom(&Xe, nullptr, &Xa, &dU, &dV, nds, nds, e);
    for (Long p = 0; p < nne; p++) {
      const Real ux = dU[p*3], uy = dU[p*3+1], uz = dU[p*3+2], vx = dV[p*3], vy = dV[p*3+1], vz = dV[p*3+2];
      const Real E = ux*ux+uy*uy+uz*uz, G = vx*vx+vy*vy+vz*vz, F = ux*vx+uy*vy+uz*vz; // 1st fundamental form JtJ
      const Real un = std::sqrt(E), vn = std::sqrt(G);
      const Real tr = E + G, dsc = std::sqrt(std::max<Real>(0, tr*tr/4 - (E*G - F*F))); // eig of [[E,F],[F,G]]
      const Real l1 = tr/2 + dsc, l2 = std::max<Real>(0, tr/2 - dsc);
      const Real aspect = (l2 > 0) ? std::sqrt(l1/l2) : (Real)1e30;
      const Real NO = std::fabs(F)/(un*vn);                              // |cos(angle(u,v))|, 0 ideal
      const Real SJ = std::sqrt(std::max<Real>(0, E*G - F*F))/(un*vn);   // sin(angle), 1 ideal, <=0 fold
      ar[cls] = std::max(ar[cls], aspect); no[cls] = std::max(no[cls], NO); sj[cls] = std::min(sj[cls], SJ); cnt[cls]++;
    }
  }
  std::cout << "\n=== diagnose_mesh_quality (order=" << order << " Naz=" << Naz << " r_fil=" << r_fil << "  Nc=" << Nc << ") ===\n";
  std::cout << "  per region: max cell ASPECT (1=square), max NON-ORTH (0=ideal), min SCALED-JAC (1=ideal, <=0=fold)\n";
  for (int c = 0; c < 5; c++)
    std::cout << "  " << name[c] << " (" << cnt[c] << " nodes): max aspect=" << ar[c] << "  max non-orth=" << no[c] << "  min scaledJ=" << sj[c] << "\n";
}
// OFF-AXIS collar mesh-quality gate for the PoU+Winslow smoother (the sphere-independent anti-overfit
// signal). Builds the all-collar sphere (every patch = collar+disk via off-axis PatchMount) and, for the
// most-distorted cube-FACE-CORNER patch, reports the collar cell ASPECT / NON-ORTH / SCALED-JAC per radial
// ring along a CORNER ray vs an EDGE-MIDPOINT ray. Acceptance: min scaledJ > 0 everywhere (no fold), and
// aspect graded (corner ray leading). Compare QJ_COLLAR_ENABLE=0 (legacy seed) vs =1 (smoothed).
template <class Real> void diagnose_mesh_quality_off(const Comm& comm, Integer order, Long PatchPerFace, Real R, Integer Naz, Real r_fil, Real grade_exp, Real R_shaft) {
  QuadElemList<Real> full = BuildAllCollarFillSphere<Real>(order, PatchPerFace, R, Naz, r_fil, grade_exp, R_shaft, -1, -1, (Real)0.40, /*with_finger=*/false, /*circularize=*/false, comm);
  const Real R_foot = R_shaft + r_fil, S = R/(Real)PatchPerFace;
  const Integer Nc = collar_Nc<Real>(R_foot, S, Naz);
  const Integer Ndisk = std::max<Integer>(1, (Integer)std::llround((double)(R_foot/(2*const_pi<Real>()*R_shaft/Naz))));
  const Long per_patch = (Long)Naz*Nc + 5*Ndisk*Ndisk;              // collar (Naz*Nc, sector-major) then disk
  // target = a cube-FACE-CORNER patch of face 0 (iu=iv=0) -> maximal gnomonic distortion.
  const Long p_target = 0*PatchPerFace*PatchPerFace + 0*PatchPerFace + 0;
  const Long base = p_target*per_patch;                            // first element of this patch's collar
  const Vector<Real>& nds = QuadElemList<Real>::ParamNodes(order);
  const Long nne = (Long)order*order;
  // per-ring aspect/nonorth/scaledJ, split corner-sector vs edge-sector.
  auto ring_stats = [&](Integer m, Vector<Real>& asp, Vector<Real>& sjmin) {
    asp.ReInit(Nc); sjmin.ReInit(Nc); Vector<Real> Xe, Xa, dU, dV;
    for (Integer ir = 0; ir < Nc; ir++) {
      const Long e = base + (Long)m*Nc + ir; Real amax = 0, smin = 1;
      full.GetGeom(&Xe, nullptr, &Xa, &dU, &dV, nds, nds, e);
      for (Long p = 0; p < nne; p++) {
        const Real ux=dU[p*3],uy=dU[p*3+1],uz=dU[p*3+2], vx=dV[p*3],vy=dV[p*3+1],vz=dV[p*3+2];
        const Real E=ux*ux+uy*uy+uz*uz, G=vx*vx+vy*vy+vz*vz, F=ux*vx+uy*vy+uz*vz;
        const Real un=std::sqrt(E), vn=std::sqrt(G), tr=E+G, dsc=std::sqrt(std::max<Real>(0,tr*tr/4-(E*G-F*F)));
        const Real l1=tr/2+dsc, l2=std::max<Real>(0,tr/2-dsc);
        amax = std::max(amax, (l2>0)?std::sqrt(l1/l2):(Real)1e30);
        smin = std::min(smin, std::sqrt(std::max<Real>(0,E*G-F*F))/(un*vn));
      }
      asp[ir]=amax; sjmin[ir]=smin;
    }
  };
  // sector 0 starts AT a corner (pi/4 offset) -> its far edge is the corner ray; sector Naz/8 centre ~ edge-mid.
  Vector<Real> aC, sC, aE, sE; ring_stats(0, aC, sC); ring_stats(std::max<Integer>(1,Naz/8), aE, sE);
  if (comm.Rank()) return;
  std::cout << "\n=== diagnose_mesh_quality_off (all-collar, PPF=" << PatchPerFace << " order=" << order
            << " Naz=" << Naz << " Nc=" << Nc << ", face-0 corner patch)  enable=" << collar_pou_cfg().enable << " ===\n";
  std::cout << "  ring : corner-ray[aspect,minSJ]   edge-ray[aspect,minSJ]  (aspect 1=square, minSJ>0=no fold)\n";
  for (Integer ir = 0; ir < Nc; ir++)
    std::cout << "  " << ir << "    : [" << std::setprecision(4) << aC[ir] << ", " << sC[ir] << "]   ["
              << aE[ir] << ", " << sE[ir] << "]\n";
  report_area<Real>(full, comm);
}
} // anonymous namespace

int main(int argc, char** argv) {
  Comm::MPI_Init(&argc, &argv);
  using Real = double;
  {
    const Comm comm = Comm::World();
    const std::string mode = (argc > 1) ? std::string(argv[1]) : "";
    const Integer ord    = (argc > 2) ? (Integer)atoi(argv[2]) : 16;
    const Integer Naz    = (argc > 3) ? (Integer)atoi(argv[3]) : 8;
    const Real r_fil     = (argc > 4) ? (Real)atof(argv[4]) : (Real)0.005;
    const Real R_shaft   = (argc > 5) ? (Real)atof(argv[5]) : (Real)0.015;
    const Long PPF       = (argc > 6) ? (Long)atoi(argv[6]) : 3;   // off-axis meshqoff only
    if (mode == "tangent")
      diagnose_tangent_alignment<Real>(comm, ord, 7, 1.0, Naz, r_fil, 1, R_shaft);
    else if (mode == "meshq")
      diagnose_mesh_quality<Real>(comm, ord, 7, 1.0, Naz, r_fil, 1, R_shaft);
    else if (mode == "meshqoff")
      diagnose_mesh_quality_off<Real>(comm, ord, PPF, 1.0, Naz, r_fil, 1, R_shaft);
    else {
      QuadElemList<Real> el = BuildCiliumStuddedSphere<Real>(ord, 7, 1.0, Naz, r_fil, -1, -1, -1, -1, 1, R_shaft);
      report_area<Real>(el);
      el.WriteVTK("vis/stud-sphere-geom", Vector<Real>(), comm);
    }
  }
  Comm::MPI_Finalize();
  return 0;
}
