/**
 * HYBRID cilium-finger BIE driver.
 *
 * Builds the studded-sphere QuadElemList BASE (cubed sphere with one pole patch = collar + fillet +
 * butterfly cap, NO shaft) and a CSBQ SlenderElemList straight-cylinder SHAFT, adds BOTH to one
 * BoundaryIntegralOp, and runs the identity checks (DL constant-density identity -> -1/2; interior
 * Green's identity) for Laplace and Stokes. The stud_sphere counterpart of ybifurc-hybrid-bie.cpp.
 *
 * Global node/density ordering is the operator's NAME-SORTED list concatenation: the lists are added
 * as "0_base" then "1_shaft", so every global array here is [base nodes ; shaft nodes] in that order.
 *
 * A mode keyword is REQUIRED: "centerfinger" (fingers on the 6 face-centre patches) or "flagella"
 * (every patch a twirling finger). The single-finger default and the "allfinger" mode were removed.
 *
 *   make bin/stud_sphere-hybrid-bie
 *   OMP_NUM_THREADS=8 ./bin/stud_sphere-hybrid-bie <centerfinger|flagella> \
 *       [tol Nbeta max_depth R_shaft Nc Naz order r_fil n_axial fourier cheb PatchPerFace flip]
 */

#include <csbq.hpp>                                       // CSBQ SlenderElemList
#include <quad_junctions/fmm_kernels.hpp>                 // SetPVFMMKer (PVFMM-safe M2M/M2L/L2L kernels)
#include <quad_junctions/stud_sphere_hybrid_geom.hpp>     // hybrid builders (base sphere-with-hole+cap, slender shaft)
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <string>

using namespace sctl;
using namespace quad_junctions;

namespace {

// Concatenate the two lists' node coords/normals in NAME-SORTED order (base then shaft).
template <class Real> void combined_nodes(const QuadElemList<Real>& base, const SlenderElemList<Real>& shaft,
                                          Vector<Real>& X, Vector<Real>& Xn, Long& Nb, Long& Ns) {
  Vector<Real> Xb, Xnb, Xs, Xns;
  base.GetNodeCoord(&Xb, &Xnb, nullptr);
  shaft.GetNodeCoord(&Xs, &Xns, nullptr);
  Nb = Xb.Dim()/3; Ns = Xs.Dim()/3;
  X.ReInit(0); Xn.ReInit(0);
  for (auto v : Xb)  X.PushBack(v);
  for (auto v : Xs)  X.PushBack(v);
  for (auto v : Xnb) Xn.PushBack(v);
  for (auto v : Xns) Xn.PushBack(v);
}

// ---- Watertightness / orientation check: int n dA = 0 for any closed, consistently-oriented surface.
//      A nonzero COMBINED flux localizes a gap or a flipped-normal region (here: wrong shaft normal). ----
template <class Real> void divergence_check(const QuadElemList<Real>& base, const SlenderElemList<Real>& shaft, const Real tol, const Comm& comm) {
  auto flux_area = [&](const auto& lst, Real f[3]) -> Real {
    Vector<Real> X, Xn, wts, dist; Vector<Long> cnt;
    lst.GetFarFieldNodes(X, Xn, wts, dist, cnt, tol);
    const Long N = wts.Dim(); Real A = 0; f[0]=f[1]=f[2]=0;
    for (Long i = 0; i < N; i++) { A += wts[i]; for (int k=0;k<3;k++) f[k] += wts[i]*Xn[i*3+k]; }
    return A;
  };
  Real fb[3], fs[3];
  Real Ab = flux_area(base, fb), As = flux_area(shaft, fs);
  Ab = GlobalReduce((double)Ab, comm, CommOp::SUM);
  As = GlobalReduce((double)As, comm, CommOp::SUM);
  for (int k = 0; k < 3; k++) { fb[k] = GlobalReduce((double)fb[k], comm, CommOp::SUM); fs[k] = GlobalReduce((double)fs[k], comm, CommOp::SUM); }
  const Real fc[3] = {fb[0]+fs[0], fb[1]+fs[1], fb[2]+fs[2]};
  auto nrm = [](const Real f[3]){ return std::sqrt(f[0]*f[0]+f[1]*f[1]+f[2]*f[2]); };
  if (!comm.Rank())
    std::cout << std::setprecision(4)
              << "  [watertight] area: base=" << Ab << " shaft=" << As << " total=" << (Ab+As) << "\n"
              << "  [watertight] |int n dA|: base=" << nrm(fb) << " shaft=" << nrm(fs)
              << "  COMBINED=" << nrm(fc) << "  (combined should be ~0 for a closed surface)\n";
}

// ---- DL constant-density identity: |DL[1]| = 1/2 on a closed, consistently-oriented surface.
//      Sign is -1/2 for OUTWARD normals, +1/2 for INWARD; auto-detected so this validates orientation
//      CONSISTENCY across the two lists regardless of the global convention (a node with the wrong sign
//      shows ~1.0 error). ----
template <class Real, class KerDL> void test_DLIdentity(const QuadElemList<Real>& base, const SlenderElemList<Real>& shaft,
                                                        const Comm& comm, const Real tol) {
  BoundaryIntegralOp<Real, KerDL> BIOp((KerDL()), false, comm);
  SetPVFMMKer(BIOp);
  BIOp.SetAccuracy(tol);
  BIOp.AddElemList(base,  "0_base");
  BIOp.AddElemList(shaft, "1_shaft");
  const Long KDIM0 = KerDL::SrcDim();
  Vector<Real> X, Xn; Long Nb, Ns; combined_nodes(base, shaft, X, Xn, Nb, Ns);
  const Long Nnode = Nb + Ns;
  Vector<Real> q(Nnode*KDIM0), U;
  for (Long i = 0; i < Nnode; i++) for (Long k = 0; k < KDIM0; k++) q[i*KDIM0+k] = k+1;
  BIOp.ComputePotential(U, q);
  SCTL_ASSERT_MSG(U.Dim() == Nnode*KerDL::TrgDim(), "hybrid node-count/ordering mismatch");
  // auto-detect the DL identity sign from the mean of U/q over component 0 (outward -> -1/2, inward -> +1/2)
  double smean = 0; for (Long i = 0; i < Nnode; i++) smean += (double)(U[i*KDIM0]/q[i*KDIM0]);
  smean = GlobalReduce(smean, comm, CommOp::SUM);
  const Real c_exp = (smean < 0) ? (Real)-0.5 : (Real)0.5;
  Real emax = 0; Long argmax = 0; Vector<Real> ce(KDIM0); ce = 0; Vector<Real> err(Nnode);
  for (Long i = 0; i < Nnode; i++) { Real ei = 0;
    for (Long k = 0; k < KDIM0; k++) { const Real e = std::fabs(U[i*KDIM0+k]/q[i*KDIM0+k] - c_exp); ce[k]=std::max(ce[k],e); ei=std::max(ei,e); if (e>emax){emax=e;argmax=i;} }
    err[i] = ei; }
  Vector<double> ced(KDIM0); for (Long k = 0; k < KDIM0; k++) ced[k] = (double)ce[k];
  GlobalReduce(ced, comm, CommOp::MAX);
  Real avg = 0; for (Long k = 0; k < KDIM0; k++) avg += ced[k]/(double)0.5; avg /= KDIM0;
  const double loc_xyz[3] = { (double)(Nnode?X[argmax*3]:0), (double)(Nnode?X[argmax*3+1]:0), (double)(Nnode?X[argmax*3+2]:0) };
  double gloc[3]; const double gemax = GlobalMaxLoc((double)emax, loc_xyz, comm, gloc);
  // base (i<Nb) vs slender shaft (i>=Nb) split -- serial only (local index maps to region only in serial).
  Real mb = 0, ms = 0;
  if (comm.Size() == 1) { for (Long i=0;i<Nnode;i++){ if(i<Nb) mb=std::max(mb,err[i]); else ms=std::max(ms,err[i]); } }
  // STUD_DUMP_ERR (serial): per-node error field for plotting -> hybrid_DL_err.csv
  if (comm.Size() == 1 && std::getenv("STUD_DUMP_ERR")) {
    FILE* fp = std::fopen("hybrid_DL_err.csv", "w");
    std::fprintf(fp, "x,y,z,r,region,err\n");   // region: 0=base(quad foot/sphere) 1=slender-shaft
    for (Long i = 0; i < Nnode; i++) { const double x=(double)X[3*i], y=(double)X[3*i+1], z=(double)X[3*i+2];
      std::fprintf(fp, "%.10g,%.10g,%.10g,%.10g,%d,%.10g\n", x, y, z, std::sqrt(x*x+y*y+z*z), (i<Nb?0:1), (double)err[i]); }
    std::fclose(fp);
    std::cout << "    [dump] wrote hybrid_DL_err.csv (" << Nnode << " nodes)\n";
  }
  if (!comm.Rank()) {
    std::cout << std::setprecision(6) << "  DL const-density identity (expect " << c_exp << "): max rel err = " << avg << "  (max abs = " << gemax << ")\n";
    std::cout << "    [loc] argmax node at (" << std::setprecision(3) << gloc[0] << "," << gloc[1] << "," << gloc[2]
              << ")  r=" << std::sqrt(gloc[0]*gloc[0]+gloc[1]*gloc[1]+gloc[2]*gloc[2]) << "\n";
    if (comm.Size() == 1) std::cout << std::setprecision(6) << "    [region max] base=" << mb << " slender-shaft=" << ms << "\n";
  }
}

// ---- Interior Green's identity (source X0 inside) over the combined surface ----
template <class Real, class KerSL, class KerDL, class KerGrad>
void test_greens_identity(const QuadElemList<Real>& base, const SlenderElemList<Real>& shaft,
                          const Comm& comm, const Real tol, const Vector<Real> X0) {
  static constexpr Integer CDIM = 3;
  KerSL ksl; KerDL kdl; KerGrad kgr;
  BoundaryIntegralOp<Real,KerSL> BIOpSL(ksl, false, comm); BoundaryIntegralOp<Real,KerDL> BIOpDL(kdl, false, comm);
  SetPVFMMKer(BIOpSL); SetPVFMMKer(BIOpDL);
  BIOpSL.SetAccuracy(tol); BIOpDL.SetAccuracy(tol);
  BIOpSL.AddElemList(base, "0_base"); BIOpSL.AddElemList(shaft, "1_shaft");
  BIOpDL.AddElemList(base, "0_base"); BIOpDL.AddElemList(shaft, "1_shaft");

  Vector<Real> X, Xn; Long Nb, Ns; combined_nodes(base, shaft, X, Xn, Nb, Ns);
  const Long N = Nb + Ns;
  Vector<Real> Fs, Fd, Uref, Us, Ud;
  { Vector<Real> Xn0{0,0,0}, F0(KerSL::SrcDim()), dU; for (auto& x : F0) x = drand48()-0.5;
    ksl.Eval(Uref, X, X0, Xn0, F0); kgr.Eval(dU, X, X0, Xn0, F0); Fd = Uref;
    constexpr Integer KDIM0 = KerSL::SrcDim(); Fs.ReInit(N*KDIM0);
    for (Long i = 0; i < N; i++) for (Integer j = 0; j < KDIM0; j++) { Real d=0; for (Long k=0;k<CDIM;k++) d += dU[(i*KDIM0+j)*CDIM+k]*Xn[i*CDIM+k]; Fs[i*KDIM0+j]=d; } }
  BIOpSL.ComputePotential(Us, Fs); BIOpDL.ComputePotential(Ud, Fd);
  // Inward vs outward normals flip both the SL/DL representation sign and the DL jump sign. Try the four
  // (jump=+-1/2) x (overall sign +-1) combinations and report the smallest relative error + which won.
  const Real jumps[2] = {(Real)-0.5, (Real)+0.5}; const Real signs[2] = {(Real)1, (Real)-1};
  Real mv = 0; for (auto x : Uref) mv = std::max<Real>(mv, std::fabs(x)); mv = GlobalReduce((double)mv, comm, CommOp::MAX);
  Real best = 1e30; Real bj = 0, bs = 0;
  for (int a = 0; a < 2; a++) for (int b = 0; b < 2; b++) {
    Real me = 0;
    for (Long i = 0; i < (Long)Uref.Dim(); i++) { const Real rep = signs[b]*(Us[i] - (Ud[i] + jumps[a]*Fd[i])); me = std::max<Real>(me, std::fabs(rep - Uref[i])); }
    me = GlobalReduce((double)me, comm, CommOp::MAX);
    if (me/mv < best) { best = me/mv; bj = jumps[a]; bs = signs[b]; }
  }
  if (!comm.Rank()) std::cout << "  Green's identity error = " << best << "  (jump=" << bj << " sign=" << bs << ")\n";
  // STUD_DUMP_ERR (serial): per-node Green's error field (best jump/sign) for plotting -> hybrid_green_err.csv
  if (comm.Size() == 1 && std::getenv("STUD_DUMP_ERR")) {
    const Integer TD = KerSL::TrgDim(); const Long Nn = Uref.Dim()/TD;
    FILE* fp = std::fopen("hybrid_green_err.csv", "w");
    std::fprintf(fp, "x,y,z,r,region,err\n");   // region: 0=base 1=slender-shaft; err = max over kernel components
    for (Long i = 0; i < Nn; i++) { Real e = 0;
      for (Integer td = 0; td < TD; td++) { const Long f = i*TD+td; const Real rep = bs*(Us[f] - (Ud[f] + bj*Fd[f])); e = std::max<Real>(e, std::fabs(rep - Uref[f])); }
      const double x=(double)X[3*i], y=(double)X[3*i+1], z=(double)X[3*i+2];
      std::fprintf(fp, "%.10g,%.10g,%.10g,%.10g,%d,%.10g\n", x, y, z, std::sqrt(x*x+y*y+z*z), (i<Nb?0:1), (double)(e/mv)); }
    std::fclose(fp);
    std::cout << "  [dump] wrote hybrid_green_err.csv (" << Nn << " nodes)\n";
  }
}

// ---- common tail: dump the mesh, set the quad scheme, and run watertightness + DL/Green identities ----
template <class Real> void run_hybrid(QuadElemList<Real>& base, SlenderElemList<Real>& shaft, const Comm& comm,
                                      const Real tol, Integer cov_q, Integer Nbeta, Integer max_depth, const std::string& tag) {
  base.WriteVTK(tag + "-base", Vector<Real>(), comm);
  shaft.WriteVTK(tag + "-shaft", Vector<Real>(), comm);
  if (!comm.Rank()) std::cout << "  wrote " << tag << "-{base,shaft}.pvtu/.vtu\n";

  base.SetQuadScheme(QuadElemList<Real>::QuadScheme::Hybrid, cov_q, Nbeta, max_depth);
  const Long nbp = GlobalReduce((Long)base.Size(), comm, CommOp::SUM), nsp = GlobalReduce((Long)shaft.Size(), comm, CommOp::SUM);
  if (!comm.Rank()) std::cout << "  base panels=" << nbp << " shaft slender-panels=" << nsp << "\n";
  divergence_check<Real>(base, shaft, tol, comm);

  const Vector<Real> X0{1.3, 1.2, 0.2};   // exterior source for interior Green's identity
  // STUD_STOKES_ONLY (env) skips Laplace; STUD_DL_ONLY skips the Green's-identity tests (which build
  // 3 operators S+D+T -> ~3x the memory of the DL near-setup, the thing that OOMs on few nodes).
  const bool stokes_only  = (std::getenv("STUD_STOKES_ONLY") != nullptr);
  const bool dl_only      = (std::getenv("STUD_DL_ONLY") != nullptr);
  const bool laplace_only = (std::getenv("STUD_LAPLACE_ONLY") != nullptr);   // skip ALL Stokes ops
  // DL-Laplace-only = STUD_DL_ONLY + STUD_LAPLACE_ONLY (Laplace DL runs; Stokes DL + both Green's skipped).
  if (!stokes_only)                   { if (!comm.Rank()) { std::cout << "[Laplace] "; } test_DLIdentity<Real, Laplace3D_DxU>(base, shaft, comm, tol); }
  if (!laplace_only)                  { if (!comm.Rank()) { std::cout << "[Stokes]  "; } test_DLIdentity<Real, Stokes3D_DxU>(base, shaft, comm, tol); }
  if (!dl_only && !stokes_only)       { if (!comm.Rank()) { std::cout << "[Laplace] "; } test_greens_identity<Real, Laplace3D_FxU, Laplace3D_DxU, Laplace3D_FxdU>(base, shaft, comm, tol, X0); }
  if (!dl_only && !laplace_only)      { if (!comm.Rank()) { std::cout << "[Stokes]  "; } test_greens_identity<Real, Stokes3D_FxU, Stokes3D_DxU, Stokes3D_FxT>(base, shaft, comm, tol, X0); }
}

} // anonymous namespace

int main(int argc, char** argv) {
  Comm::MPI_Init(&argc, &argv);
  using Real = double;
  {
    const Comm comm = Comm::World();
    // REQUIRED string mode at argv[1]:
    //   "centerfinger" = hybrid fingers ONLY on the 6 on-axis (face-centre) patches; all other patches plain
    //                    cubed sphere.
    //   "flagella"     = EVERY patch is a TWIRLING (spiral-centerline) finger (flagella_centerline.hpp).
    // The keyword shifts all remaining positional numeric args by one (o=1). The former single-finger default
    // and the "allfinger" (every-patch straight finger) mode were removed (2026-07-31) -- a mode keyword is
    // now mandatory.
    const bool flagella  = (argc > 1 && std::string(argv[1]) == "flagella");
    const bool centerfinger = (argc > 1 && std::string(argv[1]) == "centerfinger");
    SCTL_ASSERT_MSG(flagella || centerfinger,
        "stud_sphere-hybrid: a mode keyword is required -- 'centerfinger' or 'flagella' "
        "(the single-finger default and 'allfinger' modes were removed)");
    const int  o = 1;   // arg offset (a mode keyword is always present)
    // Positional CLI (numeric args follow the mode keyword; see per-arg comments).
    const Real    tol       = (argc > 1+o) ? (Real)atof(argv[1+o])  : (Real)1e-8;
    const Integer Nbeta     = (argc > 2+o) ? (Integer)atoi(argv[2+o])  : 400;
    const Integer max_depth = (argc > 3+o) ? (Integer)atoi(argv[3+o])  : 30;
    const Real    R         = (Real)1;
    const Long    PatchPerFace = (argc > 12+o) ? (Long)atoi(argv[12+o]) : (flagella ? 1 : 3);   // centerfinger default 3
    const Real    S         = R / (Real)PatchPerFace;   // patch half-width (drives the patch-relative shaft)
    // PATCH-RELATIVE cilium sizing: a cilium is self-similar to its patch. By default the shaft radius
    // R_shaft = frac*S (frac from env QJ_RSHAFT_FRAC, default 0.25 -- the thin cilium), r_fil = 0.1*R_shaft,
    // and the sphere shaft depth H_shaft = k*R_shaft (k from env QJ_HSHAFT_K, default 3). Passing a POSITIVE
    // explicit value on the CLI overrides any of them (absolute back-compat). See cilium_scale_from_patch().
    const Real    rshaft_frac = std::getenv("QJ_RSHAFT_FRAC") ? (Real)atof(std::getenv("QJ_RSHAFT_FRAC")) : (Real)0.25;
    const Real    hshaft_k    = std::getenv("QJ_HSHAFT_K")    ? (Real)atof(std::getenv("QJ_HSHAFT_K"))    : (Real)3;
    const CiliumScale<Real> csc = cilium_scale_from_patch<Real>(S, rshaft_frac, (Real)0.1, hshaft_k);
    const Real    R_shaft_arg = (argc > 4+o)  ? (Real)atof(argv[4+o])  : (Real)-1;   // >0 => absolute override
    const Real    r_fil_arg   = (argc > 8+o)  ? (Real)atof(argv[8+o])  : (Real)-1;
    const Real    H_shaft_arg = (argc > 14+o) ? (Real)atof(argv[14+o]) : (Real)-1;
    const Real    R_shaft   = (R_shaft_arg > 0) ? R_shaft_arg : csc.R_shaft;
    const Real    r_fil     = (r_fil_arg   > 0) ? r_fil_arg   : (Real)0.1 * R_shaft;
    const Real    H_shaft   = (H_shaft_arg  > 0) ? H_shaft_arg : hshaft_k * R_shaft;   // shaft depth (length ~ H_shaft - r_fil)
    const Integer Nc        = (argc > 5+o) ? (Integer)atoi(argv[5+o])  : -1;   // -1 => collar_Nc auto (~2 rings at frac 0.5)
    const Integer Naz       = (argc > 6+o) ? (Integer)atoi(argv[6+o])  : 8;
    const Integer order     = (argc > 7+o) ? (Integer)atoi(argv[7+o])  : 16;
    const Integer n_axial_in= (argc > 9+o) ? (Integer)atoi(argv[9+o])  : -1;
    const Long    fourier   = (argc > 10+o) ? (Long)atoi(argv[10+o]) : 12;
    const Long    cheb      = (argc > 11+o) ? (Long)atoi(argv[11+o]) : 10;
    // Base normal orientation: invert=1 (default) flips the whole base to INWARD normals so the surface
    // matches CSBQ's native radially-outward-from-axis slender normal (consistent orientation).
    const bool    invert    = (argc > 13+o) ? (atoi(argv[13+o]) != 0) : true;
    // Mesh-geometry knobs for the quad foot (collar+fillet+cap); defaults reproduce prior behaviour.
    const Real    grade_exp = (argc > 15+o) ? (Real)atof(argv[15+o]) : (Real)1;      // collar radial grading
    const Real    core_frac = (argc > 16+o) ? (Real)atof(argv[16+o]) : (Real)0.40;   // butterfly-cap core half-size
    const Integer cap_Naz   = (argc > 17+o) ? (Integer)atoi(argv[17+o]) : -1;        // cap azimuthal, DECOUPLED from foot Naz (-1 => = Naz)
    const Real    axial_grade = (argc > 18+o) ? (Real)atof(argv[18+o]) : (Real)0;    // slender axial clustering toward both seams (0=uniform..1=cosine)
    const Integer Nf_in     = (argc > 19+o) ? (Integer)atoi(argv[19+o]) : -1;        // fillet (rounded lip) panels; -1=auto (~1 for small r_fil)
    // (argv 20+o and 21+o are now unused dead slots -- they were the POU transition-tube depth / panel
    //  count consumed only by the removed "allfinger" mode; centerfinger/flagella don't take them.)
    // Near-quadrature coverage order for the QuadElemList base's Hybrid scheme. Must be in {6,10}
    // (the fork's RectPolar-covered tables; ybifurc-hybrid uses 6, this driver historically pinned 10).
    // Exposed to A/B the base<->slender cross-list near quadrature against ybifurc's 1e-10 configuration.
    const Integer cov_q     = (argc > 22+o) ? (Integer)atoi(argv[22+o]) : 10;
    // flagella-mode racetrack knobs: straight normal-lead + smootherstep-POU corner panels (M8: the
    // Green's-identity accuracy limiter) + radius taper panels.
    const Integer lead_panels   = (argc > 23+o) ? (Integer)atoi(argv[23+o]) : 3;
    const Integer corner_panels = (argc > 24+o) ? (Integer)atoi(argv[24+o]) : 8;
    const Integer Ntaper        = (argc > 25+o) ? (Integer)atoi(argv[25+o]) : 5;

    if (flagella) {
      FlagellaCfg<Real> cfg;
      cfg.R = R; cfg.R_shaft = R_shaft; cfg.r_fil = r_fil;
      cfg.lead_panels = lead_panels; cfg.corner_panels = corner_panels; cfg.Ntaper = Ntaper;
      // axial panels per fiber: CLI override (arg 9) or auto from the spiral arc length.
      Real u0[3]; { Real px,py,pz; FacePoint<Real>(px,py,pz,0,(Real)0,(Real)0,R); const Real pr=std::sqrt(px*px+py*py+pz*pz); u0[0]=px/pr;u0[1]=py/pr;u0[2]=pz/pr; }
      const Integer n_axial = (n_axial_in >= 1) ? n_axial_in : flagella_n_axial<Real>(cfg, u0, cfg.r_py, fourier);
      cfg.n_axial = n_axial;
      if (!comm.Rank())
        std::cout << "\n=== TWIRLING CILIA (flagella spiral) hybrid sphere (EVERY patch = QuadElemList foot [collar+fillet] + spiral-tip cap  +  one SPIRAL SlenderElemList shaft) ===\n"
                  << "  order=" << order << " Naz=" << Naz << " R_shaft=" << R_shaft << " r_py=" << cfg.r_py << " r_fil=" << r_fil
                  << " tol=" << tol << " Nbeta=" << Nbeta << " max_depth=" << max_depth
                  << " fourier=" << fourier << " cheb=" << cheb << " PatchPerFace=" << PatchPerFace << " (fingers=" << (6*PatchPerFace*PatchPerFace) << ")"
                  << " lead_panels=" << lead_panels << " corner_panels=" << corner_panels << " Ntaper=" << Ntaper
                  << " n_axial=" << n_axial << " Nc=" << Nc << " core_frac=" << core_frac << " cap_Naz=" << cap_Naz << " cov_q=" << cov_q << "\n";

      // ---- centerline-only dump (fast: skips the mesh + BIE) for the PNG shape check ----
      if (std::getenv("STUD_DUMP_CENTERLINE")) {
        if (!comm.Rank()) {
          FILE* fp = std::fopen("flagella_centerline.csv", "w");
          std::fprintf(fp, "finger,x,y,z,r\n");
          Long p = 0;
          for (Integer face = 0; face < 6; face++)
            for (Long iu = 0; iu < PatchPerFace; iu++)
              for (Long iv = 0; iv < PatchPerFace; iv++, p++) {
                const Real a_c = 2*(iu + (Real)0.5)/PatchPerFace - 1, b_c = 2*(iv + (Real)0.5)/PatchPerFace - 1;
                Real px,py,pz; FacePoint<Real>(px,py,pz,face,a_c,b_c,R); const Real pr=std::sqrt(px*px+py*py+pz*pz);
                const Real u[3] = {px/pr,py/pr,pz/pr};
                FlagellumCurve<Real> fc(cfg, u);
                const Integer ns = 240;
                for (Integer s = 0; s <= ns; s++) { const Real f = (Real)s/ns; const Vec3<Real> P = fc.point(f);
                  std::fprintf(fp, "%ld,%.10g,%.10g,%.10g,%.10g\n", (long)p, (double)P[0], (double)P[1], (double)P[2], (double)fc.radius(f)); }
              }
          std::fclose(fp);
          std::cout << "  [dump] wrote flagella_centerline.csv (" << (6*PatchPerFace*PatchPerFace) << " fingers) -- run python/plot_flagella_centerlines.py\n";
        }
        Comm::MPI_Finalize();
        return 0;
      }

      std::vector<FlagellumCurve<Real>> curves; Vector<Real> axis;
      QuadElemList<Real> base = BuildAllFingerFlagellaSphereBase<Real>(order, PatchPerFace, R, Naz, cfg, curves, axis, grade_exp, comm, invert, core_frac, Nc, cap_Naz, Nf_in);
      Long pci = -1, pcj = -1; const Real gap = flagella_min_clearance<Real>(curves, pci, pcj);
      if (!comm.Rank()) std::cout << std::setprecision(6) << "  [overlap] min finger surface clearance = " << gap
                                  << " (>0 => no overlap)  closest pair (" << pci << "," << pcj << ")\n";
      SCTL_ASSERT_MSG(gap > 0, "flagella fingers overlap -- reduce R_shaft/L or PatchPerFace");
      SlenderElemList<Real> shaft = BuildAllFingerFlagellaShaftsSlender<Real>(curves, n_axial, cheb, fourier, comm);
      if (!comm.Rank()) std::cout << std::setprecision(6) << "  shafts: spiral fibers=" << curves.size() << " n_axial/finger=" << n_axial << "\n";
      const std::string tag = "vis/stud_sphere-flagella-ppf" + std::to_string((long)PatchPerFace) + "-ord" + std::to_string((long)order);
      run_hybrid<Real>(base, shaft, comm, tol, cov_q, Nbeta, max_depth, tag);
      Comm::MPI_Finalize();
      return 0;
    }

    if (centerfinger) {
      if (!comm.Rank())
        std::cout << "\n=== HYBRID centre-finger sphere (fingers ONLY on the 6 on-axis face-centre patches; other patches PLAIN cubed sphere) ===\n"
                  << "  order=" << order << " Naz=" << Naz << " R_shaft=" << R_shaft << " r_fil=" << r_fil << " H_shaft=" << H_shaft
                  << " tol=" << tol << " Nbeta=" << Nbeta << " max_depth=" << max_depth
                  << " fourier=" << fourier << " cheb=" << cheb << " PatchPerFace=" << PatchPerFace << " (fingers=6)"
                  << " Nc=" << Nc << " grade_exp=" << grade_exp << " core_frac=" << core_frac
                  << " cap_Naz=" << cap_Naz << " axial_grade=" << axial_grade << " Nf=" << Nf_in << " cov_q=" << cov_q << "\n";
      Vector<Real> axis, a_bot, a_top, rho_bot, rho_top;
      QuadElemList<Real> base = BuildCenterFingerSphereBase<Real>(order, PatchPerFace, R, Naz, r_fil, axis, a_bot, a_top, rho_bot, rho_top, grade_exp, R_shaft, H_shaft, comm, invert, core_frac, Nc, cap_Naz, Nf_in);
      Long pi = -1, pj = -1; const Real gap = finger_min_clearance<Real>(axis, a_bot, rho_top, pi, pj);
      if (!comm.Rank()) std::cout << std::setprecision(6) << "  [overlap] min finger clearance = " << gap
                                  << " (>0 => no overlap)  closest pair (" << pi << "," << pj << ")\n";
      SCTL_ASSERT_MSG(gap > 0, "fingers overlap -- reduce PatchPerFace or R_shaft/H_shaft");
      SCTL_ASSERT_MSG(a_bot[0] > 0, "shaft too deep (a_bot<=0, crosses origin) -- reduce H_shaft/QJ_HSHAFT_K or PatchPerFace");
      Real rho_rep = 0; for (Long i = 0; i < rho_top.Dim(); i++) rho_rep += rho_top[i]; rho_rep /= std::max<Long>(1, rho_top.Dim());
      const Integer n_axial = (n_axial_in >= 1) ? n_axial_in : cilium_shaft_n_axial<Real>(rho_rep, a_bot[0], a_top[0], fourier);
      SlenderElemList<Real> shaft = BuildAllFingerShaftsSlender<Real>(axis, a_bot, a_top, rho_bot, rho_top, n_axial, cheb, fourier, comm, axial_grade);
      if (!comm.Rank()) std::cout << std::setprecision(6) << "  shafts: rho~" << rho_rep << " axial~[" << a_bot[0] << "," << a_top[0] << "] n_axial/finger=" << n_axial << "\n";
      const std::string tag = "vis/stud_sphere-centerfinger-ppf" + std::to_string((long)PatchPerFace) + "-ord" + std::to_string((long)order);
      run_hybrid<Real>(base, shaft, comm, tol, cov_q, Nbeta, max_depth, tag);
      Comm::MPI_Finalize();
      return 0;
    }

    // Unreachable: the mode-keyword assert above guarantees flagella || centerfinger, each of which returns.
    SCTL_ASSERT_MSG(false, "stud_sphere-hybrid: no mode ran (expected centerfinger or flagella)");
  }
  Comm::MPI_Finalize();
  return 0;
}
