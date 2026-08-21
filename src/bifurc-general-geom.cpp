/**
 * GEOMETRY-ONLY driver for the generalized bifurcation kernel (gen_junction_geom.hpp). NO BoundaryIntegralOp
 * -> compiles in a few minutes (vs ~20 for the BIE driver), for fast geometry iteration: reports surface
 * area, MIN quad weight (negative => folded/inverted panel), watertight closure |int n dA|, and volume, per
 * arm-region and combined, and writes the mesh VTU. Same spec grammar / CLI prefix as bifurc-general-bie.
 *
 *   make bin/bifurc-general-geom
 *   ./bin/bifurc-general-geom [spec] [level] [order] [nref] [eta_join] [Ns_trans] [s_cap] [Ncap] [alpha_deg] [sigma]
 */
#include <csbq.hpp>
#include <quad_junctions/gen_junction_geom.hpp>
#include <quad_junctions/gen_geom_io.hpp>           // exact reloadable geometry-bundle export
#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

using namespace sctl;
using namespace quad_junctions;

namespace {
using Real = double;

std::vector<Vec3<Real>> parse_spec(const std::string& spec) {
  auto from_gaps = [](const std::vector<double>& gaps) {
    std::vector<Vec3<Real>> d; double a = 0, sum = 0; for (double g : gaps) sum += g;
    SCTL_ASSERT_MSG(std::fabs(sum-360.0) < 1e-6, "gaps must sum to 360.");
    for (double g : gaps) { const double th = a*M_PI/180; d.push_back(Vec3<Real>{(Real)std::cos(th), (Real)std::sin(th), 0}); a += g; }
    return d;
  };
  auto split = [](const std::string& s, char c){ std::vector<std::string> o; std::stringstream ss(s); std::string t; while (std::getline(ss,t,c)) if(!t.empty()) o.push_back(t); return o; };
  if (spec=="y120"||spec.empty()) return from_gaps({120,120,120});
  if (spec=="cross4") return from_gaps({90,90,90,90});
  if (spec=="tri3d") { std::vector<Vec3<Real>> d; for(int k=0;k<3;k++){const double th=k*2*M_PI/3; d.push_back(gv_unit(Vec3<Real>{(Real)std::cos(th),(Real)std::sin(th),(Real)0.5}));} return d; }
  if (spec=="tetra4") { std::vector<Vec3<Real>> d={{1,1,1},{1,-1,-1},{-1,1,-1},{-1,-1,1}}; for(auto&v:d)v=gv_unit(v); return d; }
  if (spec.rfind("gaps:",0)==0){ std::vector<double> g; for(auto&t:split(spec.substr(5),',')) g.push_back(std::atof(t.c_str())); return from_gaps(g); }
  if (spec.rfind("dirs:",0)==0){ std::vector<Vec3<Real>> d; for(auto&tk:split(spec.substr(5),';')){auto c=split(tk,','); d.push_back(gv_unit(Vec3<Real>{(Real)std::atof(c[0].c_str()),(Real)std::atof(c[1].c_str()),(Real)std::atof(c[2].c_str())}));} return d; }
  SCTL_ASSERT_MSG(false,"bad spec"); return {};
}

// Area / min-Jacobian-weight / watertight / volume; returns the flux VECTOR (for combining lists).
template <class Real, class EL> void report_area(const std::string& name, const EL& el, const Comm& comm, double fout[3]=nullptr) {
  Vector<Real> X, Xn, wts, dist; Vector<Long> cnt;
  el.GetFarFieldNodes(X, Xn, wts, dist, cnt, (Real)1e-10);
  const Long N = wts.Dim(); Real A=0, minw=1e30, vol=0, f[3]={0,0,0};
  for (Long i=0;i<N;i++){ A+=wts[i]; minw=std::min(minw,wts[i]); for(int k=0;k<3;k++) f[k]+=wts[i]*Xn[i*3+k]; vol += (X[i*3]*Xn[i*3]+X[i*3+1]*Xn[i*3+1]+X[i*3+2]*Xn[i*3+2])*wts[i]/3; }
  A=GlobalReduce((double)A,comm,CommOp::SUM); vol=GlobalReduce((double)vol,comm,CommOp::SUM); minw=GlobalReduce((double)minw,comm,CommOp::MIN);
  for(int k=0;k<3;k++) f[k]=GlobalReduce((double)f[k],comm,CommOp::SUM);
  if(fout) for(int k=0;k<3;k++) fout[k]=f[k];
  if(!comm.Rank()) std::cout<<"  ["<<name<<"] area="<<std::setprecision(8)<<A<<"  minWt="<<std::setprecision(4)<<minw
    <<(minw>0?" (all Jac>0)":"  ***NEGATIVE JACOBIAN (fold)***")<<"  |int n dA|="<<std::sqrt(f[0]*f[0]+f[1]*f[1]+f[2]*f[2])<<"  vol="<<vol<<"\n";
}

} // namespace

int main(int argc, char** argv) {
  Comm::MPI_Init(&argc, &argv);
  {
    const Comm comm = Comm::World();
    const std::string spec_str = (argc>1)?std::string(argv[1]):std::string("y120");
    const Real    level  = (argc>2)?(Real)atof(argv[2]):(Real)1.5;
    const Integer ord    = (argc>3)?(Integer)atoi(argv[3]):12;
    const Integer nref   = (argc>4)?(Integer)atoi(argv[4]):1;
    const Real    etaj   = (argc>5)?(Real)atof(argv[5]):(Real)0.4;
    const Integer NsTr   = (argc>6)?(Integer)atoi(argv[6]):3;
    const Real    s_cap  = (argc>7)?(Real)atof(argv[7]):(Real)0.88;
    const Integer Ncap   = (argc>8)?(Integer)atoi(argv[8]):-1;
    const Real    alphaD = (argc>9)?(Real)atof(argv[9]):(Real)38.0;
    const Real    sigmaIn= (argc>10)?(Real)atof(argv[10]):(Real)-1;   // <0 => auto-thin for tight gaps
    const Real    clampf = (argc>11)?(Real)atof(argv[11]):(Real)0.8;

    GenSpec<Real> spec; spec.arm_dir=parse_spec(spec_str); spec.alpha_deg=alphaD; spec.alpha_clamp_frac=clampf;
    const Integer N=(Integer)spec.arm_dir.size();
    // Auto-thin sigma with the min pairwise arm angle (SAME rule as bifurc-general-bie), so the geometry
    // driver and the BIE driver agree by default (fat tubes merge in a tight gap and wreck the mesh).
    Real min_ang=M_PI;
    for(Integer i=0;i<N;i++) for(Integer j=i+1;j<N;j++){ double d=0; for(int c=0;c<3;c++) d+=(double)spec.arm_dir[i][c]*(double)spec.arm_dir[j][c]; min_ang=std::min(min_ang,(Real)std::acos(std::max(-1.0,std::min(1.0,d)))); }
    const Real min_gap_deg=min_ang*180/M_PI;
    spec.sigma = (sigmaIn>0)?sigmaIn:std::max<Real>((Real)0.075,(Real)0.15*std::min<Real>((Real)1,min_gap_deg/(Real)110));
    JuncGeom<Real> jg = build_junc_geom<Real>(spec, nref);
    if(!comm.Rank()){ std::cout<<"\n=== geom spec=\""<<spec_str<<"\" N="<<N<<" "<<(jg.coplanar?"coplanar":"non-coplanar")
      <<" order="<<ord<<" nref="<<nref<<" sigma="<<std::setprecision(4)<<spec.sigma<<" min_gap="<<min_gap_deg<<"deg alpha_eff="<<jg.alpha*180/M_PI<<"deg ===\n";
      for(Integer k=0;k<N;k++) std::cout<<"  arm "<<k<<" dir=("<<jg.u[k][0]<<","<<jg.u[k][1]<<","<<jg.u[k][2]<<")  cell_Na="<<jg.cell_Na[k]<<"\n"; }
    if(!comm.Rank() && !jg.coplanar && !jg.bigon3){
      std::cout<<"  [voro] "<<jg.vtx.size()<<" vertices:\n";
      for(size_t v=0;v<jg.vtx.size();v++) std::cout<<"    v"<<v<<"=("<<std::setprecision(4)<<jg.vtx[v][0]<<","<<jg.vtx[v][1]<<","<<jg.vtx[v][2]<<")\n";
      for(Integer k=0;k<N;k++){ std::cout<<"    cell "<<k<<" verts[";
        for(size_t e=0;e<jg.cell_vtx[k].size();e++) std::cout<<jg.cell_vtx[k][e]<<" "; std::cout<<"] np[";
        for(size_t e=0;e<jg.cell_np[k].size();e++) std::cout<<jg.cell_np[k][e]<<" "; std::cout<<"]\n"; }
    }

    // Build the coupled surface -- OR reload it from a matching reloadable bundle. When making a general
    // bifurcation we ALWAYS check GenGeomDir() (geom/) first: the validated presets are committed there, so
    // a preset run loads the exact surface instead of re-meshing. A miss (or QJ_GEOM_CACHE=0, or a
    // parameter mismatch) rebuilds and then writes the bundle back for next time.
    const Integer n_axial_geom = 3; const Long cheb_geom = 10, fourier_geom = 12;
    GenGeomParams<Real> prm;
    prm.order=ord; prm.nref=nref; prm.Ns_trans=NsTr; prm.Ncap=Ncap; prm.n_axial=n_axial_geom;
    prm.pou_kind=pou_kind(); prm.cheb_order=cheb_geom; prm.fourier_order=fourier_geom;
    prm.level=level; prm.eta_join=etaj; prm.s_cap=s_cap; prm.alpha_deg=alphaD; prm.sigma=spec.sigma; prm.clampf=clampf;
    const std::string prefix = GenGeomPath(spec_str, ord, nref);
    std::vector<Real> R0,a0,sc;
    QuadElemList<Real> junc; SlenderElemList<Real> arms; GenArmTable<Real> tab;
    if (TryLoadGenGeom<Real>(prefix, prm, junc, arms, tab, comm)) {
      for(Integer k=0;k<N;k++){ R0.push_back(tab.R0[k]); a0.push_back(tab.a0[k]); sc.push_back(tab.s_cap[k]); }
      if(!comm.Rank()){ std::cout<<"  [mesh] LOADED cached bundle "<<prefix<<".{mesh,arms} (skipped rebuild)\n";
        for(Integer k=0;k<N;k++) std::cout<<"  arm "<<k<<" R0="<<std::setprecision(6)<<R0[k]<<" axial["<<a0[k]<<","<<sc[k]<<"]\n"; }
    } else {
      Real mr=0;
      junc = BuildGenJunctionWithTransitions<Real>(spec, jg, ord, level, nref, etaj, NsTr, s_cap, R0, a0, sc, Ncap, &mr, comm);
      mr=GlobalReduce((double)mr,comm,CommOp::MAX);
      arms = BuildGenArmsSlender<Real>(jg, R0, a0, sc, n_axial_geom, cheb_geom, fourier_geom, comm);
      WriteGenGeom<Real>(prefix, junc, jg, R0, a0, sc, prm, comm);
      if(!comm.Rank()){ std::cout<<"  ray max|f-level|="<<std::setprecision(3)<<mr<<"  + wrote bundle "<<prefix<<".{mesh,arms}\n";
        for(Integer k=0;k<N;k++) std::cout<<"  arm "<<k<<" R0="<<std::setprecision(6)<<R0[k]<<" axial["<<a0[k]<<","<<sc[k]<<"]\n"; }
    }
    double fj[3]={0,0,0}, fa[3]={0,0,0};
    report_area<Real>("junc(all)", junc, comm, fj);
    { // panel edge-length aspect (max/min), like the BIE driver's verify_aspect
      Vector<Real> Xc; junc.GetNodeCoord(&Xc,nullptr,nullptr); const Long ne=junc.Size(), nn=(Long)ord*ord;
      auto nd=[&](Long e,Integer i,Integer j,int c){return Xc[(e*nn+(Long)i*ord+j)*3+c];};
      Real amax=0,asum=0; Long n3=0;
      for(Long e=0;e<ne;e++){ Real Lu=0,Lv=0;
        for(Integer j=0;j<ord;j++){Real L=0;for(Integer i=1;i<ord;i++){Real d=0;for(int c=0;c<3;c++){Real t=nd(e,i,j,c)-nd(e,i-1,j,c);d+=t*t;}L+=sqrt<Real>(d);}Lu+=L;}
        for(Integer i=0;i<ord;i++){Real L=0;for(Integer j=1;j<ord;j++){Real d=0;for(int c=0;c<3;c++){Real t=nd(e,i,j,c)-nd(e,i,j-1,c);d+=t*t;}L+=sqrt<Real>(d);}Lv+=L;}
        Lu/=ord;Lv/=ord; Real a=std::max(Lu,Lv)/std::max<Real>(std::min(Lu,Lv),(Real)1e-30); amax=std::max(amax,a);asum+=a;if(a>3)n3++; }
      if(!comm.Rank()) std::cout<<"  [aspect] panels="<<ne<<" mean="<<std::setprecision(3)<<asum/ne<<" max="<<amax<<" (#>3x="<<n3<<")\n"; }
    // Closed-model check (junc + arms). The slender arms were built (or loaded) above.
    report_area<Real>("arms", arms, comm, fa);

    if(!comm.Rank()){ double fc[3]={fj[0]+fa[0],fj[1]+fa[1],fj[2]+fa[2]};
      std::cout<<"  [COMBINED |int n dA|] = "<<std::setprecision(4)<<std::sqrt(fc[0]*fc[0]+fc[1]*fc[1]+fc[2]*fc[2])<<"  (junc+arms; ~0 iff closed)\n"; }

    // Per-region flux bucketing (serial): element emit order per arm = [Nr*Na sector][Ns_trans*Na trans][5*nc*nc cap].
    if (comm.Size()==1) {
      Vector<Real> X,Xn,wts,dist; Vector<Long> cnt; junc.GetFarFieldNodes(X,Xn,wts,dist,cnt,(Real)1e-10);
      const Integer Nr=spec.Nr0*nref, Na0=spec.Na0*nref, nc=(Ncap>0?Ncap:spec.Ncap0*nref);
      const Long e_sec=Nr*Na0, e_tr=NsTr*Na0, e_cap=5*nc*nc, e_arm=e_sec+e_tr+e_cap;
      double fsec[3]={0,0,0}, ftr[3]={0,0,0}, fcap[3]={0,0,0}; Long off=0;
      for (Long e=0;e<cnt.Dim();e++){ const Long we=e%e_arm; double* tgt = (we<e_sec)?fsec:(we<e_sec+e_tr?ftr:fcap);
        for (Long i=0;i<cnt[e];i++){ const Long g=off+i; for(int k=0;k<3;k++) tgt[k]+=wts[g]*Xn[g*3+k]; } off+=cnt[e]; }
      auto nrm=[](double*f){return std::sqrt(f[0]*f[0]+f[1]*f[1]+f[2]*f[2]);};
      std::cout<<"  [region flux] sectors="<<std::setprecision(4)<<nrm(fsec)<<" transitions="<<nrm(ftr)<<" caps="<<nrm(fcap)
               <<"  sec+tr="<<std::sqrt(std::pow(fsec[0]+ftr[0],2)+std::pow(fsec[1]+ftr[1],2)+std::pow(fsec[2]+ftr[2],2))<<"\n";
    }

    const std::string tag="vis/bifurc-general-" + GenGeomName(spec_str, ord, nref);   // spec-name + ord + nref
    junc.WriteVTK(tag+"-junc", Vector<Real>(), comm);
    if(!comm.Rank()) std::cout<<"  wrote "<<tag<<"-junc.vtu\n";
  }
  Comm::MPI_Finalize();
  return 0;
}
