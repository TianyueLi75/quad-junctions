/**
 * Check cilia_carpet-bie's combined_surface_mean over BOTH list types: a cubed-sphere QuadElemList
 * (radius R1) + a CSBQ SlenderElemList sphere (body of revolution, centerline along z in [-R2,R2],
 * radius sqrt(R2^2 - z^2)). With density=(1,0,0) the combined mean must be (1,0,0) and total_area must
 * be 4*pi*(R1^2 + R2^2) -- verifying the QuadElemList path (empty GetFarFieldDensity -> use collocation
 * density) AND the SlenderElemList path (GetFarFieldDensity fills Fff) both integrate correctly, and that
 * the two contributions add. Serial, no PVFMM (GetFarFieldNodes is quadrature, not FMM).
 *   make bin/areatest && ./bin/areatest [order PatchPerFace tol]
 */
#include <sctl.hpp>
#include <quad_junctions/collar_mount_geom.hpp>
#include <quad_junctions/quad_scheme.hpp>     // QJDefaultScheme (Duffy default, SCTL_SELF_SCHEME=hybrid opt-out)
#include <csbq/slender_element.hpp>
#include <csbq/slender_element.cpp>
#include <iomanip>
#include <iostream>

using namespace sctl;
using namespace quad_junctions;

// Replicate combined_surface_mean's per-list accum: GetFarFieldNodes -> wts; GetFarFieldDensity -> Fff
// (EMPTY for QuadElemList => use collocation density directly; FILLED for SlenderElemList). Adds this
// list's contribution to `area` and `sm[KDIM]`, and reports whether it was 1:1 / empty.
template <class Real, class LST>
void accum(const LST& lst, const Vector<Real>& sig, Integer KDIM, Real tol, Real& area, Real* sm, const char* tag) {
  Vector<Real> Xff, Xnff, wts, dist, Fff; Vector<Long> cnt;
  lst.GetFarFieldNodes(Xff, Xnff, wts, dist, cnt, tol);
  lst.GetFarFieldDensity(Fff, sig);
  const Vector<Real>& dens = (Fff.Dim() ? Fff : sig);
  const Long Nq = wts.Dim();
  Vector<Real> Xc; lst.GetNodeCoord(&Xc, nullptr, nullptr); const Long Nnode = Xc.Dim()/3;
  Real a = 0; Vector<Real> s(KDIM); s = 0;
  for (Long i = 0; i < Nq; i++) { a += wts[i]; for (Integer k = 0; k < KDIM; k++) s[k] += wts[i]*dens[i*KDIM+k]; }
  area += a; for (Integer k = 0; k < KDIM; k++) sm[k] += s[k];
  std::cout << "  [" << tag << "] Nnode=" << Nnode << " far-wts=" << Nq << " GetFarFieldDensity="
            << (Fff.Dim() ? "FILLED" : "empty") << "  area=" << a << "\n";
}

int main(int argc, char** argv) {
  Comm::MPI_Init(&argc, &argv);
  using Real = double;
  {
    Comm comm = Comm::Self();
    const Integer order        = (argc > 1) ? std::atoi(argv[1]) : 16;
    const Integer PatchPerFace = (argc > 2) ? std::atoi(argv[2]) : 4;
    const Real    tol          = (argc > 3) ? std::atof(argv[3]) : 1e-9;
    const Real    R1 = 1.0;    // quad cubed sphere radius
    const Real    R2 = 0.5;    // CSBQ slender sphere radius
    constexpr Integer KDIM = 3;

    // ---- QuadElemList: cubed sphere radius R1 ----
    Vector<Real> Xall;
    const Vector<Real>& nds = QuadElemList<Real>::ParamNodes(order);
    const Real hh = (Real)1 / PatchPerFace;
    for (Integer face = 0; face < 6; face++)
      for (Integer iu = 0; iu < PatchPerFace; iu++)
        for (Integer iv = 0; iv < PatchPerFace; iv++) {
          const Real a_c = 2*(iu + (Real)0.5)/PatchPerFace - 1, b_c = 2*(iv + (Real)0.5)/PatchPerFace - 1;
          const Mount<Real> mnt = PatchMount<Real>(face, a_c, b_c, R1);
          for (Integer i = 0; i < order; i++) { const Real X = -hh + 2*hh*nds[i];
            for (Integer j = 0; j < order; j++) { const Real Y = -hh + 2*hh*nds[j];
              const Vec3<Real> w = mnt(X, Y, (Real)0); Xall.PushBack(w[0]); Xall.PushBack(w[1]); Xall.PushBack(w[2]); } }
        }
    QuadElemList<Real> sph(order, Xall);
    sph.SetQuadScheme(quad_junctions::QJDefaultScheme<Real>(), 10, 200, 12);

    // ---- SlenderElemList: sphere radius R2 as a body of revolution about z (radius sqrt(R2^2 - z^2)) ----
    const Long Nelem = 32, cheb = 10, fourier = 16;
    Vector<Long> elem_order, forder; Vector<Real> coord, radius;
    for (Long k = 0; k < Nelem; k++) {
      elem_order.PushBack(cheb); forder.PushBack(fourier);
      const Vector<Real>& cn = SlenderElemList<Real>::CenterlineNodes(cheb);
      for (Long j = 0; j < cheb; j++) {
        const Real z = -R2 + 2*R2*(k + cn[j]) / Nelem;                     // centerline along z in [-R2, R2]
        coord.PushBack((Real)0); coord.PushBack((Real)0); coord.PushBack(z);
        radius.PushBack(std::sqrt(std::max((Real)0, R2*R2 - z*z)));        // sphere cross-section radius
      }
    }
    SlenderElemList<Real> ssph(elem_order, forder, coord, radius);

    // ---- combined accum (density (1,0,0) on both lists) ----
    Vector<Real> Xq; sph.GetNodeCoord(&Xq, nullptr, nullptr);  const Long Nq = Xq.Dim()/3;
    Vector<Real> Xs; ssph.GetNodeCoord(&Xs, nullptr, nullptr); const Long Ns = Xs.Dim()/3;
    Vector<Real> sigq(Nq*KDIM), sigs(Ns*KDIM);
    for (Long i = 0; i < Nq; i++) { sigq[i*KDIM]=1; sigq[i*KDIM+1]=0; sigq[i*KDIM+2]=0; }
    for (Long i = 0; i < Ns; i++) { sigs[i*KDIM]=1; sigs[i*KDIM+1]=0; sigs[i*KDIM+2]=0; }

    Real area = 0, sm[KDIM] = {0,0,0};
    std::cout << std::setprecision(12) << "combined_surface_mean check (density (1,0,0)):\n";
    accum<Real>(sph,  sigq, KDIM, tol, area, sm, "quad  cubed-sphere R1=1.0");
    accum<Real>(ssph, sigs, KDIM, tol, area, sm, "CSBQ slender-sphere R2=0.5");

    const Real Aq = 4*const_pi<Real>()*R1*R1, As = 4*const_pi<Real>()*R2*R2, Aexact = Aq + As;
    std::cout << "  ---- combined ----\n"
      << "  total_area = " << area << "   expect 4pi(R1^2+R2^2) = " << Aexact << " (=" << Aq << "+" << As << ")"
      << "   rel err = " << std::fabs(area-Aexact)/Aexact << "\n"
      << "  combined mean = (" << sm[0]/area << ", " << sm[1]/area << ", " << sm[2]/area << ")   (expect (1,0,0))\n";
  }
  Comm::MPI_Finalize();
  return 0;
}
