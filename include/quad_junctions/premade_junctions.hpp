/**
 * premade_junctions.hpp -- a library of VALIDATED canonical bifurcation junctions with default meshing
 * parameters, derived from the vmtk vessel network (vessels_quad_smooth_0.5.stl) by degree+branch-angle
 * clustering, and each verified through the playbook BIE identity suite (bin/bifurc-general-bie, the
 * junction + CSBQ slender-arm + hemisphere-cap config) at the defaults below (order 12 / nref 2 /
 * fourier 36, tol swept to 1e-9). Accuracy numbers are the deepest-level (tol=1e-9) identity errors;
 * raw logs in data/premade-val/.
 *
 * Each entry's `spec` is a "dirs:x,y,z;..." string -> feed to parse_dirs_spec<Real>() (gen_network_geom.hpp)
 * or set GenSpec::arm_dir, then build with PREMADE_DEFAULTS via BuildGenJunctionWithTransitions /
 * BuildGenArmsSlender. Shapes are scale-invariant (the network's x10 coordinate scaling does not affect
 * them). All 18 source junctions are now included -- j06 and j08 were previously EXCLUDED for a bigon3
 * (asymmetric non-coplanar 3-arm) junction-body leak of ~0.39/0.56, which was FIXED 2026-08-07 (the wide-
 * pair bisector-edge wrong-half bug in build_junc_geom; see gen_junction_geom.hpp `voro_edge` and
 * bifurcation_meshing.md). They are now watertight and spectral like the rest.
 *
 * SCHEME NOTE: the accuracy columns below are the DEEPEST-level errors under the DUFFY self/near scheme
 * (SCTL_SELF_SCHEME=duffy). The driver DEFAULT is RectPolar Hybrid, under which the tight/wide-pair
 * junctions floor ~10-100x higher (e.g. j06/j07 DL ~4-6e-5); run the aspect-limited junctions under Duffy
 * to reach the numbers below. The BIE-sweep banner prints [scheme=Duffy|Hybrid].
 *
 * Accuracy tiers: WIDE junctions (min branch gap >~70 deg) reach ~1e-8/1e-9 = the general-bifurcation
 * floor. TIGHT-gap junctions (aspect_limited=true, ~57-67 deg) floor at ~1e-6/1e-7 due to junction panel
 * aspect (a circular hole in a lopsided cell has an unavoidably thin annulus) -- expected per the playbook,
 * not a defect.
 */
#pragma once

#include <cstddef>
#include <string>

namespace quad_junctions {

// Validated GLOBAL default meshing parameters (identical for every premade junction below).
struct PremadeDefaults {
  int    order     = 12;
  int    nref      = 2;
  long   fourier   = 36;
  double level     = 1.5;
  double eta_join  = 0.4;
  int    Ns_trans  = 3;
  double s_cap     = 0.88;
  int    n_axial   = 12;
  double alpha_deg = 38.0;
  double clampf    = 0.8;
  double sigma     = -1.0;   // <0 => auto-thin for tight gaps (the driver rule)
  int    Ncap      = -1;     // <0 => auto
  int    pou_kind  = 1;      // smootherstep (never 0)
};
static const PremadeDefaults PREMADE_DEFAULTS = PremadeDefaults();

struct PremadeJunction {
  const char* name;         // e.g. "j00"
  int         degree;       // number of arms
  const char* spec;         // "dirs:x,y,z;..." -> parse_dirs_spec / GenSpec::arm_dir
  double      min_gap_deg;  // smallest pairwise branch angle
  // deepest-level (tol=1e-9) identity errors at PREMADE_DEFAULTS:
  double      watertight;   // |int n dA| of the combined junction+arms surface (~0 = closed)
  double      dl_laplace;   // DL const-density identity, Laplace  (max abs)
  double      dl_stokes;    // DL const-density identity, Stokes   (max abs)
  double      green_laplace;// Green's identity, Laplace
  double      green_stokes; // Green's identity, Stokes
  bool        aspect_limited; // true => tight gap, floors ~1e-6/1e-7 (expected, not a defect)
};

// -------------------------------------------------------------------------------------------------
// 18 validated canonical junctions (j06/j08 added 2026-08-07 after the bigon3 wide-pair fix).
// -------------------------------------------------------------------------------------------------
static const PremadeJunction PREMADE_JUNCTIONS[] = {
  {"j00", 3, "dirs:0.453772342,-0.064094177,-0.888809653;-0.153503825,0.596746870,0.787610150;-0.042065232,-0.997349443,-0.059368386",
   84.4, 4.539e-13, 5.08e-9, 8.62e-8, 3.11e-9, 2.42e-8, false},
  {"j01", 3, "dirs:-0.987878938,-0.153147446,0.025319251;0.300723414,0.849111944,0.434251465;0.868206872,-0.298344257,-0.396494051",
   99.46, 5.883e-14, 3.94e-9, 7.38e-8, 2.19e-9, 2.98e-8, false},
  {"j02", 3, "dirs:0.416931807,0.727012050,-0.545546833;-0.787787810,0.136386448,0.600657226;0.747071137,-0.418746164,-0.516271601",
   73.22, 3.166e-12, 1.56e-8, 1.50e-7, 4.41e-9, 7.8e-8, false},
  {"j03", 3, "dirs:-0.847416829,0.218463756,-0.483899064;0.765234495,0.578882271,-0.281622948;0.102163668,-0.749976892,0.653526776",
   112.7, 7.808e-14, 3.44e-9, 8.04e-8, 1.74e-9, 2.65e-8, false},
  {"j04", 3, "dirs:0.690485149,-0.093634976,-0.717260587;-0.257695897,0.857152276,-0.445962780;0.038183119,-0.905915655,0.421733180",
   86.46, 5.206e-13, 5.31e-9, 7.15e-8, 2.87e-9, 3.06e-8, false},
  {"j05", 3, "dirs:-0.017013204,0.967225383,-0.253348790;0.069703987,-0.432470633,-0.898949668;-0.057024883,-0.847406223,0.527873902",
   96.43, 1.44e-13, 3.76e-9, 6.94e-8, 2.13e-9, 2.1e-8, false},
  {"j06", 3, "dirs:0.252805003,0.902333751,0.349118079;0.907409028,0.404279377,-0.114747724;0.049098034,-0.998785410,0.004133683",
   56.35, 3.861e-10, 1.88e-6, 5.30e-6, 6.08e-7, 1.18e-6, true},   // wide-pair bigon3 (fixed 2026-08-07); aspect-limited (56 deg)
  {"j07", 3, "dirs:0.421294002,0.226959595,0.878066459;0.097681715,-0.661614635,0.743454341;-0.336318431,0.375815044,-0.863511995",
   57.06, 4.369e-11, 5.73e-7, 1.68e-6, 1.82e-7, 5.47e-7, true},
  {"j08", 3, "dirs:-0.745832457,-0.655558925,0.118222002;0.660444772,-0.577925497,0.479390052;0.908229019,0.114562337,-0.402486670",
   70.08, 2.507e-11, 1.24e-7, 2.06e-7, 3.95e-8, 7.63e-8, false},  // wide-pair bigon3 (fixed 2026-08-07, re-verified 2026-08-10); ~1e-7 tier
  {"j09", 4, "dirs:-0.157964683,0.810226218,0.564429478;0.552052660,-0.833150750,0.033131383;-0.780525207,-0.608847251,0.141723063;0.317115796,0.115210521,-0.941362899",
   85.35, 3.73e-13, 4.55e-9, 8.78e-8, 2.46e-9, 2.48e-8, false},
  {"j10", 4, "dirs:0.004120244,0.393324738,0.919390382;-0.135511171,-0.591813786,0.794602520;-0.216462963,0.288754322,-0.932611777;0.533097574,0.844299818,-0.054449925",
   60.18, 9.427e-11, 4.43e-7, 1.34e-6, 3.26e-8, 8.59e-8, true},
  {"j11", 4, "dirs:0.412244828,0.905802455,0.097857618;-0.096091727,-0.190211379,0.977029176;-0.788230715,0.298594399,0.538083380;0.502853483,-0.298033986,-0.811365589",
   57.0, 3.211e-11, 2.48e-7, 1.27e-6, 1.26e-8, 2.07e-7, true},
  {"j12", 4, "dirs:-0.702861913,-0.021487952,-0.711001687;0.343455822,-0.330285619,-0.879175471;0.468495148,0.868703667,0.160829832;-0.060181898,-0.389093602,0.919230280",
   67.0, 9.748e-13, 2.19e-8, 3.85e-7, 2.55e-9, 4.53e-8, false},
  {"j13", 4, "dirs:-0.004319935,-0.793171350,0.608983208;0.314807332,-0.625895156,-0.713548595;0.037784202,0.950085221,-0.309694087;-0.976603850,0.214250072,0.018488563",
   80.74, 1.064e-12, 7.63e-9, 7.45e-8, 1.8e-9, 2.14e-8, false},
  {"j14", 4, "dirs:0.055297185,0.004314871,-0.998460617;0.766200405,-0.639230742,0.065734294;-0.003528434,0.466309720,0.884614490;-0.994362349,0.106018817,-0.001878472",
   87.06, 1.648e-13, 4.5e-9, 1.04e-7, 2.53e-9, 3.0e-8, false},
  {"j15", 4, "dirs:-0.674126627,-0.534009070,-0.510281887;0.474714243,0.641570349,0.602522925;0.592261688,-0.775497982,0.218698362;-0.611321937,0.713996781,-0.341312300",
   78.17, 6.297e-13, 6.36e-9, 1.32e-7, 1.77e-9, 3.3e-8, false},
  {"j16", 4, "dirs:0.731690288,-0.082427074,0.676635130;-0.310264379,-0.133597706,0.941216058;0.215013956,-0.180406265,-0.959803406;-0.755334687,0.448817946,-0.477526922",
   65.11, 2.424e-12, 4.13e-8, 6.16e-7, 4.83e-9, 5.99e-8, true},
  {"j17", 5, "dirs:-0.174217198,-0.339468913,0.924342591;-0.449006559,0.398131087,-0.799927964;0.956314451,0.087820374,0.278837324;0.424801899,0.776707629,-0.465046886;-0.016289439,-0.939995176,-0.340798654",
   60.63, 1.185e-11, 5.77e-8, 9.43e-7, 1.93e-8, 1.13e-7, true},
};
static const int N_PREMADE_JUNCTIONS = (int)(sizeof(PREMADE_JUNCTIONS) / sizeof(PREMADE_JUNCTIONS[0]));

// (Formerly EXCLUDED: j06/j08. Their bigon3 wide-pair leak was fixed 2026-08-07 -- see the header note --
// and they are now in the list above. Before the fix both leaked the junction body ~0.39/0.56, order-
// independently, because a wide arm pair put the bisector-edge waypoint on the wrong half of the great
// circle so the cell grabbed the wrong lune.)

inline const PremadeJunction* find_premade(const std::string& name) {
  for (int i = 0; i < N_PREMADE_JUNCTIONS; i++)
    if (name == PREMADE_JUNCTIONS[i].name) return &PREMADE_JUNCTIONS[i];
  return nullptr;
}

}  // namespace quad_junctions
