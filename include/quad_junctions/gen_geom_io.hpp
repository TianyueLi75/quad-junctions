/**
 * Exact, reloadable geometry-bundle I/O for the generalized N-arm bifurcation.
 *
 * The generalized mesher (gen_junction_geom.hpp) builds the coupled surface in-process from a GenSpec
 * (field + Voronoi machinery). This header persists the RESULT of that build as a small self-contained
 * bundle that any downstream BIE / flow driver can load WITHOUT re-running the field/partition code -- the
 * same "exact QuadElemList mesh + per-arm slender frame table" contract ybifurc-hybrid-bie already uses
 * for its junction cache (.mesh + .frame), generalized to N arms and carrying the arm frames explicitly.
 *
 * Bundle = two files sharing a prefix:
 *   <prefix>.mesh   the junction QuadElemList (junction sectors + POU transitions + caps), bit-exact via
 *                   QuadElemList::Write (17-digit -> exact for double), MPI-collective (rank 0 writes).
 *   <prefix>.arms   a plain-text per-arm frame table (rank 0 writes): the slender arms are a pure function
 *                   of (u, e1, R0, a0, s_cap) per arm + (n_axial, cheb_order, fourier_order), so we store
 *                   exactly those. No field, no Voronoi, no spec needed to rebuild. The v2 header also
 *                   records the full mesh-determining parameter set (GenGeomParams) so a cache lookup can
 *                   VALIDATE a hit against the requested parameters and transparently rebuild on drift.
 *
 * ReadGenGeom(prefix, junc, arms, comm) reconstructs the SAME coupled (QuadElemList junc, SlenderElemList
 * arms) pair the mesher produced -- node/density ordering is the operator's name-sorted "0_junc" then
 * "1_arms", so hybrid_bie_tests.hpp (and any solve driver following the ybifurc-hybrid layout) applies
 * unchanged. BuildArmsSlenderFromTable duplicates BuildGenArmsSlender's CSBQ k0/k1 partition so a loaded
 * run matches a freshly-built run rank-for-rank.
 *
 * Bundles live under GenGeomDir() -- "geom/" by default, overridable with QJ_GEOM_DIR (legacy alias
 * QJ_EXPORT_GEOM). The build drivers check GenGeomDir() for a matching bundle BEFORE meshing (see
 * TryLoadGenGeom): the four validated presets (y120/cross4/tri3d/tetra4) are committed there, everything
 * else is a gitignored build artifact (geom/.gitignore). QJ_GEOM_CACHE=0 forces a rebuild.
 */
#pragma once

#include <quad_junctions/gen_junction_geom.hpp>       // Vec3, GenSpec/JuncGeom types, SlenderElemList
#include <cctype>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <string>
#include <sys/stat.h>
#include <vector>

namespace quad_junctions {
using namespace sctl;

// Filename-safe identifier for a geometry: the junction spec name + order + nref, e.g. "y120-ord12-nref2",
// "tetra4-ord8-nref1", "gaps150-150-60-ord12-nref2". Presets (y120/tetra4/cross4/tri3d) pass through as-is;
// gaps:/dirs: specs are sanitized (':' dropped, ','->'-', ';'->'_', '.'->'p', '-'->'m') so exported bundles
// and VTUs are self-identifying and never clobber a different shape at the same resolution.
inline std::string GenGeomName(const std::string& spec, Integer order, Integer nref) {
  std::string t;
  for (char c : spec) {
    if (std::isalnum((unsigned char)c)) t += c;
    else if (c == ',') t += '-';
    else if (c == ';') t += '_';
    else if (c == '.') t += 'p';
    else if (c == '-') t += 'm';
    // ':' and any other punctuation dropped
  }
  if (t.empty()) t = "spec";
  std::ostringstream os; os << t << "-ord" << order << "-nref" << nref;
  return os.str();
}

// Directory holding the reloadable geometry bundles. "geom/" by default; QJ_GEOM_DIR overrides it
// (QJ_EXPORT_GEOM is honored as a legacy alias). Always returned with a trailing '/'.
inline std::string GenGeomDir() {
  const char* d = std::getenv("QJ_GEOM_DIR");
  if (!d || !*d) d = std::getenv("QJ_EXPORT_GEOM");
  std::string s = (d && *d) ? std::string(d) : std::string("geom");
  if (s.empty()) s = "geom";
  if (s.back() != '/') s += '/';
  return s;
}

// Full bundle prefix (directory + spec-name) for a (spec, order, nref); append ".mesh"/".arms".
inline std::string GenGeomPath(const std::string& spec, Integer order, Integer nref) {
  return GenGeomDir() + GenGeomName(spec, order, nref);
}

// True iff BOTH bundle files (<prefix>.mesh and <prefix>.arms) are present and readable.
inline bool GenGeomExists(const std::string& prefix) {
  std::ifstream m(prefix + ".mesh"), a(prefix + ".arms");
  return m.good() && a.good();
}

// mkdir the directory holding `prefix` (best-effort; ignores "already exists"). Rank-0 only in practice.
inline void EnsureBundleDir(const std::string& prefix) {
  const size_t slash = prefix.find_last_of('/');
  if (slash == std::string::npos) return;
  const std::string dir = prefix.substr(0, slash);
  if (!dir.empty()) ::mkdir(dir.c_str(), 0777);   // errno ignored on purpose (EEXIST is fine)
}

// ---- Full mesh-determining parameter set, stored in the .arms v2 header so a cache hit can be VALIDATED
//      against the requested parameters. Two builds with equal GenGeomParams (and the same spec, which is
//      baked into the filename) produce the same surface bit-for-bit, so a loaded bundle is a legitimate
//      substitute for a rebuild only when the parameters match exactly. ----
template <class Real> struct GenGeomParams {
  Integer order = 0, nref = 0, Ns_trans = 0, Ncap = -1, n_axial = 3, pou_kind = 1;
  Long    cheb_order = 10, fourier_order = 12;
  Real    level = 0, eta_join = 0, s_cap = 0, alpha_deg = 0, sigma = 0, clampf = 0;
  bool operator==(const GenGeomParams& o) const {
    return order == o.order && nref == o.nref && Ns_trans == o.Ns_trans && Ncap == o.Ncap &&
           n_axial == o.n_axial && pou_kind == o.pou_kind && cheb_order == o.cheb_order &&
           fourier_order == o.fourier_order && level == o.level && eta_join == o.eta_join &&
           s_cap == o.s_cap && alpha_deg == o.alpha_deg && sigma == o.sigma && clampf == o.clampf;
  }
  bool operator!=(const GenGeomParams& o) const { return !(*this == o); }
};

// ---- Per-arm frame table = everything needed to rebuild the slender arms (no field / no partition). ----
template <class Real> struct GenArmTable {
  Integer N = 0, order = 0, n_axial = 3;
  Long cheb_order = 10, fourier_order = 12;
  std::vector<Vec3<Real>> u, e1;               // per-arm axis + azimuthal reference (beta=0 -> e1)
  std::vector<Real> R0, a0, s_cap;             // per-arm radius, axial start, axial cap-arc
  GenGeomParams<Real> params;                  // full build parameters (from the v2 header; v1 -> defaults)
  bool has_params = false;                      // false for a legacy v1 bundle (params unknown)
};

// Reconstruct the slender arms from a frame table, replicating BuildGenArmsSlender's CSBQ MPI partition
// (elements 0..N*n_axial split as k0g=Nelem*pid/Np) so a loaded run is rank-identical to a built one.
template <class Real> SlenderElemList<Real> BuildArmsSlenderFromTable(const GenArmTable<Real>& tab, const Comm& comm = Comm::Self()) {
  const Integer N = tab.N;
  const Long Nelem = (Long)N*tab.n_axial, Np = comm.Size(), pid = comm.Rank();
  const Long k0g = (Nelem*pid)/Np, k1g = (Nelem*(pid+1))/Np;
  Vector<Long> elem_order, forder; Vector<Real> coord, radius, orient; Long eg = 0;
  for (Integer k = 0; k < N; k++) {
    const Vec3<Real>& u = tab.u[k]; const Vec3<Real>& e1 = tab.e1[k];
    const Real s0 = tab.a0[k], s1 = tab.s_cap[k];
    for (Integer p = 0; p < tab.n_axial; p++, eg++) {
      if (eg < k0g || eg >= k1g) continue;
      elem_order.PushBack(tab.cheb_order); forder.PushBack(tab.fourier_order);
      const Vector<Real>& cn = SlenderElemList<Real>::CenterlineNodes(tab.cheb_order);
      for (Long j = 0; j < tab.cheb_order; j++) {
        const Real s = s0 + (s1-s0)*(p+cn[j])/tab.n_axial;
        coord.PushBack(s*u[0]); coord.PushBack(s*u[1]); coord.PushBack(s*u[2]);
        radius.PushBack(tab.R0[k]);
        orient.PushBack(e1[0]); orient.PushBack(e1[1]); orient.PushBack(e1[2]);
      }
    }
  }
  return SlenderElemList<Real>(elem_order, forder, coord, radius, orient);
}

// ---- Write the arm table (rank 0). jg supplies the per-arm frames; R0/a0/sc from the junction build; prm
//      is the full build-parameter set (its order/n_axial/cheb/fourier are the authoritative copies). ----
template <class Real> void WriteGenArmTable(const std::string& prefix, const JuncGeom<Real>& jg,
    const std::vector<Real>& R0, const std::vector<Real>& a0, const std::vector<Real>& s_cap,
    const GenGeomParams<Real>& prm, const Comm& comm = Comm::Self()) {
  if (comm.Rank()) return;
  const Integer N = jg.N;
  std::ofstream f(prefix + ".arms"); SCTL_ASSERT_MSG(f.good(), "WriteGenArmTable: cannot open .arms for write");
  f << std::setprecision(17);
  f << "# gen-geom arm-frame table v2\n";
  f << "# N order n_axial cheb_order fourier_order\n";
  f << N << " " << prm.order << " " << prm.n_axial << " " << prm.cheb_order << " " << prm.fourier_order << "\n";
  f << "# params: nref Ns_trans Ncap pou_kind  level eta_join s_cap alpha_deg sigma clampf\n";
  f << prm.nref << " " << prm.Ns_trans << " " << prm.Ncap << " " << prm.pou_kind << "  "
    << prm.level << " " << prm.eta_join << " " << prm.s_cap << " " << prm.alpha_deg << " " << prm.sigma << " " << prm.clampf << "\n";
  f << "# per arm: ux uy uz  e1x e1y e1z  R0 a0 s_cap\n";
  for (Integer k = 0; k < N; k++)
    f << jg.u[k][0] << " " << jg.u[k][1] << " " << jg.u[k][2] << "  "
      << jg.e1[k][0] << " " << jg.e1[k][1] << " " << jg.e1[k][2] << "  "
      << R0[k] << " " << a0[k] << " " << s_cap[k] << "\n";
}

// ---- Read the arm table (every rank reads the small shared text file). Handles both v1 (no params line)
//      and v2 (params line present) headers. ----
template <class Real> GenArmTable<Real> ReadGenArmTable(const std::string& prefix) {
  std::ifstream f(prefix + ".arms"); SCTL_ASSERT_MSG(f.good(), "ReadGenArmTable: cannot open .arms for read");
  // The version marker lives in the first comment line; capture it before the comment-skipping reader runs.
  std::string first; std::getline(f, first);
  const bool v2 = first.find("v2") != std::string::npos;
  GenArmTable<Real> tab;
  auto next = [&]() -> std::string { std::string l; while (std::getline(f, l)) { size_t p = l.find_first_not_of(" \t\r\n"); if (p != std::string::npos && l[p] != '#') return l; } return std::string(); };
  { std::istringstream ss(next()); ss >> tab.N >> tab.order >> tab.n_axial >> tab.cheb_order >> tab.fourier_order; }
  SCTL_ASSERT_MSG(tab.N >= 2, "ReadGenArmTable: bad N");
  GenGeomParams<Real>& p = tab.params;
  p.order = tab.order; p.n_axial = tab.n_axial; p.cheb_order = tab.cheb_order; p.fourier_order = tab.fourier_order;
  if (v2) {
    std::istringstream ss(next());
    ss >> p.nref >> p.Ns_trans >> p.Ncap >> p.pou_kind >> p.level >> p.eta_join >> p.s_cap >> p.alpha_deg >> p.sigma >> p.clampf;
    tab.has_params = true;
  }
  tab.u.resize(tab.N); tab.e1.resize(tab.N); tab.R0.resize(tab.N); tab.a0.resize(tab.N); tab.s_cap.resize(tab.N);
  for (Integer k = 0; k < tab.N; k++) {
    std::istringstream ss(next());
    ss >> tab.u[k][0] >> tab.u[k][1] >> tab.u[k][2]
       >> tab.e1[k][0] >> tab.e1[k][1] >> tab.e1[k][2]
       >> tab.R0[k] >> tab.a0[k] >> tab.s_cap[k];
  }
  return tab;
}

// ---- Write the whole bundle: <prefix>.mesh (collective) + <prefix>.arms (rank 0). Creates the target
//      directory if it does not exist (rank 0). ----
template <class Real> void WriteGenGeom(const std::string& prefix, const QuadElemList<Real>& junc, const JuncGeom<Real>& jg,
    const std::vector<Real>& R0, const std::vector<Real>& a0, const std::vector<Real>& s_cap,
    const GenGeomParams<Real>& prm, const Comm& comm = Comm::Self()) {
  if (!comm.Rank()) EnsureBundleDir(prefix);
  junc.Write(prefix + ".mesh", comm);                                  // collective: rank 0 writes the shared file
  WriteGenArmTable<Real>(prefix, jg, R0, a0, s_cap, prm, comm);
}

// ---- Read the whole bundle into the coupled (junc, arms) pair a downstream driver consumes. ----
template <class Real> void ReadGenGeom(const std::string& prefix, QuadElemList<Real>& junc, SlenderElemList<Real>& arms,
    GenArmTable<Real>* tab_out = nullptr, const Comm& comm = Comm::Self()) {
  junc.template Read<Real>(prefix + ".mesh", comm);
  const GenArmTable<Real> tab = ReadGenArmTable<Real>(prefix);
  arms = BuildArmsSlenderFromTable<Real>(tab, comm);
  if (tab_out) *tab_out = tab;
}

// ---- Cache lookup for a build driver: if a matching bundle already lives under GenGeomDir(), load it
//      (skipping the field/Voronoi/mesh build) and return true; otherwise return false so the caller
//      builds and then WriteGenGeom's the result. A hit requires GenGeomExists AND that the bundle's
//      recorded GenGeomParams equal `prm` exactly (a v1 bundle has no params and is never auto-loaded, to
//      avoid silently substituting a mesh built with unknown parameters). QJ_GEOM_CACHE=0 disables it. ----
template <class Real> bool TryLoadGenGeom(const std::string& prefix, const GenGeomParams<Real>& prm,
    QuadElemList<Real>& junc, SlenderElemList<Real>& arms, GenArmTable<Real>& tab, const Comm& comm = Comm::Self()) {
  if (const char* e = std::getenv("QJ_GEOM_CACHE")) if (std::atoi(e) == 0) return false;
  if (!GenGeomExists(prefix)) return false;
  tab = ReadGenArmTable<Real>(prefix);
  if (!tab.has_params || tab.params != prm) return false;             // stale / different params -> rebuild
  junc.template Read<Real>(prefix + ".mesh", comm);
  arms = BuildArmsSlenderFromTable<Real>(tab, comm);
  return true;
}

} // namespace quad_junctions
