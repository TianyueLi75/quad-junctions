#ifndef QUAD_JUNCTIONS_MPI_UTILS_HPP
#define QUAD_JUNCTIONS_MPI_UTILS_HPP

/**
 * Distributed-memory helpers shared by the quad-junctions drivers.
 *
 * Under MPI every rank owns only a contiguous slice of the geometry:
 *   - QuadElemList<Real>(order, X, comm)  replicate-then-slices internally (pass comm);
 *   - SlenderElemList  has no comm ctor,  so the caller must slice the element loop by
 *     rank (CSBQ's k0 = Nelem*pid/Np pattern) BEFORE Init.
 * Consequently GetNodeCoord / GetFarFieldNodes / ComputePotential all return this rank's
 * local slice, so any scalar norm / area / max-error accumulated over local nodes MUST be
 * reduced across ranks before it is compared or printed, and result prints are emitted on
 * rank 0 only. These inline helpers wrap that reduction (mirroring the GlobalReduce helpers
 * in SCTL_quad_element/src/test-quad-elem.cpp).
 */

#include <sctl.hpp>

namespace quad_junctions {
using namespace sctl;

// Reduce a local scalar across all ranks (MAX/SUM/MIN/...) and return the global value on
// every rank. inline => safe to include in multiple translation units.
inline double GlobalReduce(double x, const Comm& comm, CommOp op) {
  StaticArray<double,2> buf; buf[0] = x; buf[1] = 0;
  comm.Allreduce(buf+0, buf+1, 1, op);
  return buf[1];
}
inline Long GlobalReduce(Long x, const Comm& comm, CommOp op) {
  StaticArray<Long,2> buf; buf[0] = x; buf[1] = 0;
  comm.Allreduce(buf+0, buf+1, 1, op);
  return buf[1];
}

// Reduce a fixed-length local vector element-wise across ranks (in place, result on all ranks).
inline void GlobalReduce(Vector<double>& v, const Comm& comm, CommOp op) {
  Vector<double> out(v.Dim()); out = 0;
  comm.Allreduce(v.begin(), out.begin(), v.Dim(), op);
  v = out;
}

// SCTL's CommOp has no MAXLOC, so this reduces a local max together with the coordinates of the
// local argmax: it returns (on every rank) the global max, and writes into out_xyz the coordinates
// of the owning node -- the lowest-rank node attaining the global max (deterministic tie-break).
// Used by the DL/Green's drivers to report WHERE the worst error lives after partitioning.
inline double GlobalMaxLoc(double local_max, const double local_xyz[3], const Comm& comm, double out_xyz[3]) {
  const double gmax = GlobalReduce(local_max, comm, CommOp::MAX);
  const Long owner = GlobalReduce((Long)((local_max == gmax) ? comm.Rank() : comm.Size()), comm, CommOp::MIN);
  double loc[3] = {0, 0, 0};
  if (comm.Rank() == owner) { loc[0] = local_xyz[0]; loc[1] = local_xyz[1]; loc[2] = local_xyz[2]; }
  for (int k = 0; k < 3; k++) out_xyz[k] = GlobalReduce(loc[k], comm, CommOp::SUM);
  return gmax;
}

} // namespace quad_junctions

#endif // QUAD_JUNCTIONS_MPI_UTILS_HPP
