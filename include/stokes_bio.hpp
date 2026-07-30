// =============================================================================
// stokes_bio.hpp
//
// StokesBIO: combined-field Stokes boundary integral operator.
//
// Thin wrapper around the CSBQ BoundaryIntegralOp that assembles the unified
// single- plus double-layer operator
//
//     u(x) = SL_scal * S[mu](x) + DL_scal * D[mu](x),
//
// where S and D are the Stokes single- and double-layer potentials and mu is the
// surface density. The class manages both layer potentials together: adding
// element lists, setting target points and normals, choosing the periodicity
// (X / XY / XYZ), setting the quadrature accuracy, and applying the operator
// (ComputePotential), as well as the individual S and D applications and the
// sqrt-weight scalings used for symmetric preconditioning. The single-layer
// self-interaction uses a volume-potential correction (stokes_sl_volpot).
//
// Usage:
//   Header-only template. Include this header; the implementation in
//   stokes_bio.cpp is pulled in automatically at the bottom of the file.
//       StokesBIO<Real> op(SL_scal, DL_scal, comm);
//       op.AddElemList(elem_lst);
//       op.SetAccuracy(tol);
//       op.SetTargetCoord(X);
//       op.SetPeriodicity(sctl::Periodicity::XYZ, 1.0);
//       op.ComputePotential(U, mu);
// =============================================================================
#ifndef _UTILS_STOKESBIO_HPP_
#define _UTILS_STOKESBIO_HPP_

#include <csbq.hpp>
#include <tuple>

/**
 * PVFMM cannot handle combined field kernel. Compute SL and DL separately and add them.
 */
template <class Real> class StokesBIO {
  public:

    StokesBIO() = delete;
    StokesBIO(const StokesBIO&) = delete;
    StokesBIO& operator= (const StokesBIO&) = delete;

    StokesBIO(const Real SL_scal, const Real DL_scal, const sctl::Comm comm);

    /**
     * Set periodicity.
     *
     * @param[in] periodicity periodicity type (NONE, X, XY, XYZ).
     *
     * @param[in] period_length length of the periodic box in each dimension.
     * Must be positive if periodicity is not NONE.
     *
     * @remark Periodicity only supported in 3D and with PVFMM.
     */
    void SetPeriodicity(sctl::Periodicity periodicity, Real period_length = 0);

    /**
     * Specify quadrature accuracy tolerance.
     *
     * @param[in] tol quadrature accuracy.
     */
    void SetAccuracy(Real tol);

    /**
     * Add an element-list.
     *
     * @param[in] elem_lst an object (of type ElemLstType, derived from the
     * base class ElementListBase) that contains the description of a list of
     * elements.
     *
     * @param[in] name a string name for this element list.
     * 
     * @param[in] sl, dl booleans for whether the element list object will be added to the SL and/or DL operator.
     */
    template <class ElemLstType> void AddElemList(const ElemLstType& elem_lst, const std::string& name = std::to_string(typeid(ElemLstType).hash_code()), const bool sl = true, const bool dl = true);

    /**
     * Get const reference to an element-list.
     *
     * @param[in] name name of the element-list to return.
     *
     * @return const reference to the element-list.
     */
    template <class ElemLstType> const ElemLstType& GetElemList(const std::string& name = std::to_string(typeid(ElemLstType).hash_code())) const;

    /**
     * Delete an element-list.
     *
     * @param[in] name name of the element-list to return.
     */
    void DeleteElemList(const std::string& name);

    /**
     * Delete an element-list.
     */
    template <class ElemLstType> void DeleteElemList();

    /**
     * Set target point coordinates.
     *
     * @param[in] Xtrg the coordinates of target points in array-of-struct
     * order: {x_1, y_1, z_1, x_2, ..., x_n, y_n, z_n}
     */
    void SetTargetCoord(const sctl::Vector<Real>& Xtrg);

    /**
     * Set target point normals.
     *
     * @param[in] Xn_trg the coordinates of target points in array-of-struct
     * order: {nx_1, ny_1, nz_1, nx_2, ..., nx_n, ny_n, nz_n}
     */
    void SetTargetNormal(const sctl::Vector<Real>& Xn_trg);

    /**
     * Get local dimension of the boundary integral operator. Dim(0) is the
     * input dimension and Dim(1) is the output dimension.
     */
    sctl::Long Dim(sctl::Integer k) const;

    /**
     * Setup the boundary integral operator.
     */
    void Setup() const;

    /**
     * Clear setup data.
     */
    void ClearSetup() const;

    /**
     * Evaluate the boundary integral operator.
     *
     * @param[out] U the potential computed at each target point in
     * array-of-struct order.
     *
     * @param[in] F the charge density at each surface discretization node in
     * array-of-struct order.
     */
    void ComputePotential(sctl::Vector<Real>& U, const sctl::Vector<Real>& F) const;

    /**
     * Evaluate only the single-layer potential.
     *
     * @param[out] U the potential computed at each target point in
     * array-of-struct order.
     *
     * @param[in] F the charge density at each surface discretization node in
     * array-of-struct order.
     */
    void ComputeSL(sctl::Vector<Real>& U, const sctl::Vector<Real>& F) const;

    /**
     * Evaluate only the double-layer potential.
     *
     * @param[out] U the potential computed at each target point in
     * array-of-struct order.
     *
     * @param[in] F the charge density at each surface discretization node in
     * array-of-struct order.
     */
    void ComputeDL(sctl::Vector<Real>& U, const sctl::Vector<Real>& F) const;


    /**
     * Scale input vector by sqrt of the area of the element.
     * TODO: replace by sqrt of surface quadrature weights (not sure if it makes a difference though)
     */
    void SqrtScaling(sctl::Vector<Real>& U) const;

    /**
     * Scale input vector by inv-sqrt of the area of the element.
     * TODO: replace by inv-sqrt of surface quadrature weights (not sure if it makes a difference though)
     */
    void InvSqrtScaling(sctl::Vector<Real>& U) const;


  private:

    // In 3-periodic, this allows adding a uniform volume potential to balance the total force density on the surface.
    static void stokes_sl_volpot(sctl::Matrix<Real>& U, const sctl::Vector<Real>& X);

    const sctl::Stokes3D_FxU ker_FxU;
    const sctl::Stokes3D_DxU ker_DxU;
    const sctl::Stokes3D_FxUP ker_FxUP;
    const sctl::Stokes3D_FSxU ker_FSxU;

    const sctl::Comm comm_;
    const Real SL_scal_, DL_scal_;
    sctl::BoundaryIntegralOp<Real, sctl::Stokes3D_FxU> LayerPotenSL;
    sctl::BoundaryIntegralOp<Real, sctl::Stokes3D_DxU> LayerPotenDL;
};

#include <stokes_bio.cpp>

#endif 