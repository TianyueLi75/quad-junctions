// =============================================================================
// stokes_bio.cpp
//
// Template implementation of the StokesBIO class declared in stokes_bio.hpp.
// Not a standalone translation unit: it is included at the bottom of
// stokes_bio.hpp. See that header for the operator definition and usage.
// =============================================================================

// Single-layer volume-potential correction supplied to the FMM kernel.
template <class Real> void StokesBIO<Real>::stokes_sl_volpot(sctl::Matrix<Real>& U, const sctl::Vector<Real>& X) {
  const sctl::Long N = X.Dim() / 3;
  SCTL_ASSERT(X.Dim() == N * 3);
  if (U.Dim(0)!=3 || U.Dim(1)!=N*3) U.ReInit(3, N*3);
  for (sctl::Long i = 0; i < N; i++) {
    const auto x = X.begin() + i*3;
    const Real rx_2 = x[1]*x[1] + x[2]*x[2];
    const Real ry_2 = x[0]*x[0] + x[2]*x[2];
    const Real rz_2 = x[0]*x[0] + x[1]*x[1];
    U[0][i*3+0] = -rx_2/4; U[0][i*3+1] =       0; U[0][i*3+2] =       0;
    U[1][i*3+0] =       0; U[1][i*3+1] = -ry_2/4; U[1][i*3+2] =       0;
    U[2][i*3+0] =       0; U[2][i*3+1] =       0; U[2][i*3+2] = -rz_2/4;
  }
}

// Construct the combined-field operator with single- and double-layer weights.
template <class Real> StokesBIO<Real>::StokesBIO(const Real SL_scal, const Real DL_scal, const sctl::Comm comm)
  : comm_(comm), SL_scal_(SL_scal), DL_scal_(DL_scal), LayerPotenSL(ker_FxU, false, comm), LayerPotenDL(ker_DxU, false, comm) {
  LayerPotenSL.SetAccuracy(1e-14);
  LayerPotenDL.SetAccuracy(1e-14);
  LayerPotenSL.SetFMMKer(ker_FxU, ker_FxU, ker_FxU, ker_FxU, ker_FxU, ker_FxU, ker_FxU, ker_FxU, stokes_sl_volpot);
  LayerPotenDL.SetFMMKer(ker_DxU, ker_DxU, ker_DxU, ker_FSxU, ker_FSxU, ker_FSxU, ker_FxU, ker_FxU);
};

// Set the periodicity direction and period length of both layer potentials.
template <class Real> void StokesBIO<Real>::SetPeriodicity(sctl::Periodicity periodicity, Real period_length) {
  LayerPotenSL.SetPeriodicity(periodicity, period_length);
  LayerPotenDL.SetPeriodicity(periodicity, period_length);
}

// Set the quadrature accuracy of both layer potentials.
template <class Real> void StokesBIO<Real>::SetAccuracy(Real tol) {
  LayerPotenSL.SetAccuracy(tol);
  LayerPotenDL.SetAccuracy(tol);
}

template <class Real> template <class ElemLstType> void StokesBIO<Real>::AddElemList(const ElemLstType& elem_lst, const std::string& name, bool sl, bool dl) {
    
  if (sl) {
    LayerPotenSL.AddElemList(elem_lst, name);
  }
  if (dl) {
    LayerPotenDL.AddElemList(elem_lst, name);
  }
}

template <class Real> template <class ElemLstType> const ElemLstType& StokesBIO<Real>::GetElemList(const std::string& name) const {
  return LayerPotenDL.template GetElemList<ElemLstType>(name);
}

template <class Real> void StokesBIO<Real>::DeleteElemList(const std::string& name) {
  LayerPotenSL.DeleteElemList(name);
  LayerPotenDL.DeleteElemList(name);
}

template <class Real> template <class ElemLstType> void StokesBIO<Real>::DeleteElemList() {
  LayerPotenSL.template DeleteElemList<ElemLstType>();
  LayerPotenDL.template DeleteElemList<ElemLstType>();
}

template <class Real> void StokesBIO<Real>::SetTargetCoord(const sctl::Vector<Real>& Xtrg) {
  LayerPotenSL.SetTargetCoord(Xtrg);
  LayerPotenDL.SetTargetCoord(Xtrg);
}

template <class Real> void StokesBIO<Real>::SetTargetNormal(const sctl::Vector<Real>& Xn_trg) {
  LayerPotenSL.SetTargetNormal(Xn_trg);
  LayerPotenDL.SetTargetNormal(Xn_trg);
}

template <class Real> sctl::Long StokesBIO<Real>::Dim(sctl::Integer k) const {
  return LayerPotenDL.Dim(k);
}

template <class Real> void StokesBIO<Real>::Setup() const {
  if (SL_scal_) LayerPotenSL.Setup();
  if (DL_scal_) LayerPotenDL.Setup();
}

template <class Real> void StokesBIO<Real>::ClearSetup() const {
  LayerPotenSL.ClearSetup();
  LayerPotenDL.ClearSetup();
}

// Apply the combined-field operator: U = SL_scal * S[F] + DL_scal * D[F].
template <class Real> void StokesBIO<Real>::ComputePotential(sctl::Vector<Real>& U, const sctl::Vector<Real>& F) const {
  sctl::Vector<Real> Us, Ud;
  if (SL_scal_ && LayerPotenSL.Dim(0)) {
    if (LayerPotenSL.Dim(0) != F.Dim()) {
      sctl::Vector<Real> subF(LayerPotenSL.Dim(0), (sctl::Iterator<Real>) F.begin(), true);
      LayerPotenSL.ComputePotential(Us, subF);
    } else {
      LayerPotenSL.ComputePotential(Us, F);
    }
  } else {
    Us.ReInit(LayerPotenSL.Dim(1));
    Us.SetZero();
  }
  if (DL_scal_ && LayerPotenDL.Dim(0)) {
    LayerPotenDL.ComputePotential(Ud, F);
  } else {
    Ud.ReInit(LayerPotenDL.Dim(1));
    Ud.SetZero();
  }

  if (SL_scal_ && DL_scal_) U = Us * SL_scal_ + Ud * DL_scal_;
  else if (SL_scal_) U = Us * SL_scal_;
  else if (DL_scal_) U = Ud * DL_scal_;
  else U.SetZero();
}


template <class Real> void StokesBIO<Real>::ComputeSL(sctl::Vector<Real>& U, const sctl::Vector<Real>& F) const {
  LayerPotenSL.ComputePotential(U, F);
}

template <class Real> void StokesBIO<Real>::ComputeDL(sctl::Vector<Real>& U, const sctl::Vector<Real>& F) const {
  LayerPotenDL.ComputePotential(U, F);
}

template <class Real> void StokesBIO<Real>::SqrtScaling(sctl::Vector<Real>& U) const {
  LayerPotenSL.SqrtScaling(U);
}

template <class Real> void StokesBIO<Real>::InvSqrtScaling(sctl::Vector<Real>& U) const {
  LayerPotenSL.InvSqrtScaling(U);
}