#pragma once
#include "Dual.h"
#include "TDenseDual.h"
#include "Utils.h"

namespace SparseAD {

template<typename T>
inline T sqr(const T& x ) {
  return x*x;
}

// Project psd.
template<typename T>
inline T project_psd(const T& x) {
  return x;
}

inline Dual project_psd(const Dual& x) {
  Dual res = Dual(x).projectHessian();
  return res;
}
inline Dual project_psd(Dual&& x) {
  x.projectHessian();
  return x;
}
template<int k>
inline TDenseDual<k> project_psd(const TDenseDual<k>& x) {
  TDenseDual<k> res = TDenseDual<k>(x).projectHessian();
  return res;
}
template<int k>
inline TDenseDual<k> project_psd(TDenseDual<k>&& x) {
  x.projectHessian();
  return x;
}

// Val
template<typename T>
inline T val(const T& x) {
  return x;
}

inline double val(const Dual& x) {
  return x.val();
}

template<int k>
inline double val(const TDenseDual<k>& x) {
  return x.val();
}

template<int k>
LocalDualFactory<k> get_local_factory(const TGenericVariableFactory<Dual>& other) {
  return LocalDualFactory<k>(other);
}

template<int k>
const ValFactory<double>& get_local_factory(const ValFactory<double>& fac) {
  return fac;
}

}