#pragma once
#include "Var.h"
#include "TDenseVar.h"
#include "Utils.h"

namespace Optiz {

template<typename T>
inline T sqr(const T& x ) {
  return x*x;
}

// Project psd.
template<typename T>
inline T project_psd(const T& x) {
  return x;
}

inline Var project_psd(const Var& x) {
  Var res = Var(x).projectHessian();
  return res;
}
inline Var project_psd(Var&& x) {
  x.projectHessian();
  return x;
}
template<int k>
inline TDenseVar<k> project_psd(const TDenseVar<k>& x) {
  TDenseVar<k> res = TDenseVar<k>(x).projectHessian();
  return res;
}
template<int k>
inline TDenseVar<k> project_psd(TDenseVar<k>&& x) {
  x.projectHessian();
  return x;
}

// Val
template<typename T>
inline T val(const T& x) {
  return x;
}

inline double val(const Var& x) {
  return x.val();
}

template<int k>
inline double val(const TDenseVar<k>& x) {
  return x.val();
}

template<int k>
LocalVarFactory<k> get_local_factory(const TGenericVariableFactory<Var>& other) {
  return LocalVarFactory<k>(other);
}

template<int k>
const ValFactory<double>& get_local_factory(const ValFactory<double>& fac) {
  return fac;
}

}