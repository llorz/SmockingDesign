#if __INTELLISENSE__
#undef __ARM_NEON
#undef __ARM_NEON__
#endif

#include "../include/Utils.h"

#include <omp.h>

#include <Eigen/Eigen>

namespace SparseAD {

EnergyFunc sparse_func(int num,
                                         SparseEnergyFunc<Dual> delegate,
                                         bool project_hessian) {
  return [num, delegate,
          project_hessian](const TGenericVariableFactory<Dual>& vars) -> Dual::Tup {

double f = 0.0;
    Eigen::VectorXd grad = Eigen::VectorXd::Zero(vars.num_vars());
    std::vector<Eigen::Triplet<double>> triplets;
// Parallel compute all the values.
#pragma omp declare reduction (merge : std::vector<Eigen::Triplet<double>> : omp_out.insert(omp_out.end(), omp_in.begin(), omp_in.end()))
#pragma omp parallel for schedule(static) reduction(+: f) reduction(merge: triplets)
    for (int i = 0; i < num; i++) {
      Dual val = delegate(i, vars);
      if (project_hessian) {
        val.projectHessian(); 
      }
      f += val.val();

      for (const auto& [row, col, val2] : val.grad()) {
        #pragma omp atomic
        grad(row) += val2;
      }
      for (const auto& [row, col, val2] : val.hessian()) {
        if (row >= col) {
          triplets.push_back(Eigen::Triplet<double>(row, col, val2));
        }
      }
    }

    Eigen::SparseMatrix<double> hessian(vars.num_vars(), vars.num_vars());
    hessian.setFromTriplets(triplets.begin(), triplets.end());
    return {f, grad, hessian};
  };
}

GenericEnergyFunc<double> val_func(
    int num, SparseEnergyFunc<double> delegate) {
  return [num, delegate](const TGenericVariableFactory<double>& vars) {
  double res = 0.0;
// Parallel compute all the values.
#pragma omp parallel for schedule(static) reduction(+: res)
    for (int i = 0; i < num; i++) {
      res += delegate(i, vars);
    }
    return res;
  };
}

std::tuple<double, Eigen::VectorXd, Eigen::SparseMatrix<double>>
parallel_reduce(int n, const TGenericVariableFactory<Dual>& x, const std::function<DualOrLocalDual(int)>& func, bool proj_psd) {
  double f = 0.0;
  Eigen::VectorXd grad = Eigen::VectorXd::Zero(x.num_vars());
  std::vector<Eigen::Triplet<double>> triplets;
  #pragma omp parallel for schedule(static) reduction(+: f) reduction(merge: triplets) num_threads(omp_get_max_threads() - 1)
  for (int i = 0; i < n; i++) {
    DualOrLocalDual result = func(i);
    // If the result is dual.
    if (result.dual.has_value()) {
      Dual& val = result.dual.value();
      if (proj_psd) {
        val.projectHessian(); 
      }
      f += val.val();

      for (const auto& [row, col, val2] : val.grad()) {
        #pragma omp atomic
        grad(row) += val2;
      }
      for (const auto& [row, col, val2] : val.hessian()) {
        triplets.push_back(Eigen::Triplet<double>(row, col, val2));
      }
    // Otherwise, if it's a local dual.
    } else {
      LocalResult& res = result.local_dual.value();
      if (proj_psd) {
        res.hessian = project_hessian(res.hessian);
      }
      f += res.f;
      for (int i = 0; i < res.local_to_global.size(); i++) {
        #pragma omp atomic
        grad(res.local_to_global[i]) += res.grad(i);
      }
      for (int i = 0; i < res.local_to_global.size(); i++) {
        for (int j = 0; j < res.local_to_global.size(); j++) {
          triplets.push_back(Eigen::Triplet<double>(
            res.local_to_global[i], 
            res.local_to_global[j],
            res.hessian(i, j)));
        }
      }
    }
  }
  Eigen::SparseMatrix<double> hessian(x.num_vars(), x.num_vars());
  hessian.setFromTriplets(triplets.begin(), triplets.end());
  return {f, grad, hessian};
}

void parallel_reduce(int n, Dual::Tup& cur,
                const std::function<DualOrLocalDual(int)>& func,
                bool project_hessian) {

                }

double parallel_reduce(int n, const TGenericVariableFactory<double>& x, const std::function<double(int)>& func) {
  double f = 0.0;
  #pragma omp parallel for schedule(static) reduction(+: f) num_threads(omp_get_max_threads() - 1)
  for (int i = 0; i < n; i++) {
    f += func(i);
  }
  return f;
}

void parallel_reduce(int n, double& cur,
                       const std::function<double(int)>& func) {

                       }

}  // namespace SparseAD

