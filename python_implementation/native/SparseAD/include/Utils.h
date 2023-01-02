#pragma once
#include <omp.h>

#include <Eigen/Eigen>
#include <optional>
#include <unordered_map>
#include <vector>

#include "Dual.h"
#include "DualFactory.h"
#include "TDenseDual.h"

#pragma omp declare reduction (merge : std::vector<Eigen::Triplet<double>> : omp_out.insert(omp_out.end(), omp_in.begin(), omp_in.end()))
#define PARALLEL_ADD(x, i, num, code) SparseAD::parallel_reduce(num, x, [&](int i) -> typename SparseAD::typeMap<T>::type { \
code \
});

namespace SparseAD {

using Tup = std::tuple<double, Eigen::VectorXd, Eigen::MatrixXd>;
template <typename T>
using SparseEnergyFunc =
    std::function<T(int index, const TGenericVariableFactory<T>&)>;

using EnergyFunc =
    std::function<Dual::Tup(const TGenericVariableFactory<Dual>&)>;
template <typename T>
using GenericEnergyFunc = std::function<T(const TGenericVariableFactory<T>&)>;

EnergyFunc sparse_func(int num, SparseEnergyFunc<Dual> delegate,
                       bool project_hessian = true);

GenericEnergyFunc<double> val_func(int num, SparseEnergyFunc<double> delegate);

struct LocalResult {
  double f;
  Eigen::VectorXd grad;
  Eigen::MatrixXd hessian;
  std::vector<int> local_to_global;
};

template <int k>
class LocalDualFactory {
 public:
  LocalDualFactory(const TGenericVariableFactory<Dual>& other)
      : cur(other.current()) {
    local_to_global.reserve(k);
  }

  TDenseDual<k> operator()(int i) {
    auto local_index = global_to_local.find(i);
    if (local_index == global_to_local.end()) {
      assert(local_to_global.size() < k);
      global_to_local[i] = local_to_global.size();
      local_to_global.push_back(i);
      return TDenseDual<k>(cur(i), local_to_global.size() - 1);
    }
    return TDenseDual<k>(cur(i), local_index->second);
  }

  TDenseDual<k> operator()(int i, int j) {
    return operator()(i + j * cur.rows());
  }

  int num_vars() const { return cur.size(); }

  Eigen::VectorX<TDenseDual<k>> row(int i) {
    Eigen::VectorX<TDenseDual<k>> result(cur.cols());
    for (int j = 0; j < cur.cols(); j++) {
      result(j) = operator()(i, j);
    }
    return result;
  }

  // LocalResult get(const TDenseDual<k>& d) {
  //   const auto& [local_f, local_grad, local_hessian] = (Tup)d;
  //   LocalResult res {.f = local_f, .grad = local_grad, .hessian = local_hessian,
  //   .local_to_global = local_to_global};
  //   return res;
  // }
  LocalResult get(TDenseDual<k>&& d) {
    LocalResult res {.f = d.val(), .grad = std::move(d.grad()),
     .hessian = std::move(d.hessian()),
    .local_to_global = std::move(local_to_global)};
    return res;
  }

  const Eigen::MatrixXd& cur;
  std::vector<int> local_to_global;
  std::unordered_map<int, int> global_to_local;
};

template <int k>
using LocalEnergyFunction =
    std::function<TDenseDual<k>(int index, LocalDualFactory<k>&)>;

template <int k>
EnergyFunc sparse_func(int num, LocalEnergyFunction<k> delegate,
                       bool project_hessian = true) {
  return [num, delegate, project_hessian](
             const TGenericVariableFactory<Dual>& vars) -> Dual::Tup {
    int num_vars = vars.num_vars();
    double f = 0.0;
    Eigen::VectorXd grad = Eigen::VectorXd::Zero(num_vars);
    std::vector<Eigen::Triplet<double>> triplets;
    triplets.reserve(k * k * num);
// Parallel compute all the values.
#pragma omp parallel for schedule(static) reduction(+: f) reduction(merge: triplets) num_threads(omp_get_max_threads() - 1)
    for (int i = 0; i < num; i++) {
      LocalDualFactory<k> local_vars(vars);
      TDenseDual<k> res = delegate(i, local_vars);
      if (project_hessian) {
        res.projectHessian();
      }

      // Aggregate the result.
      const auto& [local_f, local_grad, local_hessian] = (Tup)res;
      f += local_f;

      for (int j = 0; j < local_vars.local_to_global.size(); j++) {
#pragma omp atomic
        grad(local_vars.local_to_global[j]) += local_grad(j);
      }
      for (int j = 0; j < local_vars.local_to_global.size(); j++) {
        for (int h = 0; h < local_vars.local_to_global.size(); h++) {
          triplets.push_back(Eigen::Triplet<double>(
              local_vars.local_to_global[j], local_vars.local_to_global[h],
              local_hessian(j, h)));
        }
      }
    }
    Eigen::SparseMatrix<double> hessian(num_vars, num_vars);
    hessian.setFromTriplets(triplets.begin(), triplets.end());
    return {f, grad, hessian};
  };
};

struct DualOrLocalDual {
  DualOrLocalDual(const Dual& dual) : dual(std::make_optional(dual)) {}
  DualOrLocalDual(Dual&& dual) : dual(std::make_optional(std::move(dual))) {}
  DualOrLocalDual(const LocalResult& local_dual)
      : local_dual(std::make_optional(local_dual)) {}
  DualOrLocalDual(LocalResult&& local_dual)
      : local_dual(std::make_optional(std::move(local_dual))) {}
  std::optional<Dual> dual;
  std::optional<LocalResult> local_dual;
};

template <class T>
struct typeMap {};
template<>
struct typeMap<Dual> { using type = DualOrLocalDual; };
template<>
struct typeMap<double> { using type = double; };

std::tuple<double, Eigen::VectorXd, Eigen::SparseMatrix<double>>
parallel_reduce(int n, const TGenericVariableFactory<Dual>& f,
                const std::function<DualOrLocalDual(int)>& func,
                bool project_hessian = true);

void parallel_reduce(int n, Dual::Tup& cur,
                const std::function<DualOrLocalDual(int)>& func,
                bool project_hessian = true);

double parallel_reduce(int n, const TGenericVariableFactory<double>& f,
                       const std::function<double(int)>& func);

void parallel_reduce(int n, double& cur,
                       const std::function<double(int)>& func);

}  // namespace SparseAD