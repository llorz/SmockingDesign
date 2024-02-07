#pragma once
#ifdef _OPENMP
#include <omp.h>
#endif

#include <Eigen/Eigen>
#include <fstream>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

#include "Complex.h"
#include "Var.h"
#include "VarFactory.h"
#include "TDenseVar.h"

#pragma omp declare reduction(                                        \
        merge : std::vector<Eigen::Triplet<double>> : omp_out.insert( \
                omp_out.end(), omp_in.begin(), omp_in.end()))
#define PARALLEL_ADD(x, i, num, code) \
  Optiz::parallel_reduce(num, x,      \
                         [&](int i) -> Optiz::typeMap<T>::type { code });

namespace Optiz {

using DenseValueAndDerivatives =
    std::tuple<double, Eigen::VectorXd, Eigen::MatrixXd>;
using SparseValueAndDerivatives =
    std::tuple<double, Eigen::VectorXd, std::vector<Eigen::Triplet<double>>>;

template <typename T>
using SparseEnergyFunc =
    std::function<T(int index, const TGenericVariableFactory<T>&)>;
using SparseVarEnergyFunc =
    std::function<Var(int index, const TGenericVariableFactory<Var>&)>;
using EnergyFunc = std::function<SparseValueAndDerivatives(
    const TGenericVariableFactory<Var>&)>;
template <typename T>
using GenericEnergyFunc = std::function<T(const TGenericVariableFactory<T>&)>;

EnergyFunc element_func(int num, SparseEnergyFunc<Var> delegate,
                        bool project_hessian = true);

GenericEnergyFunc<double> val_func(int num, SparseEnergyFunc<double> delegate);

struct LocalResult {
  double f;
  Eigen::VectorXd grad;
  Eigen::MatrixXd hessian;
  std::vector<int> local_to_global;
};

template <int k>
class LocalVarFactory {
 public:
  LocalVarFactory(const Eigen::Map<const Eigen::MatrixXd>& other)
      : cur(other) {
    local_to_global.reserve(k);
  }

  using Scalar = TDenseVar<k>;

  TDenseVar<k> operator()(int i) {
    auto local_index =
        std::find(local_to_global.begin(), local_to_global.end(), i);
    if (local_index == local_to_global.end()) {
      // auto local_index = global_to_local.find(i);
      // if (local_index == global_to_local.end()) {
      assert(local_to_global.size() < k);
      // global_to_local[i] = local_to_global.size();
      local_to_global.push_back(i);
      return TDenseVar<k>(cur(i), local_to_global.size() - 1);
    }
    return TDenseVar<k>(cur(i), local_index - local_to_global.begin());
  }

  TDenseVar<k> operator()(int i, int j) {
    return operator()(i + j * cur.rows());
  }

  int num_vars() const { return cur.size(); }

  Eigen::VectorX<TDenseVar<k>> row(int i) {
    Eigen::VectorX<TDenseVar<k>> result(cur.cols());
    for (int j = 0; j < cur.cols(); j++) {
      result(j) = operator()(i, j);
    }
    return result;
  }

  LocalResult get(const TDenseVar<k>& d) {
    const auto& [local_f, local_grad, local_hessian] =
        (DenseValueAndDerivatives)d;
    LocalResult res{.f = local_f,
                    .grad = local_grad,
                    .hessian = local_hessian,
                    .local_to_global = std::move(local_to_global)};
    return res;
  }
  LocalResult get(TDenseVar<k>&& d) {
    LocalResult res{.f = d.val(),
                    .grad = std::move(d.grad()),
                    .hessian = std::move(d.hessian()),
                    .local_to_global = std::move(local_to_global)};
    return res;
  }

  Eigen::Map<const Eigen::MatrixXd> cur;
  std::vector<int> local_to_global;
};

template <int k>
using LocalEnergyFunction =
    std::function<TDenseVar<k>(int index, LocalVarFactory<k>&)>;

template <int k>
EnergyFunc element_func(int num, LocalEnergyFunction<k> delegate,
                        bool project_hessian = true) {
  return [num, delegate,
          project_hessian](const TGenericVariableFactory<Var>& vars)
             -> SparseValueAndDerivatives {
    int num_vars = vars.num_vars();
    double f = 0.0;
    Eigen::VectorXd grad = Eigen::VectorXd::Zero(num_vars);
    std::vector<Eigen::Triplet<double>> triplets;
    triplets.reserve(k * k * num);
// Parallel compute all the values.
#pragma omp parallel for schedule(static) reduction(+ : f) \
    reduction(merge : triplets) num_threads(omp_get_max_threads() - 1)
    for (int i = 0; i < num; i++) {
      LocalVarFactory<k> local_vars(vars.current_mat());
      TDenseVar<k> res = delegate(i, local_vars);
      if (project_hessian) {
        res.projectHessian();
      }

      // Aggregate the result.
      const auto& [local_f, local_grad, local_hessian] =
          (DenseValueAndDerivatives)res;
      f += local_f;

      for (int j = 0; j < local_vars.local_to_global.size(); j++) {
#pragma omp atomic
        grad(local_vars.local_to_global[j]) += local_grad(j);
      }
      for (int j = 0; j < local_vars.local_to_global.size(); j++) {
        for (int h = 0; h <= j; h++) {
          int gj = local_vars.local_to_global[j],
              gh = local_vars.local_to_global[h];
          double val = local_hessian(j, h);
          // Only fill the lower triangle of the hessian.
          if (gj <= gh) {
            triplets.push_back(Eigen::Triplet<double>(gh, gj, val));
          } else {
            triplets.push_back(Eigen::Triplet<double>(gj, gh, val));
          }
        }
      }
    }
    // Eigen::SparseMatrix<double> hessian(num_vars, num_vars);
    // hessian.setFromTriplets(triplets.begin(), triplets.end());
    return {f, grad, triplets};
  };
};

struct VarOrLocalVar {
  VarOrLocalVar(const Var& var) : var(std::make_optional(var)) {}
  VarOrLocalVar(Var&& var) : var(std::make_optional(std::move(var))) {}
  VarOrLocalVar(const LocalResult& local_var)
      : local_var(std::make_optional(local_var)) {}
  VarOrLocalVar(LocalResult&& local_var)
      : local_var(std::make_optional(std::move(local_var))) {}
  std::optional<Var> var;
  std::optional<LocalResult> local_var;
};

template <class T>
struct typeMap {};
template <>
struct typeMap<Var> {
  using type = VarOrLocalVar;
};
template <>
struct typeMap<double> {
  using type = double;
};

std::tuple<double, Eigen::VectorXd, Eigen::SparseMatrix<double>>
parallel_reduce(int n, const TGenericVariableFactory<Var>& f,
                const std::function<VarOrLocalVar(int)>& func,
                bool project_hessian = true);

void parallel_reduce(int n, Var::Tup& cur,
                     const std::function<VarOrLocalVar(int)>& func,
                     bool project_hessian = true);

double parallel_reduce(int n, const TGenericVariableFactory<double>& f,
                       const std::function<double(int)>& func);

void parallel_reduce(int n, double& cur,
                     const std::function<double(int)>& func);

void write_matrix_to_file(const Eigen::MatrixXd& mat,
                          const std::string& file_name);

Eigen::MatrixXd read_matrix_from_file(const std::string& file_name);

}  // namespace Optiz