#if __INTELLISENSE__
#undef __ARM_NEON
#undef __ARM_NEON__
#endif

#include "Utils.h"

#include <Eigen/Eigen>

namespace Optiz {

EnergyFunc element_func(int num, SparseEnergyFunc<Var> delegate,
                        bool project_hessian) {
  return
      [num, delegate, project_hessian](const TGenericVariableFactory<Var>& vars)
          -> SparseValueAndDerivatives {
        double f = 0.0;
        Eigen::VectorXd grad = Eigen::VectorXd::Zero(vars.num_vars());
        std::vector<Eigen::Triplet<double>> triplets;
// Parallel compute all the values.
#pragma omp declare reduction(                                        \
        merge : std::vector<Eigen::Triplet<double>> : omp_out.insert( \
                omp_out.end(), omp_in.begin(), omp_in.end()))
#pragma omp parallel for schedule(static) reduction(+ : f) \
    reduction(merge : triplets)
        for (int i = 0; i < num; i++) {
          Var val = delegate(i, vars);
          f += val.val();

          for (const auto& [row, val2] : val.grad()) {
#pragma omp atomic
            grad(row) += val2;
          }

          if (project_hessian) {
            // val.projectHessian();
            auto [dense, inds] = project_sparse_hessian(val.hessian());
            for (int i = 0; i < dense.rows(); i++) {
              for (int j = 0; j <= i; j++) {
                if (inds[i] >= inds[j]) {
                  triplets.push_back(
                      Eigen::Triplet<double>(inds[i], inds[j], dense(i, j)));
                } else {
                  triplets.push_back(
                      Eigen::Triplet<double>(inds[j], inds[i], dense(i, j)));
                }
              }
            }
          } else {
            for (const auto& [ind, val2] : val.hessian()) {
              long row = ind / val.hessian().cols(),
                   col = ind % val.hessian().cols();
              triplets.push_back(Eigen::Triplet<double>(row, col, val2));
            }
          }
        }

        return {f, grad, triplets};
      };
}

GenericEnergyFunc<double> val_func(int num, SparseEnergyFunc<double> delegate) {
  return [num, delegate](const TGenericVariableFactory<double>& vars) {
    double res = 0.0;
// Parallel compute all the values.
#pragma omp parallel for schedule(static) reduction(+ : res)
    for (int i = 0; i < num; i++) {
      res += delegate(i, vars);
    }
    return res;
  };
}

std::tuple<double, Eigen::VectorXd, Eigen::SparseMatrix<double>>
parallel_reduce(int n, const TGenericVariableFactory<Var>& x,
                const std::function<VarOrLocalVar(int)>& func, bool proj_psd) {
  double f = 0.0;
  Eigen::VectorXd grad = Eigen::VectorXd::Zero(x.num_vars());
  std::vector<Eigen::Triplet<double>> triplets;
#pragma omp parallel for schedule(static) reduction(+ : f) \
    reduction(merge : triplets) num_threads(omp_get_max_threads() - 1)
  for (int i = 0; i < n; i++) {
    VarOrLocalVar result = func(i);
    // If the result is var.
    if (result.var.has_value()) {
      Var& val = result.var.value();
      if (proj_psd) {
        val.projectHessian();
      }
      f += val.val();

      // for (const auto& [row, col, val2] : val.grad()) {
      //   #pragma omp atomic
      //   grad(row) += val2;
      // }
      // for (const auto& [row, col, val2] : val.hessian()) {
      //   triplets.push_back(Eigen::Triplet<double>(row, col, val2));
      // }
      for (const auto& [row, val2] : val.grad()) {
#pragma omp atomic
        grad(row) += val2;
      }
      for (const auto& [ind, val2] : val.hessian()) {
        long row = ind / val.hessian().cols(), col = ind % val.hessian().cols();
        triplets.push_back(Eigen::Triplet<double>(row, col, val2));
      }
      // Otherwise, if it's a local var.
    } else {
      LocalResult& res = result.local_var.value();
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
          triplets.push_back(Eigen::Triplet<double>(res.local_to_global[i],
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

void parallel_reduce(int n, Var::Tup& cur,
                     const std::function<VarOrLocalVar(int)>& func,
                     bool project_hessian) {}

double parallel_reduce(int n, const TGenericVariableFactory<double>& x,
                       const std::function<double(int)>& func) {
  double f = 0.0;
#pragma omp parallel for schedule(static) reduction(+ : f) \
    num_threads(omp_get_max_threads() - 1)
  for (int i = 0; i < n; i++) {
    f += func(i);
  }
  return f;
}

void parallel_reduce(int n, double& cur,
                     const std::function<double(int)>& func) {}

void write_matrix_to_file(const Eigen::MatrixXd& mat,
                          const std::string& file_name) {
  std::ofstream file(file_name);
  file << mat;
  file.close();
}

Eigen::MatrixXd read_matrix_from_file(const std::string& file_name) {
  std::ifstream file(file_name);

  int cols = 0;
  std::string line;
  std::vector<std::vector<double>> mat;
  while (std::getline(file, line)) {
    std::stringstream ss(line);
    std::vector<double> row;
    double val;
    while (ss >> val) {
      row.push_back(val);
    }
    mat.push_back(row);
  }
  Eigen::MatrixXd res(mat.size(), mat[0].size());
  for (int i = 0; i < mat.size(); i++) {
    for (int j = 0; j < mat[0].size(); j++) {
      res(i, j) = mat[i][j];
    }
  }

  file.close();
  return res;
}

}  // namespace Optiz
