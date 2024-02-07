#if __INTELLISENSE__
#undef __ARM_NEON
#undef __ARM_NEON__
#endif

#include "ProjectHessian.h"

#include <iostream>
#include <set>
#include <tuple>
#include <unordered_map>
#include <vector>

#include "VectorMap.h"

namespace Optiz {

std::tuple<
    std::unordered_map<int, int>,
    std::vector<int>> static find_referenced_indices(const SelfAdjointMapMatrix&
                                                         mat) {
  // std::set<int> res;
  std::unordered_map<int, int> sp_to_dense;
  int new_index = 0;
  for (const auto& val : mat.get_values()) {
    int row = val.first / mat.cols(), col = val.first % mat.cols();
    if (sp_to_dense.try_emplace(row, new_index).second) {
      new_index++;
    }
    if (sp_to_dense.try_emplace(col, new_index).second) {
      new_index++;
    }
  }
  // Create a mapping from sparse indices to dense indices.
  std::vector<int> dense_to_sp(new_index);
  for (auto& el : sp_to_dense) {
    dense_to_sp[el.second] = el.first;
  }
  return {sp_to_dense, dense_to_sp};
}

static std::tuple<Eigen::MatrixX<double>, std::vector<int>> sparse_to_dense(
    const SelfAdjointMapMatrix& mat) {
  auto [sp_to_dense, dense_to_sp] = find_referenced_indices(mat);

  Eigen::MatrixX<double> res(sp_to_dense.size(), sp_to_dense.size());
  for (const auto& val : mat.get_values()) {
    int row = val.first / mat.cols(), col = val.first % mat.cols();
    int r = sp_to_dense[row], c = sp_to_dense[col];
    res(r, c) = val.second;
    res(c, r) = val.second;
  }
  return {res, dense_to_sp};
}

static SelfAdjointMapMatrix dense_to_sparse_selfadj(
    const Eigen::MatrixX<double>& dense, const std::vector<int>& dense_to_sp,
    int n) {
  std::vector<Eigen::Triplet<double>> triplets;
  SelfAdjointMapMatrix res(n);
  for (int i = 0; i < dense.rows(); i++) {
    for (int j = 0; j <= i; j++) {
      if (dense_to_sp[i] > dense_to_sp[j]) {
        res(dense_to_sp[i], dense_to_sp[j]) = dense(i, j);
      } else {
        res(dense_to_sp[j], dense_to_sp[i]) = dense(i, j);
      }
    }
  }
  return res;
}

static bool is_self_adjoint_positive_diagonaly_dominant(
    const Eigen::MatrixX<double>& dense) {
  for (int i = 0; i < dense.rows(); i++) {
    if (dense(i, i) <= 0) {
      return false;
    }
  }
  for (int i = 0; i < dense.rows(); i++) {
    double non_diagonal_sum = 0.0;
    for (int j = 0; j < dense.cols(); j++) {
      if (i != j) {
        non_diagonal_sum += abs((i > j) ? dense(i, j) : dense(j, i));
      }
    }
    if (dense(i, i) < non_diagonal_sum + 1e-5) {
      return false;
    }
  }
  return true;
}

Eigen::MatrixXd project_hessian(const Eigen::MatrixXd& hessian) {
  if (is_self_adjoint_positive_diagonaly_dominant(hessian)) {
    return hessian;
  }
  Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> eig(hessian);
  Eigen::VectorXd eigs = eig.eigenvalues();
  bool all_positive = true;
  for (int i = 0; i < hessian.rows(); ++i) {
    if (eigs(i) < 1e-9) {
      eigs(i) = 1e-9;
      all_positive = false;
    }
  }
  if (all_positive) {
    return hessian;
  }
  Eigen::MatrixX<double> projected =
      eig.eigenvectors() * eigs.asDiagonal() * eig.eigenvectors().transpose();
  return projected;
}

SelfAdjointMapMatrix project_hessian(const SelfAdjointMapMatrix& hessian) {
  const auto [dense, dense_to_sp] = sparse_to_dense(hessian);
  if (is_self_adjoint_positive_diagonaly_dominant(dense)) {
    return hessian;
  }
  Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> eig(dense);
  Eigen::VectorXd eigs = eig.eigenvalues();
  bool all_positive = true;
  for (int i = 0; i < dense.rows(); ++i) {
    if (eigs(i) < 1e-9) {
      eigs(i) = 1e-9;
      all_positive = false;
    }
  }
  if (all_positive) {
    return hessian;
  }
  Eigen::MatrixX<double> projected =
      eig.eigenvectors() * eigs.asDiagonal() * eig.eigenvectors().transpose();
  return dense_to_sparse_selfadj(projected, dense_to_sp, hessian.rows());
}

std::pair<Eigen::MatrixXd, std::vector<int>> project_sparse_hessian(
    const SelfAdjointMapMatrix& hessian) {
  const auto [dense, dense_to_sp] = sparse_to_dense(hessian);
  Eigen::MatrixXd projected = project_hessian(dense);
  return {projected, dense_to_sp};
}

}  // namespace Optiz
