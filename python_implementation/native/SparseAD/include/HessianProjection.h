#pragma once
#include <Eigen/Eigen>
#include <tuple>
#include <vector>
#include <unordered_map>
#include <set>
#include "SparseMapMatrix.h"

namespace SparseAD {
  
  // std::tuple<std::unordered_map<int, int>, std::vector<int>>
  //  find_referenced_indices(const Eigen::SparseMatrix<double>& mat);

  // std::tuple<Eigen::MatrixX<double>, std::vector<int>>
  //  sparse_to_dense(const Eigen::SparseMatrix<double>& mat);

  //  Eigen::SparseMatrix<double> dense_to_sparse(const Eigen::MatrixX<double>& dense,
  //  const std::vector<int>& dense_to_sp, int n);

  // bool is_positive_diagonal_dominant(const Eigen::MatrixX<double>& dense);

  // Eigen::SparseMatrix<double> project_hessian(const Eigen::SparseMatrix<double>& hessian);


  std::tuple<std::unordered_map<int, int>, std::vector<int>>
   find_referenced_indices(const SparseMapMatrix& mat);

  std::tuple<Eigen::MatrixX<double>, std::vector<int>>
   sparse_to_dense(const SparseMapMatrix& mat);

   SparseMapMatrix dense_to_sparse(const Eigen::MatrixX<double>& dense,
   const std::vector<int>& dense_to_sp, int n);

  bool is_positive_diagonal_dominant(const Eigen::MatrixX<double>& dense);

  SparseMapMatrix project_hessian(const SparseMapMatrix& hessian);

  Eigen::MatrixXd project_hessian(const Eigen::MatrixXd& hessian);

  // template<int k>
  // Eigen::Matrix<double, k, k> project_hessian(const Eigen::Matrix<double, k, k>& hessian) {
  //   if (is_positive_diagonal_dominant(hessian)) {
  //     return hessian;
  //   }
  //   Eigen::SelfAdjointEigenSolver<Eigen::MatrixX<double>> eig(hessian);
  //   Eigen::MatrixX<double> D = eig.eigenvalues().asDiagonal();
  //   bool all_positive = true;
  //   for (int i = 0; i < hessian.rows(); ++i) {
  //       if (D(i, i) < 1e-6) {
  //           D(i, i) = 1e-6;
  //           all_positive = false;
  //       }
  //   }
  //   if (all_positive) {
  //     return hessian;
  //   }
  //   Eigen::MatrixX<double> projected = eig.eigenvectors() * D * eig.eigenvectors().transpose();
  //   return projected;
  // }
} // namespace SparseAD
