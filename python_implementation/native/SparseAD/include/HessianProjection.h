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
} // namespace SparseAD
