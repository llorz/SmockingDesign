#if __INTELLISENSE__
#undef __ARM_NEON
#undef __ARM_NEON__
#endif

#include "../include/HessianProjection.h"

namespace SparseAD
{
  
// std::tuple<std::unordered_map<int, int>, std::vector<int>>
//    find_referenced_indices(const Eigen::SparseMatrix<double>& mat) {
//     std::set<int> res;
//     for (int k=0; k<mat.outerSize(); ++k) {
//       for (typename Eigen::SparseMatrix<double>::InnerIterator it(mat,k); it; ++it) {
//         res.insert(it.row());
//         res.insert(it.col());
//       }
//     }
//     // Create a mapping from sparse indices to dense indices.
//     std::unordered_map<int, int> sp_to_dense;
//     std::vector<int> dense_to_sp(res.size());
//     int new_index = 0;
//     for (int ind : res) {
//       dense_to_sp[new_index] = ind;
//       sp_to_dense[ind] = new_index++;
//     }
//     return {sp_to_dense, dense_to_sp};
//   }

//   std::tuple<Eigen::MatrixX<double>, std::vector<int>>
//    sparse_to_dense(const Eigen::SparseMatrix<double>& mat) {
//     auto [sp_to_dense, dense_to_sp] = find_referenced_indices(mat);

//     Eigen::MatrixX<double> res(sp_to_dense.size(), sp_to_dense.size());
//     std::vector<int> dense_to_sparse(sp_to_dense.size());
//     for (int k=0; k<mat.outerSize(); ++k) {
//       for (typename Eigen::SparseMatrix<double>::InnerIterator it(mat,k); it; ++it) {
//         res(sp_to_dense[it.row()], sp_to_dense[it.col()]) = it.value();
//       }
//     }
//     return {res, dense_to_sp};
//    }

//    Eigen::SparseMatrix<double> dense_to_sparse(const Eigen::MatrixX<double>& dense,
//    const std::vector<int>& dense_to_sp, int n) {
//     std::vector<Eigen::Triplet<double>> triplets;
//     for (int i = 0; i < dense.rows(); i++) {
//       for (int j = 0; j < dense.cols(); j++) {
//         triplets.push_back(Eigen::Triplet<double>(
//           dense_to_sp[i], dense_to_sp[j], dense(i, j)
//         ));
//       }
//     }
//     Eigen::SparseMatrix<double> res(n, n);
//     res.setFromTriplets(triplets.begin(), triplets.end());
//     return res;
//    }

//   bool is_positive_diagonal_dominant(const Eigen::MatrixX<double>& dense) {
//     for (int i = 0; i < dense.rows(); i++) {
//       double off_diag = 0.0;
//       for (int j = 0; j < dense.cols(); j++) {
//         if (i != j) { off_diag += abs(dense(i, j)); }
//       }
//       if (dense(i, i) < off_diag + 1e-5) { return false; }
//     }
//     return true;
//   }

//   Eigen::SparseMatrix<double> project_hessian(const Eigen::SparseMatrix<double>& hessian) {
//     const auto [dense, dense_to_sp] =  sparse_to_dense(hessian);
//     if (is_positive_diagonal_dominant(dense)) {
//       return hessian;
//     }
//     Eigen::SelfAdjointEigenSolver<Eigen::MatrixX<double>> eig(dense);
//     Eigen::MatrixX<double> D = eig.eigenvalues().asDiagonal();
//     bool all_positive = true;
//     for (int i = 0; i < dense.rows(); ++i) {
//         if (D(i, i) < 1e-6) {
//             D(i, i) = 1e-6;
//             all_positive = false;
//         }
//     }
//     if (all_positive) {
//       return hessian;
//     }
//     Eigen::MatrixX<double> projected = eig.eigenvectors() * D * eig.eigenvectors().transpose();
//     return dense_to_sparse(projected, dense_to_sp, hessian.rows());
//   }

std::tuple<std::unordered_map<int, int>, std::vector<int>>
   find_referenced_indices(const SparseMapMatrix& mat) {
    std::set<int> res;
    for (const auto& [i, row] : mat.vals()) {
      res.insert(i);
      for (const auto& [j, val] : row) {
        res.insert(j);
      }
    }
    // Create a mapping from sparse indices to dense indices.
    std::unordered_map<int, int> sp_to_dense;
    std::vector<int> dense_to_sp(res.size());
    int new_index = 0;
    for (int ind : res) {
      dense_to_sp[new_index] = ind;
      sp_to_dense[ind] = new_index++;
    }
    return {sp_to_dense, dense_to_sp};
  }

  std::tuple<Eigen::MatrixX<double>, std::vector<int>>
   sparse_to_dense(const SparseMapMatrix& mat) {
    auto [sp_to_dense, dense_to_sp] = find_referenced_indices(mat);

    Eigen::MatrixX<double> res(sp_to_dense.size(), sp_to_dense.size());
    std::vector<int> dense_to_sparse(sp_to_dense.size());
    for (const auto& [i, row] : mat.vals()) {
      for (const auto& [j, val] : row) {
        res(sp_to_dense[i], sp_to_dense[j]) = val;
      }
    }
    return {res, dense_to_sp};
   }

   SparseMapMatrix dense_to_sparse(const Eigen::MatrixX<double>& dense,
   const std::vector<int>& dense_to_sp, int n) {
    std::vector<Eigen::Triplet<double>> triplets;
    SparseMapMatrix res(n, n);
    for (int i = 0; i < dense.rows(); i++) {
      for (int j = 0; j < dense.cols(); j++) {
        res(dense_to_sp[i], dense_to_sp[j]) = dense(i, j);
      }
    }
    return res;
   }

  bool is_positive_diagonal_dominant(const Eigen::MatrixX<double>& dense) {
    for (int i = 0; i < dense.rows(); i++) {
      double off_diag = 0.0;
      for (int j = 0; j < dense.cols(); j++) {
        if (i != j) { off_diag += abs(dense(i, j)); }
      }
      if (dense(i, i) < off_diag + 1e-5) { return false; }
    }
    return true;
  }

Eigen::MatrixXd project_hessian(const Eigen::MatrixXd& hessian) {
  if (is_positive_diagonal_dominant(hessian)) {
    return hessian;
  }
  Eigen::SelfAdjointEigenSolver<Eigen::MatrixX<double>> eig(hessian);
  Eigen::MatrixX<double> D = eig.eigenvalues().asDiagonal();
  bool all_positive = true;
  for (int i = 0; i < hessian.rows(); ++i) {
      if (D(i, i) < 1e-6) {
          D(i, i) = 1e-6;
          all_positive = false;
      }
  }
  if (all_positive) {
    return hessian;
  }
  Eigen::MatrixX<double> projected = eig.eigenvectors() * D * eig.eigenvectors().transpose();
  return projected;
}

SparseMapMatrix project_hessian(const SparseMapMatrix& hessian) {
    const auto [dense, dense_to_sp] =  sparse_to_dense(hessian);
    if (is_positive_diagonal_dominant(dense)) {
      return hessian;
    }
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixX<double>> eig(dense);
    Eigen::MatrixX<double> D = eig.eigenvalues().asDiagonal();
    bool all_positive = true;
    for (int i = 0; i < dense.rows(); ++i) {
        if (D(i, i) < 1e-6) {
            D(i, i) = 1e-6;
            all_positive = false;
        }
    }
    if (all_positive) {
      return hessian;
    }
    Eigen::MatrixX<double> projected = eig.eigenvectors() * D * eig.eigenvectors().transpose();
    return dense_to_sparse(projected, dense_to_sp, hessian.rows());
  }

} // namespace SparseAD
