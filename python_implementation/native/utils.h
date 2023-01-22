#pragma once
#include <Eigen/Eigen>

namespace utils {

template<typename T>
Eigen::MatrixX<T> std_to_eig(const std::vector<std::vector<T>>& vec) {
  if (vec.size() == 0) {
    return Eigen::MatrixX<T>();
  }
  int m = vec.size();
  int n = vec[0].size();
  Eigen::MatrixX<T> result(m, n);
  for (int i = 0; i < m; i++) {
    for (int j = 0; j < n; j++) {
      result(i, j) = vec[i][j];
    }
  }
  return result;
}

template<typename T>
std::vector<std::vector<T>> eig_to_std(const Eigen::MatrixX<T>& mat) {
  std::vector<std::vector<T>> result(mat.rows(),
                                          std::vector<T>(mat.cols()));
  for (int i = 0; i < mat.rows(); i++) {
    for (int j = 0; j < mat.cols(); j++) {
      result[i][j] = mat(i, j);
    }
  }
  return result;
}


template<typename T>
Eigen::Matrix<T, 2, 2> get_local_frame(const Eigen::MatrixX<T>& verts,
const std::vector<int>& face) {
int v0 = face[0], v1 = face[1], v2 = face[2];
  const Eigen::Vector3<T> &e1 = (verts.row(v1) - verts.row(v0)),
                        e2 = verts.row(v2) - verts.row(v1);
  const Eigen::Vector3<T>& e1_rot = e1.cross(e2).cross(e1).normalized();

  Eigen::Matrix<T, 2, 2> local_mat;
  local_mat << e1.norm(), e2.dot(e1) / e1.norm(), 0,
      e2.cross(e1).norm() / e1.norm();
  return local_mat;
}

}