#if __INTELLISENSE__
#undef __ARM_NEON
#undef __ARM_NEON__
#endif

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <iostream>
#include <vector>

#include "SparseAD/include/Dual.h"
#include "SparseAD/include/Problem.h"
#include "param.h"

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

std::vector<std::vector<double>> eig_to_std(const Eigen::MatrixXd& mat) {
  std::vector<std::vector<double>> result(mat.rows(),
                                          std::vector<double>(mat.cols()));
  for (int i = 0; i < mat.rows(); i++) {
    for (int j = 0; j < mat.cols(); j++) {
      result[i][j] = mat(i, j);
    }
  }
  return result;
}

template <typename T>
T sqr(const T& x) {
  return x * x;
}

std::vector<std::vector<double>> embed_underlay(
    const std::vector<std::vector<double>>& x_vec_init,
    const std::vector<std::vector<double>>& eq_constraints) {
  Eigen::MatrixXd x_init = std_to_eig(x_vec_init);

  const auto& energy_func = [&]<typename T>(int i, auto& x) {
    int a = eq_constraints[i][0], b = eq_constraints[i][1];
    double d = eq_constraints[i][2];
    return sqr((x.row(a) - x.row(b)).norm() - d);
  };

  Eigen::MatrixXd output =
      SparseAD::Problem(x_init)
          .add_sparse_energy<4>(eq_constraints.size(), energy_func)
          .optimize()
          .current();
  return eig_to_std(output);
}

std::vector<std::vector<double>> embed_pleats(
    const std::vector<std::vector<double>>& x_underlay_init,
    const std::vector<std::vector<double>>& x_pleats_init,
    const std::vector<std::vector<double>>& eq_constraints, double var_weight,
    double max_embedding_weight, double eq_weight) {
  Eigen::MatrixXd x_pleats = std_to_eig(x_pleats_init);
  Eigen::MatrixXd x_underlay = std_to_eig(x_underlay_init);

  const auto& preserve_edge_len = [&]<typename T>(int i, auto& x) {
    int a = eq_constraints[i][0], b = eq_constraints[i][1];
    double d = eq_constraints[i][2];
    // a is an underlay node.
    if (a < x_underlay_init.size() && b >= x_underlay_init.size()) {
      b -= x_underlay_init.size();
      return sqr((x.row(b) - x_underlay.row(a)).norm() - d);
    } else if (b < x_underlay_init.size() && a >= x_underlay_init.size()) {
      a -= x_underlay_init.size();
      return sqr((x.row(a) - x_underlay.row(b)).norm() - d);
    }
    a -= x_underlay_init.size();
    b -= x_underlay_init.size();
    return sqr((x.row(a) - x.row(b)).norm() - d);
  };

  const auto& min_var_energy = [&]<typename T>(auto& x) {
    double avg = 0.0;
    for (int i = 0; i < x_pleats_init.size(); i++) {
      avg += SparseAD::val(x(i, 2));
    }
    avg /= x_pleats_init.size();
    T result = 0.0;
#pragma omp parallel for schedule(static) reduction(+ : result)
    for (int i = 0; i < x_pleats_init.size(); i++) {
      result += SparseAD::project_psd(sqr(x(i, 2) - avg));
    }
    result /= x_pleats_init.size();
    return var_weight * result;
  };

  const auto& maximize_pleat_underlay_embedding = [&]<typename T>(int i,
                                                                  auto& x) {
    int ppleat = i / x_underlay.rows();
    int punderlay = i % x_underlay.rows();
    const auto& p = x.row(ppleat);
    const auto& p2 = x_underlay.row(punderlay);
    return -((p - p2).norm());
  };

  const auto& maximize_pleat_pleat_embedding = [&]<typename T>(int i, auto& x) {
    int p1ind = i / x_pleats.rows();
    int p2ind = i % x_pleats.rows();
    if (p1ind >= p2ind) {
      return T(0.0);
    };
    const auto& p = x.row(p1ind);
    const auto& p2 = x.row(p2ind);
    return -((p - p2).norm());
  };

  return eig_to_std(
      SparseAD::Problem(x_pleats)
          // Maximize pleat-pleat distance.
          .add_sparse_energy<6>(
              sqr(x_pleats.rows()),
              SparseAD::Energy<decltype(maximize_pleat_pleat_embedding)>{
                  .weight = max_embedding_weight,
                  .provider = maximize_pleat_pleat_embedding})
          // Maximize pleat-underlay distance.
          .add_sparse_energy<3>(
              x_pleats.rows() * x_underlay.rows(),
              SparseAD::Energy<decltype(maximize_pleat_underlay_embedding)>{
                  .weight = max_embedding_weight,
                  .provider = maximize_pleat_underlay_embedding})
          // Minimize pleats 'z' variance.
          .add_energy(min_var_energy)
          // Equality constraints.
          .add_sparse_energy(
              eq_constraints.size(),
              SparseAD::Energy<decltype(preserve_edge_len)>{
                  .weight = eq_weight, .provider = preserve_edge_len})
          .optimize()
          .current());
}

Eigen::Matrix2d get_local_frame(const Eigen::MatrixXd& verts,
const std::vector<int>& face) {
int v0 = face[0], v1 = face[1], v2 = face[2];
  const Eigen::Vector3d &e1 = (verts.row(v1) - verts.row(v0)),
                        e2 = verts.row(v2) - verts.row(v1);
  const Eigen::Vector3d& e1_rot = e1.cross(e2).cross(e1).normalized();

  Eigen::Matrix2d local_mat;
  local_mat << e1.norm(), e2.dot(e1) / e1.norm(), 0,
      e2.cross(e1).norm() / e1.norm();
  return local_mat;
}

Eigen::Matrix2d get_jacobian(const Eigen::MatrixXd& verts,
  const Eigen::MatrixXd& deform_verts,
  const std::vector<int>& face) {
    // Create local frame.
    Eigen::Matrix2d local_frame = get_local_frame(verts, face);
    Eigen::Matrix2d new_local_frame = get_local_frame(deform_verts, face);
    return new_local_frame * local_frame.inverse();
  }

std::vector<double> get_isometry_distortion(
  const std::vector<std::vector<double>>& grid_verts_vec,
  const std::vector<std::vector<double>>& deform_verts_vec,
  const std::vector<std::vector<int>>& faces) {
    Eigen::MatrixXd grid_verts = std_to_eig(grid_verts_vec);
    Eigen::MatrixXd deform_verts = std_to_eig(deform_verts_vec);

    std::vector<double> result(grid_verts_vec.size());
    for (int i = 0 ; i < faces.size(); i++) {
      Eigen::Matrix2d j = get_jacobian(grid_verts, deform_verts, faces[i]);

      // Sym dirrichlet.
      // double distortion = (j.squaredNorm() + j.inverse().squaredNorm() - 4);
      // Singular values should be 1.
      Eigen::JacobiSVD<Eigen::Matrix2d> svd(j);
      double distortion = (svd.singularValues().array() - 1).square().sum();

      // Integrate the distortion around the vertices.
      for (int j = 0; j < 3; j++) {
        result[faces[i][j]] += distortion / 3;
      }
    }

    return result;
  }

PYBIND11_MODULE(cpp_smocking_solver, m) {
  m.doc() = "Ceres solver for smocking";
  m.def("embed_underlay", &embed_underlay, "Embed underlay graph.");
  m.def("embed_pleats", &embed_pleats, "Embed pleats graph.");
  m.def("bary_coords", &bary_coords, "Get isometry distortion.");
}
