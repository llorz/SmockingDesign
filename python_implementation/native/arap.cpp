#if __INTELLISENSE__
#undef __ARM_NEON
#undef __ARM_NEON__
#endif

#include "SparseAD/include/Problem.h"
#include <Eigen/Eigen>
#include <unordered_set>
#include <unordered_map>
#include "arap.h"
#include <type_traits>


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

double edge_length(const Eigen::MatrixXd& verts, int i, int j) {
  return (verts.row(i) - verts.row(j)).norm();
}


template<typename T>
Eigen::Matrix3<T> get_local_frame(auto& verts,
const std::vector<int>& face) {
int v0 = face[0], v1 = face[1], v2 = face[2];
  const Eigen::Vector3<T> vv0 {verts(v0, 0), verts(v0, 1), verts(v0, 2)};
  const Eigen::Vector3<T> vv1 {verts(v1, 0), verts(v1, 1), verts(v1, 2)};
  const Eigen::Vector3<T> vv2 {verts(v2, 0), verts(v2, 1), verts(v2, 2)};
  const Eigen::Vector3<T> e1 = vv1 - vv0, e2 = vv2 - vv1, e3 = vv0 - vv2;
  return Eigen::Matrix3<T> {{e1(0), e2(0), e3(0)},
  {e1(1), e2(1), e3(1)},
  {e1(2), e2(2), e3(2)}};
}

std::vector<std::vector<double>> run_one_step(
  const std::vector<std::vector<double>>& vertices,
  const std::vector<std::vector<int>>& faces,
  const std::vector<std::vector<double>>& edge_constraints,
  const std::vector<std::vector<double>>& constraints,
  double w_edge,
  double w_constraints,
  double w_gravity) {
    Eigen::MatrixXd verts = std_to_eig(vertices);
    std::vector<Eigen::Matrix3d> local_frame_inverses(faces.size());
    for (int i = 0; i < faces.size(); i++) {
      local_frame_inverses[i] = get_local_frame<double>(verts, faces[i]).inverse();
    }

    SparseAD::Problem prob(verts, SparseAD::Problem::Options {.num_iterations = 3});
    // "ARAP" energy.
    prob.add_sparse_energy<6>(edge_constraints.size(), [&]<typename T>(int i, auto& x) -> T {
      int v0 = edge_constraints[i][0], v1 = edge_constraints[i][1];
      double dist = edge_constraints[i][2];
      return w_edge * SparseAD::sqr((x.row(v0) - x.row(v1)).norm() - dist);
    });


    // Constraints energy for the underlay vertices.
    prob.add_sparse_energy<3>(constraints.size(), [&]<typename T>(int i, auto& x) -> T {
      int v0 = constraints[i][0];
      Eigen::Vector3d pos {constraints[i][1], constraints[i][2], constraints[i][3]};
      return w_constraints * (x.row(v0) - pos).squaredNorm();
    });

  
    // Prepare gravity energy.
    // Create a set with the constrained vertices.
    std::unordered_set<int> constraint_vertices;
    for (auto& c  : constraints) {
      constraint_vertices.insert((int)c[0]);
    }
    // Build a vector with the indices of the vertices on which gravity should be appleid.
    std::vector<int> gravity_vertices;
    gravity_vertices.reserve(vertices.size() - constraints.size());
    for (int i = 0; i < vertices.size() - constraints.size(); i++) {
      if (constraint_vertices.count(i) == 0) {
        gravity_vertices.push_back(i);
      }
    }

    prob.add_sparse_energy<1>(gravity_vertices.size(), [&]<typename T>(int i, auto& x) -> T {
      auto pos = x(gravity_vertices[i], 1);
      return -w_gravity * pos;
    });


    prob.optimize();
    return eig_to_std(prob.current());
  }


MAT prepare_constraints(
  const std::vector<std::vector<double>>& vertices,
  const std::vector<std::vector<int>>& std_faces) {
    Eigen::MatrixXd verts = std_to_eig(vertices);
    Eigen::MatrixX<int> faces = std_to_eig(std_faces);
    Eigen::MatrixXd result = verts;
    // Get the neighbors for each vertex (with a higher index).
    std::vector<std::set<int>> neighbors(vertices.size());
    int edges = 0;
    for (int i = 0; i < std_faces.size(); i++) {
      for (int j = 0; j < 3; j++) {
        int v0 = std_faces[i][j], v1 = std_faces[i][(j + 1) % 3];
        if (neighbors[std::min(v0, v1)].insert(std::max(v0, v1)).second) {
          // Count how many edges exist.
          edges++;
        }
      }
    }
    // Build constraints for the edges.
    std::vector<std::vector<double>> edge_constraints(edges);
    int counter = 0;
    for (int i = 0; i < neighbors.size(); i++) {
      for (auto neighbor : neighbors[i]) {
        edge_constraints[counter].push_back(i);
        edge_constraints[counter].push_back(neighbor);
        edge_constraints[counter].push_back(edge_length(verts, i, neighbor));
        counter++;
      }
    }

    return edge_constraints;
  }

  std::vector<std::vector<double>> run_simulation(
    const std::vector<std::vector<double>>& orig_vertices,
  const std::vector<std::vector<double>>& vertices,
  const std::vector<std::vector<int>>& faces,
  const std::vector<std::vector<double>>& constraints,
  double w_edge,
  double w_constraints,
  double w_gravity) {
    Eigen::MatrixXd verts = std_to_eig(vertices);
    Eigen::MatrixXd orig_verts = std_to_eig(orig_vertices);

    std::vector<std::unordered_set<int>> neighbors(vertices.size());
    for (int i = 0; i < faces.size(); i++) {
      for (int j = 0; j < 3; j++) {
        neighbors[faces[i][j]].insert(faces[i][(j + 1) % 3]);
        neighbors[faces[i][(j+1) % 3]].insert(faces[i][j]);
      }
    }
    std::unordered_map<int, int> constraint_indices;
    for (int i = 0; i < constraints.size(); i++) {
      constraint_indices[(int)constraints[i][0]] = i;
    }

    auto func = [&]<typename T>(int i, auto& x) {
      const auto& iter = constraint_indices.find(i);
      if (iter != constraint_indices.end()) {
        auto c = constraints[iter->second];
        int v0 = c[0];
        Eigen::Vector3d pos {c[1], c[2], c[3]};
        T res = w_constraints * (x.row(v0) - pos).squaredNorm();
        return +res;
      }
      T energy = 0.0;
      Eigen::Vector3<T> cur = x.row(i);
      Eigen::Vector3d original_cur = orig_verts.row(i);
      for (auto& n : neighbors[i]) {
        Eigen::Vector3d loc = orig_verts.row(n);
        double dist = (loc- original_cur).norm();
        energy += +SparseAD::sqr((cur - x.row(n)).norm() - dist);
      }
      return energy;
    };

    SparseAD::Problem prob(verts, SparseAD::Problem::Options {.num_iterations = 3});
    prob.add_sparse_energy(vertices.size(), SparseAD::Energy<decltype(func)> {
      .project_hessian = false,
      .provider = func
  });

    return eig_to_std(prob.optimize().current());
  }