#if __INTELLISENSE__
#undef __ARM_NEON
#undef __ARM_NEON__
#endif

#include "param.h"

#include <Eigen/Eigen>

std::vector<Eigen::Matrix3d> get_barcy_coords_projection(
    const std::vector<std::vector<double>>& uv,
    const std::vector<std::vector<int>>& faces) {
  std::vector<Eigen::Matrix3d> result(faces.size());
  for (int i = 0; i < faces.size(); i++) {
    Eigen::Vector2d x1 =
                        Eigen::Vector2d{uv[faces[i][0]][0], uv[faces[i][0]][1]},
                    x2 =
                        Eigen::Vector2d{uv[faces[i][1]][0], uv[faces[i][1]][1]},
                    x3 =
                        Eigen::Vector2d{uv[faces[i][2]][0], uv[faces[i][2]][1]};

    Eigen::Matrix3d T{{1, 1, 1}, {x1.x(), x2.x(), x3.x()}, {x1.y(), x2.y(), x3.y()}};
    result[i] = T.inverse();
  }
  return result;
}

std::vector<double> find_bary_coordinates(
    const std::vector<double>& verts,
    const std::vector<Eigen::Matrix3d>& proj) {
  Eigen::Vector3d x { 1, verts[0], verts[1] };
  for (int i = 0; i < proj.size(); i++) {
    Eigen::Vector3d lambda = proj[i] * x;
    if (lambda.x() >= 0 && lambda.y() >= 0 && lambda.z() >= 0) {
      return {(double)i, lambda[0], lambda[1], lambda[2]};
    }
  }
  return {-1, 0, 0, 0};
}

std::vector<std::vector<double>> bary_coords(
    const std::vector<std::vector<double>>& verts,
    const std::vector<std::vector<double>>& uv,
    const std::vector<std::vector<int>>& faces) {
  std::vector<std::vector<double>> result(verts.size());
  const auto& bary_coords_proj = get_barcy_coords_projection(uv, faces);
  for (int i = 0; i < verts.size(); i++) {
    result[i] = find_bary_coordinates(verts[i], bary_coords_proj);
  }
  return result;
}