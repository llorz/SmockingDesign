#include "param.h"
#include <Eigen/Eigen>

std::vector<std::vector<double>> bary_coords(const std::vector<std::vector<double>>& verts,
const std::vector<std::vector<double>>& uv,
const std::vector<std::vector<int>>& faces) {
  std::vector<Eigen::MatrixXd> bary_coords;
}