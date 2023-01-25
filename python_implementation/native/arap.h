#pragma once
#include <vector>
#include <tuple>

using MAT = std::vector<std::vector<double>>;

std::vector<std::vector<double>> run_one_step(
  const std::vector<std::vector<double>>& vertices,
  const std::vector<std::vector<int>>& faces,
  const std::vector<std::vector<double>>& edge_constraints,
  const std::vector<std::vector<double>>& constraints,
  double w_edge,
  double w_constraints,
  double w_gravity);

MAT prepare_constraints(
  const std::vector<std::vector<double>>& vertices,
  const std::vector<std::vector<int>>& faces);

std::vector<std::vector<double>> run_simulation(
  const std::vector<std::vector<double>>& vertices,
  const std::vector<std::vector<int>>& faces,
  const std::vector<std::vector<double>>& constraints,
  double w_edge,
  double w_constraints,
  double w_gravity);