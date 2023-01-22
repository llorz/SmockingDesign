#pragma once
#include <vector>

std::vector<std::vector<double>> bary_coords(const std::vector<std::vector<double>>& verts,
const std::vector<std::vector<double>>& uv,
const std::vector<std::vector<int>>& faces);

std::vector<std::vector<double>> isometry_param(const std::vector<std::vector<double>>& verts,
const std::vector<std::vector<double>>& uv,
const std::vector<std::vector<int>>& faces);