#pragma once
#include <Eigen/Eigen>

#include "SelfAdjointMapMatrix.h"

namespace Optiz {

SelfAdjointMapMatrix project_hessian(const SelfAdjointMapMatrix& hessian);

// Returns pair<dense, inds> such that the projected sparse hessian satisfies
// sparse[inds[i], inds[j]] = dense[i, j]
std::pair<Eigen::MatrixXd, std::vector<int>> project_sparse_hessian(
    const SelfAdjointMapMatrix& hessian);

Eigen::MatrixXd project_hessian(const Eigen::MatrixXd& hessian);

}  // namespace Optiz
