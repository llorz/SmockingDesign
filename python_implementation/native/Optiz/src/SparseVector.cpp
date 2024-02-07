#if __INTELLISENSE__
#undef __ARM_NEON
#undef __ARM_NEON__
#endif

#include "SparseVector.h"

#include <vector>

namespace Optiz {
SparseVector::SparseVector(long size) : _size(size) {}
SparseVector::SparseVector(SparseVector&&) noexcept = default;

Eigen::VectorXd SparseVector::to_dense() const {
  Eigen::VectorXd dense = Eigen::VectorXd::Zero(_size);
  for (const auto& val : values) {
    dense(val.first) = val.second;
  }
  return dense;
}

Eigen::SparseVector<double> SparseVector::to_sparse() const {
  Eigen::SparseVector<double> sparse(_size);
  for (const auto& val : values) {
    sparse.insert(val.first) = val.second;
  }
  return sparse;
}

double& SparseVector::operator()(long i) { return values[i]; }

double& SparseVector::insert(long i) { return values[i]; }

SparseVector& SparseVector::operator=(SparseVector&& other) = default;

SparseVector& SparseVector::add(const SparseVector& u, double alpha) {
  for (const auto& val : u.values) {
    values[val.first] += alpha * val.second;
  }
  return *this;
}

SparseVector& SparseVector::operator+=(const SparseVector& other) {
  for (const auto& val : other.values) {
    values[val.first] += val.second;
  }
  return *this;
}

SparseVector& SparseVector::operator-=(const SparseVector& other) {
  for (const auto& val : other.values) {
    values[val.first] -= val.second;
  }
  return *this;
}

SparseVector& SparseVector::operator*=(double scalar) {
  for (auto& val : values) {
    val.second *= scalar;
  }
  return *this;
}

SparseVector& SparseVector::operator/=(double scalar) {
  for (auto& val : values) {
    val.second /= scalar;
  }
  return *this;
}

}  // namespace Optiz