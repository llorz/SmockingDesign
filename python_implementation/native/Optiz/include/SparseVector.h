#pragma once
#include <Eigen/Eigen>
#include <map>
#include <tuple>
#include <unordered_map>
#include <vector>

#include "VectorMap.h"

namespace Optiz {

class SparseVector {
 public:
  SparseVector(long size);
  SparseVector(SparseVector&&) noexcept;
  SparseVector(const SparseVector&) = default;

  VectorMap<long, double>& get_values() { return values; }
  const VectorMap<long, double>& get_values() const { return values; }

  Eigen::VectorXd to_dense() const;
  Eigen::SparseVector<double> to_sparse() const;

  double& operator()(long i);
  double& insert(long i);

  SparseVector& operator=(const SparseVector&) = default;
  SparseVector& operator=(SparseVector&&);

  SparseVector& add(const SparseVector& u, double alpha = 1.0);
  SparseVector& operator+=(const SparseVector& other);
  SparseVector& operator-=(const SparseVector& other);
  SparseVector& operator*=(double scalar);
  SparseVector& operator/=(double scalar);

  inline VectorMap<long, double>::iterator begin() { return values.begin(); }
  inline VectorMap<long, double>::iterator end() { return values.end(); }
  inline VectorMap<long, double>::const_iterator begin() const {
    return values.begin();
  }
  inline VectorMap<long, double>::const_iterator end() const {
    return values.end();
  }

  inline long rows() const { return _size; }
  inline long size() const { return _size; }

 private:
  VectorMap<long, double> values;
  long _size;
};
}  // namespace Optiz