#pragma once
#include <Eigen/Eigen>
#include <map>
#include <tuple>
#include <vector>
#include "SparseVector.h"
#include "VectorMap.h"
#include <unordered_map>

namespace Optiz {

class SelfAdjointMapMatrix {
 public:
  SelfAdjointMapMatrix(long n);
  SelfAdjointMapMatrix(SelfAdjointMapMatrix&&) noexcept;
  SelfAdjointMapMatrix(const SelfAdjointMapMatrix&) = default;

  friend std::ostream& operator<<(std::ostream& s, const SelfAdjointMapMatrix& var);

  double& operator()(long i, long j);
  double& operator()(long i);
  double& insert(long i, long j);

  SelfAdjointMapMatrix& operator=(const SelfAdjointMapMatrix&) = default;
  SelfAdjointMapMatrix& operator=(SelfAdjointMapMatrix&&);

  SelfAdjointMapMatrix& operator+=(const SelfAdjointMapMatrix& other);
  SelfAdjointMapMatrix& operator-=(const SelfAdjointMapMatrix& other);
  SelfAdjointMapMatrix& operator*=(double scalar);
  SelfAdjointMapMatrix& operator/=(double scalar);

  SelfAdjointMapMatrix& add(const SelfAdjointMapMatrix& u, double alpha = 1.0);
  SelfAdjointMapMatrix& rank_update(const SparseVector& u, const SparseVector& v);
  SelfAdjointMapMatrix& rank_update(const SparseVector& u, double alpha = 1.0);

  operator std::vector<Eigen::Triplet<double>>() const;
  operator Eigen::SparseMatrix<double>() const;
  Eigen::MatrixXd to_dense() const;

  const inline VectorMap<long, double>& get_values() const {
    return values;
  }

  inline VectorMap<long, double>::iterator begin() {return values.begin();}
  inline VectorMap<long, double>::iterator end() {return values.end();}
  inline VectorMap<long, double>::const_iterator begin() const {return values.begin();}
  inline VectorMap<long, double>::const_iterator end() const {return values.end();}

  inline long rows() const {return _n;}
  inline long cols() const {return _n;}

 private:
 long _n;
 VectorMap<long, double> values;
};
}  // namespace Optiz