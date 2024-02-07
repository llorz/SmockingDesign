#pragma once
#include <Eigen/Eigen>
#include <vector>

#include "Var.h"

namespace Optiz {

template <typename T>
class TGenericVariableFactory {
 public:
  TGenericVariableFactory(const Eigen::VectorXd& current,
  const std::pair<int, int>& shape) : _current(current), shape(shape) {}

  using Scalar = T;

  virtual T operator()(int i) const = 0;
  virtual T operator()(int i, int j) const = 0;
  Eigen::Map<const Eigen::MatrixXd> current_mat() const {
    return Eigen::Map<const Eigen::MatrixXd>(_current.data(), shape.first,
                                       shape.second);
  }
  int num_vars() const { return _current.size(); }

  Eigen::RowVectorX<T> row(int i) const {
    Eigen::RowVectorX<T> result(shape.second);
    for (int j = 0; j < shape.second; j++) {
      result(j) = operator()(i, j);
    }
    return result;
  }

  const T& get(const T& v) { return v; }

  const Eigen::VectorXd& current() const { return _current; }

 protected:
  const Eigen::VectorXd& _current;
  std::pair<int, int> shape;
};

class VecVarFactory : public TGenericVariableFactory<Var> {
 public:
  VecVarFactory(const Eigen::VectorXd& init,
                 const std::vector<int>& block_start_indices);

  Var operator()(int i) const;

  Var operator()(int i, int j) const;

 private:
  const std::vector<int>& block_start_indices;
};

template <typename T>
class TVarFactory : public TGenericVariableFactory<T> {
 public:
  TVarFactory(const Eigen::VectorXd& init, const std::pair<int, int>& shape)
      : TGenericVariableFactory<T>(init, shape) {}
  T operator()(int i) const { return T(this->_current(i), i); }
  T operator()(int i, int j) const {
    int index = i + j * this->shape.first;
    return T(this->_current(index), index);
  }
};

template <typename T>
class ValFactory : public TGenericVariableFactory<T> {
 public:
  ValFactory(const Eigen::VectorXd& init, const std::pair<int, int>& shape)
      : TGenericVariableFactory<T>(init, shape) {}

  T operator()(int i) const { return this->_current(i); }

  T operator()(int i, int j) const { 
    int index = i + j * this->shape.first;
    return this->_current(index); }
};

template <typename T>
class VecValFactory : public ValFactory<T> {
 public:
  VecValFactory(const Eigen::VectorXd& init,
                const std::vector<int>& block_start_indices)
      : ValFactory<T>(init, {0, 0}), block_start_indices(block_start_indices) {}

  T operator()(int i, int j) const {
    return this->_current(block_start_indices[i] + j);
  }

 private:
  const std::vector<int>& block_start_indices;
};

using VarFactory = TVarFactory<Var>;
extern template class TVarFactory<Var>;
extern template class VecValFactory<double>;
extern template class VecValFactory<Var>;

}  // namespace Optiz