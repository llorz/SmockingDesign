#pragma once
#include <Eigen/Eigen>
#include <vector>

#include "Dual.h"

namespace SparseAD {

template <typename T>
class TGenericVariableFactory {
 public:
 TGenericVariableFactory(const Eigen::MatrixXd& current): _current(current) {}

  virtual T operator()(int i) const = 0;
  virtual T operator()(int i, int j) const = 0;
  int num_vars() const {
    return _current.size();
  }

  Eigen::MatrixX<T> row(int i) const {
    Eigen::MatrixX<T> result(1, _current.cols());
    for (int j =0 ; j < _current.cols(); j++) {
      result(j) = operator()(i, j);
    }
    return result;
  }

  const T& get(const T& v) { return v; }

  const Eigen::MatrixXd& current() const {
    return _current;
  }
protected:
  const Eigen::MatrixXd& _current;
};

class VecDualFactory : public TGenericVariableFactory<Dual> {
 public:
  VecDualFactory(const Eigen::MatrixXd& init,
                 const std::vector<int>& block_start_indices);

  Dual operator()(int i) const;

  Dual operator()(int i, int j) const;

 private:
  const std::vector<int>& block_start_indices;
};

template <typename T>
class TDualFactory : public TGenericVariableFactory<T> {
  public:
  TDualFactory(const Eigen::MatrixXd& init): TGenericVariableFactory<T>(init) { 
  }
  T operator()(int i) const {
    return T(this->_current(i), i);
  }
  T operator()(int i, int j) const {
    return T(this->_current(i, j), i + j * this->_current.rows());
  }
};
template <typename T>
class TDualValFactory : public TGenericVariableFactory<T> {
  public:
  TDualValFactory(const Eigen::MatrixXd& init): TGenericVariableFactory<T>(init) { }
  T operator()(int i) const {
    return T(this->_current(i));
  }
  T operator()(int i, int j) const {
    return T(this->_current(i, j));
  }
};
// class DualFactory : public TGenericDualFactory<Dual> {
//  public:
//   DualFactory(const Eigen::MatrixXd& init);

//   Dual operator()(int i) const;

//   Dual operator()(int i, int j) const;
// };

// class DualValFactory : public TGenericDualFactory<Dual> {
//  public:
//   DualValFactory(const Eigen::MatrixXd& init);

//   Dual operator()(int i) const;

//   Dual operator()(int i, int j) const;
// }; 

class VecDualValFactory : public TGenericVariableFactory<Dual> {
 public:
  VecDualValFactory(const Eigen::MatrixXd& init,
                    const std::vector<int>& block_start_indices);

  Dual operator()(int i) const;

  Dual operator()(int i, int j) const;
  const std::vector<int>& block_start_indices;
};

template<typename T>
class ValFactory : public TGenericVariableFactory<T> {
 public:
  ValFactory(const Eigen::MatrixXd& init): TGenericVariableFactory<T>(init) {}

  T operator()(int i) const {
    return this->_current(i);
  }

  T operator()(int i, int j) const {
    return this->_current(i, j);
  }
};

template<typename T>
class VecValFactory : public ValFactory<T> {
 public:
  VecValFactory(const Eigen::MatrixXd& init,
  const std::vector<int>& block_start_indices): ValFactory<T>(init),
  block_start_indices(block_start_indices) {}

  T operator()(int i, int j) const {
    return this->_current(block_start_indices[i] + j);
  }
  private:
  const std::vector<int>& block_start_indices;
};

using DualFactory = TDualFactory<Dual>;
extern template class TDualFactory<Dual>;
using DualValFactory = TDualValFactory<Dual>;
extern template class TDualValFactory<Dual>;
extern template class VecValFactory<double>;
extern template class VecValFactory<Dual>;

}  // namespace SparseAD