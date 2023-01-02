#include "../include/DualFactory.h"

namespace SparseAD {
  template class TDualFactory<Dual>;
  template class TDualValFactory<Dual>;
  template class VecValFactory<double>;
  template class VecValFactory<Dual>;


VecDualFactory::VecDualFactory(const Eigen::MatrixXd& init, 
const std::vector<int>& block_start_indices): TGenericVariableFactory<Dual>(init), block_start_indices(block_start_indices) {}

Dual VecDualFactory::operator()(int i) const {
  return Dual(this->_current(i), i);
}

Dual VecDualFactory::operator()(int i, int j) const {
  return operator()(block_start_indices[i] + j);
}


VecDualValFactory::VecDualValFactory(const Eigen::MatrixXd& init, 
const std::vector<int>& block_start_indices): TGenericVariableFactory<Dual>(init), block_start_indices(block_start_indices) {}

Dual VecDualValFactory::operator()(int i) const {
  return Dual(this->_current(i));
}

Dual VecDualValFactory::operator()(int i, int j) const {
  return operator()(block_start_indices[i] + j);
}

}
