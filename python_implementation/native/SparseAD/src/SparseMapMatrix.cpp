#include "../include/SparseMapMatrix.h"

#include <iostream>
#include <vector>

namespace SparseAD {

SparseMapMatrix::SparseMapMatrix(int n_rows, int n_cols)
    : n_rows(n_rows), n_cols(n_cols) {}

SparseMapMatrix::SparseMapMatrix(const Eigen::SparseMatrix<double>& mat): n_rows(mat.rows()),
n_cols(mat.cols()) {
  for (int k=0; k<mat.outerSize(); ++k) {
    for (typename Eigen::SparseMatrix<double>::InnerIterator it(mat,k); it; ++it) {
      values[it.col()][it.row()] = it.value();
    }
  }
}

SparseMapMatrix::SparseMapMatrix(SparseMapMatrix&& other) noexcept = default;

SparseMapMatrix& SparseMapMatrix::operator=(SparseMapMatrix&& other) = default;

SparseMapMatrix& SparseMapMatrix::operator+=(const SparseMapMatrix& other) {
  SPARSE_MATRIX_ITER(other, operator()(i, j) += val;)
  return *this;
}

SparseMapMatrix& SparseMapMatrix::operator*=(double scalar) {
  SPARSE_MATRIX_ITER((*this), val *= scalar;)
  return *this;
}

SparseMapMatrix& SparseMapMatrix::operator/=(double scalar) {
  SPARSE_MATRIX_ITER((*this), val /= scalar;)
  return *this;
}

SparseMapMatrix& SparseMapMatrix::operator-=(const SparseMapMatrix& other) {
  SPARSE_MATRIX_ITER(other, operator()(i, j) -= val;)
  return *this;
}

std::ostream& operator<<(std::ostream& s, const SparseMapMatrix& mat) {
  s << mat.toDense();
  return s;
}

SparseMapMatrix::RowAccess::RowAccess(SparseMapMatrix* mat, int row): mat(mat), row(row) {}
double& SparseMapMatrix::RowAccess::operator[](int col) {
  return mat->operator()(row, col);
}

SparseMapMatrix::RowAccess SparseMapMatrix::operator[](int row) {
  return RowAccess(this, row);
}

int SparseMapMatrix::rows() const {
  return n_rows;
}
int SparseMapMatrix::cols() const {
  return n_cols;
}

int SparseMapMatrix::nnz() const {
  int counter = 0;
  for (const auto& nz : *this) {
    counter++;
  }
  return counter;
}

double& SparseMapMatrix::operator()(int i, int j) {
  return values[j][i];
}

double& SparseMapMatrix::insert(int i, int j) {
  return values[j][i];
}

Eigen::MatrixXd SparseMapMatrix::toDense() const {
  Eigen::MatrixXd res = Eigen::MatrixXd::Zero(n_rows, n_cols);
  for (const auto& [i, j, val] : *this) {
    res(i, j) = val;
  }
  return res;
}

SparseMapMatrix::operator Eigen::SparseMatrix<double>() const {
  Eigen::SparseMatrix<double> res(n_rows, n_cols);
  std::vector<Eigen::Triplet<double>> triplets;
  for (const auto& [i, j, val] : *this) {
      triplets.push_back(Eigen::Triplet<double>(i, j, val));
  }
  res.setFromTriplets(triplets.begin(), triplets.end());
  return res;
}

SparseMapMatrix operator+(const SparseMapMatrix& first,
                        const SparseMapMatrix& other) {
  return SparseMapMatrix(first) += other;
}
SparseMapMatrix operator+(SparseMapMatrix&& first, const SparseMapMatrix& other) {
  first += other;
  return first;
}
SparseMapMatrix operator+(const SparseMapMatrix& first, SparseMapMatrix&& other) {
  return std::move(other) + first;
}
SparseMapMatrix operator+(SparseMapMatrix&& first, SparseMapMatrix&& other) {
  return std::move(first) + other;
}

SparseMapMatrix operator-(const SparseMapMatrix& first,
                        const SparseMapMatrix& other) {
  return SparseMapMatrix(first) -= other;
}
SparseMapMatrix operator-(SparseMapMatrix&& first, const SparseMapMatrix& other) {
  first -= other;
  return first;
}
SparseMapMatrix operator-(const SparseMapMatrix& first, SparseMapMatrix&& other) {
  // TODO: do -other += first?
  return first - other;
}
SparseMapMatrix operator-(SparseMapMatrix&& first, SparseMapMatrix&& other) {
  return std::move(first) - other;
}

SparseMapMatrix operator*(const SparseMapMatrix& a,
                        const SparseMapMatrix& b) {
  SparseMapMatrix res(a.n_rows, b.n_cols);
  SPARSE_MATRIX_ITER(b, 
    const auto& a_col = a.values.find(i);
    if (a_col == a.values.end()) { continue; }
    for (const auto& [ii, a_val] : a_col->second) {
      res(ii, j) += a_val * val;
    }
  )
  return res;
}

SparseMapMatrix operator*(const SparseMapMatrix& first, double scalar) {
  return SparseMapMatrix(first) *= scalar;
}
SparseMapMatrix operator*(SparseMapMatrix&& first, double scalar) {
  first *= scalar;
  return first;
}
SparseMapMatrix operator*(double scalar, const SparseMapMatrix& first) {
  return SparseMapMatrix(first) *= scalar;
}
SparseMapMatrix operator*(double scalar, SparseMapMatrix&& first) {
  first *= scalar;
  return first;
}
SparseMapMatrix operator/(const SparseMapMatrix& first, double scalar) {
  return SparseMapMatrix(first) /= scalar;
}
SparseMapMatrix operator/(SparseMapMatrix&& first, double scalar) {
  first /= scalar;
  return first;
}

SparseMapMatrix SparseMapMatrix::transpose() const {
  SparseMapMatrix res(n_cols, n_rows);
  SPARSE_MATRIX_ITER((*this), res(j, i) = val;)
  return res;
}

// Iterator stuff.

SparseMapMatrix::Iterator::Iterator(std::map<int, std::map<int, double>>& values, bool begin)
:values(values) {
  if (begin) {
    row_iter = values.begin();
    if (!values.empty()) {
      col_iter = row_iter->second.begin();
    }
  } else {
    row_iter = values.end();
  }
}
const SparseMapMatrix::Iterator& SparseMapMatrix::Iterator::operator++() {
  col_iter++;
 if (col_iter == row_iter->second.end()) {
  row_iter++;
  if (row_iter == values.end()) {
    col_iter = std::map<int, double>::iterator();
  } else {
    col_iter = row_iter->second.begin();
  }
 }
 return *this;
}
bool SparseMapMatrix::Iterator::operator==(const Iterator& other) const {
  return row_iter == other.row_iter && col_iter == other.col_iter;
}
bool SparseMapMatrix::Iterator::operator!=(const Iterator& other) const {
  return row_iter != other.row_iter || col_iter != other.col_iter;
}
std::tuple<int, int, double&> SparseMapMatrix::Iterator::operator*() const {
  return std::tie(col_iter->first, row_iter->first, col_iter->second);
}

// Const iterator

SparseMapMatrix::ConstIterator::ConstIterator(const std::map<int, std::map<int, double>>& values, bool begin)
:values(values) {
  if (begin) {
    row_iter = values.begin();
    if (!values.empty()) {
      col_iter = row_iter->second.begin();
    }
  } else {
    row_iter = values.end();
  }
}
const SparseMapMatrix::ConstIterator& SparseMapMatrix::ConstIterator::operator++() {
  col_iter++;
 if (col_iter == row_iter->second.end()) {
  row_iter++;
  if (row_iter == values.end()) {
    col_iter = std::map<int, double>::const_iterator();
  } else {
    col_iter = row_iter->second.begin();
  }
 }
 return *this;
}
bool SparseMapMatrix::ConstIterator::operator==(const ConstIterator& other) const {
  return row_iter == other.row_iter && col_iter == other.col_iter;
}
bool SparseMapMatrix::ConstIterator::operator!=(const ConstIterator& other) const {
  return row_iter != other.row_iter || col_iter != other.col_iter;
}
std::tuple<int, int, double> SparseMapMatrix::ConstIterator::operator*() const {
  return {col_iter->first, row_iter->first, col_iter->second};
}

}  // namespace SparseAD
