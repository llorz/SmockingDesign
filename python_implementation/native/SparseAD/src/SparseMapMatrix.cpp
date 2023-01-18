#if __INTELLISENSE__
#undef __ARM_NEON
#undef __ARM_NEON__
#endif

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
      values[it.row()][it.col()] = it.value();
    }
  }
}

SparseMapMatrix::SparseMapMatrix(SparseMapMatrix&& other) noexcept = default;

SparseMapMatrix& SparseMapMatrix::operator=(SparseMapMatrix&& other) = default;

SparseMapMatrix& SparseMapMatrix::operator+=(const SparseMapMatrix& other) {
  for (const auto& [i, row] : other.values) {
    for (const auto& [j, val] : row) {
      values[i][j] += val;
    }
  }
  return *this;
}

SparseMapMatrix& SparseMapMatrix::operator*=(double scalar) {
  for (auto& [i, row] : values) {
    for (auto& [j, val] : row) {
      val *= scalar;
    }
  }
  return *this;
}

SparseMapMatrix& SparseMapMatrix::operator/=(double scalar) {
  for (auto& [i, row] : values) {
    for (auto& [j, val] : row) {
      val /= scalar;
    }
  }
  return *this;
}

SparseMapMatrix& SparseMapMatrix::operator-=(const SparseMapMatrix& other) {
  for (const auto& [i, row] : other.values) {
    for (const auto& [j, val] : row) {
      values[i][j] -= val;
    }
  }
  return *this;
}

std::ostream& operator<<(std::ostream& s, const SparseMapMatrix& mat) {
  s << mat.toDense();
  return s;
}

std::map<int, double>& SparseMapMatrix::operator[](int row) {
  return values[row];
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
  return values[i][j];
}

double& SparseMapMatrix::insert(int i, int j) {
  return values[i][j];
}

Eigen::MatrixXd SparseMapMatrix::toDense() const {
  Eigen::MatrixXd res = Eigen::MatrixXd::Zero(n_rows, n_cols);
  for (const auto& [i, row] : values) {
    for (const auto& [j, val] : row) {
      res(i, j) = val;
    }
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
  // res[i, j] = sum_k(a_{i,k} * b_{k, j}).
  // TODO: Check which one is faster.
  // SPARSE_MATRIX_ITER(i, k, a_val, a, 
  for (const auto& [i, k, a_val] : a) {
      // Row 'k' of matrix 'b'.
      const auto& sec_row = b.values.find(k);
      if (sec_row == b.values.end()) { continue; }

      for (const auto& [j, b_val] : sec_row->second) {
        res[i][j] += a_val * b_val;
      }
  }
  // );
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
  for (const auto& [i, row] : values) {
    for (const auto& [j, val] : row) {
      res[j][i] = val;
    }
  }
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
std::tuple<int, int, double> SparseMapMatrix::Iterator::operator*() const {
  return {row_iter->first, col_iter->first, col_iter->second};
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
  return {row_iter->first, col_iter->first, col_iter->second};
}

}  // namespace SparseAD
