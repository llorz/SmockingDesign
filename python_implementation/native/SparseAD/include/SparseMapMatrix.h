
#pragma once
#include <unordered_map>
#include <Eigen/Eigen>
#include <map>
#include <tuple>

#define SPARSE_MATRIX_ITER(i, j, val, m, code) for (auto& [i, row] : m.vals()) {\
  for (auto& [j, val] : row) {\
    code \
  }\
}

namespace SparseAD {

class SparseMapMatrix {
public:
  SparseMapMatrix(int n_rows, int n_cols);
  SparseMapMatrix(const Eigen::SparseMatrix<double>& mat);
  SparseMapMatrix(SparseMapMatrix&&) noexcept;
  SparseMapMatrix(const SparseMapMatrix&) = default;

  friend std::ostream& operator<<(std::ostream& s, const SparseMapMatrix& dual);

  int rows() const;
  int cols() const;
  int nnz() const;
  inline std::map<int, std::map<int, double>>& vals() {return values;}
  inline const std::map<int, std::map<int, double>>& vals() const {return values;}
  double& operator()(int i, int j);
  double& insert(int i, int j);
  SparseMapMatrix transpose() const;
  operator Eigen::SparseMatrix<double>() const;
  Eigen::MatrixXd toDense() const;

  SparseMapMatrix& operator=(const SparseMapMatrix&) = default;
  SparseMapMatrix& operator=(SparseMapMatrix&&);
  SparseMapMatrix& operator+=(const SparseMapMatrix& other);
  SparseMapMatrix& operator-=(const SparseMapMatrix& other);
  SparseMapMatrix& operator+=(double scalar);
  SparseMapMatrix& operator-=(double scalar);
  SparseMapMatrix& operator*=(double scalar);
  SparseMapMatrix& operator/=(double scalar);

  friend SparseMapMatrix operator+(const SparseMapMatrix& first, const SparseMapMatrix& other);
  friend SparseMapMatrix operator+(SparseMapMatrix&& first, const SparseMapMatrix& other);
  friend SparseMapMatrix operator+(const SparseMapMatrix& first, SparseMapMatrix&& other);
  friend SparseMapMatrix operator+(SparseMapMatrix&& first, SparseMapMatrix&& other);

  friend SparseMapMatrix operator-(const SparseMapMatrix& first, const SparseMapMatrix& other);
  friend SparseMapMatrix operator-(SparseMapMatrix&& first, const SparseMapMatrix& other);
  friend SparseMapMatrix operator-(const SparseMapMatrix& first, SparseMapMatrix&& other);
  friend SparseMapMatrix operator-(SparseMapMatrix&& first, SparseMapMatrix&& other);

  friend SparseMapMatrix operator*(const SparseMapMatrix& first, const SparseMapMatrix& other);

  friend SparseMapMatrix operator*(const SparseMapMatrix& first, double scalar);
  friend SparseMapMatrix operator*(SparseMapMatrix&& first, double scalar);

  friend SparseMapMatrix operator/(const SparseMapMatrix& first, double scalar);
  friend SparseMapMatrix operator/(SparseMapMatrix&& first, double scalar);
  friend SparseMapMatrix operator*(double scalar, const SparseMapMatrix& first);
  friend SparseMapMatrix operator*(double scalar, SparseMapMatrix&& first);

  std::map<int, double>& operator[](int row);

  // Iterators stuff.
  struct Iterator {
    Iterator(std::map<int, std::map<int, double>>& values, bool begin);
    const Iterator& operator++();
    bool operator==(const Iterator& other) const;
    bool operator!=(const Iterator& other) const;
    std::tuple<int, int, double> operator*() const;

    std::map<int, std::map<int, double>>::iterator row_iter;
    std::map<int, double>::iterator col_iter;
    std::map<int, std::map<int, double>>& values;
  }; 
  struct ConstIterator {
    ConstIterator(const std::map<int, std::map<int, double>>& values, bool begin);
    const ConstIterator& operator++();
    bool operator==(const ConstIterator& other) const;
    bool operator!=(const ConstIterator& other) const;
    std::tuple<int, int, double> operator*() const;

    std::map<int, std::map<int, double>>::const_iterator row_iter;
    std::map<int, double>::const_iterator col_iter;
    const std::map<int, std::map<int, double>>& values;
  };

  inline Iterator begin() {return Iterator(values, true);}
  inline ConstIterator begin() const {return ConstIterator(values, true);}
  inline Iterator end() {return Iterator(values, false);}
  inline ConstIterator end() const {return ConstIterator(values, false);}
  

private:
  std::map<int, std::map<int, double>> values;
  int n_rows;
  int n_cols;
};

}