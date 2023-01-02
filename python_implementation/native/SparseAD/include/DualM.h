#pragma once

#include <Eigen/Eigen>
#include <iostream>
#include "HessianProjection.h"
#include "SparseMapMatrix.h"
#include <tuple>

namespace SparseAD {

class DualM {
 public:
 DualM();
 DualM(double val);
 DualM(const double val, int index);
 DualM(const DualM&) noexcept;
 DualM(DualM&&) noexcept;

  DualM& operator=(const DualM& other) = default;
  DualM& operator=(DualM&&);

  static void set_num_vars(int new_num_vars);

  // Getters.
  double val() const;
  inline double& val() {return _val;}
  inline SparseMapMatrix& grad() {return _grad;}
  inline SparseMapMatrix& hessian() {return psd_hess();};
  SparseMapMatrix grad() const;
  Eigen::VectorXd dense_grad() const;
  SparseMapMatrix hessian() const;
  inline const SparseMapMatrix& psd_hess() const {
    return _hessians[psd_index];
  }

  inline SparseMapMatrix& psd_hess() {
    return _hessians[psd_index];
  }

  inline const SparseMapMatrix& nsd_hess() const {
    return _hessians[1-psd_index];
  }

  inline SparseMapMatrix& nsd_hess() {
    return _hessians[1-psd_index];
  }

  using Tup = std::tuple<double, Eigen::VectorXd, Eigen::SparseMatrix<double>>;
  operator Tup() const;

  int n_vars() const;

  DualM& projectHessian();
  void reserve(int n);

  friend std::ostream& operator<<(std::ostream& s, const DualM& DualM);

  // Operators.
  DualM& operator*=(const DualM& b);
  DualM& operator*=(double b);
  DualM& operator/=(const DualM& b);
  DualM& operator/=(double b);
  DualM& operator+=(const DualM& b);
  DualM& operator+=(double b);
  DualM& operator-=(const DualM& b);
  DualM& operator-=(double b);
  DualM& chain_this(double val, double grad, double hessian);
  DualM chain(double val, double grad, double hessian) const;

  DualM inv() const;
  DualM& inv_self();
  DualM& neg();

  // Mul operator between two DualMs.
  friend DualM operator*(const DualM& a, const DualM& b);
  friend DualM operator*(DualM&& a, const DualM& b);
  friend DualM operator*(const DualM& a, DualM&& b);
  friend DualM operator*(DualM&& a, DualM&& b);
  friend DualM operator*(double b, const DualM& a);
  friend DualM operator*(const DualM& a, double b);
  friend DualM operator*(double b, DualM&& a);
  friend DualM operator*(DualM&& a, double b);

  // Div operator between two DualMs.
  friend DualM operator/(const DualM& a, const DualM& b);
  friend DualM operator/(DualM&& a, const DualM& b);
  friend DualM operator/(const DualM& a, DualM&& b);
  friend DualM operator/(DualM&& a, DualM&& b);
  friend DualM operator/(double b, const DualM& a);
  friend DualM operator/(const DualM& a, double b);
  friend DualM operator/(double b, DualM&& a);
  friend DualM operator/(DualM&& a, double b);

  // Add operator between two DualMs.
  friend DualM operator+(const DualM& a, const DualM& b);
  friend DualM operator+(DualM&& a, const DualM& b);
  friend DualM operator+(const DualM& a, DualM&& b);
  friend DualM operator+(DualM&& a, DualM&& b);
  // Add operator between DualM and double
  friend DualM operator+(double b, const DualM& a);
  friend DualM operator+(double b, DualM&& a);
  friend DualM operator+(const DualM& a, double b);
  friend DualM operator+(DualM&& a, double b);

  // Sub operator between two DualMs.
  friend DualM operator-(const DualM& a, const DualM& b);
  friend DualM operator-(DualM&& a, const DualM& b);
  friend DualM operator-(const DualM& a, DualM&& b);
  friend DualM operator-(DualM&& a, DualM&& b);
  // Sub operator between DualM and double
  friend DualM operator-(double b, const DualM& a);
  friend DualM operator-(const DualM& a, double b);
  friend DualM operator-(double b, DualM&& a);
  friend DualM operator-(DualM&& a, double b);

  friend DualM operator-(const DualM& a);
  friend DualM operator-(DualM&& a);
  friend DualM sqrt(const DualM& a);
  friend DualM sqrt(DualM&& a);
  friend DualM sqr(const DualM& a);
  friend DualM sqr(DualM&& a);
  friend DualM pow(const DualM& a, const double exponent);
  friend DualM pow(const DualM& a, const int exponent);
  friend DualM pow(DualM&& a, const int exponent);
  friend DualM pow(DualM&& a, const double exponent);
  friend DualM exp(const DualM& a);
  friend DualM exp(DualM&& a);

  
  // ----------------------- Comparisons -----------------------
  friend bool operator<(const DualM& a, const DualM& b);
  friend bool operator<=(const DualM& a, const DualM& b);
  friend bool operator>(const DualM& a, const DualM& b);
  friend bool operator>=(const DualM& a, const DualM& b);
  friend bool operator==(const DualM& a, const DualM& b);
  friend bool operator!=(const DualM& a, const DualM& b);

 private:
  double _val;
  SparseMapMatrix _grad;
  SparseMapMatrix _hessians[2];
  int psd_index;

  static int num_vars;
};

}  // namespace SparseAD

#pragma omp declare reduction (+ : SparseAD::DualM : omp_out +=  omp_in)

namespace Eigen {

/**
 * See https://eigen.tuxfamily.org/dox/TopicCustomizing_CustomScalar.html
 * and https://eigen.tuxfamily.org/dox/structEigen_1_1NumTraits.html
 */
template<>
struct NumTraits<SparseAD::DualM>: NumTraits<double>
{
    typedef SparseAD::DualM Real;
    typedef SparseAD::DualM NonInteger;
    typedef SparseAD::DualM Nested;

    enum
    {
        IsComplex = 0,
        IsInteger = 0,
        IsSigned = 1,
        RequireInitialization = 1,
        ReadCost = 1,
        AddCost = 9,
        MulCost = 9,
    };
};

template<typename BinaryOp>
struct ScalarBinaryOpTraits<SparseAD::DualM, double, BinaryOp>
{
    typedef SparseAD::DualM ReturnType;
};

template<typename BinaryOp>
struct ScalarBinaryOpTraits<double, SparseAD::DualM, BinaryOp>
{
    typedef SparseAD::DualM ReturnType;
};

}