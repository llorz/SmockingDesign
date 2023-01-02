#pragma once

#include <Eigen/Eigen>
#include <iostream>
#include "HessianProjection.h"
#include "SparseMapMatrix.h"
#include <tuple>

namespace SparseAD {

class DenseDual {
 public:
 DenseDual();
 DenseDual(double val);
 DenseDual(const double val, int index);
 DenseDual(const DenseDual&) noexcept;
 DenseDual(DenseDual&&) noexcept;

  DenseDual& operator=(const DenseDual& other) = default;
  DenseDual& operator=(DenseDual&&);

  static void set_num_vars(int new_num_vars);

  // Getters.
  double val() const;
  Eigen::MatrixXd grad() const;
  Eigen::VectorXd dense_grad() const;
  Eigen::MatrixXd hessian() const;
  using Tup = std::tuple<double, Eigen::VectorXd, Eigen::MatrixXd>;
  operator Tup() const;

  int n_vars() const;

  DenseDual& projectHessian();

  friend std::ostream& operator<<(std::ostream& s, const DenseDual& DenseDual);

  // Operators.
  DenseDual& operator*=(const DenseDual& b);
  DenseDual& operator*=(double b);
  DenseDual& operator/=(const DenseDual& b);
  DenseDual& operator/=(double b);
  DenseDual& operator+=(const DenseDual& b);
  DenseDual& operator+=(double b);
  DenseDual& operator-=(const DenseDual& b);
  DenseDual& operator-=(double b);
  DenseDual& chain_this(double val, double grad, double hessian);
  DenseDual chain(double val, double grad, double hessian) const;

  DenseDual inv() const;
  DenseDual& inv_self();
  DenseDual& neg();

  // Mul operator between two DenseDuals.
  friend DenseDual operator*(const DenseDual& a, const DenseDual& b);
  friend DenseDual operator*(DenseDual&& a, const DenseDual& b);
  friend DenseDual operator*(const DenseDual& a, DenseDual&& b);
  friend DenseDual operator*(DenseDual&& a, DenseDual&& b);
  friend DenseDual operator*(double b, const DenseDual& a);
  friend DenseDual operator*(const DenseDual& a, double b);
  friend DenseDual operator*(double b, DenseDual&& a);
  friend DenseDual operator*(DenseDual&& a, double b);

  // Div operator between two DenseDuals.
  friend DenseDual operator/(const DenseDual& a, const DenseDual& b);
  friend DenseDual operator/(DenseDual&& a, const DenseDual& b);
  friend DenseDual operator/(const DenseDual& a, DenseDual&& b);
  friend DenseDual operator/(DenseDual&& a, DenseDual&& b);
  friend DenseDual operator/(double b, const DenseDual& a);
  friend DenseDual operator/(const DenseDual& a, double b);
  friend DenseDual operator/(double b, DenseDual&& a);
  friend DenseDual operator/(DenseDual&& a, double b);

  // Add operator between two DenseDuals.
  friend DenseDual operator+(const DenseDual& a, const DenseDual& b);
  friend DenseDual operator+(DenseDual&& a, const DenseDual& b);
  friend DenseDual operator+(const DenseDual& a, DenseDual&& b);
  friend DenseDual operator+(DenseDual&& a, DenseDual&& b);
  // Add operator between DenseDual and double
  friend DenseDual operator+(double b, const DenseDual& a);
  friend DenseDual operator+(double b, DenseDual&& a);
  friend DenseDual operator+(const DenseDual& a, double b);
  friend DenseDual operator+(DenseDual&& a, double b);

  // Sub operator between two DenseDuals.
  friend DenseDual operator-(const DenseDual& a, const DenseDual& b);
  friend DenseDual operator-(DenseDual&& a, const DenseDual& b);
  friend DenseDual operator-(const DenseDual& a, DenseDual&& b);
  friend DenseDual operator-(DenseDual&& a, DenseDual&& b);
  // Sub operator between DenseDual and double
  friend DenseDual operator-(double b, const DenseDual& a);
  friend DenseDual operator-(const DenseDual& a, double b);
  friend DenseDual operator-(double b, DenseDual&& a);
  friend DenseDual operator-(DenseDual&& a, double b);

  friend DenseDual operator-(const DenseDual& a);
  friend DenseDual operator-(DenseDual&& a);
  friend DenseDual sqrt(const DenseDual& a);
  friend DenseDual sqrt(DenseDual&& a);
  friend DenseDual sqr(const DenseDual& a);
  friend DenseDual sqr(DenseDual&& a);
  friend DenseDual pow(const DenseDual& a, const double exponent);
  friend DenseDual pow(const DenseDual& a, const int exponent);
  friend DenseDual pow(DenseDual&& a, const int exponent);
  friend DenseDual pow(DenseDual&& a, const double exponent);
  friend DenseDual exp(const DenseDual& a);
  friend DenseDual exp(DenseDual&& a);

  
  // ----------------------- Comparisons -----------------------
  friend bool operator<(const DenseDual& a, const DenseDual& b);
  friend bool operator<=(const DenseDual& a, const DenseDual& b);
  friend bool operator>(const DenseDual& a, const DenseDual& b);
  friend bool operator>=(const DenseDual& a, const DenseDual& b);
  friend bool operator==(const DenseDual& a, const DenseDual& b);
  friend bool operator!=(const DenseDual& a, const DenseDual& b);

 private:
  double _val;
  Eigen::MatrixXd _grad;
  Eigen::MatrixXd _hessian;

  static int num_vars;
};

}  // namespace SparseAD

namespace Eigen {

/**
 * See https://eigen.tuxfamily.org/dox/TopicCustomizing_CustomScalar.html
 * and https://eigen.tuxfamily.org/dox/structEigen_1_1NumTraits.html
 */
template<>
struct NumTraits<SparseAD::DenseDual>: NumTraits<double>
{
    typedef SparseAD::DenseDual Real;
    typedef SparseAD::DenseDual NonInteger;
    typedef SparseAD::DenseDual Nested;

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
struct ScalarBinaryOpTraits<SparseAD::DenseDual, double, BinaryOp>
{
    typedef SparseAD::DenseDual ReturnType;
};

template<typename BinaryOp>
struct ScalarBinaryOpTraits<double, SparseAD::DenseDual, BinaryOp>
{
    typedef SparseAD::DenseDual ReturnType;
};

}