#pragma once

#include <Eigen/Eigen>
#include <iostream>
#include <tuple>

#include "HessianProjection.h"

namespace SparseAD {

template <int k>
class TDenseDual {
 public:
  TDenseDual(): TDenseDual(0.0) {}
  TDenseDual(double val)
      : _val(val),
        _grad(Eigen::VectorXd::Zero(k)),
        _hessian(Eigen::MatrixXd::Zero(k, k)) {}
  TDenseDual(const double val, int index) : TDenseDual(val) {
    _grad(index) = 1.0;
  }
  TDenseDual(TDenseDual&& other) noexcept = default;
  TDenseDual(const TDenseDual& other) noexcept = default;

  TDenseDual& operator=(const TDenseDual& other) = default;
  TDenseDual& operator=(TDenseDual&&) = default;
 
  // Getters.
  inline double val() const { return _val; }
  inline Eigen::VectorXd grad() const { return _grad; }
  inline Eigen::MatrixXd hessian() const { return _hessian; }
  using Tup = std::tuple<double, Eigen::VectorXd, Eigen::MatrixXd>;
  inline operator Tup() const { return {_val, _grad, _hessian}; }
 
  TDenseDual& projectHessian() {
    _hessian = project_hessian(_hessian);
    return *this;
  } 

  friend std::ostream& operator<<(std::ostream& s, const TDenseDual& dual) {
    s << "Val: " << dual._val << std::endl
      << "Grad: " << std::endl
      << dual._grad << std::endl
      << "Hessian: " << std::endl
      << dual._hessian << std::endl;
    return s;
  }

  // Operators.
  TDenseDual& operator*=(const TDenseDual& b) {
    Eigen::MatrixXd grad_b_grad_t = _grad * b._grad.transpose();
    _hessian = b._val * _hessian + grad_b_grad_t + grad_b_grad_t.transpose() +
               _val * b._hessian;
    _grad = b._val * _grad + _val * b._grad;
    _val *= b._val;
    return *this;
  }
  TDenseDual& operator*=(double b) {
    _val *= b;
    _grad *= b;
    _hessian *= b;
    return *this;
  }
  TDenseDual& operator/=(const TDenseDual& b) {
    _grad = (b._val * _grad - _val * b._grad) / (b._val * b._val);
    _val /= b._val;
    _hessian = (_hessian - _grad * b._grad.transpose() -
                b._grad * _grad.transpose() - _val * b._hessian) /
               b._val;
    return *this;
  }
  TDenseDual& operator/=(double b) {
    _val /= b;
    _grad /= b;
    _hessian /= b;
    return *this;
  }
  TDenseDual& operator+=(const TDenseDual& b) {
    _val += b._val;
    _grad += b._grad;
    _hessian += b._hessian;
    return *this;
  }
  TDenseDual& operator+=(double b) {
    _val += b;
    return *this;
  }
  TDenseDual& operator-=(const TDenseDual& b) {
    _val -= b._val;
    _grad -= b._grad;
    _hessian -= b._hessian;
    return *this;
  }
  TDenseDual& operator-=(double b) {
    _val -= b;
    return *this;
  }
  TDenseDual& chain_this(double val, double grad, double hessian) {
    _val = val;
    // _hessian = hessian * _grad * _grad.transpose() + grad * _hessian;
    _hessian *= grad;
    _hessian += hessian * _grad * _grad.transpose();
    _grad *= grad;
    return *this;
  }
  TDenseDual chain(double val, double grad, double hessian) const {
    TDenseDual res(*this);
    res.chain_this(val, grad, hessian);
    return res;
  }

  TDenseDual inv() const {
    double valsqr = _val * _val;
    double valcube = valsqr * _val;
    return chain(1 / _val, -1 / valsqr, 2 / valcube);
  }
  TDenseDual& inv_self() {
    double valsqr = _val * _val;
    double valcube = valsqr * _val;
    chain_this(1 / _val, -1 / valsqr, 2 / valcube);
    return *this;
  }
  TDenseDual& neg() {
    chain_this(-_val, -1.0, 0.0);
    return *this;
  }

  // Mul operator between two TDenseDuals.
  friend TDenseDual operator*(const TDenseDual& a, const TDenseDual& b) {
    TDenseDual res(a);
    res *= b;
    return res;
  }
  friend TDenseDual operator*(TDenseDual&& a, const TDenseDual& b) {
    a *= b;
    return a;
  }
  friend TDenseDual operator*(const TDenseDual& a, TDenseDual&& b) {
    return std::move(b) * a;
  }
  friend TDenseDual operator*(TDenseDual&& a, TDenseDual&& b) {
    return std::move(a) * b;
  }
  friend TDenseDual operator*(double b, const TDenseDual& a) {
    TDenseDual res = a;
    res *= b;
    return res;
  }
  friend TDenseDual operator*(const TDenseDual& a, double b) { return b * a; }
  friend TDenseDual operator*(double b, TDenseDual&& a) {
    a *= b;
    return a;
  }
  friend TDenseDual operator*(TDenseDual&& a, double b) {
    a *= b;
    return a;
  }

  // Div operator between two TDenseDuals.
  friend TDenseDual operator/(const TDenseDual& a, const TDenseDual& b) {
    TDenseDual res(a);
    res /= b;
    return res;
  }
  friend TDenseDual operator/(TDenseDual&& a, const TDenseDual& b) {
    a /= b;
    return a;
  }
  friend TDenseDual operator/(const TDenseDual& a, TDenseDual&& b) {
    return a / b;
  }
  friend TDenseDual operator/(TDenseDual&& a, TDenseDual&& b) {
    return std::move(a) / b;
  }
  friend TDenseDual operator/(double b, const TDenseDual& a) {
    TDenseDual res = a.inv();
    res *= b;
    return res;
  }
  friend TDenseDual operator/(const TDenseDual& a, double b) {
    TDenseDual res(a);
    res /= b;
    return res;
  }
  friend TDenseDual operator/(double b, TDenseDual&& a) {
    a.inv_self() *= b;
    return a;
  }
  friend TDenseDual operator/(TDenseDual&& a, double b) {
    a /= b;
    return a;
  }

  // Add operator between two TDenseDuals.
  friend TDenseDual operator+(const TDenseDual& a, const TDenseDual& b) {
    TDenseDual res(a);
    res += b;
    return res;
  }
  friend TDenseDual operator+(TDenseDual&& a, const TDenseDual& b) {
    a += b;
    return a;
  }
  friend TDenseDual operator+(const TDenseDual& a, TDenseDual&& b) {
    return std::move(b) + a;
  }
  friend TDenseDual operator+(TDenseDual&& a, TDenseDual&& b) {
    return std::move(a) + b;
  }
  // Add operator between TDenseDual and double
  friend TDenseDual operator+(double b, const TDenseDual& a) { return a + b; }
  friend TDenseDual operator+(double b, TDenseDual&& a) {
    a += b;
    return a;
  }
  friend TDenseDual operator+(const TDenseDual& a, double b) {
    TDenseDual res(a);
    res += b;
    return res;
  }
  friend TDenseDual operator+(TDenseDual&& a, double b) {
    a += b;
    return a;
  }

  // Sub operator between two TDenseDuals.
  friend TDenseDual operator-(const TDenseDual& a, const TDenseDual& b) {
    TDenseDual res(a);
    res -= b;
    return res;
  }
  friend TDenseDual operator-(TDenseDual&& a, const TDenseDual& b) {
    a -= b;
    return a;
  }
  friend TDenseDual operator-(const TDenseDual& a, TDenseDual&& b) {
    b.neg() += a;
    return b;
  }
  friend TDenseDual operator-(TDenseDual&& a, TDenseDual&& b) {
    return std::move(a) - b;
  }
  // Sub operator between TDenseDual and double
  friend TDenseDual operator-(double b, const TDenseDual& a) {
    TDenseDual res(-a);
    res += b;
    return res;
  }
  friend TDenseDual operator-(const TDenseDual& a, double b) {
    TDenseDual res(a);
    res -= b;
    return res;
  }
  friend TDenseDual operator-(double b, TDenseDual&& a) {
    a.neg() += b;
    return a;
  }
  friend TDenseDual operator-(TDenseDual&& a, double b) {
    a -= b;
    return a;
  }

  friend TDenseDual operator-(const TDenseDual& a) {
    return a.chain(-a._val, -1.0, 0.0);
  }
  friend TDenseDual operator-(TDenseDual&& a) {
    a.neg();
    return a;
  }
  friend TDenseDual sqrt(const TDenseDual& a) {
    const auto& sqrt_a = std::sqrt(a._val);
    return a.chain(sqrt_a, 0.5 / sqrt_a, -0.25 / (sqrt_a * a._val));
  }
  friend TDenseDual sqrt(TDenseDual&& a) {
    const auto& sqrt_a = std::sqrt(a._val);
    a.chain_this(sqrt_a, 0.5 / sqrt_a, -0.25 / (sqrt_a * a._val));
    return a;
  }
  friend TDenseDual sqr(const TDenseDual& a) { return a * a; }
  friend TDenseDual sqr(TDenseDual&& a) {
    a *= a;
    return a;
  }
  friend TDenseDual abs(const TDenseDual& a) { return a.chain(a._val, a._val >= 0? 1 : -1, 0); }
  friend TDenseDual abs(TDenseDual&& a) {
    a.chain_this(a._val, a._val >= 0? 1 : -1, 0);
    return a;
  }
  friend TDenseDual pow(const TDenseDual& a, const double exponent) {
    const double f2 = std::pow(a.val(), exponent - 2);
    const double f1 = f2 * a.val();
    const double f = f1 * a.val();

    return a.chain(f, exponent * f1, exponent * (exponent - 1) * f2);
  }
  friend TDenseDual pow(const TDenseDual& a, const int exponent) {
    const double f2 = std::pow(a.val(), exponent - 2);
    const double f1 = f2 * a.val();
    const double f = f1 * a.val();

    return a.chain(f, exponent * f1, exponent * (exponent - 1) * f2);
  }
  friend TDenseDual pow(TDenseDual&& a, const int exponent) {
    const double f2 = std::pow(a.val(), exponent - 2);
    const double f1 = f2 * a.val();
    const double f = f1 * a.val();
    a.chain_this(f, exponent * f1, exponent * (exponent - 1) * f2);
    return a;
  }
  friend TDenseDual pow(TDenseDual&& a, const double exponent) {
    const double f2 = std::pow(a.val(), exponent - 2);
    const double f1 = f2 * a.val();
    const double f = f1 * a.val();
    a.chain_this(f, exponent * f1, exponent * (exponent - 1) * f2);
    return a;
  }
  friend TDenseDual exp(const TDenseDual& a) {
    const double val = std::exp(a._val);
    return a.chain(val, val, val);
  }
  friend TDenseDual exp(TDenseDual&& a) {
    const double val = std::exp(a._val);
    a.chain_this(val, val, val);
    return a;
  }

  // ----------------------- Comparisons -----------------------
  friend bool operator<(const TDenseDual& a, const TDenseDual& b) { return a._val < b._val; }
  friend bool operator<=(const TDenseDual& a, const TDenseDual& b) { return a._val <= b._val; }
  friend bool operator>(const TDenseDual& a, const TDenseDual& b) { return a._val > b._val; }
  friend bool operator>=(const TDenseDual& a, const TDenseDual& b) { return a._val >= b._val; }
  friend bool operator==(const TDenseDual& a, const TDenseDual& b) { return a._val == b._val; }
  friend bool operator!=(const TDenseDual& a, const TDenseDual& b) { return a._val != b._val; }

 private:
  double _val;
  Eigen::Matrix<double, k, 1> _grad;
  Eigen::Matrix<double, k, k> _hessian;

  static int num_vars;
};

}  // namespace SparseAD

namespace Eigen {

/**
 * See https://eigen.tuxfamily.org/dox/TopicCustomizing_CustomScalar.html
 * and https://eigen.tuxfamily.org/dox/structEigen_1_1NumTraits.html
 */
template<int k>
struct NumTraits<SparseAD::TDenseDual<k>>: NumTraits<double>
{
    typedef SparseAD::TDenseDual<k> Real;
    typedef SparseAD::TDenseDual<k> NonInteger;
    typedef SparseAD::TDenseDual<k> Nested;

    enum
    {
        IsComplex = 0,
        IsInteger = 0,
        IsSigned = 1,
        RequireInitialization = 1,
        ReadCost = k * k,
        AddCost = k * k,
        MulCost = k * k,
    };
};

template<typename BinaryOp, int k>
struct ScalarBinaryOpTraits<SparseAD::TDenseDual<k>, double, BinaryOp>
{
    typedef SparseAD::TDenseDual<k> ReturnType;
};

template<typename BinaryOp, int k>
struct ScalarBinaryOpTraits<double, SparseAD::TDenseDual<k>, BinaryOp>
{
    typedef SparseAD::TDenseDual<k> ReturnType;
};

}