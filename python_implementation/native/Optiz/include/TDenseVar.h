#pragma once

#include <Eigen/Eigen>
#include <iostream>
#include <tuple>

#include "ProjectHessian.h"

namespace Optiz {

template <int k>
class TDenseVar {
 public:
  using VEC = Eigen::Matrix<double, k, 1>;
  using MAT = Eigen::Matrix<double, k, k>;

  TDenseVar() : TDenseVar(0.0) {}
  TDenseVar(double val)
      : _val(val), _grad(VEC::Zero(k)), _hessian(MAT::Zero(k, k)) {}
  TDenseVar(const double val, int index) : TDenseVar(val) {
    _grad(index) = 1.0;
  }
  TDenseVar(TDenseVar&& other) noexcept = default;
  TDenseVar(const TDenseVar& other) = default;

  TDenseVar& operator=(const TDenseVar& other) = default;
  TDenseVar& operator=(TDenseVar&&) = default;

  // Getters.
  inline double val() const { return _val; }
  inline Eigen::VectorXd grad() const { return _grad; }
  inline Eigen::MatrixXd hessian() const { return _hessian; }
  using Tup = std::tuple<double, Eigen::VectorXd, Eigen::MatrixXd>;
  inline operator Tup() const { return {_val, _grad, _hessian}; }

  TDenseVar& projectHessian() {
    _hessian = project_hessian(_hessian);
    return *this;
  }

  friend std::ostream& operator<<(std::ostream& s, const TDenseVar& var) {
    s << "Val: " << var._val << std::endl
      << "Grad: " << std::endl
      << var._grad << std::endl
      << "Hessian: " << std::endl
      << var._hessian << std::endl;
    return s;
  }

  // Operators.
  TDenseVar& operator*=(const TDenseVar& b) {
    // Eigen::Matrix<double, k, k> grad_b_grad_t = _grad * b._grad.transpose();
    // _hessian = b._val * _hessian + grad_b_grad_t + grad_b_grad_t.transpose() +
    //            _val * b._hessian;
    _hessian = (b._val * _hessian + _val * b._hessian);
    _hessian.template selfadjointView<Eigen::Lower>().rankUpdate(_grad,
    b._grad);
    _grad = b._val * _grad + _val * b._grad;
    _val *= b._val;
    return *this;
  }
  TDenseVar& operator*=(double b) {
    _val *= b;
    _grad *= b;
    _hessian *= b;
    return *this;
  }
  TDenseVar& operator/=(const TDenseVar& b) {
    _grad = (b._val * _grad - _val * b._grad) / (b._val * b._val);
    _val /= b._val;
    // _hessian.template selfadjointView<Eigen::Lower>().rankUpdate(_grad,
    // b._grad, -1.0);
    // _hessian -= _val * b._hessian;
    // _hessian /= b._val;
    _hessian = (_hessian - _grad * b._grad.transpose() -
                b._grad * _grad.transpose() - _val * b._hessian) /
               b._val;
    return *this;
    // return *this *= b.inv();
  }
  TDenseVar& operator/=(double b) {
    _val /= b;
    _grad /= b;
    _hessian /= b;
    return *this;
  }
  TDenseVar& operator+=(const TDenseVar& b) {
    _val += b._val;
    _grad += b._grad;
    _hessian += b._hessian;
    return *this;
  }
  TDenseVar& operator+=(double b) {
    _val += b;
    return *this;
  }
  TDenseVar& operator-=(const TDenseVar& b) {
    _val -= b._val;
    _grad -= b._grad;
    _hessian -= b._hessian;
    return *this;
  }
  TDenseVar& operator-=(double b) {
    _val -= b;
    return *this;
  }
  TDenseVar& chain_self(double val, double grad, double hessian) {
    _val = val;
    // _hessian = hessian * _grad * _grad.transpose() + grad * _hessian;
    _hessian *= grad;
    _hessian.template selfadjointView<Eigen::Lower>().rankUpdate(_grad,
    hessian);
    // _hessian += hessian * _grad * _grad.transpose();
    _grad *= grad;
    return *this;
  }
  TDenseVar chain(double val, double grad, double hessian) const {
    TDenseVar res(*this);
    res.chain_self(val, grad, hessian);
    return res;
  }

  TDenseVar inv() const {
    double valsqr = _val * _val;
    double valcube = valsqr * _val;
    return chain(1 / _val, -1 / valsqr, 2 / valcube);
  }
  TDenseVar& inv_self() {
    double valsqr = _val * _val;
    double valcube = valsqr * _val;
    chain_self(1 / _val, -1 / valsqr, 2 / valcube);
    return *this;
  }
  TDenseVar& neg() {
    chain_self(-_val, -1.0, 0.0);
    return *this;
  }

  // Mul operator between two TDenseVars.
  friend TDenseVar operator*(const TDenseVar& a, const TDenseVar& b) {
    TDenseVar res(a);
    res *= b;
    return res;
  }
  friend TDenseVar operator*(TDenseVar&& a, const TDenseVar& b) {
    a *= b;
    return a;
  }
  friend TDenseVar operator*(const TDenseVar& a, TDenseVar&& b) {
    return std::move(b) * a;
  }
  friend TDenseVar operator*(TDenseVar&& a, TDenseVar&& b) {
    return std::move(a) * b;
  }
  friend TDenseVar operator*(double b, const TDenseVar& a) {
    TDenseVar res = a;
    res *= b;
    return res;
  }
  friend TDenseVar operator*(const TDenseVar& a, double b) { return b * a; }
  friend TDenseVar operator*(double b, TDenseVar&& a) {
    a *= b;
    return a;
  }
  friend TDenseVar operator*(TDenseVar&& a, double b) {
    a *= b;
    return a;
  }

  // Div operator between two TDenseVars.
  friend TDenseVar operator/(const TDenseVar& a, const TDenseVar& b) {
    TDenseVar res(a);
    res /= b;
    return res;
  }
  friend TDenseVar operator/(TDenseVar&& a, const TDenseVar& b) {
    a /= b;
    return a;
  }
  friend TDenseVar operator/(const TDenseVar& a, TDenseVar&& b) {
    return a / b;
  }
  friend TDenseVar operator/(TDenseVar&& a, TDenseVar&& b) {
    return std::move(a) / b;
  }
  friend TDenseVar operator/(double b, const TDenseVar& a) {
    TDenseVar res = a.inv();
    res *= b;
    return res;
  }
  friend TDenseVar operator/(const TDenseVar& a, double b) {
    TDenseVar res(a);
    res /= b;
    return res;
  }
  friend TDenseVar operator/(double b, TDenseVar&& a) {
    a.inv_self() *= b;
    return a;
  }
  friend TDenseVar operator/(TDenseVar&& a, double b) {
    a /= b;
    return a;
  }

  // Add operator between two TDenseVars.
  friend TDenseVar operator+(const TDenseVar& a, const TDenseVar& b) {
    TDenseVar res(a);
    res += b;
    return res;
  }
  friend TDenseVar operator+(TDenseVar&& a, const TDenseVar& b) {
    a += b;
    return a;
  }
  friend TDenseVar operator+(const TDenseVar& a, TDenseVar&& b) {
    return std::move(b) + a;
  }
  friend TDenseVar operator+(TDenseVar&& a, TDenseVar&& b) {
    return std::move(a) + b;
  }
  // Add operator between TDenseVar and double
  friend TDenseVar operator+(double b, const TDenseVar& a) { return a + b; }
  friend TDenseVar operator+(double b, TDenseVar&& a) {
    a += b;
    return a;
  }
  friend TDenseVar operator+(const TDenseVar& a, double b) {
    TDenseVar res(a);
    res += b;
    return res;
  }
  friend TDenseVar operator+(TDenseVar&& a, double b) {
    a += b;
    return a;
  }

  // Sub operator between two TDenseVars.
  friend TDenseVar operator-(const TDenseVar& a, const TDenseVar& b) {
    TDenseVar res(a);
    res -= b;
    return res;
  }
  friend TDenseVar operator-(TDenseVar&& a, const TDenseVar& b) {
    a -= b;
    return a;
  }
  friend TDenseVar operator-(const TDenseVar& a, TDenseVar&& b) {
    b.neg() += a;
    return b;
  }
  friend TDenseVar operator-(TDenseVar&& a, TDenseVar&& b) {
    return std::move(a) - b;
  }
  // Sub operator between TDenseVar and double
  friend TDenseVar operator-(double b, const TDenseVar& a) {
    TDenseVar res(-a);
    res += b;
    return res;
  }
  friend TDenseVar operator-(const TDenseVar& a, double b) {
    TDenseVar res(a);
    res -= b;
    return res;
  }
  friend TDenseVar operator-(double b, TDenseVar&& a) {
    a.neg() += b;
    return a;
  }
  friend TDenseVar operator-(TDenseVar&& a, double b) {
    a -= b;
    return a;
  }

  friend TDenseVar operator-(const TDenseVar& a) {
    return a.chain(-a._val, -1.0, 0.0);
  }
  friend TDenseVar operator-(TDenseVar&& a) {
    a.neg();
    return a;
  }
  friend TDenseVar operator+(const TDenseVar& a) {
    TDenseVar res(a);
    res.projectHessian();
    return res;
  }
  friend TDenseVar operator+(TDenseVar&& a) { return a.projectHessian(); }
  friend TDenseVar sqrt(const TDenseVar& a) {
    const auto& sqrt_a = std::sqrt(a._val);
    return a.chain(sqrt_a, 0.5 / sqrt_a, -0.25 / (sqrt_a * a._val));
  }
  friend TDenseVar sqrt(TDenseVar&& a) {
    const auto& sqrt_a = std::sqrt(a._val);
    a.chain_self(sqrt_a, 0.5 / sqrt_a, -0.25 / (sqrt_a * a._val));
    return a;
  }
  friend TDenseVar sqr(const TDenseVar& a) {
    return a.chain(a._val * a._val, 2 * a._val, 2);
  }
  friend TDenseVar sqr(TDenseVar&& a) {
    a.chain_self(a._val * a._val, 2 * a._val, 2);
    return a;
  }
  friend TDenseVar abs(const TDenseVar& a) {
    return a.chain(a._val, a._val >= 0 ? 1 : -1, 0);
  }
  friend TDenseVar abs(TDenseVar&& a) {
    a.chain_self(a._val, a._val >= 0 ? 1 : -1, 0);
    return a;
  }
  friend TDenseVar pow(const TDenseVar& a, const double exponent) {
    const double f2 = std::pow(a.val(), exponent - 2);
    const double f1 = f2 * a.val();
    const double f = f1 * a.val();

    return a.chain(f, exponent * f1, exponent * (exponent - 1) * f2);
  }
  friend TDenseVar pow(const TDenseVar& a, const int exponent) {
    const double f2 = std::pow(a.val(), exponent - 2);
    const double f1 = f2 * a.val();
    const double f = f1 * a.val();

    return a.chain(f, exponent * f1, exponent * (exponent - 1) * f2);
  }
  friend TDenseVar pow(TDenseVar&& a, const int exponent) {
    const double f2 = std::pow(a.val(), exponent - 2);
    const double f1 = f2 * a.val();
    const double f = f1 * a.val();
    a.chain_self(f, exponent * f1, exponent * (exponent - 1) * f2);
    return a;
  }
  friend TDenseVar pow(TDenseVar&& a, const double exponent) {
    const double f2 = std::pow(a.val(), exponent - 2);
    const double f1 = f2 * a.val();
    const double f = f1 * a.val();
    a.chain_self(f, exponent * f1, exponent * (exponent - 1) * f2);
    return a;
  }
  friend TDenseVar exp(const TDenseVar& a) {
    const double val = std::exp(a._val);
    return a.chain(val, val, val);
  }
  friend TDenseVar exp(TDenseVar&& a) {
    const double val = std::exp(a._val);
    a.chain_self(val, val, val);
    return a;
  }
  friend TDenseVar log(const TDenseVar& a) {
    return a.chain(std::log(a._val), 1 / a._val, -1 / (a._val * a._val));
  }
  friend TDenseVar log(TDenseVar&& a) {
    a.chain_self(std::log(a._val), 1 / a._val, -1 / (a._val * a._val));
    return a;
  }

  // ----------------------- Comparisons -----------------------
  friend bool operator<(const TDenseVar& a, const TDenseVar& b) {
    return a._val < b._val;
  }
  friend bool operator<=(const TDenseVar& a, const TDenseVar& b) {
    return a._val <= b._val;
  }
  friend bool operator>(const TDenseVar& a, const TDenseVar& b) {
    return a._val > b._val;
  }
  friend bool operator>=(const TDenseVar& a, const TDenseVar& b) {
    return a._val >= b._val;
  }
  friend bool operator==(const TDenseVar& a, const TDenseVar& b) {
    return a._val == b._val;
  }
  friend bool operator!=(const TDenseVar& a, const TDenseVar& b) {
    return a._val != b._val;
  }

 private:
  double _val;
  Eigen::Matrix<double, k, 1> _grad;
  Eigen::Matrix<double, k, k> _hessian;
};

}  // namespace Optiz

namespace Eigen {

/**
 * See https://eigen.tuxfamily.org/dox/TopicCustomizing_CustomScalar.html
 * and https://eigen.tuxfamily.org/dox/structEigen_1_1NumTraits.html
 */
template <int k>
struct NumTraits<Optiz::TDenseVar<k>> : NumTraits<double> {
  typedef Optiz::TDenseVar<k> Real;
  typedef Optiz::TDenseVar<k> NonInteger;
  typedef Optiz::TDenseVar<k> Nested;

  enum {
    IsComplex = 0,
    IsInteger = 0,
    IsSigned = 1,
    RequireInitialization = 1,
    ReadCost = k * k,
    AddCost = k * k,
    MulCost = k * k * k,
  };
};

template <typename BinaryOp, int k>
struct ScalarBinaryOpTraits<Optiz::TDenseVar<k>, double, BinaryOp> {
  typedef Optiz::TDenseVar<k> ReturnType;
};

template <typename BinaryOp, int k>
struct ScalarBinaryOpTraits<double, Optiz::TDenseVar<k>, BinaryOp> {
  typedef Optiz::TDenseVar<k> ReturnType;
};

}  // namespace Eigen