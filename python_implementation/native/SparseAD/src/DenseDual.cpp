#include "../include/DenseDual.h"

#include "../include/HessianProjection.h"

namespace SparseAD {

int DenseDual::num_vars = 0;

void DenseDual::set_num_vars(int new_num_vars) { DenseDual::num_vars = new_num_vars; }

DenseDual::DenseDual(DenseDual&& other) noexcept = default;
DenseDual::DenseDual(const DenseDual& other) noexcept = default;
DenseDual::DenseDual() : DenseDual(0.0) {}
DenseDual::DenseDual(double val)
    : _val(val), _grad(Eigen::MatrixXd::Zero(num_vars, 1)),
     _hessian(Eigen::MatrixXd::Zero(num_vars, num_vars)) {}
DenseDual::DenseDual(const double val, int index) : DenseDual(val) {
  _grad(index, 0) = 1.0;
}

DenseDual& DenseDual::operator=(DenseDual&&) = default;

double DenseDual::val() const { return _val; }
Eigen::MatrixXd DenseDual::grad() const { return _grad; }
Eigen::VectorXd DenseDual::dense_grad() const { return _grad; }
Eigen::MatrixXd DenseDual::hessian() const { return _hessian; }
int DenseDual::n_vars() const { return _grad.rows(); }

DenseDual::operator std::tuple<double, Eigen::VectorXd,
                          Eigen::MatrixXd>() const {
  return {_val, _grad, _hessian};
}

std::ostream& operator<<(std::ostream& s, const DenseDual& DenseDual) {
  s << "Val: " << DenseDual._val << std::endl
    << "Grad: " << std::endl
    << DenseDual._grad << std::endl
    << "Hessian: " << std::endl
    << DenseDual._hessian << std::endl;
  return s;
}

DenseDual& DenseDual::operator*=(const DenseDual& b) {
  Eigen::MatrixXd grad_b_grad_t = _grad * b._grad.transpose();
  _hessian = b._val * _hessian + grad_b_grad_t +
             grad_b_grad_t.transpose() + _val * b._hessian;
  _grad = b._val * _grad + _val * b._grad;
  _val *= b._val;
  return *this;
}
DenseDual& DenseDual::operator*=(double b) {
  _val *= b;
  _grad *= b;
  _hessian *= b;
  return *this;
}
DenseDual& DenseDual::operator/=(const DenseDual& b) {
  _grad = (b._val * _grad - _val * b._grad) / (b._val * b._val);
  _val /= b._val;
  _hessian = (_hessian - _grad * b._grad.transpose() -
              b._grad * _grad.transpose() - _val * b._hessian) /
             b._val;
  return *this;
}
DenseDual& DenseDual::operator/=(double b) {
  _val /= b;
  _grad /= b;
  _hessian /= b;
  return *this;
}
DenseDual& DenseDual::operator+=(const DenseDual& b) {
  _val += b._val;
  _grad += b._grad;
  _hessian += b._hessian;
  return *this;
}
DenseDual& DenseDual::operator+=(double b) {
  _val += b;
  return *this;
}
DenseDual& DenseDual::operator-=(const DenseDual& b) {
  _val -= b._val;
  _grad -= b._grad;
  _hessian -= b._hessian;
  return *this;
}
DenseDual& DenseDual::operator-=(double b) {
  _val -= b;
  return *this;
}
DenseDual& DenseDual::chain_this(double val, double grad, double hessian) {
  _val = val;
  // _hessian = hessian * _grad * _grad.transpose() + grad * _hessian;
  _hessian *= grad;
  _hessian += hessian * _grad * _grad.transpose();
  _grad *= grad;
  return *this;
}

DenseDual DenseDual::chain(double val, double grad, double hessian) const {
  DenseDual res(*this);
  res.chain_this(val, grad, hessian);
  return res;
}

DenseDual DenseDual::inv() const {
  double valsqr = _val * _val;
  double valcube = valsqr * _val;
  return chain(1 / _val, -1 / valsqr, 2 / valcube);
}

DenseDual& DenseDual::inv_self() {
  double valsqr = _val * _val;
  double valcube = valsqr * _val;
  chain_this(1 / _val, -1 / valsqr, 2 / valcube);
  return *this;
}

DenseDual& DenseDual::neg() {
  chain_this(-_val, -1.0, 0.0);
  return *this;
}

DenseDual& DenseDual::projectHessian() {
  _hessian = project_hessian(_hessian);
  return *this;
}

DenseDual operator*(const DenseDual& a, const DenseDual& b) {
  DenseDual res(a);
  res *= b;
  return res;
}
DenseDual operator*(DenseDual&& a, const DenseDual& b) {
  a *= b;
  return a;
}
DenseDual operator*(const DenseDual& a, DenseDual&& b) {
  return std::move(b) * a;
}
DenseDual operator*(DenseDual&& a, DenseDual&& b) {
  return std::move(a) * b;
}

DenseDual operator*(double b, const DenseDual& a) {
  DenseDual res = a;
  res *= b;
  return res;
}
DenseDual operator*(const DenseDual& a, double b) { 
  return b * a;
}
DenseDual operator*(double b, DenseDual&& a) {
  a *= b;
  return a;
}
DenseDual operator*(DenseDual&& a, double b) {
  a *= b;
  return a;
}


DenseDual operator/(const DenseDual& a, const DenseDual& b) {
  DenseDual res(a);
  res /= b;
  return res;
}
DenseDual operator/(DenseDual&& a, const DenseDual& b) {
  a /= b;
  return a;
}
DenseDual operator/(const DenseDual& a, DenseDual&& b) {
  return a / b;
}
DenseDual operator/(DenseDual&& a, DenseDual&& b) {
  return std::move(a) / b;
}
DenseDual operator/(double b, const DenseDual& a) {
  DenseDual res = a.inv();
  res *= b;
  return res;
}
DenseDual operator/(const DenseDual& a, double b) {
  DenseDual res(a);
  res /= b;
  return res;
}
DenseDual operator/(double b, DenseDual&& a) {
  a.inv_self() *= b;
  return a;
}
DenseDual operator/(DenseDual&& a, double b) {
  a /= b;
  return a;
}
/* ------------------------ Operator+ ------------------------ */
DenseDual operator+(const DenseDual& a, const DenseDual& b) {
  DenseDual res(a);
  res += b;
  return res;
}
DenseDual operator+(DenseDual&& a, const DenseDual& b) {
  a+=b;
  return a;
}

DenseDual operator+(const DenseDual& a, DenseDual&& b) {
  return std::move(b) + a;
}
DenseDual operator+(DenseDual&& a, DenseDual&& b) {
  return std::move(a) + b;
}

DenseDual operator+(const DenseDual& a, double b) { 
  DenseDual res(a);
  res += b;
  return res;
}
DenseDual operator+(double b, const DenseDual& a) {
  return a+b;
}
DenseDual operator+(double b, DenseDual&& a) {
  a += b;
  return a;
}

DenseDual operator+(DenseDual&& a, double b) {
  a += b;
  return a;
}

/* ------------------------ Operator- ------------------------ */
DenseDual operator-(const DenseDual& a, const DenseDual& b) {
  DenseDual res(a);
  res -= b;
  return res;
}
DenseDual operator-(DenseDual&& a, const DenseDual& b) {
  a -= b;
  return a;
}
DenseDual operator-(const DenseDual& a, DenseDual&& b) {
  b.neg() += a;
  return b;
}
DenseDual operator-(DenseDual&& a, DenseDual&& b) {
  return std::move(a) - b;
}

DenseDual operator-(double b, const DenseDual& a) {
  DenseDual res(-a);
  res += b;
  return res;
}
DenseDual operator-(const DenseDual& a, double b) {
  DenseDual res(a);
  res -= b;
  return res;
}
DenseDual operator-(double b, DenseDual&& a) {
  a.neg() += b;
  return a;
}
DenseDual operator-(DenseDual&& a, double b) {
  a -= b;
  return a;
}
DenseDual operator-(const DenseDual& a) {
  return a.chain(-a._val, -1.0, 0.0);
}
DenseDual operator-(DenseDual&& a) {
  a.neg();
  return a;
}
DenseDual sqrt(const DenseDual& a) {
  const auto& sqrt_a = std::sqrt(a._val);
  return a.chain(sqrt_a, 0.5 / sqrt_a, -0.25 / (sqrt_a * a._val));
}
DenseDual sqrt(DenseDual&& a) {
  const auto& sqrt_a = std::sqrt(a._val);
  a.chain_this(sqrt_a, 0.5 / sqrt_a, -0.25 / (sqrt_a * a._val));
  return a;
}
DenseDual sqr(const DenseDual& a) { return a * a; }
DenseDual sqr(DenseDual&& a) {
  a *= a;
  return a;
}

DenseDual pow(const DenseDual& a, const int exponent) {
  const double f2 = std::pow(a.val(), exponent - 2);
  const double f1 = f2 * a.val();
  const double f = f1 * a.val();

  return a.chain(f, exponent * f1, exponent * (exponent - 1) * f2);
}
DenseDual pow(const DenseDual& a, const double exponent) {
  const double f2 = std::pow(a.val(), exponent - 2);
  const double f1 = f2 * a.val();
  const double f = f1 * a.val();

  return a.chain(f, exponent * f1, exponent * (exponent - 1) * f2);
}
DenseDual pow(DenseDual&& a, const int exponent) {
  const double f2 = std::pow(a.val(), exponent - 2);
  const double f1 = f2 * a.val();
  const double f = f1 * a.val();
  a.chain_this(f, exponent * f1, exponent * (exponent - 1) * f2);
  return a;
}
DenseDual pow(DenseDual&& a, const double exponent) {
  const double f2 = std::pow(a.val(), exponent - 2);
  const double f1 = f2 * a.val();
  const double f = f1 * a.val();
  a.chain_this(f, exponent * f1, exponent * (exponent - 1) * f2);
  return a;
}

DenseDual exp(const DenseDual& a) {
  const double val = std::exp(a._val);
  return a.chain(val, val, val);
}
DenseDual exp(DenseDual&& a) {
  const double val = std::exp(a._val);
  a.chain_this(val, val, val);
  return a;
}

// ----------------------- Comparisons -----------------------
bool operator<(const DenseDual& a, const DenseDual& b) { return a._val < b._val; }
bool operator<=(const DenseDual& a, const DenseDual& b) { return a._val <= b._val; }
bool operator>(const DenseDual& a, const DenseDual& b) { return a._val > b._val; }
bool operator>=(const DenseDual& a, const DenseDual& b) { return a._val >= b._val; }
bool operator==(const DenseDual& a, const DenseDual& b) { return a._val == b._val; }
bool operator!=(const DenseDual& a, const DenseDual& b) { return a._val != b._val; }

}  // namespace SparseAD