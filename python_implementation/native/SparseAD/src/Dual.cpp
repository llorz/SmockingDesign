#if __INTELLISENSE__
#undef __ARM_NEON
#undef __ARM_NEON__
#endif

#include "../include/Dual.h"

#include "../include/HessianProjection.h"

namespace SparseAD {

int Dual::num_vars = 0;

void Dual::set_num_vars(int new_num_vars) { Dual::num_vars = new_num_vars; }

Dual::Dual(Dual&& other) noexcept = default;
Dual::Dual(const Dual& other) noexcept = default;
Dual::Dual() : Dual(0.0) {}
Dual::Dual(double val)
    : _val(val), _grad(num_vars, 1), _hessian(num_vars, num_vars) {}
Dual::Dual(const double val, int index) : Dual(val) {
  _grad.insert(index, 0) = 1.0;
}

Dual& Dual::operator=(Dual&&) = default;

double Dual::val() const { return _val; }
SparseMapMatrix Dual::grad() const { return _grad; }
Eigen::VectorXd Dual::dense_grad() const { return _grad.toDense(); }
SparseMapMatrix Dual::hessian() const { return _hessian; }
int Dual::n_vars() const { return _grad.rows(); }

Dual::operator std::tuple<double, Eigen::VectorXd,
                          Eigen::SparseMatrix<double>>() const {
  return {_val, _grad.toDense(), _hessian};
}

std::ostream& operator<<(std::ostream& s, const Dual& dual) {
  s << "Val: " << dual._val << std::endl
    << "Grad: " << std::endl
    << dual._grad.toDense() << std::endl
    << "Hessian: " << std::endl
    << dual._hessian.toDense() << std::endl;
  return s;
}

Dual& Dual::operator*=(const Dual& b) {
  SparseMapMatrix grad_b_grad_t = _grad * b._grad.transpose();
  _hessian = b._val * _hessian + grad_b_grad_t +
             grad_b_grad_t.transpose() + _val * b._hessian;
  _grad = b._val * _grad + _val * b._grad;
  _val *= b._val;
  return *this;
}
Dual& Dual::operator*=(double b) {
  _val *= b;
  _grad *= b;
  _hessian *= b;
  return *this;
}
Dual& Dual::operator/=(const Dual& b) {
  _grad = (b._val * _grad - _val * b._grad) / (b._val * b._val);
  _val /= b._val;
  SparseMapMatrix grad_b_grad_t = _grad * b._grad.transpose();
  _hessian = (_hessian - grad_b_grad_t -
              grad_b_grad_t.transpose() - _val * b._hessian) /
             b._val;
  return *this;
}
Dual& Dual::operator/=(double b) {
  _val /= b;
  _grad /= b;
  _hessian /= b;
  return *this;
}
Dual& Dual::operator+=(const Dual& b) {
  _val += b._val;
  _grad += b._grad;
  _hessian += b._hessian;
  return *this;
}
Dual& Dual::operator+=(double b) {
  _val += b;
  return *this;
}
Dual& Dual::operator-=(const Dual& b) {
  _val -= b._val;
  _grad -= b._grad;
  _hessian -= b._hessian;
  return *this;
}
Dual& Dual::operator-=(double b) {
  _val -= b;
  return *this;
}
Dual& Dual::chain_this(double val, double grad, double hessian) {
  _val = val;
  // _hessian = hessian * _grad * _grad.transpose() + grad * _hessian;
  _hessian *= grad;
  _hessian += hessian * _grad * _grad.transpose();
  _grad *= grad;
  return *this;
}

Dual Dual::chain(double val, double grad, double hessian) const {
  Dual res(*this);
  res.chain_this(val, grad, hessian);
  return res;
}

Dual Dual::inv() const {
  double valsqr = _val * _val;
  double valcube = valsqr * _val;
  return chain(1 / _val, -1 / valsqr, 2 / valcube);
}

Dual& Dual::inv_self() {
  double valsqr = _val * _val;
  double valcube = valsqr * _val;
  chain_this(1 / _val, -1 / valsqr, 2 / valcube);
  return *this;
}

Dual& Dual::neg() {
  chain_this(-_val, -1.0, 0.0);
  return *this;
}

Dual& Dual::projectHessian() {
  _hessian = project_hessian(_hessian);
  return *this;
}

void Dual::reserve(int n) {
  // _grad.reserve(n);
  // _hessian.reserve(n*n);
}

Dual operator*(const Dual& a, const Dual& b) {
  Dual res(a);
  res *= b;
  return res;
}
Dual operator*(Dual&& a, const Dual& b) {
  a *= b;
  return a;
}
Dual operator*(const Dual& a, Dual&& b) {
  return std::move(b) * a;
}
Dual operator*(Dual&& a, Dual&& b) {
  return std::move(a) * b;
}

Dual operator*(double b, const Dual& a) {
  Dual res = a;
  res *= b;
  return res;
}
Dual operator*(const Dual& a, double b) { 
  return b * a;
}
Dual operator*(double b, Dual&& a) {
  a *= b;
  return a;
}
Dual operator*(Dual&& a, double b) {
  a *= b;
  return a;
}


Dual operator/(const Dual& a, const Dual& b) {
  Dual res(a);
  res /= b;
  return res;
}
Dual operator/(Dual&& a, const Dual& b) {
  a /= b;
  return a;
}
Dual operator/(const Dual& a, Dual&& b) {
  return a / b;
}
Dual operator/(Dual&& a, Dual&& b) {
  return std::move(a) / b;
}
Dual operator/(double b, const Dual& a) {
  Dual res = a.inv();
  res *= b;
  return res;
}
Dual operator/(const Dual& a, double b) {
  Dual res(a);
  res /= b;
  return res;
}
Dual operator/(double b, Dual&& a) {
  a.inv_self() *= b;
  return a;
}
Dual operator/(Dual&& a, double b) {
  a /= b;
  return a;
}
/* ------------------------ Operator+ ------------------------ */
Dual operator+(const Dual& a, const Dual& b) {
  Dual res(a);
  res += b;
  return res;
}
Dual operator+(Dual&& a, const Dual& b) {
  a+=b;
  return a;
}

Dual operator+(const Dual& a, Dual&& b) {
  return std::move(b) + a;
}
Dual operator+(Dual&& a, Dual&& b) {
  return std::move(a) + b;
}

Dual operator+(const Dual& a, double b) { 
  Dual res(a);
  res += b;
  return res;
}
Dual operator+(double b, const Dual& a) {
  return a+b;
}
Dual operator+(double b, Dual&& a) {
  a += b;
  return a;
}

Dual operator+(Dual&& a, double b) {
  a += b;
  return a;
}

/* ------------------------ Operator- ------------------------ */
Dual operator-(const Dual& a, const Dual& b) {
  Dual res(a);
  res -= b;
  return res;
}
Dual operator-(Dual&& a, const Dual& b) {
  a -= b;
  return a;
}
Dual operator-(const Dual& a, Dual&& b) {
  b.neg() += a;
  return b;
}
Dual operator-(Dual&& a, Dual&& b) {
  return std::move(a) - b;
}

Dual operator-(double b, const Dual& a) {
  Dual res(-a);
  res += b;
  return res;
}
Dual operator-(const Dual& a, double b) {
  Dual res(a);
  res -= b;
  return res;
}
Dual operator-(double b, Dual&& a) {
  a.neg() += b;
  return a;
}
Dual operator-(Dual&& a, double b) {
  a -= b;
  return a;
}
Dual operator-(const Dual& a) {
  return a.chain(-a._val, -1.0, 0.0);
}
Dual operator-(Dual&& a) {
  a.neg();
  return a;
}
Dual operator+(const Dual& a) {
  Dual res(a);
  res.projectHessian();
  return res;
}
Dual operator+(Dual&& a) {
  return a.projectHessian();
}
Dual sqrt(const Dual& a) {
  const auto& sqrt_a = std::sqrt(a._val);
  return a.chain(sqrt_a, 0.5 / sqrt_a, -0.25 / (sqrt_a * a._val));
}
Dual sqrt(Dual&& a) {
  const auto& sqrt_a = std::sqrt(a._val);
  a.chain_this(sqrt_a, 0.5 / sqrt_a, -0.25 / (sqrt_a * a._val));
  return a;
}
Dual sqr(const Dual& a) { 
  // return a * a; 
  return a.chain(a._val*a._val, 2 * a._val, 2);
}
Dual sqr(Dual&& a) {
  a.chain_this(a._val*a._val, 2 * a._val, 2);
  return a;
}

Dual pow(const Dual& a, const int exponent) {
  const double f2 = std::pow(a.val(), exponent - 2);
  const double f1 = f2 * a.val();
  const double f = f1 * a.val();

  return a.chain(f, exponent * f1, exponent * (exponent - 1) * f2);
}
Dual pow(const Dual& a, const double exponent) {
  const double f2 = std::pow(a.val(), exponent - 2);
  const double f1 = f2 * a.val();
  const double f = f1 * a.val();

  return a.chain(f, exponent * f1, exponent * (exponent - 1) * f2);
}
Dual pow(Dual&& a, const int exponent) {
  const double f2 = std::pow(a.val(), exponent - 2);
  const double f1 = f2 * a.val();
  const double f = f1 * a.val();
  a.chain_this(f, exponent * f1, exponent * (exponent - 1) * f2);
  return a;
}
Dual pow(Dual&& a, const double exponent) {
  const double f2 = std::pow(a.val(), exponent - 2);
  const double f1 = f2 * a.val();
  const double f = f1 * a.val();
  a.chain_this(f, exponent * f1, exponent * (exponent - 1) * f2);
  return a;
}

Dual exp(const Dual& a) {
  const double val = std::exp(a._val);
  return a.chain(val, val, val);
}
Dual exp(Dual&& a) {
  const double val = std::exp(a._val);
  a.chain_this(val, val, val);
  return a;
}

// ----------------------- Comparisons -----------------------
bool operator<(const Dual& a, const Dual& b) { return a._val < b._val; }
bool operator<=(const Dual& a, const Dual& b) { return a._val <= b._val; }
bool operator>(const Dual& a, const Dual& b) { return a._val > b._val; }
bool operator>=(const Dual& a, const Dual& b) { return a._val >= b._val; }
bool operator==(const Dual& a, const Dual& b) { return a._val == b._val; }
bool operator!=(const Dual& a, const Dual& b) { return a._val != b._val; }


}  // namespace SparseAD