#include "../include/DualM.h"

#include "../include/HessianProjection.h"

namespace SparseAD {

int DualM::num_vars = 0;

void DualM::set_num_vars(int new_num_vars) { DualM::num_vars = new_num_vars; }

DualM::DualM(DualM&& other) noexcept = default;
DualM::DualM(const DualM& other) noexcept = default;
DualM::DualM() : DualM(0.0) {}
DualM::DualM(double val)
    : _val(val), _grad(num_vars, 1), 
    _hessians {SparseMapMatrix(num_vars, num_vars),
    SparseMapMatrix(num_vars, num_vars)}, psd_index(0) {}
DualM::DualM(const double val, int index) : DualM(val) {
  _grad.insert(index, 0) = 1.0;
}

DualM& DualM::operator=(DualM&&) = default;

double DualM::val() const { return _val; }
SparseMapMatrix DualM::grad() const { return _grad; }
Eigen::VectorXd DualM::dense_grad() const { return _grad.toDense(); }
SparseMapMatrix DualM::hessian() const { return psd_hess(); }
int DualM::n_vars() const { return _grad.rows(); }

DualM::operator std::tuple<double, Eigen::VectorXd,
                          Eigen::SparseMatrix<double>>() const {
  return {_val, _grad.toDense(), psd_hess()};
}

std::ostream& operator<<(std::ostream& s, const DualM& DualM) {
  s << "Val: " << DualM._val << std::endl
    << "Grad: " << std::endl
    << DualM._grad.toDense() << std::endl
    << "Hessian: " << std::endl
    << DualM.psd_hess().toDense() << std::endl;
  return s;
}

DualM& DualM::operator*=(const DualM& b) {
    _hessians[0] *= b._val;
    _hessians[1] *= b._val;
    if (b._val < 0) {psd_index = 1 - psd_index;}
    int psdv = _val < 0? 1 - psd_index : psd_index;
    _hessians[psdv] += _val * b._hessians[b.psd_index];
    _hessians[1 - psdv] += _val * b._hessians[1 - b.psd_index];

    const auto& sum_grad = b._grad - _grad;
    _hessians[psd_index] += _grad * _grad.transpose() + b._grad * b._grad.transpose();
    _hessians[1 - psd_index] -= sum_grad * sum_grad.transpose();
 
    _grad = b._val * _grad + _val * b._grad;
    _val *= b._val;
  return *this;
}
DualM& DualM::operator*=(double b) {
  _val *= b;
    _grad *= b;
    _hessians[0] *= b;
    _hessians[1] *= b;
    if (b < 0) {psd_index = 1 - psd_index;}
  return *this;
}
DualM& DualM::operator/=(const DualM& b) {
  // _grad = (b._val * _grad - _val * b._grad) / (b._val * b._val);
  // _val /= b._val;
  // SparseMapMatrix grad_b_grad_t = _grad * b._grad.transpose();
  // _hessian = (_hessian - grad_b_grad_t -
  //             grad_b_grad_t.transpose() - _val * b._hessian) /
  //            b._val;
  return operator*=(b.inv());
}
DualM& DualM::operator/=(double b) {
  _val /= b;
  _grad /= b;
  _hessians[0] /= b;
  _hessians[1] /= b;
  if (b < 0) {psd_index = 1 - psd_index;}
  return *this;
}
DualM& DualM::operator+=(const DualM& b) {
  _val += b._val;
    _grad += b._grad;
    psd_hess() += b.psd_hess();
    nsd_hess() += b.nsd_hess();
  return *this;
}
DualM& DualM::operator+=(double b) {
  _val += b;
  return *this;
}
DualM& DualM::operator-=(const DualM& b) {
  _val -= b._val;
    _grad -= b._grad;
    psd_hess() -= b.nsd_hess();
    nsd_hess() -= b.psd_hess();
  return *this;
}
DualM& DualM::operator-=(double b) {
  _val -= b;
  return *this;
}
DualM& DualM::chain_this(double val, double grad, double hessian) {
  _val = val;
    // _hessian = hessian * _grad * _grad.transpose() + grad * _hessian;
    _hessians[0] *= grad;
    _hessians[1] *= grad;
    if (grad < 0) {psd_index = 1 - psd_index;}
    int index = hessian > 0? psd_index : 1 - psd_index;
    _hessians[index] += hessian * _grad * _grad.transpose();
    _grad *= grad;
  return *this;
}

DualM DualM::chain(double val, double grad, double hessian) const {
  DualM res(*this);
  res.chain_this(val, grad, hessian);
  return res;
}

DualM DualM::inv() const {
  double valsqr = _val * _val;
  double valcube = valsqr * _val;
  return chain(1 / _val, -1 / valsqr, 2 / valcube);
}

DualM& DualM::inv_self() {
  double valsqr = _val * _val;
  double valcube = valsqr * _val;
  chain_this(1 / _val, -1 / valsqr, 2 / valcube);
  return *this;
}

DualM& DualM::neg() {
  chain_this(-_val, -1.0, 0.0);
  return *this;
}

DualM& DualM::projectHessian() {
  return *this;
}

void DualM::reserve(int n) {
  // _grad.reserve(n);
  // _hessian.reserve(n*n);
}

DualM operator*(const DualM& a, const DualM& b) {
  DualM res(a);
  res *= b;
  return res;
}
DualM operator*(DualM&& a, const DualM& b) {
  a *= b;
  return a;
}
DualM operator*(const DualM& a, DualM&& b) {
  return std::move(b) * a;
}
DualM operator*(DualM&& a, DualM&& b) {
  return std::move(a) * b;
}

DualM operator*(double b, const DualM& a) {
  DualM res = a;
  res *= b;
  return res;
}
DualM operator*(const DualM& a, double b) { 
  return b * a;
}
DualM operator*(double b, DualM&& a) {
  a *= b;
  return a;
}
DualM operator*(DualM&& a, double b) {
  a *= b;
  return a;
}


DualM operator/(const DualM& a, const DualM& b) {
  DualM res(a);
  res /= b;
  return res;
}
DualM operator/(DualM&& a, const DualM& b) {
  a /= b;
  return a;
}
DualM operator/(const DualM& a, DualM&& b) {
  return a / b;
}
DualM operator/(DualM&& a, DualM&& b) {
  return std::move(a) / b;
}
DualM operator/(double b, const DualM& a) {
  DualM res = a.inv();
  res *= b;
  return res;
}
DualM operator/(const DualM& a, double b) {
  DualM res(a);
  res /= b;
  return res;
}
DualM operator/(double b, DualM&& a) {
  a.inv_self() *= b;
  return a;
}
DualM operator/(DualM&& a, double b) {
  a /= b;
  return a;
}
/* ------------------------ Operator+ ------------------------ */
DualM operator+(const DualM& a, const DualM& b) {
  DualM res(a);
  res += b;
  return res;
}
DualM operator+(DualM&& a, const DualM& b) {
  a+=b;
  return a;
}

DualM operator+(const DualM& a, DualM&& b) {
  return std::move(b) + a;
}
DualM operator+(DualM&& a, DualM&& b) {
  return std::move(a) + b;
}

DualM operator+(const DualM& a, double b) { 
  DualM res(a);
  res += b;
  return res;
}
DualM operator+(double b, const DualM& a) {
  return a+b;
}
DualM operator+(double b, DualM&& a) {
  a += b;
  return a;
}

DualM operator+(DualM&& a, double b) {
  a += b;
  return a;
}

/* ------------------------ Operator- ------------------------ */
DualM operator-(const DualM& a, const DualM& b) {
  DualM res(a);
  res -= b;
  return res;
}
DualM operator-(DualM&& a, const DualM& b) {
  a -= b;
  return a;
}
DualM operator-(const DualM& a, DualM&& b) {
  b.neg() += a;
  return b;
}
DualM operator-(DualM&& a, DualM&& b) {
  return std::move(a) - b;
}

DualM operator-(double b, const DualM& a) {
  DualM res(-a);
  res += b;
  return res;
}
DualM operator-(const DualM& a, double b) {
  DualM res(a);
  res -= b;
  return res;
}
DualM operator-(double b, DualM&& a) {
  a.neg() += b;
  return a;
}
DualM operator-(DualM&& a, double b) {
  a -= b;
  return a;
}
DualM operator-(const DualM& a) {
  return a.chain(-a._val, -1.0, 0.0);
}
DualM operator-(DualM&& a) {
  a.neg();
  return a;
}
DualM sqrt(const DualM& a) {
  const auto& sqrt_a = std::sqrt(a._val);
  return a.chain(sqrt_a, 0.5 / sqrt_a, -0.25 / (sqrt_a * a._val));
}
DualM sqrt(DualM&& a) {
  const auto& sqrt_a = std::sqrt(a._val);
  a.chain_this(sqrt_a, 0.5 / sqrt_a, -0.25 / (sqrt_a * a._val));
  return a;
}
DualM sqr(const DualM& a) { return a * a; }
DualM sqr(DualM&& a) {
  a *= a;
  return a;
}

DualM pow(const DualM& a, const int exponent) {
  const double f2 = std::pow(a.val(), exponent - 2);
  const double f1 = f2 * a.val();
  const double f = f1 * a.val();

  return a.chain(f, exponent * f1, exponent * (exponent - 1) * f2);
}
DualM pow(const DualM& a, const double exponent) {
  const double f2 = std::pow(a.val(), exponent - 2);
  const double f1 = f2 * a.val();
  const double f = f1 * a.val();

  return a.chain(f, exponent * f1, exponent * (exponent - 1) * f2);
}
DualM pow(DualM&& a, const int exponent) {
  const double f2 = std::pow(a.val(), exponent - 2);
  const double f1 = f2 * a.val();
  const double f = f1 * a.val();
  a.chain_this(f, exponent * f1, exponent * (exponent - 1) * f2);
  return a;
}
DualM pow(DualM&& a, const double exponent) {
  const double f2 = std::pow(a.val(), exponent - 2);
  const double f1 = f2 * a.val();
  const double f = f1 * a.val();
  a.chain_this(f, exponent * f1, exponent * (exponent - 1) * f2);
  return a;
}

DualM exp(const DualM& a) {
  const double val = std::exp(a._val);
  return a.chain(val, val, val);
}
DualM exp(DualM&& a) {
  const double val = std::exp(a._val);
  a.chain_this(val, val, val);
  return a;
}

// ----------------------- Comparisons -----------------------
bool operator<(const DualM& a, const DualM& b) { return a._val < b._val; }
bool operator<=(const DualM& a, const DualM& b) { return a._val <= b._val; }
bool operator>(const DualM& a, const DualM& b) { return a._val > b._val; }
bool operator>=(const DualM& a, const DualM& b) { return a._val >= b._val; }
bool operator==(const DualM& a, const DualM& b) { return a._val == b._val; }
bool operator!=(const DualM& a, const DualM& b) { return a._val != b._val; }


}  // namespace SparseAD