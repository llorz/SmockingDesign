#pragma once

#include <Eigen/Eigen>
#include <iostream>
#include "HessianProjection.h"
#include "SparseMapMatrix.h"
#include <tuple>

namespace SparseAD {

class Dual {
 public:
 Dual();
 Dual(double val);
 Dual(const double val, int index);
 Dual(const Dual&) noexcept;
 Dual(Dual&&) noexcept;

  Dual& operator=(const Dual& other) = default;
  Dual& operator=(Dual&&);

  static void set_num_vars(int new_num_vars);

  // Getters.
  double val() const;
  inline double& val() {return _val;}
  inline SparseMapMatrix& grad() {return _grad;}
  inline SparseMapMatrix& hessian() {return _hessian;};
  SparseMapMatrix grad() const;
  Eigen::VectorXd dense_grad() const;
  SparseMapMatrix hessian() const;
  using Tup = std::tuple<double, Eigen::VectorXd, Eigen::SparseMatrix<double>>;
  operator Tup() const;

  int n_vars() const;

  Dual& projectHessian();
  void reserve(int n);

  friend std::ostream& operator<<(std::ostream& s, const Dual& dual);

  // Operators.
  Dual& operator*=(const Dual& b);
  Dual& operator*=(double b);
  Dual& operator/=(const Dual& b);
  Dual& operator/=(double b);
  Dual& operator+=(const Dual& b);
  Dual& operator+=(double b);
  Dual& operator-=(const Dual& b);
  Dual& operator-=(double b);
  Dual& chain_this(double val, double grad, double hessian);
  Dual chain(double val, double grad, double hessian) const;

  Dual inv() const;
  Dual& inv_self();
  Dual& neg();

  // Mul operator between two duals.
  friend Dual operator*(const Dual& a, const Dual& b);
  friend Dual operator*(Dual&& a, const Dual& b);
  friend Dual operator*(const Dual& a, Dual&& b);
  friend Dual operator*(Dual&& a, Dual&& b);
  friend Dual operator*(double b, const Dual& a);
  friend Dual operator*(const Dual& a, double b);
  friend Dual operator*(double b, Dual&& a);
  friend Dual operator*(Dual&& a, double b);

  // Div operator between two duals.
  friend Dual operator/(const Dual& a, const Dual& b);
  friend Dual operator/(Dual&& a, const Dual& b);
  friend Dual operator/(const Dual& a, Dual&& b);
  friend Dual operator/(Dual&& a, Dual&& b);
  friend Dual operator/(double b, const Dual& a);
  friend Dual operator/(const Dual& a, double b);
  friend Dual operator/(double b, Dual&& a);
  friend Dual operator/(Dual&& a, double b);

  // Add operator between two duals.
  friend Dual operator+(const Dual& a, const Dual& b);
  friend Dual operator+(Dual&& a, const Dual& b);
  friend Dual operator+(const Dual& a, Dual&& b);
  friend Dual operator+(Dual&& a, Dual&& b);
  // Add operator between Dual and double
  friend Dual operator+(double b, const Dual& a);
  friend Dual operator+(double b, Dual&& a);
  friend Dual operator+(const Dual& a, double b);
  friend Dual operator+(Dual&& a, double b);

  // Sub operator between two duals.
  friend Dual operator-(const Dual& a, const Dual& b);
  friend Dual operator-(Dual&& a, const Dual& b);
  friend Dual operator-(const Dual& a, Dual&& b);
  friend Dual operator-(Dual&& a, Dual&& b);
  // Sub operator between Dual and double
  friend Dual operator-(double b, const Dual& a);
  friend Dual operator-(const Dual& a, double b);
  friend Dual operator-(double b, Dual&& a);
  friend Dual operator-(Dual&& a, double b);

  friend Dual operator-(const Dual& a);
  friend Dual operator-(Dual&& a);
  friend Dual sqrt(const Dual& a);
  friend Dual sqrt(Dual&& a);
  friend Dual sqr(const Dual& a);
  friend Dual sqr(Dual&& a);
  friend Dual abs(const Dual& a);
  friend Dual abs(Dual&& a);
  friend Dual pow(const Dual& a, const double exponent);
  friend Dual pow(const Dual& a, const int exponent);
  friend Dual pow(Dual&& a, const int exponent);
  friend Dual pow(Dual&& a, const double exponent);
  friend Dual exp(const Dual& a);
  friend Dual exp(Dual&& a);

  
  // ----------------------- Comparisons -----------------------
  friend bool operator<(const Dual& a, const Dual& b);
  friend bool operator<=(const Dual& a, const Dual& b);
  friend bool operator>(const Dual& a, const Dual& b);
  friend bool operator>=(const Dual& a, const Dual& b);
  friend bool operator==(const Dual& a, const Dual& b);
  friend bool operator!=(const Dual& a, const Dual& b);

 private:
  double _val;
  SparseMapMatrix _grad;
  SparseMapMatrix _hessian;

  static int num_vars;
};

}  // namespace SparseAD

#pragma omp declare reduction (+ : SparseAD::Dual : omp_out +=  omp_in)

namespace Eigen {

/**
 * See https://eigen.tuxfamily.org/dox/TopicCustomizing_CustomScalar.html
 * and https://eigen.tuxfamily.org/dox/structEigen_1_1NumTraits.html
 */
template<>
struct NumTraits<SparseAD::Dual>: NumTraits<double>
{
    typedef SparseAD::Dual Real;
    typedef SparseAD::Dual NonInteger;
    typedef SparseAD::Dual Nested;

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
struct ScalarBinaryOpTraits<SparseAD::Dual, double, BinaryOp>
{
    typedef SparseAD::Dual ReturnType;
};

template<typename BinaryOp>
struct ScalarBinaryOpTraits<double, SparseAD::Dual, BinaryOp>
{
    typedef SparseAD::Dual ReturnType;
};

}