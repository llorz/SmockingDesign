#pragma once
#include <Eigen/Eigen>
#include <vector>

#include "Common.h"
#include "VarFactory.h"
#include "Utils.h"

#define FACTORY_TYPE(x) typename std::decay_t<decltype(x)>::Scalar
#define IS_VAL_FACTORY(x) std::is_same_v<FACTORY_TYPE(x), double>

#define IS_ENERGY_FUNC(x) \
  std::enable_if_t<std::is_invocable_v<x, VarFactory&>, bool> = true
#define IS_TEMPLATE_ENERGY_FUNC(x) \
  std::enable_if_t<not std::is_invocable_v<x, VarFactory&>, bool> = true

#define CONVERT_TEMPLATE_ENERGY_FUNC(energy)                     \
  [&](auto& vars) {                                              \
    return energy.template operator()<FACTORY_TYPE(vars)>(vars); \
  }

#define IS_ELEMENT_FUNC(x) \
  std::enable_if_t<std::is_invocable_v<x, int, VarFactory&>, bool> = true
#define IS_TEMPLATE_ELEMENT_FUNC(x) \
  std::enable_if_t<not std::is_invocable_v<x, int, VarFactory&>, bool> = true

#define CONVERT_TEMPLATE_ELEMENT_FUNC(energy)                       \
  [&, energy](int i, auto& vars) {                                  \
    return energy.template operator()<FACTORY_TYPE(vars)>(i, vars); \
  }

#define TEMPLATE_ELEMENT_FUNC_FOR(func)                                        \
  template <typename EnergyProvider, IS_TEMPLATE_ELEMENT_FUNC(EnergyProvider)> \
  Problem& func(int num, const EnergyProvider& energy,                         \
                bool project_hessian = true) {                                 \
    return func(num, CONVERT_TEMPLATE_ELEMENT_FUNC(energy), project_hessian);  \
  }

#define TEMPLATE_FUNC_FOR(func)                                                \
  template <typename EnergyProvider, IS_TEMPLATE_ELEMENT_FUNC(EnergyProvider)> \
  Problem& func(const EnergyProvider& energy, bool project_hessian = true) {   \
    return func(CONVERT_TEMPLATE_ELEMENT_FUNC(energy), project_hessian);       \
  }

#define CONVERT_CONSTRAINT_FUNC(energy, start_ind, constraint_index)     \
  [&, energy, start_ind, constraint_index](int i,                        \
                                           auto& x) -> FACTORY_TYPE(x) { \
    if constexpr (IS_VAL_FACTORY(x)) {                                   \
      auto e = energy(i, x);                                             \
      return std::abs(e);                                                \
    } else {                                                             \
      auto e = energy(i, x);                                             \
      auto res = x(start_ind + i) * e;                                   \
                                                                         \
      return res;                                                        \
    }                                                                    \
  }

namespace Optiz {

template <typename EnergyProvider>
struct Energy {
  bool project_hessian = true;
  const EnergyProvider& provider;
};

class Problem {
 public:
  using ValueEnergyFunc = std::function<double(const ValFactory<double>&)>;
  struct Options {
    // Whether to cache the hessian pattern.
    // Set to true when the structure of the hessian is fixed.
    bool cache_pattern = true;
    // Remove unreferenced nodes when taking the step.
    bool remove_unreferenced = false;

    // Which reporting to do.
    enum report_level_enum {
      NONE = 0,
      EVERY_STEP = 1,
      SUMMARY = 2
    } report_level = EVERY_STEP;

    int num_iterations = 50;
    int line_search_iterations = 10;
    double step_decrease_factor = 0.6;

    // Stop the optimization if the relative change to the energy
    // is less than this threshold.
    double relative_change_tolerance = 1e-6;

    double gamma_theta = 1e-4;

    inline Options set_iters(int num) {
      num_iterations = num;
      return *this;
    }
    inline Options set_line_search_iters(int num) {
      line_search_iterations = num;
      return *this;
    }
    inline Options set_step_decrease_factor(double factor) {
      step_decrease_factor = factor;
      return *this;
    }
    inline Options set_report_level(report_level_enum level) {
      report_level = level;
      return *this;
    }
    inline Options set_relative_change_tolerance(double tol) {
      relative_change_tolerance = tol;
      return *this;
    }
  };

  Problem(const Eigen::MatrixXd& init);
  Problem(const Eigen::MatrixXd& init, const Options& options);
  Problem(const std::vector<Eigen::VectorXd>& init);
  Problem(const std::vector<Eigen::VectorXd>& init, const Options& options);

  Problem& optimize();

  /**
   * @brief Get the derivatives of the energy at the current state.
   *
   * @return std::tuple<double, Eigen::VectorXd&, Eigen::SparseMatrix<double>&>
   * containing the energy value, the gradient and the hessian.
   */
  std::tuple<double, Eigen::VectorXd&, Eigen::SparseMatrix<double>&>
  calc_derivatives();

  /**
   * @brief Get the value of the energy at the current state.
   *
   * @return double The energy value.
   */
  double calc_value();

  // ******************************* Energies ******************************* //
  /* Energy with autodiff and val only func. */
  struct InternalEnergy {
    EnergyFunc derivatives_func;
    ValueEnergyFunc value_func;
  };
  template <typename EnergyProvider, IS_ELEMENT_FUNC(EnergyProvider)>
  Problem& add_element_energy(int num, const EnergyProvider& energy,
                              bool project_hessian = true) {
    energies.push_back(InternalEnergy{
        .derivatives_func = element_func(num, energy, project_hessian),
        .value_func = val_func(num, energy)});
    return *this;
  }
  template <typename EnergyProvider, IS_TEMPLATE_ELEMENT_FUNC(EnergyProvider)>
  Problem& add_element_energy(int num, const EnergyProvider& energy,
                              bool project_hessian = true) {
    return add_element_energy(num, CONVERT_TEMPLATE_ELEMENT_FUNC(energy),
                              project_hessian);
  }

  template <typename EnergyProvider, IS_ELEMENT_FUNC(EnergyProvider)>
  Problem& add_eq_constraints(int num, const EnergyProvider& energy) {
    int start_ind = _cur.rows();
    int constraint_index = constraints_energies.size();
    int energy_index = energies.size();
    constraints_energies.push_back(energy_index);
    add_element_energy(
        num, CONVERT_CONSTRAINT_FUNC(energy, start_ind, constraint_index),
        false);
    _cur.conservativeResize(_cur.rows() + num, _cur.cols());
    _cur.block(_cur.rows() - num, 0, num, 1).setOnes();
    return *this;
  }

  template <int k, typename EnergyProvider,
            IS_TEMPLATE_ELEMENT_FUNC(EnergyProvider)>
  Problem& add_element_energy(int num, const EnergyProvider& energy,
                              bool project_hessian = true) {
    return add_element_energy<k>(num, CONVERT_TEMPLATE_ELEMENT_FUNC(energy),
                                 project_hessian);
  }
  template <int k, typename EnergyProvider, IS_ELEMENT_FUNC(EnergyProvider)>
  Problem& add_element_energy(int num, const EnergyProvider& energy,
                              bool project_hessian = true) {
    energies.push_back(InternalEnergy{
        .derivatives_func = element_func<k>(num, energy, project_hessian),
        .value_func = val_func(num, energy)});
    return *this;
  }

  template <typename EnergyProvider, IS_ENERGY_FUNC(EnergyProvider)>
  Problem& add_energy(EnergyProvider provider) {
    energies.push_back(
        InternalEnergy{.derivatives_func = provider, .value_func = provider});
    return *this;
  }
  template <typename EnergyProvider, IS_TEMPLATE_ENERGY_FUNC(EnergyProvider)>
  Problem& add_energy(EnergyProvider energy) {
    return add_energy(CONVERT_TEMPLATE_ENERGY_FUNC(energy));
  }

  // Log barrier energy.
  template <typename EnergyProvider, IS_ENERGY_FUNC(EnergyProvider)>
  Problem& add_log_barrier_gt_0(EnergyProvider provider) {
    return add_energy([provider](auto& vars) {
      auto val = provider(vars);
      return (val <= 0) ? std::numeric_limits<double>::infinity()
                        : -log(val);
    });
  }
  template <typename EnergyProvider, IS_TEMPLATE_ENERGY_FUNC(EnergyProvider)>
  Problem& add_log_barrier_gt_0(EnergyProvider energy) {
    return add_log_barrier_gt_0(CONVERT_TEMPLATE_ENERGY_FUNC(energy));
  }

  template <typename EnergyProvider, IS_ELEMENT_FUNC(EnergyProvider)>
  Problem& add_log_barrier_gt_0(int num, const EnergyProvider& energy,
                              bool project_hessian = true) {
    auto func = [energy](int i, auto& vars) {
      auto val = energy(i, vars);
      return (val <= 0) ? std::numeric_limits<double>::infinity()
                        : -log(val);
    };
    energies.push_back(InternalEnergy{
        .derivatives_func = element_func(num, func, project_hessian),
        .value_func = val_func(num, func)});
    return *this;
  }
  template <typename EnergyProvider, IS_TEMPLATE_ELEMENT_FUNC(EnergyProvider)>
  Problem& add_log_barrier_gt_0(int num, const EnergyProvider& energy,
                              bool project_hessian = true) {
    return add_log_barrier_gt_0(num, CONVERT_TEMPLATE_ELEMENT_FUNC(energy),
                              project_hessian);
  }

  void set_end_iteration_callback(std::function<void()> callback);

  Eigen::Map<Eigen::MatrixXd> x();
  inline double last_f() { return _last_f; }
  inline Eigen::VectorXd& last_grad() { return _last_grad; }
  inline Eigen::SparseMatrix<double>& last_hessian() { return _last_hessian; }
  Options& options();

 private:
  bool armijo_cond(double f_curr, double f_x, double step_size,
                   double dir_dot_grad, double armijo_const);

  Eigen::VectorXd line_search(const Eigen::VectorXd& cur, double f,
                              const Eigen::VectorXd& dir, double dir_dot_grad,
                              double& step_size, double& new_f);

  Eigen::VectorXd line_search_constrained(
      const Eigen::VectorXd& cur, const Eigen::VectorXd& dir, double& step_size,
      double& new_f, std::vector<std::pair<double, double>>& filter);

  ValFactory<double> val_factory(const Eigen::VectorXd& x) const;

  void analyze_pattern();

  Eigen::VectorXd factorize_and_solve();

 private:
  Options _options;
  bool first_solve;
  Eigen::VectorXd _cur;
  std::pair<int, int> _cur_shape;

  std::vector<int> block_start_indices;
  std::vector<InternalEnergy> energies;
  Eigen::SimplicialLDLT<Eigen::SparseMatrix<double>> solver;
  Eigen::SparseLU<Eigen::SparseMatrix<double>> lu_solver;
  std::function<void()> _end_iteration_callback = []() {};

  std::vector<int> constraints_energies;

  double _last_constraint_error;
  double _last_f;
  Eigen::VectorXd _last_grad;
  Eigen::SparseMatrix<double> _last_hessian;
};

}  // namespace Optiz
