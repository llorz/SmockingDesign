#pragma once
#include <Eigen/Eigen>
#include <vector>

#include "Common.h"
#include "DenseDual.h"
#include "DualFactory.h"
#include "Utils.h"

namespace SparseAD {

template <typename EnergyProvider>
struct Energy {
  double weight = 1.0;
  bool project_hessian = true;
  EnergyProvider provider;
};

class Problem {
 public:
  using EnergyFunc = std::function<Dual::Tup(const TGenericVariableFactory<Dual>&)>;
  template <typename T>
  using GenericEnergyFunc = std::function<T(const TGenericVariableFactory<T>&)>;

  using ValueEnergyFunc = std::function<double(const ValFactory<double>&)>;
  struct Options {
    // Whether to cache the hessian pattern.
    // Set to true when the structure of the hessian is fixed.
    bool cache_pattern = false;
    // Remove unreferenced nodes when taking the step.
    bool remove_unreferenced = true;
    // Whether to try -grad direction (to escape concave regions).
    bool try_grad_dir = false;

    // Which reporting to do.
    enum { NONE = 0, EVERY_STEP = 1, SUMMARY = 2 } report_level = EVERY_STEP;

    int num_iterations = 50;
    int line_search_iterations = 10;
    double step_decrease_factor = 0.6;
    std::ostream& report_stream = std::cout;
  };

  Problem(const Eigen::MatrixXd& init);
  Problem(const Eigen::MatrixXd& init, const Options& options);
  Problem(const std::vector<Eigen::VectorXd>& init);
  Problem(const std::vector<Eigen::VectorXd>& init, const Options& options);

  Problem& optimize(const EnergyFunc& func);
  Problem& optimize();

  /* Energy with autodiff and val only func. */
  struct InternalEnergy {
    EnergyFunc derivatives_func;
    ValueEnergyFunc value_func;
    double weight;
  };
  template <typename EnergyProvider>
  inline Problem& add_sparse_energy(int num, EnergyProvider provider) {
    return add_sparse_energy(num, Energy<EnergyProvider> {.provider=provider});
  }
  template <typename G>
  Problem& add_sparse_energy(int num, Energy<G> energy) {
    energies.push_back(InternalEnergy {
        .derivatives_func = sparse_func(num,
                     [energy](int num, const auto& vars) {
                       return energy.provider.template operator()<Dual>(num,
                                                                        vars);
                     }, energy.project_hessian),
         .value_func =val_func(num, [energy](int num, const auto& vars) {
           return energy.provider.template operator()<double>(num, vars);
         }),.weight = energy.weight});
    return *this;
  }
  template <int k, typename EnergyProvider>
  inline Problem& add_sparse_energy(int num, EnergyProvider provider) {
    return add_sparse_energy<k>(num, Energy<EnergyProvider> {.provider=provider});
  }
  template <int k, typename G>
  Problem& add_sparse_energy(int num, Energy<G> energy) {
    energies.push_back(InternalEnergy{
        .derivatives_func = sparse_func<k>(
            num,
            [energy](int num, auto& vars) {
              return energy.provider.template operator()<TDenseDual<k>>(num, vars);
            }, energy.project_hessian),
        .value_func = val_func(
            num,
            [energy](int num, auto& vars) {
              return energy.provider.template operator()<double>(num, vars);
            }),
        .weight = energy.weight});
    return *this;
  }

  template<typename EnergyProvider>
  Problem& add_energy(EnergyProvider provider) {
    return add_energy(Energy<EnergyProvider> {.provider=provider});
  }

  template<typename G>
  Problem& add_energy(Energy<G> energy) {
    energies.push_back(InternalEnergy{
        .derivatives_func = 
            [energy](const auto& vars) {
              return energy.provider.template operator()<Dual>(vars);
            },
        .value_func = [energy](const auto& vars) {
              return energy.provider.template operator()<double>(vars);
            },
        .weight = energy.weight});
    return *this;
  }

  Eigen::MatrixXd& current();
  Options& options();

 private:
  bool armijo_cond(double f_curr, double f_x, double step_size,
                   double dir_dot_grad, double armijo_const);

  Eigen::MatrixXd line_search(double f, const Eigen::VectorXd& dir,
                              double dir_dot_grad, const EnergyFunc& func,
                              double& step_size);
  Eigen::MatrixXd line_search(
      double f, const Eigen::VectorXd& dir, double dir_dot_grad,
      double& step_size, double& new_f);

 private:
  Options _options;
  bool first_solve;
  Eigen::MatrixXd _cur;
  std::vector<int> block_start_indices;
  std::vector<InternalEnergy> energies;
  Eigen::SimplicialLDLT<Eigen::SparseMatrix<double>> solver;
};

}  // namespace SparseAD
