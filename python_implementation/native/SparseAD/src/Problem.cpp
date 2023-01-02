#include "../include/Problem.h"

#include <chrono>
#include <tuple>
#include <unordered_map>
using namespace std::chrono;

namespace SparseAD {

#define REPORT_ITER_START(i)                                   \
  time_point<high_resolution_clock> start, stop, stop2, stop3; \
  if (_options.report_level == Options::EVERY_STEP) {          \
    std::cout << "Iteration " << i << " => ";                  \
    start = high_resolution_clock::now();                      \
  }

#define REPORT_CALC_TIME(f)                                                    \
  if (_options.report_level == Options::EVERY_STEP) {                          \
    stop = high_resolution_clock::now();                                       \
    std::cout << "f: " << f << " ("                                            \
              << duration_cast<microseconds>(stop - start).count() << " us) "; \
  }

#define REPORT_SOLVE()                                                         \
  if (_options.report_level == Options::EVERY_STEP) {                          \
    stop2 = high_resolution_clock::now();                                      \
    std::cout << "| Solve ("                                                   \
              << duration_cast<microseconds>(stop2 - stop).count() << " us) "; \
  }

#define REPORT_LINE_SEARCH(step)                                            \
  if (_options.report_level == Options::EVERY_STEP) {                       \
    stop3 = high_resolution_clock::now();                                   \
    std::cout << (step == 0 ? "| LINE SEARCH FAILED "                       \
                            : ("| step size: " + std::to_string(step)))     \
              << " (" << duration_cast<microseconds>(stop3 - stop2).count() \
              << " us)" << std::endl;                                       \
  }

#define REPORT_CONVERGENCE(iter, f)                                       \
  if (_options.report_level != Options::NONE) {                           \
    _options.report_stream << std::endl                                   \
                           << "Converged after " << iter << " iterations" \
                           << "\n";                                       \
    _options.report_stream << "Current energy: " << f << "\n";            \
  }

#define REPORT_NOT_CONVERGED(iter, f)                                          \
  if (_options.report_level != Options::NONE) {                                \
    _options.report_stream << "No convergence after " << iter << " iterations" \
                           << "\n";                                            \
    _options.report_stream << "Current energy: " << f << "\n";                 \
  }

Problem::Problem(const Eigen::MatrixXd& init) : Problem(init, {}) {}
Problem::Problem(const Eigen::MatrixXd& init, const Problem::Options& options)
    : _options(options), first_solve(true), _cur(init) {}

Problem::Problem(const std::vector<Eigen::VectorXd>& init)
    : Problem(init, {}) {}
Problem::Problem(const std::vector<Eigen::VectorXd>& init,
                 const Options& options)
    : _options(options), first_solve(true) {
  int total_size = 0;
  for (int i = 0; i < init.size(); i++) {
    total_size += init[i].size();
  }
  _cur.resize(total_size, 1);
  // Calculate the start index for each of the blocks.
  block_start_indices.resize(init.size());
  int block_start_index = 0;
  for (int i = 0; i < init.size(); i++) {
    // Store the start index.
    block_start_indices[i] = block_start_index;
    // And copy the block to the vector.
    _cur.block(block_start_index, 0, init[i].size(), 1) = init[i];
    block_start_index += init[i].size();
  }
}

// void compress(Eigen::VectorXd& grad, Eigen::SparseMatrix<double>& hessian,
//               std::unordered_map<int, int>& compressed_index_to_uncompressed) {
//   compressed_index_to_uncompressed.clear();
//   int nnz = (grad.array() != 0.0).count();
//   if (nnz == grad.size()) {
//     // All elements are referenced.
//     return;
//   }
//   std::unordered_map<int, int> unc_to_comp;
//   int cur = 0;
//   Eigen::VectorXd new_grad(nnz);
//   for (int i = 0; i < grad.size(); i++) {
//     if (grad(i) == 0) {
//       continue;
//     }
//     unc_to_comp[i] = cur;
//     compressed_index_to_uncompressed[cur] = i;
//     new_grad(cur++) = grad(i);
//   }
//   std::vector<Eigen::Triplet<double>> triplets;
//   for (int k = 0; k < hessian.outerSize(); ++k) {
//     for (typename Eigen::SparseMatrix<double>::InnerIterator it(hessian, k); it;
//          ++it) {
//       const auto& new_row = unc_to_comp.find(it.row());
//       const auto& new_col = unc_to_comp.find(it.col());
//       if (new_row == unc_to_comp.end() || new_col == unc_to_comp.end()) {
//         continue;
//       }
//       triplets.push_back(Eigen::Triplet<double>(
//           unc_to_comp[it.row()], unc_to_comp[it.col()], it.value()));
//     }
//   }
//   hessian.resize(nnz, nnz);
//   hessian.setFromTriplets(triplets.begin(), triplets.end());
//   grad = new_grad;
// }

void compress(Eigen::VectorXd& grad, Eigen::SparseMatrix<double>& hessian,
              std::unordered_map<int, int>& compressed_index_to_uncompressed) {
  compressed_index_to_uncompressed.clear();
  std::unordered_map<int, int> unc_to_comp;
  int nnz = 0;
  for (int k = 0; k < hessian.outerSize(); ++k) {
    for (typename Eigen::SparseMatrix<double>::InnerIterator it(hessian, k); it;
         ++it) {
          if (it.value() == 0) {continue;}
          auto new_row = unc_to_comp.find(it.row());
          if (new_row == unc_to_comp.end()) {
            unc_to_comp.insert({it.row(), nnz++});
          }
         }
  }
  if (nnz == grad.size()) {
    return;
  }
  
  Eigen::VectorXd new_grad(nnz);
  for (int i = 0; i < grad.size(); i++) {
    const auto& new_index = unc_to_comp.find(i);
    if (new_index == unc_to_comp.end()) {
      continue;
    }
    compressed_index_to_uncompressed.insert({new_index->second, i});
    new_grad(new_index->second) = grad(i);
  }
  std::vector<Eigen::Triplet<double>> triplets;
  for (int k = 0; k < hessian.outerSize(); ++k) {
    for (typename Eigen::SparseMatrix<double>::InnerIterator it(hessian, k); it;
         ++it) {
      if (it.value() == 0) {continue;}
      const auto& new_row = unc_to_comp.find(it.row());
      const auto& new_col = unc_to_comp.find(it.col());
      triplets.push_back(Eigen::Triplet<double>(
          new_row->second, new_col->second, it.value()));
    }
  }
  hessian.resize(nnz, nnz);
  hessian.setFromTriplets(triplets.begin(), triplets.end());
  grad = new_grad;
}

Eigen::VectorXd uncompress(
    const Eigen::VectorXd& direction,
    std::unordered_map<int, int> compressed_index_to_uncompressed, int n) {
  Eigen::VectorXd res = Eigen::VectorXd::Zero(n);
  for (int i = 0; i < direction.size(); i++) {
    res(compressed_index_to_uncompressed[i]) = direction(i);
  }
  return res;
}

Problem& Problem::optimize(const EnergyFunc& func) {
  Dual::set_num_vars(_cur.size());
  std::unordered_map<int, int> compressed_index_to_uncompressed;
  double last_energy = 0.0;
  int i = 0;
  for (; i < _options.num_iterations; i++) {
    REPORT_ITER_START(i);
    // Calculate the function and its derivatives.
    auto [f, res_grad, res_hessian] =
        block_start_indices.empty()
            ? func(DualFactory(_cur))
            : func(VecDualFactory(_cur, block_start_indices));
    REPORT_CALC_TIME(f);
    last_energy = f;
    // If remove unreferenced is true, adjust the gradient and hessian.
    if (_options.remove_unreferenced) {
      compress(res_grad, res_hessian, compressed_index_to_uncompressed);
    }

    // Find direction.
    if (!_options.cache_pattern || (_options.cache_pattern && first_solve)) {
      solver.analyzePattern(res_hessian);
      first_solve = false;
    }
    solver.factorize(res_hessian);
    Eigen::VectorXd d = solver.solve(-res_grad);
    REPORT_SOLVE();
    // Eigen::VectorXd d =
    // ((Eigen::MatrixXd)res_hessian).llt().solve(-res_grad);
    const double& dir_dot_grad = d.dot(res_grad);
    if (-0.5 * dir_dot_grad < 1e-6) {
      REPORT_CONVERGENCE(i, last_energy);
      return *this;
    }
    if (_options.remove_unreferenced &&
        !compressed_index_to_uncompressed.empty()) {
      d = uncompress(d, compressed_index_to_uncompressed, _cur.size());
    }

    // Find new value.
    double step_size;
    _cur = line_search(f, d, dir_dot_grad, func, step_size);
    REPORT_LINE_SEARCH(step_size);
  }
  REPORT_NOT_CONVERGED(i, last_energy);
  return *this;
}

std::tuple<double, Eigen::VectorXd, Eigen::SparseMatrix<double>>
calc_energy_with_derivatives(const std::vector<Problem::InternalEnergy>& energies,
                             const TGenericVariableFactory<Dual>& factory) {
  double combined_f = 0.0;
  Eigen::VectorXd grad = Eigen::VectorXd::Zero(factory.num_vars());
  Eigen::SparseMatrix<double> hessian;
  hessian.resize(factory.num_vars(), factory.num_vars());
  for (const auto& energy : energies) {
    if (energy.weight == 0.0) {
      continue;
    }
    const auto& [f, e_grad, e_hessian] = energy.derivatives_func(factory);
    combined_f += energy.weight * f;
    grad += energy.weight * e_grad;
    hessian += energy.weight * e_hessian;
  }
  return {combined_f, grad, hessian};
}

double calc_energy(const std::vector<Problem::InternalEnergy>& energies,
                   const ValFactory<double>& factory) {
  double combined_f = 0.0;
  for (const auto& energy : energies) {
    if (energy.weight == 0.0) {
      continue;
    }
    combined_f += energy.weight * energy.value_func(factory);
  }
  return combined_f;
}

Problem& Problem::optimize() {
  Dual::set_num_vars(_cur.size());
  std::unordered_map<int, int> compressed_index_to_uncompressed;
  double last_energy = 0.0;
  int i = 0;
  for (; i < _options.num_iterations; i++) {
    REPORT_ITER_START(i);
    // Calculate the function and its derivatives.
    auto [f, res_grad, res_hessian] =
        block_start_indices.empty()
            ? calc_energy_with_derivatives(energies, DualFactory(_cur))
            : calc_energy_with_derivatives(
                  energies, VecDualFactory(_cur, block_start_indices));
    REPORT_CALC_TIME(f);
    last_energy = f;
    // If remove unreferenced is true, adjust the gradient and hessian.
    if (_options.remove_unreferenced) {
      compress(res_grad, res_hessian, compressed_index_to_uncompressed);
    }

    // Find direction.
    if (!_options.cache_pattern || (_options.cache_pattern && first_solve)) {
      solver.analyzePattern(res_hessian);
      first_solve = false;
    }
    solver.factorize(res_hessian);
    Eigen::VectorXd d = solver.solve(-res_grad);
    REPORT_SOLVE();
    const double& dir_dot_grad = d.dot(res_grad);
    if (-0.5 * dir_dot_grad < 1e-6) {
      REPORT_CONVERGENCE(i, last_energy);
      return *this;
    }
    if (_options.remove_unreferenced &&
        !compressed_index_to_uncompressed.empty()) {
      d = uncompress(d, compressed_index_to_uncompressed, _cur.size());
      res_grad = uncompress(res_grad, compressed_index_to_uncompressed, _cur.size());
    }

    // Find new value.
    double step_size, new_f;
    if (!_options.try_grad_dir) {
      _cur = line_search(f, d, dir_dot_grad, step_size, new_f);
    } else {
      auto first_cand = line_search(f, d, dir_dot_grad, step_size, new_f);
      double grad_step_size, grad_f;
      auto second_cand = line_search(f, -res_grad, -res_grad.squaredNorm(), grad_step_size, grad_f);
      if (new_f < grad_f) {
        _cur = first_cand;
      } else {
        _cur = second_cand;
        step_size = grad_step_size;
      }
    }
    REPORT_LINE_SEARCH(step_size);
    if (step_size == 0) {
      break;
    }
  }
  REPORT_NOT_CONVERGED(i, last_energy);
  return *this;
}

Eigen::MatrixXd& Problem::current() { return _cur; }

Problem::Options& Problem::options() { return _options; }

bool Problem::armijo_cond(double f_curr, double f_x, double step_size,
                          double dir_dot_grad, double armijo_const) {
  return f_x <= f_curr + armijo_const * step_size * dir_dot_grad;
}

Eigen::MatrixXd Problem::line_search(double f, const Eigen::VectorXd& dir,
                                     double dir_dot_grad,
                                     const EnergyFunc& func,
                                     double& step_size) {
  step_size = 1.0;
  Eigen::MatrixXd reshaped_dir = dir.reshaped(_cur.rows(), _cur.cols());
  for (int i = 0; i < _options.line_search_iterations; i++) {
    Eigen::MatrixXd x = _cur + step_size * reshaped_dir;
    double new_f =
        block_start_indices.empty()
            ? std::get<0>(func(DualValFactory(x)))
            : std::get<0>(func(VecDualValFactory(x, block_start_indices)));
    if (armijo_cond(f, new_f, step_size, dir_dot_grad, 1e-6)) {
      return x;
    }
    step_size *= _options.step_decrease_factor;
  }
  step_size = 0.0;
  return _cur;
}

Eigen::MatrixXd Problem::line_search(double f, const Eigen::VectorXd& dir,
                                     double dir_dot_grad,
                                     double& step_size, double& new_f) {
  step_size = 1.0;
  Eigen::MatrixXd reshaped_dir = dir.reshaped(_cur.rows(), _cur.cols());
  for (int i = 0; i < _options.line_search_iterations; i++) {
    Eigen::MatrixXd x = _cur + step_size * reshaped_dir;
    new_f = block_start_indices.empty()
    ? calc_energy(energies, ValFactory<double>(x))
    : calc_energy(energies, VecValFactory<double>(x, block_start_indices));
    if (armijo_cond(f, new_f, step_size, dir_dot_grad, 1e-6)) {
      return x;
    }
    step_size *= _options.step_decrease_factor;
  }
  step_size = 0.0;
  return _cur;
}

}  // namespace SparseAD