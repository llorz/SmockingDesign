#include <ceres/ceres.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <vector>

template <typename T>
T sqr(const T& x) {
  return x * x;
}

// The energy for the pleat nodes optimization.
struct CostFunctorPleats {
  int num_vars;
  const std::vector<std::vector<double>>& underlay_nodes;
  const std::vector<std::vector<double>>& eq_constraints;
  CostFunctorPleats(int num_vars,
                    const std::vector<std::vector<double>>& underlay_nodes,
                    const std::vector<std::vector<double>>& eq_constraints)
      : num_vars(num_vars),
        underlay_nodes(underlay_nodes),
        eq_constraints(eq_constraints) {}

  /**
   * @brief Equality constraints energy
   *
   * @param x The current set of variables (pleat nodes as a flattened vector).
   * // TODO: split 'x' into 3 vectors to make it easier to read.
   */
  template <typename T>
  T equality_constraints(T const* x) const {
    T res = T(0.0);
    for (int i = 0; i < eq_constraints.size(); i++) {
      int a = eq_constraints[i][0], b = eq_constraints[i][1];
      double d = eq_constraints[i][2];
      // If 'a' is an underlay node and 'b' is a pleat node.
      if (a < underlay_nodes.size() && b >= underlay_nodes.size()) {
        b -= underlay_nodes.size();
        res += sqr(sqrt(sqr(underlay_nodes[a][0] - x[b]) +
                        sqr(underlay_nodes[a][1] - x[b + num_vars]) +
                        sqr(underlay_nodes[a][2] - x[b + num_vars * 2])) -
                   d);
        // If 'a' is a pleat node and 'b' is an underlay node.
      } else if (a >= underlay_nodes.size() && b < underlay_nodes.size()) {
        a -= underlay_nodes.size();
        res += sqr(sqrt(sqr(underlay_nodes[b][0] - x[a]) +
                        sqr(underlay_nodes[b][1] - x[a + num_vars]) +
                        sqr(underlay_nodes[b][2] - x[a + num_vars * 2])) -
                   d);
        // Otherwise, both 'a' and 'b' are pleat nodes.
      } else {
        a -= underlay_nodes.size();
        b -= underlay_nodes.size();
        res +=
            sqr(sqrt(sqr(x[b] - x[a]) + sqr(x[b + num_vars] - x[a + num_vars]) +
                     sqr(x[b + num_vars * 2] - x[a + num_vars * 2])) -
                d);
      }
    }
    return res;
  }

  /**
   * @brief Minimize variance of 'z' coordinate constraints.
   */
  template <typename T>
  T var_constraint(T const* x) const {
    T avg = T(0.0);
    for (int i = 0; i < num_vars; i++) {
      avg += x[i + num_vars * 2];
    }
    avg /= num_vars;
    T res;
    for (int i = 0; i < num_vars; i++) {
      res += sqr(x[i + num_vars * 2] - avg);
    }
    res /= num_vars - 1;
    return res;
  }

  /**
   * @brief Energy for maximizing the embedding..
   */
  template <typename T>
  T max_embedding(T const* x) const {
    T result = T(0.0);
    // For each pleat node.
    for (int i = 0; i < num_vars; i++) {
      // Add pairwaise pleats distance.
      for (int j = i + 1; j < num_vars; j++) {
        result -=
            sqrt(sqr(x[i] - x[j]) + sqr(x[i + num_vars] - x[j + num_vars]) +
                 sqr(x[i + num_vars * 2] - x[j + num_vars * 2]));
      }
      // Add pleat-udnerlay distance.
      for (const auto& n : underlay_nodes) {
        result -= sqrt(sqr(x[i] - n[0]) + sqr(x[i + num_vars] - n[1]) +
                       sqr(x[i + num_vars * 2] - n[2]));
      }
    }
    return result;
  }

  template <typename T>
  bool operator()(T const* const* x, T* residual) const {
    residual[0] = max_embedding(x[0]) + var_constraint(x[0]) +
                  1e3 * equality_constraints(x[0]);

    return true;
  }
};

/**
 * @brief Find embedding for the pleat nodes.
 *
 * @param x_underlay_init The solved position for the underlay nodes.
 * @param x_pleats_init The initialization for the pleat nodes.
 * @param eq_constraints The pleat-pleat and pleat-underlay constraints.
 * @return std::vector<std::vector<double>>
 */
std::vector<std::vector<double>> embed_pleats(
    const std::vector<std::vector<double>>& x_underlay_init,
    const std::vector<std::vector<double>>& x_pleats_init,
    const std::vector<std::vector<double>>& eq_constraints) {
  int num_vars = x_pleats_init.size();
  // Set up the cost function.
  auto* cost_func = new ceres::DynamicAutoDiffCostFunction<CostFunctorPleats>(
      new CostFunctorPleats(num_vars, x_underlay_init, eq_constraints));
  cost_func->AddParameterBlock(num_vars * 3);
  cost_func->SetNumResiduals(1);
  // Flatten the pleat positions to a single vector.
  std::vector<double> data(num_vars * 3);
  for (int i = 0; i < num_vars; i++) {
    data[i] = x_pleats_init[i][0];
    data[i + num_vars] = x_pleats_init[i][1];
    data[i + num_vars * 2] = x_pleats_init[i][2];
  }
  // Init problem.
  ceres::Problem problem;
  problem.AddResidualBlock(cost_func, NULL, {data.data()});
  // Init the solver option.
  ceres::Solver::Options solver_options;
  solver_options.num_threads = 10;
  // Doesn't seem to work with TRUST REGION for some reason...
  solver_options.line_search_direction_type =
      ceres::LineSearchDirectionType::BFGS;
  solver_options.linear_solver_type = ceres::SPARSE_NORMAL_CHOLESKY;
  solver_options.minimizer_type = ceres::MinimizerType::LINE_SEARCH;
  solver_options.minimizer_progress_to_stdout = true;
  // Optimize.
  ceres::Solver::Summary summary;
  ceres::Solve(solver_options, &problem, &summary);

  // Convert back to the output format.
  std::vector<std::vector<double>> result(num_vars);
  for (int i = 0; i < num_vars; i++) {
    result[i].push_back(data[i]);
    result[i].push_back(data[i + num_vars]);
    result[i].push_back(data[i + num_vars * 2]);
  }
  return result;
}

/**
 * @brief Energy for optimizing the underlay nodes.
 */
struct CostFunctor {
  int num_vars;
  const std::vector<std::vector<double>>& eq_constraints;
  CostFunctor(int num_vars,
              const std::vector<std::vector<double>>& eq_constraints)
      : num_vars(num_vars), eq_constraints(eq_constraints) {}

  template <typename T>
  bool operator()(T const* const* x, T* residual) const {
    residual[0] = T(0.0);
    // Sum up the energy of the equality constraints.
    for (const auto& constraint : eq_constraints) {
      int a = constraint[0], b = constraint[1];
      double d = constraint[2];

      T val = sqr(sqrt(sqr(x[0][a] - x[0][b]) +
                       sqr(x[0][a + num_vars] - x[0][b + num_vars])) -
                  d);
      residual[0] += val;
    }

    return true;
  }
};

/**
 * @brief Compute the embedding of the underlay graph.
 */
std::vector<std::vector<double>> embed_underlay(
    const std::vector<std::vector<double>>& x_vec_init,
    const std::vector<std::vector<double>>& eq_constraints) {
  // Set up the cost function.
  auto* cost_func = new ceres::DynamicAutoDiffCostFunction<CostFunctor>(
      new CostFunctor(x_vec_init.size(), eq_constraints));
  cost_func->AddParameterBlock(x_vec_init.size() * 2);
  cost_func->SetNumResiduals(1);
  // Flatten the position into a single vector.
  std::vector<double> data(x_vec_init.size() * 2);
  for (int i = 0; i < x_vec_init.size(); i++) {
    data[i] = x_vec_init[i][0];
    data[i + x_vec_init.size()] = x_vec_init[i][1];
  }

  // Set up the problem.
  ceres::Problem problem;
  problem.AddResidualBlock(cost_func, NULL, {data.data()});
  // Set up options.
  ceres::Solver::Options solver_options;
  solver_options.num_threads = 10;
  solver_options.line_search_direction_type =
      ceres::LineSearchDirectionType::BFGS;
  solver_options.linear_solver_type = ceres::SPARSE_NORMAL_CHOLESKY;
  solver_options.minimizer_type = ceres::MinimizerType::LINE_SEARCH;
  solver_options.minimizer_progress_to_stdout = true;
  ceres::Solver::Summary summary;
  ceres::Solve(solver_options, &problem, &summary);

  // Convert back to the output format.
  std::vector<std::vector<double>> result(x_vec_init.size());
  for (int i = 0; i < x_vec_init.size(); i++) {
    result[i].push_back(data[i]);
    result[i].push_back(data[i + x_vec_init.size()]);
  }
  return result;
}

PYBIND11_MODULE(cpp_smocking_solver, m) {
  m.doc() = "Ceres solver for smocking";
  m.def("embed_underlay", &embed_underlay, "Embed underlay graph.");
  m.def("embed_pleats", &embed_pleats, "Embed pleats graph.");
}