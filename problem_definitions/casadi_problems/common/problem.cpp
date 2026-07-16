#include "problem_definitions/casadi_problems/common/problem.hpp"

#include <algorithm>
#include <cmath>
#include <iostream>

namespace sip_examples::problem_definitions::casadi_problems {

void CasadiWork::resize(int iw_size, int w_size, int scratch_size) {
  iw.resize(iw_size);
  w.resize(w_size);
  scratch.resize(scratch_size);
}

auto default_casadi_problem_settings(int max_iterations) -> sip::Settings {
  return sip::Settings{
      .max_iterations = max_iterations,
      .num_iterative_refinement_steps = 0,
      .barrier =
          {
              .initial_mu = 1e-2,
              .mu_update_factor = 0.8,
              .mu_min = 1e-16,
              .mu_update_kappa = 100.0,
          },
      .penalty =
          {
              .initial_penalty_parameter = 1.0,
              .penalty_parameter_increase_factor = 1.35,
              .max_penalty_parameter = 1e9,
          },
      .termination =
          {
              .max_dual_residual = 1e-6,
              .max_constraint_violation = 1e-6,
              .max_complementarity_gap = 1e-6,
              .max_merit_slope = 1e-16,
          },
      .regularization =
          {
              .initial = 1e-3,
              .maximum = 1e9,
              .increase_factor = 4.0,
              .decrease_factor = 0.25,
          },
      .line_search =
          {
              .max_iterations = 1000,
              .skip_line_search = true,
              .enable_line_search_failures = false,
          },
      .logging =
          {
              .print_logs = false,
              .print_line_search_logs = false,
              .print_search_direction_logs = false,
              .print_derivative_check_logs = false,
          },
  };
}

void enable_all_casadi_problem_logs(sip::Settings &settings) {
  settings.logging.print_logs = true;
  settings.logging.print_line_search_logs = true;
  settings.logging.print_search_direction_logs = true;
  settings.logging.print_derivative_check_logs = true;
}

void print_casadi_problem_summary(const sip::Output &output) {
  std::cout << "iterations=" << output.num_iterations
            << " ls_iterations=" << output.num_ls_iterations
            << " primal=" << output.max_primal_violation
            << " dual=" << output.max_dual_violation << '\n';
}

void initialize_slacks_and_duals(const double *g, int s_dim, double initial_mu,
                                 double *s, double *z) {
  const double floor = std::max(initial_mu, 1e-8);
  for (int i = 0; i < s_dim; ++i) {
    s[i] = std::max(-g[i], floor);
    z[i] = initial_mu / s[i];
  }
}

void initialize_bound_slacks_and_duals(const double *lower_bounds,
                                       const double *upper_bounds, int x_dim,
                                       double initial_mu, const double *x,
                                       double *bound_s, double *bound_z) {
  const double floor = std::max(initial_mu, 1e-8);
  int side = 0;
  const auto initialize_side = [&](double constraint_value) {
    bound_s[side] = std::max(-constraint_value, floor);
    bound_z[side] = initial_mu / bound_s[side];
    ++side;
  };
  for (int variable = 0; variable < x_dim; ++variable) {
    if (lower_bounds != nullptr && std::isfinite(lower_bounds[variable])) {
      initialize_side(lower_bounds[variable] - x[variable]);
    }
    if (upper_bounds != nullptr && std::isfinite(upper_bounds[variable])) {
      initialize_side(x[variable] - upper_bounds[variable]);
    }
  }
}

} // namespace sip_examples::problem_definitions::casadi_problems
