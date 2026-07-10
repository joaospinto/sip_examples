#pragma once

#include "sip/types.hpp"

#include <algorithm>
#include <cmath>
#include <vector>

namespace sip_examples::problem_definitions::casadi_problems {

struct FlatProblemSpec {
  const char *name;
  int x_dim;
  int s_dim;
  int y_dim;
  int upper_hessian_lagrangian_nnz;
  int raw_upper_hessian_lagrangian_nnz;
  int jacobian_c_nnz;
  int jacobian_g_nnz;
  int jacobian_c_transpose_nnz;
  int jacobian_g_transpose_nnz;
  int kkt_dim;
  int kkt_nnz;
  int kkt_L_nnz;
  const int *upper_hessian_lagrangian_ind;
  const int *upper_hessian_lagrangian_indptr;
  const int *jacobian_c_ind;
  const int *jacobian_c_indptr;
  const int *jacobian_g_ind;
  const int *jacobian_g_indptr;
  const int *jacobian_c_transpose_ind;
  const int *jacobian_c_transpose_indptr;
  const int *jacobian_g_transpose_ind;
  const int *jacobian_g_transpose_indptr;
  const int *kkt_pinv;
  const double *initial_x;
};

struct OcpProblemSpec {
  const char *name;
  int num_stages;
  int state_dim;
  int control_dim;
  int c_dim;
  int g_dim;
  int theta_dim;
  const double *initial_x;
};

struct CasadiWork {
  std::vector<long long int> iw;
  std::vector<double> w;
  std::vector<double> scratch;

  void resize(int iw_size, int w_size, int scratch_size) {
    iw.resize(iw_size);
    w.resize(w_size);
    scratch.resize(scratch_size);
  }
};

inline sip::Settings default_casadi_problem_settings(const int max_iterations) {
  sip::Settings settings{
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
              .max_iterations = 100000,
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
  return settings;
}

inline void enable_all_casadi_problem_logs(sip::Settings &settings) {
  settings.logging.print_logs = true;
  settings.logging.print_line_search_logs = true;
  settings.logging.print_search_direction_logs = true;
  settings.logging.print_derivative_check_logs = true;
}

inline void initialize_slacks_and_duals(const double *g, const int s_dim,
                                        const double initial_mu, double *s,
                                        double *z) {
  const double floor = std::max(initial_mu, 1e-8);
  for (int i = 0; i < s_dim; ++i) {
    s[i] = std::max(-g[i], floor);
    z[i] = initial_mu / s[i];
  }
}

} // namespace sip_examples::problem_definitions::casadi_problems
