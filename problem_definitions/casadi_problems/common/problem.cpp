#include "problem_definitions/casadi_problems/common/problem.hpp"

#include <algorithm>
#include <cstdlib>
#include <ostream>
#include <stdexcept>

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

auto settings_configuration_from_environment(sip::Settings settings)
    -> SettingsConfiguration {
  const auto enable_alpha_max_line_search = [&settings]() {
    settings.line_search.skip_line_search = false;
    settings.line_search.start_ls_with_alpha_s_max = true;
  };
  const char *value = std::getenv("SIP_CASADI_PROBLEMS_ABLATION");
  const std::string_view ablation =
      value == nullptr ? std::string_view("default") : std::string_view(value);
  if (ablation == "line_search") {
    settings.line_search.skip_line_search = false;
  } else if (ablation == "max_regularization") {
    settings.regularization.maximum = 1e15;
  } else if (ablation == "line_search_max_regularization") {
    settings.line_search.skip_line_search = false;
    settings.regularization.maximum = 1e15;
  } else if (ablation == "fixed_penalty") {
    settings.penalty.penalty_parameter_increase_factor = 1.0;
  } else if (ablation == "line_search_fixed_penalty") {
    settings.line_search.skip_line_search = false;
    settings.penalty.penalty_parameter_increase_factor = 1.0;
  } else if (ablation == "penalty_cap_1e6") {
    settings.penalty.max_penalty_parameter = 1e6;
  } else if (ablation == "penalty_factor_1_1") {
    settings.penalty.penalty_parameter_increase_factor = 1.1;
  } else if (ablation == "long_run") {
    settings.max_iterations = 5000;
  } else if (ablation == "iterative_refinement") {
    settings.num_iterative_refinement_steps = 2;
  } else if (ablation == "predictor_corrector") {
    settings.barrier.use_predictor_corrector = true;
  } else if (ablation == "fast_barrier") {
    settings.barrier.mu_update_factor = 0.2;
  } else if (ablation == "filter_line_search") {
    settings.line_search.skip_line_search = false;
    settings.line_search.use_filter_line_search = true;
  } else if (ablation == "predictor_corrector_filter") {
    settings.barrier.use_predictor_corrector = true;
    settings.line_search.skip_line_search = false;
    settings.line_search.use_filter_line_search = true;
  } else if (ablation == "line_search_alpha_max") {
    enable_alpha_max_line_search();
  } else if (ablation == "line_search_alpha_max_5000") {
    settings.line_search.max_iterations = 5000;
    enable_alpha_max_line_search();
  } else if (ablation == "line_search_alpha_max_32_factor_attempts") {
    settings.regularization.max_attempts = 32;
    enable_alpha_max_line_search();
  } else if (ablation == "line_search_alpha_max_max_regularization") {
    settings.regularization.maximum = 1e12;
    settings.regularization.max_attempts = 32;
    enable_alpha_max_line_search();
  } else if (ablation == "line_search_alpha_max_step_aware_penalty") {
    settings.penalty.scale_violation_reduction_with_step_size = true;
    enable_alpha_max_line_search();
  } else if (ablation ==
             "line_search_alpha_max_step_aware_penalty_max_regularization") {
    settings.penalty.scale_violation_reduction_with_step_size = true;
    settings.regularization.maximum = 1e12;
    settings.regularization.max_attempts = 32;
    enable_alpha_max_line_search();
  } else if (ablation != "default") {
    throw std::invalid_argument("unknown SIP_CASADI_PROBLEMS_ABLATION mode");
  }
  return {.settings = settings, .ablation = ablation};
}

void enable_all_casadi_problem_logs(sip::Settings &settings) {
  settings.logging.print_logs = true;
  settings.logging.print_line_search_logs = true;
  settings.logging.print_search_direction_logs = true;
  settings.logging.print_derivative_check_logs = true;
}

void print_result(std::ostream &stream, const std::string_view ablation,
                  const sip::Output &output) {
  stream << "ablation=" << ablation << " status=" << output.exit_status
         << " iterations=" << output.num_iterations
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

} // namespace sip_examples::problem_definitions::casadi_problems
