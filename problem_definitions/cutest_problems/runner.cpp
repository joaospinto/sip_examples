#include "problem_definitions/casadi_problems/common/problem.hpp"
#include "problem_definitions/cutest_problems/cutest_problem.hpp"
#include "problem_definitions/cutest_problems/qp_scaling.hpp"

#include "sip/sip.hpp"
#include "sip_qdldl/sip_qdldl.hpp"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <exception>
#include <functional>
#include <iostream>
#include <optional>
#include <vector>

namespace sip_examples::problem_definitions::cutest_problems {
namespace {

auto run(const char *runtime_path, const char *problem_library_path,
         const char *outsdif_path) -> sip::Output {
  CutestProblem problem(runtime_path, problem_library_path, outsdif_path);
  const bool is_quadratic_program = problem.is_quadratic_program();
  std::cout << "problem_type=" << (is_quadratic_program ? "QP" : "NLP")
            << std::endl;
  const int x_dim = problem.x_dim();
  const int y_dim = problem.equality_dim();
  const int s_dim = problem.inequality_dim();
  const int kkt_dim = x_dim + y_dim + s_dim;

  auto settings = casadi_problems::default_casadi_problem_settings(1000);
  settings.line_search.skip_line_search = false;
  settings.line_search.max_iterations = 5000;
  settings.regularization.maximum = 1e12;
  settings.regularization.max_attempts = 24;
  settings.termination.max_merit_slope = 1e-24;
  if (is_quadratic_program) {
    settings.mode = sip::Mode::PRIMAL_DUAL_PROXIMAL_IPM;
    settings.num_iterative_refinement_steps = 1;
    settings.barrier.use_predictor_corrector = true;
    settings.barrier.use_adaptive_proximal_updates = true;
    settings.barrier.mu_update_factor = 0.2;
    settings.regularization.initial = 3e-5;
    settings.regularization.first_positive = 1e-12;
    settings.regularization.decrease_factor = 0.15;
    settings.regularization.maximum_factorization_shift = 1e-4;
    settings.penalty.initial_penalty_parameter = 1e4;
    settings.penalty.max_penalty_parameter = 1e12;
    settings.line_search.skip_line_search = true;
    settings.line_search.tau = 0.99;
  } else {
    settings.line_search.use_filter_line_search = true;
    settings.line_search.filter_min_total_line_search_iterations = 300;
  }
  if (std::getenv("SIP_CUTEST_PRINT_LOGS") != nullptr) {
    casadi_problems::enable_all_casadi_problem_logs(settings);
  }

  sip_qdldl::Workspace qdldl_workspace;
  qdldl_workspace.reserve(kkt_dim, problem.kkt_nnz(), problem.kkt_l_nnz());

  auto &model_output = problem.model_output();
  std::optional<ScaledQp> scaled_qp;
  if (is_quadratic_program) {
    std::vector<double> zero_x(x_dim, 0.0);
    std::vector<double> zero_y(y_dim, 0.0);
    std::vector<double> zero_z(s_dim, 0.0);
    problem.evaluate_values(zero_x.data());
    problem.evaluate_derivatives(zero_x.data(), zero_y.data(), zero_z.data());
    scaled_qp.emplace(model_output, problem.lower_bounds(),
                      problem.upper_bounds());
  }
  const double *model_x = nullptr;
  const double *model_y = nullptr;
  const double *model_z = nullptr;
  bool derivatives_current = is_quadratic_program;
  const auto model_callback =
      [&](const sip::ModelCallbackInput &input) -> void {
    model_x = input.x;
    model_y = input.y;
    model_z = input.z;
    if (input.new_x) {
      if (is_quadratic_program) {
        scaled_qp->evaluate_values(input.x, model_output);
      } else {
        problem.evaluate_values(input.x);
      }
    }
    if (!is_quadratic_program && (input.new_x || input.new_y || input.new_z)) {
      derivatives_current = false;
    }
  };
  const auto ensure_derivatives = [&]() -> void {
    if (!derivatives_current) {
      problem.evaluate_derivatives(model_x, model_y, model_z);
      derivatives_current = true;
    }
  };
  const sip_qdldl::Settings qdldl_settings{
      .permute_kkt_system = true,
      .kkt_pinv = problem.kkt_pinv(),
  };
  sip_qdldl::CallbackProvider callback_provider(qdldl_settings, model_output,
                                                qdldl_workspace);

  const auto factor = [&callback_provider, &ensure_derivatives](
                          const double *w, const double *r1, const double *r2,
                          const double *r3) -> bool {
    ensure_derivatives();
    return callback_provider.factor(w, r1, r2, r3);
  };
  const auto solve = [&callback_provider](const double *b, double *v) -> void {
    callback_provider.solve(b, v);
  };
  const auto add_kx_to_y =
      [&callback_provider, &ensure_derivatives](
          const double *w, const double *r1, const double *r2, const double *r3,
          const double *x_x, const double *x_y, const double *x_z, double *y_x,
          double *y_y, double *y_z) -> void {
    ensure_derivatives();
    callback_provider.add_Kx_to_y(w, r1, r2, r3, x_x, x_y, x_z, y_x, y_y, y_z);
  };
  const auto add_hx_to_y = [&callback_provider, &ensure_derivatives](
                               const double *x, double *y) -> void {
    ensure_derivatives();
    callback_provider.add_Hx_to_y(x, y);
  };
  const auto add_cx_to_y = [&callback_provider, &ensure_derivatives](
                               const double *x, double *y) -> void {
    ensure_derivatives();
    callback_provider.add_Cx_to_y(x, y);
  };
  const auto add_ctx_to_y = [&callback_provider, &ensure_derivatives](
                                const double *x, double *y) -> void {
    ensure_derivatives();
    callback_provider.add_CTx_to_y(x, y);
  };
  const auto add_gx_to_y = [&callback_provider, &ensure_derivatives](
                               const double *x, double *y) -> void {
    ensure_derivatives();
    callback_provider.add_Gx_to_y(x, y);
  };
  const auto add_gtx_to_y = [&callback_provider, &ensure_derivatives](
                                const double *x, double *y) -> void {
    ensure_derivatives();
    callback_provider.add_GTx_to_y(x, y);
  };
  const auto get_f = [&model_output]() -> double { return model_output.f; };
  const auto get_grad_f = [&model_output,
                           &ensure_derivatives]() -> const double * {
    ensure_derivatives();
    return model_output.gradient_f;
  };
  const auto get_c = [&model_output]() -> const double * {
    return model_output.c;
  };
  const auto get_g = [&model_output]() -> const double * {
    return model_output.g;
  };
  const auto timeout_callback = []() -> bool { return false; };
  const double *lower_bounds = is_quadratic_program
                                   ? scaled_qp->scaled_lower_bounds().data()
                                   : problem.lower_bounds();
  const double *upper_bounds = is_quadratic_program
                                   ? scaled_qp->scaled_upper_bounds().data()
                                   : problem.upper_bounds();

  sip::Input input{
      .factor = std::cref(factor),
      .solve = std::cref(solve),
      .add_Kx_to_y = std::cref(add_kx_to_y),
      .add_Hx_to_y = std::cref(add_hx_to_y),
      .add_Cx_to_y = std::cref(add_cx_to_y),
      .add_CTx_to_y = std::cref(add_ctx_to_y),
      .add_Gx_to_y = std::cref(add_gx_to_y),
      .add_GTx_to_y = std::cref(add_gtx_to_y),
      .get_f = std::cref(get_f),
      .get_grad_f = std::cref(get_grad_f),
      .get_c = std::cref(get_c),
      .get_g = std::cref(get_g),
      .model_callback = std::cref(model_callback),
      .timeout_callback = std::cref(timeout_callback),
      .lower_bounds = lower_bounds,
      .upper_bounds = upper_bounds,
      .residual_scaling =
          is_quadratic_program
              ? sip::Input::ResidualScaling{
                    .dual = scaled_qp->primal_variable_scaling().data(),
                    .equality =
                        scaled_qp->equality_residual_scaling().data(),
                    .inequality =
                        scaled_qp->inequality_residual_scaling().data(),
                    .bound_inequality =
                        scaled_qp->bound_residual_scaling().data(),
                }
              : sip::Input::ResidualScaling{},
      .dimensions =
          {
              .x_dim = x_dim,
              .s_dim = s_dim,
              .y_dim = y_dim,
          },
  };

  const int num_bound_sides = input.num_bound_sides();
  sip::Workspace workspace;
  workspace.reserve(x_dim, s_dim, y_dim, num_bound_sides, settings);

  if (is_quadratic_program) {
    scaled_qp->to_scaled_primal(problem.initial_x().data(), workspace.vars.x);
  } else {
    std::copy(problem.initial_x().begin(), problem.initial_x().end(),
              workspace.vars.x);
  }
  std::fill_n(workspace.vars.y, y_dim, 0.0);
  std::fill_n(workspace.vars.z, s_dim, 1.0);
  casadi_problems::initialize_bound_slacks_and_duals(
      lower_bounds, upper_bounds, x_dim, settings.barrier.initial_mu,
      workspace.vars.x, workspace.vars.bound_s, workspace.vars.bound_z);

  model_callback({.x = workspace.vars.x,
                  .y = workspace.vars.y,
                  .z = workspace.vars.z,
                  .new_x = true,
                  .new_y = true,
                  .new_z = true});
  casadi_problems::initialize_slacks_and_duals(
      model_output.g, s_dim, settings.barrier.initial_mu, workspace.vars.s,
      workspace.vars.z);

  sip::Output output = sip::solve(input, settings, workspace);
  if (is_quadratic_program) {
    const QpResiduals residuals = scaled_qp->residuals(
        workspace.vars.x, workspace.vars.y, workspace.vars.z, workspace.vars.s,
        workspace.vars.bound_s, workspace.vars.bound_z, model_output);
    output.max_primal_violation = residuals.primal;
    output.max_dual_violation = residuals.dual;
    if (output.exit_status == sip::Status::SOLVED &&
        !(residuals.primal < settings.termination.max_constraint_violation &&
          residuals.dual < settings.termination.max_dual_residual &&
          residuals.complementarity <
              settings.termination.max_complementarity_gap)) {
      output.exit_status = sip::Status::FAILED_CHECK;
    }
  }
  qdldl_workspace.free();
  workspace.free();
  return output;
}

} // namespace
} // namespace sip_examples::problem_definitions::cutest_problems

int main(int argc, char **argv) {
  if (argc != 4) {
    std::cerr << "usage: cutest_runner CUTEST_RUNTIME PROBLEM_LIBRARY "
                 "OUTSDIF\n";
    return 2;
  }
  try {
    const sip::Output output =
        sip_examples::problem_definitions::cutest_problems::run(
            argv[1], argv[2], argv[3]);
    std::cout << "status=" << output.exit_status
              << " iterations=" << output.num_iterations
              << " ls_iterations=" << output.num_ls_iterations
              << " primal=" << output.max_primal_violation
              << " dual=" << output.max_dual_violation << '\n';
    return output.exit_status == sip::Status::SOLVED ? 0 : 1;
  } catch (const std::exception &error) {
    std::cerr << error.what() << '\n';
    return 2;
  }
}
