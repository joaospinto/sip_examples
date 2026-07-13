#include "problem_definitions/casadi_problems/common/problem.hpp"
#include "problem_definitions/cutest_problems/cutest_problem.hpp"

#include "sip/sip.hpp"
#include "sip_qdldl/sip_qdldl.hpp"

#include <algorithm>
#include <cerrno>
#include <charconv>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <exception>
#include <functional>
#include <iostream>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>

namespace sip_examples::problem_definitions::cutest_problems {
namespace {

auto positive_environment_double(const char *name) -> std::optional<double> {
  const char *text = std::getenv(name);
  if (text == nullptr) {
    return std::nullopt;
  }
  char *end = nullptr;
  errno = 0;
  const double value = std::strtod(text, &end);
  if (errno != 0 || end == text || *end != '\0' || !std::isfinite(value) ||
      value <= 0.0) {
    throw std::invalid_argument(std::string(name) +
                                " must be a positive finite number");
  }
  return value;
}

auto nonnegative_environment_int(const char *name) -> std::optional<int> {
  const char *value = std::getenv(name);
  if (value == nullptr) {
    return std::nullopt;
  }
  const std::string_view text(value);
  int parsed = 0;
  const auto [end, error] =
      std::from_chars(text.data(), text.data() + text.size(), parsed);
  if (error != std::errc{} || end != text.data() + text.size() || parsed < 0) {
    throw std::invalid_argument(std::string(name) +
                                " must be a nonnegative integer");
  }
  return parsed;
}

auto run(const char *runtime_path, const char *problem_library_path,
         const char *outsdif_path, bool use_qp_settings) -> sip::Output {
  CutestProblem problem(runtime_path, problem_library_path, outsdif_path,
                        CutestProblem::SymbolicData::SipQdldl);
  const int x_dim = problem.x_dim();
  const int y_dim = problem.equality_dim();
  const int s_dim = problem.inequality_dim();
  const int kkt_dim = x_dim + y_dim + s_dim;

  auto settings = casadi_problems::default_casadi_problem_settings(1000);
  settings.line_search.skip_line_search = false;
  settings.line_search.use_filter_line_search = !use_qp_settings;
  settings.regularization.max_attempts = 24;
  if (const auto maximum =
          positive_environment_double("SIP_CUTEST_REGULARIZATION_MAXIMUM")) {
    settings.regularization.maximum = *maximum;
    settings.regularization.max_attempts = 128;
  }
  if (const auto decrease_factor =
          positive_environment_double(
              "SIP_CUTEST_REGULARIZATION_DECREASE_FACTOR")) {
    if (*decrease_factor > 1.0) {
      throw std::invalid_argument(
          "SIP_CUTEST_REGULARIZATION_DECREASE_FACTOR must not exceed 1");
    }
    settings.regularization.decrease_factor = *decrease_factor;
  }
  if (const auto filter_min_ls =
          nonnegative_environment_int("SIP_CUTEST_FILTER_MIN_LS")) {
    settings.line_search.filter_min_total_line_search_iterations =
        *filter_min_ls;
  }
  if (const auto refinement_steps = nonnegative_environment_int(
          "SIP_CUTEST_ITERATIVE_REFINEMENT_STEPS")) {
    settings.num_iterative_refinement_steps = *refinement_steps;
  }
  if (use_qp_settings) {
    settings.barrier.mu_update_factor = 0.2;
    settings.barrier.use_predictor_corrector =
        std::getenv("SIP_CUTEST_PREDICTOR_CORRECTOR") != nullptr;
    settings.line_search.skip_line_search =
        settings.barrier.use_predictor_corrector;
    settings.line_search.max_iterations = 5000;
    settings.penalty.scale_violation_reduction_with_step_size = true;
    settings.regularization.initial = 3e-5;
    if (const auto penalty =
            positive_environment_double("SIP_CUTEST_INITIAL_PENALTY")) {
      settings.penalty.initial_penalty_parameter = *penalty;
    }
  }
  if (std::getenv("SIP_CUTEST_PRINT_LOGS") != nullptr) {
    casadi_problems::enable_all_casadi_problem_logs(settings);
  }

  sip::Workspace workspace;
  workspace.reserve(x_dim, s_dim, y_dim,
                    sip::FilterWorkspace::required_capacity(settings));

  sip_qdldl::Workspace qdldl_workspace;
  qdldl_workspace.reserve(kkt_dim, problem.kkt_nnz(), problem.kkt_l_nnz());

  auto &model_output = problem.model_output();
  const bool print_diagnostics =
      std::getenv("SIP_CUTEST_DIAGNOSTICS") != nullptr;
  int factor_count = 0;
  int solve_count = 0;
  int value_evaluation_count = 0;
  int derivative_evaluation_count = 0;
  double factor_seconds = 0.0;
  double solve_seconds = 0.0;
  const double *model_x = nullptr;
  const double *model_y = nullptr;
  const double *model_z = nullptr;
  bool derivatives_current = false;
  const auto model_callback = [&](const sip::ModelCallbackInput &input) -> void {
    model_x = input.x;
    model_y = input.y;
    model_z = input.z;
    if (input.new_x) {
      problem.evaluate_values(input.x);
      ++value_evaluation_count;
    }
    if (input.new_x || input.new_y || input.new_z) {
      derivatives_current = false;
    }
  };
  const auto ensure_derivatives = [&]() -> void {
    if (!derivatives_current) {
      problem.evaluate_derivatives(model_x, model_y, model_z);
      derivatives_current = true;
      ++derivative_evaluation_count;
    }
  };
  const sip_qdldl::Settings qdldl_settings{
      .permute_kkt_system = true,
      .kkt_pinv = problem.kkt_pinv(),
  };
  sip_qdldl::CallbackProvider callback_provider(qdldl_settings, model_output,
                                                qdldl_workspace);

  const auto factor = [&callback_provider, &model_output, &ensure_derivatives,
                       print_diagnostics,
                       &factor_count, &factor_seconds, y_dim, s_dim,
                       max_regularization = settings.regularization.maximum](
                          const double *w, double r1, const double *r2,
                          const double *r3) -> bool {
    ensure_derivatives();
    const auto start = std::chrono::steady_clock::now();
    const bool succeeded = callback_provider.factor(w, r1, r2, r3);
    factor_seconds += std::chrono::duration<double>(
                          std::chrono::steady_clock::now() - start)
                          .count();
    ++factor_count;
    if (print_diagnostics) {
      std::cerr << "factor r1=" << r1 << " succeeded=" << succeeded << '\n';
      if (!succeeded && r1 >= max_regularization) {
        const auto nonfinite_count = [](const double *values, int size) {
          return std::count_if(values, values + size, [](double value) {
            return !std::isfinite(value);
          });
        };
        const auto sparse_nonfinite_count = [&](const auto &matrix) {
          return nonfinite_count(matrix.data,
                                 matrix.indptr[matrix.cols]);
        };
        const auto max_abs = [](const double *values, int size) {
          double result = 0.0;
          for (int i = 0; i < size; ++i) {
            result = std::max(result, std::fabs(values[i]));
          }
          return result;
        };
        const auto sparse_max_abs = [&](const auto &matrix) {
          return max_abs(matrix.data, matrix.indptr[matrix.cols]);
        };
        std::cerr << "terminal_factor_nonfinite w="
                  << nonfinite_count(w, s_dim)
                  << " r2=" << nonfinite_count(r2, y_dim)
                  << " r3=" << nonfinite_count(r3, s_dim)
                  << " hessian="
                  << sparse_nonfinite_count(
                         model_output.upper_hessian_lagrangian)
                  << " jacobian_c="
                  << sparse_nonfinite_count(model_output.jacobian_c)
                  << " jacobian_g="
                  << sparse_nonfinite_count(model_output.jacobian_g) << '\n'
                  << "terminal_factor_max_abs w=" << max_abs(w, s_dim)
                  << " r2=" << max_abs(r2, y_dim)
                  << " r3=" << max_abs(r3, s_dim)
                  << " hessian="
                  << sparse_max_abs(model_output.upper_hessian_lagrangian)
                  << " jacobian_c="
                  << sparse_max_abs(model_output.jacobian_c)
                  << " jacobian_g="
                  << sparse_max_abs(model_output.jacobian_g) << '\n';
      }
    }
    return succeeded;
  };
  const auto solve = [&callback_provider, &solve_count,
                      &solve_seconds](const double *b, double *v) -> void {
    const auto start = std::chrono::steady_clock::now();
    callback_provider.solve(b, v);
    solve_seconds += std::chrono::duration<double>(
                         std::chrono::steady_clock::now() - start)
                         .count();
    ++solve_count;
  };
  const auto add_kx_to_y =
      [&callback_provider](const double *w, double r1, const double *r2,
                           const double *r3, const double *x_x,
                           const double *x_y, const double *x_z, double *y_x,
                           double *y_y, double *y_z) -> void {
    callback_provider.add_Kx_to_y(w, r1, r2, r3, x_x, x_y, x_z, y_x, y_y, y_z);
  };
  const auto add_hx_to_y = [&callback_provider](const double *x,
                                                double *y) -> void {
    callback_provider.add_Hx_to_y(x, y);
  };
  const auto add_cx_to_y = [&callback_provider](const double *x,
                                                double *y) -> void {
    callback_provider.add_Cx_to_y(x, y);
  };
  const auto add_ctx_to_y = [&callback_provider](const double *x,
                                                 double *y) -> void {
    callback_provider.add_CTx_to_y(x, y);
  };
  const auto add_gx_to_y = [&callback_provider](const double *x,
                                                double *y) -> void {
    callback_provider.add_Gx_to_y(x, y);
  };
  const auto add_gtx_to_y = [&callback_provider](const double *x,
                                                 double *y) -> void {
    callback_provider.add_GTx_to_y(x, y);
  };
  const auto get_f = [&model_output]() -> double { return model_output.f; };
  const auto get_grad_f = [&model_output, &ensure_derivatives]() -> const double * {
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
      .dimensions =
          {
              .x_dim = x_dim,
              .s_dim = s_dim,
              .y_dim = y_dim,
          },
  };

  std::copy(problem.initial_x().begin(), problem.initial_x().end(),
            workspace.vars.x);
  std::fill_n(workspace.vars.y, y_dim, 0.0);
  std::fill_n(workspace.vars.z, s_dim, 1.0);

  model_callback({.x = workspace.vars.x,
                  .y = workspace.vars.y,
                  .z = workspace.vars.z,
                  .new_x = true,
                  .new_y = true,
                  .new_z = true});
  if (print_diagnostics) {
    ensure_derivatives();
    const int hessian_nnz =
        model_output.upper_hessian_lagrangian
            .indptr[model_output.upper_hessian_lagrangian.cols];
    const auto nonfinite_count = [](const double *values, int size) {
      return std::count_if(values, values + size,
                           [](double value) { return !std::isfinite(value); });
    };
    const auto [hessian_min, hessian_max] = std::minmax_element(
        model_output.upper_hessian_lagrangian.data,
        model_output.upper_hessian_lagrangian.data + hessian_nnz);
    std::cerr << "dimensions x=" << x_dim << " y=" << y_dim << " s=" << s_dim
              << " kkt_nnz=" << problem.kkt_nnz()
              << " kkt_l_nnz=" << problem.kkt_l_nnz() << '\n'
              << "model f=" << model_output.f << " gradient_nonfinite="
              << nonfinite_count(model_output.gradient_f, x_dim)
              << " hessian_nonfinite="
              << nonfinite_count(model_output.upper_hessian_lagrangian.data,
                                 hessian_nnz)
              << " hessian_min=" << *hessian_min
              << " hessian_max=" << *hessian_max << '\n';
  }
  casadi_problems::initialize_slacks_and_duals(
      model_output.g, s_dim, settings.barrier.initial_mu, workspace.vars.s,
      workspace.vars.z);

  const sip::Output output = sip::solve(input, settings, workspace);
  if (print_diagnostics) {
    std::cerr << "linear_system factor_count=" << factor_count
              << " factor_seconds=" << factor_seconds
              << " solve_count=" << solve_count
              << " solve_seconds=" << solve_seconds
              << " value_evaluations=" << value_evaluation_count
              << " derivative_evaluations=" << derivative_evaluation_count
              << '\n';
  }
  double max_complementarity = 0.0;
  double mean_complementarity = 0.0;
  for (int i = 0; i < s_dim; ++i) {
    const double complementarity = workspace.vars.s[i] * workspace.vars.z[i];
    max_complementarity = std::max(max_complementarity, complementarity);
    mean_complementarity += complementarity;
  }
  mean_complementarity /= std::max(s_dim, 1);
  std::cout << "mean_complementarity=" << mean_complementarity
            << " max_complementarity=" << max_complementarity << '\n';
  qdldl_workspace.free();
  workspace.free();
  return output;
}

} // namespace
} // namespace sip_examples::problem_definitions::cutest_problems

int main(int argc, char **argv) {
  if (argc != 5) {
    std::cerr << "usage: cutest_runner CUTEST_RUNTIME PROBLEM_LIBRARY OUTSDIF "
                 "USE_QP_SETTINGS\n";
    return 2;
  }
  const std::string_view use_qp_settings_arg(argv[4]);
  if (use_qp_settings_arg != "0" && use_qp_settings_arg != "1") {
    std::cerr << "USE_QP_SETTINGS must be 0 or 1\n";
    return 2;
  }
  try {
    const sip::Output output =
        sip_examples::problem_definitions::cutest_problems::run(
            argv[1], argv[2], argv[3], use_qp_settings_arg == "1");
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
