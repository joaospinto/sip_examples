#include "problem_definitions/casadi_problems/common/problem.hpp"
#include "problem_definitions/cutest_problems/cutest_problem.hpp"
#include "problem_definitions/cutest_problems/scaled_qp_model.hpp"

#include "sip/sip.hpp"
#include "sip_qdldl/sip_qdldl.hpp"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <exception>
#include <functional>
#include <iostream>
#include <limits>
#include <optional>
#include <string_view>
#include <vector>

namespace sip_examples::problem_definitions::cutest_problems {
namespace {

auto run(const char *runtime_path, const char *problem_library_path,
         const char *outsdif_path, bool use_qp_settings) -> sip::Output {
  CutestProblem problem(runtime_path, problem_library_path, outsdif_path);
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
  if (use_qp_settings) {
    settings.mode = sip::Mode::PROXIMAL_PREDICTOR_CORRECTOR_QP;
    settings.line_search.skip_line_search = true;
    settings.line_search.tau = 0.99;
    settings.regularization.max_attempts = 40;
  } else {
    const bool has_constraints = y_dim + s_dim > 0;
    settings.mode = has_constraints && !problem.is_quadratic_program()
                        ? sip::Mode::DUAL_PROXIMAL_IPM
                        : sip::Mode::REGULARIZED_IPM;
    if (settings.mode == sip::Mode::DUAL_PROXIMAL_IPM) {
      settings.penalty.initial_penalty_parameter =
          sip::PenaltySettings{}.initial_penalty_parameter;
    }
    settings.line_search.use_filter_line_search = true;
    settings.line_search.filter_min_total_line_search_iterations = 300;
  }
  if (std::getenv("SIP_CUTEST_PRINT_LOGS") != nullptr) {
    casadi_problems::enable_all_casadi_problem_logs(settings);
  }

  sip::Workspace workspace;
  workspace.reserve(x_dim, s_dim, y_dim, settings);

  sip_qdldl::Workspace qdldl_workspace;
  qdldl_workspace.reserve(kkt_dim, s_dim, problem.kkt_nnz(),
                          problem.kkt_l_nnz());

  auto &model_output = problem.model_output();
  QpScaling scaling(x_dim, y_dim, s_dim);
  std::optional<ScaledQpModel> qp_model;
  if (use_qp_settings) {
    std::vector<double> zero_x(x_dim, 0.0);
    std::vector<double> zero_y(y_dim, 0.0);
    std::vector<double> zero_z(s_dim, 0.0);
    problem.evaluate_values(zero_x.data());
    problem.evaluate_derivatives(zero_x.data(), zero_y.data(), zero_z.data());
    scaling.compute(model_output);
    if (!scaling.has_material_effect()) {
      scaling.set_identity();
    }
    qp_model.emplace(model_output, scaling);
  }
  const double *model_x = nullptr;
  const double *model_y = nullptr;
  const double *model_z = nullptr;
  bool derivatives_current = use_qp_settings;
  const auto model_callback =
      [&](const sip::ModelCallbackInput &input) -> void {
    if (!use_qp_settings) {
      model_x = input.x;
      model_y = input.y;
      model_z = input.z;
    }
    if (input.new_x) {
      if (use_qp_settings) {
        qp_model->evaluate_values(input.x, scaling, model_output);
      } else {
        problem.evaluate_values(model_x);
      }
    }
    if (!use_qp_settings && (input.new_x || input.new_y || input.new_z)) {
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

  const auto factor = [&callback_provider, &ensure_derivatives, &model_output,
                       x_dim](const double *w, double r1, const double *r2,
                              const double *r3,
                              double factorization_regularization) -> bool {
    ensure_derivatives();
    const bool succeeded =
        callback_provider.factor(w, r1, r2, r3, factorization_regularization);
    if (!succeeded &&
        std::getenv("SIP_CUTEST_PRINT_FACTOR_DIAGNOSTICS") != nullptr) {
      const auto &hessian = model_output.upper_hessian_lagrangian;
      double minimum_diagonal = std::numeric_limits<double>::infinity();
      double maximum_diagonal = -std::numeric_limits<double>::infinity();
      double maximum_absolute_value = 0.0;
      double maximum_absolute_gradient = 0.0;
      int nonfinite_values = 0;
      for (int col = 0; col < hessian.cols; ++col) {
        for (int index = hessian.indptr[col]; index < hessian.indptr[col + 1];
             ++index) {
          const double value = hessian.data[index];
          if (!std::isfinite(value)) {
            ++nonfinite_values;
          } else {
            maximum_absolute_value =
                std::max(maximum_absolute_value, std::abs(value));
          }
          if (hessian.ind[index] == col) {
            minimum_diagonal = std::min(minimum_diagonal, value);
            maximum_diagonal = std::max(maximum_diagonal, value);
          }
        }
      }
      for (int i = 0; i < x_dim; ++i) {
        maximum_absolute_gradient = std::max(
            maximum_absolute_gradient, std::abs(model_output.gradient_f[i]));
      }
      std::cerr << "factorization_failed r1=" << r1
                << " numerical_regularization=" << factorization_regularization
                << " hessian_max_abs=" << maximum_absolute_value
                << " hessian_diagonal_min=" << minimum_diagonal
                << " hessian_diagonal_max=" << maximum_diagonal
                << " hessian_nonfinite=" << nonfinite_values
                << " gradient_max_abs=" << maximum_absolute_gradient
                << " objective=" << model_output.f << '\n';
    }
    return succeeded;
  };
  const auto solve = [&callback_provider](const double *b, double *v) -> void {
    callback_provider.solve(b, v);
  };
  const auto add_kx_to_y = [&callback_provider, &ensure_derivatives](
                               const double *w, double r1, const double *r2,
                               const double *r3, const double *x_x,
                               const double *x_y, const double *x_z,
                               double *y_x, double *y_y, double *y_z) -> void {
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
      .residual_scaling =
          {
              .dual = use_qp_settings ? scaling.primal().data() : nullptr,
              .equality = use_qp_settings ? scaling.equality().data() : nullptr,
              .inequality =
                  use_qp_settings ? scaling.inequality().data() : nullptr,
          },
      .dimensions =
          {
              .x_dim = x_dim,
              .s_dim = s_dim,
              .y_dim = y_dim,
          },
  };

  if (use_qp_settings) {
    scaling.to_scaled_primal(problem.initial_x().data(), workspace.vars.x);
  } else {
    std::copy(problem.initial_x().begin(), problem.initial_x().end(),
              workspace.vars.x);
  }
  std::fill_n(workspace.vars.y, y_dim, 0.0);
  std::fill_n(workspace.vars.z, s_dim, 1.0);

  model_callback({.x = workspace.vars.x,
                  .y = workspace.vars.y,
                  .z = workspace.vars.z,
                  .new_x = true,
                  .new_y = true,
                  .new_z = true});
  casadi_problems::initialize_slacks_and_duals(
      model_output.g, s_dim, settings.barrier.initial_mu, workspace.vars.s,
      workspace.vars.z);

  const sip::Output output = sip::solve(input, settings, workspace);
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
