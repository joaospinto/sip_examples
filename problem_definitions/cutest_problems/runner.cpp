#include "problem_definitions/casadi_problems/common/problem.hpp"
#include "problem_definitions/cutest_problems/cutest_problem.hpp"

#include "sip/sip.hpp"
#include "sip_qdldl/sip_qdldl.hpp"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <exception>
#include <functional>
#include <iostream>
#include <string_view>
#include <vector>

namespace sip_examples::problem_definitions::cutest_problems {
namespace {

struct ModelScaling {
  explicit ModelScaling(int x_dim, int y_dim, int z_dim)
      : x(x_dim, 1.0), y(y_dim, 1.0), z(z_dim, 1.0),
        dual_residual(x_dim, 1.0), row_norm(x_dim + y_dim + z_dim) {}

  void compute(const sip_qdldl::ModelCallbackOutput &output) {
    const int x_dim = static_cast<int>(x.size());
    const int y_dim = static_cast<int>(y.size());
    const int z_dim = static_cast<int>(z.size());
    for (int iteration = 0; iteration < 10; ++iteration) {
      std::fill(row_norm.begin(), row_norm.end(), 0.0);
      const auto update_norm = [&](const int lhs, const int rhs,
                                   const double value) {
        const double magnitude = std::abs(value);
        row_norm[lhs] = std::max(row_norm[lhs], magnitude);
        row_norm[rhs] = std::max(row_norm[rhs], magnitude);
      };
      const auto &hessian = output.upper_hessian_lagrangian;
      for (int col = 0; col < x_dim; ++col) {
        for (int index = hessian.indptr[col]; index < hessian.indptr[col + 1];
             ++index) {
          const int row = hessian.ind[index];
          update_norm(row, col, hessian.data[index] * x[row] * x[col]);
        }
      }
      const auto &jacobian_c = output.jacobian_c;
      for (int col = 0; col < y_dim; ++col) {
        for (int index = jacobian_c.indptr[col];
             index < jacobian_c.indptr[col + 1]; ++index) {
          const int row = jacobian_c.ind[index];
          update_norm(row, x_dim + col,
                      jacobian_c.data[index] * x[row] * y[col]);
        }
      }
      const auto &jacobian_g = output.jacobian_g;
      for (int col = 0; col < z_dim; ++col) {
        for (int index = jacobian_g.indptr[col];
             index < jacobian_g.indptr[col + 1]; ++index) {
          const int row = jacobian_g.ind[index];
          update_norm(row, x_dim + y_dim + col,
                      jacobian_g.data[index] * x[row] * z[col]);
        }
      }

      double max_change = 0.0;
      for (double &norm : row_norm) {
        norm = norm < 1e-4 ? 1.0 : std::min(norm, 1e4);
        norm = 1.0 / std::sqrt(norm);
        max_change = std::max(max_change, std::abs(1.0 - norm));
      }
      for (int i = 0; i < x_dim; ++i) {
        x[i] *= row_norm[i];
      }
      for (int i = 0; i < y_dim; ++i) {
        y[i] *= row_norm[x_dim + i];
      }
      for (int i = 0; i < z_dim; ++i) {
        z[i] *= row_norm[x_dim + y_dim + i];
      }
      if (max_change <= 1e-3) {
        break;
      }
    }
  }

  void compute_nlp(const sip_qdldl::ModelCallbackOutput &output) {
    constexpr double kMaximumDerivative = 100.0;
    double maximum_gradient = 0.0;
    for (int i = 0; i < static_cast<int>(x.size()); ++i) {
      maximum_gradient =
          std::max(maximum_gradient, std::abs(output.gradient_f[i]));
    }
    if (maximum_gradient > kMaximumDerivative) {
      objective = kMaximumDerivative / maximum_gradient;
      std::fill(dual_residual.begin(), dual_residual.end(), objective);
    }

    const auto scale_constraint_rows = [](const sip_qdldl::SparseMatrix &matrix,
                                          std::vector<double> &scales) {
      for (int col = 0; col < matrix.cols; ++col) {
        double maximum = 0.0;
        for (int index = matrix.indptr[col]; index < matrix.indptr[col + 1];
             ++index) {
          maximum = std::max(maximum, std::abs(matrix.data[index]));
        }
        if (maximum > kMaximumDerivative) {
          scales[col] = kMaximumDerivative / maximum;
        }
      }
    };
    scale_constraint_rows(output.jacobian_c, y);
    scale_constraint_rows(output.jacobian_g, z);
  }

  auto is_identity() const -> bool {
    return objective == 1.0 &&
           std::all_of(y.begin(), y.end(),
                       [](const double value) { return value == 1.0; }) &&
           std::all_of(z.begin(), z.end(),
                       [](const double value) { return value == 1.0; });
  }

  void scale_values(sip_qdldl::ModelCallbackOutput &output) const {
    output.f *= objective;
    for (int i = 0; i < static_cast<int>(y.size()); ++i) {
      output.c[i] *= y[i];
    }
    for (int i = 0; i < static_cast<int>(z.size()); ++i) {
      output.g[i] *= z[i];
    }
  }

  void scale_derivatives(sip_qdldl::ModelCallbackOutput &output) const {
    for (int i = 0; i < static_cast<int>(x.size()); ++i) {
      output.gradient_f[i] *= objective;
    }
    auto &hessian = output.upper_hessian_lagrangian;
    for (int index = 0; index < hessian.indptr[hessian.cols]; ++index) {
      hessian.data[index] *= objective;
    }
    auto &jacobian_c = output.jacobian_c;
    for (int col = 0; col < jacobian_c.cols; ++col) {
      for (int index = jacobian_c.indptr[col];
           index < jacobian_c.indptr[col + 1]; ++index) {
        jacobian_c.data[index] *= y[col];
      }
    }
    auto &jacobian_g = output.jacobian_g;
    for (int col = 0; col < jacobian_g.cols; ++col) {
      for (int index = jacobian_g.indptr[col];
           index < jacobian_g.indptr[col + 1]; ++index) {
        jacobian_g.data[index] *= z[col];
      }
    }
  }

  std::vector<double> x;
  std::vector<double> y;
  std::vector<double> z;
  std::vector<double> dual_residual;
  std::vector<double> row_norm;
  double objective{1.0};
};

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
  settings.regularization.max_attempts = 40;
  settings.termination.max_merit_slope = 1e-24;
  if (use_qp_settings) {
    settings.barrier.mu_update_factor = 0.2;
    settings.regularization.initial = 3e-5;
    settings.regularization.decrease_factor = 0.15;
  } else {
    settings.line_search.use_filter_line_search = true;
    settings.line_search.filter_min_total_line_search_iterations = 300;
  }
  if (std::getenv("SIP_CUTEST_PRINT_LOGS") != nullptr) {
    casadi_problems::enable_all_casadi_problem_logs(settings);
  }
  sip::Workspace workspace;
  workspace.reserve(x_dim, s_dim, y_dim, settings);

  sip_qdldl::Workspace qdldl_workspace;
  qdldl_workspace.reserve(kkt_dim, problem.kkt_nnz(), problem.kkt_l_nnz());

  auto &model_output = problem.model_output();
  ModelScaling scaling(x_dim, y_dim, s_dim);
  bool scaling_enabled = false;
  std::vector<double> original_y(y_dim);
  std::vector<double> original_z(s_dim);
  std::fill(original_y.begin(), original_y.end(), 0.0);
  std::fill(original_z.begin(), original_z.end(), 0.0);
  problem.evaluate_values(problem.initial_x().data());
  problem.evaluate_derivatives(problem.initial_x().data(), original_y.data(),
                               original_z.data());
  if (use_qp_settings) {
    scaling.compute(model_output);
    double minimum_scale = 1.0;
    double maximum_scale = 1.0;
    const auto include_range = [&](const std::vector<double> &values) {
      if (values.empty()) {
        return;
      }
      const auto [minimum, maximum] =
          std::minmax_element(values.begin(), values.end());
      minimum_scale = std::min(minimum_scale, *minimum);
      maximum_scale = std::max(maximum_scale, *maximum);
    };
    include_range(scaling.y);
    include_range(scaling.z);
    scaling_enabled = maximum_scale >= 1e3 * minimum_scale;
    if (!scaling_enabled) {
      std::fill(scaling.y.begin(), scaling.y.end(), 1.0);
      std::fill(scaling.z.begin(), scaling.z.end(), 1.0);
    }
  } else {
    scaling.compute_nlp(model_output);
    scaling_enabled = !scaling.is_identity();
    settings.barrier.initial_mu *= scaling.objective;
    settings.barrier.mu_min *= scaling.objective;
    if (y_dim == 0 && s_dim == 0) {
      settings.regularization.initial *= scaling.objective;
      settings.regularization.first_positive *= scaling.objective;
    }
    settings.termination.max_complementarity_gap *= scaling.objective;
    settings.termination.max_merit_slope *= scaling.objective;
    settings.line_search.min_merit_slope_to_skip_line_search *=
        scaling.objective;
  }
  const double *model_x = nullptr;
  const double *model_y = nullptr;
  const double *model_z = nullptr;
  bool derivatives_current = false;
  const auto model_callback =
      [&](const sip::ModelCallbackInput &input) -> void {
    if (scaling_enabled) {
      for (int i = 0; i < y_dim; ++i) {
        original_y[i] = scaling.y[i] * input.y[i] / scaling.objective;
      }
      for (int i = 0; i < s_dim; ++i) {
        original_z[i] = scaling.z[i] * input.z[i] / scaling.objective;
      }
      model_x = input.x;
      model_y = original_y.data();
      model_z = original_z.data();
    } else {
      model_x = input.x;
      model_y = input.y;
      model_z = input.z;
    }
    if (input.new_x) {
      problem.evaluate_values(model_x);
      if (scaling_enabled) {
        scaling.scale_values(model_output);
      }
    }
    if (input.new_x || input.new_y || input.new_z) {
      derivatives_current = false;
    }
  };
  const auto ensure_derivatives = [&]() -> void {
    if (!derivatives_current) {
      problem.evaluate_derivatives(model_x, model_y, model_z);
      if (scaling_enabled) {
        scaling.scale_derivatives(model_output);
      }
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
                          const double *w, double r1, const double *r2,
                          const double *r3) -> bool {
    ensure_derivatives();
    return callback_provider.factor(w, r1, r2, r3);
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
              .dual = scaling_enabled ? scaling.dual_residual.data() : nullptr,
              .equality = scaling_enabled ? scaling.y.data() : nullptr,
              .inequality = scaling_enabled ? scaling.z.data() : nullptr,
          },
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
  casadi_problems::initialize_slacks_and_duals(
      model_output.g, s_dim, settings.barrier.initial_mu, workspace.vars.s,
      workspace.vars.z);

  sip::Output output = sip::solve(input, settings, workspace);
  if (scaling_enabled) {
    for (int i = 0; i < y_dim; ++i) {
      original_y[i] =
          scaling.y[i] * workspace.vars.y[i] / scaling.objective;
    }
    for (int i = 0; i < s_dim; ++i) {
      original_z[i] =
          scaling.z[i] * workspace.vars.z[i] / scaling.objective;
    }
    problem.evaluate_values(workspace.vars.x);
    problem.evaluate_derivatives(workspace.vars.x, original_y.data(),
                                 original_z.data());
    std::vector<double> dual(model_output.gradient_f,
                             model_output.gradient_f + x_dim);
    const auto add_transpose_product =
        [&](const sip_qdldl::SparseMatrix &matrix,
            const std::vector<double> &multipliers) {
          for (int col = 0; col < matrix.cols; ++col) {
            for (int index = matrix.indptr[col]; index < matrix.indptr[col + 1];
                 ++index) {
              dual[matrix.ind[index]] += matrix.data[index] * multipliers[col];
            }
          }
        };
    add_transpose_product(model_output.jacobian_c, original_y);
    add_transpose_product(model_output.jacobian_g, original_z);
    output.max_dual_violation = 0.0;
    for (double value : dual) {
      output.max_dual_violation =
          std::max(output.max_dual_violation, std::abs(value));
    }
    output.max_primal_violation = 0.0;
    for (int i = 0; i < y_dim; ++i) {
      output.max_primal_violation =
          std::max(output.max_primal_violation, std::abs(model_output.c[i]));
    }
    for (int i = 0; i < s_dim; ++i) {
      output.max_primal_violation =
          std::max(output.max_primal_violation, model_output.g[i]);
    }
  }
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
