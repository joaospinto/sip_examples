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
#include <limits>
#include <optional>
#include <string_view>
#include <vector>

namespace sip_examples::problem_definitions::cutest_problems {
namespace {

struct ModelScaling {
  explicit ModelScaling(int x_dim, int y_dim, int z_dim)
      : x(x_dim, 1.0), y(y_dim, 1.0), z(z_dim, 1.0), dual_residual(x_dim, 1.0),
        row_norm(x_dim + y_dim + z_dim) {}

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
    double maximum_derivative = 0.0;
    for (int i = 0; i < static_cast<int>(x.size()); ++i) {
      maximum_derivative =
          std::max(maximum_derivative, std::abs(output.gradient_f[i]));
    }
    const auto &hessian = output.upper_hessian_lagrangian;
    for (int index = 0; index < hessian.indptr[hessian.cols]; ++index) {
      maximum_derivative =
          std::max(maximum_derivative, std::abs(hessian.data[index]));
    }
    if (maximum_derivative > kMaximumDerivative) {
      objective = kMaximumDerivative / maximum_derivative;
      std::fill(dual_residual.begin(), dual_residual.end(), objective);
    }
  }

  auto is_identity() const -> bool {
    return objective == 1.0 &&
           std::all_of(x.begin(), x.end(),
                       [](const double value) { return value == 1.0; }) &&
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
      output.gradient_f[i] *= objective * x[i];
    }
    auto &hessian = output.upper_hessian_lagrangian;
    for (int col = 0; col < hessian.cols; ++col) {
      for (int index = hessian.indptr[col]; index < hessian.indptr[col + 1];
           ++index) {
        hessian.data[index] *= objective * x[hessian.ind[index]] * x[col];
      }
    }
    auto &jacobian_c = output.jacobian_c;
    for (int col = 0; col < jacobian_c.cols; ++col) {
      for (int index = jacobian_c.indptr[col];
           index < jacobian_c.indptr[col + 1]; ++index) {
        jacobian_c.data[index] *= y[col] * x[jacobian_c.ind[index]];
      }
    }
    auto &jacobian_g = output.jacobian_g;
    for (int col = 0; col < jacobian_g.cols; ++col) {
      for (int index = jacobian_g.indptr[col];
           index < jacobian_g.indptr[col + 1]; ++index) {
        jacobian_g.data[index] *= z[col] * x[jacobian_g.ind[index]];
      }
    }
  }

  void compute_residual_scaling() {
    for (int i = 0; i < static_cast<int>(x.size()); ++i) {
      dual_residual[i] = objective * x[i];
    }
  }

  void to_scaled_primal(const double *original, double *scaled) const {
    for (int i = 0; i < static_cast<int>(x.size()); ++i) {
      scaled[i] = original[i] / x[i];
    }
  }

  void to_original_variables(const double *scaled_x, const double *scaled_y,
                             const double *scaled_z, double *original_x,
                             double *original_y, double *original_z) const {
    for (int i = 0; i < static_cast<int>(x.size()); ++i) {
      original_x[i] = x[i] * scaled_x[i];
    }
    for (int i = 0; i < static_cast<int>(y.size()); ++i) {
      original_y[i] = y[i] * scaled_y[i] / objective;
    }
    for (int i = 0; i < static_cast<int>(z.size()); ++i) {
      original_z[i] = z[i] * scaled_z[i] / objective;
    }
  }

  std::vector<double> x;
  std::vector<double> y;
  std::vector<double> z;
  std::vector<double> dual_residual;
  std::vector<double> row_norm;
  double objective{1.0};
};

void add_compensated(const double term, double &sum, double &correction) {
  const double next = sum + term;
  correction += std::abs(sum) >= std::abs(term) ? (sum - next) + term
                                                : (term - next) + sum;
  sum = next;
}

class AffineQpModel {
public:
  AffineQpModel(sip_qdldl::ModelCallbackOutput &output,
                const ModelScaling &scaling)
      : scaled_linear_(output.gradient_f, output.gradient_f + scaling.x.size()),
        scaled_constant_(scaling.objective * output.f),
        equality_rhs_(scaling.y.size()), inequality_rhs_(scaling.z.size()),
        jacobian_c_(output.jacobian_c.data,
                    output.jacobian_c.data +
                        output.jacobian_c.indptr[output.jacobian_c.cols]),
        jacobian_g_(output.jacobian_g.data,
                    output.jacobian_g.data +
                        output.jacobian_g.indptr[output.jacobian_g.cols]) {
    for (int i = 0; i < static_cast<int>(scaled_linear_.size()); ++i) {
      scaled_linear_[i] *= scaling.objective * scaling.x[i];
    }
    for (int i = 0; i < static_cast<int>(equality_rhs_.size()); ++i) {
      equality_rhs_[i] = -output.c[i];
    }
    for (int i = 0; i < static_cast<int>(inequality_rhs_.size()); ++i) {
      inequality_rhs_[i] = -output.g[i];
    }
    scaling.scale_derivatives(output);
  }

  void evaluate_values(const double *x, const ModelScaling &scaling,
                       sip_qdldl::ModelCallbackOutput &output) const {
    std::copy(scaled_linear_.begin(), scaled_linear_.end(), output.gradient_f);
    sip_qdldl::add_Ax_to_y_where_A_upper_symmetric(
        output.upper_hessian_lagrangian, x, output.gradient_f);

    double objective = scaled_constant_;
    double correction = 0.0;
    for (int i = 0; i < static_cast<int>(scaled_linear_.size()); ++i) {
      add_compensated(x[i] * (scaled_linear_[i] + output.gradient_f[i]) / 2.0,
                      objective, correction);
    }
    output.f = objective + correction;
    evaluate_constraints(x, scaling, output);
  }

  void evaluate_constraints(const double *x, const ModelScaling &scaling,
                            sip_qdldl::ModelCallbackOutput &output) const {
    evaluate_affine_rows(output.jacobian_c, jacobian_c_, equality_rhs_,
                         scaling.x, scaling.y, x, output.c);
    evaluate_affine_rows(output.jacobian_g, jacobian_g_, inequality_rhs_,
                         scaling.x, scaling.z, x, output.g);
  }

private:
  static void evaluate_affine_rows(const sip_qdldl::SparseMatrix &matrix,
                                   const std::vector<double> &data,
                                   const std::vector<double> &rhs,
                                   const std::vector<double> &variable_scale,
                                   const std::vector<double> &row_scale,
                                   const double *x, double *values) {
    for (int row = 0; row < matrix.cols; ++row) {
      double value = -rhs[row];
      double correction = 0.0;
      for (int index = matrix.indptr[row]; index < matrix.indptr[row + 1];
           ++index) {
        const int variable = matrix.ind[index];
        add_compensated(data[index] * variable_scale[variable] * x[variable],
                        value, correction);
      }
      values[row] = row_scale[row] * (value + correction);
    }
  }

  std::vector<double> scaled_linear_;
  double scaled_constant_;
  std::vector<double> equality_rhs_;
  std::vector<double> inequality_rhs_;
  std::vector<double> jacobian_c_;
  std::vector<double> jacobian_g_;
};

auto run(const char *runtime_path, const char *problem_library_path,
         const char *outsdif_path, bool use_qp_settings) -> sip::Output {
  CutestProblem problem(runtime_path, problem_library_path, outsdif_path);
  const int x_dim = problem.x_dim();
  const int y_dim = problem.equality_dim();
  const int s_dim = problem.inequality_dim();
  const int kkt_dim = x_dim + y_dim + s_dim;

  auto settings = casadi_problems::default_casadi_problem_settings(2000);
  settings.line_search.skip_line_search = false;
  settings.line_search.max_iterations = 5000;
  settings.regularization.maximum = 1e12;
  settings.regularization.max_attempts = 40;
  settings.termination.max_merit_slope = 1e-24;
  if (use_qp_settings) {
    settings.barrier.mu_update_factor = 0.2;
    settings.barrier.use_predictor_corrector = true;
    settings.line_search.skip_line_search = true;
    settings.line_search.tau = 0.99;
    settings.num_iterative_refinement_steps = 1;
    settings.regularization.initial = 3e-5;
    settings.regularization.decrease_factor = 0.15;
  } else {
    settings.line_search.use_filter_line_search = true;
    settings.line_search.filter_min_total_line_search_iterations = 300;
  }
  if (std::getenv("SIP_CUTEST_PRINT_LOGS") != nullptr) {
    casadi_problems::enable_all_casadi_problem_logs(settings);
  }
  if (std::getenv("SIP_CUTEST_PRINT_ITERATION_LOGS") != nullptr) {
    settings.logging.print_logs = true;
  }
  if (const char *max_iterations = std::getenv("SIP_CUTEST_MAX_ITERATIONS");
      max_iterations != nullptr) {
    settings.max_iterations = std::stoi(max_iterations);
  }
  sip::Workspace workspace;
  workspace.reserve(x_dim, s_dim, y_dim, settings);

  sip_qdldl::Workspace qdldl_workspace;
  qdldl_workspace.reserve(kkt_dim, s_dim, problem.kkt_nnz(),
                          problem.kkt_l_nnz());

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
  std::optional<AffineQpModel> qp_model;
  if (use_qp_settings) {
    std::vector<double> zero_x(x_dim, 0.0);
    problem.evaluate_values(zero_x.data());
    problem.evaluate_derivatives(zero_x.data(), original_y.data(),
                                 original_z.data());
    scaling.compute(model_output);
    scaling_enabled = true;
    qp_model.emplace(model_output, scaling);
  } else {
    if (y_dim == 0 && s_dim == 0) {
      scaling.compute_nlp(model_output);
      double maximum_hessian = 0.0;
      const auto &hessian = model_output.upper_hessian_lagrangian;
      for (int index = 0; index < hessian.indptr[hessian.cols]; ++index) {
        maximum_hessian =
            std::max(maximum_hessian, std::abs(hessian.data[index]));
      }
      const double hessian_regularization_limit = std::min(
          std::numeric_limits<double>::max(), 10.0 * x_dim * maximum_hessian);
      settings.regularization.maximum = std::max(
          settings.regularization.maximum, hessian_regularization_limit);
    } else {
      scaling.compute(model_output);
    }
    scaling_enabled = !scaling.is_identity();
    settings.regularization.initial *= scaling.objective;
    settings.regularization.first_positive *= scaling.objective;
    settings.regularization.maximum *= scaling.objective;
    settings.termination.max_merit_slope *= scaling.objective;
    settings.termination.max_cost_change *= scaling.objective;
    settings.line_search.min_merit_slope_to_skip_line_search *=
        scaling.objective;
  }
  scaling.compute_residual_scaling();
  const double *model_x = nullptr;
  const double *model_y = nullptr;
  const double *model_z = nullptr;
  bool derivatives_current = use_qp_settings;
  std::vector<double> original_x(x_dim);
  const auto model_callback =
      [&](const sip::ModelCallbackInput &input) -> void {
    if (scaling_enabled) {
      scaling.to_original_variables(input.x, input.y, input.z,
                                    original_x.data(), original_y.data(),
                                    original_z.data());
      model_x = original_x.data();
      model_y = original_y.data();
      model_z = original_z.data();
    } else {
      model_x = input.x;
      model_y = input.y;
      model_z = input.z;
    }
    if (input.new_x) {
      if (use_qp_settings) {
        qp_model->evaluate_values(input.x, scaling, model_output);
      } else {
        problem.evaluate_values(model_x);
        if (scaling_enabled) {
          scaling.scale_values(model_output);
        }
      }
    }
    if (!use_qp_settings && (input.new_x || input.new_y || input.new_z)) {
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
                          const double *r3,
                          double factorization_regularization) -> bool {
    ensure_derivatives();
    return callback_provider.factor(w, r1, r2, r3,
                                    factorization_regularization);
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

  scaling.to_scaled_primal(problem.initial_x().data(), workspace.vars.x);
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
    scaling.to_original_variables(workspace.vars.x, workspace.vars.y,
                                  workspace.vars.z, original_x.data(),
                                  original_y.data(), original_z.data());
    if (use_qp_settings) {
      qp_model->evaluate_constraints(workspace.vars.x, scaling, model_output);
      for (int i = 0; i < y_dim; ++i) {
        model_output.c[i] /= scaling.y[i];
      }
      for (int i = 0; i < s_dim; ++i) {
        model_output.g[i] /= scaling.z[i];
      }
    } else {
      problem.evaluate_values(original_x.data());
    }
    problem.evaluate_derivatives(original_x.data(), original_y.data(),
                                 original_z.data());
    std::vector<double> dual(model_output.gradient_f,
                             model_output.gradient_f + x_dim);
    std::vector<double> dual_correction(x_dim, 0.0);
    const auto add_transpose_product =
        [&](const sip_qdldl::SparseMatrix &matrix,
            const std::vector<double> &multipliers) {
          for (int col = 0; col < matrix.cols; ++col) {
            for (int index = matrix.indptr[col]; index < matrix.indptr[col + 1];
                 ++index) {
              const int row = matrix.ind[index];
              add_compensated(matrix.data[index] * multipliers[col], dual[row],
                              dual_correction[row]);
            }
          }
        };
    add_transpose_product(model_output.jacobian_c, original_y);
    add_transpose_product(model_output.jacobian_g, original_z);
    for (int i = 0; i < x_dim; ++i) {
      dual[i] += dual_correction[i];
    }
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
    if (output.exit_status == sip::Status::SOLVED &&
        !(output.max_primal_violation <
              settings.termination.max_constraint_violation &&
          output.max_dual_violation < settings.termination.max_dual_residual)) {
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
