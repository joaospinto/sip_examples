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

void print_array_diagnostics(const char *name, const double *values,
                             const int size) {
  int nonfinite = 0;
  double maximum = 0.0;
  for (int i = 0; i < size; ++i) {
    if (std::isfinite(values[i])) {
      maximum = std::max(maximum, std::abs(values[i]));
    } else {
      ++nonfinite;
      std::cerr << name << "[" << i << "]=" << values[i] << '\n';
    }
  }
  std::cerr << name << ": size=" << size << " nonfinite=" << nonfinite
            << " max_abs=" << maximum << '\n';
}

void print_sparse_diagnostics(const char *name,
                              const sip_qdldl::SparseMatrix &matrix) {
  print_array_diagnostics(name, matrix.data, matrix.indptr[matrix.cols]);
  for (int col = 0; col < matrix.cols; ++col) {
    for (int index = matrix.indptr[col]; index < matrix.indptr[col + 1];
         ++index) {
      if (!std::isfinite(matrix.data[index])) {
        std::cerr << name << " nonfinite at (" << matrix.ind[index] << ", "
                  << col << ")\n";
      }
    }
  }
}

struct ModelScaling {
  explicit ModelScaling(int x_dim, int y_dim, int z_dim)
      : x(x_dim, 1.0), y(y_dim, 1.0), z(z_dim, 1.0),
        primal_regularization(x_dim, 1.0), dual_residual(x_dim, 1.0),
        equality_residual(y_dim, 1.0), inequality_residual(z_dim, 1.0),
        row_norm(x_dim + y_dim + z_dim) {}

  void compute(const sip_qdldl::ModelCallbackOutput &output) {
    const double minimum_scale =
        std::sqrt(std::numeric_limits<double>::epsilon());
    const double maximum_scale = 1.0 / minimum_scale;
    const int x_dim = static_cast<int>(x.size());
    const int y_dim = static_cast<int>(y.size());
    const int z_dim = static_cast<int>(z.size());
    for (int iteration = 0; iteration < 40; ++iteration) {
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
        x[i] = std::clamp(x[i] * row_norm[i], minimum_scale, maximum_scale);
      }
      for (int i = 0; i < y_dim; ++i) {
        y[i] = std::clamp(y[i] * row_norm[x_dim + i], minimum_scale,
                          maximum_scale);
      }
      for (int i = 0; i < z_dim; ++i) {
        z[i] = std::clamp(z[i] * row_norm[x_dim + y_dim + i], minimum_scale,
                          maximum_scale);
      }
      if (max_change <= 1e-3) {
        break;
      }
    }
  }

  void scale_derivatives(sip_qdldl::ModelCallbackOutput &output) const {
    for (int i = 0; i < static_cast<int>(x.size()); ++i) {
      output.gradient_f[i] *= x[i];
    }
    auto &hessian = output.upper_hessian_lagrangian;
    for (int col = 0; col < hessian.cols; ++col) {
      for (int index = hessian.indptr[col]; index < hessian.indptr[col + 1];
           ++index) {
        hessian.data[index] *= x[hessian.ind[index]] * x[col];
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

  void scale_constraint_values(sip_qdldl::ModelCallbackOutput &output) const {
    for (int i = 0; i < static_cast<int>(y.size()); ++i) {
      output.c[i] *= y[i];
    }
    for (int i = 0; i < static_cast<int>(z.size()); ++i) {
      output.g[i] *= z[i];
    }
  }

  void compute_residual_scaling() {
    for (int i = 0; i < static_cast<int>(x.size()); ++i) {
      primal_regularization[i] = x[i] * x[i];
      dual_residual[i] = 1.0 / x[i];
    }
    for (int i = 0; i < static_cast<int>(y.size()); ++i) {
      equality_residual[i] = 1.0 / y[i];
    }
    for (int i = 0; i < static_cast<int>(z.size()); ++i) {
      inequality_residual[i] = 1.0 / z[i];
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
      original_y[i] = y[i] * scaled_y[i];
    }
    for (int i = 0; i < static_cast<int>(z.size()); ++i) {
      original_z[i] = z[i] * scaled_z[i];
    }
  }

  std::vector<double> x;
  std::vector<double> y;
  std::vector<double> z;
  std::vector<double> primal_regularization;
  std::vector<double> dual_residual;
  std::vector<double> equality_residual;
  std::vector<double> inequality_residual;
  std::vector<double> row_norm;
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
        scaled_constant_(output.f), equality_rhs_(scaling.y.size()),
        inequality_rhs_(scaling.z.size()),
        jacobian_c_(output.jacobian_c.data,
                    output.jacobian_c.data +
                        output.jacobian_c.indptr[output.jacobian_c.cols]),
        jacobian_g_(output.jacobian_g.data,
                    output.jacobian_g.data +
                        output.jacobian_g.indptr[output.jacobian_g.cols]) {
    for (int i = 0; i < static_cast<int>(scaled_linear_.size()); ++i) {
      scaled_linear_[i] *= scaling.x[i];
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
  const bool scale_nlp =
      !use_qp_settings && std::getenv("SIP_CUTEST_SCALE_NLP") != nullptr;
  const bool equilibrate_primal_regularization =
      !use_qp_settings &&
      std::getenv("SIP_CUTEST_EQUILIBRATE_PRIMAL_REGULARIZATION") != nullptr;
  const bool use_model_scaling = use_qp_settings || scale_nlp;

  auto settings = casadi_problems::default_casadi_problem_settings(1000);
  settings.line_search.skip_line_search = false;
  settings.line_search.max_iterations = 5000;
  settings.regularization.maximum = 1e12;
  settings.regularization.max_attempts = 26;
  if (std::getenv("SIP_CUTEST_EXTEND_REGULARIZATION_RANGE") != nullptr) {
    settings.regularization.maximum = std::numeric_limits<double>::max();
    settings.regularization.max_attempts = 40;
  }
  settings.termination.max_merit_slope = 1e-24;
  if (use_qp_settings) {
    settings.barrier.mu_update_factor = 0.2;
    settings.barrier.use_predictor_corrector = true;
    settings.line_search.skip_line_search = true;
    settings.line_search.tau = 0.99;
    settings.num_iterative_refinement_steps = 1;
    settings.regularization.max_attempts = 40;
    settings.regularization.initial = 3e-5;
    settings.regularization.decrease_factor = 0.15;
  } else {
    settings.line_search.use_filter_line_search = true;
    settings.line_search.filter_min_total_line_search_iterations = 300;
    settings.line_search.use_primal_dual_merit =
        std::getenv("SIP_CUTEST_USE_PRIMAL_DUAL_MERIT") != nullptr;
    if (const char *weight = std::getenv("SIP_CUTEST_PRIMAL_DUAL_MERIT_WEIGHT");
        weight != nullptr) {
      settings.line_search.primal_dual_merit_weight = std::stod(weight);
    }
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
  settings.line_search.start_ls_with_alpha_s_max =
      std::getenv("SIP_CUTEST_START_LS_WITH_ALPHA_S_MAX") != nullptr;
  sip::Workspace workspace;
  workspace.reserve(x_dim, s_dim, y_dim, settings);

  sip_qdldl::Workspace qdldl_workspace;
  qdldl_workspace.reserve(kkt_dim, problem.kkt_nnz(), problem.kkt_l_nnz());

  auto &model_output = problem.model_output();
  ModelScaling scaling(x_dim, y_dim, s_dim);
  std::vector<double> original_y(y_dim);
  std::vector<double> original_z(s_dim);
  std::fill(original_y.begin(), original_y.end(), 0.0);
  std::fill(original_z.begin(), original_z.end(), 0.0);
  std::optional<AffineQpModel> qp_model;
  if (use_model_scaling || equilibrate_primal_regularization) {
    std::vector<double> zero_x(x_dim, 0.0);
    const double *scaling_x =
        use_qp_settings ? zero_x.data() : problem.initial_x().data();
    problem.evaluate_values(scaling_x);
    problem.evaluate_derivatives(scaling_x, original_y.data(),
                                 original_z.data());
    scaling.compute(model_output);
    scaling.compute_residual_scaling();
    if (equilibrate_primal_regularization && !scale_nlp) {
      std::fill(scaling.primal_regularization.begin(),
                scaling.primal_regularization.end(), 1.0);
    }
  }
  if (use_qp_settings) {
    qp_model.emplace(model_output, scaling);
  }
  const double *model_x = nullptr;
  const double *model_y = nullptr;
  const double *model_z = nullptr;
  bool derivatives_current = use_qp_settings;
  std::vector<double> original_x(x_dim);
  const auto model_callback =
      [&](const sip::ModelCallbackInput &input) -> void {
    if (use_model_scaling) {
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
        if (scale_nlp) {
          scaling.scale_constraint_values(model_output);
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
      if (use_model_scaling) {
        scaling.scale_derivatives(model_output);
      }
      derivatives_current = true;
    }
  };
  const sip_qdldl::Settings qdldl_settings{
      .permute_kkt_system = true,
      .compensate_constraint_products = use_qp_settings,
      .primal_regularization_scaling =
          scale_nlp || equilibrate_primal_regularization
              ? scaling.primal_regularization.data()
              : nullptr,
      .kkt_pinv = problem.kkt_pinv(),
      .constant_singleton_inequalities =
          use_qp_settings || problem.all_inequalities_are_variable_bounds()
              ? problem.constant_singleton_inequalities()
              : nullptr,
  };
  sip_qdldl::CallbackProvider callback_provider(qdldl_settings, model_output,
                                                qdldl_workspace);
  const auto update_primal_regularization_scaling = [&](const double r1) {
    if (!equilibrate_primal_regularization || scale_nlp) {
      return;
    }
    const bool use_equilibrated_regularization =
        r1 > settings.regularization.initial;
    for (int i = 0; i < x_dim; ++i) {
      scaling.primal_regularization[i] =
          use_equilibrated_regularization
              ? std::max(1.0, 1.0 / (scaling.x[i] * scaling.x[i]))
              : 1.0;
    }
  };
  const bool print_model_diagnostics =
      std::getenv("SIP_CUTEST_PRINT_MODEL_DIAGNOSTICS") != nullptr;
  const bool print_factor_diagnostics =
      std::getenv("SIP_CUTEST_PRINT_FACTOR_DIAGNOSTICS") != nullptr;
  bool printed_model_diagnostics = false;

  const auto factor =
      [&callback_provider, &ensure_derivatives,
       &update_primal_regularization_scaling, &model_output, &model_x,
       &printed_model_diagnostics, print_model_diagnostics,
       print_factor_diagnostics, x_dim, y_dim,
       s_dim](const double *w, double r1, const double *r2, const double *r3,
              double factorization_regularization) -> bool {
    update_primal_regularization_scaling(r1);
    ensure_derivatives();
    if (print_model_diagnostics && !printed_model_diagnostics) {
      std::cerr << "f=" << model_output.f << '\n';
      print_array_diagnostics("x", model_x, x_dim);
      print_array_diagnostics("gradient_f", model_output.gradient_f, x_dim);
      print_array_diagnostics("c", model_output.c, y_dim);
      print_array_diagnostics("g", model_output.g, s_dim);
      print_sparse_diagnostics("hessian",
                               model_output.upper_hessian_lagrangian);
      print_sparse_diagnostics("jacobian_c", model_output.jacobian_c);
      print_sparse_diagnostics("jacobian_g", model_output.jacobian_g);
      printed_model_diagnostics = true;
    }
    const bool success =
        callback_provider.factor(w, r1, r2, r3, factorization_regularization);
    if (print_factor_diagnostics) {
      std::cerr << "target_regularization=" << r1
                << " factorization_regularization="
                << factorization_regularization << " success=" << success
                << '\n';
    }
    return success;
  };
  const auto solve = [&callback_provider](const double *b, double *v) -> void {
    callback_provider.solve(b, v);
  };
  const auto add_kx_to_y = [&callback_provider, &ensure_derivatives,
                            &update_primal_regularization_scaling](
                               const double *w, double r1, const double *r2,
                               const double *r3, const double *x_x,
                               const double *x_y, const double *x_z,
                               double *y_x, double *y_y, double *y_z) -> void {
    update_primal_regularization_scaling(r1);
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
              .dual =
                  use_model_scaling ? scaling.dual_residual.data() : nullptr,
              .equality = use_model_scaling ? scaling.equality_residual.data()
                                            : nullptr,
              .inequality = use_model_scaling
                                ? scaling.inequality_residual.data()
                                : nullptr,
          },
      .dimensions =
          {
              .x_dim = x_dim,
              .s_dim = s_dim,
              .y_dim = y_dim,
          },
  };

  const auto initialize_primal = [&](const double *initial_x) {
    if (use_model_scaling) {
      scaling.to_scaled_primal(initial_x, workspace.vars.x);
    } else {
      std::copy_n(initial_x, x_dim, workspace.vars.x);
    }
  };
  if (!use_qp_settings &&
      std::getenv("SIP_CUTEST_PUSH_INITIAL_BOUNDS") != nullptr) {
    problem.initialize_x(original_x.data(), 1e-2, 1e-2);
    initialize_primal(original_x.data());
  } else {
    initialize_primal(problem.initial_x().data());
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

  sip::Output output = sip::solve(input, settings, workspace);
  if (use_model_scaling) {
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
