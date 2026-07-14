#pragma once

#include "problem_definitions/casadi_problems/common/problem.hpp"
#include "sip/sip.hpp"
#include "sip_qdldl/sip_qdldl.hpp"

#include <algorithm>
#include <cmath>
#include <functional>
#include <vector>

namespace sip_examples::problem_definitions::casadi_problems {

struct FlatQdldlResult {
  sip::Output output;
  std::vector<double> x;
};

namespace detail {

inline void copy_sparse_pattern(const int nnz, const int cols, const int *ind,
                                const int *indptr,
                                sip_qdldl::SparseMatrix &matrix) {
  std::copy_n(ind, nnz, matrix.ind);
  std::copy_n(indptr, cols + 1, matrix.indptr);
}

} // namespace detail

template <typename GeneratedProblem>
FlatQdldlResult run_flat_qdldl(const sip::Settings &settings) {
  const auto &spec = GeneratedProblem::flat_spec();

  sip::Workspace workspace;
  workspace.reserve(spec.x_dim, spec.s_dim, spec.y_dim, settings);

  sip_qdldl::ModelCallbackOutput mco;
  constexpr bool kIsUpperHessianLagrangianTransposed = false;
  constexpr bool kIsJacobianCTransposed = true;
  constexpr bool kIsJacobianGTransposed = true;
  mco.reserve(spec.x_dim, spec.s_dim, spec.y_dim,
              spec.upper_hessian_lagrangian_nnz, spec.jacobian_c_transpose_nnz,
              spec.jacobian_g_transpose_nnz, kIsJacobianCTransposed,
              kIsJacobianGTransposed);

  mco.upper_hessian_lagrangian.rows = spec.x_dim;
  mco.upper_hessian_lagrangian.cols = spec.x_dim;
  mco.upper_hessian_lagrangian.is_transposed =
      kIsUpperHessianLagrangianTransposed;
  detail::copy_sparse_pattern(spec.upper_hessian_lagrangian_nnz, spec.x_dim,
                              spec.upper_hessian_lagrangian_ind,
                              spec.upper_hessian_lagrangian_indptr,
                              mco.upper_hessian_lagrangian);

  mco.jacobian_c.rows = spec.x_dim;
  mco.jacobian_c.cols = spec.y_dim;
  mco.jacobian_c.is_transposed = kIsJacobianCTransposed;
  detail::copy_sparse_pattern(spec.jacobian_c_transpose_nnz, spec.y_dim,
                              spec.jacobian_c_transpose_ind,
                              spec.jacobian_c_transpose_indptr, mco.jacobian_c);

  mco.jacobian_g.rows = spec.x_dim;
  mco.jacobian_g.cols = spec.s_dim;
  mco.jacobian_g.is_transposed = kIsJacobianGTransposed;
  detail::copy_sparse_pattern(spec.jacobian_g_transpose_nnz, spec.s_dim,
                              spec.jacobian_g_transpose_ind,
                              spec.jacobian_g_transpose_indptr, mco.jacobian_g);

  auto work = GeneratedProblem::make_flat_work();
  std::vector<double> c_data(spec.jacobian_c_nnz);
  std::vector<double> g_data(spec.jacobian_g_nnz);

  const double *model_x = nullptr;
  const double *model_y = nullptr;
  const double *model_z = nullptr;
  bool derivatives_current = false;
  auto model_callback = [&](const sip::ModelCallbackInput &mci) -> void {
    model_x = mci.x;
    model_y = mci.y;
    model_z = mci.z;
    if (mci.new_x) {
      GeneratedProblem::eval_flat_values(mci.x, &mco.f, mco.c, mco.g, work);
    }
    if (mci.new_x || mci.new_y || mci.new_z) {
      derivatives_current = false;
    }
  };
  const auto ensure_derivatives = [&]() -> void {
    if (!derivatives_current) {
      GeneratedProblem::eval_flat_qdldl(
          model_x, model_y, model_z, &mco.f, mco.gradient_f, mco.c, mco.g,
          mco.upper_hessian_lagrangian.data, c_data.data(), g_data.data(),
          mco.jacobian_c.data, mco.jacobian_g.data, work);
      derivatives_current = true;
    }
  };

  sip_qdldl::Workspace qdldl_workspace;
  qdldl_workspace.reserve(spec.kkt_dim, spec.kkt_nnz, spec.kkt_L_nnz);

  const sip_qdldl::Settings qdldl_settings{
      .permute_kkt_system = true,
      .kkt_pinv = spec.kkt_pinv,
  };
  auto callback_provider =
      sip_qdldl::CallbackProvider(qdldl_settings, mco, qdldl_workspace);

  const auto factor = [&callback_provider, &ensure_derivatives](
                          const double *w, const double r1, const double *r2,
                          const double *r3) -> bool {
    ensure_derivatives();
    return callback_provider.factor(w, r1, r2, r3);
  };
  const auto solve = [&callback_provider](const double *b, double *v) -> void {
    callback_provider.solve(b, v);
  };
  const auto add_Kx_to_y =
      [&callback_provider](const double *w, const double r1, const double *r2,
                           const double *r3, const double *x_x,
                           const double *x_y, const double *x_z, double *y_x,
                           double *y_y, double *y_z) -> void {
    callback_provider.add_Kx_to_y(w, r1, r2, r3, x_x, x_y, x_z, y_x, y_y, y_z);
  };
  const auto add_Hx_to_y = [&callback_provider](const double *x,
                                                double *y) -> void {
    callback_provider.add_Hx_to_y(x, y);
  };
  const auto add_Cx_to_y = [&callback_provider](const double *x,
                                                double *y) -> void {
    callback_provider.add_Cx_to_y(x, y);
  };
  const auto add_CTx_to_y = [&callback_provider](const double *x,
                                                 double *y) -> void {
    callback_provider.add_CTx_to_y(x, y);
  };
  const auto add_Gx_to_y = [&callback_provider](const double *x,
                                                double *y) -> void {
    callback_provider.add_Gx_to_y(x, y);
  };
  const auto add_GTx_to_y = [&callback_provider](const double *x,
                                                 double *y) -> void {
    callback_provider.add_GTx_to_y(x, y);
  };
  const auto get_f = [&mco]() -> double { return mco.f; };
  const auto get_grad_f = [&mco, &ensure_derivatives]() -> const double * {
    ensure_derivatives();
    return mco.gradient_f;
  };
  const auto get_c = [&mco]() -> const double * { return mco.c; };
  const auto get_g = [&mco]() -> const double * { return mco.g; };
  const auto timeout_callback = []() -> bool { return false; };

  sip::Input input{
      .factor = std::cref(factor),
      .solve = std::cref(solve),
      .add_Kx_to_y = std::cref(add_Kx_to_y),
      .add_Hx_to_y = std::cref(add_Hx_to_y),
      .add_Cx_to_y = std::cref(add_Cx_to_y),
      .add_CTx_to_y = std::cref(add_CTx_to_y),
      .add_Gx_to_y = std::cref(add_Gx_to_y),
      .add_GTx_to_y = std::cref(add_GTx_to_y),
      .get_f = std::cref(get_f),
      .get_grad_f = std::cref(get_grad_f),
      .get_c = std::cref(get_c),
      .get_g = std::cref(get_g),
      .model_callback = std::cref(model_callback),
      .timeout_callback = std::cref(timeout_callback),
      .dimensions =
          {
              .x_dim = spec.x_dim,
              .s_dim = spec.s_dim,
              .y_dim = spec.y_dim,
          },
  };

  std::copy_n(spec.initial_x, spec.x_dim, workspace.vars.x);
  std::fill_n(workspace.vars.y, spec.y_dim, 0.0);
  std::fill_n(workspace.vars.z, spec.s_dim, 1.0);
  model_callback({.x = workspace.vars.x,
                  .y = workspace.vars.y,
                  .z = workspace.vars.z,
                  .new_x = true,
                  .new_y = true,
                  .new_z = true});
  initialize_slacks_and_duals(mco.g, spec.s_dim, settings.barrier.initial_mu,
                              workspace.vars.s, workspace.vars.z);

  auto output = sip::solve(input, settings, workspace);
  std::vector<double> x(workspace.vars.x, workspace.vars.x + spec.x_dim);

  qdldl_workspace.free();
  workspace.free();
  mco.free();

  return {.output = output, .x = std::move(x)};
}

} // namespace sip_examples::problem_definitions::casadi_problems
