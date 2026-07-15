#pragma once

#include "problem_definitions/casadi_problems/common/problem.hpp"
#include "sip-slacg/helpers/helpers.hpp"
#include "sip/sip.hpp"

#include <algorithm>
#include <functional>
#include <vector>

namespace sip_examples::problem_definitions::casadi_problems {

struct FlatSlacgResult {
  sip::Output output;
  std::vector<double> x;
};

template <typename GeneratedProblem>
FlatSlacgResult run_flat_slacg(const sip::Settings &settings) {
  const auto &spec = GeneratedProblem::flat_spec();

  sip_examples::ModelCallbackOutput mco;
  mco.reserve(spec.x_dim, spec.s_dim, spec.y_dim,
              spec.upper_hessian_lagrangian_nnz, spec.jacobian_c_nnz,
              spec.jacobian_g_nnz);

  auto work = GeneratedProblem::make_flat_work();
  std::vector<double> jacobian_c_transpose(spec.jacobian_c_transpose_nnz);
  std::vector<double> jacobian_g_transpose(spec.jacobian_g_transpose_nnz);

  auto model_callback = [&](const sip::ModelCallbackInput &mci) -> void {
    GeneratedProblem::eval_flat_slacg(
        mci.x, mci.y, mci.z, &mco.f, mco.gradient_f, mco.c, mco.g,
        mco.upper_hessian_lagrangian, mco.jacobian_c, mco.jacobian_g,
        jacobian_c_transpose.data(), jacobian_g_transpose.data(), work);
  };

  std::vector<double> LT_data(::sip_examples::problem_definitions::
                                  casadi_problems::generated_problem::L_nnz);
  std::vector<double> D_diag(::sip_examples::problem_definitions::
                                 casadi_problems::generated_problem::core_dim);
  std::vector<double> border_solution(
      ::sip_examples::problem_definitions::casadi_problems::generated_problem::
          border_solution_size);
  std::vector<double> border_factor(
      ::sip_examples::problem_definitions::casadi_problems::generated_problem::
          border_factor_size);
  std::vector<double> factor_r2(spec.y_dim);
  std::vector<double> factor_r3(spec.s_dim);

  const auto factor = [&](const double *w, const double r1, const double *r2,
                          const double *r3,
                          const double factorization_regularization) -> bool {
    const double *regularized_r2 = r2;
    const double *regularized_r3 = r3;
    if (factorization_regularization != 0.0) {
      for (int i = 0; i < spec.y_dim; ++i) {
        factor_r2[i] = r2[i] + factorization_regularization;
      }
      for (int i = 0; i < spec.s_dim; ++i) {
        factor_r3[i] = r3[i] + factorization_regularization;
      }
      regularized_r2 = factor_r2.data();
      regularized_r3 = factor_r3.data();
    }
    return ::sip_examples::problem_definitions::casadi_problems::
        generated_problem::ldlt_factor(
            mco.upper_hessian_lagrangian, mco.jacobian_c, mco.jacobian_g, w,
            r1 + factorization_regularization, regularized_r2, regularized_r3,
            LT_data.data(), D_diag.data(), border_solution.data(),
            border_factor.data());
  };
  const auto solve = [&](const double *b, double *v) -> void {
    return ::sip_examples::problem_definitions::casadi_problems::
        generated_problem::ldlt_solve(LT_data.data(), D_diag.data(),
                                      border_solution.data(),
                                      border_factor.data(), b, v);
  };
  const auto add_Kx_to_y =
      [&mco](const double *w, const double r1, const double *r2,
             const double *r3, const double *x_x, const double *x_y,
             const double *x_z, double *y_x, double *y_y, double *y_z) -> void {
    return ::sip_examples::problem_definitions::casadi_problems::
        generated_problem::add_Kx_to_y(mco.upper_hessian_lagrangian,
                                       mco.jacobian_c, mco.jacobian_g, w, r1,
                                       r2, r3, x_x, x_y, x_z, y_x, y_y, y_z);
  };
  const auto add_Hx_to_y = [&mco](const double *x, double *y) -> void {
    return ::sip_examples::problem_definitions::casadi_problems::
        generated_problem::add_upper_symmetric_Hx_to_y(
            mco.upper_hessian_lagrangian, x, y);
  };
  const auto add_Cx_to_y = [&mco](const double *x, double *y) -> void {
    return ::sip_examples::problem_definitions::casadi_problems::
        generated_problem::add_Cx_to_y(mco.jacobian_c, x, y);
  };
  const auto add_CTx_to_y = [&mco](const double *x, double *y) -> void {
    return ::sip_examples::problem_definitions::casadi_problems::
        generated_problem::add_CTx_to_y(mco.jacobian_c, x, y);
  };
  const auto add_Gx_to_y = [&mco](const double *x, double *y) -> void {
    return ::sip_examples::problem_definitions::casadi_problems::
        generated_problem::add_Gx_to_y(mco.jacobian_g, x, y);
  };
  const auto add_GTx_to_y = [&mco](const double *x, double *y) -> void {
    return ::sip_examples::problem_definitions::casadi_problems::
        generated_problem::add_GTx_to_y(mco.jacobian_g, x, y);
  };
  const auto get_f = [&mco]() -> double { return mco.f; };
  const auto get_grad_f = [&mco]() -> const double * { return mco.gradient_f; };
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

  sip::Workspace workspace;
  workspace.reserve(spec.x_dim, spec.s_dim, spec.y_dim, settings);
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

  workspace.free();
  mco.free();

  return {.output = output, .x = std::move(x)};
}

} // namespace sip_examples::problem_definitions::casadi_problems
