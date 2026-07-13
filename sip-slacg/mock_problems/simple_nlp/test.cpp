#include "sip-slacg/mock_problems/simple_nlp/kkt_codegen.hpp"

#include "problem_definitions/mock_problems/simple_nlp/problem.hpp"
#include "sip-slacg/helpers/helpers.hpp"
#include "sip/sip.hpp"
#include <gtest/gtest.h>

namespace sip_examples {
namespace problem = ::sip_examples::problem_definitions::simple_nlp;

struct LDLTCallbackProvider {
  double *LT_data;
  double *D_diag;

  auto ldlt_factor(const double *upper_H_data, const double *C_data,
                   const double *G_data, const double *w, const double r1,
                   const double *r2, const double *r3) -> bool {
    return ::sip_examples::ldlt_factor(upper_H_data, C_data, G_data, w, r1, r2,
                                       r3, LT_data, D_diag);
  }

  void ldlt_solve(const double *b, double *v) {
    return ::sip_examples::ldlt_solve(LT_data, D_diag, b, v);
  }

  auto reserve(int L_nnz, int kkt_dim) -> void {
    LT_data = new double[L_nnz];
    D_diag = new double[kkt_dim];
  }

  auto free() -> void {
    delete[] LT_data;
    delete[] D_diag;
  }
};

TEST(SimpleNLP, Problem1) {
  ModelCallbackOutput mco;
  mco.reserve(problem::kXDim, problem::kSDim, problem::kYDim,
              problem::kUpperHessianLagrangianNnz, problem::kJacobianCNnz,
              problem::kJacobianGNnz);

  const auto model_callback = [&mco](const sip::ModelCallbackInput &mci) {
    if (!mci.new_x) {
      return;
    }
    problem::evaluate_slacg(mci, &mco.f, mco.gradient_f, mco.g,
                            mco.upper_hessian_lagrangian, mco.jacobian_g);
  };

  LDLTCallbackProvider ldlt_callback_provider;
  ldlt_callback_provider.reserve(problem::kSlacgLNnz, problem::kKktDim);

  const auto timeout_callback = []() { return false; };

  const auto factor = [&ldlt_callback_provider,
                       &mco](const double *w, const double r1, const double *r2,
                             const double *r3) -> bool {
    return ldlt_callback_provider.ldlt_factor(mco.upper_hessian_lagrangian,
                                              mco.jacobian_c, mco.jacobian_g, w,
                                              r1, r2, r3);
  };

  const auto solve = [&ldlt_callback_provider](const double *b, double *v) {
    return ldlt_callback_provider.ldlt_solve(b, v);
  };

  const auto _add_Kx_to_y =
      [&mco](const double *w, const double r1, const double *r2,
             const double *r3, const double *x_x, const double *x_y,
             const double *x_z, double *y_x, double *y_y, double *y_z) -> void {
    return add_Kx_to_y(mco.upper_hessian_lagrangian, mco.jacobian_c,
                       mco.jacobian_g, w, r1, r2, r3, x_x, x_y, x_z, y_x, y_y,
                       y_z);
  };

  const auto _add_Hx_to_y = [&mco](const double *x, double *y) -> void {
    return add_upper_symmetric_Hx_to_y(mco.upper_hessian_lagrangian, x, y);
  };

  const auto _add_Cx_to_y = [&mco](const double *x, double *y) -> void {
    return add_Cx_to_y(mco.jacobian_c, x, y);
  };

  const auto _add_CTx_to_y = [&mco](const double *x, double *y) -> void {
    return add_CTx_to_y(mco.jacobian_c, x, y);
  };

  const auto _add_Gx_to_y = [&mco](const double *x, double *y) -> void {
    return add_Gx_to_y(mco.jacobian_g, x, y);
  };

  const auto _add_GTx_to_y = [&mco](const double *x, double *y) -> void {
    return add_GTx_to_y(mco.jacobian_g, x, y);
  };

  const auto get_f = [&mco]() -> double { return mco.f; };

  const auto get_grad_f = [&mco]() -> double * { return mco.gradient_f; };

  const auto get_c = [&mco]() -> double * { return mco.c; };

  const auto get_g = [&mco]() -> double * { return mco.g; };

  sip::Input input{
      .factor = std::cref(factor),
      .solve = std::cref(solve),
      .add_Kx_to_y = std::cref(_add_Kx_to_y),
      .add_Hx_to_y = std::cref(_add_Hx_to_y),
      .add_Cx_to_y = std::cref(_add_Cx_to_y),
      .add_CTx_to_y = std::cref(_add_CTx_to_y),
      .add_Gx_to_y = std::cref(_add_Gx_to_y),
      .add_GTx_to_y = std::cref(_add_GTx_to_y),
      .get_f = std::cref(get_f),
      .get_grad_f = std::cref(get_grad_f),
      .get_c = std::cref(get_c),
      .get_g = std::cref(get_g),
      .model_callback = std::cref(model_callback),
      .timeout_callback = std::cref(timeout_callback),
      .dimensions =
          {
              .x_dim = problem::kXDim,
              .s_dim = problem::kSDim,
              .y_dim = problem::kYDim,
          },
  };

  sip::Settings settings = problem::settings();

  sip::Workspace workspace;
  workspace.reserve(
      problem::kXDim, problem::kSDim, problem::kYDim,
      sip::FilterWorkspace::required_capacity(settings));
  problem::initialize(workspace);

  const auto output = sip::solve(input, settings, workspace);

  EXPECT_EQ(output.exit_status, sip::Status::SOLVED);

  EXPECT_NEAR(workspace.vars.x[0], -1.15747396, 1e-3);
  EXPECT_NEAR(workspace.vars.x[1], -4.31975162, 1e-3);

  workspace.free();
  ldlt_callback_provider.free();
  mco.free();
}

} // namespace sip_examples
