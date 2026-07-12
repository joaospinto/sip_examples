#include "problem_definitions/mock_problems/simple_nlp/problem.hpp"
#include "sip/sip.hpp"
#include "sip_qdldl/sip_qdldl.hpp"

#include <gtest/gtest.h>

namespace sip_examples {
namespace problem = ::sip_examples::problem_definitions::simple_nlp;

TEST(SimpleNLP, Problem1) {
  sip::Settings settings = problem::settings();
  sip::Workspace workspace;

  workspace.reserve(problem::kXDim, problem::kSDim, problem::kYDim);

  sip_qdldl::ModelCallbackOutput mco;
  mco.reserve(problem::kXDim, problem::kSDim, problem::kYDim,
              problem::kUpperHessianLagrangianNnz, problem::kJacobianCNnz,
              problem::kJacobianGNnz, true, true);
  problem::configure_qdldl_sparsity(mco);

  auto model_callback = [&mco](const sip::ModelCallbackInput &mci) -> void {
    if (!mci.new_x) {
      return;
    }
    problem::evaluate_qdldl(mci, &mco.f, mco.gradient_f, mco.g,
                            mco.upper_hessian_lagrangian.data,
                            mco.jacobian_g.data);
  };

  sip_qdldl::Workspace sip_qdldl_workspace;
  sip_qdldl_workspace.reserve(problem::kKktDim, problem::kQdldlKktNnz,
                              problem::kQdldlKktLNnz);

  const sip_qdldl::Settings sip_qdldl_settings{
      .permute_kkt_system = true,
      .kkt_pinv = problem::kKktPinv.data(),
  };

  auto callback_provider =
      sip_qdldl::CallbackProvider(sip_qdldl_settings, mco, sip_qdldl_workspace);

  const auto timeout_callback = []() { return false; };

  const auto ldlt_factor =
      [&callback_provider](const double *w, const double r1, const double *r2,
                           const double *r3) -> bool {
    return callback_provider.factor(w, r1, r2, r3);
  };

  const auto ldlt_solve = [&callback_provider](const double *b,
                                               double *v) -> void {
    return callback_provider.solve(b, v);
  };

  const auto add_Kx_to_y =
      [&callback_provider](const double *w, const double r1, const double *r2,
                           const double *r3, const double *x_x,
                           const double *x_y, const double *x_z, double *y_x,
                           double *y_y, double *y_z) -> void {
    return callback_provider.add_Kx_to_y(w, r1, r2, r3, x_x, x_y, x_z, y_x, y_y,
                                         y_z);
  };

  const auto add_Hx_to_y = [&callback_provider](const double *x,
                                                double *y) -> void {
    return callback_provider.add_Hx_to_y(x, y);
  };

  const auto add_Cx_to_y = [&callback_provider](const double *x,
                                                double *y) -> void {
    return callback_provider.add_Cx_to_y(x, y);
  };

  const auto add_CTx_to_y = [&callback_provider](const double *x,
                                                 double *y) -> void {
    return callback_provider.add_CTx_to_y(x, y);
  };

  const auto add_Gx_to_y = [&callback_provider](const double *x,
                                                double *y) -> void {
    return callback_provider.add_Gx_to_y(x, y);
  };

  const auto add_GTx_to_y = [&callback_provider](const double *x,
                                                 double *y) -> void {
    return callback_provider.add_GTx_to_y(x, y);
  };

  const auto get_f = [&mco]() -> double { return mco.f; };

  const auto get_grad_f = [&mco]() -> double * { return mco.gradient_f; };

  const auto get_c = [&mco]() -> double * { return mco.c; };

  const auto get_g = [&mco]() -> double * { return mco.g; };

  sip::Input input{
      .factor = std::cref(ldlt_factor),
      .solve = std::cref(ldlt_solve),
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
              .x_dim = problem::kXDim,
              .s_dim = problem::kSDim,
              .y_dim = problem::kYDim,
          },
  };

  problem::initialize(workspace);

  const auto output = solve(input, settings, workspace);

  EXPECT_EQ(output.exit_status, sip::Status::SOLVED);

  EXPECT_NEAR(workspace.vars.x[0], -1.15747396, 1e-3);
  EXPECT_NEAR(workspace.vars.x[1], -4.31975162, 1e-3);

  sip_qdldl_workspace.free();
  workspace.free();
  mco.free();
}

} // namespace sip_examples
