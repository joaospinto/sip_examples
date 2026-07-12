#include "problem_definitions/mock_problems/simple_branched_lqr/problem.hpp"
#include "sip/sip.hpp"
#include "sip_qdldl/sip_qdldl.hpp"

#include <gtest/gtest.h>

namespace sip_examples {
namespace problem = ::sip_examples::problem_definitions::simple_branched_lqr;

TEST(SimpleBranchedLQR, QDLDL) {
  const sip::Settings settings = problem::settings();
  sip::Workspace workspace;
  workspace.reserve(problem::kXDim, problem::kSDim, problem::kYDim,
                    sip::FilterWorkspace::required_capacity(settings));

  sip_qdldl::ModelCallbackOutput mco;
  mco.reserve(problem::kXDim, problem::kSDim, problem::kYDim,
              problem::kUpperHessianLagrangianNnz, problem::kJacobianCNnz,
              problem::kJacobianGNnz, true, true);
  problem::configure_qdldl_sparsity(mco);

  const auto model_callback = [&mco](const sip::ModelCallbackInput &mci) {
    if (!mci.new_x) {
      return;
    }
    problem::evaluate_qdldl(mci, &mco.f, mco.gradient_f, mco.c,
                            mco.upper_hessian_lagrangian.data,
                            mco.jacobian_c.data);
  };

  sip_qdldl::Workspace qdldl_workspace;
  qdldl_workspace.reserve(problem::kKktDim, problem::kQdldlKktNnz,
                          problem::kQdldlKktLNnz);
  const sip_qdldl::Settings qdldl_settings;
  sip_qdldl::CallbackProvider callback_provider(qdldl_settings, mco,
                                                qdldl_workspace);

  const auto factor = [&callback_provider](const double *w, const double r1,
                                           const double *r2, const double *r3) {
    return callback_provider.factor(w, r1, r2, r3);
  };
  const auto solve = [&callback_provider](const double *b, double *v) {
    callback_provider.solve(b, v);
  };
  const auto add_Kx_to_y = [&callback_provider](
                               const double *w, const double r1,
                               const double *r2, const double *r3,
                               const double *x_x, const double *x_y,
                               const double *x_z, double *y_x, double *y_y,
                               double *y_z) {
    callback_provider.add_Kx_to_y(w, r1, r2, r3, x_x, x_y, x_z, y_x, y_y, y_z);
  };
  const auto add_Hx_to_y = [&callback_provider](const double *x, double *y) {
    callback_provider.add_Hx_to_y(x, y);
  };
  const auto add_Cx_to_y = [&callback_provider](const double *x, double *y) {
    callback_provider.add_Cx_to_y(x, y);
  };
  const auto add_CTx_to_y = [&callback_provider](const double *x, double *y) {
    callback_provider.add_CTx_to_y(x, y);
  };
  const auto add_Gx_to_y = [&callback_provider](const double *x, double *y) {
    callback_provider.add_Gx_to_y(x, y);
  };
  const auto add_GTx_to_y = [&callback_provider](const double *x, double *y) {
    callback_provider.add_GTx_to_y(x, y);
  };
  const auto get_f = [&mco]() { return mco.f; };
  const auto get_grad_f = [&mco]() { return mco.gradient_f; };
  const auto get_c = [&mco]() { return mco.c; };
  const auto get_g = [&mco]() { return mco.g; };
  const auto timeout_callback = []() { return false; };

  const sip::Input input{
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
              .x_dim = problem::kXDim,
              .s_dim = problem::kSDim,
              .y_dim = problem::kYDim,
          },
  };

  problem::initialize(workspace);
  const auto output = sip::solve(input, settings, workspace);

  ASSERT_EQ(output.exit_status, sip::Status::SOLVED);
  for (int i = 0; i < problem::kXDim; ++i) {
    EXPECT_NEAR(workspace.vars.x[i], problem::kSolution[i], 1e-8);
  }

  qdldl_workspace.free();
  workspace.free();
  mco.free();
}

} // namespace sip_examples
