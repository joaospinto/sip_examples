#include "problem_definitions/mock_problems/simple_nlp/problem.hpp"
#include "problem_definitions/unit_residual_scaling.hpp"
#include "sip/sip.hpp"
#include "sip_qdldl/sip_qdldl.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <limits>
#include <string>

namespace sip_examples {
namespace problem = ::sip_examples::problem_definitions::simple_nlp;

struct DerivativeRequestCase {
  sip::Mode mode;
  bool debug;
  bool eager;
  bool skip_line_search;
  bool incorrect_gradient;
  bool direction_only;
};

class SimpleNLP : public ::testing::TestWithParam<DerivativeRequestCase> {};

TEST_P(SimpleNLP, DerivativeRequests) {
  const auto config = GetParam();
  sip::Settings settings = problem::settings();
  settings.mode = config.mode;
  settings.logging.print_logs = config.debug;
  settings.logging.print_line_search_logs = config.debug;
  settings.logging.print_search_direction_logs = config.debug;
  settings.logging.print_derivative_check_logs = config.debug;
  settings.logging.only_check_search_direction_slope = config.direction_only;
  settings.line_search.skip_line_search = config.skip_line_search;
  if (config.incorrect_gradient || config.skip_line_search) {
    settings.max_iterations = 1;
  }
  sip::Workspace workspace;

  workspace.reserve(problem::kXDim, problem::kSDim, problem::kYDim, 0,
                    settings);

  sip_qdldl::ModelCallbackOutput mco;
  mco.reserve(problem::kXDim, problem::kSDim, problem::kYDim,
              problem::kUpperHessianLagrangianNnz, problem::kJacobianCNnz,
              problem::kJacobianGNnz, true, true);
  problem::configure_qdldl_sparsity(mco);

  int value_requests = 0;
  int derivative_requests = 0;
  int deferred_requests_at_same_x = 0;
  bool derivatives_requested = false;
  const auto require_derivatives = [&]() {
    EXPECT_TRUE(derivatives_requested);
  };
  auto model_callback = [&](const sip::ModelCallbackInput &mci) -> void {
    if (mci.need_derivatives) {
      ++derivative_requests;
      if (!mci.new_x && !derivatives_requested) {
        ++deferred_requests_at_same_x;
      }
    } else {
      ++value_requests;
    }
    derivatives_requested = mci.need_derivatives;
    if (config.eager || mci.need_derivatives) {
      auto evaluation_input = mci;
      evaluation_input.need_derivatives = true;
      problem::evaluate_qdldl(evaluation_input, &mco.f, mco.gradient_f, mco.g,
                              mco.upper_hessian_lagrangian.data,
                              mco.jacobian_g.data);
      if (config.incorrect_gradient) {
        for (int i = 0; i < problem::kXDim; ++i) {
          mco.gradient_f[i] = -mco.gradient_f[i];
        }
      }
    } else {
      mco.f = mci.x[1] * (5.0 + mci.x[0]);
      mco.g[0] = 5.0 - mci.x[0] * mci.x[1];
      mco.g[1] = mci.x[0] * mci.x[0] + mci.x[1] * mci.x[1] - 20.0;
      // Make stale derivative reads fail even if the model point is unchanged.
      const double nan = std::numeric_limits<double>::quiet_NaN();
      std::fill_n(mco.gradient_f, problem::kXDim, nan);
      std::fill_n(mco.upper_hessian_lagrangian.data,
                  problem::kUpperHessianLagrangianNnz, nan);
      std::fill_n(mco.jacobian_g.data, problem::kJacobianGNnz, nan);
    }
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

  const auto ldlt_factor = [&callback_provider, &require_derivatives](
                               const double *w, const double *r1,
                               const double *r2, const double *r3) -> bool {
    require_derivatives();
    return callback_provider.factor(w, r1, r2, r3);
  };

  const auto ldlt_solve = [&callback_provider](const double *b,
                                               double *v) -> void {
    return callback_provider.solve(b, v);
  };

  const auto add_Kx_to_y =
      [&callback_provider, &require_derivatives](
          const double *w, const double *r1, const double *r2, const double *r3,
          const double *x_x, const double *x_y, const double *x_z, double *y_x,
          double *y_y, double *y_z) -> void {
    require_derivatives();
    return callback_provider.add_Kx_to_y(w, r1, r2, r3, x_x, x_y, x_z, y_x, y_y,
                                         y_z);
  };

  const auto add_Hx_to_y = [&callback_provider, &require_derivatives](
                               const double *x, double *y) -> void {
    require_derivatives();
    return callback_provider.add_Hx_to_y(x, y);
  };

  const auto add_Cx_to_y = [&callback_provider, &require_derivatives](
                               const double *x, double *y) -> void {
    require_derivatives();
    return callback_provider.add_Cx_to_y(x, y);
  };

  const auto add_CTx_to_y = [&callback_provider, &require_derivatives](
                                const double *x, double *y) -> void {
    require_derivatives();
    return callback_provider.add_CTx_to_y(x, y);
  };

  const auto add_Gx_to_y = [&callback_provider, &require_derivatives](
                               const double *x, double *y) -> void {
    require_derivatives();
    return callback_provider.add_Gx_to_y(x, y);
  };

  const auto add_GTx_to_y = [&callback_provider, &require_derivatives](
                                const double *x, double *y) -> void {
    require_derivatives();
    return callback_provider.add_GTx_to_y(x, y);
  };

  const auto get_f = [&mco]() -> double { return mco.f; };

  const auto get_grad_f = [&]() -> double * {
    require_derivatives();
    return mco.gradient_f;
  };

  const auto get_c = [&mco]() -> double * { return mco.c; };

  const auto get_g = [&mco]() -> double * { return mco.g; };

  const problem_definitions::UnitResidualScaling residual_scaling(
      problem::kXDim, problem::kSDim, problem::kYDim);

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
      .residual_scaling = residual_scaling.get(),
      .dimensions =
          {
              .x_dim = problem::kXDim,
              .s_dim = problem::kSDim,
              .y_dim = problem::kYDim,
          },
  };

  problem::initialize(workspace);

  if (config.incorrect_gradient) {
    ::testing::internal::CaptureStdout();
  }
  const auto output = solve(input, settings, workspace);
  if (config.incorrect_gradient) {
    const auto logs = ::testing::internal::GetCapturedStdout();
    EXPECT_NE(logs.find("f/c/g"), std::string::npos);
  }

  EXPECT_GT(value_requests, 0);
  EXPECT_GT(derivative_requests, 0);
  EXPECT_GT(deferred_requests_at_same_x, 0);
  EXPECT_TRUE(derivatives_requested);
  if (settings.max_iterations == 1) {
    EXPECT_EQ(output.exit_status, sip::Status::ITERATION_LIMIT);
  } else {
    EXPECT_EQ(output.exit_status, sip::Status::SOLVED);
    EXPECT_NEAR(workspace.vars.x[0], -1.15747396, 1e-3);
    EXPECT_NEAR(workspace.vars.x[1], -4.31975162, 1e-3);
  }

  sip_qdldl_workspace.free();
  workspace.free();
  mco.free();
}

INSTANTIATE_TEST_SUITE_P(
    ModelCallbacks, SimpleNLP,
    ::testing::Values(DerivativeRequestCase{sip::Mode::REGULARIZED_IPM, false,
                                            false, false, false, false},
                      DerivativeRequestCase{sip::Mode::REGULARIZED_IPM, true,
                                            false, false, false, false},
                      DerivativeRequestCase{sip::Mode::REGULARIZED_IPM, true,
                                            true, false, false, false},
                      DerivativeRequestCase{sip::Mode::PRIMAL_PROXIMAL_IPM,
                                            true, false, false, false, false},
                      DerivativeRequestCase{sip::Mode::PRIMAL_DUAL_PROXIMAL_IPM,
                                            true, false, false, false, false},
                      DerivativeRequestCase{sip::Mode::REGULARIZED_IPM, false,
                                            false, true, false, false},
                      DerivativeRequestCase{sip::Mode::PRIMAL_DUAL_PROXIMAL_IPM,
                                            false, false, true, false, false},
                      DerivativeRequestCase{sip::Mode::REGULARIZED_IPM, true,
                                            false, false, true, false},
                      DerivativeRequestCase{sip::Mode::REGULARIZED_IPM, true,
                                            true, false, true, false},
                      DerivativeRequestCase{sip::Mode::PRIMAL_DUAL_PROXIMAL_IPM,
                                            true, false, false, true, true}));

} // namespace sip_examples
