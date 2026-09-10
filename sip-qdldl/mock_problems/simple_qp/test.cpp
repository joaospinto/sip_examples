#include "problem_definitions/mock_problems/simple_qp/problem.hpp"
#include "sip/sip.hpp"
#include "sip_qdldl/sip_qdldl.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <functional>
#include <gtest/gtest.h>
#include <vector>

namespace sip_examples {
namespace problem = ::sip_examples::problem_definitions::simple_qp;

namespace {

void solve_problem(const sip::Mode mode, const bool use_mem_assign,
                   const double objective_scale = 1.0,
                   const double dual_s_max = 0.0,
                   const int timeout_after_callbacks = 0,
                   const bool skip_line_search = false) {
  sip::Settings settings = problem::settings();
  settings.mode = mode;
  settings.logging = {.print_logs = false,
                      .print_line_search_logs = false,
                      .print_search_direction_logs = false,
                      .print_derivative_check_logs = false};
  settings.termination.dual_residual_s_max = dual_s_max;
  settings.line_search.skip_line_search = skip_line_search;
  const std::array<double, 2> variable_scale =
      objective_scale == 1.0 ? std::array{1.0, 1.0} : std::array{4.0, 0.5};
  const double equality_scale = objective_scale == 1.0 ? 1.0 : 3.0;
  std::array<double, 2> lower, upper;
  for (int i = 0; i < 2; ++i) {
    lower[i] = problem::kLowerBounds[i] * variable_scale[i];
    upper[i] = problem::kUpperBounds[i] * variable_scale[i];
  }
  int callbacks = 0;
  int derivative_callbacks = 0;

  sip_qdldl::ModelCallbackOutput mco;
  mco.reserve(problem::kXDim, problem::kSDim, problem::kYDim,
              problem::kUpperHessianLagrangianNnz, problem::kJacobianCNnz,
              problem::kJacobianGNnz, true, true);
  problem::configure_qdldl_sparsity(mco);

  auto model_callback = [&](const sip::ModelCallbackInput &mci) -> void {
    ++callbacks;
    derivative_callbacks += mci.need_derivatives;
    if (!mci.new_x && !mci.need_derivatives) {
      return;
    }
    std::array<double, 2> x{mci.x[0] / variable_scale[0],
                            mci.x[1] / variable_scale[1]};
    auto original = mci;
    original.x = x.data();
    problem::evaluate(original, &mco.f, mco.gradient_f, mco.c,
                      mco.upper_hessian_lagrangian.data, mco.jacobian_c.data);
    mco.f *= objective_scale;
    mco.c[0] *= equality_scale;
    if (mci.need_derivatives) {
      for (int i = 0; i < 2; ++i) {
        mco.gradient_f[i] *= objective_scale / variable_scale[i];
        mco.jacobian_c.data[i] *= equality_scale / variable_scale[i];
      }
      mco.upper_hessian_lagrangian.data[0] *=
          objective_scale / (variable_scale[0] * variable_scale[0]);
      mco.upper_hessian_lagrangian.data[1] *=
          objective_scale / (variable_scale[0] * variable_scale[1]);
      mco.upper_hessian_lagrangian.data[2] *=
          objective_scale / (variable_scale[1] * variable_scale[1]);
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

  const auto timeout_callback = [&]() {
    return callbacks >= timeout_after_callbacks;
  };

  const auto ldlt_factor =
      [&callback_provider](const double *w, const double *r1, const double *r2,
                           const double *r3) -> bool {
    return callback_provider.factor(w, r1, r2, r3);
  };

  const auto ldlt_solve = [&callback_provider](const double *b,
                                               double *v) -> void {
    return callback_provider.solve(b, v);
  };

  const auto add_Kx_to_y =
      [&callback_provider](const double *w, const double *r1, const double *r2,
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
      .timeout_callback = timeout_after_callbacks > 0
                              ? sip::Input::TimeoutCallback(timeout_callback)
                              : sip::Input::TimeoutCallback{},
      .lower_bounds = lower.data(),
      .upper_bounds = upper.data(),
      .scaling = {.objective = objective_scale,
                  .variable = variable_scale.data(),
                  .equality = &equality_scale},
      .dimensions =
          {
              .x_dim = problem::kXDim,
              .s_dim = problem::kSDim,
              .y_dim = problem::kYDim,
          },
  };

  const int num_bound_sides = input.num_bound_sides();
  sip::Workspace workspace;
  std::vector<unsigned char> workspace_memory;
  if (use_mem_assign) {
    workspace_memory.resize(
        sip::Workspace::num_bytes(problem::kXDim, problem::kSDim,
                                  problem::kYDim, num_bound_sides, settings));
    EXPECT_EQ(workspace.mem_assign(problem::kXDim, problem::kSDim,
                                   problem::kYDim, num_bound_sides, settings,
                                   workspace_memory.data()),
              workspace_memory.size());
  } else {
    workspace.reserve(problem::kXDim, problem::kSDim, problem::kYDim,
                      num_bound_sides, settings);
  }
  problem::initialize(workspace, num_bound_sides);
  for (int side = 0; side < num_bound_sides; ++side) {
    const double v = variable_scale[side / 2];
    workspace.vars.bound_s[side] *= v;
    workspace.vars.bound_z[side] *= objective_scale / v;
  }

  const auto output = solve(input, settings, workspace);

  const double x0 = workspace.vars.x[0] / variable_scale[0];
  const double x1 = workspace.vars.x[1] / variable_scale[1];
  const double y = workspace.vars.y[0] * equality_scale / objective_scale;
  std::array<double, 4> bound_z;
  double multiplier_sum = std::abs(y);
  double complementarity = 0;
  for (int side = 0; side < num_bound_sides; ++side) {
    bound_z[side] = workspace.vars.bound_z[side] * variable_scale[side / 2] /
                    objective_scale;
    multiplier_sum += std::abs(bound_z[side]);
    complementarity = std::max(
        complementarity, workspace.vars.bound_s[side] *
                             workspace.vars.bound_z[side] / objective_scale);
  }
  const double raw_dual =
      std::max(std::abs(4 * x0 + x1 + 1 + y - bound_z[0] + bound_z[1]),
               std::abs(x0 + 2 * x1 + 1 + y - bound_z[2] + bound_z[3]));
  EXPECT_NEAR(output.max_dual_violation, raw_dual, 2e-14);
  const double primal =
      std::max({std::abs(x0 + x1 - 1), -x0, x0 - 0.7, -x1, x1 - 0.7});
  EXPECT_NEAR(output.max_primal_violation, primal, 2e-14);
  if (timeout_after_callbacks > 0) {
    EXPECT_EQ(output.exit_status, sip::Status::TIMEOUT);
    EXPECT_DOUBLE_EQ(x0, 0);
    EXPECT_DOUBLE_EQ(x1, 0);
    EXPECT_EQ(callbacks, timeout_after_callbacks);
    if (timeout_after_callbacks <= 2) {
      EXPECT_EQ(derivative_callbacks, 1);
    }
  } else {
    EXPECT_EQ(output.exit_status, sip::Status::SOLVED);
    EXPECT_NEAR(x0, 0.3, 1e-6);
    EXPECT_NEAR(x1, 0.7, 1e-6);
    const double normalization =
        dual_s_max == 0.0 ? 1.0
                          : std::max(1.0, multiplier_sum / 5 / dual_s_max);
    EXPECT_LT(raw_dual / normalization,
              settings.termination.max_dual_residual + 2e-14);
    EXPECT_LT(complementarity, settings.termination.max_complementarity_gap);
  }

  sip_qdldl_workspace.free();
  if (!use_mem_assign) {
    workspace.free();
  }
  mco.free();
}

} // namespace

TEST(SimpleQP, FromOSQPRepo) {
  constexpr std::array modes{
      sip::Mode::REGULARIZED_IPM,
      sip::Mode::PRIMAL_PROXIMAL_IPM,
      sip::Mode::PRIMAL_DUAL_PROXIMAL_IPM,
  };
  for (const sip::Mode mode : modes) {
    for (const bool use_mem_assign : {false, true}) {
      SCOPED_TRACE(static_cast<int>(mode));
      SCOPED_TRACE(use_mem_assign);
      solve_problem(mode, use_mem_assign);
    }
  }
}

TEST(SimpleQP, ScalingPreservesSolutionAndOriginalCoordinateResiduals) {
  for (const auto mode :
       {sip::Mode::REGULARIZED_IPM, sip::Mode::PRIMAL_PROXIMAL_IPM,
        sip::Mode::PRIMAL_DUAL_PROXIMAL_IPM}) {
    for (const double objective_scale : {0.1, 9.0}) {
      for (const double dual_s_max : {0.0, 0.01}) {
        SCOPED_TRACE(static_cast<int>(mode));
        SCOPED_TRACE(objective_scale);
        SCOPED_TRACE(dual_s_max);
        solve_problem(mode, true, objective_scale, dual_s_max);
      }
    }
  }
}

TEST(SimpleQP, TimeoutPreservesAcceptedPointAndOriginalCoordinateResiduals) {
  for (const auto mode :
       {sip::Mode::REGULARIZED_IPM, sip::Mode::PRIMAL_DUAL_PROXIMAL_IPM}) {
    for (const bool skip_line_search : {false, true}) {
      for (int callbacks = 1; callbacks <= 3; ++callbacks) {
        SCOPED_TRACE(static_cast<int>(mode));
        SCOPED_TRACE(skip_line_search);
        SCOPED_TRACE(callbacks);
        solve_problem(mode, false, 9.0, 0.0, callbacks, skip_line_search);
      }
    }
  }
}

} // namespace sip_examples
