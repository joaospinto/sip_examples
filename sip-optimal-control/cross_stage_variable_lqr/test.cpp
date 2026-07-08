#include "sip_optimal_control/sip_optimal_control.hpp"

#include <Eigen/Core>

#include <array>
#include <gtest/gtest.h>

namespace sip_examples {
namespace {

constexpr int kNumStages = 1;
constexpr int kStateDim = 1;
constexpr int kControlDim = 1;
constexpr int kCDim = 1;
constexpr int kGDim = 0;
constexpr int kThetaDim = 1;

auto run_solver(::sip::optimal_control::Workspace &workspace) {
  sip::Settings settings{
      .max_iterations = 100,
      .barrier =
          {
              .initial_mu = 1e-3,
              .mu_update_factor = 0.5,
          },
      .penalty =
          {
              .initial_penalty_parameter = 1e3,
          },
      .termination =
          {
              .max_dual_residual = 1e-9,
              .max_constraint_violation = 1e-9,
              .max_complementarity_gap = 1e-9,
          },
      .line_search =
          {
              .max_iterations = 1000,
          },
      .logging =
          {
              .print_logs = false,
              .print_line_search_logs = false,
              .print_search_direction_logs = false,
              .print_derivative_check_logs = false,
          },
  };

  const auto model_callback =
      [&](const ::sip::optimal_control::ModelCallbackInput &mci) -> void {
    auto &mco = workspace.model_callback_output;

    const double x0 = mci.states[0][0];
    const double u0 = mci.controls[0][0];
    const double x1 = mci.states[1][0];
    const double theta = mci.theta[0];

    mco.f = 0.5 * x0 * x0 + 0.5 * u0 * u0 + 0.5 * x1 * x1 + 0.5 * theta * theta;

    mco.df_dx[0][0] = x0;
    mco.df_du[0][0] = u0;
    mco.df_dx[1][0] = x1;
    mco.df_dtheta[0] = theta;

    mco.dyn_res[0][0] = -x0;
    mco.dyn_res[1][0] = x0 + u0 + theta - x1;
    mco.ddyn_dx[0][0] = 1.0;
    mco.ddyn_du[0][0] = 1.0;
    mco.ddyn_dtheta[0][0] = 1.0;

    mco.c[0][0] = 0.0;
    mco.c[1][0] = x1 - 1.0;
    mco.dc_dx[0][0] = 0.0;
    mco.dc_du[0][0] = 0.0;
    mco.dc_dtheta[0][0] = 0.0;
    mco.dc_dx[1][0] = 1.0;
    mco.dc_dtheta[1][0] = 0.0;

    mco.d2L_dx2[0][0] = 1.0;
    mco.d2L_dxdu[0][0] = 0.0;
    mco.d2L_du2[0][0] = 1.0;
    mco.d2L_dxdtheta[0][0] = 0.0;
    mco.d2L_dudtheta[0][0] = 0.0;
    mco.d2L_dx2[1][0] = 1.0;
    mco.d2L_dxdtheta[1][0] = 0.0;
    mco.d2L_dtheta2[0] = 1.0;
  };

  const auto timeout_callback = []() -> bool { return false; };

  const auto input = ::sip::optimal_control::Input{
      .model_callback = std::cref(model_callback),
      .timeout_callback = std::cref(timeout_callback),
      .dimensions = {
          .num_stages = kNumStages,
          .state_dim = kStateDim,
          .control_dim = kControlDim,
          .c_dim = kCDim,
          .g_dim = kGDim,
          .theta_dim = kThetaDim,
      }};

  for (int i = 0; i < input.dimensions.get_x_dim(); ++i) {
    workspace.sip_workspace.vars.x[i] = 0.0;
  }
  for (int i = 0; i < input.dimensions.get_y_dim(); ++i) {
    workspace.sip_workspace.vars.y[i] = 0.0;
  }

  return solve(input, settings, workspace);
}

void expect_solution(const ::sip::optimal_control::Workspace &workspace) {
  constexpr int kThetaOffset =
      kNumStages * (kStateDim + kControlDim) + kStateDim;
  EXPECT_NEAR(workspace.sip_workspace.vars.x[0], 0.0, 1e-6);
  EXPECT_NEAR(workspace.sip_workspace.vars.x[1], 0.5, 1e-6);
  EXPECT_NEAR(workspace.sip_workspace.vars.x[2], 1.0, 1e-6);
  EXPECT_NEAR(workspace.sip_workspace.vars.x[kThetaOffset], 0.5, 1e-6);
}

} // namespace

TEST(CrossStageVariableLQR, WithMemAssign) {
  ::sip::optimal_control::Workspace workspace;
  constexpr int kWorkspaceSize = ::sip::optimal_control::Workspace::num_bytes(
      kStateDim, kControlDim, kNumStages, kCDim, kGDim, kThetaDim);
  std::array<unsigned char, kWorkspaceSize> workspace_bytes;
  workspace.mem_assign(kStateDim, kControlDim, kNumStages, kCDim, kGDim,
                       workspace_bytes.data(), kThetaDim);

  const auto output = run_solver(workspace);

  ASSERT_EQ(output.exit_status, ::sip::Status::SOLVED);
  expect_solution(workspace);
}

TEST(CrossStageVariableLQR, WithReserve) {
  ::sip::optimal_control::Workspace workspace;
  workspace.reserve(kStateDim, kControlDim, kNumStages, kCDim, kGDim,
                    kThetaDim);

  const auto output = run_solver(workspace);

  ASSERT_EQ(output.exit_status, ::sip::Status::SOLVED);
  expect_solution(workspace);

  workspace.free(kNumStages);
}

} // namespace sip_examples
