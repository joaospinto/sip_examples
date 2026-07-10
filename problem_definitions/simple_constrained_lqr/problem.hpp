#pragma once

#include "sip_optimal_control/sip_optimal_control.hpp"

#include <Eigen/Core>

#include <functional>

namespace sip_examples::problem_definitions::simple_constrained_lqr {

constexpr double kDt = 0.1;

constexpr int kStateDim = 2;
constexpr int kControlDim = 1;
constexpr int kNumStages = 100;
constexpr int kCDim = 1;
constexpr int kGDim = 2;

inline auto run_solver(::sip::optimal_control::Workspace &workspace) {
  sip::Settings settings{
      .max_iterations = 300,
      .barrier =
          {
              .mu_update_factor = 0.9,
          },
      .penalty =
          {
              .penalty_parameter_increase_factor = 1.2,
          },
      .termination =
          {
              .max_dual_residual = 1e-9,
              .max_constraint_violation = 1e-9,
              .max_complementarity_gap = 1e-9,
              .max_merit_slope = 1e-16,
          },
      .line_search =
          {
              .max_iterations = 5000,
          },
  };

  const auto model_callback =
      [&](const ::sip::optimal_control::ModelCallbackInput &mci) -> void {
    auto &mco = workspace.model_callback_output;

    mco.f = 0.0;

    const auto s_0 = Eigen::Vector2d{0.0, 10.0};

    auto dyn_res = Eigen::Map<Eigen::VectorXd>(mco.dyn_res[0], kStateDim);

    const auto x_0 =
        Eigen::Map<const Eigen::VectorXd>(mci.states[0], kStateDim);

    dyn_res = s_0 - x_0;

    for (int i = 0; i < kNumStages; ++i) {
      const auto x =
          Eigen::Map<const Eigen::VectorXd>(mci.states[i], kStateDim);
      const auto u =
          Eigen::Map<const Eigen::VectorXd>(mci.controls[i], kControlDim);
      const auto next_x =
          Eigen::Map<const Eigen::VectorXd>(mci.states[i + 1], kStateDim);
      auto grad_x_f = Eigen::Map<Eigen::VectorXd>(mco.df_dx[i], kStateDim);
      auto grad_u_f = Eigen::Map<Eigen::VectorXd>(mco.df_du[i], kControlDim);

      auto dyn_res = Eigen::Map<Eigen::VectorXd>(mco.dyn_res[i + 1], kStateDim);
      auto ddyn_dx =
          Eigen::Map<Eigen::MatrixXd>(mco.ddyn_dx[i], kStateDim, kStateDim);
      auto ddyn_du =
          Eigen::Map<Eigen::MatrixXd>(mco.ddyn_du[i], kStateDim, kControlDim);

      auto c = Eigen::Map<Eigen::VectorXd>(mco.c[i], kCDim);
      auto dc_dx = Eigen::Map<Eigen::MatrixXd>(mco.dc_dx[i], kCDim, kStateDim);
      auto dc_du =
          Eigen::Map<Eigen::MatrixXd>(mco.dc_du[i], kCDim, kControlDim);

      auto g = Eigen::Map<Eigen::VectorXd>(mco.g[i], kGDim);
      auto dg_dx = Eigen::Map<Eigen::MatrixXd>(mco.dg_dx[i], kGDim, kStateDim);
      auto dg_du =
          Eigen::Map<Eigen::MatrixXd>(mco.dg_du[i], kGDim, kControlDim);

      auto d2L_dx2 =
          Eigen::Map<Eigen::MatrixXd>(mco.d2L_dx2[i], kStateDim, kStateDim);
      auto d2L_dxdu =
          Eigen::Map<Eigen::MatrixXd>(mco.d2L_dxdu[i], kStateDim, kControlDim);
      auto d2L_du2 =
          Eigen::Map<Eigen::MatrixXd>(mco.d2L_du2[i], kControlDim, kControlDim);

      mco.f +=
          0.5 * x[0] * x[0] + 0.5 * 0.1 * x[1] * x[1] + 0.5 * 0.1 * u[0] * u[0];

      grad_x_f(0) = x[0];
      grad_x_f(1) = 0.1 * x[1];

      grad_u_f(0) = 0.1 * u[0];

      // Double integrator:
      // dx/dt = [1  kDt] x + [ kDt^2 / 2 ] u
      //         [0  1 ]     [    kDt    ]
      ddyn_dx(0, 0) = 1.0;
      ddyn_dx(0, 1) = kDt;
      ddyn_dx(1, 0) = 0.0;
      ddyn_dx(1, 1) = 1.0;

      ddyn_du(0, 0) = 0.5 * kDt * kDt;
      ddyn_du(1, 0) = kDt;

      dyn_res = ddyn_dx * x + ddyn_du * u - next_x;

      c.setZero();
      dc_dx.setZero();
      dc_du.setZero();

      g(0) = u[0] - 2.0;
      g(1) = -2.0 - u[0];
      dg_dx.setZero();
      dg_du(0, 0) = 1.0;
      dg_du(1, 0) = -1.0;

      d2L_dx2(0, 0) = 1.0;
      d2L_dx2(0, 1) = 0.0;
      d2L_dx2(1, 0) = 0.0;
      d2L_dx2(1, 1) = 0.1;

      d2L_dxdu.setZero();

      d2L_du2(0, 0) = 0.1;
    }

    const auto x =
        Eigen::Map<const Eigen::VectorXd>(mci.states[kNumStages], kStateDim);

    auto grad_x_f =
        Eigen::Map<Eigen::VectorXd>(mco.df_dx[kNumStages], kStateDim);

    auto c = Eigen::Map<Eigen::VectorXd>(mco.c[kNumStages], kCDim);
    auto dc_dx =
        Eigen::Map<Eigen::MatrixXd>(mco.dc_dx[kNumStages], kCDim, kStateDim);

    auto g = Eigen::Map<Eigen::VectorXd>(mco.g[kNumStages], kGDim);
    auto dg_dx =
        Eigen::Map<Eigen::MatrixXd>(mco.dg_dx[kNumStages], kGDim, kStateDim);

    auto d2L_dx2 = Eigen::Map<Eigen::MatrixXd>(mco.d2L_dx2[kNumStages],
                                               kStateDim, kStateDim);

    mco.f += 0.5 * x[0] * x[0] + 0.5 * 0.1 * x[1] * x[1];

    grad_x_f(0) = x[0];
    grad_x_f(1) = 0.1 * x[1];

    c(0) = x[1];
    dc_dx(0, 0) = 0.0;
    dc_dx(0, 1) = 1.0;

    g.setZero();
    dg_dx.setZero();

    d2L_dx2(0, 0) = 1.0;
    d2L_dx2(0, 1) = 0.0;
    d2L_dx2(1, 0) = 0.0;
    d2L_dx2(1, 1) = 0.1;
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
      }};

  const int x_dim = kNumStages * (kStateDim + kControlDim) + kStateDim;
  const int y_dim = (kCDim + kStateDim) * (kNumStages + 1);
  const int z_dim = kGDim * (kNumStages + 1);

  // Initialize the variables.
  for (int i = 0; i < x_dim; ++i) {
    workspace.sip_workspace.vars.x[i] = 0.0;
  }

  for (int i = 0; i < z_dim; ++i) {
    workspace.sip_workspace.vars.s[i] = 1.0;
    workspace.sip_workspace.vars.z[i] = 1.0;
  }

  for (int i = 0; i < y_dim; ++i) {
    workspace.sip_workspace.vars.y[i] = 0.0;
  }

  return ::sip::optimal_control::solve(input, settings, workspace);
}

} // namespace sip_examples::problem_definitions::simple_constrained_lqr
