#include "sip_optimal_control/sip_optimal_control.hpp"

#include <Eigen/Core>

#include <gtest/gtest.h>

namespace sip_examples {

namespace {
constexpr double dt = 0.1;

constexpr int state_dim = 2;
constexpr int control_dim = 1;
constexpr int num_stages = 100;
constexpr int c_dim = 1;
constexpr int g_dim = 2;

auto run_solver(::sip::optimal_control::Workspace &workspace) {
  sip::Settings settings{
      .max_kkt_violation = 1e-9,
      .max_merit_slope = 1e-16,
      .mu_update_factor = 0.9,
      .penalty_parameter_increase_factor = 2.0,
      .enable_elastics = true,
      .elastic_var_cost_coeff = 1e6,
  };

  const auto model_callback =
      [&](const ::sip::optimal_control::ModelCallbackInput &mci) -> void {
    auto &mco = workspace.model_callback_output;

    mco.f = 0.0;

    const auto s_0 = Eigen::Vector2d{0.0, 10.0};

    auto dyn_res = Eigen::Map<Eigen::VectorXd>(mco.dyn_res[0], state_dim);

    const auto x_0 =
        Eigen::Map<const Eigen::VectorXd>(mci.states[0], state_dim);

    dyn_res = s_0 - x_0;

    for (int i = 0; i < num_stages; ++i) {
      const auto x =
          Eigen::Map<const Eigen::VectorXd>(mci.states[i], state_dim);
      const auto u =
          Eigen::Map<const Eigen::VectorXd>(mci.controls[i], control_dim);
      const auto next_x =
          Eigen::Map<const Eigen::VectorXd>(mci.states[i + 1], state_dim);
      auto grad_x_f = Eigen::Map<Eigen::VectorXd>(mco.df_dx[i], state_dim);
      auto grad_u_f = Eigen::Map<Eigen::VectorXd>(mco.df_du[i], control_dim);

      auto dyn_res = Eigen::Map<Eigen::VectorXd>(mco.dyn_res[i + 1], state_dim);
      auto ddyn_dx =
          Eigen::Map<Eigen::MatrixXd>(mco.ddyn_dx[i], state_dim, state_dim);
      auto ddyn_du =
          Eigen::Map<Eigen::MatrixXd>(mco.ddyn_du[i], state_dim, control_dim);

      auto c = Eigen::Map<Eigen::VectorXd>(mco.c[i], c_dim);
      auto dc_dx = Eigen::Map<Eigen::MatrixXd>(mco.dc_dx[i], c_dim, state_dim);
      auto dc_du =
          Eigen::Map<Eigen::MatrixXd>(mco.dc_du[i], c_dim, control_dim);

      auto g = Eigen::Map<Eigen::VectorXd>(mco.g[i], g_dim);
      auto dg_dx = Eigen::Map<Eigen::MatrixXd>(mco.dg_dx[i], g_dim, state_dim);
      auto dg_du =
          Eigen::Map<Eigen::MatrixXd>(mco.dg_du[i], g_dim, control_dim);

      auto d2L_dx2 =
          Eigen::Map<Eigen::MatrixXd>(mco.d2L_dx2[i], state_dim, state_dim);
      auto d2L_dxdu =
          Eigen::Map<Eigen::MatrixXd>(mco.d2L_dxdu[i], state_dim, control_dim);
      auto d2L_du2 =
          Eigen::Map<Eigen::MatrixXd>(mco.d2L_du2[i], control_dim, control_dim);

      mco.f +=
          0.5 * x[0] * x[0] + 0.5 * 0.1 * x[1] * x[1] + 0.5 * 0.1 * u[0] * u[0];

      grad_x_f(0) = x[0];
      grad_x_f(1) = 0.1 * x[1];

      grad_u_f(0) = 0.1 * u[0];

      // Double integrator:
      // dx/dt = [1  dt] x + [ dt^2 / 2 ] u
      //         [0  1 ]     [    dt    ]
      ddyn_dx(0, 0) = 1.0;
      ddyn_dx(0, 1) = dt;
      ddyn_dx(1, 0) = 0.0;
      ddyn_dx(1, 1) = 1.0;

      ddyn_du(0, 0) = 0.5 * dt * dt;
      ddyn_du(1, 0) = dt;

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
        Eigen::Map<const Eigen::VectorXd>(mci.states[num_stages], state_dim);

    auto grad_x_f =
        Eigen::Map<Eigen::VectorXd>(mco.df_dx[num_stages], state_dim);

    auto c = Eigen::Map<Eigen::VectorXd>(mco.c[num_stages], c_dim);
    auto dc_dx =
        Eigen::Map<Eigen::MatrixXd>(mco.dc_dx[num_stages], c_dim, state_dim);

    auto g = Eigen::Map<Eigen::VectorXd>(mco.g[num_stages], g_dim);
    auto dg_dx =
        Eigen::Map<Eigen::MatrixXd>(mco.dg_dx[num_stages], g_dim, state_dim);

    auto d2L_dx2 = Eigen::Map<Eigen::MatrixXd>(mco.d2L_dx2[num_stages],
                                               state_dim, state_dim);

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
          .num_stages = num_stages,
          .state_dim = state_dim,
          .control_dim = control_dim,
          .c_dim = c_dim,
          .g_dim = g_dim,
      }};

  const int x_dim = num_stages * (state_dim + control_dim) + state_dim;
  const int y_dim = (c_dim + state_dim) * (num_stages + 1);
  const int z_dim = g_dim * (num_stages + 1);

  // Initialize the variables.
  for (int i = 0; i < x_dim; ++i) {
    workspace.sip_workspace.vars.x[i] = 0.0;
  }

  for (int i = 0; i < z_dim; ++i) {
    workspace.sip_workspace.vars.s[i] = 1.0;
    workspace.sip_workspace.vars.z[i] = 1.0;
    workspace.sip_workspace.vars.e[i] = 0.0;
  }

  for (int i = 0; i < y_dim; ++i) {
    workspace.sip_workspace.vars.y[i] = 0.0;
  }

  return solve(input, settings, workspace);
}
} // namespace

TEST(SimpleConstrainedLQR, Problem1WithMemAssign) {
  ::sip::optimal_control::Workspace workspace;
  constexpr int kWorkspaceSize = ::sip::optimal_control::Workspace::num_bytes(
      state_dim, control_dim, num_stages, c_dim, g_dim);
  std::array<unsigned char, kWorkspaceSize> workspace_bytes;
  workspace.mem_assign(state_dim, control_dim, num_stages, c_dim, g_dim,
                       workspace_bytes.data());

  const auto output = run_solver(workspace);

  ASSERT_EQ(output.exit_status, ::sip::Status::SOLVED);
}

TEST(SimpleConstrainedLQR, Problem1WithReserve) {
  ::sip::optimal_control::Workspace workspace;
  workspace.reserve(state_dim, control_dim, num_stages, c_dim, g_dim);
  constexpr int kWorkspaceSize = ::sip::optimal_control::Workspace::num_bytes(
      state_dim, control_dim, num_stages, c_dim, g_dim);
  std::array<unsigned char, kWorkspaceSize> workspace_bytes;
  workspace.mem_assign(state_dim, control_dim, num_stages, c_dim, g_dim,
                       workspace_bytes.data());

  const auto output = run_solver(workspace);

  ASSERT_EQ(output.exit_status, ::sip::Status::SOLVED);
}

} // namespace sip_examples
