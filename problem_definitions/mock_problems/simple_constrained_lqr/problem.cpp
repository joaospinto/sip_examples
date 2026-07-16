#include "problem_definitions/mock_problems/simple_constrained_lqr/problem.hpp"

#include <Eigen/Core>

#include <algorithm>
#include <array>
#include <limits>

namespace sip_examples::problem_definitions::simple_constrained_lqr {

const std::array<int, kNumEdges + 1> kStateDims = [] {
  std::array<int, kNumEdges + 1> values{};
  values.fill(kStateDim);
  return values;
}();
const std::array<int, kNumEdges> kControlDims = [] {
  std::array<int, kNumEdges> values{};
  values.fill(kControlDim);
  return values;
}();
const std::array<int, kNumEdges + 1> kCDims = [] {
  std::array<int, kNumEdges + 1> values{};
  values.fill(kCDim);
  return values;
}();
const std::array<int, kNumEdges + 1> kGDims = [] {
  std::array<int, kNumEdges + 1> values{};
  values.fill(kGDim);
  return values;
}();
const std::array<int, kNumEdges> kEdgeParents = [] {
  std::array<int, kNumEdges> values{};
  for (int edge = 0; edge < kNumEdges; ++edge) {
    values[edge] = edge;
  }
  return values;
}();
const std::array<int, kNumEdges> kEdgeChildren = [] {
  std::array<int, kNumEdges> values{};
  for (int edge = 0; edge < kNumEdges; ++edge) {
    values[edge] = edge + 1;
  }
  return values;
}();

const ::sip::optimal_control::Dimensions kDimensions{
    0, kStateDims.data(), kControlDims.data(), kCDims.data(), kGDims.data()};
const ::sip::optimal_control::Topology kTopology{
    kNumEdges, 0, kEdgeParents.data(), kEdgeChildren.data()};
const std::array<double, kXDim> kLowerBounds = [] {
  std::array<double, kXDim> values{};
  values.fill(-std::numeric_limits<double>::infinity());
  for (int edge = 0; edge < kNumEdges; ++edge) {
    values[edge * (kStateDim + kControlDim) + kStateDim] = -2.0;
  }
  return values;
}();
const std::array<double, kXDim> kUpperBounds = [] {
  std::array<double, kXDim> values{};
  values.fill(std::numeric_limits<double>::infinity());
  for (int edge = 0; edge < kNumEdges; ++edge) {
    values[edge * (kStateDim + kControlDim) + kStateDim] = 2.0;
  }
  return values;
}();

auto settings() -> sip::Settings {
  return sip::Settings{
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
}

auto run_solver(const ::sip::optimal_control::Dimensions &dimensions,
                const ::sip::optimal_control::Topology &topology,
                ::sip::optimal_control::Workspace &workspace) -> sip::Output {
  const auto solver_settings = settings();
  const ::sip::optimal_control::Input input{
      .dimensions = dimensions,
      .topology = topology,
      .model_callback =
          [&](const ::sip::optimal_control::ModelCallbackInput &mci) -> void {
        auto &mco = workspace.model_callback_output;

        mco.f = 0.0;

        const auto s_0 = Eigen::Vector2d{0.0, 10.0};

        auto dyn_res = Eigen::Map<Eigen::VectorXd>(mco.dyn_res[0], kStateDim);

        const auto x_0 =
            Eigen::Map<const Eigen::VectorXd>(mci.states[0], kStateDim);

        dyn_res = s_0 - x_0;

        for (int i = 0; i < kNumEdges; ++i) {
          const auto x =
              Eigen::Map<const Eigen::VectorXd>(mci.states[i], kStateDim);
          const auto u =
              Eigen::Map<const Eigen::VectorXd>(mci.controls[i], kControlDim);
          const auto next_x =
              Eigen::Map<const Eigen::VectorXd>(mci.states[i + 1], kStateDim);
          auto grad_x_f = Eigen::Map<Eigen::VectorXd>(mco.df_dx[i], kStateDim);
          auto grad_u_f =
              Eigen::Map<Eigen::VectorXd>(mco.df_du[i], kControlDim);

          auto dyn_res =
              Eigen::Map<Eigen::VectorXd>(mco.dyn_res[i + 1], kStateDim);
          auto ddyn_dx =
              Eigen::Map<Eigen::MatrixXd>(mco.ddyn_dx[i], kStateDim, kStateDim);
          auto ddyn_du = Eigen::Map<Eigen::MatrixXd>(mco.ddyn_du[i], kStateDim,
                                                     kControlDim);

          auto c = Eigen::Map<Eigen::VectorXd>(mco.c[i], kCDim);
          auto dc_dx =
              Eigen::Map<Eigen::MatrixXd>(mco.dc_dx[i], kCDim, kStateDim);
          auto dc_du =
              Eigen::Map<Eigen::MatrixXd>(mco.dc_du[i], kCDim, kControlDim);

          auto d2L_dx2 =
              Eigen::Map<Eigen::MatrixXd>(mco.d2L_dx2[i], kStateDim, kStateDim);
          auto d2L_dxdu = Eigen::Map<Eigen::MatrixXd>(mco.d2L_dxdu[i],
                                                      kStateDim, kControlDim);
          auto d2L_du2 = Eigen::Map<Eigen::MatrixXd>(mco.d2L_du2[i],
                                                     kControlDim, kControlDim);

          mco.f += 0.5 * x[0] * x[0] + 0.5 * 0.1 * x[1] * x[1] +
                   0.5 * 0.1 * u[0] * u[0];

          grad_x_f(0) = x[0];
          grad_x_f(1) = 0.1 * x[1];
          grad_u_f(0) = 0.1 * u[0];

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

          d2L_dx2(0, 0) = 1.0;
          d2L_dx2(0, 1) = 0.0;
          d2L_dx2(1, 0) = 0.0;
          d2L_dx2(1, 1) = 0.1;
          d2L_dxdu.setZero();
          d2L_du2(0, 0) = 0.1;
        }

        const auto x =
            Eigen::Map<const Eigen::VectorXd>(mci.states[kNumEdges], kStateDim);

        auto grad_x_f =
            Eigen::Map<Eigen::VectorXd>(mco.df_dx[kNumEdges], kStateDim);

        auto c = Eigen::Map<Eigen::VectorXd>(mco.c[kNumEdges], kCDim);
        auto dc_dx =
            Eigen::Map<Eigen::MatrixXd>(mco.dc_dx[kNumEdges], kCDim, kStateDim);

        auto d2L_dx2 = Eigen::Map<Eigen::MatrixXd>(mco.d2L_dx2[kNumEdges],
                                                   kStateDim, kStateDim);

        mco.f += 0.5 * x[0] * x[0] + 0.5 * 0.1 * x[1] * x[1];

        grad_x_f(0) = x[0];
        grad_x_f(1) = 0.1 * x[1];

        c(0) = x[1];
        dc_dx(0, 0) = 0.0;
        dc_dx(0, 1) = 1.0;

        d2L_dx2(0, 0) = 1.0;
        d2L_dx2(0, 1) = 0.0;
        d2L_dx2(1, 0) = 0.0;
        d2L_dx2(1, 1) = 0.1;
      },
      .timeout_callback = []() { return false; },
      .lower_bounds = kLowerBounds.data(),
      .upper_bounds = kUpperBounds.data(),
  };

  const int x_dim = kXDim;
  const int y_dim = (kCDim + kStateDim) * (kNumEdges + 1);
  const int z_dim = kGDim * (kNumEdges + 1);

  for (int i = 0; i < x_dim; ++i) {
    workspace.sip_workspace.vars.x[i] = 0.0;
  }
  for (int i = 0; i < z_dim; ++i) {
    workspace.sip_workspace.vars.s[i] = 1.0;
    workspace.sip_workspace.vars.z[i] = 1.0;
  }
  std::fill_n(workspace.sip_workspace.vars.bound_s, kNumBoundSides, 2.0);
  std::fill_n(workspace.sip_workspace.vars.bound_z, kNumBoundSides,
              solver_settings.barrier.initial_mu / 2.0);
  for (int i = 0; i < y_dim; ++i) {
    workspace.sip_workspace.vars.y[i] = 0.0;
  }

  return ::sip::optimal_control::solve(input, solver_settings, workspace);
}

} // namespace sip_examples::problem_definitions::simple_constrained_lqr
