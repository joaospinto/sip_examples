#include "problem_definitions/mock_problems/simple_constrained_lqr/problem.hpp"
#include "problem_definitions/unit_residual_scaling.hpp"

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
const std::array<int, kNumEdges + 1> kNodeCDims = [] {
  std::array<int, kNumEdges + 1> values{};
  values.back() = kCDim;
  return values;
}();
const std::array<int, kNumEdges + 1> kNodeGDims{};
const std::array<int, kNumEdges> kEdgeCDims{};
const std::array<int, kNumEdges> kEdgeGDims{};
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

const ::sip::optimal_control::Dimensions kDimensions{0,
                                                     kStateDims.data(),
                                                     kControlDims.data(),
                                                     kNodeCDims.data(),
                                                     kNodeGDims.data(),
                                                     kEdgeCDims.data(),
                                                     kEdgeGDims.data()};
const ::sip::optimal_control::Topology kTopology{
    kNumEdges, 0, kEdgeParents.data(), kEdgeChildren.data()};
const std::array<double, kStateDim> kInitialState = {0.0, 10.0};
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
  const UnitResidualScaling residual_scaling(
      dimensions.get_x_dim(topology.num_edges),
      dimensions.get_z_dim(topology.num_edges),
      dimensions.get_y_dim(topology.num_edges));
  const ::sip::optimal_control::Input input{
      .dimensions = dimensions,
      .topology = topology,
      .initial_state = kInitialState.data(),
      .model_callback =
          [&](const ::sip::optimal_control::ModelCallbackInput &mci,
              ::sip::optimal_control::ModelCallbackOutput &mco) -> void {
        for (int node = 0; node <= kNumEdges; ++node) {
          auto &node_output = mco.nodes[node];
          node_output.f = 0.0;
          Eigen::Map<Eigen::VectorXd>(node_output.df_dx, kStateDim).setZero();
          Eigen::Map<Eigen::MatrixXd>(node_output.d2L_dx2, kStateDim, kStateDim)
              .setZero();
        }
        for (int i = 0; i < kNumEdges; ++i) {
          const auto &edge_input = mci.edges[i];
          auto &edge_output = mco.edges[i];
          const auto x = Eigen::Map<const Eigen::VectorXd>(
              edge_input.parent_state, kStateDim);
          const auto u = Eigen::Map<const Eigen::VectorXd>(edge_input.control,
                                                           kControlDim);
          const auto next_x = Eigen::Map<const Eigen::VectorXd>(
              edge_input.child_state, kStateDim);
          auto grad_x_f =
              Eigen::Map<Eigen::VectorXd>(edge_output.df_dx, kStateDim);
          auto grad_u_f =
              Eigen::Map<Eigen::VectorXd>(edge_output.df_du, kControlDim);
          auto dyn_res =
              Eigen::Map<Eigen::VectorXd>(edge_output.dyn_res, kStateDim);
          auto ddyn_dx = Eigen::Map<Eigen::MatrixXd>(edge_output.ddyn_dx,
                                                     kStateDim, kStateDim);
          auto ddyn_du = Eigen::Map<Eigen::MatrixXd>(edge_output.ddyn_du,
                                                     kStateDim, kControlDim);
          auto d2L_dx2 = Eigen::Map<Eigen::MatrixXd>(edge_output.d2L_dx2,
                                                     kStateDim, kStateDim);
          auto d2L_dxdu = Eigen::Map<Eigen::MatrixXd>(edge_output.d2L_dxdu,
                                                      kStateDim, kControlDim);
          auto d2L_du2 = Eigen::Map<Eigen::MatrixXd>(edge_output.d2L_du2,
                                                     kControlDim, kControlDim);

          edge_output.f = 0.5 * x[0] * x[0] + 0.5 * 0.1 * x[1] * x[1] +
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

          d2L_dx2(0, 0) = 1.0;
          d2L_dx2(0, 1) = 0.0;
          d2L_dx2(1, 0) = 0.0;
          d2L_dx2(1, 1) = 0.1;
          d2L_dxdu.setZero();
          d2L_du2(0, 0) = 0.1;
        }

        const auto x = Eigen::Map<const Eigen::VectorXd>(
            mci.nodes[kNumEdges].state, kStateDim);
        auto &terminal = mco.nodes[kNumEdges];
        auto grad_x_f = Eigen::Map<Eigen::VectorXd>(terminal.df_dx, kStateDim);
        auto c = Eigen::Map<Eigen::VectorXd>(terminal.c, kCDim);
        auto dc_dx =
            Eigen::Map<Eigen::MatrixXd>(terminal.dc_dx, kCDim, kStateDim);
        auto d2L_dx2 =
            Eigen::Map<Eigen::MatrixXd>(terminal.d2L_dx2, kStateDim, kStateDim);

        terminal.f = 0.5 * x[0] * x[0] + 0.5 * 0.1 * x[1] * x[1];

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
      .residual_scaling = residual_scaling.get(),
  };

  const int x_dim = kXDim;
  const int y_dim = dimensions.get_y_dim(kNumEdges);
  const int z_dim = dimensions.get_z_dim(kNumEdges);

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
