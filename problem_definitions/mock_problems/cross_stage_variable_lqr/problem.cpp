#include "problem_definitions/mock_problems/cross_stage_variable_lqr/problem.hpp"
#include "problem_definitions/unit_residual_scaling.hpp"

#include <array>

namespace sip_examples::problem_definitions::cross_stage_variable_lqr {

const std::array<int, 2> kStateDims = {1, 1};
const std::array<int, 1> kControlDims = {1};
const std::array<int, 2> kNodeCDims = {0, 1};
const std::array<int, 2> kNodeGDims = {0, 0};
const std::array<int, 1> kEdgeCDims = {0};
const std::array<int, 1> kEdgeGDims = {0};
const std::array<int, 1> kEdgeParents = {0};
const std::array<int, 1> kEdgeChildren = {1};

const ::sip::optimal_control::Dimensions kDimensions{
    kThetaDim,         kStateDims.data(), kControlDims.data(),
    kNodeCDims.data(), kNodeGDims.data(), kEdgeCDims.data(),
    kEdgeGDims.data()};
const ::sip::optimal_control::Topology kTopology{
    kNumEdges, 0, kEdgeParents.data(), kEdgeChildren.data()};
const std::array<double, kStateDim> kInitialState = {0.0};

auto settings() -> sip::Settings {
  return sip::Settings{
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
        const double x0 = mci.nodes[0].state[0];
        const double u0 = mci.edges[0].control[0];
        const double x1 = mci.nodes[1].state[0];
        const double theta = mci.theta[0];

        auto &root = mco.nodes[0];
        root.f = 0.0;
        root.df_dx[0] = 0.0;
        root.df_dtheta[0] = 0.0;
        root.d2L_dx2[0] = 0.0;
        root.d2L_dxdtheta[0] = 0.0;
        root.d2L_dtheta2[0] = 0.0;

        auto &edge = mco.edges[0];
        edge.f = 0.5 * x0 * x0 + 0.5 * u0 * u0 + 0.5 * theta * theta;
        edge.df_dx[0] = x0;
        edge.df_du[0] = u0;
        edge.df_dtheta[0] = theta;
        edge.dyn_res[0] = x0 + u0 + theta - x1;
        edge.ddyn_dx[0] = 1.0;
        edge.ddyn_du[0] = 1.0;
        edge.ddyn_dtheta[0] = 1.0;
        edge.d2L_dx2[0] = 1.0;
        edge.d2L_dxdu[0] = 0.0;
        edge.d2L_du2[0] = 1.0;
        edge.d2L_dxdtheta[0] = 0.0;
        edge.d2L_dudtheta[0] = 0.0;
        edge.d2L_dtheta2[0] = 1.0;

        auto &terminal = mco.nodes[1];
        terminal.f = 0.5 * x1 * x1;
        terminal.df_dx[0] = x1;
        terminal.df_dtheta[0] = 0.0;
        terminal.c[0] = x1 - 1.0;
        terminal.dc_dx[0] = 1.0;
        terminal.dc_dtheta[0] = 0.0;
        terminal.d2L_dx2[0] = 1.0;
        terminal.d2L_dxdtheta[0] = 0.0;
        terminal.d2L_dtheta2[0] = 0.0;
      },
      .timeout_callback = []() { return false; },
      .residual_scaling = residual_scaling.get(),
  };

  for (int i = 0; i < input.dimensions.get_x_dim(kNumEdges); ++i) {
    workspace.sip_workspace.vars.x[i] = 0.0;
  }
  for (int i = 0; i < input.dimensions.get_y_dim(kNumEdges); ++i) {
    workspace.sip_workspace.vars.y[i] = 0.0;
  }

  return ::sip::optimal_control::solve(input, solver_settings, workspace);
}

} // namespace sip_examples::problem_definitions::cross_stage_variable_lqr
