#include "problem_definitions/mock_problems/cross_stage_variable_lqr/problem.hpp"
#include "problem_definitions/unit_residual_scaling.hpp"

#include <array>

namespace sip_examples::problem_definitions::cross_stage_variable_lqr {

const std::array<int, 2> kStateDims = {1, 1};
const std::array<int, 1> kControlDims = {1};
const std::array<int, 2> kCDims = {1, 1};
const std::array<int, 2> kGDims = {0, 0};
const std::array<int, 1> kEdgeParents = {0};
const std::array<int, 1> kEdgeChildren = {1};

const ::sip::optimal_control::Dimensions kDimensions{
    kThetaDim, kStateDims.data(), kControlDims.data(), kCDims.data(),
    kGDims.data()};
const ::sip::optimal_control::Topology kTopology{
    kNumEdges, 0, kEdgeParents.data(), kEdgeChildren.data()};

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
      dimensions.get_z_dim(topology.num_nodes()),
      dimensions.get_y_dim(topology.num_nodes()));
  const ::sip::optimal_control::Input input{
      .dimensions = dimensions,
      .topology = topology,
      .model_callback =
          [&](const ::sip::optimal_control::ModelCallbackInput &mci) -> void {
        auto &mco = workspace.model_callback_output;

        const double x0 = mci.states[0][0];
        const double u0 = mci.controls[0][0];
        const double x1 = mci.states[1][0];
        const double theta = mci.theta[0];

        mco.f =
            0.5 * x0 * x0 + 0.5 * u0 * u0 + 0.5 * x1 * x1 + 0.5 * theta * theta;

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
      },
      .timeout_callback = []() { return false; },
      .residual_scaling = residual_scaling.get(),
  };

  for (int i = 0; i < input.dimensions.get_x_dim(kNumEdges); ++i) {
    workspace.sip_workspace.vars.x[i] = 0.0;
  }
  for (int i = 0; i < input.dimensions.get_y_dim(kNumEdges + 1); ++i) {
    workspace.sip_workspace.vars.y[i] = 0.0;
  }

  return ::sip::optimal_control::solve(input, solver_settings, workspace);
}

} // namespace sip_examples::problem_definitions::cross_stage_variable_lqr
