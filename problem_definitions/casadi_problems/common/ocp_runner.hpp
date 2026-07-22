#pragma once

#include "problem_definitions/casadi_problems/common/ocp_problem.hpp"
#include "problem_definitions/casadi_problems/common/problem.hpp"
#include "problem_definitions/unit_residual_scaling.hpp"
#include "sip_optimal_control/sip_optimal_control.hpp"

#include <algorithm>
#include <functional>
#include <vector>

namespace sip_examples::problem_definitions::casadi_problems {

struct OcpResult {
  sip::Output output;
  std::vector<double> x;
};

template <typename GeneratedProblem>
OcpResult run_ocp(const sip::Settings &settings) {
  const auto &spec = GeneratedProblem::ocp_spec();

  ::sip::optimal_control::Workspace workspace;
  const int num_edges = spec.num_edges;
  const int num_nodes = num_edges + 1;
  const int x_dim = spec.dimensions.get_x_dim(num_edges);
  const int y_dim = spec.dimensions.get_y_dim(num_edges);
  const int z_dim = spec.dimensions.get_z_dim(num_edges);
  const UnitResidualScaling residual_scaling(x_dim, z_dim, y_dim);

  auto work = GeneratedProblem::make_ocp_work();
  const auto model_callback =
      [&](const ::sip::optimal_control::ModelCallbackInput &mci,
          ::sip::optimal_control::ModelCallbackOutput &mco) -> void {
    GeneratedProblem::eval_ocp(mci, mco, work);
  };
  const auto timeout_callback = []() -> bool { return false; };

  ::sip::optimal_control::Input input{
      .dimensions = spec.dimensions,
      .topology = {num_edges, spec.topology_root, spec.edge_parents,
                   spec.edge_children},
      .initial_state = spec.initial_state,
      .model_callback = std::cref(model_callback),
      .timeout_callback = std::cref(timeout_callback),
      .lower_bounds = spec.lower_bounds,
      .upper_bounds = spec.upper_bounds,
      .residual_scaling = residual_scaling.get(),
  };
  workspace.reserve(input.dimensions, input.topology, input.num_bound_sides(),
                    settings);

  std::copy_n(spec.initial_x, x_dim, workspace.sip_workspace.vars.x);
  std::fill_n(workspace.sip_workspace.vars.y, y_dim, 0.0);
  std::fill_n(workspace.sip_workspace.vars.z, z_dim, 1.0);
  initialize_bound_slacks_and_duals(
      spec.lower_bounds, spec.upper_bounds, x_dim, settings.barrier.initial_mu,
      workspace.sip_workspace.vars.x, workspace.sip_workspace.vars.bound_s,
      workspace.sip_workspace.vars.bound_z);

  auto &model_input = workspace.model_callback_input;
  model_input.theta =
      workspace.sip_workspace.vars.x + workspace.stagewise_x_dim;
  for (int node = 0; node < num_nodes; ++node) {
    model_input.nodes[node] = {
        .node = node,
        .state =
            workspace.sip_workspace.vars.x + workspace.x_state_offsets[node],
        .equality_constraint_multipliers =
            workspace.sip_workspace.vars.y + workspace.y_node_c_offsets[node],
        .inequality_constraint_multipliers =
            workspace.sip_workspace.vars.z + workspace.z_node_offsets[node],
    };
  }
  for (int edge = 0; edge < num_edges; ++edge) {
    const int parent = spec.edge_parents[edge];
    const int child = spec.edge_children[edge];
    model_input.edges[edge] = {
        .edge = edge,
        .parent = parent,
        .child = child,
        .parent_state =
            workspace.sip_workspace.vars.x + workspace.x_state_offsets[parent],
        .control =
            workspace.sip_workspace.vars.x + workspace.x_control_offsets[edge],
        .child_state =
            workspace.sip_workspace.vars.x + workspace.x_state_offsets[child],
        .costate =
            workspace.sip_workspace.vars.y + workspace.y_dyn_offsets[child],
        .equality_constraint_multipliers =
            workspace.sip_workspace.vars.y + workspace.y_edge_c_offsets[edge],
        .inequality_constraint_multipliers =
            workspace.sip_workspace.vars.z + workspace.z_edge_offsets[edge],
    };
  }

  model_callback(model_input, workspace.model_callback_output);
  if (z_dim > 0) {
    double *s = workspace.sip_workspace.vars.s;
    double *dual = workspace.sip_workspace.vars.z;
    for (int node = 0; node < num_nodes; ++node) {
      const int g_dim = input.dimensions.get_node_g_dim(node);
      initialize_slacks_and_duals(workspace.model_callback_output.nodes[node].g,
                                  g_dim, settings.barrier.initial_mu, s, dual);
      s += g_dim;
      dual += g_dim;
    }
    for (int edge = 0; edge < num_edges; ++edge) {
      const int g_dim = input.dimensions.get_edge_g_dim(edge);
      initialize_slacks_and_duals(workspace.model_callback_output.edges[edge].g,
                                  g_dim, settings.barrier.initial_mu, s, dual);
      s += g_dim;
      dual += g_dim;
    }
  }

  auto output = ::sip::optimal_control::solve(input, settings, workspace);
  std::vector<double> result_x(workspace.sip_workspace.vars.x,
                               workspace.sip_workspace.vars.x + x_dim);

  workspace.free(input.topology);
  return {.output = output, .x = std::move(result_x)};
}

} // namespace sip_examples::problem_definitions::casadi_problems
