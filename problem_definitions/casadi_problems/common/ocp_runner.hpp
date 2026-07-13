#pragma once

#include "problem_definitions/casadi_problems/common/ocp_problem.hpp"
#include "problem_definitions/casadi_problems/common/problem.hpp"
#include "sip_optimal_control/sip_optimal_control.hpp"

#include <algorithm>
#include <functional>
#include <vector>

namespace sip_examples::problem_definitions::casadi_problems {

struct OcpResult {
  sip::Output output;
  std::vector<double> x;
};

void bind_ocp_model_callback_input(
    const ::sip::optimal_control::Dimensions &dimensions,
    const ::sip::optimal_control::Topology &topology,
    ::sip::optimal_control::Workspace &workspace);

template <typename GeneratedProblem>
OcpResult run_ocp(const sip::Settings &settings) {
  const auto &spec = GeneratedProblem::ocp_spec();

  ::sip::optimal_control::Workspace workspace;
  const int num_edges = spec.num_edges;

  auto work = GeneratedProblem::make_ocp_work();
  const auto model_callback =
      [&](const ::sip::optimal_control::ModelCallbackInput &mci) -> void {
    GeneratedProblem::eval_ocp(mci, workspace.model_callback_output, work);
  };
  const auto timeout_callback = []() -> bool { return false; };

  ::sip::optimal_control::Input input{
      .dimensions = spec.dimensions,
      .topology = {num_edges, spec.topology_root, spec.edge_parents,
                   spec.edge_children},
      .model_callback = std::cref(model_callback),
      .timeout_callback = std::cref(timeout_callback),
  };
  workspace.reserve(input.dimensions, input.topology,
                    sip::FilterWorkspace::required_capacity(settings));

  const int x_dim = input.dimensions.get_x_dim(input.topology.num_edges);
  const int y_dim = input.dimensions.get_y_dim(input.topology.num_nodes());
  const int z_dim = input.dimensions.get_z_dim(input.topology.num_nodes());
  std::copy_n(spec.initial_x, x_dim, workspace.sip_workspace.vars.x);
  std::fill_n(workspace.sip_workspace.vars.y, y_dim, 0.0);
  std::fill_n(workspace.sip_workspace.vars.z, z_dim, 1.0);

  bind_ocp_model_callback_input(input.dimensions, input.topology, workspace);

  model_callback(workspace.model_callback_input);
  if (z_dim > 0) {
    double *s = workspace.sip_workspace.vars.s;
    double *dual = workspace.sip_workspace.vars.z;
    for (int i = 0; i <= num_edges; ++i) {
      const int g_dim = input.dimensions.get_g_dim(i);
      initialize_slacks_and_duals(workspace.model_callback_output.g[i], g_dim,
                                  settings.barrier.initial_mu, s, dual);
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
