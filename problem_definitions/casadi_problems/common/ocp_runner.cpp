#include "problem_definitions/casadi_problems/common/ocp_runner.hpp"

namespace sip_examples::problem_definitions::casadi_problems {

void bind_ocp_model_callback_input(
    const ::sip::optimal_control::Dimensions &dimensions,
    const ::sip::optimal_control::Topology &topology,
    ::sip::optimal_control::Workspace &workspace) {
  double *x = workspace.sip_workspace.vars.x;
  for (int edge = 0; edge < topology.num_edges; ++edge) {
    workspace.model_callback_input.states[edge] = x;
    x += dimensions.get_state_dim(edge);
    workspace.model_callback_input.controls[edge] = x;
    x += dimensions.get_control_dim(edge);
  }
  workspace.model_callback_input.states[topology.num_edges] = x;
  x += dimensions.get_state_dim(topology.num_edges);
  workspace.model_callback_input.theta = x;

  double *y = workspace.sip_workspace.vars.y;
  for (int node = 0; node < topology.num_nodes(); ++node) {
    workspace.model_callback_input.costates[node] = y;
    y += dimensions.get_state_dim(node);
    workspace.model_callback_input.equality_constraint_multipliers[node] = y;
    y += dimensions.get_c_dim(node);
  }

  double *z = workspace.sip_workspace.vars.z;
  for (int node = 0; node < topology.num_nodes(); ++node) {
    workspace.model_callback_input.inequality_constraint_multipliers[node] = z;
    z += dimensions.get_g_dim(node);
  }
}

} // namespace sip_examples::problem_definitions::casadi_problems
