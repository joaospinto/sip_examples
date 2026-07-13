#include GENERATED_HEADER

#include "problem_definitions/casadi_problems/common/model_signature.hpp"
#include "problem_definitions/casadi_problems/common/ocp_runner.hpp"
#include "sip_optimal_control/helpers.hpp"

#include <algorithm>
#include <functional>
#include <iostream>
#include <vector>

namespace casadi_problems = sip_examples::problem_definitions::casadi_problems;
using casadi_problems::generated_problem::Problem;

auto main() -> int {
  const auto &spec = Problem::ocp_spec();
  const int num_edges = spec.num_edges;
  auto work = Problem::make_ocp_work();
  ::sip::optimal_control::Workspace workspace;

  const auto model_callback =
      [&](const ::sip::optimal_control::ModelCallbackInput &input) -> void {
    Problem::eval_ocp(input, workspace.model_callback_output, work);
  };
  const auto timeout_callback = []() -> bool { return false; };
  ::sip::optimal_control::Input input{
      .dimensions = spec.dimensions,
      .topology = {num_edges, spec.topology_root, spec.edge_parents,
                   spec.edge_children},
      .model_callback = std::cref(model_callback),
      .timeout_callback = std::cref(timeout_callback),
  };
  workspace.reserve(input.dimensions, input.topology, 1);

  const int x_dim = input.dimensions.get_x_dim(num_edges);
  const int y_dim = input.dimensions.get_y_dim(input.topology.num_nodes());
  const int z_dim = input.dimensions.get_z_dim(input.topology.num_nodes());
  std::copy_n(spec.initial_x, x_dim, workspace.sip_workspace.vars.x);
  const auto y = casadi_problems::scaled_sequence(y_dim, 0.02);
  const auto z = casadi_problems::scaled_sequence(z_dim, 0.03);
  std::copy(y.begin(), y.end(), workspace.sip_workspace.vars.y);
  std::copy(z.begin(), z.end(), workspace.sip_workspace.vars.z);
  casadi_problems::bind_ocp_model_callback_input(input.dimensions,
                                                 input.topology, workspace);
  model_callback(workspace.model_callback_input);

  casadi_problems::ModelSignature signature{
      .x_dim = x_dim,
      .y_dim = y_dim,
      .z_dim = z_dim,
      .f = workspace.model_callback_output.f,
      .gradient_f = std::vector<double>(x_dim),
      .c = std::vector<double>(y_dim),
      .g = std::vector<double>(z_dim),
      .hessian_product = std::vector<double>(x_dim),
      .jacobian_c_product = std::vector<double>(y_dim),
      .jacobian_c_transpose_product = std::vector<double>(x_dim),
      .jacobian_g_product = std::vector<double>(z_dim),
      .jacobian_g_transpose_product = std::vector<double>(x_dim),
  };

  double *gradient = signature.gradient_f.data();
  for (int edge = 0; edge < num_edges; ++edge) {
    const int state_dim = input.dimensions.get_state_dim(edge);
    const int control_dim = input.dimensions.get_control_dim(edge);
    std::copy_n(workspace.model_callback_output.df_dx[edge], state_dim,
                gradient);
    gradient += state_dim;
    std::copy_n(workspace.model_callback_output.df_du[edge], control_dim,
                gradient);
    gradient += control_dim;
  }
  std::copy_n(workspace.model_callback_output.df_dx[num_edges],
              input.dimensions.get_state_dim(num_edges), gradient);
  gradient += input.dimensions.get_state_dim(num_edges);
  std::copy_n(workspace.model_callback_output.df_dtheta,
              input.dimensions.theta_dim, gradient);

  double *c = signature.c.data();
  double *g = signature.g.data();
  for (int node = 0; node <= num_edges; ++node) {
    const int state_dim = input.dimensions.get_state_dim(node);
    const int c_dim = input.dimensions.get_c_dim(node);
    const int g_dim = input.dimensions.get_g_dim(node);
    std::copy_n(workspace.model_callback_output.dyn_res[node], state_dim, c);
    c += state_dim;
    std::copy_n(workspace.model_callback_output.c[node], c_dim, c);
    c += c_dim;
    std::copy_n(workspace.model_callback_output.g[node], g_dim, g);
    g += g_dim;
  }

  const auto x_direction = casadi_problems::scaled_sequence(x_dim, 0.01);
  const auto y_direction = casadi_problems::scaled_sequence(y_dim, -0.02);
  const auto z_direction = casadi_problems::scaled_sequence(z_dim, -0.03);
  ::sip::optimal_control::CallbackProvider callback_provider(input, workspace);
  callback_provider.add_Hx_to_y(x_direction.data(),
                                signature.hessian_product.data());
  callback_provider.add_Cx_to_y(x_direction.data(),
                                signature.jacobian_c_product.data());
  callback_provider.add_CTx_to_y(y_direction.data(),
                                 signature.jacobian_c_transpose_product.data());
  callback_provider.add_Gx_to_y(x_direction.data(),
                                signature.jacobian_g_product.data());
  callback_provider.add_GTx_to_y(z_direction.data(),
                                 signature.jacobian_g_transpose_product.data());

  casadi_problems::print_model_signature(signature, std::cout);
  workspace.free(input.topology);
  return 0;
}
