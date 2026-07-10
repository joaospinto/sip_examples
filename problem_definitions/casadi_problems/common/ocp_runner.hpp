#pragma once

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

template <typename GeneratedProblem>
OcpResult run_ocp(const sip::Settings &settings) {
  const auto &spec = GeneratedProblem::ocp_spec();

  ::sip::optimal_control::Workspace workspace;
  workspace.reserve(spec.state_dim, spec.control_dim, spec.num_stages,
                    spec.c_dim, spec.g_dim, spec.theta_dim);

  auto work = GeneratedProblem::make_ocp_work();
  const auto model_callback =
      [&](const ::sip::optimal_control::ModelCallbackInput &mci) -> void {
    GeneratedProblem::eval_ocp(mci, workspace.model_callback_output, work);
  };
  const auto timeout_callback = []() -> bool { return false; };

  const ::sip::optimal_control::Input input{
      .model_callback = std::cref(model_callback),
      .timeout_callback = std::cref(timeout_callback),
      .dimensions =
          {
              .num_stages = spec.num_stages,
              .state_dim = spec.state_dim,
              .control_dim = spec.control_dim,
              .c_dim = spec.c_dim,
              .g_dim = spec.g_dim,
              .theta_dim = spec.theta_dim,
          },
  };

  const int x_dim = input.dimensions.get_x_dim();
  const int y_dim = input.dimensions.get_y_dim();
  const int z_dim = input.dimensions.get_z_dim();
  std::copy_n(spec.initial_x, x_dim, workspace.sip_workspace.vars.x);
  std::fill_n(workspace.sip_workspace.vars.y, y_dim, 0.0);
  std::fill_n(workspace.sip_workspace.vars.z, z_dim, 1.0);

  workspace.model_callback_input.theta =
      workspace.sip_workspace.vars.x +
      input.dimensions.get_stagewise_x_dim();
  double *x = workspace.sip_workspace.vars.x;
  for (int i = 0; i < spec.num_stages; ++i) {
    workspace.model_callback_input.states[i] = x;
    x += spec.state_dim;
    workspace.model_callback_input.controls[i] = x;
    x += spec.control_dim;
  }
  workspace.model_callback_input.states[spec.num_stages] = x;

  double *y = workspace.sip_workspace.vars.y;
  for (int i = 0; i <= spec.num_stages; ++i) {
    workspace.model_callback_input.costates[i] = y;
    y += spec.state_dim;
    workspace.model_callback_input.equality_constraint_multipliers[i] = y;
    y += spec.c_dim;
  }

  double *z = workspace.sip_workspace.vars.z;
  for (int i = 0; i <= spec.num_stages; ++i) {
    workspace.model_callback_input.inequality_constraint_multipliers[i] = z;
    z += spec.g_dim;
  }

  model_callback(workspace.model_callback_input);
  if (z_dim > 0) {
    for (int i = 0; i <= spec.num_stages; ++i) {
      initialize_slacks_and_duals(
          workspace.model_callback_output.g[i], spec.g_dim,
          settings.barrier.initial_mu,
          workspace.sip_workspace.vars.s + i * spec.g_dim,
          workspace.sip_workspace.vars.z + i * spec.g_dim);
    }
  }

  auto output = ::sip::optimal_control::solve(input, settings, workspace);
  std::vector<double> result_x(workspace.sip_workspace.vars.x,
                               workspace.sip_workspace.vars.x + x_dim);

  workspace.free(spec.num_stages);
  return {.output = output, .x = std::move(result_x)};
}

} // namespace sip_examples::problem_definitions::casadi_problems
