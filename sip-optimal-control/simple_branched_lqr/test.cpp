#include "sip_optimal_control/sip_optimal_control.hpp"

#include <array>
#include <cstddef>
#include <functional>

#include <gtest/gtest.h>

namespace sip_examples {
namespace {

constexpr int kNumStages = 2;
constexpr int kStateDim = 1;
constexpr int kControlDim = 1;
constexpr int kCDim = 1;
constexpr int kGDim = 0;

constexpr std::array<int, kNumStages + 1> kStateDims = {1, 1, 1};
constexpr std::array<int, kNumStages> kControlDims = {1, 1};
constexpr std::array<int, kNumStages + 1> kCDims = {1, 1, 1};
constexpr std::array<int, kNumStages + 1> kGDims = {0, 0, 0};

struct BranchTopology {
  std::array<int, kNumStages> parent = {0, 0};
  std::array<int, kNumStages> child = {1, 2};

  static int root(const void *) { return 0; }

  static int edge_parent(const void *context, const int edge) {
    return static_cast<const BranchTopology *>(context)->parent[edge];
  }

  static int edge_child(const void *context, const int edge) {
    return static_cast<const BranchTopology *>(context)->child[edge];
  }
};

auto dimensions() -> ::sip::optimal_control::Dimensions {
  return {
      .num_stages = kNumStages,
      .state_dim = kStateDim,
      .control_dim = kControlDim,
      .c_dim = kCDim,
      .g_dim = kGDim,
      .state_dims = kStateDims.data(),
      .control_dims = kControlDims.data(),
      .c_dims = kCDims.data(),
      .g_dims = kGDims.data(),
  };
}

auto topology(const BranchTopology &branch_topology)
    -> ::sip::optimal_control::Input::Topology {
  return {
      .context = &branch_topology,
      .root = BranchTopology::root,
      .edge_parent = BranchTopology::edge_parent,
      .edge_child = BranchTopology::edge_child,
  };
}

void evaluate_model(const ::sip::optimal_control::ModelCallbackInput &mci,
                    ::sip::optimal_control::Workspace &workspace) {
  auto &mco = workspace.model_callback_output;

  const double x0 = mci.states[0][0];
  const double x1 = mci.states[1][0];
  const double x2 = mci.states[2][0];
  const double u0 = mci.controls[0][0];
  const double u1 = mci.controls[1][0];

  mco.f = 0.5 * u0 * u0 + 0.5 * u1 * u1;

  for (int node = 0; node <= kNumStages; ++node) {
    mco.df_dx[node][0] = 0.0;
    mco.c[node][0] = 0.0;
    mco.dc_dx[node][0] = 0.0;
    mco.d2L_dx2[node][0] = 0.0;
  }
  for (int edge = 0; edge < kNumStages; ++edge) {
    mco.df_du[edge][0] = mci.controls[edge][0];
    mco.ddyn_dx[edge][0] = 1.0;
    mco.ddyn_du[edge][0] = 1.0;
    mco.dc_du[edge][0] = 0.0;
    mco.d2L_dxdu[edge][0] = 0.0;
    mco.d2L_du2[edge][0] = 1.0;
  }

  mco.dyn_res[0][0] = -x0;
  mco.dyn_res[1][0] = x0 + u0 - x1;
  mco.dyn_res[2][0] = x0 + u1 - x2;

  mco.c[1][0] = x1 - 1.0;
  mco.c[2][0] = x2 + 1.0;
  mco.dc_dx[1][0] = 1.0;
  mco.dc_dx[2][0] = 1.0;
}

auto solve_branched_problem(::sip::optimal_control::Workspace &workspace) {
  const auto dims = dimensions();
  const BranchTopology branch_topology;

  const auto model_callback =
      [&](const ::sip::optimal_control::ModelCallbackInput &mci) -> void {
    evaluate_model(mci, workspace);
  };
  const auto timeout_callback = []() -> bool { return false; };

  const ::sip::optimal_control::Input input{
      .model_callback = std::cref(model_callback),
      .timeout_callback = std::cref(timeout_callback),
      .dimensions = dims,
      .topology = topology(branch_topology),
  };

  auto settings = sip::Settings{
      .max_iterations = 100,
      .termination =
          {
              .max_dual_residual = 1e-9,
              .max_constraint_violation = 1e-9,
              .max_complementarity_gap = 1e-9,
          },
      .line_search =
          {
              .max_iterations = 100,
              .skip_line_search = true,
          },
  };

  for (int i = 0; i < dims.get_x_dim(); ++i) {
    workspace.sip_workspace.vars.x[i] = 0.0;
  }
  for (int i = 0; i < dims.get_y_dim(); ++i) {
    workspace.sip_workspace.vars.y[i] = 0.0;
  }

  return ::sip::optimal_control::solve(input, settings, workspace);
}

void expect_solution(const ::sip::optimal_control::Workspace &workspace) {
  EXPECT_NEAR(workspace.sip_workspace.vars.x[0], 0.0, 1e-8);
  EXPECT_NEAR(workspace.sip_workspace.vars.x[1], 1.0, 1e-8);
  EXPECT_NEAR(workspace.sip_workspace.vars.x[2], 1.0, 1e-8);
  EXPECT_NEAR(workspace.sip_workspace.vars.x[3], -1.0, 1e-8);
  EXPECT_NEAR(workspace.sip_workspace.vars.x[4], -1.0, 1e-8);
}

TEST(SimpleBranchedLQR, WithMemAssign) {
  const auto dims = dimensions();
  constexpr int kWorkspaceCapacity = 65536;
  std::array<unsigned char, kWorkspaceCapacity> workspace_bytes{};
  ASSERT_LE(::sip::optimal_control::Workspace::num_bytes(dims),
            static_cast<int>(workspace_bytes.size()));

  ::sip::optimal_control::Workspace workspace;
  workspace.mem_assign(dims, workspace_bytes.data());

  const auto output = solve_branched_problem(workspace);

  ASSERT_EQ(output.exit_status, ::sip::Status::SOLVED);
  expect_solution(workspace);
}

TEST(SimpleBranchedLQR, WithReserve) {
  const auto dims = dimensions();
  ::sip::optimal_control::Workspace workspace;
  workspace.reserve(dims);

  const auto output = solve_branched_problem(workspace);

  ASSERT_EQ(output.exit_status, ::sip::Status::SOLVED);
  expect_solution(workspace);

  workspace.free(kNumStages);
}

} // namespace
} // namespace sip_examples
