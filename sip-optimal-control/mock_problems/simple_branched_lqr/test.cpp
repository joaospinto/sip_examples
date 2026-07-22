#include "problem_definitions/mock_problems/simple_branched_lqr/problem.hpp"
#include "problem_definitions/unit_residual_scaling.hpp"
#include "sip_optimal_control/sip_optimal_control.hpp"

#include <algorithm>
#include <array>
#include <vector>

#include <gtest/gtest.h>

namespace sip_examples {
namespace problem = ::sip_examples::problem_definitions::simple_branched_lqr;
namespace {

auto solve_branched_problem(const ::sip::optimal_control::Input &input,
                            ::sip::optimal_control::Workspace &workspace) {
  problem::initialize(workspace);
  return ::sip::optimal_control::solve(input, problem::settings(), workspace);
}

void expect_solution(const double *x) {
  for (int i = 0; i < problem::kXDim; ++i) {
    EXPECT_NEAR(x[i], problem::kSolution[i], 1e-8);
  }
}

TEST(SimpleBranchedLQR, WithMemAssign) {
  std::array<unsigned char,
             ::sip::optimal_control::Dimensions::num_bytes(problem::kNumEdges)>
      dimension_bytes{};
  ::sip::optimal_control::Dimensions dimensions;
  ASSERT_EQ(dimensions.mem_assign(problem::kNumEdges, dimension_bytes.data()),
            static_cast<int>(dimension_bytes.size()));
  dimensions.set_uniform(problem::kNumEdges, problem::kStateDim,
                         problem::kControlDim, 0, 0, 0, 0);
  std::copy(problem::kNodeCDims.begin(), problem::kNodeCDims.end(),
            const_cast<int *>(dimensions.node_c_dims));

  std::array<unsigned char,
             ::sip::optimal_control::Topology::num_bytes(problem::kNumEdges)>
      topology_bytes{};
  ::sip::optimal_control::Topology topology;
  ASSERT_EQ(topology.mem_assign(problem::kNumEdges, topology_bytes.data()),
            static_cast<int>(topology_bytes.size()));
  topology.set_tree(0, problem::kEdgeParents.data(),
                    problem::kEdgeChildren.data());

  ::sip::optimal_control::Workspace workspace;
  const problem_definitions::UnitResidualScaling residual_scaling(
      problem::kXDim, problem::kSDim, problem::kOcpYDim);
  const ::sip::optimal_control::Input input{
      .dimensions = dimensions,
      .topology = topology,
      .initial_state = problem::kInitialState.data(),
      .model_callback =
          [](const ::sip::optimal_control::ModelCallbackInput &mci,
             ::sip::optimal_control::ModelCallbackOutput &mco) {
            problem::evaluate_optimal_control(mci, mco);
          },
      .timeout_callback = []() { return false; },
      .residual_scaling = residual_scaling.get(),
  };
  std::vector<unsigned char> workspace_bytes(
      ::sip::optimal_control::Workspace::num_bytes(
          input.dimensions, input.topology, input.num_bound_sides(),
          problem::settings()));
  ASSERT_EQ(workspace.mem_assign(input.dimensions, input.topology,
                                 input.num_bound_sides(), problem::settings(),
                                 workspace_bytes.data()),
            static_cast<int>(workspace_bytes.size()));

  const auto output = solve_branched_problem(input, workspace);

  ASSERT_EQ(output.exit_status, ::sip::Status::SOLVED);
  expect_solution(workspace.sip_workspace.vars.x);
}

TEST(SimpleBranchedLQR, WithReserve) {
  ::sip::optimal_control::Dimensions dimensions;
  dimensions.reserve(problem::kNumEdges);
  dimensions.set_uniform(problem::kNumEdges, problem::kStateDim,
                         problem::kControlDim, 0, 0, 0, 0);
  std::copy(problem::kNodeCDims.begin(), problem::kNodeCDims.end(),
            const_cast<int *>(dimensions.node_c_dims));
  ::sip::optimal_control::Topology topology;
  topology.reserve(problem::kNumEdges);
  topology.set_tree(0, problem::kEdgeParents.data(),
                    problem::kEdgeChildren.data());

  ::sip::optimal_control::Workspace workspace;
  const problem_definitions::UnitResidualScaling residual_scaling(
      problem::kXDim, problem::kSDim, problem::kOcpYDim);
  const ::sip::optimal_control::Input input{
      .dimensions = dimensions,
      .topology = topology,
      .initial_state = problem::kInitialState.data(),
      .model_callback =
          [](const ::sip::optimal_control::ModelCallbackInput &mci,
             ::sip::optimal_control::ModelCallbackOutput &mco) {
            problem::evaluate_optimal_control(mci, mco);
          },
      .timeout_callback = []() { return false; },
      .residual_scaling = residual_scaling.get(),
  };
  workspace.reserve(input.dimensions, input.topology, input.num_bound_sides(),
                    problem::settings());

  const auto output = solve_branched_problem(input, workspace);

  ASSERT_EQ(output.exit_status, ::sip::Status::SOLVED);
  expect_solution(workspace.sip_workspace.vars.x);

  workspace.free(input.topology);
  topology.free();
  dimensions.free();
}

} // namespace
} // namespace sip_examples
