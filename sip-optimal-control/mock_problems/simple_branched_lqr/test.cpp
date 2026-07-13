#include "problem_definitions/mock_problems/simple_branched_lqr/problem.hpp"
#include "sip_optimal_control/sip_optimal_control.hpp"

#include <array>

#include <gtest/gtest.h>

namespace sip_examples {
namespace problem = ::sip_examples::problem_definitions::simple_branched_lqr;
namespace {

constexpr int kFilterCapacity =
    ::sip::FilterWorkspace::required_capacity(::sip::Settings{});

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
                         problem::kControlDim, problem::kCDim, problem::kGDim);

  std::array<unsigned char,
             ::sip::optimal_control::Topology::num_bytes(problem::kNumEdges)>
      topology_bytes{};
  ::sip::optimal_control::Topology topology;
  ASSERT_EQ(topology.mem_assign(problem::kNumEdges, topology_bytes.data()),
            static_cast<int>(topology_bytes.size()));
  topology.set_tree(0, problem::kEdgeParents.data(),
                    problem::kEdgeChildren.data());

  ::sip::optimal_control::Workspace workspace;
  const ::sip::optimal_control::Input input{
      .dimensions = dimensions,
      .topology = topology,
      .model_callback =
          [&workspace](const ::sip::optimal_control::ModelCallbackInput &mci) {
            problem::evaluate_optimal_control(mci,
                                              workspace.model_callback_output);
          },
      .timeout_callback = []() { return false; },
  };
  constexpr int kWorkspaceSize = ::sip::optimal_control::Workspace::num_bytes(
      problem::kStateDim, problem::kControlDim, problem::kNumEdges,
      problem::kCDim, problem::kGDim, 0, kFilterCapacity);
  std::array<unsigned char, kWorkspaceSize> workspace_bytes{};
  ASSERT_EQ(kFilterCapacity,
            ::sip::FilterWorkspace::required_capacity(problem::settings()));
  ASSERT_EQ(::sip::optimal_control::Workspace::num_bytes(
                input.dimensions, input.topology, kFilterCapacity),
            static_cast<int>(workspace_bytes.size()));
  ASSERT_EQ(workspace.mem_assign(input.dimensions, input.topology,
                                 kFilterCapacity, workspace_bytes.data()),
            static_cast<int>(workspace_bytes.size()));

  const auto output = solve_branched_problem(input, workspace);

  ASSERT_EQ(output.exit_status, ::sip::Status::SOLVED);
  expect_solution(workspace.sip_workspace.vars.x);
}

TEST(SimpleBranchedLQR, WithReserve) {
  ::sip::optimal_control::Dimensions dimensions;
  dimensions.reserve(problem::kNumEdges);
  dimensions.set_uniform(problem::kNumEdges, problem::kStateDim,
                         problem::kControlDim, problem::kCDim, problem::kGDim);
  ::sip::optimal_control::Topology topology;
  topology.reserve(problem::kNumEdges);
  topology.set_tree(0, problem::kEdgeParents.data(),
                    problem::kEdgeChildren.data());

  ::sip::optimal_control::Workspace workspace;
  const ::sip::optimal_control::Input input{
      .dimensions = dimensions,
      .topology = topology,
      .model_callback =
          [&workspace](const ::sip::optimal_control::ModelCallbackInput &mci) {
            problem::evaluate_optimal_control(mci,
                                              workspace.model_callback_output);
          },
      .timeout_callback = []() { return false; },
  };
  workspace.reserve(
      input.dimensions, input.topology,
      ::sip::FilterWorkspace::required_capacity(problem::settings()));

  const auto output = solve_branched_problem(input, workspace);

  ASSERT_EQ(output.exit_status, ::sip::Status::SOLVED);
  expect_solution(workspace.sip_workspace.vars.x);

  workspace.free(input.topology);
  topology.free();
  dimensions.free();
}

} // namespace
} // namespace sip_examples
