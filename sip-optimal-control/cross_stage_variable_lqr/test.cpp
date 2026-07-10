#include "problem_definitions/cross_stage_variable_lqr/problem.hpp"

#include <array>

#include <gtest/gtest.h>

namespace sip_examples {
namespace problem = ::sip_examples::problem_definitions::cross_stage_variable_lqr;

namespace {
void expect_solution(const ::sip::optimal_control::Workspace &workspace) {
  EXPECT_NEAR(workspace.sip_workspace.vars.x[0], 0.0, 1e-6);
  EXPECT_NEAR(workspace.sip_workspace.vars.x[1], 0.5, 1e-6);
  EXPECT_NEAR(workspace.sip_workspace.vars.x[2], 1.0, 1e-6);
  EXPECT_NEAR(workspace.sip_workspace.vars.x[problem::kThetaOffset], 0.5, 1e-6);
}
} // namespace

TEST(CrossStageVariableLQR, WithMemAssign) {
  ::sip::optimal_control::Workspace workspace;
  constexpr int kWorkspaceSize = ::sip::optimal_control::Workspace::num_bytes(
      problem::kStateDim, problem::kControlDim, problem::kNumStages,
      problem::kCDim, problem::kGDim, problem::kThetaDim);
  std::array<unsigned char, kWorkspaceSize> workspace_bytes;
  workspace.mem_assign(problem::kStateDim, problem::kControlDim,
                       problem::kNumStages, problem::kCDim, problem::kGDim,
                       workspace_bytes.data(), problem::kThetaDim);

  const auto output = problem::run_solver(workspace);

  ASSERT_EQ(output.exit_status, ::sip::Status::SOLVED);
  expect_solution(workspace);
}

TEST(CrossStageVariableLQR, WithReserve) {
  ::sip::optimal_control::Workspace workspace;
  workspace.reserve(problem::kStateDim, problem::kControlDim,
                    problem::kNumStages, problem::kCDim, problem::kGDim,
                    problem::kThetaDim);

  const auto output = problem::run_solver(workspace);

  ASSERT_EQ(output.exit_status, ::sip::Status::SOLVED);
  expect_solution(workspace);

  workspace.free(problem::kNumStages);
}

} // namespace sip_examples
