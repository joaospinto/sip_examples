#include "problem_definitions/simple_constrained_lqr/problem.hpp"

#include <array>

#include <gtest/gtest.h>

namespace sip_examples {
namespace problem = ::sip_examples::problem_definitions::simple_constrained_lqr;

TEST(SimpleConstrainedLQR, Problem1WithMemAssign) {
  ::sip::optimal_control::Workspace workspace;
  constexpr int kWorkspaceSize = ::sip::optimal_control::Workspace::num_bytes(
      problem::kStateDim, problem::kControlDim, problem::kNumStages,
      problem::kCDim, problem::kGDim);
  std::array<unsigned char, kWorkspaceSize> workspace_bytes;
  workspace.mem_assign(problem::kStateDim, problem::kControlDim,
                       problem::kNumStages, problem::kCDim, problem::kGDim,
                       workspace_bytes.data());

  const auto output = problem::run_solver(workspace);

  ASSERT_EQ(output.exit_status, ::sip::Status::SOLVED);
}

TEST(SimpleConstrainedLQR, Problem1WithReserve) {
  ::sip::optimal_control::Workspace workspace;
  workspace.reserve(problem::kStateDim, problem::kControlDim,
                    problem::kNumStages, problem::kCDim, problem::kGDim);

  const auto output = problem::run_solver(workspace);

  ASSERT_EQ(output.exit_status, ::sip::Status::SOLVED);
}

} // namespace sip_examples
