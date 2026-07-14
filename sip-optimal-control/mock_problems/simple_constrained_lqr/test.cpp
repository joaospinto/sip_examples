#include "problem_definitions/mock_problems/simple_constrained_lqr/problem.hpp"

#include <array>

#include <gtest/gtest.h>

namespace sip_examples {
namespace problem = ::sip_examples::problem_definitions::simple_constrained_lqr;
TEST(SimpleConstrainedLQR, Problem1WithMemAssign) {
  ::sip::optimal_control::Workspace workspace;
  constexpr int kWorkspaceSize = ::sip::optimal_control::Workspace::num_bytes(
      problem::kStateDim, problem::kControlDim, problem::kNumEdges,
      problem::kCDim, problem::kGDim, 0, ::sip::Settings{});
  std::array<unsigned char, kWorkspaceSize> workspace_bytes;
  ASSERT_EQ(::sip::optimal_control::Workspace::num_bytes(
                problem::kDimensions, problem::kTopology, problem::settings()),
            static_cast<int>(workspace_bytes.size()));
  workspace.mem_assign(problem::kDimensions, problem::kTopology,
                       problem::settings(), workspace_bytes.data());

  const auto output =
      problem::run_solver(problem::kDimensions, problem::kTopology, workspace);

  ASSERT_EQ(output.exit_status, ::sip::Status::SOLVED);
}

TEST(SimpleConstrainedLQR, Problem1WithReserve) {
  ::sip::optimal_control::Workspace workspace;
  workspace.reserve(problem::kDimensions, problem::kTopology,
                    problem::settings());

  const auto output =
      problem::run_solver(problem::kDimensions, problem::kTopology, workspace);

  ASSERT_EQ(output.exit_status, ::sip::Status::SOLVED);
  workspace.free(problem::kTopology);
}

} // namespace sip_examples
