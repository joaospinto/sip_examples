#include "problem_definitions/mock_problems/simple_constrained_lqr/problem.hpp"

#include <array>
#include <vector>

#include <gtest/gtest.h>

namespace sip_examples {
namespace problem = ::sip_examples::problem_definitions::simple_constrained_lqr;
TEST(SimpleConstrainedLQR, Problem1WithMemAssign) {
  ::sip::optimal_control::Workspace workspace;
  std::vector<unsigned char> workspace_bytes(
      ::sip::optimal_control::Workspace::num_bytes(
          problem::kDimensions, problem::kTopology, problem::kNumBoundSides,
          problem::settings()));
  workspace.mem_assign(problem::kDimensions, problem::kTopology,
                       problem::kNumBoundSides, problem::settings(),
                       workspace_bytes.data());

  const auto output =
      problem::run_solver(problem::kDimensions, problem::kTopology, workspace);

  ASSERT_EQ(output.exit_status, ::sip::Status::SOLVED);
}

TEST(SimpleConstrainedLQR, Problem1WithReserve) {
  ::sip::optimal_control::Workspace workspace;
  workspace.reserve(problem::kDimensions, problem::kTopology,
                    problem::kNumBoundSides, problem::settings());

  const auto output =
      problem::run_solver(problem::kDimensions, problem::kTopology, workspace);

  ASSERT_EQ(output.exit_status, ::sip::Status::SOLVED);
  workspace.free(problem::kTopology);
}

} // namespace sip_examples
