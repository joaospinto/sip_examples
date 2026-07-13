#include "problem_definitions/mock_problems/simple_constrained_lqr/problem.hpp"

#include <array>

#include <gtest/gtest.h>

namespace sip_examples {
namespace problem = ::sip_examples::problem_definitions::simple_constrained_lqr;
namespace {

constexpr int kFilterCapacity =
    ::sip::FilterWorkspace::required_capacity(::sip::Settings{});

} // namespace

TEST(SimpleConstrainedLQR, Problem1WithMemAssign) {
  ::sip::optimal_control::Workspace workspace;
  constexpr int kWorkspaceSize = ::sip::optimal_control::Workspace::num_bytes(
      problem::kStateDim, problem::kControlDim, problem::kNumEdges,
      problem::kCDim, problem::kGDim, 0, kFilterCapacity);
  std::array<unsigned char, kWorkspaceSize> workspace_bytes;
  ASSERT_EQ(kFilterCapacity,
            ::sip::FilterWorkspace::required_capacity(problem::settings()));
  workspace.mem_assign(problem::kDimensions, problem::kTopology,
                       kFilterCapacity, workspace_bytes.data());

  const auto output =
      problem::run_solver(problem::kDimensions, problem::kTopology, workspace);

  ASSERT_EQ(output.exit_status, ::sip::Status::SOLVED);
}

TEST(SimpleConstrainedLQR, Problem1WithReserve) {
  ::sip::optimal_control::Workspace workspace;
  workspace.reserve(problem::kDimensions, problem::kTopology, kFilterCapacity);

  const auto output =
      problem::run_solver(problem::kDimensions, problem::kTopology, workspace);

  ASSERT_EQ(output.exit_status, ::sip::Status::SOLVED);
  workspace.free(problem::kTopology);
}

} // namespace sip_examples
