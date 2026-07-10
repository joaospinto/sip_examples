#include GENERATED_HEADER
#include GENERATED_KKT_HEADER

#include "problem_definitions/casadi_problems/common/flat_slacg_runner.hpp"

#include <cstdlib>

#include <gtest/gtest.h>

namespace sip_examples::problem_definitions::casadi_problems {

TEST(CasadiFlatSlacg, SolvesGeneratedProblem) {
  auto settings = generated_problem::Problem::settings();
  if (std::getenv("SIP_CASADI_PROBLEMS_PRINT_LOGS") != nullptr) {
    enable_all_casadi_problem_logs(settings);
  }
  const auto result = run_flat_slacg<generated_problem::Problem>(settings);
  EXPECT_EQ(result.output.exit_status, sip::Status::SOLVED)
      << "iterations=" << result.output.num_iterations
      << " ls_iterations=" << result.output.num_ls_iterations
      << " primal=" << result.output.max_primal_violation
      << " dual=" << result.output.max_dual_violation;
}

} // namespace sip_examples::problem_definitions::casadi_problems
