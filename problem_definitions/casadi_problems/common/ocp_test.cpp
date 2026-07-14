#include GENERATED_HEADER

#include "problem_definitions/casadi_problems/common/ocp_runner.hpp"

#include <cstdlib>

#include <gtest/gtest.h>

namespace sip_examples::problem_definitions::casadi_problems {

TEST(CasadiOcp, SolvesGeneratedProblem) {
  auto settings = generated_problem::Problem::settings();
  const bool print_logs =
      std::getenv("SIP_CASADI_PROBLEMS_PRINT_LOGS") != nullptr;
  if (print_logs) {
    enable_all_casadi_problem_logs(settings);
  }
  const auto result = run_ocp<generated_problem::Problem>(settings);
  if (print_logs) {
    print_casadi_problem_summary(result.output);
  }
  EXPECT_EQ(result.output.exit_status, sip::Status::SOLVED)
      << "iterations=" << result.output.num_iterations
      << " ls_iterations=" << result.output.num_ls_iterations
      << " primal=" << result.output.max_primal_violation
      << " dual=" << result.output.max_dual_violation;
}

} // namespace sip_examples::problem_definitions::casadi_problems
