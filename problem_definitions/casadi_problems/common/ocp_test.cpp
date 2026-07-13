#include GENERATED_HEADER

#include "problem_definitions/casadi_problems/common/ocp_runner.hpp"

#include <cstdlib>
#include <iostream>

#include <gtest/gtest.h>

namespace sip_examples::problem_definitions::casadi_problems {

TEST(CasadiOcp, SolvesGeneratedProblem) {
  auto configuration = settings_configuration_from_environment(
      generated_problem::Problem::settings());
  if (std::getenv("SIP_CASADI_PROBLEMS_PRINT_LOGS") != nullptr) {
    enable_all_casadi_problem_logs(configuration.settings);
  }
  const auto result =
      run_ocp<generated_problem::Problem>(configuration.settings);
  if (std::getenv("SIP_CASADI_PROBLEMS_PRINT_LOGS") != nullptr) {
    print_result(std::cout, configuration.ablation, result.output);
  }
  EXPECT_EQ(result.output.exit_status, sip::Status::SOLVED)
      << "iterations=" << result.output.num_iterations
      << " ls_iterations=" << result.output.num_ls_iterations
      << " primal=" << result.output.max_primal_violation
      << " dual=" << result.output.max_dual_violation;
}

} // namespace sip_examples::problem_definitions::casadi_problems
