#include GENERATED_HEADER

#include "problem_definitions/casadi_problems/common/flat_qdldl_runner.hpp"

#include <cstdlib>

#include <gtest/gtest.h>

namespace sip_examples::problem_definitions::casadi_problems {

TEST(CasadiFlatQdldl, SolvesGeneratedProblem) {
  auto settings = generated_problem::Problem::settings();
  settings.proximal.use_primal_center =
      std::getenv("SIP_CASADI_USE_PRIMAL_CENTER") != nullptr;
  settings.proximal.use_dual_center =
      std::getenv("SIP_CASADI_USE_DUAL_CENTER") != nullptr;
  settings.barrier.use_predictor_corrector =
      std::getenv("SIP_CASADI_USE_PREDICTOR_CORRECTOR") != nullptr;
  settings.barrier.initialize_primal_dual_variables =
      std::getenv("SIP_CASADI_INITIALIZE_PRIMAL_DUAL_VARIABLES") != nullptr;
  const bool print_logs =
      std::getenv("SIP_CASADI_PROBLEMS_PRINT_LOGS") != nullptr;
  if (print_logs) {
    enable_all_casadi_problem_logs(settings);
  }
  const auto result = run_flat_qdldl<generated_problem::Problem>(settings);
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
