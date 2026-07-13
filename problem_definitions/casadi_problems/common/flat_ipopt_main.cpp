#include GENERATED_HEADER

#include "problem_definitions/casadi_problems/common/flat_ipopt_model.hpp"

#include <exception>
#include <iostream>

namespace sip_examples::problem_definitions::casadi_problems {
namespace {

auto run(const char *ipopt_library_path) -> int {
  using Problem = generated_problem::Problem;
  const GeneratedFlatProblem generated{
      .spec = &Problem::flat_spec(),
      .make_work = Problem::make_flat_work,
      .evaluate_values = Problem::eval_flat_values,
      .evaluate = Problem::eval_flat_qdldl,
  };
  FlatIpoptModel model(generated);
  const auto result = ipopt::solve(ipopt_library_path, model, {});
  ipopt::print_result(std::cout, result);
  return result.solved ? 0 : 1;
}

} // namespace
} // namespace sip_examples::problem_definitions::casadi_problems

int main(int argc, char **argv) {
  if (argc != 2) {
    std::cerr << "usage: flat_ipopt_main IPOPT_LIBRARY\n";
    return 2;
  }
  try {
    return sip_examples::problem_definitions::casadi_problems::run(argv[1]);
  } catch (const std::exception &exception) {
    std::cerr << exception.what() << '\n';
    return 2;
  }
}
