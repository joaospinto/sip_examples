#include "problem_definitions/cutest_problems/cutest_problem.hpp"
#include "problem_definitions/ipopt/ipopt.hpp"

#include <cstdlib>
#include <exception>
#include <iostream>
#include <stdexcept>
#include <string_view>

namespace sip_examples::problem_definitions::cutest_problems {
namespace {

class CutestIpoptModel final : public ipopt::Model {
public:
  explicit CutestIpoptModel(CutestProblem &problem) : problem_(problem) {}

  int x_dim() const override { return problem_.x_dim(); }
  int equality_dim() const override { return problem_.equality_dim(); }
  int inequality_dim() const override { return problem_.inequality_dim(); }
  const double *initial_x() const override {
    return problem_.initial_x().data();
  }
  sip_qdldl::ModelCallbackOutput &output() override {
    return problem_.model_output();
  }
  void evaluate_values(const double *x) override {
    problem_.evaluate_values(x);
  }
  void evaluate_first_derivatives(const double *x) override {
    problem_.evaluate_objective_and_gradient(x);
    problem_.evaluate_constraints_and_jacobian(x);
  }
  void evaluate_lagrangian_hessian(const double *x, const double *y,
                                   const double *z) override {
    problem_.evaluate_lagrangian_hessian(x, y, z);
  }

private:
  CutestProblem &problem_;
};

auto run(const char *ipopt_library_path, const char *runtime_path,
         const char *problem_library_path, const char *outsdif_path) -> int {
  CutestProblem problem(runtime_path, problem_library_path, outsdif_path,
                        CutestProblem::SymbolicData::DerivativesOnly);
  CutestIpoptModel model(problem);
  ipopt::Settings settings;
  const std::string_view ablation = [] {
    const char *value = std::getenv("IPOPT_ABLATION");
    return value == nullptr ? std::string_view("default")
                            : std::string_view(value);
  }();
  if (ablation == "mehrotra") {
    settings.use_mehrotra_algorithm = true;
  } else if (ablation == "no_scaling") {
    settings.disable_nlp_scaling = true;
  } else if (ablation == "limited_memory") {
    settings.use_limited_memory_hessian = true;
  } else if (ablation != "default") {
    throw std::invalid_argument("unknown IPOPT_ABLATION mode");
  }

  const auto result = ipopt::solve(ipopt_library_path, model, settings);
  std::cout << "ablation=" << ablation << ' ';
  ipopt::print_result(std::cout, result);
  return result.solved ? 0 : 1;
}

} // namespace
} // namespace sip_examples::problem_definitions::cutest_problems

int main(int argc, char **argv) {
  if (argc != 6) {
    std::cerr << "usage: ipopt_native_runner IPOPT_LIBRARY CUTEST_RUNTIME "
                 "PROBLEM_LIBRARY OUTSDIF USE_QP_SETTINGS\n";
    return 2;
  }
  try {
    return sip_examples::problem_definitions::cutest_problems::run(
        argv[1], argv[2], argv[3], argv[4]);
  } catch (const std::exception &exception) {
    std::cerr << exception.what() << '\n';
    return 2;
  }
}
