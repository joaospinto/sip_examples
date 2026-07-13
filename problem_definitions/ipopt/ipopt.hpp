#pragma once

#include "sip_qdldl/sip_qdldl.hpp"

#include <iosfwd>
#include <string_view>

namespace sip_examples::problem_definitions::ipopt {

class Model {
public:
  virtual ~Model();

  virtual int x_dim() const = 0;
  virtual int equality_dim() const = 0;
  virtual int inequality_dim() const = 0;
  virtual const double *initial_x() const = 0;
  virtual sip_qdldl::ModelCallbackOutput &output() = 0;
  virtual void evaluate_values(const double *x) = 0;
  virtual void evaluate_first_derivatives(const double *x) = 0;
  virtual void evaluate_lagrangian_hessian(const double *x, const double *y,
                                           const double *z) = 0;
};

struct Settings {
  int max_iterations = 1000;
  int print_level = 0;
  bool use_mehrotra_algorithm = false;
  bool disable_nlp_scaling = false;
  bool use_limited_memory_hessian = false;
};

struct Configuration {
  Settings settings;
  std::string_view ablation;
};

struct Result {
  int ipopt_status;
  int num_iterations;
  int num_line_search_iterations;
  double max_primal_violation;
  double max_dual_violation;
  double max_complementarity;
  double objective;
  double solve_time;
  bool solved;
};

auto solve(const char *ipopt_library_path, Model &model,
           const Settings &settings) -> Result;
auto configuration_from_environment() -> Configuration;
void print_result(std::ostream &stream, const Result &result);

} // namespace sip_examples::problem_definitions::ipopt
