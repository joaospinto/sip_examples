#include "problem_definitions/cutest_problems/cutest_problem.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <dlfcn.h>
#include <exception>
#include <iostream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace sip_examples::problem_definitions::cutest_problems {
namespace {

using EvalF = bool (*)(int, double *, bool, double *, void *);
using EvalG = bool (*)(int, double *, bool, int, double *, void *);
using EvalGradF = bool (*)(int, double *, bool, double *, void *);
using EvalJacG = bool (*)(int, double *, bool, int, int, int *, int *,
                          double *, void *);
using EvalH = bool (*)(int, double *, bool, double, int, double *, bool, int,
                       int *, int *, double *, void *);
using Intermediate = bool (*)(int, int, double, double, double, double, double,
                              double, double, double, int, void *);

template <typename Function>
auto load_symbol(void *handle, const char *name) -> Function {
  dlerror();
  void *symbol = dlsym(handle, name);
  if (const char *error = dlerror(); error != nullptr) {
    throw std::runtime_error("failed to load IPOPT symbol " +
                             std::string(name) + ": " + error);
  }
  return reinterpret_cast<Function>(symbol);
}

class IpoptApi {
public:
  explicit IpoptApi(const char *library_path)
      : handle_(dlopen(library_path, RTLD_NOW | RTLD_LOCAL)) {
    if (handle_ == nullptr) {
      throw std::runtime_error("failed to load IPOPT: " +
                               std::string(dlerror()));
    }
    create = load_symbol<Create>(handle_, "CreateIpoptProblem");
    free = load_symbol<Free>(handle_, "FreeIpoptProblem");
    add_string = load_symbol<AddString>(handle_, "AddIpoptStrOption");
    add_number = load_symbol<AddNumber>(handle_, "AddIpoptNumOption");
    add_integer = load_symbol<AddInteger>(handle_, "AddIpoptIntOption");
    set_intermediate =
        load_symbol<SetIntermediate>(handle_, "SetIntermediateCallback");
    solve = load_symbol<Solve>(handle_, "IpoptSolve");
  }

  ~IpoptApi() { dlclose(handle_); }

  using Create = void *(*)(int, double *, double *, int, double *, double *,
                            int, int, int, EvalF, EvalG, EvalGradF, EvalJacG,
                            EvalH);
  using Free = void (*)(void *);
  using AddString = bool (*)(void *, const char *, const char *);
  using AddNumber = bool (*)(void *, const char *, double);
  using AddInteger = bool (*)(void *, const char *, int);
  using SetIntermediate = bool (*)(void *, Intermediate);
  using Solve = int (*)(void *, double *, double *, double *, double *,
                        double *, double *, void *);

  Create create;
  Free free;
  AddString add_string;
  AddNumber add_number;
  AddInteger add_integer;
  SetIntermediate set_intermediate;
  Solve solve;

private:
  void *handle_;
};

struct Context {
  explicit Context(CutestProblem &problem)
      : problem(problem), y(problem.equality_dim(), 0.0),
        z(problem.inequality_dim(), 0.0) {
    const auto &hessian = problem.model_output().upper_hessian_lagrangian;
    hessian_with_objective.resize(hessian.indptr[hessian.cols]);
  }

  template <typename Function> auto invoke(Function &&function) -> bool {
    try {
      function();
      return true;
    } catch (const std::exception &exception) {
      error = exception.what();
      return false;
    }
  }

  CutestProblem &problem;
  std::vector<double> y;
  std::vector<double> z;
  std::vector<double> hessian_with_objective;
  std::string error;
  int iterations = 0;
  int line_search_iterations = 0;
};

auto evaluate_f(int, double *x, bool, double *value, void *user_data) -> bool {
  auto &context = *static_cast<Context *>(user_data);
  return context.invoke([&] {
    context.problem.evaluate_objective_and_gradient(x);
    *value = context.problem.model_output().f;
  });
}

auto evaluate_grad_f(int n, double *x, bool, double *gradient,
                     void *user_data) -> bool {
  auto &context = *static_cast<Context *>(user_data);
  return context.invoke([&] {
    context.problem.evaluate_objective_and_gradient(x);
    std::copy_n(context.problem.model_output().gradient_f, n, gradient);
  });
}

auto evaluate_g(int, double *x, bool, int, double *constraints,
                void *user_data) -> bool {
  auto &context = *static_cast<Context *>(user_data);
  return context.invoke([&] {
    context.problem.evaluate_constraints_and_jacobian(x);
    const auto &output = context.problem.model_output();
    std::copy_n(output.c, context.y.size(), constraints);
    std::copy_n(output.g, context.z.size(), constraints + context.y.size());
  });
}

auto evaluate_jacobian(int, double *x, bool, int, int, int *rows, int *cols,
                       double *values, void *user_data) -> bool {
  auto &context = *static_cast<Context *>(user_data);
  return context.invoke([&] {
    auto &output = context.problem.model_output();
    if (values == nullptr) {
      int offset = 0;
      for (int constraint = 0; constraint < output.jacobian_c.cols;
           ++constraint) {
        for (int index = output.jacobian_c.indptr[constraint];
             index < output.jacobian_c.indptr[constraint + 1]; ++index) {
          rows[offset] = constraint;
          cols[offset++] = output.jacobian_c.ind[index];
        }
      }
      for (int constraint = 0; constraint < output.jacobian_g.cols;
           ++constraint) {
        for (int index = output.jacobian_g.indptr[constraint];
             index < output.jacobian_g.indptr[constraint + 1]; ++index) {
          rows[offset] = static_cast<int>(context.y.size()) + constraint;
          cols[offset++] = output.jacobian_g.ind[index];
        }
      }
      return;
    }
    context.problem.evaluate_constraints_and_jacobian(x);
    const int equality_nnz =
        output.jacobian_c.indptr[output.jacobian_c.cols];
    const int inequality_nnz =
        output.jacobian_g.indptr[output.jacobian_g.cols];
    std::copy_n(output.jacobian_c.data, equality_nnz, values);
    std::copy_n(output.jacobian_g.data, inequality_nnz, values + equality_nnz);
  });
}

auto evaluate_hessian(int, double *x, bool, double objective_factor, int,
                      double *multipliers, bool, int, int *rows, int *cols,
                      double *values, void *user_data) -> bool {
  auto &context = *static_cast<Context *>(user_data);
  return context.invoke([&] {
    auto &hessian = context.problem.model_output().upper_hessian_lagrangian;
    if (values == nullptr) {
      int offset = 0;
      for (int col = 0; col < hessian.cols; ++col) {
        for (int index = hessian.indptr[col]; index < hessian.indptr[col + 1];
             ++index) {
          rows[offset] = col;
          cols[offset++] = hessian.ind[index];
        }
      }
      return;
    }

    const int y_dim = static_cast<int>(context.y.size());
    const int z_dim = static_cast<int>(context.z.size());
    const int hessian_nnz = hessian.indptr[hessian.cols];
    if (objective_factor != 0.0) {
      for (int i = 0; i < y_dim; ++i) {
        context.y[i] = multipliers[i] / objective_factor;
      }
      for (int i = 0; i < z_dim; ++i) {
        context.z[i] = multipliers[y_dim + i] / objective_factor;
      }
      context.problem.evaluate_lagrangian_hessian(x, context.y.data(),
                                                  context.z.data());
      for (int i = 0; i < hessian_nnz; ++i) {
        values[i] = objective_factor * hessian.data[i];
      }
      return;
    }

    if (y_dim > 0) {
      std::copy_n(multipliers, y_dim, context.y.data());
    }
    if (z_dim > 0) {
      std::copy_n(multipliers + y_dim, z_dim, context.z.data());
    context.problem.evaluate_lagrangian_hessian(x, context.y.data(),
                                                context.z.data());
    std::copy_n(hessian.data, hessian_nnz,
                context.hessian_with_objective.data());
    std::fill(context.y.begin(), context.y.end(), 0.0);
    std::fill(context.z.begin(), context.z.end(), 0.0);
    context.problem.evaluate_lagrangian_hessian(x, context.y.data(),
                                                context.z.data());
    for (int i = 0; i < hessian_nnz; ++i) {
      values[i] = context.hessian_with_objective[i] - hessian.data[i];
    }
  });
}

auto intermediate_callback(int, int iteration, double, double, double, double,
                           double, double, double, double,
                           int line_search_trials, void *user_data) -> bool {
  auto &context = *static_cast<Context *>(user_data);
  context.iterations = std::max(context.iterations, iteration);
  context.line_search_iterations += line_search_trials;
  return true;
}

void require_option(bool accepted, const char *name) {
  if (!accepted) {
    throw std::runtime_error("IPOPT rejected option " + std::string(name));
  }
}

auto run(const char *ipopt_library_path, const char *runtime_path,
         const char *problem_library_path, const char *outsdif_path) -> int {
  CutestProblem problem(runtime_path, problem_library_path, outsdif_path,
                        CutestProblem::SymbolicData::DerivativesOnly);
  Context context(problem);
  auto &output = problem.model_output();
  const int x_dim = problem.x_dim();
  const int y_dim = problem.equality_dim();
  const int z_dim = problem.inequality_dim();
  const int constraint_dim = y_dim + z_dim;
  const int jacobian_nnz = output.jacobian_c.indptr[output.jacobian_c.cols] +
                           output.jacobian_g.indptr[output.jacobian_g.cols];
  const int hessian_nnz =
      output.upper_hessian_lagrangian
          .indptr[output.upper_hessian_lagrangian.cols];

  constexpr double infinity = 1e19;
  std::vector<double> x_lower(x_dim, -infinity);
  std::vector<double> x_upper(x_dim, infinity);
  std::vector<double> constraint_lower(constraint_dim, -infinity);
  std::vector<double> constraint_upper(constraint_dim, 0.0);
  std::fill_n(constraint_lower.data(), y_dim, 0.0);

  IpoptApi api(ipopt_library_path);
  void *ipopt_problem =
      api.create(x_dim, x_lower.data(), x_upper.data(), constraint_dim,
                 constraint_lower.data(), constraint_upper.data(),
                 jacobian_nnz, hessian_nnz, 0, evaluate_f, evaluate_g,
                 evaluate_grad_f, evaluate_jacobian, evaluate_hessian);
  if (ipopt_problem == nullptr) {
    throw std::runtime_error("CreateIpoptProblem failed");
  }

  constexpr double internal_tolerance = 1e-7;
  require_option(api.add_number(ipopt_problem, "tol", internal_tolerance),
                 "tol");
  require_option(api.add_number(ipopt_problem, "constr_viol_tol",
                                internal_tolerance),
                 "constr_viol_tol");
  require_option(api.add_number(ipopt_problem, "dual_inf_tol",
                                internal_tolerance),
                 "dual_inf_tol");
  require_option(api.add_number(ipopt_problem, "compl_inf_tol",
                                internal_tolerance),
                 "compl_inf_tol");
  require_option(api.add_integer(ipopt_problem, "acceptable_iter", 0),
                 "acceptable_iter");
  require_option(api.add_integer(ipopt_problem, "max_iter", 1000),
                 "max_iter");
  require_option(api.add_integer(ipopt_problem, "print_level", 0),
                 "print_level");
  require_option(api.add_string(ipopt_problem, "sb", "yes"), "sb");
  require_option(api.add_number(ipopt_problem, "bound_relax_factor", 0.0),
                 "bound_relax_factor");
  require_option(api.set_intermediate(ipopt_problem, intermediate_callback),
                 "intermediate_callback");

  const std::string_view ablation = [] {
    const char *value = std::getenv("IPOPT_ABLATION");
    return value == nullptr ? std::string_view("default")
                            : std::string_view(value);
  }();
  if (ablation == "mehrotra") {
    require_option(
        api.add_string(ipopt_problem, "mehrotra_algorithm", "yes"),
        "mehrotra_algorithm");
  } else if (ablation == "no_scaling") {
    require_option(api.add_string(ipopt_problem, "nlp_scaling_method", "none"),
                   "nlp_scaling_method");
  } else if (ablation == "limited_memory") {
    require_option(api.add_string(ipopt_problem, "hessian_approximation",
                                  "limited-memory"),
                   "hessian_approximation");
  } else if (ablation != "default") {
    api.free(ipopt_problem);
    throw std::invalid_argument("unknown IPOPT_ABLATION mode");
  }

  std::vector<double> x = problem.initial_x();
  std::vector<double> constraints(constraint_dim, 0.0);
  std::vector<double> multipliers(constraint_dim, 0.0);
  std::vector<double> x_lower_multipliers(x_dim, 0.0);
  std::vector<double> x_upper_multipliers(x_dim, 0.0);
  double objective = 0.0;
  const auto start = std::chrono::steady_clock::now();
  const int status = api.solve(
      ipopt_problem, x.data(), constraints.data(), &objective,
      multipliers.data(), x_lower_multipliers.data(),
      x_upper_multipliers.data(), &context);
  const auto elapsed = std::chrono::steady_clock::now() - start;
  api.free(ipopt_problem);
  if (!context.error.empty()) {
    throw std::runtime_error(context.error);
  }

  if (y_dim > 0) {
    std::copy_n(multipliers.data(), y_dim, context.y.data());
  }
  if (z_dim > 0) {
    std::copy_n(multipliers.data() + y_dim, z_dim, context.z.data());
  }
  problem.evaluate(x.data(), context.y.data(), context.z.data());
  double primal_residual = 0.0;
  double complementarity = 0.0;
  for (int i = 0; i < y_dim; ++i) {
    primal_residual = std::max(primal_residual, std::fabs(output.c[i]));
  }
  for (int i = 0; i < z_dim; ++i) {
    primal_residual = std::max(primal_residual, output.g[i]);
    complementarity =
        std::max(complementarity, std::fabs(output.g[i] * context.z[i]));
  }
  std::vector<double> lagrangian_gradient(output.gradient_f,
                                          output.gradient_f + x_dim);
  sip_qdldl::add_ATx_to_y(output.jacobian_c, context.y.data(),
                          lagrangian_gradient.data());
  sip_qdldl::add_ATx_to_y(output.jacobian_g, context.z.data(),
                          lagrangian_gradient.data());
  double dual_residual = 0.0;
  for (double value : lagrangian_gradient) {
    dual_residual = std::max(dual_residual, std::fabs(value));
  }

  const double duration =
      std::chrono::duration<double>(elapsed).count();
  const bool solved =
      (status == 0 || status == 1) && primal_residual <= 1e-6 &&
      dual_residual <= 1e-6 && complementarity <= 1e-6;
  std::cout << "ablation=" << ablation << " ipopt_status=" << status
            << " status=" << (solved ? "SOLVED" : "FAILED")
            << " iterations=" << context.iterations
            << " ls_iterations=" << context.line_search_iterations
            << " primal=" << primal_residual << " dual=" << dual_residual
            << " max_complementarity=" << complementarity
            << " objective=" << objective << " solve_time=" << duration
            << '\n';
  return solved ? 0 : 1;
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
