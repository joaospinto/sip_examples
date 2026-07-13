#include "problem_definitions/ipopt/ipopt.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <dlfcn.h>
#include <ostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace sip_examples::problem_definitions::ipopt {
namespace {

using EvalF = bool (*)(int, double *, bool, double *, void *);
using EvalG = bool (*)(int, double *, bool, int, double *, void *);
using EvalGradF = bool (*)(int, double *, bool, double *, void *);
using EvalJacG = bool (*)(int, double *, bool, int, int, int *, int *, double *,
                          void *);
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

class Api {
public:
  explicit Api(const char *library_path)
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

  ~Api() { dlclose(handle_); }

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
  explicit Context(Model &model)
      : model(model), y(model.equality_dim(), 0.0),
        z(model.inequality_dim(), 0.0) {
    model.evaluate_values(model.initial_x());
    const auto &output = model.output();
    for (int i = 0; i < model.equality_dim(); ++i) {
      if (output.jacobian_c.indptr[i] != output.jacobian_c.indptr[i + 1]) {
        active_equalities.push_back(i);
      } else if (output.c[i] != 0.0) {
        throw std::runtime_error("constant equality constraint is infeasible");
      }
    }
    for (int i = 0; i < model.inequality_dim(); ++i) {
      if (output.jacobian_g.indptr[i] != output.jacobian_g.indptr[i + 1]) {
        active_inequalities.push_back(i);
      } else if (output.g[i] > 0.0) {
        throw std::runtime_error(
            "constant inequality constraint is infeasible");
      }
    }
    const auto &hessian = model.output().upper_hessian_lagrangian;
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

  void invalidate(bool new_x) {
    if (new_x) {
      values_current = false;
      first_derivatives_current = false;
    }
  }

  void ensure_values(const double *x) {
    if (!values_current) {
      model.evaluate_values(x);
      values_current = true;
    }
  }

  void ensure_first_derivatives(const double *x) {
    if (!first_derivatives_current) {
      model.evaluate_first_derivatives(x);
      values_current = true;
      first_derivatives_current = true;
    }
  }

  Model &model;
  std::vector<double> y;
  std::vector<double> z;
  std::vector<int> active_equalities;
  std::vector<int> active_inequalities;
  std::vector<double> hessian_with_objective;
  std::string error;
  int iterations = 0;
  int line_search_iterations = 0;
  bool values_current = false;
  bool first_derivatives_current = false;
};

auto evaluate_f(int, double *x, bool new_x, double *value, void *user_data)
    -> bool {
  auto &context = *static_cast<Context *>(user_data);
  return context.invoke([&] {
    context.invalidate(new_x);
    context.ensure_values(x);
    *value = context.model.output().f;
  });
}

auto evaluate_grad_f(int n, double *x, bool new_x, double *gradient,
                     void *user_data) -> bool {
  auto &context = *static_cast<Context *>(user_data);
  return context.invoke([&] {
    context.invalidate(new_x);
    context.ensure_first_derivatives(x);
    std::copy_n(context.model.output().gradient_f, n, gradient);
  });
}

auto evaluate_g(int, double *x, bool new_x, int, double *constraints,
                void *user_data) -> bool {
  auto &context = *static_cast<Context *>(user_data);
  return context.invoke([&] {
    context.invalidate(new_x);
    context.ensure_values(x);
    const auto &output = context.model.output();
    int offset = 0;
    for (int index : context.active_equalities) {
      constraints[offset++] = output.c[index];
    }
    for (int index : context.active_inequalities) {
      constraints[offset++] = output.g[index];
    }
  });
}

auto evaluate_jacobian(int, double *x, bool new_x, int, int, int *rows,
                       int *cols, double *values, void *user_data) -> bool {
  auto &context = *static_cast<Context *>(user_data);
  return context.invoke([&] {
    auto &output = context.model.output();
    if (values == nullptr) {
      int offset = 0;
      int active_constraint = 0;
      for (int constraint : context.active_equalities) {
        for (int index = output.jacobian_c.indptr[constraint];
             index < output.jacobian_c.indptr[constraint + 1]; ++index) {
          rows[offset] = active_constraint;
          cols[offset++] = output.jacobian_c.ind[index];
        }
        ++active_constraint;
      }
      for (int constraint : context.active_inequalities) {
        for (int index = output.jacobian_g.indptr[constraint];
             index < output.jacobian_g.indptr[constraint + 1]; ++index) {
          rows[offset] = active_constraint;
          cols[offset++] = output.jacobian_g.ind[index];
        }
        ++active_constraint;
      }
      return;
    }
    context.invalidate(new_x);
    context.ensure_first_derivatives(x);
    int offset = 0;
    for (int constraint : context.active_equalities) {
      const int begin = output.jacobian_c.indptr[constraint];
      const int count = output.jacobian_c.indptr[constraint + 1] - begin;
      std::copy_n(output.jacobian_c.data + begin, count, values + offset);
      offset += count;
    }
    for (int constraint : context.active_inequalities) {
      const int begin = output.jacobian_g.indptr[constraint];
      const int count = output.jacobian_g.indptr[constraint + 1] - begin;
      std::copy_n(output.jacobian_g.data + begin, count, values + offset);
      offset += count;
    }
  });
}

auto evaluate_hessian(int, double *x, bool, double objective_factor, int,
                      double *multipliers, bool, int, int *rows, int *cols,
                      double *values, void *user_data) -> bool {
  auto &context = *static_cast<Context *>(user_data);
  return context.invoke([&] {
    auto &hessian = context.model.output().upper_hessian_lagrangian;
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

    const int hessian_nnz = hessian.indptr[hessian.cols];
    std::fill(context.y.begin(), context.y.end(), 0.0);
    std::fill(context.z.begin(), context.z.end(), 0.0);
    if (objective_factor != 0.0) {
      int offset = 0;
      for (int index : context.active_equalities) {
        context.y[index] = multipliers[offset++] / objective_factor;
      }
      for (int index : context.active_inequalities) {
        context.z[index] = multipliers[offset++] / objective_factor;
      }
      context.model.evaluate_lagrangian_hessian(x, context.y.data(),
                                                context.z.data());
      for (int i = 0; i < hessian_nnz; ++i) {
        values[i] = objective_factor * hessian.data[i];
      }
      return;
    }

    int offset = 0;
    for (int index : context.active_equalities) {
      context.y[index] = multipliers[offset++];
    }
    for (int index : context.active_inequalities) {
      context.z[index] = multipliers[offset++];
    }
    context.model.evaluate_lagrangian_hessian(x, context.y.data(),
                                              context.z.data());
    std::copy_n(hessian.data, hessian_nnz,
                context.hessian_with_objective.data());
    std::fill(context.y.begin(), context.y.end(), 0.0);
    std::fill(context.z.begin(), context.z.end(), 0.0);
    context.model.evaluate_lagrangian_hessian(x, context.y.data(),
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

} // namespace

Model::~Model() = default;

auto configuration_from_environment() -> Configuration {
  const char *value = std::getenv("IPOPT_ABLATION");
  const std::string_view ablation =
      value == nullptr ? std::string_view("default") : std::string_view(value);
  Settings settings;
  if (ablation == "mehrotra") {
    settings.use_mehrotra_algorithm = true;
  } else if (ablation == "no_scaling") {
    settings.disable_nlp_scaling = true;
  } else if (ablation == "limited_memory") {
    settings.use_limited_memory_hessian = true;
  } else if (ablation == "accept_every_trial_step") {
    settings.accept_every_trial_step = true;
  } else if (ablation == "no_soc") {
    settings.disable_second_order_corrections = true;
  } else if (ablation == "no_watchdog") {
    settings.disable_watchdog = true;
  } else if (ablation == "trace") {
    settings.print_level = 5;
  } else if (ablation == "no_least_square_multipliers") {
    settings.disable_least_square_multiplier_initialization = true;
  } else if (ablation == "mu_based_bound_multipliers") {
    settings.use_mu_based_bound_multiplier_initialization = true;
  } else if (ablation == "sip_initialization") {
    settings.initial_mu = 1e-2;
    settings.use_mu_based_bound_multiplier_initialization = true;
    settings.disable_least_square_multiplier_initialization = true;
  } else if (ablation != "default") {
    throw std::invalid_argument("unknown IPOPT_ABLATION mode");
  }
  return {.settings = settings, .ablation = ablation};
}

auto solve(const char *ipopt_library_path, Model &model,
           const Settings &settings) -> Result {
  Context context(model);
  auto &output = model.output();
  const int x_dim = model.x_dim();
  const int y_dim = model.equality_dim();
  const int z_dim = model.inequality_dim();
  const int active_y_dim = static_cast<int>(context.active_equalities.size());
  const int active_z_dim = static_cast<int>(context.active_inequalities.size());
  const int constraint_dim = active_y_dim + active_z_dim;
  int jacobian_nnz = 0;
  for (int index : context.active_equalities) {
    jacobian_nnz +=
        output.jacobian_c.indptr[index + 1] - output.jacobian_c.indptr[index];
  }
  for (int index : context.active_inequalities) {
    jacobian_nnz +=
        output.jacobian_g.indptr[index + 1] - output.jacobian_g.indptr[index];
  }
  const int hessian_nnz = output.upper_hessian_lagrangian
                              .indptr[output.upper_hessian_lagrangian.cols];

  constexpr double infinity = 1e19;
  std::vector<double> x_lower(x_dim, -infinity);
  std::vector<double> x_upper(x_dim, infinity);
  std::vector<double> constraint_lower(constraint_dim, -infinity);
  std::vector<double> constraint_upper(constraint_dim, 0.0);
  std::fill_n(constraint_lower.data(), active_y_dim, 0.0);

  Api api(ipopt_library_path);
  void *ipopt_problem =
      api.create(x_dim, x_lower.data(), x_upper.data(), constraint_dim,
                 constraint_lower.data(), constraint_upper.data(), jacobian_nnz,
                 hessian_nnz, 0, evaluate_f, evaluate_g, evaluate_grad_f,
                 evaluate_jacobian, evaluate_hessian);
  if (ipopt_problem == nullptr) {
    throw std::runtime_error("CreateIpoptProblem failed");
  }

  constexpr double internal_tolerance = 1e-7;
  require_option(api.add_number(ipopt_problem, "tol", internal_tolerance),
                 "tol");
  require_option(
      api.add_number(ipopt_problem, "constr_viol_tol", internal_tolerance),
      "constr_viol_tol");
  require_option(
      api.add_number(ipopt_problem, "dual_inf_tol", internal_tolerance),
      "dual_inf_tol");
  require_option(
      api.add_number(ipopt_problem, "compl_inf_tol", internal_tolerance),
      "compl_inf_tol");
  require_option(api.add_integer(ipopt_problem, "acceptable_iter", 0),
                 "acceptable_iter");
  require_option(
      api.add_integer(ipopt_problem, "max_iter", settings.max_iterations),
      "max_iter");
  require_option(
      api.add_integer(ipopt_problem, "print_level", settings.print_level),
      "print_level");
  require_option(api.add_number(ipopt_problem, "mu_init", settings.initial_mu),
                 "mu_init");
  require_option(api.add_string(ipopt_problem, "sb", "yes"), "sb");
  require_option(api.add_number(ipopt_problem, "bound_relax_factor", 0.0),
                 "bound_relax_factor");
  require_option(api.set_intermediate(ipopt_problem, intermediate_callback),
                 "intermediate_callback");
  if (settings.use_mehrotra_algorithm) {
    require_option(api.add_string(ipopt_problem, "mehrotra_algorithm", "yes"),
                   "mehrotra_algorithm");
  }
  if (settings.disable_nlp_scaling) {
    require_option(api.add_string(ipopt_problem, "nlp_scaling_method", "none"),
                   "nlp_scaling_method");
  }
  if (settings.use_limited_memory_hessian) {
    require_option(api.add_string(ipopt_problem, "hessian_approximation",
                                  "limited-memory"),
                   "hessian_approximation");
  }
  if (settings.accept_every_trial_step) {
    require_option(
        api.add_string(ipopt_problem, "accept_every_trial_step", "yes"),
        "accept_every_trial_step");
  }
  if (settings.disable_second_order_corrections) {
    require_option(api.add_integer(ipopt_problem, "max_soc", 0), "max_soc");
  }
  if (settings.disable_watchdog) {
    require_option(
        api.add_integer(ipopt_problem, "watchdog_shortened_iter_trigger", 0),
        "watchdog_shortened_iter_trigger");
  }
  if (settings.use_mu_based_bound_multiplier_initialization) {
    require_option(api.add_string(ipopt_problem, "bound_mult_init_method",
                                  "mu-based"),
                   "bound_mult_init_method");
  }
  if (settings.disable_least_square_multiplier_initialization) {
    require_option(api.add_number(ipopt_problem, "constr_mult_init_max", 0.0),
                   "constr_mult_init_max");
  }

  std::vector<double> x(model.initial_x(), model.initial_x() + x_dim);
  std::vector<double> constraints(constraint_dim, 0.0);
  std::vector<double> multipliers(constraint_dim, 0.0);
  std::vector<double> x_lower_multipliers(x_dim, 0.0);
  std::vector<double> x_upper_multipliers(x_dim, 0.0);
  double objective = 0.0;
  const auto start = std::chrono::steady_clock::now();
  const int status =
      api.solve(ipopt_problem, x.data(), constraints.data(), &objective,
                multipliers.data(), x_lower_multipliers.data(),
                x_upper_multipliers.data(), &context);
  const auto elapsed = std::chrono::steady_clock::now() - start;
  api.free(ipopt_problem);
  if (!context.error.empty()) {
    throw std::runtime_error(context.error);
  }

  std::fill(context.y.begin(), context.y.end(), 0.0);
  std::fill(context.z.begin(), context.z.end(), 0.0);
  int multiplier_offset = 0;
  for (int index : context.active_equalities) {
    context.y[index] = multipliers[multiplier_offset++];
  }
  for (int index : context.active_inequalities) {
    context.z[index] = multipliers[multiplier_offset++];
  }
  model.evaluate_first_derivatives(x.data());
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

  const double duration = std::chrono::duration<double>(elapsed).count();
  const bool solved = (status == 0 || status == 1) && primal_residual <= 1e-6 &&
                      dual_residual <= 1e-6 && complementarity <= 1e-6;
  return {
      .ipopt_status = status,
      .num_iterations = context.iterations,
      .num_line_search_iterations = context.line_search_iterations,
      .max_primal_violation = primal_residual,
      .max_dual_violation = dual_residual,
      .max_complementarity = complementarity,
      .objective = objective,
      .solve_time = duration,
      .solved = solved,
  };
}

void print_result(std::ostream &stream, const Result &result) {
  stream << "ipopt_status=" << result.ipopt_status
         << " status=" << (result.solved ? "SOLVED" : "FAILED")
         << " iterations=" << result.num_iterations
         << " ls_iterations=" << result.num_line_search_iterations
         << " primal=" << result.max_primal_violation
         << " dual=" << result.max_dual_violation
         << " max_complementarity=" << result.max_complementarity
         << " objective=" << result.objective
         << " solve_time=" << result.solve_time << '\n';
}

} // namespace sip_examples::problem_definitions::ipopt
