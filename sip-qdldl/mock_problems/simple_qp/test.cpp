#include "problem_definitions/mock_problems/simple_qp/problem.hpp"
#include "sip/sip.hpp"
#include "sip_qdldl/sip_qdldl.hpp"

#include <array>
#include <functional>
#include <gtest/gtest.h>
#include <vector>

namespace sip_examples {
namespace problem = ::sip_examples::problem_definitions::simple_qp;

namespace {

void solve_problem(const sip::Mode mode, const bool use_mem_assign) {
  sip::Settings settings = problem::settings();
  settings.mode = mode;

  sip_qdldl::ModelCallbackOutput mco;
  mco.reserve(problem::kXDim, problem::kSDim, problem::kYDim,
              problem::kUpperHessianLagrangianNnz, problem::kJacobianCNnz,
              problem::kJacobianGNnz, true, true);
  problem::configure_qdldl_sparsity(mco);

  auto model_callback = [&mco](const sip::ModelCallbackInput &mci) -> void {
    if (!mci.new_x) {
      return;
    }
    problem::evaluate(mci, &mco.f, mco.gradient_f, mco.c,
                      mco.upper_hessian_lagrangian.data, mco.jacobian_c.data);
  };

  sip_qdldl::Workspace sip_qdldl_workspace;
  sip_qdldl_workspace.reserve(problem::kKktDim, problem::kQdldlKktNnz,
                              problem::kQdldlKktLNnz);

  const sip_qdldl::Settings sip_qdldl_settings{
      .permute_kkt_system = true,
      .kkt_pinv = problem::kKktPinv.data(),
  };

  auto callback_provider =
      sip_qdldl::CallbackProvider(sip_qdldl_settings, mco, sip_qdldl_workspace);

  const auto timeout_callback = []() { return false; };

  const auto ldlt_factor =
      [&callback_provider](const double *w, const double *r1, const double *r2,
                           const double *r3) -> bool {
    return callback_provider.factor(w, r1, r2, r3);
  };

  const auto ldlt_solve = [&callback_provider](const double *b,
                                               double *v) -> void {
    return callback_provider.solve(b, v);
  };

  const auto add_Kx_to_y =
      [&callback_provider](const double *w, const double *r1, const double *r2,
                           const double *r3, const double *x_x,
                           const double *x_y, const double *x_z, double *y_x,
                           double *y_y, double *y_z) -> void {
    return callback_provider.add_Kx_to_y(w, r1, r2, r3, x_x, x_y, x_z, y_x, y_y,
                                         y_z);
  };

  const auto add_Hx_to_y = [&callback_provider](const double *x,
                                                double *y) -> void {
    return callback_provider.add_Hx_to_y(x, y);
  };

  const auto add_Cx_to_y = [&callback_provider](const double *x,
                                                double *y) -> void {
    return callback_provider.add_Cx_to_y(x, y);
  };

  const auto add_CTx_to_y = [&callback_provider](const double *x,
                                                 double *y) -> void {
    return callback_provider.add_CTx_to_y(x, y);
  };

  const auto add_Gx_to_y = [&callback_provider](const double *x,
                                                double *y) -> void {
    return callback_provider.add_Gx_to_y(x, y);
  };

  const auto add_GTx_to_y = [&callback_provider](const double *x,
                                                 double *y) -> void {
    return callback_provider.add_GTx_to_y(x, y);
  };

  const auto get_f = [&mco]() -> double { return mco.f; };

  const auto get_grad_f = [&mco]() -> double * { return mco.gradient_f; };

  const auto get_c = [&mco]() -> double * { return mco.c; };

  const auto get_g = [&mco]() -> double * { return mco.g; };

  sip::Input input{
      .factor = std::cref(ldlt_factor),
      .solve = std::cref(ldlt_solve),
      .add_Kx_to_y = std::cref(add_Kx_to_y),
      .add_Hx_to_y = std::cref(add_Hx_to_y),
      .add_Cx_to_y = std::cref(add_Cx_to_y),
      .add_CTx_to_y = std::cref(add_CTx_to_y),
      .add_Gx_to_y = std::cref(add_Gx_to_y),
      .add_GTx_to_y = std::cref(add_GTx_to_y),
      .get_f = std::cref(get_f),
      .get_grad_f = std::cref(get_grad_f),
      .get_c = std::cref(get_c),
      .get_g = std::cref(get_g),
      .model_callback = std::cref(model_callback),
      .timeout_callback = std::cref(timeout_callback),
      .lower_bounds = problem::kLowerBounds.data(),
      .upper_bounds = problem::kUpperBounds.data(),
      .dimensions =
          {
              .x_dim = problem::kXDim,
              .s_dim = problem::kSDim,
              .y_dim = problem::kYDim,
          },
  };

  const int num_bound_sides = input.num_bound_sides();
  sip::Workspace workspace;
  std::vector<unsigned char> workspace_memory;
  if (use_mem_assign) {
    workspace_memory.resize(
        sip::Workspace::num_bytes(problem::kXDim, problem::kSDim,
                                  problem::kYDim, num_bound_sides, settings));
    EXPECT_EQ(workspace.mem_assign(problem::kXDim, problem::kSDim,
                                   problem::kYDim, num_bound_sides, settings,
                                   workspace_memory.data()),
              workspace_memory.size());
  } else {
    workspace.reserve(problem::kXDim, problem::kSDim, problem::kYDim,
                      num_bound_sides, settings);
  }
  problem::initialize(workspace, num_bound_sides);

  const auto output = solve(input, settings, workspace);

  EXPECT_EQ(output.exit_status, sip::Status::SOLVED);

  EXPECT_NEAR(workspace.vars.x[0], 0.3, 1e-6);
  EXPECT_NEAR(workspace.vars.x[1], 0.7, 1e-6);

  sip_qdldl_workspace.free();
  if (!use_mem_assign) {
    workspace.free();
  }
  mco.free();
}

} // namespace

TEST(SimpleQP, FromOSQPRepo) {
  constexpr std::array modes{
      sip::Mode::REGULARIZED_IPM,
      sip::Mode::PRIMAL_PROXIMAL_IPM,
      sip::Mode::PRIMAL_DUAL_PROXIMAL_IPM,
  };
  for (const sip::Mode mode : modes) {
    for (const bool use_mem_assign : {false, true}) {
      SCOPED_TRACE(static_cast<int>(mode));
      SCOPED_TRACE(use_mem_assign);
      solve_problem(mode, use_mem_assign);
    }
  }
}

} // namespace sip_examples
