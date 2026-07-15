#include "problem_definitions/mock_problems/simple_qp/problem.hpp"
#include "sip/sip.hpp"
#include "sip_qdldl/sip_qdldl.hpp"

#include <array>
#include <gtest/gtest.h>

namespace sip_examples {
namespace problem = ::sip_examples::problem_definitions::simple_qp;

template <bool kUsePreallocatedMemory, bool kUseProximalMethod>
void run_simple_qp() {
  sip::Settings settings = problem::settings();
  settings.proximal_qp.enabled = kUseProximalMethod;
  if constexpr (kUseProximalMethod) {
    settings.line_search.skip_line_search = true;
    settings.line_search.tau = 0.99;
  }
  sip::Workspace workspace;
  constexpr sip::Settings capacity_settings{
      .proximal_qp = {.enabled = kUseProximalMethod},
  };
  constexpr int kSipWorkspaceBytes = sip::Workspace::num_bytes(
      problem::kXDim, problem::kSDim, problem::kYDim, capacity_settings);
  alignas(double) std::array<unsigned char, kSipWorkspaceBytes>
      sip_workspace_memory;
  if constexpr (kUsePreallocatedMemory) {
    ASSERT_EQ(workspace.mem_assign(problem::kXDim, problem::kSDim,
                                   problem::kYDim, settings,
                                   sip_workspace_memory.data()),
              kSipWorkspaceBytes);
  } else {
    workspace.reserve(problem::kXDim, problem::kSDim, problem::kYDim, settings);
  }

  sip_qdldl::ModelCallbackOutput mco;
  constexpr int kModelOutputBytes = sip_qdldl::ModelCallbackOutput::num_bytes(
      problem::kXDim, problem::kSDim, problem::kYDim,
      problem::kUpperHessianLagrangianNnz, problem::kJacobianCNnz,
      problem::kJacobianGNnz, true, true);
  alignas(double) std::array<unsigned char, kModelOutputBytes>
      model_output_memory;
  if constexpr (kUsePreallocatedMemory) {
    ASSERT_EQ(mco.mem_assign(problem::kXDim, problem::kSDim, problem::kYDim,
                             problem::kUpperHessianLagrangianNnz,
                             problem::kJacobianCNnz, problem::kJacobianGNnz,
                             true, true, model_output_memory.data()),
              kModelOutputBytes);
  } else {
    mco.reserve(problem::kXDim, problem::kSDim, problem::kYDim,
                problem::kUpperHessianLagrangianNnz, problem::kJacobianCNnz,
                problem::kJacobianGNnz, true, true);
  }
  problem::configure_qdldl_sparsity(mco);

  auto model_callback = [&mco](const sip::ModelCallbackInput &mci) -> void {
    if (!mci.new_x) {
      return;
    }
    problem::evaluate(mci, &mco.f, mco.gradient_f, mco.c, mco.g,
                      mco.upper_hessian_lagrangian.data, mco.jacobian_c.data,
                      mco.jacobian_g.data);
  };

  sip_qdldl::Workspace sip_qdldl_workspace;
  constexpr int kQdldlWorkspaceBytes = sip_qdldl::Workspace::num_bytes(
      problem::kKktDim, problem::kQdldlKktNnz, problem::kQdldlKktLNnz);
  alignas(double) std::array<unsigned char, kQdldlWorkspaceBytes>
      qdldl_workspace_memory;
  if constexpr (kUsePreallocatedMemory) {
    ASSERT_EQ(sip_qdldl_workspace.mem_assign(
                  problem::kKktDim, problem::kQdldlKktNnz,
                  problem::kQdldlKktLNnz, qdldl_workspace_memory.data()),
              kQdldlWorkspaceBytes);
  } else {
    sip_qdldl_workspace.reserve(problem::kKktDim, problem::kQdldlKktNnz,
                                problem::kQdldlKktLNnz);
  }

  const sip_qdldl::Settings sip_qdldl_settings{
      .permute_kkt_system = true,
      .kkt_pinv = problem::kKktPinv.data(),
  };

  auto callback_provider =
      sip_qdldl::CallbackProvider(sip_qdldl_settings, mco, sip_qdldl_workspace);

  const auto timeout_callback = []() { return false; };

  const auto ldlt_factor =
      [&callback_provider](const double *w, const double r1, const double *r2,
                           const double *r3,
                           const double factorization_regularization) -> bool {
    return callback_provider.factor(w, r1, r2, r3,
                                    factorization_regularization);
  };

  const auto ldlt_solve = [&callback_provider](const double *b,
                                               double *v) -> void {
    return callback_provider.solve(b, v);
  };

  const auto add_Kx_to_y =
      [&callback_provider](const double *w, const double r1, const double *r2,
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
      .dimensions =
          {
              .x_dim = problem::kXDim,
              .s_dim = problem::kSDim,
              .y_dim = problem::kYDim,
          },
  };

  problem::initialize(workspace);

  const auto output = solve(input, settings, workspace);

  EXPECT_EQ(output.exit_status, sip::Status::SOLVED);

  EXPECT_NEAR(workspace.vars.x[0], 0.3, 1e-2);
  EXPECT_NEAR(workspace.vars.x[1], 0.7, 1e-2);

  if constexpr (!kUsePreallocatedMemory) {
    sip_qdldl_workspace.free();
    workspace.free();
    mco.free();
  }
}

TEST(SimpleQP, StandardWithReserve) { run_simple_qp<false, false>(); }

TEST(SimpleQP, StandardWithMemAssign) { run_simple_qp<true, false>(); }

TEST(SimpleQP, ProximalWithReserve) { run_simple_qp<false, true>(); }

TEST(SimpleQP, ProximalWithMemAssign) { run_simple_qp<true, true>(); }

} // namespace sip_examples
