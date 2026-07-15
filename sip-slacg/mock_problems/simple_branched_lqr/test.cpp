#include "sip-slacg/mock_problems/simple_branched_lqr/kkt_codegen.hpp"

#include "problem_definitions/mock_problems/simple_branched_lqr/problem.hpp"
#include "sip-slacg/helpers/helpers.hpp"
#include "sip/sip.hpp"

#include <array>
#include <gtest/gtest.h>

namespace sip_examples {
namespace problem = ::sip_examples::problem_definitions::simple_branched_lqr;
namespace {

struct LDLTCallbackProvider {
  double *LT_data;
  double *D_diag;
  double *border_solution;
  double *border_factor;

  auto factor(const double *upper_H_data, const double *C_data,
              const double *G_data, const double *w, const double r1,
              const double *r2, const double *r3) -> bool {
    return ldlt_factor(upper_H_data, C_data, G_data, w, r1, r2, r3, LT_data,
                       D_diag, border_solution, border_factor);
  }

  void solve(const double *b, double *v) {
    ldlt_solve(LT_data, D_diag, border_solution, border_factor, b, v);
  }

  void reserve() {
    LT_data = new double[L_nnz];
    D_diag = new double[core_dim];
    border_solution = new double[border_solution_size];
    border_factor = new double[border_factor_size];
  }

  void free() {
    delete[] LT_data;
    delete[] D_diag;
    delete[] border_solution;
    delete[] border_factor;
  }
};

TEST(SimpleBranchedLQR, SLACG) {
  ModelCallbackOutput mco;
  mco.reserve(problem::kXDim, problem::kSDim, problem::kYDim,
              problem::kUpperHessianLagrangianNnz, problem::kJacobianCNnz,
              problem::kJacobianGNnz);

  const auto model_callback = [&mco](const sip::ModelCallbackInput &mci) {
    if (!mci.new_x) {
      return;
    }
    problem::evaluate_slacg(mci, &mco.f, mco.gradient_f, mco.c,
                            mco.upper_hessian_lagrangian, mco.jacobian_c);
  };

  LDLTCallbackProvider callback_provider;
  callback_provider.reserve();

  std::array<double, problem::kYDim> factor_r2;
  std::array<double, problem::kSDim> factor_r3;

  const auto factor = [&callback_provider, &mco, &factor_r2,
                       &factor_r3](const double *w, const double r1,
                                   const double *r2, const double *r3,
                                   const double factorization_regularization) {
    for (int i = 0; i < problem::kYDim; ++i) {
      factor_r2[i] = r2[i] + factorization_regularization;
    }
    for (int i = 0; i < problem::kSDim; ++i) {
      factor_r3[i] = r3[i] + factorization_regularization;
    }
    return callback_provider.factor(
        mco.upper_hessian_lagrangian, mco.jacobian_c, mco.jacobian_g, w,
        r1 + factorization_regularization, factor_r2.data(), factor_r3.data());
  };
  const auto solve = [&callback_provider](const double *b, double *v) {
    callback_provider.solve(b, v);
  };
  const auto add_Kx_to_y =
      [&mco](const double *w, const double r1, const double *r2,
             const double *r3, const double *x_x, const double *x_y,
             const double *x_z, double *y_x, double *y_y, double *y_z) {
        ::sip_examples::add_Kx_to_y(mco.upper_hessian_lagrangian,
                                    mco.jacobian_c, mco.jacobian_g, w, r1, r2,
                                    r3, x_x, x_y, x_z, y_x, y_y, y_z);
      };
  const auto add_Hx_to_y = [&mco](const double *x, double *y) {
    add_upper_symmetric_Hx_to_y(mco.upper_hessian_lagrangian, x, y);
  };
  const auto add_Cx_to_y = [&mco](const double *x, double *y) {
    ::sip_examples::add_Cx_to_y(mco.jacobian_c, x, y);
  };
  const auto add_CTx_to_y = [&mco](const double *x, double *y) {
    ::sip_examples::add_CTx_to_y(mco.jacobian_c, x, y);
  };
  const auto add_Gx_to_y = [&mco](const double *x, double *y) {
    ::sip_examples::add_Gx_to_y(mco.jacobian_g, x, y);
  };
  const auto add_GTx_to_y = [&mco](const double *x, double *y) {
    ::sip_examples::add_GTx_to_y(mco.jacobian_g, x, y);
  };
  const auto get_f = [&mco]() { return mco.f; };
  const auto get_grad_f = [&mco]() { return mco.gradient_f; };
  const auto get_c = [&mco]() { return mco.c; };
  const auto get_g = [&mco]() { return mco.g; };
  const auto timeout_callback = []() { return false; };

  const sip::Input input{
      .factor = std::cref(factor),
      .solve = std::cref(solve),
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

  const sip::Settings settings = problem::settings();
  sip::Workspace workspace;
  workspace.reserve(problem::kXDim, problem::kSDim, problem::kYDim, settings);
  problem::initialize(workspace);
  const auto output = sip::solve(input, settings, workspace);

  ASSERT_EQ(output.exit_status, sip::Status::SOLVED);
  for (int i = 0; i < problem::kXDim; ++i) {
    EXPECT_NEAR(workspace.vars.x[i], problem::kSolution[i], 1e-8);
  }

  workspace.free();
  callback_provider.free();
  mco.free();
}

} // namespace
} // namespace sip_examples
