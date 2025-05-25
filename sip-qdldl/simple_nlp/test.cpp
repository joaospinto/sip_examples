#include "sip/sip.hpp"
#include "sip_qdldl/sip_qdldl.hpp"

#include <gtest/gtest.h>

namespace sip_examples {

TEST(SimpleNLP, Problem1) {
  sip::Settings settings{
      .max_kkt_violation = 1e-12,
      .max_merit_slope = 1e-24,
      .enable_elastics = true,
      .elastic_var_cost_coeff = 1e6,
  };
  sip::Workspace workspace;

  constexpr int x_dim = 2;
  constexpr int s_dim = 2;
  constexpr int y_dim = 0;
  constexpr int kkt_dim = x_dim + s_dim + y_dim;

  constexpr bool is_upper_hessian_lagrangian_transposed = false;
  constexpr bool is_jacobian_c_transposed = true;
  constexpr bool is_jacobian_g_transposed = true;

  constexpr int kkt_nnz = 9;
  constexpr int kkt_L_nnz = 11;

  workspace.reserve(x_dim, s_dim, y_dim);

  sip_qdldl::ModelCallbackOutput mco;
  constexpr int upper_hessian_lagrangian_nnz = 3;
  constexpr int jacobian_c_nnz = 0;
  constexpr int jacobian_g_nnz = 4;
  mco.reserve(x_dim, s_dim, y_dim, upper_hessian_lagrangian_nnz, jacobian_c_nnz,
              jacobian_g_nnz, is_jacobian_c_transposed,
              is_jacobian_g_transposed);

  // NOTE: only the upper triangle of upper_hessian_lagrangian should be filled.
  mco.upper_hessian_lagrangian.ind[0] = 0;
  mco.upper_hessian_lagrangian.ind[1] = 0;
  mco.upper_hessian_lagrangian.ind[2] = 1;
  mco.upper_hessian_lagrangian.indptr[0] = 0;
  mco.upper_hessian_lagrangian.indptr[1] = 1;
  mco.upper_hessian_lagrangian.indptr[2] = 3;

  mco.upper_hessian_lagrangian.rows = x_dim;
  mco.upper_hessian_lagrangian.cols = x_dim;

  mco.jacobian_c.rows = x_dim;
  mco.jacobian_c.cols = y_dim;
  mco.jacobian_c.indptr[0] = 0;
  mco.jacobian_c.is_transposed = is_jacobian_c_transposed;

  mco.jacobian_g.rows = x_dim;
  mco.jacobian_g.cols = s_dim;
  mco.jacobian_g.ind[0] = 0;
  mco.jacobian_g.ind[1] = 1;
  mco.jacobian_g.ind[2] = 0;
  mco.jacobian_g.ind[3] = 1;
  mco.jacobian_g.indptr[0] = 0;
  mco.jacobian_g.indptr[1] = 2;
  mco.jacobian_g.indptr[2] = 4;
  mco.jacobian_g.is_transposed = is_jacobian_g_transposed;

  auto model_callback = [&mco](const sip::ModelCallbackInput &mci) -> void {
    if (!mci.new_x) {
      return;
    }
    mco.f = mci.x[1] * (5.0 + mci.x[0]);

    mco.gradient_f[0] = mci.x[1];
    mco.gradient_f[1] = 5.0 + mci.x[0];

    // NOTE: a positive definite Hessian approximation is expected;
    //       the eigenvalues of the real Hessian are +-1,
    //       so we add (1 + 1e-6) to shift them.
    mco.upper_hessian_lagrangian.data[0] = 1.0 + 1e-6;
    mco.upper_hessian_lagrangian.data[1] = 1.0;
    mco.upper_hessian_lagrangian.data[2] = 1.0 + 1e-6;
    mco.upper_hessian_lagrangian.is_transposed =
        is_upper_hessian_lagrangian_transposed;

    // No equality constraints, so we don't set mco.c.

    mco.g[0] = 5.0 - mci.x[0] * mci.x[1];
    mco.g[1] = mci.x[0] * mci.x[0] + mci.x[1] * mci.x[1] - 20.0;

    mco.jacobian_g.data[0] = -mci.x[1];
    mco.jacobian_g.data[1] = -mci.x[0];
    mco.jacobian_g.data[2] = 2 * mci.x[0];
    mco.jacobian_g.data[3] = 2 * mci.x[1];
  };

  const auto kkt_pinv = std::array{3, 2, 1, 0};

  sip_qdldl::Workspace sip_qdldl_workspace;
  sip_qdldl_workspace.reserve(kkt_dim, kkt_nnz, kkt_L_nnz);

  const sip_qdldl::Settings sip_qdldl_settings{
      .permute_kkt_system = true,
      .kkt_pinv = kkt_pinv.data(),
  };

  auto callback_provider =
      sip_qdldl::CallbackProvider(sip_qdldl_settings, mco, sip_qdldl_workspace);

  const auto timeout_callback = []() { return false; };

  const auto ldlt_factor =
      [&callback_provider](const double *w, const double r1, const double r2,
                           const double r3) -> void {
    return callback_provider.factor(w, r1, r2, r3);
  };

  const auto ldlt_solve = [&callback_provider](const double *b,
                                               double *v) -> void {
    return callback_provider.solve(b, v);
  };

  const auto add_Kx_to_y =
      [&callback_provider](const double *w, const double r1, const double r2,
                           const double r3, const double *x_x,
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
              .x_dim = x_dim,
              .s_dim = s_dim,
              .y_dim = y_dim,
          },
  };

  for (int i = 0; i < x_dim; ++i) {
    workspace.vars.x[i] = 0.0;
  }

  for (int i = 0; i < s_dim; ++i) {
    workspace.vars.s[i] = 1.0;
    workspace.vars.z[i] = 1.0;
    workspace.vars.e[i] = 0.0;
  }

  for (int i = 0; i < y_dim; ++i) {
    workspace.vars.y[i] = 0.0;
  }

  const auto output = solve(input, settings, workspace);

  EXPECT_EQ(output.exit_status, sip::Status::SOLVED);

  EXPECT_NEAR(workspace.vars.x[0], -1.15747396, 1e-3);
  EXPECT_NEAR(workspace.vars.x[1], -4.31975162, 1e-3);

  sip_qdldl_workspace.free();
  workspace.free();
  mco.free();
}

} // namespace sip_examples
