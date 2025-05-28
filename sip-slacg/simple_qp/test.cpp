#include "sip-slacg/simple_qp/kkt_codegen.hpp"

#include "sip-slacg/helpers/helpers.hpp"
#include "sip/sip.hpp"
#include <gtest/gtest.h>

namespace sip_examples {

struct LDLTCallbackProvider {
  double *LT_data;
  double *D_diag;

  void ldlt_factor(const double *upper_H_data, const double *C_data,
                   const double *G_data, const double *w, const double r1,
                   const double r2, const double r3) {
    return ::sip_examples::ldlt_factor(upper_H_data, C_data, G_data, w, r1, r2,
                                       r3, LT_data, D_diag);
  }

  void ldlt_solve(const double *b, double *v) {
    return ::sip_examples::ldlt_solve(LT_data, D_diag, b, v);
  }

  auto reserve(int L_nnz, int kkt_dim) -> void {
    LT_data = new double[L_nnz];
    D_diag = new double[kkt_dim];
  }

  auto free() -> void {
    delete[] LT_data;
    delete[] D_diag;
  }
};

TEST(SimpleQP, FromOSQPRepo) {
  ModelCallbackOutput mco;
  constexpr int upper_hessian_lagrangian_nnz = 3;
  constexpr int jacobian_c_nnz = 2;
  constexpr int jacobian_g_nnz = 4;
  constexpr int L_nnz = 7;
  mco.reserve(x_dim, z_dim, y_dim, upper_hessian_lagrangian_nnz, jacobian_c_nnz,
              jacobian_g_nnz);

  const auto model_callback = [&mco](const sip::ModelCallbackInput &mci) {
    if (!mci.new_x) {
      return;
    }
    mco.f = 0.5 * (4.0 * mci.x[0] * mci.x[0] + 2.0 * mci.x[0] * mci.x[1] +
                   2.0 * mci.x[1] * mci.x[1]) +
            mci.x[0] + mci.x[1];

    mco.gradient_f[0] = 4.0 * mci.x[0] + 1.0 * mci.x[1] + 1.0;
    mco.gradient_f[1] = 1.0 * mci.x[0] + 2.0 * mci.x[1] + 1.0;

    // NOTE: only the upper triangle should be filled.
    mco.upper_hessian_lagrangian[0] = 4.0;
    mco.upper_hessian_lagrangian[1] = 1.0;
    mco.upper_hessian_lagrangian[2] = 2.0;

    mco.c[0] = mci.x[0] + mci.x[1] - 1.0;

    mco.jacobian_c[0] = 1.0;
    mco.jacobian_c[1] = 1.0;

    mco.g[0] = mci.x[0] - 0.7;
    mco.g[1] = -mci.x[0] - 0.0;
    mco.g[2] = mci.x[1] - 0.7;
    mco.g[3] = -mci.x[1] - 0.0;

    mco.jacobian_g[0] = 1.0;
    mco.jacobian_g[1] = -1.0;
    mco.jacobian_g[2] = 1.0;
    mco.jacobian_g[3] = -1.0;
  };

  LDLTCallbackProvider ldlt_callback_provider;
  constexpr int kkt_dim = x_dim + z_dim + y_dim;
  ldlt_callback_provider.reserve(L_nnz, kkt_dim);

  const auto timeout_callback = []() { return false; };

  const auto factor = [&ldlt_callback_provider,
                       &mco](const double *w, const double r1, const double r2,
                             const double r3) {
    return ldlt_callback_provider.ldlt_factor(mco.upper_hessian_lagrangian,
                                              mco.jacobian_c, mco.jacobian_g, w,
                                              r1, r2, r3);
  };
  const auto solve = [&ldlt_callback_provider](const double *b, double *v) {
    return ldlt_callback_provider.ldlt_solve(b, v);
  };

  const auto _add_Kx_to_y =
      [&mco](const double *w, const double r1, const double r2, const double r3,
             const double *x_x, const double *x_y, const double *x_z,
             double *y_x, double *y_y, double *y_z) -> void {
    return add_Kx_to_y(mco.upper_hessian_lagrangian, mco.jacobian_c,
                       mco.jacobian_g, w, r1, r2, r3, x_x, x_y, x_z, y_x, y_y,
                       y_z);
  };

  const auto _add_Hx_to_y = [&mco](const double *x, double *y) -> void {
    return add_upper_symmetric_Hx_to_y(mco.upper_hessian_lagrangian, x, y);
  };

  const auto _add_Cx_to_y = [&mco](const double *x, double *y) -> void {
    return add_Cx_to_y(mco.jacobian_c, x, y);
  };

  const auto _add_CTx_to_y = [&mco](const double *x, double *y) -> void {
    return add_CTx_to_y(mco.jacobian_c, x, y);
  };

  const auto _add_Gx_to_y = [&mco](const double *x, double *y) -> void {
    return add_Gx_to_y(mco.jacobian_g, x, y);
  };

  const auto _add_GTx_to_y = [&mco](const double *x, double *y) -> void {
    return add_GTx_to_y(mco.jacobian_g, x, y);
  };

  const auto get_f = [&mco]() -> double { return mco.f; };

  const auto get_grad_f = [&mco]() -> double * { return mco.gradient_f; };

  const auto get_c = [&mco]() -> double * { return mco.c; };

  const auto get_g = [&mco]() -> double * { return mco.g; };

  sip::Input input{
      .factor = std::cref(factor),
      .solve = std::cref(solve),
      .add_Kx_to_y = std::cref(_add_Kx_to_y),
      .add_Hx_to_y = std::cref(_add_Hx_to_y),
      .add_Cx_to_y = std::cref(_add_Cx_to_y),
      .add_CTx_to_y = std::cref(_add_CTx_to_y),
      .add_Gx_to_y = std::cref(_add_Gx_to_y),
      .add_GTx_to_y = std::cref(_add_GTx_to_y),
      .get_f = std::cref(get_f),
      .get_grad_f = std::cref(get_grad_f),
      .get_c = std::cref(get_c),
      .get_g = std::cref(get_g),
      .model_callback = std::cref(model_callback),
      .timeout_callback = std::cref(timeout_callback),
      .dimensions =
          {
              .x_dim = x_dim,
              .s_dim = z_dim,
              .y_dim = y_dim,
          },
  };

  sip::Settings settings{.max_kkt_violation = 1e-12,
                         .max_merit_slope = 1e-24,
                         .enable_elastics = true,
                         .elastic_var_cost_coeff = 1e6};

  sip::Workspace workspace;
  workspace.reserve(x_dim, z_dim, y_dim);

  for (int i = 0; i < x_dim; ++i) {
    workspace.vars.x[i] = 0.0;
  }

  for (int i = 0; i < z_dim; ++i) {
    workspace.vars.s[i] = 1.0;
    workspace.vars.z[i] = 1.0;
    workspace.vars.e[i] = 0.0;
  }

  for (int i = 0; i < y_dim; ++i) {
    workspace.vars.y[i] = 0.0;
  }

  const auto output = sip::solve(input, settings, workspace);

  EXPECT_EQ(output.exit_status, sip::Status::SOLVED);

  EXPECT_NEAR(workspace.vars.x[0], 0.3, 1e-2);
  EXPECT_NEAR(workspace.vars.x[1], 0.7, 1e-2);

  workspace.free();
  ldlt_callback_provider.free();
  mco.free();
}

} // namespace sip_examples
