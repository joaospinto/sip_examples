#include "sip-slacg/simple_qp/kkt_codegen.hpp"

#include "sip/sip.hpp"
#include <gtest/gtest.h>

namespace sip_examples {

void _model_callback(const sip::ModelCallbackInput &mci,
                     sip::ModelCallbackOutput &mco) {
  if (!mci.new_x) {
    return;
  }
  mco.f = 0.5 * (4.0 * mci.x[0] * mci.x[0] + 2.0 * mci.x[0] * mci.x[1] +
                 2.0 * mci.x[1] * mci.x[1]) +
          mci.x[0] + mci.x[1];

  mco.gradient_f[0] = 4.0 * mci.x[0] + 1.0 * mci.x[1] + 1.0;
  mco.gradient_f[1] = 1.0 * mci.x[0] + 2.0 * mci.x[1] + 1.0;

  // NOTE: only the upper triangle should be filled.
  mco.upper_hessian_lagrangian.rows = x_dim;
  mco.upper_hessian_lagrangian.cols = x_dim;
  mco.upper_hessian_lagrangian.ind[0] = 0;
  mco.upper_hessian_lagrangian.ind[1] = 0;
  mco.upper_hessian_lagrangian.ind[2] = 1;
  mco.upper_hessian_lagrangian.indptr[0] = 0;
  mco.upper_hessian_lagrangian.indptr[1] = 1;
  mco.upper_hessian_lagrangian.indptr[2] = 3;
  mco.upper_hessian_lagrangian.data[0] = 4.0;
  mco.upper_hessian_lagrangian.data[1] = 1.0;
  mco.upper_hessian_lagrangian.data[2] = 2.0;
  mco.upper_hessian_lagrangian.is_transposed = false;

  mco.c[0] = mci.x[0] + mci.x[1] - 1.0;

  mco.jacobian_c.rows = y_dim;
  mco.jacobian_c.cols = x_dim;
  mco.jacobian_c.ind[0] = 0;
  mco.jacobian_c.ind[1] = 0;
  mco.jacobian_c.indptr[0] = 0;
  mco.jacobian_c.indptr[1] = 1;
  mco.jacobian_c.indptr[2] = 2;
  mco.jacobian_c.data[0] = 1.0;
  mco.jacobian_c.data[1] = 1.0;
  mco.jacobian_c.is_transposed = false;

  mco.g[0] = mci.x[0] - 0.7;
  mco.g[1] = -mci.x[0] - 0.0;
  mco.g[2] = mci.x[1] - 0.7;
  mco.g[3] = -mci.x[1] - 0.0;

  mco.jacobian_g.rows = z_dim;
  mco.jacobian_g.cols = x_dim;
  mco.jacobian_g.ind[0] = 0;
  mco.jacobian_g.ind[1] = 1;
  mco.jacobian_g.ind[2] = 2;
  mco.jacobian_g.ind[3] = 3;
  mco.jacobian_g.indptr[0] = 0;
  mco.jacobian_g.indptr[1] = 2;
  mco.jacobian_g.indptr[2] = 4;
  mco.jacobian_g.data[0] = 1.0;
  mco.jacobian_g.data[1] = -1.0;
  mco.jacobian_g.data[2] = 1.0;
  mco.jacobian_g.data[3] = -1.0;
  mco.jacobian_g.is_transposed = false;
}

struct CallbackProvider {
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

  // To dynamically allocate the required memory.
  auto reserve(int L_nnz, int kkt_dim) -> void {
    LT_data = new double[L_nnz];
    D_diag = new double[kkt_dim];
  }

  auto free() -> void;
};

TEST(SimpleQP, FromOSQPRepo) {
  sip::ModelCallbackOutput _mco;
  constexpr int upper_hessian_lagrangian_nnz = 3;
  constexpr int jacobian_c_nnz = 2;
  constexpr int jacobian_g_nnz = 4;
  constexpr int L_nnz = 7;
  constexpr bool is_jacobian_c_transposed = false;
  constexpr bool is_jacobian_g_transposed = false;
  _mco.reserve(x_dim, z_dim, y_dim, upper_hessian_lagrangian_nnz,
               jacobian_c_nnz, jacobian_g_nnz, is_jacobian_c_transposed,
               is_jacobian_g_transposed);

  auto model_callback = [&](const sip::ModelCallbackInput &mci,
                            sip::ModelCallbackOutput **mco) -> void {
    _model_callback(mci, _mco);
    *mco = &_mco;
  };

  CallbackProvider callback_provider;
  constexpr int kkt_dim = x_dim + z_dim + y_dim;
  callback_provider.reserve(L_nnz, kkt_dim);

  const auto ldlt_factor =
      [&callback_provider](const double *upper_H_data, const double *C_data,
                           const double *G_data, const double *w,
                           const double r1, const double r2, const double r3) {
        return callback_provider.ldlt_factor(upper_H_data, C_data, G_data, w,
                                             r1, r2, r3);
      };
  const auto ldlt_solve = [&callback_provider](const double *b, double *v) {
    return callback_provider.ldlt_solve(b, v);
  };

  const auto timeout_callback = []() { return false; };

  sip::Input input{
      .factor = std::cref(ldlt_factor),
      .solve = std::cref(ldlt_solve),
      .add_Kx_to_y = &add_Kx_to_y,
      .add_upper_symmetric_Hx_to_y = &add_upper_symmetric_Hx_to_y,
      .add_Cx_to_y = &add_Cx_to_y,
      .add_CTx_to_y = &add_CTx_to_y,
      .add_Gx_to_y = &add_Gx_to_y,
      .add_GTx_to_y = &add_GTx_to_y,
      .model_callback = std::cref(model_callback),
      .timeout_callback = std::cref(timeout_callback),
  };

  sip::Settings settings{.max_kkt_violation = 1e-12,
                         .max_merit_slope = 1e-24,
                         .enable_elastics = true,
                         .elastic_var_cost_coeff = 1e6,
                         .print_line_search_logs = true};

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
}

} // namespace sip_examples
