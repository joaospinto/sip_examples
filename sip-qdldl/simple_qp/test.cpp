#include "sip/sip.hpp"
#include "sip_qdldl/sip_qdldl.hpp"

#include <gtest/gtest.h>

namespace sip_examples {

TEST(SimpleQP, FromOSQPRepo) {
  sip::Settings settings{.max_aug_kkt_violation = 1e-12,
                         .enable_elastics = true,
                         .elastic_var_cost_coeff = 1e6};
  sip::Workspace workspace;

  constexpr int x_dim = 2;
  constexpr int s_dim = 4;
  constexpr int y_dim = 1;
  constexpr int kkt_dim = x_dim + s_dim + y_dim;

  constexpr bool is_upper_hessian_lagrangian_transposed = false;
  constexpr bool is_jacobian_c_transposed = true;
  constexpr bool is_jacobian_g_transposed = true;

  constexpr int kkt_nnz = 14;
  constexpr int kkt_L_nnz = 15;

  workspace.reserve(x_dim, s_dim, y_dim, kkt_L_nnz);

  auto _model_callback = [](const sip::ModelCallbackInput &mci,
                            sip::ModelCallbackOutput &mco) -> void {
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
    mco.upper_hessian_lagrangian.is_transposed =
        is_upper_hessian_lagrangian_transposed;

    mco.c[0] = mci.x[0] + mci.x[1] - 1.0;

    mco.jacobian_c.rows = x_dim;
    mco.jacobian_c.cols = y_dim;
    mco.jacobian_c.ind[0] = 0;
    mco.jacobian_c.ind[1] = 1;
    mco.jacobian_c.indptr[0] = 0;
    mco.jacobian_c.indptr[1] = 2;
    mco.jacobian_c.data[0] = 1.0;
    mco.jacobian_c.data[1] = 1.0;
    mco.jacobian_c.is_transposed = is_jacobian_c_transposed;

    mco.g[0] = mci.x[0] - 0.7;
    mco.g[1] = -mci.x[0] - 0.0;
    mco.g[2] = mci.x[1] - 0.7;
    mco.g[3] = -mci.x[1] - 0.0;

    mco.jacobian_g.rows = x_dim;
    mco.jacobian_g.cols = s_dim;
    mco.jacobian_g.ind[0] = 0;
    mco.jacobian_g.ind[1] = 0;
    mco.jacobian_g.ind[2] = 1;
    mco.jacobian_g.ind[3] = 1;
    mco.jacobian_g.indptr[0] = 0;
    mco.jacobian_g.indptr[1] = 1;
    mco.jacobian_g.indptr[2] = 2;
    mco.jacobian_g.indptr[3] = 3;
    mco.jacobian_g.indptr[4] = 4;
    mco.jacobian_g.data[0] = 1.0;
    mco.jacobian_g.data[1] = -1.0;
    mco.jacobian_g.data[2] = 1.0;
    mco.jacobian_g.data[3] = -1.0;
    mco.jacobian_g.is_transposed = is_jacobian_g_transposed;
  };

  sip::ModelCallbackOutput _mco;
  constexpr int upper_hessian_lagrangian_nnz = 3;
  constexpr int jacobian_c_nnz = 2;
  constexpr int jacobian_g_nnz = 4;
  _mco.reserve(x_dim, s_dim, y_dim, upper_hessian_lagrangian_nnz,
               jacobian_c_nnz, jacobian_g_nnz, is_jacobian_c_transposed,
               is_jacobian_g_transposed);

  auto model_callback = [&](const sip::ModelCallbackInput &mci,
                            sip::ModelCallbackOutput **mco) -> void {
    _model_callback(mci, _mco);
    *mco = &_mco;
  };

  const auto kkt_pinv = std::array{6, 5, 4, 3, 2, 1, 0};

  sip_qdldl::Workspace sip_qdldl_workspace;
  sip_qdldl_workspace.reserve(kkt_dim, kkt_nnz, kkt_L_nnz);

  const sip_qdldl::Settings sip_qdldl_settings{
      .permute_kkt_system = true,
      .kkt_pinv = kkt_pinv.data(),
  };

  auto x_data = std::array{1.0, 1.0};
  sip::ModelCallbackInput x{
      .x = x_data.data(),
      .y = nullptr, // Unused.
      .z = nullptr, // Unused.
  };
  sip::ModelCallbackOutput y;
  y.reserve(x_dim, s_dim, y_dim, upper_hessian_lagrangian_nnz, jacobian_c_nnz,
            jacobian_g_nnz, is_jacobian_c_transposed, is_jacobian_g_transposed);
  _model_callback(x, y);

  const auto upper_H_pattern = sip_qdldl::ConstSparseMatrix(
      y.upper_hessian_lagrangian.rows, y.upper_hessian_lagrangian.cols,
      y.upper_hessian_lagrangian.ind, y.upper_hessian_lagrangian.indptr,
      y.upper_hessian_lagrangian.data,
      y.upper_hessian_lagrangian.is_transposed);

  const auto C_pattern = sip_qdldl::ConstSparseMatrix(
      y.jacobian_c.rows, y.jacobian_c.cols, y.jacobian_c.ind,
      y.jacobian_c.indptr, y.jacobian_c.data, y.jacobian_c.is_transposed);

  const auto G_pattern = sip_qdldl::ConstSparseMatrix(
      y.jacobian_g.rows, y.jacobian_g.cols, y.jacobian_g.ind,
      y.jacobian_g.indptr, y.jacobian_g.data, y.jacobian_g.is_transposed);

  auto callback_provider =
      sip_qdldl::CallbackProvider(upper_H_pattern, C_pattern, G_pattern,
                                  sip_qdldl_settings, sip_qdldl_workspace);

  const auto timeout_callback = []() { return false; };

  const auto ldlt_factor =
      [&callback_provider](const double *H_data, const double *C_data,
                           const double *G_data, const double *w,
                           const double r1, const double r2, const double r3,
                           double *LT_data, double *D_diag) -> void {
    return callback_provider.ldlt_factor(H_data, C_data, G_data, w, r1, r2, r3,
                                         LT_data, D_diag);
  };

  const auto ldlt_solve =
      [&callback_provider](const double *LT_data, const double *D_diag,
                           const double *b, double *v) -> void {
    return callback_provider.ldlt_solve(LT_data, D_diag, b, v);
  };

  const auto add_Kx_to_y =
      [&callback_provider](
          const double *H_data, const double *C_data, const double *G_data,
          const double *w, const double r1, const double r2, const double r3,
          const double *x_x, const double *x_y, const double *x_z, double *y_x,
          double *y_y, double *y_z) -> void {
    return callback_provider.add_Kx_to_y(H_data, C_data, G_data, w, r1, r2, r3,
                                         x_x, x_y, x_z, y_x, y_y, y_z);
  };

  const auto add_upper_symmetric_Hx_to_y =
      [&callback_provider](const double *H_data, const double *x,
                           double *y) -> void {
    return callback_provider.add_upper_symmetric_Hx_to_y(H_data, x, y);
  };

  const auto add_Cx_to_y = [&callback_provider](const double *C_data,
                                                const double *x,
                                                double *y) -> void {
    return callback_provider.add_Cx_to_y(C_data, x, y);
  };

  const auto add_CTx_to_y = [&callback_provider](const double *C_data,
                                                 const double *x,
                                                 double *y) -> void {
    return callback_provider.add_CTx_to_y(C_data, x, y);
  };

  const auto add_Gx_to_y = [&callback_provider](const double *G_data,
                                                const double *x,
                                                double *y) -> void {
    return callback_provider.add_Gx_to_y(G_data, x, y);
  };

  const auto add_GTx_to_y = [&callback_provider](const double *G_data,
                                                 const double *x,
                                                 double *y) -> void {
    return callback_provider.add_GTx_to_y(G_data, x, y);
  };

  sip::Input input{
      .ldlt_factor = std::cref(ldlt_factor),
      .ldlt_solve = std::cref(ldlt_solve),
      .add_Kx_to_y = std::cref(add_Kx_to_y),
      .add_upper_symmetric_Hx_to_y = std::cref(add_upper_symmetric_Hx_to_y),
      .add_Cx_to_y = std::cref(add_Cx_to_y),
      .add_CTx_to_y = std::cref(add_CTx_to_y),
      .add_Gx_to_y = std::cref(add_Gx_to_y),
      .add_GTx_to_y = std::cref(add_GTx_to_y),
      .model_callback = std::cref(model_callback),
      .timeout_callback = std::cref(timeout_callback),
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

  EXPECT_NEAR(workspace.vars.x[0], 0.3, 1e-5);
  EXPECT_NEAR(workspace.vars.x[1], 0.7, 1e-5);

  sip_qdldl_workspace.free();
  workspace.free();
}

} // namespace sip_examples
