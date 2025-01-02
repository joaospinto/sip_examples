#include "sip/sip.hpp"
#include "sip_qdldl/sip_qdldl.hpp"

#include <gtest/gtest.h>

namespace sip_examples {

TEST(SimpleNLP, SYMMETRIC_DIRECT_4x4) {
  sip::Settings settings{.max_kkt_violation = 1e-12,
                         .enable_elastics = true,
                         .elastic_var_cost_coeff = 1e6};
  sip::Workspace workspace;
  sip::Output output;

  constexpr int x_dim = 2;
  constexpr int s_dim = 2;
  constexpr int y_dim = 0;

  constexpr bool is_upper_hessian_f_transposed = false;
  constexpr bool is_jacobian_c_transposed = true;
  constexpr bool is_jacobian_g_transposed = true;

  workspace.reserve(x_dim, s_dim, y_dim);

  auto _model_callback = [](const sip::ModelCallbackInput &mci,
                            sip::ModelCallbackOutput &mco) -> void {
    mco.f = mci.x[1] * (5.0 + mci.x[0]);

    mco.gradient_f[0] = mci.x[1];
    mco.gradient_f[1] = 5.0 + mci.x[0];

    // NOTE: a positive definite Hessian approximation is expected.
    mco.upper_hessian_f.rows = x_dim;
    mco.upper_hessian_f.cols = x_dim;
    mco.upper_hessian_f.ind[0] = 0;
    mco.upper_hessian_f.ind[1] = 0;
    mco.upper_hessian_f.ind[2] = 1;
    mco.upper_hessian_f.indptr[0] = 0;
    mco.upper_hessian_f.indptr[1] = 1;
    mco.upper_hessian_f.indptr[2] = 3;
    // NOTE: only the upper triangle should be filled.
    //       the eigenvalues of the real Hessian are +-1,
    //       so we add (1 + 1e-6) to shift them.
    mco.upper_hessian_f.data[0] = 1.0 + 1e-6;
    mco.upper_hessian_f.data[1] = 1.0;
    mco.upper_hessian_f.data[2] = 1.0 + 1e-6;
    mco.upper_hessian_f.is_transposed = is_upper_hessian_f_transposed;

    // No equality constraints, so we don't set mco.c.

    mco.jacobian_c.rows = x_dim;
    mco.jacobian_c.cols = y_dim;
    mco.jacobian_c.indptr[0] = 0;
    mco.jacobian_c.is_transposed = is_jacobian_c_transposed;

    mco.g[0] = 5.0 - mci.x[0] * mci.x[1];
    mco.g[1] = mci.x[0] * mci.x[0] + mci.x[1] * mci.x[1] - 20.0;

    mco.jacobian_g.rows = x_dim;
    mco.jacobian_g.cols = s_dim;
    mco.jacobian_g.ind[0] = 0;
    mco.jacobian_g.ind[1] = 1;
    mco.jacobian_g.ind[2] = 0;
    mco.jacobian_g.ind[3] = 1;
    mco.jacobian_g.indptr[0] = 0;
    mco.jacobian_g.indptr[1] = 2;
    mco.jacobian_g.indptr[2] = 4;
    mco.jacobian_g.data[0] = -mci.x[1];
    mco.jacobian_g.data[1] = -mci.x[0];
    mco.jacobian_g.data[2] = 2 * mci.x[0];
    mco.jacobian_g.data[3] = 2 * mci.x[1];
    mco.jacobian_g.is_transposed = is_jacobian_g_transposed;
  };

  sip::ModelCallbackOutput _mco;
  constexpr int upper_hessian_f_nnz = 3;
  constexpr int jacobian_c_nnz = 0;
  constexpr int jacobian_g_nnz = 4;
  _mco.reserve(x_dim, s_dim, y_dim, upper_hessian_f_nnz, jacobian_c_nnz,
               jacobian_g_nnz, is_jacobian_c_transposed,
               is_jacobian_g_transposed);

  auto model_callback = [&](const sip::ModelCallbackInput &mci,
                            sip::ModelCallbackOutput **mco) -> void {
    _model_callback(mci, _mco);
    *mco = &_mco;
  };

  const auto kkt_p = std::array{5, 4, 3, 2, 1, 0};
  const auto kkt_pinv = std::array{5, 4, 3, 2, 1, 0};

  const auto lin_sys_formulation =
      sip_qdldl::Settings::LinearSystemFormulation::SYMMETRIC_DIRECT_4x4;
  constexpr int dummy_upper_jac_g_t_jac_g_nnz = 0;
  constexpr int dummy_upper_hessian_f_plus_upper_jac_g_t_jac_g_nnz = 0;
  constexpr int kkt_L_nnz = 11;

  sip_qdldl::Workspace sip_qdldl_workspace;
  sip_qdldl_workspace.reserve(
      lin_sys_formulation, x_dim, s_dim, y_dim, upper_hessian_f_nnz,
      jacobian_c_nnz, jacobian_g_nnz, dummy_upper_jac_g_t_jac_g_nnz,
      dummy_upper_hessian_f_plus_upper_jac_g_t_jac_g_nnz, kkt_L_nnz);

  const auto lin_sys_solver =
      [&](const double *c, const double *g, const double *grad_f,
          const double *H_data, const double *C_data, const double *G_data,
          const double *s, const double *y, const double *z, const double *e,
          const double mu, const double p, const double r1, const double r2,
          const double r3, double *dx, double *ds, double *dy, double *dz,
          double *de, double &kkt_error, double &lin_sys_error) {
        sip_qdldl::Input input{
            .c = c,
            .g = g,
            .grad_f = grad_f,
            .H = sip_qdldl::ConstSparseMatrix(
                x_dim, x_dim, _mco.upper_hessian_f.ind,
                _mco.upper_hessian_f.indptr, H_data,
                is_upper_hessian_f_transposed),
            .C = sip_qdldl::ConstSparseMatrix(x_dim, y_dim, _mco.jacobian_c.ind,
                                              _mco.jacobian_c.indptr, C_data,
                                              is_jacobian_c_transposed),
            .G = sip_qdldl::ConstSparseMatrix(x_dim, s_dim, _mco.jacobian_g.ind,
                                              _mco.jacobian_g.indptr, G_data,
                                              is_jacobian_g_transposed),
            .s = s,
            .y = y,
            .z = z,
            .e = e,
            .mu = mu,
            .p = p,
            .r1 = r1,
            .r2 = r2,
            .r3 = r3,
        };
        sip_qdldl::Settings settings{
            .lin_sys_formulation = lin_sys_formulation,
            .enable_elastics = true,
            .permute_kkt_system = true,
            .kkt_p = kkt_p.data(),
            .kkt_pinv = kkt_pinv.data(),
        };
        sip_qdldl::Output output{
            .dx = dx,
            .ds = ds,
            .dy = dy,
            .dz = dz,
            .de = de,
            .kkt_error = 0.0,
            .lin_sys_error = 0.0,
        };
        sip_qdldl::compute_search_direction(input, settings,
                                            sip_qdldl_workspace, output);
        kkt_error = output.kkt_error;
        lin_sys_error = output.lin_sys_error;
      };

  const auto timeout_callback = []() { return false; };

  sip::Input input{
      .model_callback = std::cref(model_callback),
      .lin_sys_solver = std::cref(lin_sys_solver),
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

  solve(input, settings, workspace, output);

  EXPECT_EQ(output.exit_status, sip::Status::SOLVED);

  EXPECT_NEAR(workspace.vars.x[0], -1.15747396, 1e-6);
  EXPECT_NEAR(workspace.vars.x[1], -4.31975162, 1e-6);

  sip_qdldl_workspace.free();
  workspace.free();
}

TEST(SimpleNLP, SYMMETRIC_INDIRECT_3x3) {
  sip::Settings settings{.max_kkt_violation = 1e-12,
                         .enable_elastics = true,
                         .elastic_var_cost_coeff = 1e6};
  sip::Workspace workspace;
  sip::Output output;

  constexpr int x_dim = 2;
  constexpr int s_dim = 2;
  constexpr int y_dim = 0;

  constexpr bool is_upper_hessian_f_transposed = false;
  constexpr bool is_jacobian_c_transposed = true;
  constexpr bool is_jacobian_g_transposed = true;

  workspace.reserve(x_dim, s_dim, y_dim);

  auto _model_callback = [](const sip::ModelCallbackInput &mci,
                            sip::ModelCallbackOutput &mco) -> void {
    mco.f = mci.x[1] * (5.0 + mci.x[0]);

    mco.gradient_f[0] = mci.x[1];
    mco.gradient_f[1] = 5.0 + mci.x[0];

    // NOTE: a positive definite Hessian approximation is expected.
    mco.upper_hessian_f.rows = x_dim;
    mco.upper_hessian_f.cols = x_dim;
    mco.upper_hessian_f.ind[0] = 0;
    mco.upper_hessian_f.ind[1] = 0;
    mco.upper_hessian_f.ind[2] = 1;
    mco.upper_hessian_f.indptr[0] = 0;
    mco.upper_hessian_f.indptr[1] = 1;
    mco.upper_hessian_f.indptr[2] = 3;
    // NOTE: only the upper triangle should be filled.
    //       the eigenvalues of the real Hessian are +-1,
    //       so we add (1 + 1e-6) to shift them.
    mco.upper_hessian_f.data[0] = 1.0 + 1e-6;
    mco.upper_hessian_f.data[1] = 1.0;
    mco.upper_hessian_f.data[2] = 1.0 + 1e-6;
    mco.upper_hessian_f.is_transposed = is_upper_hessian_f_transposed;

    // No equality constraints, so we don't set mco.c.

    mco.jacobian_c.rows = x_dim;
    mco.jacobian_c.cols = y_dim;
    mco.jacobian_c.indptr[0] = 0;
    mco.jacobian_c.is_transposed = is_jacobian_c_transposed;

    mco.g[0] = 5.0 - mci.x[0] * mci.x[1];
    mco.g[1] = mci.x[0] * mci.x[0] + mci.x[1] * mci.x[1] - 20.0;

    mco.jacobian_g.rows = x_dim;
    mco.jacobian_g.cols = s_dim;
    mco.jacobian_g.ind[0] = 0;
    mco.jacobian_g.ind[1] = 1;
    mco.jacobian_g.ind[2] = 0;
    mco.jacobian_g.ind[3] = 1;
    mco.jacobian_g.indptr[0] = 0;
    mco.jacobian_g.indptr[1] = 2;
    mco.jacobian_g.indptr[2] = 4;
    mco.jacobian_g.data[0] = -mci.x[1];
    mco.jacobian_g.data[1] = -mci.x[0];
    mco.jacobian_g.data[2] = 2 * mci.x[0];
    mco.jacobian_g.data[3] = 2 * mci.x[1];
    mco.jacobian_g.is_transposed = is_jacobian_g_transposed;
  };

  sip::ModelCallbackOutput _mco;
  constexpr int upper_hessian_f_nnz = 3;
  constexpr int jacobian_c_nnz = 0;
  constexpr int jacobian_g_nnz = 4;
  _mco.reserve(x_dim, s_dim, y_dim, upper_hessian_f_nnz, jacobian_c_nnz,
               jacobian_g_nnz, is_jacobian_c_transposed,
               is_jacobian_g_transposed);

  auto model_callback = [&](const sip::ModelCallbackInput &mci,
                            sip::ModelCallbackOutput **mco) -> void {
    _model_callback(mci, _mco);
    *mco = &_mco;
  };

  const auto kkt_p = std::array{3, 2, 1, 0};
  const auto kkt_pinv = std::array{3, 2, 1, 0};

  const auto lin_sys_formulation =
      sip_qdldl::Settings::LinearSystemFormulation::SYMMETRIC_INDIRECT_3x3;
  constexpr int dummy_upper_jac_g_t_jac_g_nnz = 0;
  constexpr int dummy_upper_hessian_f_plus_upper_jac_g_t_jac_g_nnz = 0;
  constexpr int kkt_L_nnz = 5;

  sip_qdldl::Workspace sip_qdldl_workspace;
  sip_qdldl_workspace.reserve(
      lin_sys_formulation, x_dim, s_dim, y_dim, upper_hessian_f_nnz,
      jacobian_c_nnz, jacobian_g_nnz, dummy_upper_jac_g_t_jac_g_nnz,
      dummy_upper_hessian_f_plus_upper_jac_g_t_jac_g_nnz, kkt_L_nnz);

  const auto lin_sys_solver =
      [&](const double *c, const double *g, const double *grad_f,
          const double *H_data, const double *C_data, const double *G_data,
          const double *s, const double *y, const double *z, const double *e,
          const double mu, const double p, const double r1, const double r2,
          const double r3, double *dx, double *ds, double *dy, double *dz,
          double *de, double &kkt_error, double &lin_sys_error) {
        sip_qdldl::Input input{
            .c = c,
            .g = g,
            .grad_f = grad_f,
            .H = sip_qdldl::ConstSparseMatrix(
                x_dim, x_dim, _mco.upper_hessian_f.ind,
                _mco.upper_hessian_f.indptr, H_data,
                is_upper_hessian_f_transposed),
            .C = sip_qdldl::ConstSparseMatrix(x_dim, y_dim, _mco.jacobian_c.ind,
                                              _mco.jacobian_c.indptr, C_data,
                                              is_jacobian_c_transposed),
            .G = sip_qdldl::ConstSparseMatrix(x_dim, s_dim, _mco.jacobian_g.ind,
                                              _mco.jacobian_g.indptr, G_data,
                                              is_jacobian_g_transposed),
            .s = s,
            .y = y,
            .z = z,
            .e = e,
            .mu = mu,
            .p = p,
            .r1 = r1,
            .r2 = r2,
            .r3 = r3,
        };
        sip_qdldl::Settings settings{
            .lin_sys_formulation = lin_sys_formulation,
            .enable_elastics = true,
            .permute_kkt_system = true,
            .kkt_p = kkt_p.data(),
            .kkt_pinv = kkt_pinv.data(),
        };
        sip_qdldl::Output output{
            .dx = dx,
            .ds = ds,
            .dy = dy,
            .dz = dz,
            .de = de,
            .kkt_error = 0.0,
            .lin_sys_error = 0.0,
        };
        sip_qdldl::compute_search_direction(input, settings,
                                            sip_qdldl_workspace, output);
        kkt_error = output.kkt_error;
        lin_sys_error = output.lin_sys_error;
      };

  const auto timeout_callback = []() { return false; };

  sip::Input input{
      .model_callback = std::cref(model_callback),
      .lin_sys_solver = std::cref(lin_sys_solver),
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

  solve(input, settings, workspace, output);

  EXPECT_EQ(output.exit_status, sip::Status::SOLVED);

  EXPECT_NEAR(workspace.vars.x[0], -1.15747396, 1e-6);
  EXPECT_NEAR(workspace.vars.x[1], -4.31975162, 1e-6);

  sip_qdldl_workspace.free();
  workspace.free();
}

TEST(SimpleNLP, SYMMETRIC_INDIRECT_2x2) {
  sip::Settings settings{.max_kkt_violation = 1e-12,
                         .enable_elastics = true,
                         .elastic_var_cost_coeff = 1e6};
  sip::Workspace workspace;
  sip::Output output;

  constexpr int x_dim = 2;
  constexpr int s_dim = 2;
  constexpr int y_dim = 0;

  constexpr bool is_upper_hessian_f_transposed = false;
  constexpr bool is_jacobian_c_transposed = true;
  constexpr bool is_jacobian_g_transposed = false;

  workspace.reserve(x_dim, s_dim, y_dim);

  auto _model_callback = [](const sip::ModelCallbackInput &mci,
                            sip::ModelCallbackOutput &mco) -> void {
    mco.f = mci.x[1] * (5.0 + mci.x[0]);

    mco.gradient_f[0] = mci.x[1];
    mco.gradient_f[1] = 5.0 + mci.x[0];

    // NOTE: a positive definite Hessian approximation is expected.
    mco.upper_hessian_f.rows = x_dim;
    mco.upper_hessian_f.cols = x_dim;
    mco.upper_hessian_f.ind[0] = 0;
    mco.upper_hessian_f.ind[1] = 0;
    mco.upper_hessian_f.ind[2] = 1;
    mco.upper_hessian_f.indptr[0] = 0;
    mco.upper_hessian_f.indptr[1] = 1;
    mco.upper_hessian_f.indptr[2] = 3;
    // NOTE: only the upper triangle should be filled.
    //       the eigenvalues of the real Hessian are +-1,
    //       so we add (1 + 1e-6) to shift them.
    mco.upper_hessian_f.data[0] = 1.0 + 1e-6;
    mco.upper_hessian_f.data[1] = 1.0;
    mco.upper_hessian_f.data[2] = 1.0 + 1e-6;
    mco.upper_hessian_f.is_transposed = is_upper_hessian_f_transposed;

    // No equality constraints, so we don't set mco.c.

    mco.jacobian_c.rows = x_dim;
    mco.jacobian_c.cols = y_dim;
    mco.jacobian_c.indptr[0] = 0;
    mco.jacobian_c.is_transposed = is_jacobian_c_transposed;

    mco.g[0] = 5.0 - mci.x[0] * mci.x[1];
    mco.g[1] = mci.x[0] * mci.x[0] + mci.x[1] * mci.x[1] - 20.0;

    mco.jacobian_g.rows = s_dim;
    mco.jacobian_g.cols = x_dim;
    mco.jacobian_g.ind[0] = 0;
    mco.jacobian_g.ind[1] = 1;
    mco.jacobian_g.ind[2] = 0;
    mco.jacobian_g.ind[3] = 1;
    mco.jacobian_g.indptr[0] = 0;
    mco.jacobian_g.indptr[1] = 2;
    mco.jacobian_g.indptr[2] = 4;
    mco.jacobian_g.data[0] = -mci.x[1];
    mco.jacobian_g.data[1] = 2 * mci.x[0];
    mco.jacobian_g.data[2] = -mci.x[0];
    mco.jacobian_g.data[3] = 2 * mci.x[1];
    mco.jacobian_g.is_transposed = is_jacobian_g_transposed;
  };

  sip::ModelCallbackOutput _mco;
  constexpr int upper_hessian_f_nnz = 3;
  constexpr int jacobian_c_nnz = 0;
  constexpr int jacobian_g_nnz = 4;
  _mco.reserve(x_dim, s_dim, y_dim, upper_hessian_f_nnz, jacobian_c_nnz,
               jacobian_g_nnz, is_jacobian_c_transposed,
               is_jacobian_g_transposed);

  auto model_callback = [&](const sip::ModelCallbackInput &mci,
                            sip::ModelCallbackOutput **mco) -> void {
    _model_callback(mci, _mco);
    *mco = &_mco;
  };

  const auto kkt_p = std::array{1, 0};
  const auto kkt_pinv = std::array{1, 0};

  const auto lin_sys_formulation =
      sip_qdldl::Settings::LinearSystemFormulation::SYMMETRIC_INDIRECT_2x2;
  constexpr int upper_jac_g_t_jac_g_nnz = 3;
  constexpr int upper_hessian_f_plus_upper_jac_g_t_jac_g_nnz = 3;
  constexpr int kkt_L_nnz = 3;

  sip_qdldl::Workspace sip_qdldl_workspace;
  sip_qdldl_workspace.reserve(
      lin_sys_formulation, x_dim, s_dim, y_dim, upper_hessian_f_nnz,
      jacobian_c_nnz, jacobian_g_nnz, upper_jac_g_t_jac_g_nnz,
      upper_hessian_f_plus_upper_jac_g_t_jac_g_nnz, kkt_L_nnz);

  const auto lin_sys_solver =
      [&](const double *c, const double *g, const double *grad_f,
          const double *H_data, const double *C_data, const double *G_data,
          const double *s, const double *y, const double *z, const double *e,
          const double mu, const double p, const double r1, const double r2,
          const double r3, double *dx, double *ds, double *dy, double *dz,
          double *de, double &kkt_error, double &lin_sys_error) {
        sip_qdldl::Input input{
            .c = c,
            .g = g,
            .grad_f = grad_f,
            .H = sip_qdldl::ConstSparseMatrix(
                x_dim, x_dim, _mco.upper_hessian_f.ind,
                _mco.upper_hessian_f.indptr, H_data,
                is_upper_hessian_f_transposed),
            .C = sip_qdldl::ConstSparseMatrix(x_dim, y_dim, _mco.jacobian_c.ind,
                                              _mco.jacobian_c.indptr, C_data,
                                              is_jacobian_c_transposed),
            .G = sip_qdldl::ConstSparseMatrix(s_dim, x_dim, _mco.jacobian_g.ind,
                                              _mco.jacobian_g.indptr, G_data,
                                              is_jacobian_g_transposed),
            .s = s,
            .y = y,
            .z = z,
            .e = e,
            .mu = mu,
            .p = p,
            .r1 = r1,
            .r2 = r2,
            .r3 = r3,
        };
        sip_qdldl::Settings settings{
            .lin_sys_formulation = lin_sys_formulation,
            .enable_elastics = true,
            .permute_kkt_system = true,
            .kkt_p = kkt_p.data(),
            .kkt_pinv = kkt_pinv.data(),
        };
        sip_qdldl::Output output{
            .dx = dx,
            .ds = ds,
            .dy = dy,
            .dz = dz,
            .de = de,
            .kkt_error = 0.0,
            .lin_sys_error = 0.0,
        };
        sip_qdldl::compute_search_direction(input, settings,
                                            sip_qdldl_workspace, output);
        kkt_error = output.kkt_error;
        lin_sys_error = output.lin_sys_error;
      };

  const auto timeout_callback = []() { return false; };

  sip::Input input{
      .model_callback = std::cref(model_callback),
      .lin_sys_solver = std::cref(lin_sys_solver),
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

  solve(input, settings, workspace, output);

  EXPECT_EQ(output.exit_status, sip::Status::SOLVED);

  EXPECT_NEAR(workspace.vars.x[0], -1.15747396, 1e-6);
  EXPECT_NEAR(workspace.vars.x[1], -4.31975162, 1e-6);

  sip_qdldl_workspace.free();
  workspace.free();
}

} // namespace sip_examples
