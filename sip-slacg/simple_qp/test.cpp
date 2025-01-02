#include "sip-slacg/simple_qp/kkt_codegen.hpp"

#include "sip/sip.hpp"
#include <gtest/gtest.h>

#include <algorithm>
#include <array>

namespace sip_examples {

namespace {
constexpr int x_dim = 2;
constexpr int s_dim = 4;
constexpr int y_dim = 1;
} // namespace

void _model_callback(const sip::ModelCallbackInput &mci,
                     sip::ModelCallbackOutput &mco) {
  mco.f = 0.5 * (4.0 * mci.x[0] * mci.x[0] + 2.0 * mci.x[0] * mci.x[1] +
                 2.0 * mci.x[1] * mci.x[1]) +
          mci.x[0] + mci.x[1];

  mco.gradient_f[0] = 4.0 * mci.x[0] + 1.0 * mci.x[1] + 1.0;
  mco.gradient_f[1] = 1.0 * mci.x[0] + 2.0 * mci.x[1] + 1.0;

  // NOTE: only the upper triangle should be filled.
  mco.upper_hessian_f.rows = x_dim;
  mco.upper_hessian_f.cols = x_dim;
  mco.upper_hessian_f.ind[0] = 0;
  mco.upper_hessian_f.ind[1] = 0;
  mco.upper_hessian_f.ind[2] = 1;
  mco.upper_hessian_f.indptr[0] = 0;
  mco.upper_hessian_f.indptr[1] = 1;
  mco.upper_hessian_f.indptr[2] = 3;
  mco.upper_hessian_f.data[0] = 4.0;
  mco.upper_hessian_f.data[1] = 1.0;
  mco.upper_hessian_f.data[2] = 2.0;
  mco.upper_hessian_f.is_transposed = false;

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

  mco.jacobian_g.rows = s_dim;
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

TEST(SimpleQPFromOSQPRepo, SYMMETRIC_INDIRECT_3x3) {
  sip::ModelCallbackOutput _mco;
  constexpr int upper_hessian_f_nnz = 3;
  constexpr int jacobian_c_nnz = 2;
  constexpr int jacobian_g_nnz = 4;
  constexpr bool is_jacobian_c_transposed = false;
  constexpr bool is_jacobian_g_transposed = false;
  _mco.reserve(x_dim, s_dim, y_dim, upper_hessian_f_nnz, jacobian_c_nnz,
               jacobian_g_nnz, is_jacobian_c_transposed,
               is_jacobian_g_transposed);

  auto model_callback = [&](const sip::ModelCallbackInput &mci,
                            sip::ModelCallbackOutput **mco) -> void {
    _model_callback(mci, _mco);
    *mco = &_mco;
  };

  const auto timeout_callback = []() { return false; };

  sip::Input input{
      .model_callback = std::cref(model_callback),
      .lin_sys_solver = &newton_kkt_solver,
      .timeout_callback = std::cref(timeout_callback),
  };

  sip::Settings settings{.max_kkt_violation = 1e-12,
                         .enable_elastics = true,
                         .elastic_var_cost_coeff = 1e6};

  sip::Workspace workspace;
  workspace.reserve(x_dim, s_dim, y_dim);

  sip::Output output;

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

  sip::solve(input, settings, workspace, output);

  EXPECT_EQ(output.exit_status, sip::Status::SOLVED);

  EXPECT_NEAR(workspace.vars.x[0], 0.3, 1e-6);
  EXPECT_NEAR(workspace.vars.x[1], 0.7, 1e-6);

  workspace.free();
}

} // namespace sip_examples
