#include "sip-slacg/simple_nlp/kkt_codegen.hpp"

#include "sip/sip.hpp"
#include <gtest/gtest.h>

#include <algorithm>
#include <array>

namespace sip_examples {

namespace {
constexpr int x_dim = 2;
constexpr int s_dim = 2;
constexpr int y_dim = 0;
} // namespace

void _model_callback(const sip::ModelCallbackInput &mci,
                     sip::ModelCallbackOutput &mco) {
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
  mco.upper_hessian_f.is_transposed = false;

  // No equality constraints, so we don't set mco.c.

  mco.jacobian_c.rows = y_dim;
  mco.jacobian_c.cols = x_dim;
  mco.jacobian_c.indptr[0] = 0;
  mco.jacobian_c.is_transposed = false;

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
  mco.jacobian_g.is_transposed = false;
}

TEST(SimpleNLP, SYMMETRIC_INDIRECT_3x3) {
  sip::ModelCallbackOutput _mco;
  constexpr int upper_hessian_f_nnz = 3;
  constexpr int jacobian_c_nnz = 0;
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

  sip::Input input{.model_callback = std::cref(model_callback),
                   .lin_sys_solver = &newton_kkt_solver};

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

  EXPECT_NEAR(workspace.vars.x[0], -1.15747396, 1e-6);
  EXPECT_NEAR(workspace.vars.x[1], -4.31975162, 1e-6);

  workspace.free();
}

} // namespace sip_examples
