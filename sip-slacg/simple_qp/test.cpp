#include "sip-slacg/simple_qp/C_ops.hpp"
#include "sip-slacg/simple_qp/G_ops.hpp"
#include "sip-slacg/simple_qp/H_ops.hpp"
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

void lin_sys_solver(const sip::ModelCallbackOutput &mco, const double *s,
                    const double *y, const double *z, const double *e,
                    const double mu, const double p, const double r1,
                    const double r2, const double r3,
                    [[maybe_unused]] const double new_lhs, double *dx,
                    double *ds, double *dy, double *dz, double *de,
                    double &kkt_error, double &lin_sys_error) {
  assert(!mco.jacobian_c.is_transposed && !mco.jacobian_g.is_transposed);

  std::array<double, s_dim> w;

  for (int i = 0; i < s_dim; ++i) {
    w[i] = s[i] / z[i];
  }

  std::array<double, L_nnz> LT_data;
  std::array<double, dim> D_diag;

  const double r3p = r3 + 1.0 / p;

  ldlt_factor(mco.upper_hessian_f.data, mco.jacobian_c.data,
              mco.jacobian_g.data, w.data(), r1, r2, r3p, LT_data.data(),
              D_diag.data());

  std::array<double, dim> b;
  double *bx = b.data();
  double *by = bx + x_dim;
  double *bz = by + y_dim;

  for (int i = 0; i < x_dim; ++i) {
    bx[i] = mco.gradient_f[i];
  }

  C_ops::add_ATx_to_y(mco.jacobian_c.data, y, bx);
  G_ops::add_ATx_to_y(mco.jacobian_g.data, z, bx);

  for (int i = 0; i < y_dim; ++i) {
    by[i] = mco.c[i];
  }

  for (int i = 0; i < s_dim; ++i) {
    bz[i] = mco.g[i] + mu / z[i] - z[i] / p;
  }

  for (int i = 0; i < dim; ++i) {
    b[i] = -b[i];
  }

  std::array<double, dim> v;

  ldlt_solve(LT_data.data(), D_diag.data(), b.data(), v.data());

  std::array<double, dim> residual;
  for (int i = 0; i < dim; ++i) {
    residual[i] = -b[i];
  }

  add_Kx_to_y(mco.upper_hessian_f.data, mco.jacobian_c.data,
              mco.jacobian_g.data, w.data(), r1, r2, r3p, v.data(),
              residual.data());

  lin_sys_error = 0.0;
  for (int i = 0; i < dim; ++i) {
    lin_sys_error = std::max(lin_sys_error, std::fabs(residual[i]));
  }

  kkt_error = 0.0;
  for (int i = 0; i < x_dim; ++i) {
    kkt_error = std::max(kkt_error, std::fabs(b[i]));
  }
  for (int i = 0; i < y_dim; ++i) {
    kkt_error = std::max(kkt_error, std::fabs(mco.c[i]));
  }
  for (int i = 0; i < s_dim; ++i) {
    kkt_error = std::max(kkt_error, std::fabs(mco.g[i] + s[i] + e[i]));
  }

  auto it = v.begin();
  std::copy(it, it + x_dim, dx);
  it += x_dim;
  std::copy(it, it + y_dim, dy);
  it += y_dim;
  std::copy(it, it + s_dim, dz);
  it += s_dim;

  // Any of these formulas can be used to recover ds,
  // but the second one is preferred for being cheaper to compute.
  // ds = z / p -(g(x) + s) + (gamma_z + 1 / p) * dz - G @ dx
  // ds = -Z^{-1} S dz - s + mu / z.
  for (int i = 0; i < s_dim; ++i) {
    ds[i] = -s[i] / z[i] * dz[i] - s[i] + mu / z[i];
  }

  // de = (-dz - (pe + z)) / p
  for (int i = 0; i < s_dim; ++i) {
    de[i] = (-dz[i] - p * e[i] - z[i]) / p;
  }
}

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

  sip::Input input{.model_callback = std::cref(model_callback),
                   .lin_sys_solver = &lin_sys_solver};

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
