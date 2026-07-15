#include "problem_definitions/mock_problems/simple_nlp/problem.hpp"

namespace sip_examples::problem_definitions::simple_nlp {
namespace {

void evaluate_common(const sip::ModelCallbackInput &mci, double *f,
                     double *gradient_f, double *upper_hessian_lagrangian,
                     double *g) {
  *f = mci.x[1] * (5.0 + mci.x[0]);
  gradient_f[0] = mci.x[1];
  gradient_f[1] = 5.0 + mci.x[0];
  upper_hessian_lagrangian[0] = 2.0 * mci.z[1];
  upper_hessian_lagrangian[1] = 1.0 - mci.z[0];
  upper_hessian_lagrangian[2] = 2.0 * mci.z[1];
  g[0] = 5.0 - mci.x[0] * mci.x[1];
  g[1] = mci.x[0] * mci.x[0] + mci.x[1] * mci.x[1] - 20.0;
}

} // namespace

auto settings() -> sip::Settings {
  return sip::Settings{
      .mode = sip::Mode::PRIMAL_PROXIMAL_IPM,
      .termination =
          {
              .max_dual_residual = 1e-12,
              .max_constraint_violation = 1e-12,
              .max_complementarity_gap = 1e-12,
              .max_merit_slope = 1e-24,
          },
  };
}

void initialize(sip::Workspace &workspace) {
  for (int i = 0; i < kXDim; ++i) {
    workspace.vars.x[i] = 0.0;
  }
  for (int i = 0; i < kSDim; ++i) {
    workspace.vars.s[i] = 1.0;
    workspace.vars.z[i] = 1.0;
  }
}

void configure_qdldl_sparsity(sip_qdldl::ModelCallbackOutput &mco) {
  mco.upper_hessian_lagrangian.rows = kXDim;
  mco.upper_hessian_lagrangian.cols = kXDim;
  mco.upper_hessian_lagrangian.ind[0] = 0;
  mco.upper_hessian_lagrangian.ind[1] = 0;
  mco.upper_hessian_lagrangian.ind[2] = 1;
  mco.upper_hessian_lagrangian.indptr[0] = 0;
  mco.upper_hessian_lagrangian.indptr[1] = 1;
  mco.upper_hessian_lagrangian.indptr[2] = 3;
  mco.upper_hessian_lagrangian.is_transposed = false;

  mco.jacobian_c.rows = kXDim;
  mco.jacobian_c.cols = kYDim;
  mco.jacobian_c.indptr[0] = 0;
  mco.jacobian_c.is_transposed = true;

  mco.jacobian_g.rows = kXDim;
  mco.jacobian_g.cols = kSDim;
  mco.jacobian_g.ind[0] = 0;
  mco.jacobian_g.ind[1] = 1;
  mco.jacobian_g.ind[2] = 0;
  mco.jacobian_g.ind[3] = 1;
  mco.jacobian_g.indptr[0] = 0;
  mco.jacobian_g.indptr[1] = 2;
  mco.jacobian_g.indptr[2] = 4;
  mco.jacobian_g.is_transposed = true;
}

void evaluate_qdldl(const sip::ModelCallbackInput &mci, double *f,
                    double *gradient_f, double *g,
                    double *upper_hessian_lagrangian, double *jacobian_g) {
  evaluate_common(mci, f, gradient_f, upper_hessian_lagrangian, g);
  jacobian_g[0] = -mci.x[1];
  jacobian_g[1] = -mci.x[0];
  jacobian_g[2] = 2.0 * mci.x[0];
  jacobian_g[3] = 2.0 * mci.x[1];
}

void evaluate_slacg(const sip::ModelCallbackInput &mci, double *f,
                    double *gradient_f, double *g,
                    double *upper_hessian_lagrangian, double *jacobian_g) {
  evaluate_common(mci, f, gradient_f, upper_hessian_lagrangian, g);
  jacobian_g[0] = -mci.x[1];
  jacobian_g[1] = 2.0 * mci.x[0];
  jacobian_g[2] = -mci.x[0];
  jacobian_g[3] = 2.0 * mci.x[1];
}

} // namespace sip_examples::problem_definitions::simple_nlp
