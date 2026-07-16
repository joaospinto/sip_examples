#include "problem_definitions/mock_problems/simple_qp/problem.hpp"

#include <algorithm>

namespace sip_examples::problem_definitions::simple_qp {

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
  mco.jacobian_c.ind[0] = 0;
  mco.jacobian_c.ind[1] = 1;
  mco.jacobian_c.indptr[0] = 0;
  mco.jacobian_c.indptr[1] = 2;
  mco.jacobian_c.is_transposed = true;

  mco.jacobian_g.rows = kXDim;
  mco.jacobian_g.cols = 0;
  mco.jacobian_g.indptr[0] = 0;
  mco.jacobian_g.is_transposed = true;
}

void evaluate(const sip::ModelCallbackInput &mci, double *f, double *gradient_f,
              double *c, double *upper_hessian_lagrangian, double *jacobian_c) {
  *f = 0.5 * (4.0 * mci.x[0] * mci.x[0] + 2.0 * mci.x[0] * mci.x[1] +
              2.0 * mci.x[1] * mci.x[1]) +
       mci.x[0] + mci.x[1];

  gradient_f[0] = 4.0 * mci.x[0] + mci.x[1] + 1.0;
  gradient_f[1] = mci.x[0] + 2.0 * mci.x[1] + 1.0;

  upper_hessian_lagrangian[0] = 4.0;
  upper_hessian_lagrangian[1] = 1.0;
  upper_hessian_lagrangian[2] = 2.0;

  c[0] = mci.x[0] + mci.x[1] - 1.0;
  jacobian_c[0] = 1.0;
  jacobian_c[1] = 1.0;
}

auto settings() -> sip::Settings {
  return sip::Settings{
      .termination =
          {
              .max_dual_residual = 1e-12,
              .max_constraint_violation = 1e-12,
              .max_complementarity_gap = 1e-12,
              .max_merit_slope = 1e-24,
          },
  };
}

void initialize(sip::Workspace &workspace, const int num_bound_sides) {
  workspace.vars.x[0] = 0.0;
  workspace.vars.x[1] = 0.0;
  workspace.vars.y[0] = 0.0;
  std::fill_n(workspace.vars.bound_s, num_bound_sides, 1.0);
  std::fill_n(workspace.vars.bound_z, num_bound_sides, 1.0);
}

} // namespace sip_examples::problem_definitions::simple_qp
