#include "problem_definitions/casadi_problems/common/flat_ipopt_model.hpp"

#include <algorithm>

namespace sip_examples::problem_definitions::casadi_problems {
namespace {

void copy_pattern(int nnz, int columns, const int *indices,
                  const int *column_pointers, sip_qdldl::SparseMatrix &matrix) {
  std::copy_n(indices, nnz, matrix.ind);
  std::copy_n(column_pointers, columns + 1, matrix.indptr);
}

} // namespace

FlatIpoptModel::FlatIpoptModel(const GeneratedFlatProblem &problem)
    : problem_(problem), work_(problem.make_work()),
      y_(problem.spec->y_dim, 0.0), z_(problem.spec->s_dim, 0.0),
      jacobian_c_(problem.spec->jacobian_c_nnz),
      jacobian_g_(problem.spec->jacobian_g_nnz) {
  const auto &spec = *problem_.spec;
  output_.reserve(
      spec.x_dim, spec.s_dim, spec.y_dim, spec.upper_hessian_lagrangian_nnz,
      spec.jacobian_c_transpose_nnz, spec.jacobian_g_transpose_nnz, true, true);

  output_.upper_hessian_lagrangian.rows = spec.x_dim;
  output_.upper_hessian_lagrangian.cols = spec.x_dim;
  output_.upper_hessian_lagrangian.is_transposed = false;
  copy_pattern(spec.upper_hessian_lagrangian_nnz, spec.x_dim,
               spec.upper_hessian_lagrangian_ind,
               spec.upper_hessian_lagrangian_indptr,
               output_.upper_hessian_lagrangian);

  output_.jacobian_c.rows = spec.x_dim;
  output_.jacobian_c.cols = spec.y_dim;
  output_.jacobian_c.is_transposed = true;
  copy_pattern(spec.jacobian_c_transpose_nnz, spec.y_dim,
               spec.jacobian_c_transpose_ind, spec.jacobian_c_transpose_indptr,
               output_.jacobian_c);

  output_.jacobian_g.rows = spec.x_dim;
  output_.jacobian_g.cols = spec.s_dim;
  output_.jacobian_g.is_transposed = true;
  copy_pattern(spec.jacobian_g_transpose_nnz, spec.s_dim,
               spec.jacobian_g_transpose_ind, spec.jacobian_g_transpose_indptr,
               output_.jacobian_g);
}

FlatIpoptModel::~FlatIpoptModel() { output_.free(); }

int FlatIpoptModel::x_dim() const { return problem_.spec->x_dim; }

int FlatIpoptModel::equality_dim() const { return problem_.spec->y_dim; }

int FlatIpoptModel::inequality_dim() const { return problem_.spec->s_dim; }

const double *FlatIpoptModel::initial_x() const {
  return problem_.spec->initial_x;
}

sip_qdldl::ModelCallbackOutput &FlatIpoptModel::output() { return output_; }

void FlatIpoptModel::evaluate_values(const double *x) {
  problem_.evaluate_values(x, &output_.f, output_.c, output_.g, work_);
}

void FlatIpoptModel::evaluate_first_derivatives(const double *x) {
  std::fill(y_.begin(), y_.end(), 0.0);
  std::fill(z_.begin(), z_.end(), 0.0);
  evaluate(x, y_.data(), z_.data());
}

void FlatIpoptModel::evaluate_lagrangian_hessian(const double *x,
                                                 const double *y,
                                                 const double *z) {
  evaluate(x, y, z);
}

void FlatIpoptModel::evaluate(const double *x, const double *y,
                              const double *z) {
  problem_.evaluate(x, y, z, &output_.f, output_.gradient_f, output_.c,
                    output_.g, output_.upper_hessian_lagrangian.data,
                    jacobian_c_.data(), jacobian_g_.data(),
                    output_.jacobian_c.data, output_.jacobian_g.data, work_);
}

} // namespace sip_examples::problem_definitions::casadi_problems
