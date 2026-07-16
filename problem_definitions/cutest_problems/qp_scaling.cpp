#include "problem_definitions/cutest_problems/qp_scaling.hpp"

#include <algorithm>
#include <cmath>
#include <limits>

namespace sip_examples::problem_definitions::cutest_problems {
namespace {

void add_compensated(const double term, double &sum, double &correction) {
  const double next = sum + term;
  correction += std::abs(sum) >= std::abs(term) ? (sum - next) + term
                                                : (term - next) + sum;
  sum = next;
}

auto finite_max(const double current, const double candidate) -> double {
  return std::isfinite(candidate) ? std::max(current, candidate)
                                  : std::numeric_limits<double>::infinity();
}

void evaluate_affine_rows(const sip_qdldl::SparseMatrix &matrix,
                          const std::vector<double> &data,
                          const std::vector<double> &rhs,
                          const std::vector<double> &variable_scaling,
                          const std::vector<double> &row_scaling,
                          const double *x, double *values) {
  for (int row = 0; row < matrix.cols; ++row) {
    double value = -rhs[row];
    double correction = 0.0;
    for (int index = matrix.indptr[row]; index < matrix.indptr[row + 1];
         ++index) {
      const int variable = matrix.ind[index];
      add_compensated(data[index] * variable_scaling[variable] * x[variable],
                      value, correction);
    }
    values[row] = row_scaling[row] * (value + correction);
  }
}

} // namespace

ScaledQp::ScaledQp(sip_qdldl::ModelCallbackOutput &output,
                   const double *lower_bounds, const double *upper_bounds)
    : x_scaling_(output.upper_hessian_lagrangian.cols, 1.0),
      equality_scaling_(output.jacobian_c.cols, 1.0),
      inequality_scaling_(output.jacobian_g.cols, 1.0),
      row_norm_(x_scaling_.size() + equality_scaling_.size() +
                inequality_scaling_.size()),
      scaled_lower_bounds_(x_scaling_.size()),
      scaled_upper_bounds_(x_scaling_.size()),
      scaled_linear_(output.gradient_f, output.gradient_f + x_scaling_.size()),
      scaled_constant_(output.f), equality_rhs_(equality_scaling_.size()),
      inequality_rhs_(inequality_scaling_.size()),
      jacobian_c_(output.jacobian_c.data,
                  output.jacobian_c.data +
                      output.jacobian_c.indptr[output.jacobian_c.cols]),
      jacobian_g_(output.jacobian_g.data,
                  output.jacobian_g.data +
                      output.jacobian_g.indptr[output.jacobian_g.cols]) {
  compute_scaling(output);
  for (int i = 0; i < static_cast<int>(x_scaling_.size()); ++i) {
    scaled_lower_bounds_[i] = lower_bounds[i] / x_scaling_[i];
    scaled_upper_bounds_[i] = upper_bounds[i] / x_scaling_[i];
    if (std::isfinite(lower_bounds[i])) {
      bound_scaling_.push_back(1.0 / x_scaling_[i]);
    }
    if (std::isfinite(upper_bounds[i])) {
      bound_scaling_.push_back(1.0 / x_scaling_[i]);
    }
    scaled_linear_[i] *= x_scaling_[i];
  }
  for (int i = 0; i < static_cast<int>(equality_rhs_.size()); ++i) {
    equality_rhs_[i] = -output.c[i];
  }
  for (int i = 0; i < static_cast<int>(inequality_rhs_.size()); ++i) {
    inequality_rhs_[i] = -output.g[i];
  }
  scale_derivatives(output);
}

const std::vector<double> &ScaledQp::primal_variable_scaling() const {
  return x_scaling_;
}

const std::vector<double> &ScaledQp::equality_residual_scaling() const {
  return equality_scaling_;
}

const std::vector<double> &ScaledQp::inequality_residual_scaling() const {
  return inequality_scaling_;
}

const std::vector<double> &ScaledQp::bound_residual_scaling() const {
  return bound_scaling_;
}

const std::vector<double> &ScaledQp::scaled_lower_bounds() const {
  return scaled_lower_bounds_;
}

const std::vector<double> &ScaledQp::scaled_upper_bounds() const {
  return scaled_upper_bounds_;
}

void ScaledQp::compute_scaling(const sip_qdldl::ModelCallbackOutput &output) {
  const int x_dim = static_cast<int>(x_scaling_.size());
  const int y_dim = static_cast<int>(equality_scaling_.size());
  const int z_dim = static_cast<int>(inequality_scaling_.size());
  for (int iteration = 0; iteration < 10; ++iteration) {
    std::fill(row_norm_.begin(), row_norm_.end(), 0.0);
    const auto update_norm = [&](const int lhs, const int rhs,
                                 const double value) {
      const double magnitude = std::abs(value);
      row_norm_[lhs] = std::max(row_norm_[lhs], magnitude);
      row_norm_[rhs] = std::max(row_norm_[rhs], magnitude);
    };
    const auto &hessian = output.upper_hessian_lagrangian;
    for (int col = 0; col < x_dim; ++col) {
      for (int index = hessian.indptr[col]; index < hessian.indptr[col + 1];
           ++index) {
        const int row = hessian.ind[index];
        update_norm(row, col,
                    hessian.data[index] * x_scaling_[row] * x_scaling_[col]);
      }
    }
    const auto &jacobian_c = output.jacobian_c;
    for (int col = 0; col < y_dim; ++col) {
      for (int index = jacobian_c.indptr[col];
           index < jacobian_c.indptr[col + 1]; ++index) {
        const int row = jacobian_c.ind[index];
        update_norm(row, x_dim + col,
                    jacobian_c.data[index] * x_scaling_[row] *
                        equality_scaling_[col]);
      }
    }
    const auto &jacobian_g = output.jacobian_g;
    for (int col = 0; col < z_dim; ++col) {
      for (int index = jacobian_g.indptr[col];
           index < jacobian_g.indptr[col + 1]; ++index) {
        const int row = jacobian_g.ind[index];
        update_norm(row, x_dim + y_dim + col,
                    jacobian_g.data[index] * x_scaling_[row] *
                        inequality_scaling_[col]);
      }
    }

    double max_change = 0.0;
    for (double &norm : row_norm_) {
      norm = norm < 1e-4 ? 1.0 : std::min(norm, 1e4);
      norm = 1.0 / std::sqrt(norm);
      max_change = std::max(max_change, std::abs(1.0 - norm));
    }
    for (int i = 0; i < x_dim; ++i) {
      x_scaling_[i] *= row_norm_[i];
    }
    for (int i = 0; i < y_dim; ++i) {
      equality_scaling_[i] *= row_norm_[x_dim + i];
    }
    for (int i = 0; i < z_dim; ++i) {
      inequality_scaling_[i] *= row_norm_[x_dim + y_dim + i];
    }
    if (max_change <= 1e-3) {
      break;
    }
  }
}

void ScaledQp::scale_derivatives(sip_qdldl::ModelCallbackOutput &output) const {
  for (int i = 0; i < static_cast<int>(x_scaling_.size()); ++i) {
    output.gradient_f[i] *= x_scaling_[i];
  }
  auto &hessian = output.upper_hessian_lagrangian;
  for (int col = 0; col < hessian.cols; ++col) {
    for (int index = hessian.indptr[col]; index < hessian.indptr[col + 1];
         ++index) {
      output.upper_hessian_lagrangian.data[index] *=
          x_scaling_[hessian.ind[index]] * x_scaling_[col];
    }
  }
  auto &jacobian_c = output.jacobian_c;
  for (int col = 0; col < jacobian_c.cols; ++col) {
    for (int index = jacobian_c.indptr[col]; index < jacobian_c.indptr[col + 1];
         ++index) {
      jacobian_c.data[index] *=
          equality_scaling_[col] * x_scaling_[jacobian_c.ind[index]];
    }
  }
  auto &jacobian_g = output.jacobian_g;
  for (int col = 0; col < jacobian_g.cols; ++col) {
    for (int index = jacobian_g.indptr[col]; index < jacobian_g.indptr[col + 1];
         ++index) {
      jacobian_g.data[index] *=
          inequality_scaling_[col] * x_scaling_[jacobian_g.ind[index]];
    }
  }
}

void ScaledQp::to_scaled_primal(const double *original, double *scaled) const {
  for (int i = 0; i < static_cast<int>(x_scaling_.size()); ++i) {
    scaled[i] = original[i] / x_scaling_[i];
  }
}

void ScaledQp::evaluate_values(const double *scaled_x,
                               sip_qdldl::ModelCallbackOutput &output) const {
  std::copy(scaled_linear_.begin(), scaled_linear_.end(), output.gradient_f);
  sip_qdldl::add_Ax_to_y_where_A_upper_symmetric(
      output.upper_hessian_lagrangian, scaled_x, output.gradient_f);

  double objective = scaled_constant_;
  double correction = 0.0;
  for (int i = 0; i < static_cast<int>(scaled_linear_.size()); ++i) {
    add_compensated(scaled_x[i] * (scaled_linear_[i] + output.gradient_f[i]) /
                        2.0,
                    objective, correction);
  }
  output.f = objective + correction;
  evaluate_constraints(scaled_x, output);
}

void ScaledQp::evaluate_constraints(
    const double *scaled_x, sip_qdldl::ModelCallbackOutput &output) const {
  evaluate_affine_rows(output.jacobian_c, jacobian_c_, equality_rhs_,
                       x_scaling_, equality_scaling_, scaled_x, output.c);
  evaluate_affine_rows(output.jacobian_g, jacobian_g_, inequality_rhs_,
                       x_scaling_, inequality_scaling_, scaled_x, output.g);
}

auto ScaledQp::residuals(const double *scaled_x, const double *scaled_y,
                         const double *scaled_z, const double *slacks,
                         const double *bound_slacks, const double *bound_duals,
                         sip_qdldl::ModelCallbackOutput &output) const
    -> QpResiduals {
  evaluate_values(scaled_x, output);

  std::vector<double> dual(output.gradient_f,
                           output.gradient_f + x_scaling_.size());
  std::vector<double> dual_correction(x_scaling_.size(), 0.0);
  const auto add_transpose_product = [&](const sip_qdldl::SparseMatrix &matrix,
                                         const double *multipliers) {
    for (int col = 0; col < matrix.cols; ++col) {
      for (int index = matrix.indptr[col]; index < matrix.indptr[col + 1];
           ++index) {
        const int row = matrix.ind[index];
        add_compensated(matrix.data[index] * multipliers[col], dual[row],
                        dual_correction[row]);
      }
    }
  };
  add_transpose_product(output.jacobian_c, scaled_y);
  add_transpose_product(output.jacobian_g, scaled_z);

  int side_index = 0;
  for (int i = 0; i < static_cast<int>(x_scaling_.size()); ++i) {
    if (std::isfinite(scaled_lower_bounds_[i])) {
      add_compensated(-bound_duals[side_index], dual[i], dual_correction[i]);
      ++side_index;
    }
    if (std::isfinite(scaled_upper_bounds_[i])) {
      add_compensated(bound_duals[side_index], dual[i], dual_correction[i]);
      ++side_index;
    }
  }

  QpResiduals result{0.0, 0.0, 0.0};
  for (int i = 0; i < static_cast<int>(x_scaling_.size()); ++i) {
    result.dual = finite_max(
        result.dual, std::abs((dual[i] + dual_correction[i]) / x_scaling_[i]));
  }
  for (int i = 0; i < static_cast<int>(equality_scaling_.size()); ++i) {
    result.primal =
        finite_max(result.primal, std::abs(output.c[i] / equality_scaling_[i]));
  }
  for (int i = 0; i < static_cast<int>(inequality_scaling_.size()); ++i) {
    result.primal =
        finite_max(result.primal, output.g[i] / inequality_scaling_[i]);
    result.complementarity =
        finite_max(result.complementarity, std::abs(slacks[i] * scaled_z[i]));
  }
  side_index = 0;
  for (int i = 0; i < static_cast<int>(x_scaling_.size()); ++i) {
    if (std::isfinite(scaled_lower_bounds_[i])) {
      result.primal =
          finite_max(result.primal, (scaled_lower_bounds_[i] - scaled_x[i]) /
                                        bound_scaling_[side_index]);
      result.complementarity =
          finite_max(result.complementarity, std::abs(bound_slacks[side_index] *
                                                      bound_duals[side_index]));
      ++side_index;
    }
    if (std::isfinite(scaled_upper_bounds_[i])) {
      result.primal =
          finite_max(result.primal, (scaled_x[i] - scaled_upper_bounds_[i]) /
                                        bound_scaling_[side_index]);
      result.complementarity =
          finite_max(result.complementarity, std::abs(bound_slacks[side_index] *
                                                      bound_duals[side_index]));
      ++side_index;
    }
  }
  return result;
}

} // namespace sip_examples::problem_definitions::cutest_problems
