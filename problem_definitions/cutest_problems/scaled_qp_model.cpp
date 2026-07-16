#include "problem_definitions/cutest_problems/scaled_qp_model.hpp"

#include <algorithm>
#include <cmath>

namespace sip_examples::problem_definitions::cutest_problems {
namespace {

void add_compensated(const double term, double &sum, double &correction) {
  const double next = sum + term;
  correction += std::abs(sum) >= std::abs(term) ? (sum - next) + term
                                                : (term - next) + sum;
  sum = next;
}

} // namespace

QpScaling::QpScaling(const int x_dim, const int y_dim, const int z_dim)
    : primal_(x_dim, 1.0), equality_(y_dim, 1.0), inequality_(z_dim, 1.0),
      row_norm_(x_dim + y_dim + z_dim) {}

void QpScaling::compute(const sip_qdldl::ModelCallbackOutput &output) {
  const int x_dim = static_cast<int>(primal_.size());
  const int y_dim = static_cast<int>(equality_.size());
  const int z_dim = static_cast<int>(inequality_.size());
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
                    hessian.data[index] * primal_[row] * primal_[col]);
      }
    }
    const auto &jacobian_c = output.jacobian_c;
    for (int col = 0; col < y_dim; ++col) {
      for (int index = jacobian_c.indptr[col];
           index < jacobian_c.indptr[col + 1]; ++index) {
        const int row = jacobian_c.ind[index];
        update_norm(row, x_dim + col,
                    jacobian_c.data[index] * primal_[row] * equality_[col]);
      }
    }
    const auto &jacobian_g = output.jacobian_g;
    for (int col = 0; col < z_dim; ++col) {
      for (int index = jacobian_g.indptr[col];
           index < jacobian_g.indptr[col + 1]; ++index) {
        const int row = jacobian_g.ind[index];
        update_norm(row, x_dim + y_dim + col,
                    jacobian_g.data[index] * primal_[row] * inequality_[col]);
      }
    }

    double max_change = 0.0;
    for (double &norm : row_norm_) {
      norm = norm < 1e-4 ? 1.0 : std::min(norm, 1e4);
      norm = 1.0 / std::sqrt(norm);
      max_change = std::max(max_change, std::abs(1.0 - norm));
    }
    for (int i = 0; i < x_dim; ++i) {
      primal_[i] *= row_norm_[i];
    }
    for (int i = 0; i < y_dim; ++i) {
      equality_[i] *= row_norm_[x_dim + i];
    }
    for (int i = 0; i < z_dim; ++i) {
      inequality_[i] *= row_norm_[x_dim + y_dim + i];
    }
    if (max_change <= 1e-3) {
      break;
    }
  }
}

auto QpScaling::has_material_effect() const -> bool {
  double minimum = 1.0;
  double maximum = 1.0;
  const auto include_range = [&](const std::vector<double> &values) {
    if (values.empty()) {
      return;
    }
    const auto [range_minimum, range_maximum] =
        std::minmax_element(values.begin(), values.end());
    minimum = std::min(minimum, *range_minimum);
    maximum = std::max(maximum, *range_maximum);
  };
  include_range(primal_);
  include_range(equality_);
  include_range(inequality_);
  return maximum >= 1e3 * minimum;
}

void QpScaling::set_identity() {
  std::fill(primal_.begin(), primal_.end(), 1.0);
  std::fill(equality_.begin(), equality_.end(), 1.0);
  std::fill(inequality_.begin(), inequality_.end(), 1.0);
}

void QpScaling::scale_derivatives(
    sip_qdldl::ModelCallbackOutput &output) const {
  for (int i = 0; i < static_cast<int>(primal_.size()); ++i) {
    output.gradient_f[i] *= primal_[i];
  }
  auto &hessian = output.upper_hessian_lagrangian;
  for (int col = 0; col < hessian.cols; ++col) {
    for (int index = hessian.indptr[col]; index < hessian.indptr[col + 1];
         ++index) {
      hessian.data[index] *= primal_[hessian.ind[index]] * primal_[col];
    }
  }
  auto &jacobian_c = output.jacobian_c;
  for (int col = 0; col < jacobian_c.cols; ++col) {
    for (int index = jacobian_c.indptr[col]; index < jacobian_c.indptr[col + 1];
         ++index) {
      jacobian_c.data[index] *= equality_[col] * primal_[jacobian_c.ind[index]];
    }
  }
  auto &jacobian_g = output.jacobian_g;
  for (int col = 0; col < jacobian_g.cols; ++col) {
    for (int index = jacobian_g.indptr[col]; index < jacobian_g.indptr[col + 1];
         ++index) {
      jacobian_g.data[index] *=
          inequality_[col] * primal_[jacobian_g.ind[index]];
    }
  }
}

void QpScaling::to_scaled_primal(const double *original, double *scaled) const {
  for (int i = 0; i < static_cast<int>(primal_.size()); ++i) {
    scaled[i] = original[i] / primal_[i];
  }
}

auto QpScaling::primal() const -> const std::vector<double> & {
  return primal_;
}

auto QpScaling::equality() const -> const std::vector<double> & {
  return equality_;
}

auto QpScaling::inequality() const -> const std::vector<double> & {
  return inequality_;
}

ScaledQpModel::ScaledQpModel(sip_qdldl::ModelCallbackOutput &output,
                             const QpScaling &scaling)
    : scaled_linear_(output.gradient_f,
                     output.gradient_f + scaling.primal().size()),
      scaled_constant_(output.f), equality_rhs_(scaling.equality().size()),
      inequality_rhs_(scaling.inequality().size()),
      jacobian_c_(output.jacobian_c.data,
                  output.jacobian_c.data +
                      output.jacobian_c.indptr[output.jacobian_c.cols]),
      jacobian_g_(output.jacobian_g.data,
                  output.jacobian_g.data +
                      output.jacobian_g.indptr[output.jacobian_g.cols]) {
  for (int i = 0; i < static_cast<int>(scaled_linear_.size()); ++i) {
    scaled_linear_[i] *= scaling.primal()[i];
  }
  for (int i = 0; i < static_cast<int>(equality_rhs_.size()); ++i) {
    equality_rhs_[i] = -output.c[i];
  }
  for (int i = 0; i < static_cast<int>(inequality_rhs_.size()); ++i) {
    inequality_rhs_[i] = -output.g[i];
  }
  scaling.scale_derivatives(output);
}

void ScaledQpModel::evaluate_values(
    const double *x, const QpScaling &scaling,
    sip_qdldl::ModelCallbackOutput &output) const {
  std::copy(scaled_linear_.begin(), scaled_linear_.end(), output.gradient_f);
  sip_qdldl::add_Ax_to_y_where_A_upper_symmetric(
      output.upper_hessian_lagrangian, x, output.gradient_f);

  double objective = scaled_constant_;
  double correction = 0.0;
  for (int i = 0; i < static_cast<int>(scaled_linear_.size()); ++i) {
    add_compensated(x[i] * (scaled_linear_[i] + output.gradient_f[i]) / 2.0,
                    objective, correction);
  }
  output.f = objective + correction;
  evaluate_affine_rows(output.jacobian_c, jacobian_c_, equality_rhs_,
                       scaling.primal(), scaling.equality(), x, output.c);
  evaluate_affine_rows(output.jacobian_g, jacobian_g_, inequality_rhs_,
                       scaling.primal(), scaling.inequality(), x, output.g);
}

void ScaledQpModel::evaluate_affine_rows(
    const sip_qdldl::SparseMatrix &matrix, const std::vector<double> &data,
    const std::vector<double> &rhs, const std::vector<double> &variable_scale,
    const std::vector<double> &row_scale, const double *x, double *values) {
  for (int row = 0; row < matrix.cols; ++row) {
    double value = -rhs[row];
    double correction = 0.0;
    for (int index = matrix.indptr[row]; index < matrix.indptr[row + 1];
         ++index) {
      const int variable = matrix.ind[index];
      add_compensated(data[index] * variable_scale[variable] * x[variable],
                      value, correction);
    }
    values[row] = row_scale[row] * (value + correction);
  }
}

} // namespace sip_examples::problem_definitions::cutest_problems
