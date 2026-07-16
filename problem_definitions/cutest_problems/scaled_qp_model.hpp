#pragma once

#include "sip_qdldl/sip_qdldl.hpp"

#include <vector>

namespace sip_examples::problem_definitions::cutest_problems {

class QpScaling {
public:
  QpScaling(int x_dim, int y_dim, int z_dim);

  void compute(const sip_qdldl::ModelCallbackOutput &output);
  auto has_material_effect() const -> bool;
  void set_identity();
  void scale_derivatives(sip_qdldl::ModelCallbackOutput &output) const;
  void to_scaled_primal(const double *original, double *scaled) const;

  auto primal() const -> const std::vector<double> &;
  auto equality() const -> const std::vector<double> &;
  auto inequality() const -> const std::vector<double> &;

private:
  std::vector<double> primal_;
  std::vector<double> equality_;
  std::vector<double> inequality_;
  std::vector<double> row_norm_;
};

class ScaledQpModel {
public:
  ScaledQpModel(sip_qdldl::ModelCallbackOutput &output,
                const QpScaling &scaling);

  void evaluate_values(const double *x, const QpScaling &scaling,
                       sip_qdldl::ModelCallbackOutput &output) const;

private:
  static void evaluate_affine_rows(const sip_qdldl::SparseMatrix &matrix,
                                   const std::vector<double> &data,
                                   const std::vector<double> &rhs,
                                   const std::vector<double> &variable_scale,
                                   const std::vector<double> &row_scale,
                                   const double *x, double *values);

  std::vector<double> scaled_linear_;
  double scaled_constant_;
  std::vector<double> equality_rhs_;
  std::vector<double> inequality_rhs_;
  std::vector<double> jacobian_c_;
  std::vector<double> jacobian_g_;
};

} // namespace sip_examples::problem_definitions::cutest_problems
