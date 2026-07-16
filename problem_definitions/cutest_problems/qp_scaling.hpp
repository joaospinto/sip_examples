#pragma once

#include "sip_qdldl/sip_qdldl.hpp"

#include <vector>

namespace sip_examples::problem_definitions::cutest_problems {

struct QpResiduals {
  double primal;
  double dual;
  double complementarity;
};

class ScaledQp {
public:
  ScaledQp(sip_qdldl::ModelCallbackOutput &output, const double *lower_bounds,
           const double *upper_bounds);

  const std::vector<double> &primal_variable_scaling() const;
  const std::vector<double> &equality_residual_scaling() const;
  const std::vector<double> &inequality_residual_scaling() const;
  const std::vector<double> &bound_residual_scaling() const;
  const std::vector<double> &scaled_lower_bounds() const;
  const std::vector<double> &scaled_upper_bounds() const;

  void to_scaled_primal(const double *original, double *scaled) const;
  void evaluate_values(const double *scaled_x,
                       sip_qdldl::ModelCallbackOutput &output) const;
  auto residuals(const double *scaled_x, const double *scaled_y,
                 const double *scaled_z, const double *slacks,
                 const double *bound_slacks, const double *bound_duals,
                 sip_qdldl::ModelCallbackOutput &output) const -> QpResiduals;

private:
  void compute_scaling(const sip_qdldl::ModelCallbackOutput &output);
  void scale_derivatives(sip_qdldl::ModelCallbackOutput &output) const;
  void evaluate_constraints(const double *scaled_x,
                            sip_qdldl::ModelCallbackOutput &output) const;

  std::vector<double> x_scaling_;
  std::vector<double> equality_scaling_;
  std::vector<double> inequality_scaling_;
  std::vector<double> bound_scaling_;
  std::vector<double> row_norm_;
  std::vector<double> scaled_lower_bounds_;
  std::vector<double> scaled_upper_bounds_;
  std::vector<double> scaled_linear_;
  double scaled_constant_;
  std::vector<double> equality_rhs_;
  std::vector<double> inequality_rhs_;
  std::vector<double> jacobian_c_;
  std::vector<double> jacobian_g_;
};

} // namespace sip_examples::problem_definitions::cutest_problems
