#pragma once

#include "problem_definitions/casadi_problems/common/problem.hpp"
#include "problem_definitions/ipopt/ipopt.hpp"

#include <vector>

namespace sip_examples::problem_definitions::casadi_problems {

struct GeneratedFlatProblem {
  const FlatProblemSpec *spec;
  CasadiWork (*make_work)();
  void (*evaluate_values)(const double *, double *, double *, double *,
                          CasadiWork &);
  void (*evaluate)(const double *, const double *, const double *, double *,
                   double *, double *, double *, double *, double *, double *,
                   double *, double *, CasadiWork &);
};

class FlatIpoptModel final : public ipopt::Model {
public:
  explicit FlatIpoptModel(const GeneratedFlatProblem &problem);
  ~FlatIpoptModel() override;

  FlatIpoptModel(const FlatIpoptModel &) = delete;
  FlatIpoptModel &operator=(const FlatIpoptModel &) = delete;

  int x_dim() const override;
  int equality_dim() const override;
  int inequality_dim() const override;
  const double *initial_x() const override;
  sip_qdldl::ModelCallbackOutput &output() override;
  void evaluate_values(const double *x) override;
  void evaluate_first_derivatives(const double *x) override;
  void evaluate_lagrangian_hessian(const double *x, const double *y,
                                   const double *z) override;

private:
  void evaluate(const double *x, const double *y, const double *z);

  GeneratedFlatProblem problem_;
  CasadiWork work_;
  sip_qdldl::ModelCallbackOutput output_;
  std::vector<double> y_;
  std::vector<double> z_;
  std::vector<double> jacobian_c_;
  std::vector<double> jacobian_g_;
};

} // namespace sip_examples::problem_definitions::casadi_problems
