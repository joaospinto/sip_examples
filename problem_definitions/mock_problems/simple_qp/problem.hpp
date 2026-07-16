#pragma once

#include "sip/sip.hpp"
#include "sip_qdldl/sip_qdldl.hpp"

#include <array>

namespace sip_examples::problem_definitions::simple_qp {

constexpr int kXDim = 2;
constexpr int kSDim = 0;
constexpr int kYDim = 1;
constexpr int kKktDim = kXDim + kSDim + kYDim;
constexpr int kUpperHessianLagrangianNnz = 3;
constexpr int kJacobianCNnz = 2;
constexpr int kJacobianGNnz = 0;
constexpr int kQdldlKktNnz = 6;
constexpr int kQdldlKktLNnz = 3;
constexpr std::array<int, kKktDim> kKktPinv = {2, 1, 0};
constexpr std::array<double, kXDim> kLowerBounds = {0.0, 0.0};
constexpr std::array<double, kXDim> kUpperBounds = {0.7, 0.7};

auto settings() -> sip::Settings;
void initialize(sip::Workspace &workspace, int num_bound_sides);
void configure_qdldl_sparsity(sip_qdldl::ModelCallbackOutput &mco);
void evaluate(const sip::ModelCallbackInput &mci, double *f, double *gradient_f,
              double *c, double *upper_hessian_lagrangian, double *jacobian_c);

} // namespace sip_examples::problem_definitions::simple_qp
