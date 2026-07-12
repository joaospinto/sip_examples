#pragma once

#include "sip/sip.hpp"
#include "sip_qdldl/sip_qdldl.hpp"

#include <array>

namespace sip_examples::problem_definitions::simple_nlp {

constexpr int kXDim = 2;
constexpr int kSDim = 2;
constexpr int kYDim = 0;
constexpr int kKktDim = kXDim + kSDim + kYDim;
constexpr int kUpperHessianLagrangianNnz = 3;
constexpr int kJacobianCNnz = 0;
constexpr int kJacobianGNnz = 4;
constexpr int kQdldlKktNnz = 9;
constexpr int kQdldlKktLNnz = 11;
constexpr int kSlacgLNnz = 5;
constexpr std::array<int, kKktDim> kKktPinv = {3, 2, 1, 0};

auto settings() -> sip::Settings;
void initialize(sip::Workspace &workspace);
void configure_qdldl_sparsity(sip_qdldl::ModelCallbackOutput &mco);
void evaluate_qdldl(const sip::ModelCallbackInput &mci, double *f,
                    double *gradient_f, double *g,
                    double *upper_hessian_lagrangian, double *jacobian_g);
void evaluate_slacg(const sip::ModelCallbackInput &mci, double *f,
                    double *gradient_f, double *g,
                    double *upper_hessian_lagrangian, double *jacobian_g);

} // namespace sip_examples::problem_definitions::simple_nlp
