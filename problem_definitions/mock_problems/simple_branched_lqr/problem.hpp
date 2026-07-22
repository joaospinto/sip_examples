#pragma once

#include "sip/sip.hpp"
#include "sip_optimal_control/sip_optimal_control.hpp"
#include "sip_qdldl/sip_qdldl.hpp"

#include <array>

namespace sip_examples::problem_definitions::simple_branched_lqr {

constexpr int kNumEdges = 2;
constexpr int kNumNodes = kNumEdges + 1;
constexpr int kStateDim = 1;
constexpr int kControlDim = 1;
constexpr int kCDim = 1;
constexpr int kGDim = 0;

constexpr std::array<int, kNumEdges> kEdgeParents = {0, 0};
constexpr std::array<int, kNumEdges> kEdgeChildren = {1, 2};
constexpr std::array<int, kNumNodes> kNodeCDims = {0, 1, 1};
constexpr std::array<int, kNumNodes> kNodeGDims = {0, 0, 0};
constexpr std::array<int, kNumEdges> kEdgeCDims = {0, 0};
constexpr std::array<int, kNumEdges> kEdgeGDims = {0, 0};
constexpr std::array<double, kStateDim> kInitialState = {0.0};

constexpr int kXDim = 5;
constexpr int kSDim = 0;
constexpr int kYDim = 6;
constexpr int kOcpYDim = 5;
constexpr int kKktDim = kXDim + kSDim + kYDim;
constexpr int kUpperHessianLagrangianNnz = 5;
constexpr int kJacobianCNnz = 9;
constexpr int kJacobianGNnz = 0;
constexpr int kQdldlKktNnz = kUpperHessianLagrangianNnz + kJacobianCNnz + kYDim;
constexpr int kQdldlKktLNnz = kKktDim * (kKktDim - 1) / 2;
constexpr std::array<double, kXDim> kSolution = {0.0, 1.0, 1.0, -1.0, -1.0};

auto settings() -> sip::Settings;
void initialize(sip::Workspace &workspace);
void configure_qdldl_sparsity(sip_qdldl::ModelCallbackOutput &mco);

void evaluate_qdldl(const sip::ModelCallbackInput &mci, double *f,
                    double *gradient_f, double *c,
                    double *upper_hessian_lagrangian, double *jacobian_c);
void evaluate_slacg(const sip::ModelCallbackInput &mci, double *f,
                    double *gradient_f, double *c,
                    double *upper_hessian_lagrangian, double *jacobian_c);

void evaluate_optimal_control(
    const sip::optimal_control::ModelCallbackInput &mci,
    sip::optimal_control::ModelCallbackOutput &mco);
void initialize(sip::optimal_control::Workspace &workspace);

} // namespace sip_examples::problem_definitions::simple_branched_lqr
