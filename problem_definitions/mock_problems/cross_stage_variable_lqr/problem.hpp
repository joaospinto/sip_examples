#pragma once

#include "sip_optimal_control/sip_optimal_control.hpp"

#include <array>

namespace sip_examples::problem_definitions::cross_stage_variable_lqr {

constexpr int kNumEdges = 1;
constexpr int kStateDim = 1;
constexpr int kControlDim = 1;
constexpr int kCDim = 1;
constexpr int kGDim = 0;
constexpr int kThetaDim = 1;
constexpr int kThetaOffset = kNumEdges * (kStateDim + kControlDim) + kStateDim;

extern const std::array<int, 2> kStateDims;
extern const std::array<int, 1> kControlDims;
extern const std::array<int, 2> kCDims;
extern const std::array<int, 2> kGDims;
extern const std::array<int, 1> kEdgeParents;
extern const std::array<int, 1> kEdgeChildren;
extern const ::sip::optimal_control::Dimensions kDimensions;
extern const ::sip::optimal_control::Topology kTopology;

auto settings() -> sip::Settings;
auto run_solver(const ::sip::optimal_control::Dimensions &dimensions,
                const ::sip::optimal_control::Topology &topology,
                ::sip::optimal_control::Workspace &workspace) -> sip::Output;

} // namespace sip_examples::problem_definitions::cross_stage_variable_lqr
