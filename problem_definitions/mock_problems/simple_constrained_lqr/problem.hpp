#pragma once

#include "sip_optimal_control/sip_optimal_control.hpp"

#include <array>

namespace sip_examples::problem_definitions::simple_constrained_lqr {

constexpr double kDt = 0.1;
constexpr int kStateDim = 2;
constexpr int kControlDim = 1;
constexpr int kNumEdges = 100;
constexpr int kCDim = 1;
constexpr int kGDim = 2;

extern const std::array<int, kNumEdges + 1> kStateDims;
extern const std::array<int, kNumEdges> kControlDims;
extern const std::array<int, kNumEdges + 1> kCDims;
extern const std::array<int, kNumEdges + 1> kGDims;
extern const std::array<int, kNumEdges> kEdgeParents;
extern const std::array<int, kNumEdges> kEdgeChildren;
extern const ::sip::optimal_control::Dimensions kDimensions;
extern const ::sip::optimal_control::Topology kTopology;

auto settings() -> sip::Settings;
auto run_solver(const ::sip::optimal_control::Dimensions &dimensions,
                const ::sip::optimal_control::Topology &topology,
                ::sip::optimal_control::Workspace &workspace) -> sip::Output;

} // namespace sip_examples::problem_definitions::simple_constrained_lqr
