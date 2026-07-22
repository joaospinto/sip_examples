#pragma once

#include "sip_optimal_control/types.hpp"

namespace sip_examples::problem_definitions::casadi_problems {

struct OcpProblemSpec {
  const char *name;
  int num_edges;
  ::sip::optimal_control::Dimensions dimensions;
  int topology_root;
  const int *edge_parents;
  const int *edge_children;
  const double *initial_state;
  const double *initial_x;
  const double *lower_bounds;
  const double *upper_bounds;
};

} // namespace sip_examples::problem_definitions::casadi_problems
