#pragma once

#include "sip/types.hpp"

#include <vector>

namespace sip_examples::problem_definitions {

class UnitResidualScaling {
public:
  UnitResidualScaling(int x_dim, int s_dim, int y_dim);

  auto get() const -> ::sip::Input::Scaling;

private:
  std::vector<double> variable_;
  std::vector<double> equality_;
  std::vector<double> inequality_;
};

} // namespace sip_examples::problem_definitions
