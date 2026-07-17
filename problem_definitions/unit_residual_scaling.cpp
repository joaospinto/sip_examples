#include "problem_definitions/unit_residual_scaling.hpp"

namespace sip_examples::problem_definitions {

UnitResidualScaling::UnitResidualScaling(const int x_dim, const int s_dim,
                                         const int y_dim)
    : dual_(x_dim, 1.0), equality_(y_dim, 1.0), inequality_(s_dim, 1.0),
      variable_bound_(x_dim, 1.0) {}

auto UnitResidualScaling::get() const -> ::sip::Input::ResidualScaling {
  return {
      .dual = dual_.data(),
      .equality = equality_.data(),
      .inequality = inequality_.data(),
      .variable_bound = variable_bound_.data(),
  };
}

} // namespace sip_examples::problem_definitions
