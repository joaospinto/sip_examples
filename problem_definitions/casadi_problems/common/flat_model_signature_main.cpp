#include GENERATED_HEADER

#include "problem_definitions/casadi_problems/common/model_signature.hpp"

#include <algorithm>
#include <iostream>
#include <vector>

namespace casadi_problems = sip_examples::problem_definitions::casadi_problems;
using casadi_problems::generated_problem::Problem;

auto main() -> int {
  const auto &spec = Problem::flat_spec();
  auto work = Problem::make_flat_work();

  std::vector<double> x(spec.initial_x, spec.initial_x + spec.x_dim);
  const auto y = casadi_problems::scaled_sequence(spec.y_dim, 0.02);
  const auto z = casadi_problems::scaled_sequence(spec.s_dim, 0.03);
  const auto x_direction = casadi_problems::scaled_sequence(spec.x_dim, 0.01);
  const auto y_direction = casadi_problems::scaled_sequence(spec.y_dim, -0.02);
  const auto z_direction = casadi_problems::scaled_sequence(spec.s_dim, -0.03);

  casadi_problems::ModelSignature signature{
      .x_dim = spec.x_dim,
      .y_dim = spec.y_dim,
      .z_dim = spec.s_dim,
      .gradient_f = std::vector<double>(spec.x_dim),
      .c = std::vector<double>(spec.y_dim),
      .g = std::vector<double>(spec.s_dim),
      .hessian_product = std::vector<double>(spec.x_dim),
      .jacobian_c_product = std::vector<double>(spec.y_dim),
      .jacobian_c_transpose_product = std::vector<double>(spec.x_dim),
      .jacobian_g_product = std::vector<double>(spec.s_dim),
      .jacobian_g_transpose_product = std::vector<double>(spec.x_dim),
  };
  std::vector<double> hessian(spec.upper_hessian_lagrangian_nnz);
  std::vector<double> jacobian_c(spec.jacobian_c_nnz);
  std::vector<double> jacobian_g(spec.jacobian_g_nnz);
  std::vector<double> jacobian_c_transpose(spec.jacobian_c_transpose_nnz);
  std::vector<double> jacobian_g_transpose(spec.jacobian_g_transpose_nnz);

  Problem::eval_flat_qdldl(
      x.data(), y.data(), z.data(), &signature.f, signature.gradient_f.data(),
      signature.c.data(), signature.g.data(), hessian.data(), jacobian_c.data(),
      jacobian_g.data(), jacobian_c_transpose.data(),
      jacobian_g_transpose.data(), work);

  casadi_problems::add_symmetric_upper_csc_product(
      spec.x_dim, spec.upper_hessian_lagrangian_ind,
      spec.upper_hessian_lagrangian_indptr, hessian.data(), x_direction.data(),
      signature.hessian_product.data());
  casadi_problems::add_csc_product(spec.y_dim, spec.x_dim, spec.jacobian_c_ind,
                                   spec.jacobian_c_indptr, jacobian_c.data(),
                                   x_direction.data(),
                                   signature.jacobian_c_product.data());
  casadi_problems::add_csc_product(
      spec.x_dim, spec.y_dim, spec.jacobian_c_transpose_ind,
      spec.jacobian_c_transpose_indptr, jacobian_c_transpose.data(),
      y_direction.data(), signature.jacobian_c_transpose_product.data());
  casadi_problems::add_csc_product(spec.s_dim, spec.x_dim, spec.jacobian_g_ind,
                                   spec.jacobian_g_indptr, jacobian_g.data(),
                                   x_direction.data(),
                                   signature.jacobian_g_product.data());
  casadi_problems::add_csc_product(
      spec.x_dim, spec.s_dim, spec.jacobian_g_transpose_ind,
      spec.jacobian_g_transpose_indptr, jacobian_g_transpose.data(),
      z_direction.data(), signature.jacobian_g_transpose_product.data());

  casadi_problems::print_model_signature(signature, std::cout);
  return 0;
}
