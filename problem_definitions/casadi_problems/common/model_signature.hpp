#pragma once

#include <iosfwd>
#include <vector>

namespace sip_examples::problem_definitions::casadi_problems {

struct ModelSignature {
  int x_dim;
  int y_dim;
  int z_dim;
  double f;
  std::vector<double> gradient_f;
  std::vector<double> c;
  std::vector<double> g;
  std::vector<double> hessian_product;
  std::vector<double> jacobian_c_product;
  std::vector<double> jacobian_c_transpose_product;
  std::vector<double> jacobian_g_product;
  std::vector<double> jacobian_g_transpose_product;
};

auto scaled_sequence(int size, double scale) -> std::vector<double>;

void add_csc_product(int rows, int cols, const int *indices, const int *indptr,
                     const double *data, const double *x, double *y);

void add_symmetric_upper_csc_product(int size, const int *indices,
                                     const int *indptr, const double *data,
                                     const double *x, double *y);

void print_model_signature(const ModelSignature &signature,
                           std::ostream &stream);

} // namespace sip_examples::problem_definitions::casadi_problems
