#include "problem_definitions/casadi_problems/common/model_signature.hpp"

#include <iomanip>
#include <ostream>
#include <string_view>

namespace sip_examples::problem_definitions::casadi_problems {

auto scaled_sequence(const int size, const double scale)
    -> std::vector<double> {
  std::vector<double> values(size);
  for (int i = 0; i < size; ++i) {
    values[i] =
        scale * static_cast<double>(i + 1) / static_cast<double>(size + 1);
  }
  return values;
}

void add_csc_product(const int rows, const int cols, const int *indices,
                     const int *indptr, const double *data, const double *x,
                     double *y) {
  (void)rows;
  for (int col = 0; col < cols; ++col) {
    for (int index = indptr[col]; index < indptr[col + 1]; ++index) {
      y[indices[index]] += data[index] * x[col];
    }
  }
}

void add_symmetric_upper_csc_product(const int size, const int *indices,
                                     const int *indptr, const double *data,
                                     const double *x, double *y) {
  for (int col = 0; col < size; ++col) {
    for (int index = indptr[col]; index < indptr[col + 1]; ++index) {
      const int row = indices[index];
      y[row] += data[index] * x[col];
      if (row != col) {
        y[col] += data[index] * x[row];
      }
    }
  }
}

namespace {

void print_vector(const std::string_view name,
                  const std::vector<double> &values, std::ostream &stream) {
  stream << name << ' ' << values.size();
  for (const double value : values) {
    stream << ' ' << value;
  }
  stream << '\n';
}

} // namespace

void print_model_signature(const ModelSignature &signature,
                           std::ostream &stream) {
  stream << std::setprecision(17);
  stream << "dimensions 3 " << signature.x_dim << ' ' << signature.y_dim << ' '
         << signature.z_dim << '\n';
  stream << "objective 1 " << signature.f << '\n';
  print_vector("gradient_f", signature.gradient_f, stream);
  print_vector("c", signature.c, stream);
  print_vector("g", signature.g, stream);
  print_vector("hessian_product", signature.hessian_product, stream);
  print_vector("jacobian_c_product", signature.jacobian_c_product, stream);
  print_vector("jacobian_c_transpose_product",
               signature.jacobian_c_transpose_product, stream);
  print_vector("jacobian_g_product", signature.jacobian_g_product, stream);
  print_vector("jacobian_g_transpose_product",
               signature.jacobian_g_transpose_product, stream);
}

} // namespace sip_examples::problem_definitions::casadi_problems
