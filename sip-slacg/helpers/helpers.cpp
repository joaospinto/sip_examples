#include "sip-slacg/helpers/helpers.hpp"

namespace sip_examples {

void ModelCallbackOutput::reserve(int x_dim, int s_dim, int y_dim,
                                  int upper_hessian_lagrangian_nnz,
                                  int jacobian_c_nnz, int jacobian_g_nnz) {
  gradient_f = new double[x_dim];
  upper_hessian_lagrangian = new double[upper_hessian_lagrangian_nnz];
  c = new double[y_dim];
  jacobian_c = new double[jacobian_c_nnz];
  g = new double[s_dim];
  jacobian_g = new double[jacobian_g_nnz];
}

void ModelCallbackOutput::free() {
  delete[] gradient_f;
  delete[] upper_hessian_lagrangian;
  delete[] c;
  delete[] jacobian_c;
  delete[] g;
  delete[] jacobian_g;
}

auto ModelCallbackOutput::mem_assign(int x_dim, int s_dim, int y_dim,
                                     int upper_hessian_lagrangian_nnz,
                                     int jacobian_c_nnz, int jacobian_g_nnz,
                                     unsigned char *mem_ptr) -> int {
  int cum_size = 0;
  gradient_f = reinterpret_cast<double *>(mem_ptr + cum_size);
  cum_size += x_dim * sizeof(double);
  upper_hessian_lagrangian = reinterpret_cast<double *>(mem_ptr + cum_size);
  cum_size += upper_hessian_lagrangian_nnz * sizeof(double);
  c = reinterpret_cast<double *>(mem_ptr + cum_size);
  cum_size += y_dim * sizeof(double);
  jacobian_c = reinterpret_cast<double *>(mem_ptr + cum_size);
  cum_size += jacobian_c_nnz * sizeof(double);
  g = reinterpret_cast<double *>(mem_ptr + cum_size);
  cum_size += s_dim * sizeof(double);
  jacobian_g = reinterpret_cast<double *>(mem_ptr + cum_size);
  cum_size += jacobian_g_nnz * sizeof(double);
  return cum_size;
}

} // namespace sip_examples
