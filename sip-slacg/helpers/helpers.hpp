#pragma once

namespace sip_examples {

struct ModelCallbackOutput {
  // NOTE: all sparse matrices should be represented in CSC format.

  // The objective and its first derivative.
  double f;
  double *gradient_f;

  // The Hessian of the Lagrangian.
  // NOTE:
  // 1. Only the upper triangle should be filled in upper_hessian_lagrangian.
  // 2. upper_hessian_lagrangian should be a positive definite approximation.
  // 3. An positive definite approximation of the Hessian of f is often used.
  double *upper_hessian_lagrangian;

  // The equality constraints and their first derivative.
  double *c;
  double *jacobian_c;

  // The inequality constraints and their first derivative.
  double *g;
  double *jacobian_g;

  // To dynamically allocate the required memory.
  void reserve(int x_dim, int s_dim, int y_dim,
               int upper_hessian_lagrangian_nnz, int jacobian_c_nnz,
               int jacobian_g_nnz);
  void free();

  // For using pre-allocated (possibly statically allocated) memory.
  auto mem_assign(int x_dim, int s_dim, int y_dim,
                  int upper_hessian_lagrangian_nnz, int jacobian_c_nnz,
                  int jacobian_g_nnz, unsigned char *mem_ptr) -> int;

  // For knowing how much memory to pre-allocate.
  static constexpr auto num_bytes(int x_dim, int s_dim, int y_dim,
                                  int upper_hessian_lagrangian_nnz,
                                  int jacobian_c_nnz, int jacobian_g_nnz)
      -> int {
    return (x_dim + s_dim + y_dim + upper_hessian_lagrangian_nnz +
            jacobian_c_nnz + jacobian_g_nnz) *
           sizeof(double);
  }
};

} // namespace sip_examples
