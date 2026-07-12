#pragma once

#include "sip/types.hpp"

#include <vector>

namespace sip_examples::problem_definitions::casadi_problems {

struct FlatProblemSpec {
  const char *name;
  int x_dim;
  int s_dim;
  int y_dim;
  int upper_hessian_lagrangian_nnz;
  int raw_upper_hessian_lagrangian_nnz;
  int jacobian_c_nnz;
  int jacobian_g_nnz;
  int jacobian_c_transpose_nnz;
  int jacobian_g_transpose_nnz;
  int kkt_dim;
  int kkt_nnz;
  int kkt_L_nnz;
  const int *upper_hessian_lagrangian_ind;
  const int *upper_hessian_lagrangian_indptr;
  const int *jacobian_c_ind;
  const int *jacobian_c_indptr;
  const int *jacobian_g_ind;
  const int *jacobian_g_indptr;
  const int *jacobian_c_transpose_ind;
  const int *jacobian_c_transpose_indptr;
  const int *jacobian_g_transpose_ind;
  const int *jacobian_g_transpose_indptr;
  const int *kkt_pinv;
  const double *initial_x;
};

struct CasadiWork {
  std::vector<long long int> iw;
  std::vector<double> w;
  std::vector<double> scratch;

  void resize(int iw_size, int w_size, int scratch_size);
};

auto default_casadi_problem_settings(int max_iterations) -> sip::Settings;
void enable_all_casadi_problem_logs(sip::Settings &settings);
void initialize_slacks_and_duals(const double *g, int s_dim, double initial_mu,
                                 double *s, double *z);

} // namespace sip_examples::problem_definitions::casadi_problems
