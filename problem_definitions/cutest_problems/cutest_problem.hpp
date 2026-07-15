#pragma once

#include "sip_qdldl/sip_qdldl.hpp"

#include <cstdint>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

namespace sip_examples::problem_definitions::cutest_problems {

class CutestProblem {
public:
  CutestProblem(const std::string &runtime_path,
                const std::string &problem_library_path,
                const std::string &outsdif_path);
  ~CutestProblem();

  CutestProblem(const CutestProblem &) = delete;
  CutestProblem &operator=(const CutestProblem &) = delete;

  int x_dim() const;
  int equality_dim() const;
  int inequality_dim() const;
  const std::vector<double> &initial_x() const;

  sip_qdldl::ModelCallbackOutput &model_output();
  void evaluate_values(const double *x);
  void evaluate_derivatives(const double *x, const double *y, const double *z);
  int kkt_nnz() const;
  int kkt_l_nnz() const;
  const int *kkt_pinv() const;
  const std::uint8_t *constant_singleton_inequalities() const;

private:
  struct Api;

  enum class Source { Variable, Constraint };

  struct Term {
    Source source;
    int index;
    double sign;
    double offset;
  };

  enum class JacobianKind { Equality, Inequality };

  struct JacobianScatter {
    JacobianKind kind;
    int index;
    double sign;
  };

  void open(const std::string &runtime_path,
            const std::string &problem_library_path,
            const std::string &outsdif_path);
  void setup();
  void build_terms();
  void build_sparse_patterns();
  void append_bound_terms(Source source, int index, double lower, double upper,
                          bool equality);
  void initialize_variable_jacobian(const std::vector<Term> &terms,
                                    sip_qdldl::SparseMatrix &jacobian);
  void reset_jacobians();
  void scatter_first_derivatives(int nnz);
  void scatter_hessian(int nnz);
  void prepare_original_multipliers(const double *y, const double *z);
  void add_constraint_multipliers(const std::vector<Term> &terms,
                                  const double *multipliers);
  void evaluate_objective(const double *x, bool calculate_gradient);
  void evaluate_constraints(const double *x, bool calculate_jacobian);
  double term_value(const Term &term, const double *x) const;
  bool is_constrained() const;

  void *runtime_handle_{nullptr};
  Api *api_{nullptr};
  int input_unit_{42};
  bool routines_loaded_{false};
  bool input_open_{false};
  bool setup_complete_{false};
  int n_{0};
  int m_{0};
  std::vector<double> initial_x_;
  std::vector<double> variable_lower_;
  std::vector<double> variable_upper_;
  std::vector<double> constraint_lower_;
  std::vector<double> constraint_upper_;
  std::vector<double> original_multipliers_;
  std::vector<double> original_constraints_;
  std::unique_ptr<bool[]> equality_flags_;
  std::unique_ptr<bool[]> linear_flags_;

  std::vector<Term> equality_terms_;
  std::vector<Term> inequality_terms_;
  std::vector<std::uint8_t> constant_singleton_inequalities_;
  std::vector<std::vector<int>> original_jacobian_variables_;

  int original_jacobian_capacity_{0};
  std::vector<double> original_jacobian_values_;
  std::vector<int> original_jacobian_variables_buffer_;
  std::vector<int> original_jacobian_constraints_buffer_;
  std::unordered_map<std::uint64_t, std::vector<JacobianScatter>>
      jacobian_scatter_;

  int original_hessian_capacity_{0};
  std::vector<double> original_hessian_values_;
  std::vector<int> original_hessian_rows_;
  std::vector<int> original_hessian_cols_;
  std::unordered_map<std::uint64_t, int> hessian_scatter_;

  sip_qdldl::ModelCallbackOutput model_output_;
  int kkt_nnz_{0};
  int kkt_l_nnz_{0};
  std::vector<int> kkt_pinv_;
};

} // namespace sip_examples::problem_definitions::cutest_problems
