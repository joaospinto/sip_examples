#include "problem_definitions/cutest_problems/cutest_problem.hpp"

#include <Eigen/OrderingMethods>
#include <Eigen/SparseCore>

#include <cutest_c.h>
#include <qdldl.h>

#include <algorithm>
#include <cmath>
#include <dlfcn.h>
#include <stdexcept>
#include <string_view>
#include <utility>

namespace sip_examples::problem_definitions::cutest_problems {
namespace {

constexpr double kCutestInfinity = 1e20;

template <typename Function>
auto load_symbol(void *handle, const char *name) -> Function {
  dlerror();
  void *symbol = dlsym(handle, name);
  if (const char *error = dlerror(); error != nullptr) {
    throw std::runtime_error("failed to load CUTEst symbol " +
                             std::string(name) + ": " + error);
  }
  return reinterpret_cast<Function>(symbol);
}

void check_status(int status, const char *operation) {
  if (status != 0) {
    throw std::runtime_error(std::string(operation) +
                             " failed with CUTEst status " +
                             std::to_string(status));
  }
}

auto coordinate_key(int row, int col) -> std::uint64_t {
  return (static_cast<std::uint64_t>(static_cast<std::uint32_t>(col)) << 32) |
         static_cast<std::uint32_t>(row);
}

auto is_finite_cutest_bound(double value) -> bool {
  return std::isfinite(value) && std::abs(value) < kCutestInfinity;
}

void sort_and_unique(std::vector<int> &indices) {
  std::sort(indices.begin(), indices.end());
  indices.erase(std::unique(indices.begin(), indices.end()), indices.end());
}

void assign_pattern(const std::vector<std::vector<int>> &columns, int rows,
                    bool is_transposed, sip_qdldl::SparseMatrix &matrix) {
  matrix.rows = rows;
  matrix.cols = static_cast<int>(columns.size());
  matrix.is_transposed = is_transposed;
  int offset = 0;
  matrix.indptr[0] = 0;
  for (int col = 0; col < matrix.cols; ++col) {
    for (int row : columns[col]) {
      matrix.ind[offset++] = row;
    }
    matrix.indptr[col + 1] = offset;
  }
}

auto pattern_nnz(const std::vector<std::vector<int>> &columns) -> int {
  int result = 0;
  for (const auto &column : columns) {
    result += static_cast<int>(column.size());
  }
  return result;
}

auto find_entry(const sip_qdldl::SparseMatrix &matrix, int col, int row)
    -> int {
  const int begin = matrix.indptr[col];
  const int end = matrix.indptr[col + 1];
  const int *position =
      std::lower_bound(matrix.ind + begin, matrix.ind + end, row);
  if (position == matrix.ind + end || *position != row) {
    throw std::runtime_error("CUTEst derivative entry is absent from pattern");
  }
  return static_cast<int>(position - matrix.ind);
}

} // namespace

struct CutestProblem::Api {
  using LoadRoutines = void (*)(const char *);
  using UnloadRoutines = void (*)();
  using FortranOpen = void (*)(const int *, const char *, int *);
  using FortranClose = void (*)(const int *, int *);
  using Cdimen = void (*)(int *, const int *, int *, int *);
  using Usetup = void (*)(int *, const int *, const int *, const int *, int *,
                          double *, double *, double *);
  using Csetup = void (*)(int *, const int *, const int *, const int *, int *,
                          int *, double *, double *, double *, double *,
                          double *, double *, bool *, bool *, const int *,
                          const int *, const int *);
  using Classification = void (*)(int *, const int *, char *);
  using Uofg = void (*)(int *, const int *, const double *, double *, double *,
                        const bool *);
  using Cofg = void (*)(int *, const int *, const double *, double *, double *,
                        const bool *);
  using Ugrsh = void (*)(int *, const int *, const double *, double *, int *,
                         const int *, double *, int *, int *);
  using Cdimscj = void (*)(int *, int *);
  using Csjp = void (*)(int *, int *, const int *, int *, int *);
  using Ccfsg = void (*)(int *, const int *, const int *, const double *,
                         double *, int *, const int *, double *, int *, int *,
                         const bool *);
  using Csgrsh = void (*)(int *, const int *, const int *, const double *,
                          const double *, const bool *, int *, const int *,
                          double *, int *, int *, int *, const int *, double *,
                          int *, int *);
  using Udimsh = void (*)(int *, int *);
  using Cdimsh = void (*)(int *, int *);
  using Ushp = void (*)(int *, const int *, int *, const int *, int *, int *);
  using Cshp = void (*)(int *, const int *, int *, const int *, int *, int *);
  using Terminate = void (*)(int *);

  explicit Api(void *handle)
      : load_routines(
            load_symbol<LoadRoutines>(handle, "cutest_load_routines_")),
        unload_routines(
            load_symbol<UnloadRoutines>(handle, "cutest_unload_routines_")),
        fortran_open(load_symbol<FortranOpen>(handle, "fortran_open_")),
        fortran_close(load_symbol<FortranClose>(handle, "fortran_close_")),
        cdimen(load_symbol<Cdimen>(handle, "cutest_cdimen_")),
        usetup(load_symbol<Usetup>(handle, "cutest_usetup_")),
        csetup(load_symbol<Csetup>(handle, "cutest_cint_csetup_")),
        classification(
            load_symbol<Classification>(handle, "cutest_cint_classification_")),
        uofg(load_symbol<Uofg>(handle, "cutest_cint_uofg_")),
        cofg(load_symbol<Cofg>(handle, "cutest_cint_cofg_")),
        ugrsh(load_symbol<Ugrsh>(handle, "cutest_ugrsh_c_")),
        cdimscj(load_symbol<Cdimscj>(handle, "cutest_cdimscj_")),
        csjp(load_symbol<Csjp>(handle, "cutest_csjp_c_")),
        ccfsg(load_symbol<Ccfsg>(handle, "cutest_ccfsg_c_")),
        csgrsh(load_symbol<Csgrsh>(handle, "cutest_csgrsh_c_")),
        udimsh(load_symbol<Udimsh>(handle, "cutest_udimsh_")),
        cdimsh(load_symbol<Cdimsh>(handle, "cutest_cdimsh_")),
        ushp(load_symbol<Ushp>(handle, "cutest_ushp_c_")),
        cshp(load_symbol<Cshp>(handle, "cutest_cshp_c_")),
        uterminate(load_symbol<Terminate>(handle, "cutest_uterminate_")),
        cterminate(load_symbol<Terminate>(handle, "cutest_cterminate_")) {}

  LoadRoutines load_routines;
  UnloadRoutines unload_routines;
  FortranOpen fortran_open;
  FortranClose fortran_close;
  Cdimen cdimen;
  Usetup usetup;
  Csetup csetup;
  Classification classification;
  Uofg uofg;
  Cofg cofg;
  Ugrsh ugrsh;
  Cdimscj cdimscj;
  Csjp csjp;
  Ccfsg ccfsg;
  Csgrsh csgrsh;
  Udimsh udimsh;
  Cdimsh cdimsh;
  Ushp ushp;
  Cshp cshp;
  Terminate uterminate;
  Terminate cterminate;
};

CutestProblem::CutestProblem(const std::string &runtime_path,
                             const std::string &problem_library_path,
                             const std::string &outsdif_path) {
  open(runtime_path, problem_library_path, outsdif_path);
  setup();
  build_terms();
  build_sparse_patterns();
}

CutestProblem::~CutestProblem() {
  if (api_ != nullptr) {
    int status = 0;
    if (setup_complete_) {
      if (is_constrained()) {
        api_->cterminate(&status);
      } else {
        api_->uterminate(&status);
      }
    }
    if (input_open_) {
      status = 0;
      api_->fortran_close(&input_unit_, &status);
    }
    if (routines_loaded_) {
      api_->unload_routines();
    }
    delete api_;
  }
  if (runtime_handle_ != nullptr) {
    dlclose(runtime_handle_);
  }
  model_output_.free();
}

void CutestProblem::open(const std::string &runtime_path,
                         const std::string &problem_library_path,
                         const std::string &outsdif_path) {
  runtime_handle_ = dlopen(runtime_path.c_str(), RTLD_NOW | RTLD_GLOBAL);
  if (runtime_handle_ == nullptr) {
    throw std::runtime_error("failed to load CUTEst runtime: " +
                             std::string(dlerror()));
  }
  api_ = new Api(runtime_handle_);
  api_->load_routines(problem_library_path.c_str());
  routines_loaded_ = true;

  int status = 0;
  api_->fortran_open(&input_unit_, outsdif_path.c_str(), &status);
  check_status(status, "opening OUTSDIF");
  input_open_ = true;
}

void CutestProblem::setup() {
  int status = 0;
  char classification[31]{};
  api_->classification(&status, &input_unit_, classification);
  check_status(status, "CUTEST_classification");
  constexpr std::string_view quadratic_objectives = "NCLQ";
  constexpr std::string_view linear_constraints = "UXBNL";
  is_quadratic_program_ =
      quadratic_objectives.find(classification[0]) != std::string_view::npos &&
      linear_constraints.find(classification[1]) != std::string_view::npos;

  api_->cdimen(&status, &input_unit_, &n_, &m_);
  check_status(status, "CUTEST_cdimen");
  if (n_ <= 0 || m_ < 0) {
    throw std::runtime_error("CUTEst returned invalid problem dimensions");
  }

  initial_x_.resize(n_);
  variable_lower_.resize(n_);
  variable_upper_.resize(n_);

  constexpr int output_unit = 6;
  constexpr int io_buffer = 11;
  if (!is_constrained()) {
    api_->usetup(&status, &input_unit_, &output_unit, &io_buffer, &n_,
                 initial_x_.data(), variable_lower_.data(),
                 variable_upper_.data());
    check_status(status, "CUTEST_usetup");
  } else {
    original_multipliers_.resize(m_);
    constraint_lower_.resize(m_);
    constraint_upper_.resize(m_);
    original_constraints_.resize(m_);
    equality_flags_ = std::make_unique<bool[]>(m_);
    linear_flags_ = std::make_unique<bool[]>(m_);
    constexpr int equality_order = 0;
    constexpr int linear_order = 0;
    constexpr int nonlinear_variable_order = 0;
    api_->csetup(&status, &input_unit_, &output_unit, &io_buffer, &n_, &m_,
                 initial_x_.data(), variable_lower_.data(),
                 variable_upper_.data(), original_multipliers_.data(),
                 constraint_lower_.data(), constraint_upper_.data(),
                 equality_flags_.get(), linear_flags_.get(), &equality_order,
                 &linear_order, &nonlinear_variable_order);
    check_status(status, "CUTEST_csetup");
  }
  setup_complete_ = true;
}

void CutestProblem::build_terms() {
  for (int i = 0; i < n_; ++i) {
    append_bound_terms(Source::Variable, i, variable_lower_[i],
                       variable_upper_[i], false);
  }

  for (int i = 0; i < m_; ++i) {
    append_bound_terms(Source::Constraint, i, constraint_lower_[i],
                       constraint_upper_[i], equality_flags_[i]);
  }

  constant_singleton_inequalities_.reserve(inequality_terms_.size());
  for (const Term &term : inequality_terms_) {
    constant_singleton_inequalities_.push_back(
        term.source == Source::Variable ? 1 : 0);
  }
}

void CutestProblem::append_bound_terms(Source source, int index, double lower,
                                       double upper, bool equality) {
  const bool has_lower = is_finite_cutest_bound(lower);
  const bool has_upper = is_finite_cutest_bound(upper);
  if (equality || (has_lower && has_upper && lower == upper)) {
    if (!has_lower) {
      throw std::runtime_error("CUTEst equality has no finite target");
    }
    equality_terms_.push_back(
        {.source = source, .index = index, .sign = 1.0, .offset = -lower});
    return;
  }
  if (has_lower) {
    inequality_terms_.push_back(
        {.source = source, .index = index, .sign = -1.0, .offset = lower});
  }
  if (has_upper) {
    inequality_terms_.push_back(
        {.source = source, .index = index, .sign = 1.0, .offset = -upper});
  }
}

void CutestProblem::build_sparse_patterns() {
  original_jacobian_variables_.resize(m_);
  if (is_constrained()) {
    int status = 0;
    api_->cdimscj(&status, &original_jacobian_capacity_);
    check_status(status, "CUTEST_cdimscj");
    const int derivative_capacity = original_jacobian_capacity_ + n_;
    original_jacobian_values_.resize(derivative_capacity);
    original_jacobian_variables_buffer_.resize(derivative_capacity);
    original_jacobian_constraints_buffer_.resize(derivative_capacity);
    int nnz = original_jacobian_capacity_;
    api_->csjp(&status, &nnz, &original_jacobian_capacity_,
               original_jacobian_variables_buffer_.data(),
               original_jacobian_constraints_buffer_.data());
    check_status(status, "CUTEST_csjp");
    for (int i = 0; i < nnz; ++i) {
      const int variable = original_jacobian_variables_buffer_[i];
      const int constraint = original_jacobian_constraints_buffer_[i];
      if (variable < 0 || variable >= n_ || constraint < 0 ||
          constraint >= m_) {
        throw std::runtime_error("CUTEst returned invalid Jacobian indices");
      }
      original_jacobian_variables_[constraint].push_back(variable);
    }
    for (auto &variables : original_jacobian_variables_) {
      sort_and_unique(variables);
    }
  }

  int status = 0;
  if (is_constrained()) {
    api_->cdimsh(&status, &original_hessian_capacity_);
  } else {
    api_->udimsh(&status, &original_hessian_capacity_);
  }
  check_status(status, "CUTEst Hessian dimensioning");
  original_hessian_values_.resize(original_hessian_capacity_);
  original_hessian_rows_.resize(original_hessian_capacity_);
  original_hessian_cols_.resize(original_hessian_capacity_);
  int hessian_nnz = original_hessian_capacity_;
  if (is_constrained()) {
    api_->cshp(&status, &n_, &hessian_nnz, &original_hessian_capacity_,
               original_hessian_rows_.data(), original_hessian_cols_.data());
  } else {
    api_->ushp(&status, &n_, &hessian_nnz, &original_hessian_capacity_,
               original_hessian_rows_.data(), original_hessian_cols_.data());
  }
  check_status(status, "CUTEst Hessian pattern");

  std::vector<std::vector<int>> hessian_columns(n_);
  for (int i = 0; i < hessian_nnz; ++i) {
    int row = original_hessian_rows_[i];
    int col = original_hessian_cols_[i];
    if (row < 0 || row >= n_ || col < 0 || col >= n_) {
      throw std::runtime_error("CUTEst returned invalid Hessian indices");
    }
    if (row > col) {
      std::swap(row, col);
    }
    hessian_columns[col].push_back(row);
  }
  for (int col = 0; col < n_; ++col) {
    hessian_columns[col].push_back(col);
    sort_and_unique(hessian_columns[col]);
  }

  auto term_columns = [this](const std::vector<Term> &terms) {
    std::vector<std::vector<int>> columns;
    columns.reserve(terms.size());
    for (const Term &term : terms) {
      if (term.source == Source::Variable) {
        columns.push_back({term.index});
      } else {
        columns.push_back(original_jacobian_variables_[term.index]);
      }
    }
    return columns;
  };
  auto equality_columns = term_columns(equality_terms_);
  auto inequality_columns = term_columns(inequality_terms_);

  const int hessian_pattern_nnz = pattern_nnz(hessian_columns);
  const int equality_pattern_nnz = pattern_nnz(equality_columns);
  const int inequality_pattern_nnz = pattern_nnz(inequality_columns);
  model_output_.reserve(n_, static_cast<int>(inequality_terms_.size()),
                        static_cast<int>(equality_terms_.size()),
                        hessian_pattern_nnz, equality_pattern_nnz,
                        inequality_pattern_nnz, true, true);
  assign_pattern(hessian_columns, n_, false,
                 model_output_.upper_hessian_lagrangian);
  assign_pattern(equality_columns, n_, true, model_output_.jacobian_c);
  assign_pattern(inequality_columns, n_, true, model_output_.jacobian_g);

  for (int col = 0; col < n_; ++col) {
    for (int index = model_output_.upper_hessian_lagrangian.indptr[col];
         index < model_output_.upper_hessian_lagrangian.indptr[col + 1];
         ++index) {
      const int row = model_output_.upper_hessian_lagrangian.ind[index];
      hessian_scatter_.emplace(coordinate_key(row, col), index);
    }
  }

  auto add_jacobian_scatter = [this](const std::vector<Term> &terms,
                                     JacobianKind kind,
                                     const sip_qdldl::SparseMatrix &matrix) {
    for (int col = 0; col < static_cast<int>(terms.size()); ++col) {
      const Term &term = terms[col];
      if (term.source != Source::Constraint) {
        matrix.data[find_entry(matrix, col, term.index)] = term.sign;
        continue;
      }
      for (int variable : original_jacobian_variables_[term.index]) {
        jacobian_scatter_[coordinate_key(variable, term.index)].push_back(
            {.kind = kind,
             .index = find_entry(matrix, col, variable),
             .sign = term.sign});
      }
    }
  };
  add_jacobian_scatter(equality_terms_, JacobianKind::Equality,
                       model_output_.jacobian_c);
  add_jacobian_scatter(inequality_terms_, JacobianKind::Inequality,
                       model_output_.jacobian_g);

  const int equality_dim = static_cast<int>(equality_terms_.size());
  const int inequality_dim = static_cast<int>(inequality_terms_.size());
  const int kkt_dim = n_ + equality_dim + inequality_dim;
  kkt_nnz_ = hessian_pattern_nnz + equality_pattern_nnz + equality_dim +
             inequality_pattern_nnz + inequality_dim;
  std::vector<int> kkt_indptr(kkt_dim + 1);
  std::vector<int> kkt_indices;
  kkt_indices.reserve(kkt_nnz_);
  for (int col = 0; col < n_; ++col) {
    kkt_indptr[col] = static_cast<int>(kkt_indices.size());
    for (int index = model_output_.upper_hessian_lagrangian.indptr[col];
         index < model_output_.upper_hessian_lagrangian.indptr[col + 1];
         ++index) {
      kkt_indices.push_back(model_output_.upper_hessian_lagrangian.ind[index]);
    }
  }
  for (int col = 0; col < equality_dim; ++col) {
    const int kkt_col = n_ + col;
    kkt_indptr[kkt_col] = static_cast<int>(kkt_indices.size());
    for (int index = model_output_.jacobian_c.indptr[col];
         index < model_output_.jacobian_c.indptr[col + 1]; ++index) {
      kkt_indices.push_back(model_output_.jacobian_c.ind[index]);
    }
    kkt_indices.push_back(kkt_col);
  }
  for (int col = 0; col < inequality_dim; ++col) {
    const int kkt_col = n_ + equality_dim + col;
    kkt_indptr[kkt_col] = static_cast<int>(kkt_indices.size());
    for (int index = model_output_.jacobian_g.indptr[col];
         index < model_output_.jacobian_g.indptr[col + 1]; ++index) {
      kkt_indices.push_back(model_output_.jacobian_g.ind[index]);
    }
    kkt_indices.push_back(kkt_col);
  }
  kkt_indptr[kkt_dim] = static_cast<int>(kkt_indices.size());
  if (static_cast<int>(kkt_indices.size()) != kkt_nnz_) {
    throw std::runtime_error("internal CUTEst KKT sparsity mismatch");
  }

  Eigen::SparseMatrix<double> kkt_pattern(kkt_dim, kkt_dim);
  std::vector<Eigen::Triplet<double>> kkt_triplets;
  kkt_triplets.reserve(kkt_nnz_);
  for (int col = 0; col < kkt_dim; ++col) {
    for (int index = kkt_indptr[col]; index < kkt_indptr[col + 1]; ++index) {
      kkt_triplets.emplace_back(kkt_indices[index], col, 1.0);
    }
  }
  kkt_pattern.setFromTriplets(kkt_triplets.begin(), kkt_triplets.end());

  Eigen::AMDOrdering<int> amd;
  Eigen::PermutationMatrix<Eigen::Dynamic, Eigen::Dynamic, int> permutation;
  amd(kkt_pattern.selfadjointView<Eigen::Upper>(), permutation);
  // Eigen stores new-to-old indices; SIP-QDLDL consumes old-to-new indices.
  kkt_pinv_.resize(kkt_dim);
  for (int new_index = 0; new_index < kkt_dim; ++new_index) {
    kkt_pinv_[permutation.indices()[new_index]] = new_index;
  }

  std::vector<std::vector<int>> permuted_columns(kkt_dim);
  for (int col = 0; col < kkt_dim; ++col) {
    for (int index = kkt_indptr[col]; index < kkt_indptr[col + 1]; ++index) {
      const int permuted_row = kkt_pinv_[kkt_indices[index]];
      const int permuted_col = kkt_pinv_[col];
      permuted_columns[std::max(permuted_row, permuted_col)].push_back(
          std::min(permuted_row, permuted_col));
    }
  }

  std::vector<int> permuted_indptr(kkt_dim + 1);
  std::vector<int> permuted_indices;
  permuted_indices.reserve(kkt_nnz_);
  for (int col = 0; col < kkt_dim; ++col) {
    permuted_indptr[col] = static_cast<int>(permuted_indices.size());
    auto &rows = permuted_columns[col];
    std::sort(rows.begin(), rows.end());
    permuted_indices.insert(permuted_indices.end(), rows.begin(), rows.end());
  }
  permuted_indptr[kkt_dim] = static_cast<int>(permuted_indices.size());

  std::vector<int> elimination_tree(kkt_dim);
  std::vector<int> column_counts(kkt_dim);
  std::vector<int> workspace(kkt_dim);
  kkt_l_nnz_ = QDLDL_etree(kkt_dim, permuted_indptr.data(),
                           permuted_indices.data(), workspace.data(),
                           column_counts.data(), elimination_tree.data());
  if (kkt_l_nnz_ == -2) {
    throw std::runtime_error(
        "AMD-ordered KKT factor exceeds QDLDL's index capacity");
  }
  if (kkt_l_nnz_ < 0) {
    throw std::runtime_error("QDLDL rejected the CUTEst KKT sparsity pattern");
  }
}

void CutestProblem::evaluate_objective(const double *x,
                                       const bool calculate_gradient) {
  int status = 0;
  if (is_constrained()) {
    api_->cofg(&status, &n_, x, &model_output_.f, model_output_.gradient_f,
               &calculate_gradient);
  } else {
    api_->uofg(&status, &n_, x, &model_output_.f, model_output_.gradient_f,
               &calculate_gradient);
  }
  check_status(status, "CUTEst objective evaluation");
}

void CutestProblem::evaluate_constraints(const double *x,
                                         const bool calculate_jacobian) {
  if (calculate_jacobian) {
    reset_jacobians();
  }
  if (is_constrained()) {
    int status = 0;
    int nnz = original_jacobian_capacity_;
    api_->ccfsg(&status, &n_, &m_, x, original_constraints_.data(), &nnz,
                &original_jacobian_capacity_, original_jacobian_values_.data(),
                original_jacobian_variables_buffer_.data(),
                original_jacobian_constraints_buffer_.data(),
                &calculate_jacobian);
    check_status(status, "CUTEst constraint evaluation");
    if (calculate_jacobian) {
      scatter_first_derivatives(nnz);
    }
  }

  for (int i = 0; i < equality_dim(); ++i) {
    model_output_.c[i] = term_value(equality_terms_[i], x);
  }
  for (int i = 0; i < inequality_dim(); ++i) {
    model_output_.g[i] = term_value(inequality_terms_[i], x);
  }
}

void CutestProblem::initialize_variable_jacobian(
    const std::vector<Term> &terms, sip_qdldl::SparseMatrix &jacobian) {
  for (int col = 0; col < static_cast<int>(terms.size()); ++col) {
    const Term &term = terms[col];
    if (term.source == Source::Variable) {
      jacobian.data[find_entry(jacobian, col, term.index)] = term.sign;
    }
  }
}

void CutestProblem::reset_jacobians() {
  std::fill_n(model_output_.jacobian_c.data,
              model_output_.jacobian_c.indptr[model_output_.jacobian_c.cols],
              0.0);
  std::fill_n(model_output_.jacobian_g.data,
              model_output_.jacobian_g.indptr[model_output_.jacobian_g.cols],
              0.0);
  initialize_variable_jacobian(equality_terms_, model_output_.jacobian_c);
  initialize_variable_jacobian(inequality_terms_, model_output_.jacobian_g);
}

void CutestProblem::scatter_first_derivatives(const int nnz) {
  for (int i = 0; i < nnz; ++i) {
    const int variable = original_jacobian_variables_buffer_[i];
    const int function = original_jacobian_constraints_buffer_[i];
    if (variable < 0 || variable >= n_ || function < -1 || function >= m_) {
      throw std::runtime_error("CUTEst returned invalid derivative indices");
    }
    if (function < 0) {
      model_output_.gradient_f[variable] = original_jacobian_values_[i];
      continue;
    }
    const auto scatter =
        jacobian_scatter_.find(coordinate_key(variable, function));
    if (scatter == jacobian_scatter_.end()) {
      continue;
    }
    for (const JacobianScatter &entry : scatter->second) {
      double *data = entry.kind == JacobianKind::Equality
                         ? model_output_.jacobian_c.data
                         : model_output_.jacobian_g.data;
      data[entry.index] = entry.sign * original_jacobian_values_[i];
    }
  }
}

void CutestProblem::prepare_original_multipliers(const double *y,
                                                 const double *z) {
  std::fill(original_multipliers_.begin(), original_multipliers_.end(), 0.0);
  add_constraint_multipliers(equality_terms_, y);
  add_constraint_multipliers(inequality_terms_, z);
}

void CutestProblem::scatter_hessian(const int nnz) {
  std::fill_n(model_output_.upper_hessian_lagrangian.data,
              model_output_.upper_hessian_lagrangian
                  .indptr[model_output_.upper_hessian_lagrangian.cols],
              0.0);
  for (int i = 0; i < nnz; ++i) {
    int row = original_hessian_rows_[i];
    int col = original_hessian_cols_[i];
    if (row > col) {
      std::swap(row, col);
    }
    const auto position = hessian_scatter_.find(coordinate_key(row, col));
    if (position == hessian_scatter_.end()) {
      throw std::runtime_error(
          "CUTEst Hessian value is absent from its declared pattern");
    }
    model_output_.upper_hessian_lagrangian.data[position->second] =
        original_hessian_values_[i];
  }
}

void CutestProblem::add_constraint_multipliers(const std::vector<Term> &terms,
                                               const double *multipliers) {
  for (int i = 0; i < static_cast<int>(terms.size()); ++i) {
    const Term &term = terms[i];
    if (term.source == Source::Constraint) {
      original_multipliers_[term.index] += term.sign * multipliers[i];
    }
  }
}

void CutestProblem::evaluate_values(const double *x) {
  evaluate_objective(x, false);
  evaluate_constraints(x, false);
}

void CutestProblem::evaluate_derivatives(const double *x, const double *y,
                                         const double *z) {
  int status = 0;
  int hessian_nnz = original_hessian_capacity_;
  reset_jacobians();
  if (is_constrained()) {
    prepare_original_multipliers(y, z);
    std::fill_n(model_output_.gradient_f, n_, 0.0);
    int derivative_nnz = static_cast<int>(original_jacobian_values_.size());
    const int derivative_capacity = derivative_nnz;
    constexpr bool gradient_of_lagrangian = false;
    api_->csgrsh(&status, &n_, &m_, x, original_multipliers_.data(),
                 &gradient_of_lagrangian, &derivative_nnz, &derivative_capacity,
                 original_jacobian_values_.data(),
                 original_jacobian_variables_buffer_.data(),
                 original_jacobian_constraints_buffer_.data(), &hessian_nnz,
                 &original_hessian_capacity_, original_hessian_values_.data(),
                 original_hessian_rows_.data(), original_hessian_cols_.data());
    check_status(status, "CUTEst derivative evaluation");
    scatter_first_derivatives(derivative_nnz);
  } else {
    api_->ugrsh(&status, &n_, x, model_output_.gradient_f, &hessian_nnz,
                &original_hessian_capacity_, original_hessian_values_.data(),
                original_hessian_rows_.data(), original_hessian_cols_.data());
    check_status(status, "CUTEst derivative evaluation");
  }
  scatter_hessian(hessian_nnz);
}

double CutestProblem::term_value(const Term &term, const double *x) const {
  const double source_value = term.source == Source::Variable
                                  ? x[term.index]
                                  : original_constraints_[term.index];
  return term.sign * source_value + term.offset;
}

int CutestProblem::x_dim() const { return n_; }

int CutestProblem::equality_dim() const {
  return static_cast<int>(equality_terms_.size());
}

int CutestProblem::inequality_dim() const {
  return static_cast<int>(inequality_terms_.size());
}

bool CutestProblem::is_quadratic_program() const {
  return is_quadratic_program_;
}

const std::vector<double> &CutestProblem::initial_x() const {
  return initial_x_;
}

sip_qdldl::ModelCallbackOutput &CutestProblem::model_output() {
  return model_output_;
}

int CutestProblem::kkt_nnz() const { return kkt_nnz_; }

int CutestProblem::kkt_l_nnz() const { return kkt_l_nnz_; }

const int *CutestProblem::kkt_pinv() const { return kkt_pinv_.data(); }

const std::uint8_t *CutestProblem::constant_singleton_inequalities() const {
  return constant_singleton_inequalities_.empty()
             ? nullptr
             : constant_singleton_inequalities_.data();
}

bool CutestProblem::is_constrained() const { return m_ > 0; }

} // namespace sip_examples::problem_definitions::cutest_problems
