#include "problem_definitions/cutest_problems/cutest_problem.hpp"

#include <Eigen/SparseCore>

#include <piqp/piqp.hpp>

#include <algorithm>
#include <cstdlib>
#include <exception>
#include <iostream>
#include <stdexcept>
#include <string_view>
#include <vector>

namespace sip_examples::problem_definitions::cutest_problems {
namespace {

auto symmetric_hessian(const sip_qdldl::SparseMatrix &source)
    -> Eigen::SparseMatrix<double> {
  std::vector<Eigen::Triplet<double>> triplets;
  const int nnz = source.indptr[source.cols];
  triplets.reserve(2 * nnz);
  for (int col = 0; col < source.cols; ++col) {
    for (int index = source.indptr[col]; index < source.indptr[col + 1];
         ++index) {
      const int row = source.ind[index];
      const double value = source.data[index];
      triplets.emplace_back(row, col, value);
      if (row != col) {
        triplets.emplace_back(col, row, value);
      }
    }
  }
  Eigen::SparseMatrix<double> result(source.rows, source.cols);
  result.setFromTriplets(triplets.begin(), triplets.end());
  result.makeCompressed();
  return result;
}

auto transpose_jacobian(const sip_qdldl::SparseMatrix &source)
    -> Eigen::SparseMatrix<double> {
  std::vector<Eigen::Triplet<double>> triplets;
  triplets.reserve(source.indptr[source.cols]);
  for (int constraint = 0; constraint < source.cols; ++constraint) {
    for (int index = source.indptr[constraint];
         index < source.indptr[constraint + 1]; ++index) {
      triplets.emplace_back(constraint, source.ind[index], source.data[index]);
    }
  }
  Eigen::SparseMatrix<double> result(source.cols, source.rows);
  result.setFromTriplets(triplets.begin(), triplets.end());
  result.makeCompressed();
  return result;
}

auto run(const char *runtime_path, const char *problem_library_path,
         const char *outsdif_path) -> piqp::Status {
  CutestProblem problem(runtime_path, problem_library_path, outsdif_path,
                        CutestProblem::SymbolicData::DerivativesOnly);
  const int x_dim = problem.x_dim();
  const int y_dim = problem.equality_dim();
  const int s_dim = problem.inequality_dim();

  std::vector<double> y(y_dim, 0.0);
  std::vector<double> z(s_dim, 0.0);
  problem.evaluate(problem.initial_x().data(), y.data(), z.data());
  const auto &output = problem.model_output();

  const Eigen::SparseMatrix<double> P =
      symmetric_hessian(output.upper_hessian_lagrangian);
  const Eigen::SparseMatrix<double> A = transpose_jacobian(output.jacobian_c);
  const Eigen::SparseMatrix<double> G = transpose_jacobian(output.jacobian_g);
  const Eigen::Map<const Eigen::VectorXd> x0(problem.initial_x().data(), x_dim);
  const Eigen::Map<const Eigen::VectorXd> gradient(output.gradient_f, x_dim);
  const Eigen::Map<const Eigen::VectorXd> equality(output.c, y_dim);
  const Eigen::Map<const Eigen::VectorXd> inequality(output.g, s_dim);

  const Eigen::VectorXd c = gradient - P * x0;
  const Eigen::VectorXd b = A * x0 - equality;
  const Eigen::VectorXd h_u = G * x0 - inequality;

  piqp::SparseSolver<double> solver;
  solver.settings().kkt_solver = piqp::KKTSolver::sparse_ldlt;
  solver.settings().eps_abs = 1e-6;
  solver.settings().eps_rel = 0.0;
  solver.settings().check_duality_gap = false;
  solver.settings().max_iter = 1000;
  solver.settings().compute_timings = true;
  const std::string_view ablation = [] {
    const char *value = std::getenv("PIQP_ABLATION");
    return value == nullptr ? std::string_view("default")
                            : std::string_view(value);
  }();
  if (ablation == "no_preconditioner") {
    solver.settings().preconditioner_iter = 0;
  } else if (ablation == "no_predictor_corrector") {
    solver.settings().use_predictor_corrector = false;
  } else if (ablation == "no_iterative_refinement") {
    solver.settings().iterative_refinement_max_iter = 0;
  } else if (ablation == "scale_cost") {
    solver.settings().preconditioner_scale_cost = true;
  } else if (ablation == "low_proximal") {
    solver.settings().rho_init = 1e-12;
    solver.settings().delta_init = 1e-12;
  } else if (ablation != "default") {
    throw std::invalid_argument("unknown PIQP_ABLATION mode");
  }
  solver.setup(P, c, A, b, G, piqp::nullopt, h_u, piqp::nullopt, piqp::nullopt);
  const piqp::Status status = solver.solve();
  const auto &result = solver.result();
  const auto complementarity = result.s_u.array() * result.z_u.array();
  const double max_complementarity =
      complementarity.size() == 0 ? 0.0 : complementarity.maxCoeff();
  const auto &info = solver.result().info;
  std::cout << "ablation=" << ablation
            << " status=" << piqp::status_to_string(status)
            << " iterations=" << info.iter << " ls_iterations=0"
            << " primal=" << info.primal_res << " dual=" << info.dual_res
            << " gap=" << info.duality_gap
            << " mean_complementarity=" << info.mu
            << " max_complementarity=" << max_complementarity
            << " setup_time=" << info.setup_time
            << " solve_time=" << info.solve_time
            << " kkt_factor_time=" << info.kkt_factor_time
            << " kkt_solve_time=" << info.kkt_solve_time << '\n';
  return status;
}

} // namespace
} // namespace sip_examples::problem_definitions::cutest_problems

int main(int argc, char **argv) {
  if (argc != 5) {
    std::cerr << "usage: piqp_runner CUTEST_RUNTIME PROBLEM_LIBRARY OUTSDIF "
                 "USE_QP_SETTINGS\n";
    return 2;
  }
  const std::string_view use_qp_settings_arg(argv[4]);
  if (use_qp_settings_arg != "1") {
    std::cerr << "PIQP runner only supports QP corpus problems\n";
    return 2;
  }
  try {
    const piqp::Status status =
        sip_examples::problem_definitions::cutest_problems::run(
            argv[1], argv[2], argv[3]);
    return status == piqp::Status::PIQP_SOLVED ? 0 : 1;
  } catch (const std::exception &error) {
    std::cerr << error.what() << '\n';
    return 2;
  }
}
