#include GENERATED_HEADER

#include "problem_definitions/casadi_problems/common/ocp_runner.hpp"

#include <cmath>
#include <cstdlib>
#include <limits>
#include <vector>

#include <gtest/gtest.h>

namespace sip_examples::problem_definitions::casadi_problems {

namespace {

struct CheckedValueOnlyProblem : generated_problem::Problem {
  static void eval_ocp(const ::sip::optimal_control::ModelCallbackInput &mci,
                       ::sip::optimal_control::ModelCallbackOutput &mco,
                       CasadiWork &work) {
    if (mci.need_derivatives) {
      Problem::eval_ocp(mci, mco, work);
      return;
    }
    const auto &spec = ocp_spec();
    const auto &dims = spec.dimensions;
    const auto visit_derivatives = [&](const auto &visit) {
      for (int node = 0; node <= spec.num_edges; ++node) {
        const int n = dims.get_state_dim(node);
        visit(mco.nodes[node].df_dx, n);
        visit(mco.nodes[node].d2L_dx2, n * n);
      }
      for (int edge = 0; edge < spec.num_edges; ++edge) {
        const int m = dims.get_control_dim(edge);
        visit(mco.edges[edge].df_du, m);
        visit(mco.edges[edge].d2L_du2, m * m);
      }
    };
    constexpr double sentinel = 123456789.0;
    visit_derivatives(
        [sentinel](double *data, int size) { std::fill_n(data, size, sentinel); });
    Problem::eval_ocp(mci, mco, work);
    visit_derivatives([sentinel](const double *data, int size) {
      for (int i = 0; i < size; ++i) {
        EXPECT_EQ(data[i], sentinel);
      }
    });
    const auto values = [&]() {
      std::vector<double> out;
      const auto append = [&](const double *data, int size) {
        for (int i = 0; i < size; ++i) {
          out.push_back(data[i]);
        }
      };
      for (int node = 0; node <= spec.num_edges; ++node) {
        out.push_back(mco.nodes[node].f);
        append(mco.nodes[node].c, dims.get_node_c_dim(node));
        append(mco.nodes[node].g, dims.get_node_g_dim(node));
      }
      for (int edge = 0; edge < spec.num_edges; ++edge) {
        out.push_back(mco.edges[edge].f);
        append(mco.edges[edge].dyn_res,
               dims.get_state_dim(spec.edge_children[edge]));
        append(mco.edges[edge].c, dims.get_edge_c_dim(edge));
        append(mco.edges[edge].g, dims.get_edge_g_dim(edge));
      }
      return out;
    };
    const auto value_only = values();
    auto full_input = mci;
    full_input.need_derivatives = true;
    Problem::eval_ocp(full_input, mco, work);
    const auto full_values = values();
    ASSERT_EQ(value_only.size(), full_values.size());
    for (size_t i = 0; i < value_only.size(); ++i) {
      if (std::isnan(full_values[i])) {
        EXPECT_TRUE(std::isnan(value_only[i]));
      } else if (std::isinf(full_values[i])) {
        EXPECT_EQ(value_only[i], full_values[i]);
      } else {
        EXPECT_NEAR(value_only[i], full_values[i],
                    1e-12 * std::max(1.0, std::abs(full_values[i])));
      }
    }
    visit_derivatives([](double *data, int size) {
      std::fill_n(data, size, std::numeric_limits<double>::quiet_NaN());
    });
  }
};

} // namespace

TEST(CasadiOcp, ValueOnlyEvaluationPreservesDerivativesAndMatchesValues) {
  auto settings = generated_problem::Problem::settings();
  settings.max_iterations = 1;
  const auto result = run_ocp<CheckedValueOnlyProblem>(settings);
  EXPECT_NE(result.output.exit_status, sip::Status::FACTORIZATION_FAILURE);
  EXPECT_NE(result.output.exit_status, sip::Status::FAILED_CHECK);
}

TEST(CasadiOcp, SolvesGeneratedProblem) {
  auto settings = generated_problem::Problem::settings();
  const bool print_logs =
      std::getenv("SIP_CASADI_PROBLEMS_PRINT_LOGS") != nullptr;
  if (print_logs) {
    enable_all_casadi_problem_logs(settings);
  }
  const auto result = run_ocp<generated_problem::Problem>(settings);
  if (print_logs) {
    print_casadi_problem_summary(result.output);
  }
  EXPECT_EQ(result.output.exit_status, sip::Status::SOLVED)
      << "iterations=" << result.output.num_iterations
      << " ls_iterations=" << result.output.num_ls_iterations
      << " primal=" << result.output.max_primal_violation
      << " dual=" << result.output.max_dual_violation;
}

} // namespace sip_examples::problem_definitions::casadi_problems
