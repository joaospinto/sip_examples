#include "problem_definitions/mock_problems/simple_branched_lqr/problem.hpp"

namespace sip_examples::problem_definitions::simple_branched_lqr {
namespace {

constexpr std::array<std::array<double, kXDim>, kYDim> kJacobianC = {{
    {-1.0, 0.0, 0.0, 0.0, 0.0},
    {1.0, 1.0, -1.0, 0.0, 0.0},
    {1.0, 0.0, 0.0, 1.0, -1.0},
    {0.0, 0.0, 0.0, 0.0, 0.0},
    {0.0, 0.0, 1.0, 0.0, 0.0},
    {0.0, 0.0, 0.0, 0.0, 1.0},
}};

constexpr auto count_nonzeros() -> int {
  int count = 0;
  for (const auto &row : kJacobianC) {
    for (const double value : row) {
      count += value != 0.0;
    }
  }
  return count;
}

static_assert(count_nonzeros() == kJacobianCNnz);

struct Evaluation {
  double f;
  std::array<double, kXDim> gradient_f;
  std::array<double, kXDim> hessian_diagonal;
  std::array<double, kYDim> c;
};

auto evaluate_common(const std::array<double, kXDim> &x) -> Evaluation {
  const double x0 = x[0];
  const double u0 = x[1];
  const double x1 = x[2];
  const double u1 = x[3];
  const double x2 = x[4];

  return Evaluation{
      .f = 0.5 * u0 * u0 + 0.5 * u1 * u1,
      .gradient_f = {0.0, u0, 0.0, u1, 0.0},
      .hessian_diagonal = {0.0, 1.0, 0.0, 1.0, 0.0},
      .c = {-x0, x0 + u0 - x1, x0 + u1 - x2, 0.0, x1 - 1.0, x2 + 1.0},
  };
}

void copy_flat_evaluation(const Evaluation &evaluation, double *f,
                          double *gradient_f, double *c,
                          double *upper_hessian_lagrangian) {
  *f = evaluation.f;
  for (int i = 0; i < kXDim; ++i) {
    gradient_f[i] = evaluation.gradient_f[i];
    upper_hessian_lagrangian[i] = evaluation.hessian_diagonal[i];
  }
  for (int i = 0; i < kYDim; ++i) {
    c[i] = evaluation.c[i];
  }
}

} // namespace

auto settings() -> sip::Settings {
  return sip::Settings{
      .max_iterations = 100,
      .termination =
          {
              .max_dual_residual = 1e-9,
              .max_constraint_violation = 1e-9,
              .max_complementarity_gap = 1e-9,
          },
      .line_search =
          {
              .max_iterations = 100,
              .skip_line_search = true,
          },
  };
}

void initialize(sip::Workspace &workspace) {
  for (int i = 0; i < kXDim; ++i) {
    workspace.vars.x[i] = 0.0;
  }
  for (int i = 0; i < kYDim; ++i) {
    workspace.vars.y[i] = 0.0;
  }
}

void configure_qdldl_sparsity(sip_qdldl::ModelCallbackOutput &mco) {
  mco.upper_hessian_lagrangian.rows = kXDim;
  mco.upper_hessian_lagrangian.cols = kXDim;
  for (int i = 0; i < kXDim; ++i) {
    mco.upper_hessian_lagrangian.ind[i] = i;
    mco.upper_hessian_lagrangian.indptr[i] = i;
  }
  mco.upper_hessian_lagrangian.indptr[kXDim] = kXDim;
  mco.upper_hessian_lagrangian.is_transposed = false;

  mco.jacobian_c.rows = kXDim;
  mco.jacobian_c.cols = kYDim;
  int k = 0;
  for (int constraint = 0; constraint < kYDim; ++constraint) {
    mco.jacobian_c.indptr[constraint] = k;
    for (int variable = 0; variable < kXDim; ++variable) {
      if (kJacobianC[constraint][variable] != 0.0) {
        mco.jacobian_c.ind[k++] = variable;
      }
    }
  }
  mco.jacobian_c.indptr[kYDim] = k;
  mco.jacobian_c.is_transposed = true;

  mco.jacobian_g.rows = kXDim;
  mco.jacobian_g.cols = kSDim;
  mco.jacobian_g.indptr[0] = 0;
  mco.jacobian_g.is_transposed = true;
}

void evaluate_qdldl(const sip::ModelCallbackInput &mci, double *f,
                    double *gradient_f, double *c,
                    double *upper_hessian_lagrangian, double *jacobian_c) {
  const std::array<double, kXDim> x = {mci.x[0], mci.x[1], mci.x[2], mci.x[3],
                                       mci.x[4]};
  copy_flat_evaluation(evaluate_common(x), f, gradient_f, c,
                       upper_hessian_lagrangian);
  int k = 0;
  for (int constraint = 0; constraint < kYDim; ++constraint) {
    for (int variable = 0; variable < kXDim; ++variable) {
      if (kJacobianC[constraint][variable] != 0.0) {
        jacobian_c[k++] = kJacobianC[constraint][variable];
      }
    }
  }
}

void evaluate_slacg(const sip::ModelCallbackInput &mci, double *f,
                    double *gradient_f, double *c,
                    double *upper_hessian_lagrangian, double *jacobian_c) {
  const std::array<double, kXDim> x = {mci.x[0], mci.x[1], mci.x[2], mci.x[3],
                                       mci.x[4]};
  copy_flat_evaluation(evaluate_common(x), f, gradient_f, c,
                       upper_hessian_lagrangian);
  int k = 0;
  for (int variable = 0; variable < kXDim; ++variable) {
    for (int constraint = 0; constraint < kYDim; ++constraint) {
      if (kJacobianC[constraint][variable] != 0.0) {
        jacobian_c[k++] = kJacobianC[constraint][variable];
      }
    }
  }
}

void evaluate_optimal_control(
    const sip::optimal_control::ModelCallbackInput &mci,
    sip::optimal_control::ModelCallbackOutput &mco) {
  const std::array<double, kXDim> x = {
      mci.nodes[0].state[0], mci.edges[0].control[0], mci.nodes[1].state[0],
      mci.edges[1].control[0], mci.nodes[2].state[0]};
  const Evaluation evaluation = evaluate_common(x);

  for (int node = 0; node < kNumNodes; ++node) {
    const int state_index = 2 * node;
    auto &output = mco.nodes[node];
    output.f = 0.0;
    output.df_dx[0] = evaluation.gradient_f[state_index];
    output.d2L_dx2[0] = evaluation.hessian_diagonal[state_index];
    if (kNodeCDims[node] > 0) {
      output.c[0] = evaluation.c[kNumNodes + node];
      output.dc_dx[0] = kJacobianC[kNumNodes + node][state_index];
    }
  }
  for (int edge = 0; edge < kNumEdges; ++edge) {
    const int control_index = 2 * edge + 1;
    const int parent_state_index = 2 * kEdgeParents[edge];
    auto &output = mco.edges[edge];
    output.f = 0.5 * x[control_index] * x[control_index];
    output.df_dx[0] = 0.0;
    output.df_du[0] = evaluation.gradient_f[control_index];
    output.dyn_res[0] = evaluation.c[edge + 1];
    output.ddyn_dx[0] = kJacobianC[edge + 1][parent_state_index];
    output.ddyn_du[0] = kJacobianC[edge + 1][control_index];
    output.d2L_dx2[0] = 0.0;
    output.d2L_dxdu[0] = 0.0;
    output.d2L_du2[0] = evaluation.hessian_diagonal[control_index];
  }
}

void initialize(sip::optimal_control::Workspace &workspace) {
  for (int i = 0; i < kXDim; ++i) {
    workspace.sip_workspace.vars.x[i] = 0.0;
  }
  for (int i = 0; i < kOcpYDim; ++i) {
    workspace.sip_workspace.vars.y[i] = 0.0;
  }
}

} // namespace sip_examples::problem_definitions::simple_branched_lqr
