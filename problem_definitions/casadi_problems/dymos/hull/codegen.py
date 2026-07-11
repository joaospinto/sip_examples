import casadi as ca
import numpy as np

from problem_definitions.casadi_problems.codegen_common import ProblemData, main as codegen_main
from problem_definitions.casadi_problems.dymos.common import rk4_step


def make_problem() -> ProblemData:
    T = 40
    n = 2
    m = 0
    dt = 5.0 / T

    def ode(x, u, theta):
        del u
        control = theta[0]
        return ca.vertcat(control, 0.5 * control * control)

    def dynamics(x, u, theta):
        return rk4_step(ode, x, u, theta, dt)

    def cost(x, u, theta, terminal):
        del u, theta
        return 2.5 * x[0] * x[0] + x[1] if terminal else ca.SX(0.0)

    def equalities(x, u, theta, terminal):
        del x, u, theta, terminal
        return ca.SX.zeros(0, 1)

    def inequalities(x, u, theta, terminal):
        del x, u, theta, terminal
        return ca.SX.zeros(0, 1)

    x0 = np.array([1.5, 0.0])
    xf_guess = np.array([0.5, 1.0])
    return ProblemData(
        name="dymos/hull",
        T=T,
        n=n,
        m=m,
        theta_dim=1,
        c_dim=0,
        g_dim=0,
        x0=x0,
        X_init=np.linspace(x0, xf_guess, T + 1),
        U_init=np.zeros((T, m)),
        theta_init=np.array([0.8]),
        max_iterations=500,
        settings_override_cpp="""
  settings.penalty.initial_penalty_parameter = 1.0;
  settings.barrier.initial_mu = 1e-3;
  settings.regularization.initial = 1e-6;
  settings.line_search.skip_line_search = false;
""",
        cost=cost,
        dynamics=dynamics,
        equalities=equalities,
        inequalities=inequalities,
    )


if __name__ == "__main__":
    codegen_main(make_problem)
