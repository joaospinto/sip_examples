import casadi as ca
import numpy as np

from problem_definitions.casadi_problems.codegen_common import ProblemData, main as codegen_main
from problem_definitions.casadi_problems.dymos.common import rk4_step


def make_problem() -> ProblemData:
    T = 80
    n = 2
    m = 1
    tf = 10.0
    dt = tf / T

    def ode(x, u, theta):
        del theta
        return ca.vertcat(-x[0] + u[0], 0.5 * (x[0] * x[0] + u[0] * u[0]))

    def dynamics(x, u, theta):
        return rk4_step(ode, x, u, theta, dt)

    def cost(x, u, theta, terminal):
        del u, theta
        return x[1] if terminal else ca.SX(0.0)

    def equalities(x, u, theta, terminal):
        del u, theta
        return ca.vertcat(x[0] - 1.0) if terminal else ca.SX.zeros(1, 1)

    def inequalities(x, u, theta, terminal):
        del x, u, theta, terminal
        return ca.SX.zeros(0, 1)

    x0 = np.array([1.5, 0.0])
    xf_guess = np.array([1.0, 10.0])
    return ProblemData(
        name="dymos/hypersensitive",
        T=T,
        n=n,
        m=m,
        theta_dim=0,
        c_dim=1,
        g_dim=0,
        x0=x0,
        X_init=np.linspace(x0, xf_guess, T + 1),
        U_init=np.linspace([-0.6], [2.4], T),
        theta_init=np.zeros(0),
        max_iterations=1000,
        settings_override_cpp="""
  settings.penalty.initial_penalty_parameter = 10.0;
  settings.penalty.penalty_parameter_increase_factor = 1.5;
  settings.barrier.initial_mu = 1e-3;
  settings.regularization.initial = 1e-3;
  settings.line_search.skip_line_search = false;
""",
        cost=cost,
        dynamics=dynamics,
        equalities=equalities,
        inequalities=inequalities,
    )


if __name__ == "__main__":
    codegen_main(make_problem)
