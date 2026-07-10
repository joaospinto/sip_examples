import casadi as ca
import numpy as np

from problem_definitions.casadi_problems.codegen_common import ProblemData, main as codegen_main
from problem_definitions.casadi_problems.dymos.common import control_bounds, rk4_step


def make_problem() -> ProblemData:
    T = 80
    n = 3
    m = 1
    dt = 10.0 / T

    def ode(x, u, theta):
        del theta
        x0 = x[0]
        x1 = x[1]
        control = u[0]
        return ca.vertcat(
            (1.0 - x1 * x1) * x0 - x1 + control,
            x0,
            x0 * x0 + x1 * x1 + control * control,
        )

    def dynamics(x, u, theta):
        return rk4_step(ode, x, u, theta, dt)

    def cost(x, u, theta, terminal):
        del u, theta
        return x[2] if terminal else ca.SX(0.0)

    def equalities(x, u, theta, terminal):
        del u, theta
        return ca.vertcat(x[0], x[1]) if terminal else ca.SX.zeros(2, 1)

    def inequalities(x, u, theta, terminal):
        del x, theta, terminal
        return control_bounds(u, [-0.75], [1.0])

    x0 = np.array([1.0, 1.0, 0.0])
    xf_guess = np.array([0.0, 0.0, 10.0])
    return ProblemData(
        name="dymos/vanderpol",
        T=T,
        n=n,
        m=m,
        theta_dim=0,
        c_dim=2,
        g_dim=2,
        x0=x0,
        X_init=np.linspace(x0, xf_guess, T + 1),
        U_init=-0.75 * np.ones((T, m)),
        theta_init=np.zeros(0),
        max_iterations=1500,
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
