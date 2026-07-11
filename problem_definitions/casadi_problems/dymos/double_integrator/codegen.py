import casadi as ca
import numpy as np

from problem_definitions.casadi_problems.codegen_common import ProblemData, main as codegen_main
from problem_definitions.casadi_problems.dymos.common import control_bounds, rk4_step


def make_problem() -> ProblemData:
    T = 40
    n = 2
    m = 1
    dt = 1.0 / T

    def ode(x, u, theta):
        del theta
        return ca.vertcat(x[1], u[0])

    def dynamics(x, u, theta):
        return rk4_step(ode, x, u, theta, dt)

    def cost(x, u, theta, terminal):
        del u, theta
        return -x[0] if terminal else ca.SX(0.0)

    def equalities(x, u, theta, terminal):
        del u, theta
        return ca.vertcat(x[1]) if terminal else ca.SX.zeros(1, 1)

    def inequalities(x, u, theta, terminal):
        del x, theta, terminal
        return control_bounds(u, [-1.0], [1.0])

    x0 = np.array([0.0, 0.0])
    xf_guess = np.array([0.25, 0.0])
    u_guess = np.ones((T, m))
    u_guess[T // 2 :, 0] = -1.0
    return ProblemData(
        name="dymos/double_integrator",
        T=T,
        n=n,
        m=m,
        theta_dim=0,
        c_dim=1,
        g_dim=2,
        x0=x0,
        X_init=np.linspace(x0, xf_guess, T + 1),
        U_init=u_guess,
        theta_init=np.zeros(0),
        max_iterations=500,
        settings_override_cpp="""
  settings.penalty.initial_penalty_parameter = 10.0;
  settings.penalty.penalty_parameter_increase_factor = 1.5;
  settings.barrier.initial_mu = 1e-3;
  settings.line_search.skip_line_search = false;
""",
        cost=cost,
        dynamics=dynamics,
        equalities=equalities,
        inequalities=inequalities,
    )


if __name__ == "__main__":
    codegen_main(make_problem)
