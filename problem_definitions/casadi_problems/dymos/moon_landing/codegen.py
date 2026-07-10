import casadi as ca
import numpy as np

from problem_definitions.casadi_problems.codegen_common import ProblemData, main as codegen_main
from problem_definitions.casadi_problems.dymos.common import control_bounds, rk4_step


def make_problem() -> ProblemData:
    T = 60
    n = 3
    m = 1
    dtau = 1.0 / T
    thrust_upper = 1.227
    mass_lower = 1e-3
    duration_lower = 0.2
    duration_upper = 3.0

    def ode(x, u, theta):
        duration = theta[0]
        thrust = u[0]
        return duration * ca.vertcat(
            x[1],
            -1.0 + thrust / x[2],
            -thrust / 2.349,
        )

    def dynamics(x, u, theta):
        return rk4_step(ode, x, u, theta, dtau)

    def cost(x, u, theta, terminal):
        del u, theta
        return -x[2] if terminal else ca.SX(0.0)

    def equalities(x, u, theta, terminal):
        del u, theta
        return ca.vertcat(x[0], x[1]) if terminal else ca.SX.zeros(2, 1)

    def inequalities(x, u, theta, terminal):
        control_ineq = ca.vertcat(-1.0, -1.0) if terminal else control_bounds(u, [0.0], [thrust_upper])
        return ca.vertcat(
            control_ineq,
            mass_lower - x[2],
            duration_lower - theta[0],
            theta[0] - duration_upper,
        )

    x0 = np.array([1.0, -0.783, 1.0])
    xf_guess = np.array([0.0, 0.0, 0.4])
    return ProblemData(
        name="dymos/moon_landing",
        T=T,
        n=n,
        m=m,
        theta_dim=1,
        c_dim=2,
        g_dim=5,
        x0=x0,
        X_init=np.linspace(x0, xf_guess, T + 1),
        U_init=np.linspace([0.0], [thrust_upper], T),
        theta_init=np.array([1.4]),
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
