import casadi as ca
import numpy as np

from problem_definitions.casadi_problems.codegen_common import ProblemData, main as codegen_main
from problem_definitions.casadi_problems.dymos.common import control_bounds, rk4_step


def make_problem() -> ProblemData:
    T = 40
    n = 3
    m = 1
    dtau = 1.0 / T
    gravity = 9.80665
    target = np.array([10.0, 5.0])
    theta_lower = np.deg2rad(0.01)
    theta_upper = np.deg2rad(179.9)
    duration_lower = 0.5
    duration_upper = 10.0

    def ode(x, u, theta):
        duration = theta[0]
        angle = u[0]
        return duration * ca.vertcat(
            x[2] * ca.sin(angle),
            -x[2] * ca.cos(angle),
            gravity * ca.cos(angle),
        )

    def dynamics(x, u, theta):
        return rk4_step(ode, x, u, theta, dtau)

    def cost(x, u, theta, terminal):
        del x, u
        return theta[0] if terminal else ca.SX(0.0)

    def equalities(x, u, theta, terminal):
        del u, theta
        return ca.vertcat(x[0] - target[0], x[1] - target[1]) if terminal else ca.SX.zeros(2, 1)

    def inequalities(x, u, theta, terminal):
        del x
        control_ineq = ca.vertcat(-1.0, -1.0) if terminal else control_bounds(u, [theta_lower], [theta_upper])
        return ca.vertcat(
            control_ineq,
            duration_lower - theta[0],
            theta[0] - duration_upper,
        )

    x0 = np.array([0.0, 10.0, 0.0])
    xf_guess = np.array([target[0], target[1], 9.9])
    return ProblemData(
        name="dymos/brachistochrone",
        T=T,
        n=n,
        m=m,
        theta_dim=1,
        c_dim=2,
        g_dim=4,
        x0=x0,
        X_init=np.linspace(x0, xf_guess, T + 1),
        U_init=np.linspace([np.deg2rad(5.0)], [np.deg2rad(100.5)], T),
        theta_init=np.array([2.0]),
        max_iterations=1000,
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
