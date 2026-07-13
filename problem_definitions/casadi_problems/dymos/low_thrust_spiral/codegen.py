import casadi as ca
import numpy as np

from problem_definitions.casadi_problems.codegen_common import ProblemData, main as codegen_main
from problem_definitions.casadi_problems.dymos.common import rk4_step


THRUST = 1.0e-3


def make_problem() -> ProblemData:
    T = 50
    n = 4
    m = 1
    dtau = 1.0 / T

    def ode(x, u, theta):
        duration = theta[0]
        r = x[0]
        vr = x[2]
        vt = x[3]
        alpha = u[0]
        return duration * ca.vertcat(
            vr,
            vt / r,
            vt**2 / r - 1.0 / r**2 + THRUST * ca.sin(alpha),
            -vr * vt / r + THRUST * ca.cos(alpha),
        )

    def dynamics(x, u, theta):
        return rk4_step(ode, x, u, theta, dtau)

    def cost(x, u, theta, terminal):
        del x, u
        return theta[0] / 100.0 if terminal else ca.SX(0.0)

    def equalities(x, u, theta, terminal):
        del u, theta
        if terminal:
            return ca.vertcat(
                x[0] - 6.0,
                x[2],
                x[3] - 1.0 / np.sqrt(6.0),
            )
        return ca.SX.zeros(3, 1)

    def inequalities(x, u, theta, terminal):
        del u, terminal
        return ca.vertcat(
            0.5 - x[0],
            x[0] - 6.5,
            (100.0 - theta[0]) / 100.0,
            (theta[0] - 500.0) / 100.0,
        )

    theta_init = np.array([300.0])
    X_init = np.column_stack(
        (
            np.linspace(1.0, 6.0, T + 1),
            np.zeros(T + 1),
            np.zeros(T + 1),
            np.linspace(1.0, 1.0 / np.sqrt(6.0), T + 1),
        )
    )
    return ProblemData(
        name="dymos/low_thrust_spiral",
        T=T,
        n=n,
        m=m,
        theta_dim=1,
        c_dim=3,
        g_dim=4,
        x0=np.array([1.0, 0.0, 0.0, 1.0]),
        X_init=X_init,
        U_init=np.zeros((T, 1)),
        theta_init=theta_init,
        max_iterations=1000,
        cost=cost,
        dynamics=dynamics,
        equalities=equalities,
        inequalities=inequalities,
        settings_override_cpp="""
  settings.penalty.initial_penalty_parameter = 10.0;
  settings.penalty.penalty_parameter_increase_factor = 1.5;
  settings.barrier.initial_mu = 1e-3;
  settings.barrier.mu_update_factor = 0.2;
  settings.line_search.skip_line_search = false;
""",
    )


if __name__ == "__main__":
    codegen_main(make_problem)
