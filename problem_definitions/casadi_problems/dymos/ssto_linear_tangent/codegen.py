import casadi as ca
import numpy as np

from problem_definitions.casadi_problems.codegen_common import ProblemData, main as codegen_main
from problem_definitions.casadi_problems.dymos.common import rk4_step


GRAVITY = 1.61544
THRUST = 3.0 * 50000.0 * GRAVITY
ISP = 1.0e6


def _numpy_rollout(T, theta):
    duration, a_ctrl, b_ctrl = theta
    dt = duration / T
    X = np.zeros((T + 1, 6))
    X[0] = [0.0, 0.0, 0.0, 0.0, 1.0e-6, 50000.0]

    def ode(x):
        pitch = np.arctan(a_ctrl * x[0] + b_ctrl)
        return np.array(
            [
                1.0,
                x[3],
                x[4],
                THRUST * np.cos(pitch) / x[5],
                THRUST * np.sin(pitch) / x[5] - GRAVITY,
                -THRUST / (GRAVITY * ISP),
            ]
        )

    for i in range(T):
        x = X[i]
        k1 = ode(x)
        k2 = ode(x + 0.5 * dt * k1)
        k3 = ode(x + 0.5 * dt * k2)
        k4 = ode(x + dt * k3)
        X[i + 1] = x + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
    return X


def make_problem() -> ProblemData:
    T = 10
    n = 6
    m = 0
    dtau = 1.0 / T
    target_y = 1.85e5
    target_vx = 1627.0

    def ode(x, u, theta):
        del u
        duration = theta[0]
        a_ctrl = theta[1]
        b_ctrl = theta[2]
        pitch = ca.atan(a_ctrl * x[0] + b_ctrl)
        return duration * ca.vertcat(
            1.0,
            x[3],
            x[4],
            THRUST * ca.cos(pitch) / x[5],
            THRUST * ca.sin(pitch) / x[5] - GRAVITY,
            -THRUST / (GRAVITY * ISP),
        )

    def dynamics(x, u, theta):
        return rk4_step(ode, x, u, theta, dtau)

    def cost(x, u, theta, terminal):
        del x, u
        return 0.01 * theta[0] if terminal else ca.SX(0.0)

    def equalities(x, u, theta, terminal):
        del u, theta
        if terminal:
            return ca.vertcat(
                (x[2] - target_y) / 1.0e5,
                (x[3] - target_vx) / 1000.0,
                x[4] / 1000.0,
            )
        return ca.SX.zeros(3, 1)

    def inequalities(x, u, theta, terminal):
        del u, terminal
        return ca.vertcat(
            -x[1] / 1.0e5,
            -x[2] / 1.0e5,
            -x[3] / 1000.0,
            10.0 - theta[0],
            theta[0] - 1000.0,
        )

    theta_init = np.array([481.717002, -0.00828050507, 2.74740120])
    x0 = np.array([0.0, 0.0, 0.0, 0.0, 1.0e-6, 50000.0])
    return ProblemData(
        name="dymos/ssto_linear_tangent",
        T=T,
        n=n,
        m=m,
        theta_dim=3,
        c_dim=3,
        g_dim=5,
        x0=x0,
        X_init=_numpy_rollout(T, theta_init),
        U_init=np.zeros((T, 0)),
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
  settings.line_search.skip_line_search = false;
""",
    )


if __name__ == "__main__":
    codegen_main(make_problem)
