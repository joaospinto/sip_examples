import casadi as ca
import numpy as np

from problem_definitions.casadi_problems.codegen_common import ProblemData, main as codegen_main
from problem_definitions.casadi_problems.dymos.common import control_bounds, rk4_step


SIGMA = 5.4915e-5
GRAVITY = 32.174
C_EXHAUST = 1580.9425
H0 = 23800.0


def _numpy_rollout(T, duration, controls):
    dt = duration / T
    X = np.zeros((T + 1, 3))
    X[0] = [0.0, 0.0, 3.0]

    def ode(x, thrust):
        h, v, m = x
        return np.array(
            [
                v,
                (thrust - SIGMA * v**2 * np.exp(-h / H0)) / m - GRAVITY,
                -thrust / C_EXHAUST,
            ]
        )

    for i in range(T):
        x = X[i]
        u = controls[i]
        k1 = ode(x, u)
        k2 = ode(x + 0.5 * dt * k1, u)
        k3 = ode(x + 0.5 * dt * k2, u)
        k4 = ode(x + dt * k3, u)
        X[i + 1] = x + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
    return X


def make_problem() -> ProblemData:
    T = 20
    n = 3
    m = 1
    dtau = 1.0 / T

    def ode(x, u, theta):
        duration = theta[0]
        thrust = u[0]
        return duration * ca.vertcat(
            x[1],
            (thrust - SIGMA * x[1] ** 2 * ca.exp(-x[0] / H0)) / x[2] - GRAVITY,
            -thrust / C_EXHAUST,
        )

    def dynamics(x, u, theta):
        return rk4_step(ode, x, u, theta, dtau)

    def cost(x, u, theta, terminal):
        del u, theta
        return -x[0] / 10000.0 if terminal else ca.SX(0.0)

    def equalities(x, u, theta, terminal):
        del u, theta
        if terminal:
            return ca.vertcat((x[2] - 1.0) / 3.0)
        return ca.SX.zeros(1, 1)

    def inequalities(x, u, theta, terminal):
        del terminal
        return ca.vertcat(
            -x[0] / 20000.0,
            -x[1] / 1000.0,
            (1.0e-3 - x[2]) / 3.0,
            (x[2] - 3.0) / 3.0,
            control_bounds(u, [0.0], [200.0]) / 200.0,
            (20.0 - theta[0]) / 100.0,
            (theta[0] - 80.0) / 100.0,
        )

    theta_init = np.array([42.71313102824429])
    controls = np.array(
        [
            200.0,
            200.0,
            200.0,
            200.0,
            200.0,
            200.0,
            92.6966912,
            58.5928070,
            64.0131998,
            63.3218215,
            1.89677396,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
        ]
    )

    return ProblemData(
        name="dymos/goddard_rocket",
        T=T,
        n=n,
        m=m,
        theta_dim=1,
        c_dim=1,
        g_dim=8,
        x0=np.array([0.0, 0.0, 3.0]),
        X_init=_numpy_rollout(T, theta_init[0], controls),
        U_init=controls.reshape(T, 1),
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
