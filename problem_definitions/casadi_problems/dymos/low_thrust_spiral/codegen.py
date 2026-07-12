import casadi as ca
import numpy as np

from problem_definitions.casadi_problems.codegen_common import ProblemData, main as codegen_main
from problem_definitions.casadi_problems.dymos.common import rk4_step


THRUST = 1.0e-3


def _numpy_ode(x, alpha):
    r = x[0]
    vr = x[2]
    vt = x[3]
    return np.array(
        [
            vr,
            vt / r,
            vt**2 / r - 1.0 / r**2 + THRUST * np.sin(alpha),
            -vr * vt / r + THRUST * np.cos(alpha),
        ]
    )


def _numpy_rk4(x, alpha, dt):
    k1 = _numpy_ode(x, alpha)
    k2 = _numpy_ode(x + 0.5 * dt * k1, alpha)
    k3 = _numpy_ode(x + 0.5 * dt * k2, alpha)
    k4 = _numpy_ode(x + dt * k3, alpha)
    return x + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


def _numpy_rollout(T, duration, controls):
    X = np.zeros((T + 1, 4))
    X[0] = [1.0, 0.0, 0.0, 1.0]
    dt = duration / T
    for i, alpha in enumerate(controls):
        X[i + 1] = _numpy_rk4(X[i], alpha, dt)
    return X


def _control_guess(T):
    knots = np.array(
        [
            0.513010503618,
            0.420949927,
            0.263136244,
            0.187872467,
            0.355116622,
            0.0697373118,
            0.514351161,
            -0.322442309,
            0.651320747,
            0.0726823221,
            -0.200744295,
            0.544729019,
        ]
    )
    return np.interp(np.linspace(0, len(knots) - 1, T), np.arange(len(knots)), knots)


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

    theta_init = np.array([250.00723817247032])
    controls = _control_guess(T)
    return ProblemData(
        name="dymos/low_thrust_spiral",
        T=T,
        n=n,
        m=m,
        theta_dim=1,
        c_dim=3,
        g_dim=4,
        x0=np.array([1.0, 0.0, 0.0, 1.0]),
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
