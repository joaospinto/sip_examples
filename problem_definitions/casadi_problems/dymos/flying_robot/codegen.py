import casadi as ca
import numpy as np

from problem_definitions.casadi_problems.codegen_common import ProblemData, main as codegen_main
from problem_definitions.casadi_problems.dymos.common import control_bounds, rk4_step


def _bang_bang_guess(T, duration):
    X = np.zeros((T + 1, 4))
    U = np.zeros((T, 2))
    for i in range(T + 1):
        t = duration * i / T
        if t <= duration / 2.0:
            ax = -1.0
            ay = 1.0
            X[i, 0] = 0.5 * ax * t * t
            X[i, 1] = 0.5 * ay * t * t
            X[i, 2] = ax * t
            X[i, 3] = ay * t
        else:
            s = t - duration / 2.0
            X[i, 0] = -50.0 - 10.0 * s + 0.5 * s * s
            X[i, 1] = 50.0 + 10.0 * s - 0.5 * s * s
            X[i, 2] = -10.0 + s
            X[i, 3] = 10.0 - s
    for i in range(T):
        t = duration * (i + 0.5) / T
        U[i] = [-1.0, 1.0] if t <= duration / 2.0 else [1.0, -1.0]
    return X, U


def make_problem() -> ProblemData:
    T = 60
    n = 4
    m = 2
    dtau = 1.0 / T
    duration_lower = 0.1
    duration_upper = 1.0e4
    target = np.array([-100.0, 100.0, 0.0, 0.0])

    def ode(x, u, theta):
        duration = theta[0]
        return duration * ca.vertcat(x[2], x[3], u[0], u[1])

    def dynamics(x, u, theta):
        return rk4_step(ode, x, u, theta, dtau)

    def cost(x, u, theta, terminal):
        del x, u
        return theta[0] if terminal else ca.SX(0.0)

    def equalities(x, u, theta, terminal):
        del u, theta
        return x - target if terminal else ca.SX.zeros(4, 1)

    def inequalities(x, u, theta, terminal):
        del x
        control_ineq = control_bounds(u, [-1.0, -1.0], [1.0, 1.0])
        return ca.vertcat(
            control_ineq,
            duration_lower - theta[0],
            theta[0] - duration_upper,
        )

    x0 = np.array([0.0, 0.0, 0.0, 0.0])
    X_init, U_init = _bang_bang_guess(T, 20.0)
    return ProblemData(
        name="dymos/flying_robot",
        T=T,
        n=n,
        m=m,
        theta_dim=1,
        c_dim=4,
        g_dim=6,
        x0=x0,
        X_init=X_init,
        U_init=U_init,
        theta_init=np.array([20.0]),
        max_iterations=1000,
        settings_override_cpp="""
  settings.penalty.initial_penalty_parameter = 10.0;
  settings.penalty.penalty_parameter_increase_factor = 1.5;
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
