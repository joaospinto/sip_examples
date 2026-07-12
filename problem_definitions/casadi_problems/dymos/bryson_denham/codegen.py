import casadi as ca
import numpy as np

from problem_definitions.casadi_problems.codegen_common import ProblemData, main as codegen_main


def rk4_step(ode, x, u, theta, dt):
    k1 = ode(x, u, theta)
    k2 = ode(x + 0.5 * dt * k1, u, theta)
    k3 = ode(x + 0.5 * dt * k2, u, theta)
    k4 = ode(x + dt * k3, u, theta)
    return x + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


def make_problem() -> ProblemData:
    T = 60
    n = 3
    m = 1
    dt = 1.0 / T

    def ode(x, u, theta):
        del theta
        return ca.vertcat(x[1], u[0], 0.5 * u[0] * u[0])

    def dynamics(x, u, theta):
        return rk4_step(ode, x, u, theta, dt)

    def cost(x, u, theta, terminal):
        del u, theta
        return x[2] if terminal else ca.SX(0.0)

    def equalities(x, u, theta, terminal):
        del u, theta
        return ca.vertcat(x[0], x[1] + 1.0) if terminal else ca.SX.zeros(2, 1)

    def inequalities(x, u, theta, terminal):
        del u, theta, terminal
        return ca.vertcat(x[0] - 1.0 / 9.0)

    x0 = np.array([0.0, 1.0, 0.0])
    xf_guess = np.array([0.0, -1.0, 5.0])
    return ProblemData(
        name="dymos/bryson_denham",
        T=T,
        n=n,
        m=m,
        theta_dim=0,
        c_dim=2,
        g_dim=1,
        x0=x0,
        X_init=np.linspace(x0, xf_guess, T + 1),
        U_init=-2.0 * np.ones((T, m)),
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
