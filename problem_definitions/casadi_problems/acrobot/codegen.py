import casadi as ca
import numpy as np

from problem_definitions.casadi_problems.codegen_common import ProblemData, main as codegen_main


def make_problem() -> ProblemData:
    T = 50
    n = 4
    m = 1
    dt = 0.1
    goal = np.array([np.pi, 0.0, 0.0, 0.0])
    m1, m2, l1, lc1, lc2, I1, I2 = (1.0, 1.0, 1.0, 0.5, 0.5, 1.0, 1.0)

    def ode(x, u, theta):
        del theta
        g = 9.8
        th1, th2, dth1, dth2 = x[0], x[1], x[2], x[3]
        d1 = (
            m1 * lc1**2
            + m2 * (l1**2 + lc2**2 + 2 * l1 * lc2 * ca.cos(th2))
            + I1
            + I2
        )
        d2 = m2 * (lc2**2 + l1 * lc2 * ca.cos(th2)) + I2
        phi2 = m2 * lc2 * g * ca.cos(th1 + th2 - np.pi / 2.0)
        phi1 = (
            -m2 * l1 * lc2 * dth2**2 * ca.sin(th2)
            - 2 * m2 * l1 * lc2 * dth2 * dth1 * ca.sin(th2)
            + (m1 * lc1 + m2 * l1) * g * ca.cos(th1 - np.pi / 2.0)
            + phi2
        )
        ddth2 = (
            u[0] + d2 / d1 * phi1 - m2 * l1 * lc2 * dth1**2 * ca.sin(th2) - phi2
        ) / (m2 * lc2**2 + I2 - d2**2 / d1)
        ddth1 = -(d2 * ddth2 + phi1) / d1
        return ca.vertcat(dth1, dth2, ddth1, ddth2)

    def dynamics(x, u, theta):
        return x + dt * ode(x, u, theta)

    def cost(x, u, theta, terminal):
        del theta
        delta = x - goal
        terminal_cost = 0.5 * 1000.0 * ca.dot(delta, delta)
        stage_cost = 0.5 * 0.1 * ca.dot(delta, delta) + 0.5 * 0.01 * ca.dot(u, u)
        return terminal_cost if terminal else stage_cost

    def equalities(x, u, theta, terminal):
        del x, u, theta, terminal
        return ca.SX.zeros(0, 1)

    def inequalities(x, u, theta, terminal):
        del x, u, theta, terminal
        return ca.SX.zeros(0, 1)

    x0 = np.zeros(n)
    return ProblemData(
        name="acrobot",
        T=T,
        n=n,
        m=m,
        theta_dim=0,
        c_dim=0,
        g_dim=0,
        x0=x0,
        X_init=np.zeros((T + 1, n)),
        U_init=np.zeros((T, m)),
        theta_init=np.zeros(0),
        max_iterations=1000,
        settings_override_cpp="""
  settings.penalty.initial_penalty_parameter = 100.0;
  settings.penalty.penalty_parameter_increase_factor = 1.5;
  settings.line_search.skip_line_search = false;
""",
        cost=cost,
        dynamics=dynamics,
        equalities=equalities,
        inequalities=inequalities,
    )




if __name__ == "__main__":
    codegen_main(make_problem)
