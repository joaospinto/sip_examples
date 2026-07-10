import casadi as ca
import numpy as np

from problem_definitions.casadi_problems.codegen_common import ProblemData, main as codegen_main


def make_problem() -> ProblemData:
    T = 50
    n = 4
    m = 1
    dt = 0.1
    goal = np.array([0.0, np.pi, 0.0, 0.0])
    mc, mp, length = 10.0, 1.0, 0.5

    def ode(x, u, theta):
        del theta
        g = 9.81
        qd1, qd2 = x[2], x[3]
        s = ca.sin(x[1])
        c = ca.cos(x[1])
        H = ca.vertcat(
            ca.horzcat(mc + mp, mp * length * c),
            ca.horzcat(mp * length * c, mp * length * length),
        )
        C = ca.vertcat(
            ca.horzcat(0.0, -mp * qd2 * length * s),
            ca.horzcat(0.0, 0.0),
        )
        G = ca.vertcat(0.0, mp * g * length * s)
        B = ca.vertcat(1.0, 0.0)
        qd = ca.vertcat(qd1, qd2)
        drift = ca.vertcat(qd, -ca.solve(H, C @ qd + G))
        act = ca.vertcat(ca.SX.zeros(2, 1), ca.solve(H, B))
        return drift + act * u[0]

    def dynamics(x, u, theta):
        return x + dt * ode(x, u, theta)

    def cost(x, u, theta, terminal):
        del theta
        err = x - goal
        stage = 0.1 * ca.dot(err, err) + 0.01 * ca.dot(u, u)
        final = 1000.0 * ca.dot(err, err)
        return final if terminal else stage

    def equalities(x, u, theta, terminal):
        del u, theta
        return x - goal if terminal else ca.SX.zeros(n, 1)

    def inequalities(x, u, theta, terminal):
        del x, theta
        return -ca.SX.ones(2, 1) if terminal else ca.vertcat(u[0] - 5.0, -5.0 - u[0])

    x0 = np.array([0.0, 0.2, 0.0, -0.1])
    return ProblemData(
        name="cartpole",
        T=T,
        n=n,
        m=m,
        theta_dim=0,
        c_dim=n,
        g_dim=2,
        x0=x0,
        X_init=np.linspace(x0, goal, T + 1),
        U_init=np.zeros((T, m)),
        theta_init=np.zeros(0),
        max_iterations=1000,
        settings_override_cpp="""
  settings.penalty.initial_penalty_parameter = 1.0;
  settings.penalty.penalty_parameter_increase_factor = 1.5;
  settings.barrier.initial_mu = 1e-6;
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
