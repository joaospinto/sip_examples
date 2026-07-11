import casadi as ca
import numpy as np

from problem_definitions.casadi_problems.codegen_common import ProblemData, main as codegen_main
from problem_definitions.casadi_problems.dymos.common import control_bounds, rk4_step


def make_problem() -> ProblemData:
    T = 60
    n = 6
    m = 3
    dtau = 1.0 / T
    arm_length = 5.0
    duration_lower = 0.0
    target = np.array([4.5, 2.0 * np.pi / 3.0, np.pi / 4.0, 0.0, 0.0, 0.0])

    def ode(x, u, theta):
        duration = theta[0]
        denom = (arm_length - x[0]) ** 3 + x[0] ** 3
        return duration * ca.vertcat(
            x[3],
            x[4],
            x[5],
            u[0] / arm_length,
            3.0 * u[1] / (denom * ca.sin(x[2]) ** 2),
            3.0 * u[2] / denom,
        )

    def dynamics(x, u, theta):
        return rk4_step(ode, x, u, theta, dtau)

    def cost(x, u, theta, terminal):
        del x, u
        return theta[0] if terminal else ca.SX(0.0)

    def equalities(x, u, theta, terminal):
        del u, theta
        return x - target if terminal else ca.SX.zeros(6, 1)

    def inequalities(x, u, theta, terminal):
        del x, terminal
        return ca.vertcat(
            control_bounds(u, [-1.0, -1.0, -1.0], [1.0, 1.0, 1.0]),
            duration_lower - theta[0],
        )

    x0 = np.array([4.5, 0.0, np.pi / 4.0, 0.0, 0.0, 0.0])
    return ProblemData(
        name="dymos/robot_arm",
        T=T,
        n=n,
        m=m,
        theta_dim=1,
        c_dim=6,
        g_dim=7,
        x0=x0,
        X_init=np.linspace(x0, target, T + 1),
        U_init=np.zeros((T, m)),
        theta_init=np.array([10.0]),
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
