import casadi as ca
import numpy as np

from problem_definitions.casadi_problems.codegen_common import (
    GraphEdge,
    GraphProblemData,
    main as codegen_main,
)
from problem_definitions.casadi_problems.dymos.common import control_bounds, rk4_step


def make_problem() -> GraphProblemData:
    num_steps = 60
    theta_init = np.array([102.479])
    times = np.linspace(0.0, 1.0, num_steps + 1)
    x_init = [np.array([-0.5 + time, 0.07 * time]) for time in times]
    controls = [np.array([1.0 if time < 0.55 else -1.0]) for time in times[:-1]]

    def ode(x, u, theta):
        del theta
        return ca.vertcat(x[1], 0.001 * u[0] - 0.0025 * ca.cos(3.0 * x[0]))

    def dynamics(x, u, theta):
        return rk4_step(ode, x, u, theta, theta[0] / num_steps)

    edges = [GraphEdge(i, i + 1, 1, dynamics) for i in range(num_steps)]
    terminal = num_steps

    def root_residual(x, theta):
        del theta
        return ca.vertcat(-0.5 - x[0], -x[1])

    def cost(node, x, theta):
        del x
        return theta[0] / 1000.0 if node == terminal else ca.SX(0.0)

    def equalities(node, x, theta, outgoing_controls):
        del node, x, theta, outgoing_controls
        return ca.SX.zeros(0, 1)

    def inequalities(node, x, theta, outgoing_controls):
        pieces = [
            -1.2 - x[0],
            x[0] - 0.5,
            -0.07 - x[1],
            x[1] - 0.07,
        ]
        if outgoing_controls:
            pieces.append(control_bounds(outgoing_controls[0], [-1.0], [1.0]))
        if node == 0:
            pieces.extend([0.05 - theta[0], theta[0] - 10000.0])
        if node == terminal:
            pieces.extend([0.5 - x[0], -x[1]])
        return ca.vertcat(*pieces)

    g_dims = [6 for _ in range(num_steps + 1)]
    g_dims[0] = 8

    return GraphProblemData(
        name="dymos/mountain_car",
        state_dims=[2 for _ in range(num_steps + 1)],
        edges=edges,
        theta_dim=1,
        c_dims=[0 for _ in range(num_steps + 1)],
        g_dims=g_dims,
        X_init=x_init,
        U_init=controls,
        theta_init=theta_init,
        max_iterations=1000,
        root_residual=root_residual,
        cost=cost,
        equalities=equalities,
        inequalities=inequalities,
    )


if __name__ == "__main__":
    codegen_main(make_problem)
