import casadi as ca
import numpy as np

from problem_definitions.casadi_problems.codegen_common import (
    GraphEdge,
    GraphProblemData,
    main as codegen_main,
)
from problem_definitions.casadi_problems.dymos.common import rk4_step


def make_problem() -> GraphProblemData:
    num_steps = 30
    times = np.linspace(0.0, 1.0, num_steps + 1)
    x_init = [np.array([1.0 - 2.0 * time, 0.0, 4.0 * time]) for time in times]
    controls = [np.array([-5.0]) for _ in range(num_steps)]

    def ode(x, u, theta):
        del theta
        return ca.vertcat(u[0], x[0], 0.5 * u[0] ** 2)

    def dynamics(x, u, theta):
        return rk4_step(ode, x, u, theta, 1.0 / num_steps)

    edges = [GraphEdge(i, i + 1, 1, dynamics) for i in range(num_steps)]
    terminal = num_steps

    def root_residual(x, theta):
        del theta
        return ca.vertcat(1.0 - x[0], -x[1], -x[2])

    def cost(node, x, theta):
        del theta
        return x[2] if node == terminal else ca.SX(0.0)

    def equalities(node, x, theta, outgoing_controls):
        del theta, outgoing_controls
        if node == terminal:
            return ca.vertcat(x[0] + 1.0, x[1])
        return ca.SX.zeros(0, 1)

    def inequalities(node, x, theta, outgoing_controls):
        del node, theta, outgoing_controls
        return ca.vertcat(x[1] - 0.1)

    c_dims = [0 for _ in range(num_steps + 1)]
    c_dims[terminal] = 2

    return GraphProblemData(
        name="dymos/breakwell",
        state_dims=[3 for _ in range(num_steps + 1)],
        edges=edges,
        theta_dim=0,
        c_dims=c_dims,
        g_dims=[1 for _ in range(num_steps + 1)],
        X_init=x_init,
        U_init=controls,
        theta_init=np.zeros(0),
        max_iterations=1000,
        root_residual=root_residual,
        cost=cost,
        equalities=equalities,
        inequalities=inequalities,
    )


if __name__ == "__main__":
    codegen_main(make_problem)
