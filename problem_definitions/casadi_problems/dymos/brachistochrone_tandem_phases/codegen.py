import casadi as ca
import numpy as np

from problem_definitions.casadi_problems.codegen_common import (
    GraphEdge,
    GraphProblemData,
    main as codegen_main,
)
from problem_definitions.casadi_problems.dymos.common import control_bounds, rk4_step


DEG_TO_RAD = np.pi / 180.0
GRAVITY = 9.80665


def make_problem() -> GraphProblemData:
    num_steps = 20
    theta_init = np.array([2.0])
    times = np.linspace(0.0, 1.0, num_steps + 1)
    starts = np.array([0.0, 10.0, 0.0, 0.0])
    finishes = np.array([10.0, 5.0, 9.9, np.sqrt(125.0)])
    x_init = [starts + time * (finishes - starts) for time in times]
    controls = [
        np.array([(5.0 + time * (100.5 - 5.0)) * DEG_TO_RAD])
        for time in times[:-1]
    ]

    def ode(x, u, theta):
        del theta
        angle = u[0]
        arclength_rate = (
            ca.sqrt(1.0 + (1.0 / ca.tan(angle)) ** 2)
            * x[2]
            * ca.sin(angle)
        )
        return ca.vertcat(
            x[2] * ca.sin(angle),
            -x[2] * ca.cos(angle),
            GRAVITY * ca.cos(angle),
            arclength_rate,
        )

    def dynamics(x, u, theta):
        return rk4_step(ode, x, u, theta, theta[0] / num_steps)

    edges = [GraphEdge(i, i + 1, 1, dynamics) for i in range(num_steps)]
    terminal = num_steps

    def root_residual(x, theta):
        del theta
        return starts - x

    def cost(node, x, theta):
        del theta
        return x[3] if node == terminal else ca.SX(0.0)

    def equalities(node, x, theta, outgoing_controls):
        del theta, outgoing_controls
        if node == terminal:
            return ca.vertcat(x[0] - 10.0, x[1] - 5.0)
        return ca.SX.zeros(0, 1)

    def inequalities(node, x, theta, outgoing_controls):
        pieces = []
        if outgoing_controls:
            pieces.append(
                control_bounds(
                    outgoing_controls[0] / DEG_TO_RAD,
                    [0.01],
                    [179.9],
                )
            )
        if node == 0:
            pieces.extend([0.5 - theta[0], theta[0] - 10.0])
        return ca.vertcat(*pieces) if pieces else ca.SX.zeros(0, 1)

    c_dims = [0 for _ in range(num_steps + 1)]
    c_dims[terminal] = 2
    g_dims = [2 for _ in range(num_steps)] + [0]
    g_dims[0] = 4

    return GraphProblemData(
        name="dymos/brachistochrone_tandem_phases",
        state_dims=[4 for _ in range(num_steps + 1)],
        edges=edges,
        theta_dim=1,
        c_dims=c_dims,
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
