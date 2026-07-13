import casadi as ca
import numpy as np

from problem_definitions.casadi_problems.codegen_common import (
    GraphEdge,
    GraphProblemData,
    main as codegen_main,
)
from problem_definitions.casadi_problems.dymos.common import control_bounds, rk4_step


GRAVITY = 9.80665
RHO_REF = 1.225
ALTITUDE_SCALE = 8.44e3
CD_AREA = 0.5 * 7.069
THRUST = 2.1e6
ISP = 265.2
STATE_SCALES = np.array([1.0e5, 1.0e5, 1.0e3, 1.0e3, 1.0e3])


def make_problem() -> GraphProblemData:
    num_steps = 12
    duration = np.array([150.0])
    times = np.linspace(0.0, 1.0, num_steps + 1)
    x_init = [
        np.array(
            [
                1.15e5 * time,
                1.85e5 * time,
                1.0,
                1.0e-6 * (1.0 - time),
                117000.0 + (1163.0 - 117000.0) * time,
            ]
        )
        for time in times
    ]
    controls = [np.array([1.5 + (-0.76 - 1.5) * time]) for time in times[:-1]]

    def ode(x, u, theta):
        del theta
        rho = RHO_REF * ca.exp(-x[1] / ALTITUDE_SCALE)
        return ca.vertcat(
            x[2],
            x[3],
            (THRUST * ca.cos(u[0]) - CD_AREA * 0.5 * rho * x[2] ** 2) / x[4],
            (THRUST * ca.sin(u[0]) - CD_AREA * 0.5 * rho * x[3] ** 2) / x[4] - GRAVITY,
            -THRUST / (GRAVITY * ISP),
        )

    def dynamics(x, u, theta):
        return rk4_step(ode, x, u, theta, theta[0] / num_steps)

    edges = [GraphEdge(i, i + 1, 1, dynamics) for i in range(num_steps)]
    terminal = num_steps

    def root_residual(x, theta):
        del theta
        return ca.vertcat(-x[0], -x[1], 1.0 - x[2], 1.0e-6 - x[3], 117000.0 - x[4])

    def cost(node, x, theta):
        del x
        return 0.01 * theta[0] if node == terminal else ca.SX(0.0)

    def equalities(node, x, theta, outgoing_controls):
        del theta, outgoing_controls
        if node == terminal:
            return ca.vertcat(
                (x[1] - 1.85e5) / 1.0e5,
                (x[2] - 7796.6961) / 1.0e3,
                x[3] / 1.0e3,
            )
        return ca.SX.zeros(0, 1)

    def inequalities(node, x, theta, outgoing_controls):
        del x
        pieces = []
        if outgoing_controls:
            pieces.append(control_bounds(outgoing_controls[0], [-1.57], [1.57]))
        if node == 0:
            pieces.extend([10.0 - theta[0], theta[0] - 500.0])
        return ca.vertcat(*pieces) if pieces else ca.SX.zeros(0, 1)

    c_dims = [0 for _ in range(num_steps + 1)]
    c_dims[terminal] = 3
    g_dims = [2 for _ in range(num_steps)] + [0]
    g_dims[0] = 4

    return GraphProblemData(
        name="dymos/ssto_earth",
        state_dims=[5 for _ in range(num_steps + 1)],
        edges=edges,
        theta_dim=1,
        c_dims=c_dims,
        g_dims=g_dims,
        X_init=x_init,
        U_init=controls,
        theta_init=duration,
        max_iterations=1000,
        root_residual=root_residual,
        cost=cost,
        equalities=equalities,
        inequalities=inequalities,
        state_scales=[STATE_SCALES.copy() for _ in range(num_steps + 1)],
    )


if __name__ == "__main__":
    codegen_main(make_problem)
