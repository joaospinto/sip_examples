import casadi as ca
import numpy as np

from problem_definitions.casadi_problems.codegen_common import (
    GraphEdge,
    GraphProblemData,
    main as codegen_main,
)
from problem_definitions.casadi_problems.dymos.atmosphere_1976 import (
    linear_density_casadi,
    linear_density_numpy,
)
from problem_definitions.casadi_problems.dymos.common import rk4_step


DEG_TO_RAD = np.pi / 180.0
GRAVITY = 9.80665
DENSITY = 7870.0
DRAG_COEFFICIENT = 0.5


def _mass(radius):
    return (4.0 / 3.0) * DENSITY * np.pi * radius**3


def _area(radius):
    return np.pi * radius**2


def _numpy_ode(x, theta):
    radius = theta[1]
    density = linear_density_numpy(x[1])
    drag = 0.5 * density * x[3] ** 2 * _area(radius) * DRAG_COEFFICIENT
    return np.array(
        [
            x[3] * np.cos(x[2]),
            x[3] * np.sin(x[2]),
            -(GRAVITY / x[3]) * np.cos(x[2]),
            -drag / _mass(radius) - GRAVITY * np.sin(x[2]),
        ]
    )


def _numpy_rollout(num_steps, theta):
    states = [np.array([0.0, 0.0, theta[2], theta[3]])]
    step = theta[0] / num_steps
    for _ in range(num_steps):
        x = states[-1]
        k1 = _numpy_ode(x, theta)
        k2 = _numpy_ode(x + 0.5 * step * k1, theta)
        k3 = _numpy_ode(x + 0.5 * step * k2, theta)
        k4 = _numpy_ode(x + step * k3, theta)
        states.append(x + step * (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0)
    return states


def make_problem() -> GraphProblemData:
    num_steps = 10
    theta_init = np.array([20.0, 0.04, 45.0 * DEG_TO_RAD, 500.0])
    trajectory_init = _numpy_rollout(num_steps, theta_init)

    def ode(x, u, theta):
        del u
        radius = theta[1]
        density = linear_density_casadi(x[1])
        drag = 0.5 * density * x[3] ** 2 * _area(radius) * DRAG_COEFFICIENT
        return ca.vertcat(
            x[3] * ca.cos(x[2]),
            x[3] * ca.sin(x[2]),
            -(GRAVITY / x[3]) * ca.cos(x[2]),
            -drag / _mass(radius) - GRAVITY * ca.sin(x[2]),
        )

    def dynamics(x, u, theta):
        return rk4_step(ode, x, u, theta, theta[0] / num_steps)

    def initialize_state(x, u, theta):
        del x, u
        return ca.vertcat(0.0, 0.0, theta[2], theta[3])

    edges = [GraphEdge(0, 1, 0, initialize_state)]
    edges.extend(
        GraphEdge(i + 1, i + 2, 0, dynamics) for i in range(num_steps)
    )
    terminal = num_steps + 1

    def root_residual(x, theta):
        del theta
        return ca.vertcat(-x[0])

    def cost(node, x, theta):
        del theta
        return -x[0] / 1000.0 if node == terminal else ca.SX(0.0)

    def equalities(node, x, theta, outgoing_controls):
        del theta, outgoing_controls
        return ca.vertcat(x[1] / 1000.0) if node == terminal else ca.SX.zeros(0, 1)

    def inequalities(node, x, theta, outgoing_controls):
        del x, outgoing_controls
        if node != 0:
            return ca.SX.zeros(0, 1)
        energy = 0.5 * _mass(theta[1]) * theta[3] ** 2
        return ca.vertcat(
            (10.0 - theta[0]) / 100.0,
            (theta[0] - 1000.0) / 100.0,
            0.01 - theta[1],
            theta[1] - 0.10,
            (25.0 * DEG_TO_RAD - theta[2]) / DEG_TO_RAD,
            (theta[2] - 60.0 * DEG_TO_RAD) / DEG_TO_RAD,
            (100.0 - theta[3]) / 100.0,
            (theta[3] - 800.0) / 100.0,
            -energy / 100000.0,
            (energy - 400000.0) / 100000.0,
        )

    c_dims = [0 for _ in range(num_steps + 2)]
    c_dims[terminal] = 1
    g_dims = [0 for _ in range(num_steps + 2)]
    g_dims[0] = 10

    return GraphProblemData(
        name="dymos/cannonball_implicit_duration",
        state_dims=[1] + [4 for _ in range(num_steps + 1)],
        edges=edges,
        theta_dim=4,
        c_dims=c_dims,
        g_dims=g_dims,
        X_init=[np.zeros(1)] + trajectory_init,
        U_init=[np.zeros(0) for _ in edges],
        theta_init=theta_init,
        max_iterations=1000,
        root_residual=root_residual,
        cost=cost,
        equalities=equalities,
        inequalities=inequalities,
        settings_override_cpp="""
  settings.regularization.initial = 1e-6;
""",
    )


if __name__ == "__main__":
    codegen_main(make_problem)
