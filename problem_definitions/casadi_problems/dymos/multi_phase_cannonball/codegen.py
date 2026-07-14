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
CD = 0.5

DURATION_ASCENT = 0
DURATION_DESCENT = 1
RADIUS = 2
INITIAL_GAM = 3
INITIAL_V = 4

R_REF = 1000.0
H_REF = 1000.0
GAM_REF = 1.0
V_REF = 100.0
ENERGY_REF = 100000.0


def _mass(radius):
    return (4.0 / 3.0) * np.pi * DENSITY * radius**3


def _area(radius):
    return np.pi * radius**2


def _numpy_ode(x, theta):
    radius = theta[RADIUS]
    r, h, gam, v = x
    del r
    rho = linear_density_numpy(h)
    drag = 0.5 * rho * v**2 * _area(radius) * CD
    return np.array(
        [
            v * np.cos(gam),
            v * np.sin(gam),
            -(GRAVITY / v) * np.cos(gam),
            -drag / _mass(radius) - GRAVITY * np.sin(gam),
        ]
    )


def _numpy_rk4(x, dt, theta):
    k1 = _numpy_ode(x, theta)
    k2 = _numpy_ode(x + 0.5 * dt * k1, theta)
    k3 = _numpy_ode(x + 0.5 * dt * k2, theta)
    k4 = _numpy_ode(x + dt * k3, theta)
    return x + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


def make_problem() -> GraphProblemData:
    segments = 10
    state_dims = []
    x_init = []
    edges = []
    u_init = []
    node_phase = []

    theta_init = np.array(
        [
            10.6480458,
            16.1342959,
            0.0420296584,
            0.558734304,
            571.714824,
        ]
    )

    def add_node(value, phase):
        state_dims.append(len(value))
        x_init.append(np.asarray(value, dtype=float))
        node_phase.append(phase)
        return len(state_dims) - 1

    def add_edge(parent, child, dynamics):
        edges.append(GraphEdge(parent, child, 0, np.zeros(0), dynamics))
        u_init.append(np.zeros(0))

    def ode(x, u, theta):
        del u
        radius = theta[RADIUS]
        rho = linear_density_casadi(x[1])
        drag = 0.5 * rho * x[3] ** 2 * _area(radius) * CD
        return ca.vertcat(
            x[3] * ca.cos(x[2]),
            x[3] * ca.sin(x[2]),
            -(GRAVITY / x[3]) * ca.cos(x[2]),
            -drag / _mass(radius) - GRAVITY * ca.sin(x[2]),
        )

    def step(duration_index):
        def dyn(x, u, theta, parameters):
            del parameters
            return rk4_step(ode, x, u, theta, theta[duration_index] / segments)

        return dyn

    def launch_initial_state(x, u, theta, parameters):
        del x, u, parameters
        return ca.vertcat(0.0, 0.0, theta[INITIAL_GAM], theta[INITIAL_V])

    root = add_node([0.0], "root")
    ascent_initial = add_node(
        [0.0, 0.0, theta_init[INITIAL_GAM], theta_init[INITIAL_V]], "ascent"
    )
    add_edge(root, ascent_initial, launch_initial_state)
    nodes = [ascent_initial]
    for _ in range(segments):
        parent = nodes[-1]
        child = add_node(x_init[parent], "ascent")
        add_edge(parent, child, step(DURATION_ASCENT))
        nodes.append(child)
    ascent_terminal = nodes[-1]
    for _ in range(segments):
        parent = nodes[-1]
        child = add_node(x_init[parent], "descent")
        add_edge(parent, child, step(DURATION_DESCENT))
        nodes.append(child)
    descent_terminal = nodes[-1]

    x_init[ascent_initial] = np.array(
        [0.0, 0.0, theta_init[INITIAL_GAM], theta_init[INITIAL_V]]
    )
    for i in range(segments):
        x_init[nodes[i + 1]] = _numpy_rk4(
            x_init[nodes[i]], theta_init[DURATION_ASCENT] / segments, theta_init
        )
    for i in range(segments):
        x_init[nodes[segments + i + 1]] = _numpy_rk4(
            x_init[nodes[segments + i]],
            theta_init[DURATION_DESCENT] / segments,
            theta_init,
        )

    c_dims = [0 for _ in state_dims]
    c_dims[ascent_terminal] = 1
    c_dims[descent_terminal] = 1

    g_dims = [0 for _ in state_dims]
    g_dims[root] = 8

    def root_residual(x, theta):
        del theta
        return ca.vertcat(-x[0])

    def cost(node, x, theta):
        del theta
        if node == descent_terminal:
            return -x[0] / R_REF
        return ca.SX(0.0)

    def equalities(node, x, theta, outgoing_controls, outgoing_parameters):
        del theta, outgoing_controls, outgoing_parameters
        if node == ascent_terminal:
            return ca.vertcat(x[2] / GAM_REF)
        if node == descent_terminal:
            return ca.vertcat(x[1] / H_REF)
        return ca.SX.zeros(0, 1)

    def inequalities(node, x, theta, outgoing_controls, outgoing_parameters):
        del x, outgoing_controls, outgoing_parameters
        if node != root:
            return ca.SX.zeros(0, 1)
        mass = _mass(theta[RADIUS])
        energy = 0.5 * mass * theta[INITIAL_V] ** 2
        return ca.vertcat(
            (1.0 - theta[DURATION_ASCENT]) / 100.0,
            (theta[DURATION_ASCENT] - 100.0) / 100.0,
            (0.5 - theta[DURATION_DESCENT]) / 100.0,
            (theta[DURATION_DESCENT] - 100.0) / 100.0,
            0.01 - theta[RADIUS],
            theta[RADIUS] - 0.10,
            -energy / ENERGY_REF,
            (energy - 400000.0) / ENERGY_REF,
        )

    return GraphProblemData(
        name="dymos/multi_phase_cannonball",
        state_dims=state_dims,
        edges=edges,
        theta_dim=5,
        c_dims=c_dims,
        g_dims=g_dims,
        X_init=x_init,
        U_init=u_init,
        theta_init=theta_init,
        max_iterations=1000,
        root_residual=root_residual,
        cost=cost,
        equalities=equalities,
        inequalities=inequalities,
        settings_override_cpp="""
  settings.penalty.initial_penalty_parameter = 10.0;
  settings.penalty.penalty_parameter_increase_factor = 1.5;
  settings.barrier.initial_mu = 1e-3;
  settings.line_search.skip_line_search = false;
""",
    )


if __name__ == "__main__":
    codegen_main(make_problem)
