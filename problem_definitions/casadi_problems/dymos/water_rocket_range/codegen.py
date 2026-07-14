import casadi as ca
import numpy as np

from problem_definitions.casadi_problems.codegen_common import (
    GraphEdge,
    GraphProblemData,
    main as codegen_main,
)
from problem_definitions.casadi_problems.dymos.atmosphere_1976 import (
    atmosphere_casadi,
    atmosphere_numpy,
)
from problem_definitions.casadi_problems.dymos.common import rk4_step


G = 9.80665
CD = 0.3450
CL = 0.0
S_REF = np.pi * 106e-3**2 / 4.0
A_OUT = np.pi * 22e-3**2 / 4.0
K_EXPANSION = 1.2
V_BOTTLE = 2.0e-3
RHO_WATER = 1000.0
DEG_TO_RAD = np.pi / 180.0

D_PROP = 0
D_BALLISTIC = 1
D_DESCENT = 2
INITIAL_GAM = 3
INITIAL_WATER = 4
M_EMPTY = 5

PROP_REF = np.array([100.0, 10.0, 1.0, 100.0, 10.0, 1.0])
BALLISTIC_REF = np.array([100.0, 100.0, 1.0, 100.0])


def _prop_ode_numpy(x, theta):
    r, h, gam, v, p_bar, water_l = x
    del r
    p_ambient, rho = atmosphere_numpy(h)
    pressure = p_bar * 1.0e5
    water_m3 = water_l * 1.0e-3
    exhaust_speed = np.sqrt(2.0 * (pressure - p_ambient) / RHO_WATER)
    water_dot_m3 = -exhaust_speed * A_OUT
    pressure_dot_bar = (
        pressure * K_EXPANSION * water_dot_m3 / (V_BOTTLE - water_m3) / 1.0e5
    )
    thrust = RHO_WATER * exhaust_speed**2 * A_OUT
    mass = theta[M_EMPTY] + RHO_WATER * water_m3
    q = 0.5 * rho * v**2
    drag = q * S_REF * CD
    lift = q * S_REF * CL
    return np.array(
        [
            v * np.cos(gam),
            v * np.sin(gam),
            lift / (mass * v) - (G / v) * np.cos(gam),
            (thrust - drag) / mass - G * np.sin(gam),
            pressure_dot_bar,
            water_dot_m3 * 1000.0,
        ]
    )


def _ballistic_ode_numpy(x, theta):
    r, h, gam, v = x
    del r
    _, rho = atmosphere_numpy(h)
    q = 0.5 * rho * v**2
    drag = q * S_REF * CD
    lift = q * S_REF * CL
    mass = theta[M_EMPTY]
    return np.array(
        [
            v * np.cos(gam),
            v * np.sin(gam),
            lift / (mass * v) - (G / v) * np.cos(gam),
            -drag / mass - G * np.sin(gam),
        ]
    )


def _numpy_rk4(ode, x, dt, theta):
    k1 = ode(x, theta)
    k2 = ode(x + 0.5 * dt * k1, theta)
    k3 = ode(x + 0.5 * dt * k2, theta)
    k4 = ode(x + dt * k3, theta)
    return x + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


def _rollout_phase(ode, x0, duration, segments, theta):
    X = [np.asarray(x0, dtype=float)]
    dt = duration / segments
    for _ in range(segments):
        X.append(_numpy_rk4(ode, X[-1], dt, theta))
    return X


def make_water_rocket_problem(name, objective, theta_init) -> GraphProblemData:
    if objective not in ("range", "height"):
        raise ValueError(f"Unknown water rocket objective: {objective}")

    prop_segments = 4
    ballistic_segments = 4
    descent_segments = 4

    theta_init = np.asarray(theta_init, dtype=float)

    state_dims = []
    x_init = []
    edges = []
    u_init = []
    node_kind = []
    edge_kind = []

    def add_node(value, kind):
        state_dims.append(len(value))
        x_init.append(np.asarray(value, dtype=float))
        node_kind.append(kind)
        return len(state_dims) - 1

    def add_edge(parent, child, dynamics, kind):
        edges.append(GraphEdge(parent, child, 0, np.zeros(0), dynamics))
        u_init.append(np.zeros(0))
        edge_kind.append(kind)

    def prop_ode(x, u, theta):
        del u
        p_ambient, rho = atmosphere_casadi(x[1])
        pressure = x[4] * 1.0e5
        water_m3 = x[5] * 1.0e-3
        exhaust_speed = ca.sqrt(2.0 * (pressure - p_ambient) / RHO_WATER)
        water_dot_m3 = -exhaust_speed * A_OUT
        pressure_dot_bar = (
            pressure * K_EXPANSION * water_dot_m3 / (V_BOTTLE - water_m3) / 1.0e5
        )
        thrust = RHO_WATER * exhaust_speed**2 * A_OUT
        mass = theta[M_EMPTY] + RHO_WATER * water_m3
        q = 0.5 * rho * x[3] ** 2
        drag = q * S_REF * CD
        lift = q * S_REF * CL
        return ca.vertcat(
            x[3] * ca.cos(x[2]),
            x[3] * ca.sin(x[2]),
            lift / (mass * x[3]) - (G / x[3]) * ca.cos(x[2]),
            (thrust - drag) / mass - G * ca.sin(x[2]),
            pressure_dot_bar,
            water_dot_m3 * 1000.0,
        )

    def ballistic_ode(x, u, theta):
        del u
        _, rho = atmosphere_casadi(x[1])
        q = 0.5 * rho * x[3] ** 2
        drag = q * S_REF * CD
        lift = q * S_REF * CL
        mass = theta[M_EMPTY]
        return ca.vertcat(
            x[3] * ca.cos(x[2]),
            x[3] * ca.sin(x[2]),
            lift / (mass * x[3]) - (G / x[3]) * ca.cos(x[2]),
            -drag / mass - G * ca.sin(x[2]),
        )

    def prop_step(x, u, theta, parameters):
        del parameters
        return rk4_step(prop_ode, x, u, theta, theta[D_PROP] / prop_segments)

    def ballistic_step(x, u, theta, parameters):
        del parameters
        return rk4_step(
            ballistic_ode, x, u, theta, theta[D_BALLISTIC] / ballistic_segments
        )

    def descent_step(x, u, theta, parameters):
        del parameters
        return rk4_step(ballistic_ode, x, u, theta, theta[D_DESCENT] / descent_segments)

    def initialize_propulsion(x, u, theta, parameters):
        del x, u, parameters
        return ca.vertcat(
            0.0,
            0.0,
            theta[INITIAL_GAM],
            0.1,
            6.5,
            theta[INITIAL_WATER],
        )

    def link_prop_to_ballistic(x, u, theta, parameters):
        del u, theta, parameters
        return x[:4]

    def link_ballistic_to_descent(x, u, theta, parameters):
        del u, theta, parameters
        return x

    root = add_node([0.0], "root")
    prop_start = add_node(
        [0.0, 0.0, theta_init[INITIAL_GAM], 0.1, 6.5, theta_init[INITIAL_WATER]], "prop"
    )
    add_edge(root, prop_start, initialize_propulsion, "init")

    prop_guess = _rollout_phase(
        _prop_ode_numpy,
        x_init[prop_start],
        theta_init[D_PROP],
        prop_segments,
        theta_init,
    )
    prop_nodes = [prop_start]
    for value in prop_guess[1:]:
        child = add_node(value, "prop")
        add_edge(prop_nodes[-1], child, prop_step, "prop")
        prop_nodes.append(child)

    ballistic_start = add_node(prop_guess[-1][:4], "ballistic")
    add_edge(prop_nodes[-1], ballistic_start, link_prop_to_ballistic, "link")

    ballistic_guess = _rollout_phase(
        _ballistic_ode_numpy,
        x_init[ballistic_start],
        theta_init[D_BALLISTIC],
        ballistic_segments,
        theta_init,
    )
    ballistic_nodes = [ballistic_start]
    for value in ballistic_guess[1:]:
        child = add_node(value, "ballistic")
        add_edge(ballistic_nodes[-1], child, ballistic_step, "ballistic")
        ballistic_nodes.append(child)

    descent_start = add_node(ballistic_guess[-1], "descent")
    add_edge(ballistic_nodes[-1], descent_start, link_ballistic_to_descent, "link")

    descent_guess = _rollout_phase(
        _ballistic_ode_numpy,
        x_init[descent_start],
        theta_init[D_DESCENT],
        descent_segments,
        theta_init,
    )
    descent_nodes = [descent_start]
    for value in descent_guess[1:]:
        child = add_node(value, "descent")
        add_edge(descent_nodes[-1], child, descent_step, "descent")
        descent_nodes.append(child)

    prop_terminal = prop_nodes[-1]
    ballistic_terminal = ballistic_nodes[-1]
    descent_terminal = descent_nodes[-1]

    c_dims = [0 for _ in state_dims]
    c_dims[prop_terminal] = 1
    c_dims[ballistic_terminal] = 1
    c_dims[descent_terminal] = 1

    def g_dim_for_node(node):
        if node == root:
            return 13
        if node_kind[node] == "prop":
            return 5
        if node_kind[node] == "ballistic":
            return 1
        return 0

    g_dims = [g_dim_for_node(node) for node in range(len(state_dims))]

    def root_residual(x, theta):
        del theta
        return ca.vertcat(-x[0])

    def cost(node, x, theta):
        del theta
        if objective == "range" and node == descent_terminal:
            return -x[0] / 100.0
        if objective == "height" and node == ballistic_terminal:
            return -x[1] / 100.0
        return ca.SX(0.0)

    def equalities(node, x, theta, outgoing_controls, outgoing_parameters):
        del theta, outgoing_controls, outgoing_parameters
        if node == prop_terminal:
            return ca.vertcat(x[5])
        if node == ballistic_terminal:
            return ca.vertcat(x[2])
        if node == descent_terminal:
            return ca.vertcat(x[1] / 100.0)
        return ca.SX.zeros(0, 1)

    def inequalities(node, x, theta, outgoing_controls, outgoing_parameters):
        del outgoing_controls, outgoing_parameters
        pieces = []
        if node == root:
            pieces.extend(
                [
                    (0.001 - theta[D_PROP]) / 0.1,
                    (theta[D_PROP] - 0.5) / 0.1,
                    0.001 - theta[D_BALLISTIC],
                    (theta[D_BALLISTIC] - 10.0) / 10.0,
                    (0.5 - (theta[D_PROP] + theta[D_BALLISTIC])) / 10.0,
                    ((theta[D_PROP] + theta[D_BALLISTIC]) - 100.0) / 10.0,
                    (0.5 - theta[D_DESCENT]) / 10.0,
                    (theta[D_DESCENT] - 100.0) / 10.0,
                    0.001 - theta[M_EMPTY],
                    theta[M_EMPTY] - 1.0,
                    -theta[INITIAL_GAM],
                    (theta[INITIAL_GAM] - 85.0 * DEG_TO_RAD) / DEG_TO_RAD,
                    -theta[INITIAL_WATER],
                ]
            )
        elif node_kind[node] == "prop":
            pieces.extend(
                [
                    -x[2],
                    (x[2] - 85.0 * DEG_TO_RAD) / DEG_TO_RAD,
                    (0.01 - x[3]) / 100.0,
                    (1.02 - x[4]) / 10.0,
                    -x[5],
                ]
            )
        elif node_kind[node] == "ballistic":
            pieces.extend(
                [
                    (x[2] - 89.0 * DEG_TO_RAD) / DEG_TO_RAD,
                ]
            )
        return ca.vertcat(*pieces) if pieces else ca.SX.zeros(0, 1)

    return GraphProblemData(
        name=name,
        state_dims=state_dims,
        edges=edges,
        theta_dim=len(theta_init),
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


def make_problem() -> GraphProblemData:
    return make_water_rocket_problem(
        "dymos/water_rocket_range",
        "range",
        np.array(
            [
                0.1140,
                1.9480,
                2.3600,
                46.0 * DEG_TO_RAD,
                1.026,
                0.189,
            ]
        ),
    )


if __name__ == "__main__":
    codegen_main(make_problem)
