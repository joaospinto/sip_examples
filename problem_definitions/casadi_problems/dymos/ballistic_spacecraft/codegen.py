import casadi as ca
import numpy as np

from problem_definitions.casadi_problems.codegen_common import (
    GraphEdge,
    GraphProblemData,
    main as codegen_main,
)
from problem_definitions.casadi_problems.dymos.common import rk4_step


MU_SUN = 1.32712440041279419e11
KMPAU = 149597870.700
DAY = 86400.0
TU_SUN = 2.0 * np.pi * np.sqrt(KMPAU**3 / MU_SUN)

EARTH_ELEMENTS = np.array(
    [
        1.4959826115044251e8,
        1.6711230000000001e-2,
        -2.6720990848033185e-7,
        1.7966014740491711,
        0.0,
        -4.3163916976385941e-2,
    ]
)
MARS_ELEMENTS = np.array(
    [
        2.2794382242757303e8,
        9.3394099999999994e-2,
        3.2283205424889293e-2,
        -1.2826977680479821,
        8.6497712974974172e-1,
        3.3806682838789720e-1,
    ]
)

T_INITIAL = 0
DURATION = 1
V0_START = 2


def _solve_kepler_numpy(mean_anomaly, eccentricity):
    eccentric_anomaly = mean_anomaly - eccentricity * np.sin(mean_anomaly)
    for _ in range(15):
        residual = (
            eccentric_anomaly
            - eccentricity * np.sin(eccentric_anomaly)
            - mean_anomaly
        )
        jacobian = 1.0 - eccentricity * np.cos(eccentric_anomaly)
        eccentric_anomaly -= residual / jacobian
    return eccentric_anomaly


def _solve_kepler_ca(mean_anomaly, eccentricity):
    eccentric_anomaly = mean_anomaly - eccentricity * ca.sin(mean_anomaly)
    for _ in range(15):
        residual = (
            eccentric_anomaly
            - eccentricity * ca.sin(eccentric_anomaly)
            - mean_anomaly
        )
        jacobian = 1.0 - eccentricity * ca.cos(eccentric_anomaly)
        eccentric_anomaly -= residual / jacobian
    return eccentric_anomaly


def _ephemeris_numpy(elements, t):
    a, e, inc, raan, argp, mean_anomaly_0 = elements
    mean_motion = np.sqrt(MU_SUN / a**3)
    mean_anomaly = mean_anomaly_0 + mean_motion * t
    eccentric_anomaly = _solve_kepler_numpy(mean_anomaly, e)
    true_anomaly = 2.0 * np.arctan2(
        np.sqrt(1.0 + e) * np.sin(eccentric_anomaly / 2.0),
        np.sqrt(1.0 - e) * np.cos(eccentric_anomaly / 2.0),
    )
    radius = a * (1.0 - e**2) / (1.0 + e * np.cos(true_anomaly))
    speed = np.sqrt(2.0 * MU_SUN / radius - MU_SUN / a)
    gamma = np.arctan2(e * np.sin(true_anomaly), 1.0 + e * np.cos(true_anomaly))

    cos_theta_argp = np.cos(true_anomaly + argp)
    sin_theta_argp = np.sin(true_anomaly + argp)
    cos_raan = np.cos(raan)
    sin_raan = np.sin(raan)
    cos_inc = np.cos(inc)
    sin_inc = np.sin(inc)

    r = np.array(
        [
            radius * (cos_theta_argp * cos_raan - sin_theta_argp * cos_inc * sin_raan),
            radius * (cos_theta_argp * sin_raan + sin_theta_argp * cos_inc * cos_raan),
            radius * sin_theta_argp * sin_inc,
        ]
    )

    cos_theta_omega_gamma = np.cos(true_anomaly + argp - gamma)
    sin_theta_omega_gamma = np.sin(true_anomaly + argp - gamma)
    v = np.array(
        [
            speed * (-sin_theta_omega_gamma * cos_raan - cos_theta_omega_gamma * cos_inc * sin_raan),
            speed * (-sin_theta_omega_gamma * sin_raan + cos_theta_omega_gamma * cos_inc * cos_raan),
            speed * cos_theta_omega_gamma * sin_inc,
        ]
    )
    return r, v


def _ephemeris_ca(elements, t):
    a, e, inc, raan, argp, mean_anomaly_0 = [float(v) for v in elements]
    mean_motion = np.sqrt(MU_SUN / a**3)
    mean_anomaly = mean_anomaly_0 + mean_motion * t
    eccentric_anomaly = _solve_kepler_ca(mean_anomaly, e)
    true_anomaly = 2.0 * ca.atan2(
        np.sqrt(1.0 + e) * ca.sin(eccentric_anomaly / 2.0),
        np.sqrt(1.0 - e) * ca.cos(eccentric_anomaly / 2.0),
    )
    radius = a * (1.0 - e**2) / (1.0 + e * ca.cos(true_anomaly))
    speed = ca.sqrt(2.0 * MU_SUN / radius - MU_SUN / a)
    gamma = ca.atan2(e * ca.sin(true_anomaly), 1.0 + e * ca.cos(true_anomaly))

    cos_theta_argp = ca.cos(true_anomaly + argp)
    sin_theta_argp = ca.sin(true_anomaly + argp)
    cos_raan = np.cos(raan)
    sin_raan = np.sin(raan)
    cos_inc = np.cos(inc)
    sin_inc = np.sin(inc)

    r = ca.vertcat(
        radius * (cos_theta_argp * cos_raan - sin_theta_argp * cos_inc * sin_raan),
        radius * (cos_theta_argp * sin_raan + sin_theta_argp * cos_inc * cos_raan),
        radius * sin_theta_argp * sin_inc,
    )

    cos_theta_omega_gamma = ca.cos(true_anomaly + argp - gamma)
    sin_theta_omega_gamma = ca.sin(true_anomaly + argp - gamma)
    v = ca.vertcat(
        speed * (-sin_theta_omega_gamma * cos_raan - cos_theta_omega_gamma * cos_inc * sin_raan),
        speed * (-sin_theta_omega_gamma * sin_raan + cos_theta_omega_gamma * cos_inc * cos_raan),
        speed * cos_theta_omega_gamma * sin_inc,
    )
    return r, v


def _numpy_ode(x):
    r = x[:3]
    v = x[3:]
    r_mag = np.linalg.norm(r)
    return np.concatenate([v, -MU_SUN * r / r_mag**3])


def _numpy_rk4(x, dt):
    k1 = _numpy_ode(x)
    k2 = _numpy_ode(x + 0.5 * dt * k1)
    k3 = _numpy_ode(x + 0.5 * dt * k2)
    k4 = _numpy_ode(x + dt * k3)
    return x + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


def _numpy_rollout(T, theta):
    earth_r, _ = _ephemeris_numpy(EARTH_ELEMENTS, theta[T_INITIAL] * DAY)
    X = [np.concatenate([earth_r, theta[V0_START : V0_START + 3]])]
    dt = theta[DURATION] * DAY / T
    for _ in range(T):
        X.append(_numpy_rk4(X[-1], dt))
    return X


def make_problem() -> GraphProblemData:
    segments = 10
    theta_init = np.array(
        [
            -360.322721,
            243.669506,
            -31.5171666,
            -8.65413717,
            -0.0187973520,
        ]
    )

    state_dims = []
    x_init = []
    edges = []
    u_init = []

    def add_node(value):
        state_dims.append(len(value))
        x_init.append(np.asarray(value, dtype=float))
        return len(state_dims) - 1

    def add_edge(parent, child, dynamics):
        edges.append(GraphEdge(parent, child, 0, dynamics))
        u_init.append(np.zeros(0))

    def ode(x, u, theta):
        del u, theta
        r = x[:3]
        v = x[3:]
        r_mag = ca.sqrt(ca.dot(r, r))
        return ca.vertcat(v, -MU_SUN * r / r_mag**3)

    def step(x, u, theta):
        return rk4_step(ode, x, u, theta, theta[DURATION] * DAY / segments)

    def initialize_state(x, u, theta):
        del x, u
        earth_r, _ = _ephemeris_ca(EARTH_ELEMENTS, theta[T_INITIAL] * DAY)
        return ca.vertcat(earth_r, theta[V0_START : V0_START + 3])

    root = add_node([0.0])
    rollout = _numpy_rollout(segments, theta_init)
    first = add_node(rollout[0])
    add_edge(root, first, initialize_state)
    nodes = [first]
    for value in rollout[1:]:
        child = add_node(value)
        add_edge(nodes[-1], child, step)
        nodes.append(child)

    terminal = nodes[-1]

    c_dims = [0 for _ in state_dims]
    c_dims[terminal] = 3
    g_dims = [0 for _ in state_dims]
    g_dims[root] = 1

    def root_residual(x, theta):
        del theta
        return ca.vertcat(-x[0])

    def cost(node, x, theta):
        del x
        if node != root:
            return ca.SX(0.0)
        _, earth_v = _ephemeris_ca(EARTH_ELEMENTS, theta[T_INITIAL] * DAY)
        v_inf = theta[V0_START : V0_START + 3] - earth_v
        return ca.dot(v_inf, v_inf) / 100.0

    def equalities(node, x, theta, outgoing_controls):
        del outgoing_controls
        if node == terminal:
            mars_r, _ = _ephemeris_ca(
                MARS_ELEMENTS, (theta[T_INITIAL] + theta[DURATION]) * DAY
            )
            return (x[:3] - mars_r) / 1.0e8
        return ca.SX.zeros(0, 1)

    def inequalities(node, x, theta, outgoing_controls):
        del x, outgoing_controls
        if node == root:
            return ca.vertcat((0.01 * TU_SUN / DAY - theta[DURATION]) / 100.0)
        return ca.SX.zeros(0, 1)

    return GraphProblemData(
        name="dymos/ballistic_spacecraft",
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


if __name__ == "__main__":
    codegen_main(make_problem)
