import casadi as ca
import numpy as np
from scipy import interpolate

from problem_definitions.casadi_problems.codegen_common import (
    GraphEdge,
    GraphProblemData,
    main as codegen_main,
)


MASS = 800.0
GRAVITY = 9.8
A_CG_FRONT = 1.8
B_CG_REAR = 1.6
HALF_TRACK_WIDTH = 0.73
YAW_INERTIA = 450.0
AIR_DENSITY = 1.2
BRAKE_BIAS = 0.62
K_LAMBDA = 44.0
COP = 1.6
CG_HEIGHT = 0.3
ROLL_STIFFNESS = 0.5
CL_A = 4.0
CD_A = 2.0
TAU_X = 0.2
TAU_Y = 0.2
MU0_X = 1.68
MU0_Y = 1.68
K_MU = -0.0
POWER_LIMIT = 960000.0
POWER_REF = 100000.0
STATE_REFS = np.array([100.0, 4.0, 40.0, 0.15, 0.01, 0.3, 8.0, 8.0])
RK4_SUBSTEPS = 4

T_IDX = 0
N_IDX = 1
V_IDX = 2
ALPHA_IDX = 3
LAMBDA_IDX = 4
OMEGA_IDX = 5
AX_IDX = 6
AY_IDX = 7


OVAL_TRACK_SEGMENTS = np.array(
    [
        [0.0, 150.0, 0.0, -1.0],
        [1.0, np.pi / 2.0, 50.0, 0.0],
        [0.0, 100.0, 0.0, -1.0],
        [1.0, np.pi / 2.0, 90.0, 0.0],
        [0.0, 300.0, 0.0, -1.0],
        [1.0, np.pi / 2.0, 50.0, 0.0],
        [0.0, 100.0, 0.0, -1.0],
        [1.0, np.pi / 2.0, 90.0, 0.0],
        [0.0, 155.0, 0.0, -1.0],
    ]
)


def _track_length(segments):
    total = 0.0
    for kind, length, radius, _ in segments:
        total += length if kind == 0.0 else length * radius
    return total


def _track_points(segments):
    pos = np.array([0.0, 0.0])
    direction = np.array([1.0, 0.0])
    points = [[0.0, 0.0]]
    for kind, length, radius, side in segments:
        if radius == 0.0:
            endpoint = pos + direction * length
            for j in range(1, int(length) - 1):
                if j % 5 == 0:
                    points.append(pos + direction * j)
            pos = endpoint
            continue

        normal = (
            np.array([-direction[1], direction[0]])
            if side == 0.0
            else np.array([direction[1], -direction[0]])
        )
        center = pos + radius * normal
        theta_line = np.arctan2(direction[1], direction[0])
        theta_0 = np.arctan2(pos[1] - center[1], pos[0] - center[0])
        if side == 0.0:
            theta_end = theta_0 + length
            direction = np.array(
                [np.cos(theta_line + length), np.sin(theta_line + length)]
            )
        else:
            theta_end = theta_0 - length
            direction = np.array(
                [np.cos(theta_line - length), np.sin(theta_line - length)]
            )
        theta_vector = np.linspace(theta_0, theta_end, 100)
        xs = center[0] + radius * np.cos(theta_vector)
        ys = center[1] + radius * np.sin(theta_vector)
        for j in range(len(xs)):
            if j % 10 == 0:
                points.append([xs[j], ys[j]])
        pos = np.array([xs[-1], ys[-1]])
    return np.asarray(points)


def _curvature_table():
    tck, _ = interpolate.splprep(_track_points(OVAL_TRACK_SEGMENTS).T, s=0.0, k=5)
    unew = np.arange(0.0, 1.0, 0.0001)
    first = interpolate.splev(unew, tck, der=1)
    second = interpolate.splev(unew, tck, der=2)
    return (first[0] * second[1] - first[1] * second[0]) / (
        first[0] ** 2 + first[1] ** 2
    ) ** (3.0 / 2.0)


def _curvature_at(curv, track_length, s):
    index = np.floor((s / track_length) * len(curv))
    index = int(np.minimum(index, len(curv) - 1))
    return float(curv[index])


def _thrust_split(thrust):
    positive = ca.if_else(thrust > 0.0, 1.0, 0.0)
    brake = (positive - 1.0) * thrust
    throttle = positive * thrust
    return throttle, brake


def _model_quantities(x, u, kappa):
    t, n, V, alpha, lamb, omega, ax, ay = [x[i] for i in range(8)]
    del t
    delta = u[0]
    thrust = u[1]
    wheelbase = A_CG_FRONT + B_CG_REAR

    downforce = 0.5 * AIR_DENSITY * CL_A * V**2
    downforce_rear = downforce * (COP / wheelbase)
    downforce_front = downforce * (1.0 - COP / wheelbase)
    n_fl = (
        (MASS * GRAVITY / 2.0) * (B_CG_REAR / wheelbase)
        + (MASS / 4.0)
        * (-(ax * CG_HEIGHT) / wheelbase + ay * ROLL_STIFFNESS * CG_HEIGHT / HALF_TRACK_WIDTH)
        + downforce_front / 2.0
    )
    n_fr = (
        (MASS * GRAVITY / 2.0) * (B_CG_REAR / wheelbase)
        + (MASS / 4.0)
        * (-(ax * CG_HEIGHT) / wheelbase - ay * ROLL_STIFFNESS * CG_HEIGHT / HALF_TRACK_WIDTH)
        + downforce_front / 2.0
    )
    n_rl = (
        (MASS * GRAVITY / 2.0) * (A_CG_FRONT / wheelbase)
        + (MASS / 4.0)
        * ((ax * CG_HEIGHT) / wheelbase + ay * (1.0 - ROLL_STIFFNESS) * CG_HEIGHT / HALF_TRACK_WIDTH)
        + downforce_rear / 2.0
    )
    n_rr = (
        (MASS * GRAVITY / 2.0) * (A_CG_FRONT / wheelbase)
        + (MASS / 4.0)
        * ((ax * CG_HEIGHT) / wheelbase - ay * (1.0 - ROLL_STIFFNESS) * CG_HEIGHT / HALF_TRACK_WIDTH)
        + downforce_rear / 2.0
    )

    throttle, brake = _thrust_split(thrust)
    s_fl = -(MASS * GRAVITY / 2.0) * brake * BRAKE_BIAS
    s_fr = -(MASS * GRAVITY / 2.0) * brake * BRAKE_BIAS
    s_rl = (MASS * GRAVITY / 2.0) * (throttle - brake * (1.0 - BRAKE_BIAS))
    s_rr = (MASS * GRAVITY / 2.0) * (throttle - brake * (1.0 - BRAKE_BIAS))

    f_rr = n_rr * K_LAMBDA * (lamb + omega * (B_CG_REAR + lamb * HALF_TRACK_WIDTH) / V)
    f_rl = n_rl * K_LAMBDA * (lamb + omega * (B_CG_REAR - lamb * HALF_TRACK_WIDTH) / V)
    f_fr = n_fr * K_LAMBDA * (
        lamb + delta - omega * (A_CG_FRONT - lamb * HALF_TRACK_WIDTH) / V
    )
    f_fl = n_fl * K_LAMBDA * (
        lamb + delta - omega * (A_CG_FRONT + lamb * HALF_TRACK_WIDTH) / V
    )

    f_all = f_fl + f_fr + f_rl + f_rr
    s_all = s_fl + s_fr + s_rl + s_rr
    f_front = f_fl + f_fr
    s_front = s_fl + s_fr
    drag = 0.5 * AIR_DENSITY * CD_A * V**2
    sin_alpha_minus_lamb = ca.sin(alpha - lamb)
    cos_alpha_minus_lamb = ca.cos(alpha - lamb)
    sdot = V * cos_alpha_minus_lamb / (1.0 - n * kappa)
    ndot = V * sin_alpha_minus_lamb
    alphadot = omega - kappa * V * cos_alpha_minus_lamb / (1.0 - n * kappa)
    vdot = s_all / MASS - delta * f_front / MASS - drag / MASS - omega * V * lamb
    lambdadot = (
        omega
        - vdot * lamb / V
        - delta * s_front / (MASS * V)
        - f_all / (MASS * V)
    )
    omegadot = (
        A_CG_FRONT * (f_fr + f_fl) / YAW_INERTIA
        - B_CG_REAR * (f_rr + f_rl) / YAW_INERTIA
        + HALF_TRACK_WIDTH * (-s_rr + s_rl - s_fr + s_fl) / YAW_INERTIA
    )
    axdot = (vdot + omega * V * lamb - ax) / TAU_X
    aydot = (omega * V - (V * lambdadot + vdot * lamb) - ay) / TAU_Y
    power = V * s_all

    def tire_constraint(S, F, N, N0):
        mu_x = MU0_X + K_MU * N / N0
        mu_y = MU0_Y + K_MU * N / N0
        return (S / (N * mu_x)) ** 2 + (F / (N * mu_y)) ** 2

    n0_rr = (MASS * GRAVITY / 2.0) * (A_CG_FRONT / wheelbase)
    n0_rl = n0_rr
    n0_fr = (MASS * GRAVITY / 2.0) * (B_CG_REAR / wheelbase)
    n0_fl = n0_fr
    tire_constraints = ca.vertcat(
        tire_constraint(s_rr, f_rr, n_rr, n0_rr),
        tire_constraint(s_rl, f_rl, n_rl, n0_rl),
        tire_constraint(s_fr, f_fr, n_fr, n0_fr),
        tire_constraint(s_fl, f_fl, n_fl, n0_fl),
    )

    dx_ds = ca.vertcat(
        1.0 / sdot,
        ndot / sdot,
        vdot / sdot,
        alphadot / sdot,
        lambdadot / sdot,
        omegadot / sdot,
        axdot / sdot,
        aydot / sdot,
    )
    return dx_ds, power, tire_constraints


def _numpy_model(x, u, kappa):
    x_ca = ca.DM(x)
    u_ca = ca.DM(u)
    dx, _, _ = _model_quantities(x_ca, u_ca, kappa)
    return np.array(dx).reshape(-1)


def _numpy_rk4(x, u, kappas, ds):
    k1 = _numpy_model(x, u, kappas[0])
    k2 = _numpy_model(x + 0.5 * ds * k1, u, kappas[1])
    k3 = _numpy_model(x + 0.5 * ds * k2, u, kappas[1])
    k4 = _numpy_model(x + ds * k3, u, kappas[2])
    return x + (ds / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


def make_problem() -> GraphProblemData:
    segments = 20
    track_length = _track_length(OVAL_TRACK_SEGMENTS)
    ds = track_length / segments
    curv = _curvature_table()

    theta_init = np.array([0.0, 20.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    initial_state = np.array([0.0, *theta_init])
    control_guess = np.array([0.0, 0.1])

    state_dims = [0]
    x_init = [np.zeros(0)]
    edges = []
    u_init = []

    def add_node(value):
        state_dims.append(8)
        x_init.append(np.asarray(value, dtype=float))
        return len(state_dims) - 1

    first = add_node(initial_state)

    def initial_dynamics(x, u, theta):
        del x, u
        return ca.vertcat(0.0, theta)

    edges.append(GraphEdge(0, first, 0, initial_dynamics))
    u_init.append(np.zeros(0))

    nodes = [first]
    for _ in range(segments):
        nodes.append(add_node(x_init[-1]))

    for edge_index in range(segments):
        parent = nodes[edge_index]
        child = nodes[edge_index + 1]
        s0 = edge_index * ds
        sub_ds = ds / RK4_SUBSTEPS
        kappa_steps = []
        for substep in range(RK4_SUBSTEPS):
            sub_s0 = s0 + substep * sub_ds
            kappa_steps.append(
                [
                    _curvature_at(curv, track_length, sub_s0),
                    _curvature_at(curv, track_length, sub_s0 + 0.5 * sub_ds),
                    _curvature_at(curv, track_length, sub_s0 + sub_ds),
                ]
            )

        def make_step(kappa_values):
            def step(x, u, theta):
                del theta
                x_next = x
                for kappa_step in kappa_values:
                    k1, _, _ = _model_quantities(x_next, u, kappa_step[0])
                    k2, _, _ = _model_quantities(
                        x_next + 0.5 * sub_ds * k1, u, kappa_step[1]
                    )
                    k3, _, _ = _model_quantities(
                        x_next + 0.5 * sub_ds * k2, u, kappa_step[1]
                    )
                    k4, _, _ = _model_quantities(x_next + sub_ds * k3, u, kappa_step[2])
                    x_next = x_next + (sub_ds / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
                return x_next

            return step

        edges.append(GraphEdge(parent, child, 2, make_step(kappa_steps)))
        u_init.append(control_guess.copy())

    for i, node in enumerate(nodes):
        x_init[node] = np.array(
            [
                100.0 * i / segments,
                0.0,
                20.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
            ]
        )

    c_dims = [0 for _ in state_dims]
    g_dims = [0 for _ in state_dims]
    c_dims[nodes[-1]] = 7
    for node in nodes[:-1]:
        g_dims[node] = 8
    g_dims[nodes[-1]] = 3

    edge_by_parent = {edge.parent: i for i, edge in enumerate(edges)}
    final_node = nodes[-1]

    def root_residual(x, theta):
        del x, theta
        return ca.SX.zeros(0, 1)

    def cost(node, x, theta):
        del theta
        return x[T_IDX] if node == final_node else ca.SX(0.0)

    def equalities(node, x, theta, outgoing_controls):
        del outgoing_controls
        if node == final_node:
            return (x[1:] - theta) / STATE_REFS[1:]
        return ca.SX.zeros(0, 1)

    def inequalities(node, x, theta, outgoing_controls):
        del theta
        pieces = [
            -x[T_IDX] / 100.0,
            (-4.0 - x[N_IDX]) / 4.0,
            (x[N_IDX] - 4.0) / 4.0,
        ]
        if node != final_node:
            edge_index = edge_by_parent[node]
            s0 = (edge_index - 1) * ds
            kappa = _curvature_at(curv, track_length, s0)
            _, power, tire_constraints = _model_quantities(x, outgoing_controls[0], kappa)
            pieces.append((power - POWER_LIMIT) / POWER_REF)
            for i in range(4):
                pieces.append(tire_constraints[i] - 1.0)
        return ca.vertcat(*pieces)

    return GraphProblemData(
        name="dymos/racecar",
        state_dims=state_dims,
        edges=edges,
        theta_dim=7,
        c_dims=c_dims,
        g_dims=g_dims,
        X_init=x_init,
        U_init=u_init,
        theta_init=theta_init,
        max_iterations=1000,
        dyn_refs=[np.ones(dim) if dim == 0 else STATE_REFS.copy() for dim in state_dims],
        root_residual=root_residual,
        cost=cost,
        equalities=equalities,
        inequalities=inequalities,
        settings_override_cpp="""
  settings.line_search.skip_line_search = false;
  settings.line_search.start_ls_with_alpha_s_max = true;
  settings.regularization.maximum = 1e12;
""",
    )


if __name__ == "__main__":
    codegen_main(make_problem)
