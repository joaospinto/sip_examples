import casadi as ca
import numpy as np

from problem_definitions.casadi_problems.codegen_common import (
    GraphEdge,
    GraphProblemData,
    main as codegen_main,
)
from problem_definitions.casadi_problems.dymos.common import control_bounds, rk4_step


FT_TO_M = 0.3048
KNOT_TO_MPS = 0.5144444444444445
LBM_TO_KG = 0.45359237
LBF_TO_N = 4.4482216152605
DEG_TO_RAD = np.pi / 180.0

DURATION_BR = 0
DURATION_RTO = 1
DURATION_V1 = 2
DURATION_ROTATE = 3
DURATION_CLIMB = 4
THETA_FIELD_LENGTH = 5
THETA_ALPHA_LINK = 6

R_REF = 1000.0
V_REF = 100.0
H_REF = 1.0
GAM_REF = 0.05
ALPHA_REF = 10.0 * DEG_TO_RAD
FORCE_REF = 100000.0


def make_problem() -> GraphProblemData:
    segments = 6

    rho = 1.225
    s_ref = 124.7
    cd0 = 0.03
    cl0 = 0.5
    cl_max = 2.0
    alpha_max = np.radians(10.0)
    h_w = 1.0
    aspect_ratio = 9.45
    oswald_e = 0.801
    span = 35.7
    gravity = 9.80665
    mass = 174200.0 * LBM_TO_KG
    t_nominal = 27000.0 * 2.0 * LBF_TO_N
    t_engine_out = 27000.0 * LBF_TO_N
    t_shutdown = 0.0
    mu_nominal = 0.03
    mu_braking = 0.3
    h_runway = 0.0

    state_dims = []
    x_init = []
    edges = []
    u_init = []
    node_phase = []
    edge_phase = []
    phase_nodes = {}

    def add_node(dim, value, phase):
        state_dims.append(dim)
        x_init.append(np.asarray(value, dtype=float))
        node_phase.append(phase)
        return len(state_dims) - 1

    def add_edge(parent, child, control_dim, dynamics, phase, control_guess=None):
        edges.append(GraphEdge(parent, child, control_dim, np.zeros(0), dynamics))
        edge_phase.append(phase)
        if control_guess is None:
            control_guess = np.zeros(control_dim)
        u_init.append(np.asarray(control_guess, dtype=float))
        return len(edges) - 1

    def aero_runway(x, alpha, thrust, mu_r):
        v = x[1]
        h = h_runway
        weight = mass * gravity
        cl = cl0 + (alpha / alpha_max) * (cl_max - cl0)
        k_nom = 1.0 / (np.pi * aspect_ratio * oswald_e)
        b = span / 2.0
        height_ratio = (h + h_w) / b
        fact = height_ratio * ca.sqrt(height_ratio)
        k = k_nom * 33.0 * fact / (1.0 + 33.0 * fact)
        q = 0.5 * rho * v**2
        lift = q * s_ref * cl
        drag = q * s_ref * (cd0 + k * cl**2)
        c_alpha = ca.cos(alpha)
        s_alpha = ca.sin(alpha)
        f_r = mass * gravity - lift * c_alpha - thrust * s_alpha
        v_dot = (thrust * c_alpha - drag - f_r * mu_r) / mass
        return {
            "dynamics": ca.vertcat(v, v_dot),
            "F_r": f_r,
            "v_over_v_stall": v / ca.sqrt(2.0 * weight / rho / s_ref / cl_max),
        }

    def aero_climb(x, alpha, thrust):
        r, h, v, gam = x[0], x[1], x[2], x[3]
        del r
        weight = mass * gravity
        cl = cl0 + (alpha / alpha_max) * (cl_max - cl0)
        k_nom = 1.0 / (np.pi * aspect_ratio * oswald_e)
        b = span / 2.0
        height_ratio = (h + h_w) / b
        fact = height_ratio * ca.sqrt(height_ratio)
        k = k_nom * 33.0 * fact / (1.0 + 33.0 * fact)
        q = 0.5 * rho * v**2
        lift = q * s_ref * cl
        drag = q * s_ref * (cd0 + k * cl**2)
        c_alpha = ca.cos(alpha)
        s_alpha = ca.sin(alpha)
        c_gam = ca.cos(gam)
        s_gam = ca.sin(gam)
        v_dot = (thrust * c_alpha - drag) / mass - gravity * s_gam
        gam_dot = (thrust * s_alpha + lift) / (mass * v) - (gravity / v) * c_gam
        return {
            "dynamics": ca.vertcat(v * c_gam, v * s_gam, v_dot, gam_dot),
            "v_over_v_stall": v / ca.sqrt(2.0 * weight / rho / s_ref / cl_max),
        }

    def phase_dt(theta, duration_index):
        return theta[duration_index] / segments

    def runway_step(duration_index, thrust, mu_r, alpha_value=None):
        def dyn(x, u, theta, parameters):
            del parameters
            alpha = alpha_value if alpha_value is not None else u[0]

            def ode(x_ode, u_ode, theta_ode):
                del u_ode, theta_ode
                return aero_runway(x_ode, alpha, thrust, mu_r)["dynamics"]

            return rk4_step(
                ode, x, ca.SX.zeros(0, 1), theta, phase_dt(theta, duration_index)
            )

        return dyn

    def rotate_step(edge_in_phase):
        def dyn(x, u, theta, parameters):
            del u, parameters
            h = phase_dt(theta, DURATION_ROTATE)

            def ode_at(alpha):
                def ode(x_ode, u_ode, theta_ode):
                    del u_ode, theta_ode
                    return aero_runway(x_ode, alpha, t_engine_out, mu_nominal)[
                        "dynamics"
                    ]

                return ode

            alpha0 = theta[THETA_ALPHA_LINK] * edge_in_phase / segments
            alpha_mid = theta[THETA_ALPHA_LINK] * (edge_in_phase + 0.5) / segments
            alpha1 = theta[THETA_ALPHA_LINK] * (edge_in_phase + 1.0) / segments
            k1 = ode_at(alpha0)(x, ca.SX.zeros(0, 1), theta)
            k2 = ode_at(alpha_mid)(x + 0.5 * h * k1, ca.SX.zeros(0, 1), theta)
            k3 = ode_at(alpha_mid)(x + 0.5 * h * k2, ca.SX.zeros(0, 1), theta)
            k4 = ode_at(alpha1)(x + h * k3, ca.SX.zeros(0, 1), theta)
            return x + (h / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

        return dyn

    def climb_step(x, u, theta, parameters):
        del parameters

        def ode(x_ode, u_ode, theta_ode):
            del theta_ode
            return aero_climb(x_ode, u_ode[0], t_engine_out)["dynamics"]

        return rk4_step(ode, x, u, theta, phase_dt(theta, DURATION_CLIMB))

    def link_rv(x, u, theta, parameters):
        del u, theta, parameters
        return x

    def link_rotate_to_climb(x, u, theta, parameters):
        del u, theta, parameters
        return ca.vertcat(x[0], 0.0, x[1], 0.0)

    def add_phase(
        name,
        start_node,
        final_value,
        duration_index,
        dynamics,
        control_dim=0,
        control_guess=None,
    ):
        nodes = [start_node]
        start = x_init[start_node]
        guesses = np.linspace(start, np.asarray(final_value, dtype=float), segments + 1)
        for value in guesses[1:]:
            nodes.append(add_node(guesses.shape[1], value, name))
        for i in range(segments):
            add_edge(nodes[i], nodes[i + 1], control_dim, dynamics, name, control_guess)
        phase_nodes[name] = nodes
        return nodes[-1]

    br_initial = add_node(2, [0.0, 1.0e-4], "br_to_v1")
    br_final = add_phase(
        "br_to_v1",
        br_initial,
        [1200.0, 70.0],
        DURATION_BR,
        runway_step(DURATION_BR, t_nominal, mu_nominal, 0.0),
    )
    add_phase(
        "rto",
        br_final,
        [2197.0, 1.0e-4],
        DURATION_RTO,
        runway_step(DURATION_RTO, t_shutdown, mu_braking, 0.0),
    )
    v1_final = add_phase(
        "v1_to_vr",
        br_final,
        [1600.0, 80.0],
        DURATION_V1,
        runway_step(DURATION_V1, t_engine_out, mu_nominal, 0.0),
    )

    rotate_start = add_node(2, x_init[v1_final], "rotate")
    add_edge(v1_final, rotate_start, 0, link_rv, "link_v1_rotate")
    rotate_nodes = [rotate_start]
    rotate_guesses = np.linspace(
        x_init[rotate_start], np.array([1800.0, 85.0]), segments + 1
    )
    for value in rotate_guesses[1:]:
        rotate_nodes.append(add_node(2, value, "rotate"))
    for i in range(segments):
        add_edge(
            rotate_nodes[i],
            rotate_nodes[i + 1],
            0,
            rotate_step(i),
            "rotate",
        )
    phase_nodes["rotate"] = rotate_nodes
    rotate_final = rotate_nodes[-1]

    climb_start = add_node(
        4, [x_init[rotate_final][0], 0.0, x_init[rotate_final][1], 0.0], "climb"
    )
    add_edge(rotate_final, climb_start, 0, link_rotate_to_climb, "link_rotate_climb")
    add_phase(
        "climb",
        climb_start,
        [2197.0, 35.0 * FT_TO_M, 170.0 * KNOT_TO_MPS, 5.0 * DEG_TO_RAD],
        DURATION_CLIMB,
        climb_step,
        control_dim=1,
        control_guess=np.array([5.0 * DEG_TO_RAD]),
    )

    theta_init = np.array(
        [
            28.12256353,
            28.15292765,
            8.29989208,
            1.06218046,
            4.02118536,
            2197.0,
            0.09735854740699715,
        ]
    )

    climb_controls = np.array(
        [0.09735855, 0.07399866, 0.13706639, 0.15410363, 0.14712257, 0.10481729]
    )
    climb_index = 0
    for edge_index, phase in enumerate(edge_phase):
        if phase == "climb":
            u_init[edge_index] = np.array([climb_controls[climb_index]])
            climb_index += 1

    for edge_index, edge in enumerate(edges):
        x_sym = ca.SX.sym("x", state_dims[edge.parent])
        u_sym = ca.SX.sym("u", edge.control_dim)
        theta_sym = ca.SX.sym("theta", len(theta_init))
        step = ca.Function(
            f"warm_step_{edge_index}",
            [x_sym, u_sym, theta_sym],
            [edge.dynamics(x_sym, u_sym, theta_sym, ca.DM(edge.parameters))],
        )
        x_init[edge.child] = np.array(
            step(x_init[edge.parent], u_init[edge_index], theta_init)
        ).reshape(-1)

    rto_final = phase_nodes["rto"][-1]
    v1_to_vr_final = phase_nodes["v1_to_vr"][-1]
    rotate_terminal = phase_nodes["rotate"][-1]
    climb_first = phase_nodes["climb"][0]
    climb_final = phase_nodes["climb"][-1]

    c_dims = [0 for _ in state_dims]
    g_dims = [0 for _ in state_dims]

    node_equalities = {
        rto_final: ["rto_final_v", "rto_field_length"],
        rotate_terminal: ["rotate_final_normal_force"],
        climb_first: ["climb_initial_alpha"],
        climb_final: ["climb_final_h", "climb_final_gam", "climb_field_length"],
    }
    for node, constraints in node_equalities.items():
        c_dims[node] = len(constraints)

    def outgoing_control_bounds(node):
        bounds = []
        for edge_index, edge in enumerate(edges):
            if edge.parent != node or edge.control_dim == 0:
                continue
            if edge_phase[edge_index] == "rotate":
                bounds.append((0.0, 10.0 * DEG_TO_RAD))
            elif edge_phase[edge_index] == "climb":
                bounds.append((-10.0 * DEG_TO_RAD, 15.0 * DEG_TO_RAD))
        return bounds

    def base_inequality_count(node):
        count = state_dims[node]
        count += 2 * len(outgoing_control_bounds(node))
        if node == br_initial:
            count += 12
        if node == v1_to_vr_final:
            count += 1
        if node == climb_final:
            count += 1
        if node_phase[node] == "climb":
            count += 1
        return count

    for node in range(len(state_dims)):
        g_dims[node] = base_inequality_count(node)

    def root_residual(x, theta):
        del theta
        return ca.vertcat(0.0 - x[0], 1.0e-4 - x[1])

    def cost(node, x, theta):
        del theta
        if node == rto_final:
            return x[0] / R_REF
        return ca.SX(0.0)

    def equalities(node, x, theta, outgoing_controls, outgoing_parameters):
        del outgoing_parameters
        pieces = []
        for kind in node_equalities.get(node, []):
            if kind == "rto_final_v":
                pieces.append(x[1] / V_REF)
            elif kind == "rto_field_length":
                pieces.append((x[0] - theta[THETA_FIELD_LENGTH]) / R_REF)
            elif kind == "rotate_final_normal_force":
                pieces.append(
                    aero_runway(x, theta[THETA_ALPHA_LINK], t_engine_out, mu_nominal)[
                        "F_r"
                    ]
                    / FORCE_REF
                )
            elif kind == "climb_initial_alpha":
                pieces.append(
                    (outgoing_controls[0][0] - theta[THETA_ALPHA_LINK]) / ALPHA_REF
                )
            elif kind == "climb_final_h":
                pieces.append((x[1] - 35.0 * FT_TO_M) / (35.0 * FT_TO_M))
            elif kind == "climb_final_gam":
                pieces.append((x[3] - 5.0 * DEG_TO_RAD) / (5.0 * DEG_TO_RAD))
            elif kind == "climb_field_length":
                pieces.append((x[0] - theta[THETA_FIELD_LENGTH]) / R_REF)
            else:
                raise ValueError(kind)
        return ca.vertcat(*pieces) if pieces else ca.SX.zeros(0, 1)

    def inequalities(node, x, theta, outgoing_controls, outgoing_parameters):
        del outgoing_parameters
        pieces = []
        state_refs = (
            [R_REF, V_REF] if state_dims[node] == 2 else [R_REF, H_REF, V_REF, GAM_REF]
        )
        pieces.extend([-x[i] / state_refs[i] for i in range(state_dims[node])])
        for control, (lower, upper) in zip(
            outgoing_controls, outgoing_control_bounds(node)
        ):
            pieces.append(
                control_bounds(
                    control / ALPHA_REF, [lower / ALPHA_REF], [upper / ALPHA_REF]
                )
            )
        if node == br_initial:
            duration_bounds = [
                (DURATION_BR, 1.0, 1000.0, 10.0),
                (DURATION_RTO, 1.0, 1000.0, 1.0),
                (DURATION_V1, 1.0, 1000.0, 1.0),
                (DURATION_ROTATE, 1.0, 5.0, 1.0),
                (DURATION_CLIMB, 1.0, 100.0, 1.0),
            ]
            for index, lower, upper, ref in duration_bounds:
                pieces.append((lower - theta[index]) / ref)
                pieces.append((theta[index] - upper) / ref)
            pieces.append((0.0 - theta[THETA_ALPHA_LINK]) / ALPHA_REF)
            pieces.append((theta[THETA_ALPHA_LINK] - 10.0 * DEG_TO_RAD) / ALPHA_REF)
        if node == v1_to_vr_final:
            pieces.append(
                (1.2 - aero_runway(x, 0.0, t_engine_out, mu_nominal)["v_over_v_stall"])
                / 100.0
            )
        if node_phase[node] == "climb":
            pieces.append((x[3] - 5.0 * DEG_TO_RAD) / (5.0 * DEG_TO_RAD))
        if node == climb_final:
            pieces.append(
                (
                    1.25
                    - aero_climb(x, theta[THETA_ALPHA_LINK], t_engine_out)[
                        "v_over_v_stall"
                    ]
                )
                / 1.25
            )
        return ca.vertcat(*pieces) if pieces else ca.SX.zeros(0, 1)

    return GraphProblemData(
        name="dymos/balanced_field",
        state_dims=state_dims,
        edges=edges,
        theta_dim=7,
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
  settings.penalty.initial_penalty_parameter = 1000.0;
  settings.penalty.penalty_parameter_increase_factor = 1.5;
  settings.barrier.initial_mu = 1e-3;
  settings.line_search.skip_line_search = false;
  settings.line_search.start_ls_with_alpha_s_max = true;
  settings.regularization.maximum = 1e12;
""",
    )


if __name__ == "__main__":
    codegen_main(make_problem)
