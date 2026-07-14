import casadi as ca
import numpy as np

from problem_definitions.casadi_problems.codegen_common import (
    GraphEdge,
    GraphProblemData,
    main as codegen_main,
)
from problem_definitions.casadi_problems.dymos.common import control_bounds, rk4_step


DEG_TO_RAD = np.pi / 180.0

DURATION_BURN1 = 0
DURATION_COAST = 1
DURATION_BURN2 = 2
THETA_ACCEL_LINK = 3

STATE_REF = np.array([1.0, 1.0, 1.0, 1.0, 0.1, 0.1])


def make_problem() -> GraphProblemData:
    segments = 5
    c_burn = 1.5
    c_coast = 1.0

    state_dims = []
    x_init = []
    edges = []
    u_init = []
    node_phase = []
    edge_phase = []
    phase_nodes = {}

    def add_node(value, phase):
        state_dims.append(6)
        x_init.append(np.asarray(value, dtype=float))
        node_phase.append(phase)
        return len(state_dims) - 1

    def add_edge(parent, child, control_dim, dynamics, phase, control_guess=None):
        edges.append(GraphEdge(parent, child, control_dim, dynamics))
        edge_phase.append(phase)
        if control_guess is None:
            control_guess = np.zeros(control_dim)
        u_init.append(np.asarray(control_guess, dtype=float))
        return len(edges) - 1

    def ode(x, u1, c):
        r = x[0]
        vr = x[2]
        vt = x[3]
        accel = x[4]
        return ca.vertcat(
            vr,
            vt / r,
            vt**2 / r - 1.0 / r**2 + accel * ca.sin(u1),
            -vr * vt / r + accel * ca.cos(u1),
            accel**2 / c,
            accel,
        )

    def phase_dt(theta, duration_index):
        return theta[duration_index] / segments

    def burn_step(duration_index, c):
        def dyn(x, u, theta):
            def f(x_ode, u_ode, theta_ode):
                del theta_ode
                return ode(x_ode, u_ode[0], c)

            return rk4_step(f, x, u, theta, phase_dt(theta, duration_index))

        return dyn

    def coast_step(x, u, theta):
        del u

        def f(x_ode, u_ode, theta_ode):
            del u_ode, theta_ode
            return ode(x_ode, 0.0, c_coast)

        return rk4_step(f, x, ca.SX.zeros(0, 1), theta, phase_dt(theta, DURATION_COAST))

    def link_burn1_to_coast(x, u, theta):
        del u, theta
        return ca.vertcat(x[0], x[1], x[2], x[3], 0.0, x[5])

    def link_coast_to_burn2(x, u, theta):
        del u
        return ca.vertcat(x[0], x[1], x[2], x[3], theta[THETA_ACCEL_LINK], x[5])

    def add_phase(name, start_node, final_guess, duration_index, dynamics, controls):
        nodes = [start_node]
        guesses = np.linspace(x_init[start_node], np.asarray(final_guess, dtype=float), segments + 1)
        for value in guesses[1:]:
            nodes.append(add_node(value, name))
        phase_nodes[name] = nodes
        for i in range(segments):
            if controls is None:
                add_edge(nodes[i], nodes[i + 1], 0, dynamics, name)
            else:
                add_edge(nodes[i], nodes[i + 1], 1, dynamics, name, [controls[i]])
        return nodes[-1]

    burn1_controls = np.linspace(-3.5, 13.0, segments) * DEG_TO_RAD
    burn2_controls = np.zeros(segments)

    root = add_node([1.0, 0.0, 0.0, 1.0, 0.1, 0.0], "burn1")
    burn1_final = add_phase(
        "burn1",
        root,
        [1.5, 1.7, 0.0, 1.0, 0.0, 0.1],
        DURATION_BURN1,
        burn_step(DURATION_BURN1, c_burn),
        burn1_controls,
    )

    coast_start = add_node([1.3, 2.18, 0.3285, 0.97, 0.0, 0.0], "coast")
    add_edge(burn1_final, coast_start, 0, link_burn1_to_coast, "link_burn1_coast")
    coast_final = add_phase(
        "coast",
        coast_start,
        [1.5, 1.7, 0.0, 1.0, 0.0, 0.0],
        DURATION_COAST,
        coast_step,
        None,
    )

    burn2_start = add_node([1.0, 0.0, 0.0, 1.0, 0.1, 0.1], "burn2")
    add_edge(coast_final, burn2_start, 0, link_coast_to_burn2, "link_coast_burn2")
    add_phase(
        "burn2",
        burn2_start,
        [3.0, 4.0, 0.0, np.sqrt(1.0 / 3.0), 0.0, 0.2],
        DURATION_BURN2,
        burn_step(DURATION_BURN2, c_burn),
        burn2_controls,
    )

    theta_init = np.array([2.25, 3.0, 1.75, 0.1])

    burn1_terminal = phase_nodes["burn1"][-1]
    burn2_terminal = phase_nodes["burn2"][-1]

    c_dims = [0 for _ in state_dims]
    c_dims[burn1_terminal] = 1
    c_dims[burn2_terminal] = 3

    def outgoing_control_bounds(node):
        bounds = []
        for edge_index, edge in enumerate(edges):
            if edge.parent != node or edge.control_dim == 0:
                continue
            if edge_phase[edge_index] == "burn1":
                bounds.append((-30.0 * DEG_TO_RAD, 30.0 * DEG_TO_RAD))
            elif edge_phase[edge_index] == "burn2":
                bounds.append((-90.0 * DEG_TO_RAD, 90.0 * DEG_TO_RAD))
        return bounds

    g_dims = []
    for node in range(len(state_dims)):
        count = 2 * len(outgoing_control_bounds(node))
        if node == root:
            count += 6
        if node_phase[node] == "burn2":
            count += 1
        g_dims.append(count)

    def root_residual(x, theta):
        del theta
        return ca.vertcat(1.0 - x[0], -x[1], -x[2], 1.0 - x[3], 0.1 - x[4], -x[5])

    def cost(node, x, theta):
        del theta
        if node == burn2_terminal:
            return 100.0 * x[5]
        return ca.SX(0.0)

    def equalities(node, x, theta, outgoing_controls):
        del outgoing_controls
        pieces = []
        if node == burn1_terminal:
            pieces.append((x[4] - theta[THETA_ACCEL_LINK]) / STATE_REF[4])
        if node == burn2_terminal:
            pieces.extend(
                [
                    x[0] - 3.0,
                    x[2],
                    x[3] - np.sqrt(1.0 / 3.0),
                ]
            )
        return ca.vertcat(*pieces) if pieces else ca.SX.zeros(0, 1)

    def inequalities(node, x, theta, outgoing_controls):
        pieces = []
        for control, (lower, upper) in zip(outgoing_controls, outgoing_control_bounds(node)):
            pieces.append(control_bounds(control / DEG_TO_RAD, [lower / DEG_TO_RAD], [upper / DEG_TO_RAD]))
        if node == root:
            duration_bounds = [
                (DURATION_BURN1, 0.5, 10.0, 1.0),
                (DURATION_COAST, 0.5, 50.0, 50.0),
                (DURATION_BURN2, 0.5, 10.0, 1.0),
            ]
            for index, lower, upper, ref in duration_bounds:
                pieces.append((lower - theta[index]) / ref)
                pieces.append((theta[index] - upper) / ref)
        if node_phase[node] == "burn2":
            pieces.append(-x[5])
        return ca.vertcat(*pieces) if pieces else ca.SX.zeros(0, 1)

    return GraphProblemData(
        name="dymos/finite_burn_orbit_raise",
        state_dims=state_dims,
        edges=edges,
        theta_dim=4,
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
  settings.line_search.use_filter_line_search = true;
  settings.line_search.filter_min_total_line_search_iterations = 300;
""",
    )


if __name__ == "__main__":
    codegen_main(make_problem)
