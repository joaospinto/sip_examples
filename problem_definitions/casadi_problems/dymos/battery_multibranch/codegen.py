import casadi as ca
import numpy as np

from problem_definitions.casadi_problems.codegen_common import (
    GraphEdge,
    GraphProblemData,
    main as codegen_main,
)
from problem_definitions.casadi_problems.dymos.common import control_bounds, rk4_step


TRAIN_SOC = np.array([0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0])
AKIMA_C = np.array(
    [
        [
            -1.923076923076894,
            -9.945609945610054,
            3.180606060606042,
            2.053333333333388,
            -30.555555555555195,
            -6.250000000000003,
        ],
        [
            1.0256410256410349,
            2.175602175602198,
            -1.376969696969688,
            0.07333333333331704,
            7.638888888888806,
            -0.41666666666662755,
        ],
        [
            0.4166666666666637,
            0.5641025641025638,
            0.5454545454545447,
            0.45333333333333364,
            0.8750000000000023,
            1.1041666666666683,
        ],
        [3.5, 3.55, 3.65, 3.75, 3.9, 4.1],
    ]
)


def akima_v_oc(soc):
    pieces = []
    for i in range(TRAIN_SOC.size - 1):
        dx = soc - TRAIN_SOC[i]
        value = (
            AKIMA_C[0, i] * dx**3
            + AKIMA_C[1, i] * dx**2
            + AKIMA_C[2, i] * dx
            + AKIMA_C[3, i]
        )
        if i == 0:
            cond = soc <= TRAIN_SOC[i + 1]
        elif i == TRAIN_SOC.size - 2:
            cond = soc >= TRAIN_SOC[i]
        else:
            cond = ca.logic_and(soc >= TRAIN_SOC[i], soc <= TRAIN_SOC[i + 1])
        pieces.append((cond, value))
    result = pieces[-1][1]
    for cond, value in reversed(pieces[:-1]):
        result = ca.if_else(cond, value, result)
    return result


def power_balance(soc, current, num_battery, num_motor):
    r_0 = 0.025
    power_out_gearbox = 3.6
    i_pack = num_battery * current
    v_l = akima_v_oc(soc) - current * r_0
    p_pack = i_pack * v_l
    efficiency = 0.9 - 0.3 * i_pack / num_motor
    return p_pack - power_out_gearbox / efficiency


def make_problem() -> GraphProblemData:
    segments_per_phase = 5
    duration = 3600.0
    dt = duration / segments_per_phase

    state_dims = [1]
    x_init = [np.array([0.0])]
    edges = []
    u_init = []
    c_dims = [0]
    g_dims = [0]
    phase_nodes = {}

    def add_node(initial_soc):
        state_dims.append(1)
        x_init.append(np.array([initial_soc]))
        c_dims.append(0)
        g_dims.append(0)
        return len(state_dims) - 1

    first = add_node(1.0)
    edges.append(GraphEdge(0, first, 0, lambda x, u, theta: ca.vertcat(1.0)))
    u_init.append(np.zeros(0))

    def add_phase(name, start_node, end_guess, num_battery, num_motor):
        nodes = [start_node]
        soc0 = x_init[start_node][0]
        guesses = np.linspace(soc0, end_guess, segments_per_phase + 1)
        for value in guesses[1:]:
            nodes.append(add_node(value))
        current_guess = 1.05 * (soc0 - end_guess)

        def ode(x, u, theta):
            del x, theta
            return ca.vertcat(-u[0] / (3600.0 * 1.05))

        for i in range(segments_per_phase):
            def dyn(x, u, theta, ode=ode):
                return rk4_step(ode, x, u, theta, dt)

            edges.append(GraphEdge(nodes[i], nodes[i + 1], 1, dyn))
            u_init.append(np.array([current_guess]))
        phase_nodes[name] = nodes
        return nodes[-1]

    phase0_final = add_phase("phase0", first, 0.63464982, 3, 3)
    add_phase("phase1", phase0_final, 0.23794217, 3, 3)
    add_phase("phase1_bfail", phase0_final, 0.0281523, 2, 3)
    add_phase("phase1_mfail", phase0_final, 0.18625395, 3, 2)

    edge_model = {}
    for edge_index, edge in enumerate(edges):
        if edge.control_dim == 0:
            continue
        child = edge.child
        if child in phase_nodes["phase0"][1:]:
            edge_model[edge_index] = (3, 3)
        elif child in phase_nodes["phase1"][1:]:
            edge_model[edge_index] = (3, 3)
        elif child in phase_nodes["phase1_bfail"][1:]:
            edge_model[edge_index] = (2, 3)
        elif child in phase_nodes["phase1_mfail"][1:]:
            edge_model[edge_index] = (3, 2)
        c_dims[edge.parent] += 1
        g_dims[edge.parent] += 2

    outgoing_edge_models = {node: [] for node in range(len(state_dims))}
    for edge_index, edge in enumerate(edges):
        if edge.control_dim == 1:
            outgoing_edge_models[edge.parent].append(edge_model[edge_index])

    def root_residual(x, theta):
        del theta
        return ca.vertcat(-x[0])

    def cost(node, x, theta):
        del node, x, theta
        return ca.SX(0.0)

    def equalities(node, x, theta, outgoing_controls):
        del theta
        pieces = []
        for control, (num_battery, num_motor) in zip(
            outgoing_controls, outgoing_edge_models[node]
        ):
            pieces.append(power_balance(x[0], control[0], num_battery, num_motor))
        return ca.vertcat(*pieces) if pieces else ca.SX.zeros(0, 1)

    def inequalities(node, x, theta, outgoing_controls):
        del node, x, theta
        pieces = [control_bounds(control, [0.0], [50.0]) for control in outgoing_controls]
        return ca.vertcat(*pieces) if pieces else ca.SX.zeros(0, 1)

    return GraphProblemData(
        name="dymos/battery_multibranch",
        state_dims=state_dims,
        edges=edges,
        theta_dim=0,
        c_dims=c_dims,
        g_dims=g_dims,
        X_init=x_init,
        U_init=u_init,
        theta_init=np.zeros(0),
        max_iterations=500,
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
