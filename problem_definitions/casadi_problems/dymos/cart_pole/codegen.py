import casadi as ca
import numpy as np

from problem_definitions.casadi_problems.codegen_common import (
    GraphEdge,
    GraphProblemData,
    main as codegen_main,
)
from problem_definitions.casadi_problems.dymos.common import control_bounds


X = 0
X_DOT = 1
THETA = 2
THETA_DOT = 3
ENERGY = 4


def make_problem() -> GraphProblemData:
    segments = 12
    duration = 2.0
    dt = duration / segments
    m_cart = 1.0
    m_pole = 0.3
    l_pole = 0.5
    gravity = 9.81

    state_dims = []
    x_init = []
    edges = []
    u_init = []

    def add_node(value):
        state_dims.append(5)
        x_init.append(np.asarray(value, dtype=float))
        return len(state_dims) - 1

    def add_edge(parent, child, control_guess):
        edges.append(GraphEdge(parent, child, 1, np.zeros(0), dynamics))
        u_init.append(np.asarray([control_guess], dtype=float))

    def ode(x, force):
        theta = x[THETA]
        omega = x[THETA_DOT]
        sint = ca.sin(theta)
        cost = ca.cos(theta)
        det = m_pole * l_pole * cost**2 - l_pole * (m_cart + m_pole)
        x_dotdot = (
            -m_pole * l_pole * gravity * sint * cost
            - l_pole * (force + m_pole * l_pole * omega**2 * sint)
        ) / det
        theta_dotdot = (
            (m_cart + m_pole) * gravity * sint
            + cost * (force + m_pole * l_pole * omega**2 * sint)
        ) / det
        return ca.vertcat(x[X_DOT], x_dotdot, x[THETA_DOT], theta_dotdot, force**2)

    def dynamics(x, u, theta, parameters):
        del theta, parameters

        def f(x_ode, force):
            return ode(x_ode, force)

        k1 = f(x, u[0])
        k2 = f(x + dt * k1 / 2.0, u[0])
        k3 = f(x + dt * k2 / 2.0, u[0])
        k4 = f(x + dt * k3, u[0])
        return x + dt * (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0

    def interp(time, values):
        return np.interp(time, [0.0, 1.0, 2.0], values)

    for i in range(segments + 1):
        t = i * dt
        add_node(
            [
                interp(t, [0.0, 1.0, 1.0]),
                interp(t, [0.0, 0.1, 0.0]),
                interp(t, [0.0, np.pi / 2.0, np.pi]),
                interp(t, [0.0, 1.0, 0.0]),
                interp(t, [0.0, 30.0, 60.0]),
            ]
        )
    for i in range(segments):
        add_edge(i, i + 1, interp(i * dt, [3.0, -1.0, 0.0]))

    penultimate = segments - 1
    terminal = segments
    c_dims = [0 for _ in state_dims]
    c_dims[penultimate] = 1
    c_dims[terminal] = 4

    g_dims = []
    for node in range(len(state_dims)):
        count = 2
        if node < segments:
            count += 2
        g_dims.append(count)

    def root_residual(x, theta):
        del theta
        return ca.vertcat(-x[X], -x[X_DOT], -x[THETA], -x[THETA_DOT], -x[ENERGY])

    def cost(node, x, theta):
        del theta
        if node == terminal:
            return x[ENERGY]
        return ca.SX(0.0)

    def equalities(node, x, theta, outgoing_controls, outgoing_parameters):
        del theta, outgoing_parameters
        if node == penultimate:
            return ca.vertcat(outgoing_controls[0][0])
        if node == terminal:
            return ca.vertcat(x[X] - 1.0, x[X_DOT], x[THETA] - np.pi, x[THETA_DOT])
        return ca.SX.zeros(0, 1)

    def inequalities(node, x, theta, outgoing_controls, outgoing_parameters):
        del theta, outgoing_parameters
        pieces = [
            -2.0 - x[X],
            x[X] - 2.0,
        ]
        if node < segments:
            pieces.append(control_bounds(outgoing_controls[0], [-20.0], [20.0]))
        return ca.vertcat(*pieces)

    return GraphProblemData(
        name="dymos/cart_pole",
        state_dims=state_dims,
        edges=edges,
        theta_dim=0,
        c_dims=c_dims,
        g_dims=g_dims,
        X_init=x_init,
        U_init=u_init,
        theta_init=np.zeros(0),
        max_iterations=1000,
        root_residual=root_residual,
        cost=cost,
        equalities=equalities,
        inequalities=inequalities,
        settings_override_cpp="""
  settings.penalty.initial_penalty_parameter = 100.0;
  settings.penalty.penalty_parameter_increase_factor = 1.5;
  settings.barrier.initial_mu = 1e-3;
  settings.line_search.skip_line_search = false;
""",
    )


if __name__ == "__main__":
    codegen_main(make_problem)
