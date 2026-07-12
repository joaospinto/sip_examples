import casadi as ca
import numpy as np

from problem_definitions.casadi_problems.codegen_common import ProblemData


def make_quadpendulum_problem(with_theta: bool) -> ProblemData:
    T = 200
    n = 8
    m = 2
    dt = 0.025
    mass = 0.486
    mass_pole = 0.2 * mass
    grav = 9.81
    l_quad = 0.25
    l_pole = 2.0 * l_quad
    j_quad = 0.00383
    fric = 0.01
    u_hover = 0.5 * (mass + mass_pole) * grav * np.ones(m)
    r_joint = 0.05 * l_quad
    r_tip = 0.1 * l_quad
    r_t = 0.3 * l_quad
    obs = (
        (np.array([-1.0, 0.5]), 0.5),
        (np.array([0.75, -1.0]), 0.75),
        (np.array([-2.0, -1.0]), 0.5),
        (np.array([2.0, 1.0]), 0.5),
    )
    world_lo = np.array([-4.0, -2.0])
    world_hi = np.array([4.0, 2.0])
    theta_lim = 3.0 * np.pi / 4.0
    ctrl_lo = 0.1 * mass * grav * np.ones(m)
    ctrl_hi = 3.0 * mass * grav * np.ones(m)
    pos_0 = np.array([-2.5, 1.5, 0.0, 0.0])
    pos_g = np.array([3.0, -1.5, 0.0, np.pi])
    goal = np.concatenate((pos_g, np.zeros(4)))
    weights = np.array((0.01, 0.05, 5.0, 10.0))
    qt = np.array((10.0, 10.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0))

    def mass_matrix(q):
        phi = q[-1]
        a = mass + mass_pole
        b = mass_pole * l_pole * ca.cos(phi)
        c = mass_pole * l_pole * ca.sin(phi)
        d = mass_pole * l_pole * l_pole
        return ca.vertcat(
            ca.horzcat(a, 0.0, 0.0, b),
            ca.horzcat(0.0, a, 0.0, c),
            ca.horzcat(0.0, 0.0, j_quad, 0.0),
            ca.horzcat(b, c, 0.0, d),
        )

    def ode(x, u, theta):
        del theta
        q = x[:4]
        qd = x[4:]
        phi = q[-1]
        theta_quad = q[2]
        M = mass_matrix(q)
        dphi = qd[-1]
        dM_dphi = ca.vertcat(
            ca.horzcat(0.0, 0.0, 0.0, -mass_pole * l_pole * ca.sin(phi)),
            ca.horzcat(0.0, 0.0, 0.0, mass_pole * l_pole * ca.cos(phi)),
            ca.horzcat(0.0, 0.0, 0.0, 0.0),
            ca.horzcat(
                -mass_pole * l_pole * ca.sin(phi),
                mass_pole * l_pole * ca.cos(phi),
                0.0,
                0.0,
            ),
        )
        Mdot_qd = (dphi * dM_dphi) @ qd
        dKE_dphi = 0.5 * ca.dot(qd, dM_dphi @ qd)
        dPE_dq1 = (mass + mass_pole) * grav
        dPE_dphi = mass_pole * grav * l_pole * ca.sin(phi)
        dL_dq = ca.vertcat(0.0, -dPE_dq1, 0.0, dKE_dphi - dPE_dphi)
        torque_fric_pole = -fric * (qd[-1] - qd[-2])
        F_q = ca.vertcat(
            -ca.sum1(u) * ca.sin(theta_quad),
            ca.sum1(u) * ca.cos(theta_quad),
            (u[0] - u[1]) * l_quad - torque_fric_pole,
            torque_fric_pole,
        )
        qdd = ca.solve(M, F_q + dL_dq - Mdot_qd)
        return ca.vertcat(qd, qdd)

    def dynamics(x, u, theta):
        return x + dt * ode(x, u, theta)

    def cost(x, u, theta, terminal):
        delta = x - goal
        pos_cost = ca.dot(delta[:3], delta[:3]) + (1.0 + ca.cos(x[3]))
        ctrl_cost = ca.dot(u - u_hover, u - u_hover)
        stage_cost = weights[0] * pos_cost + weights[1] * ctrl_cost
        term_cost = weights[2] * ca.dot(delta, qt * delta)
        margin = ca.sum1(theta) if with_theta else 0.0
        # The original benchmark attaches this global-theta reward at t=0.
        # Since theta is cross-stage, attaching it once at the terminal stage
        # gives the same NLP derivatives while keeping a single reusable
        # non-terminal OCP stage function.
        return 0.5 * (term_cost if terminal else stage_cost) - (
            weights[3] * margin if with_theta and terminal else 0.0
        )

    def geometry(q):
        pos = q[:2]
        theta_quad = q[2]
        phi = q[-1]
        R = ca.vertcat(
            ca.horzcat(ca.cos(theta_quad), -ca.sin(theta_quad)),
            ca.horzcat(ca.sin(theta_quad), ca.cos(theta_quad)),
        )
        offsets = [
            np.array([0.0, 0.0]),
            np.array([0.0, 0.15 * l_quad]),
            np.array([-l_quad, 0.3 * l_quad]),
            np.array([l_quad, 0.3 * l_quad]),
        ]
        centers = [pos + R @ off for off in offsets]
        pole_tip = pos + ca.vertcat(l_pole * ca.sin(phi), -l_pole * ca.cos(phi))
        centers.append(pole_tip)
        return centers, np.array([r_joint, l_quad, r_t, r_t, r_tip]), pos, pole_tip

    def equalities(x, u, theta, terminal):
        del u, theta
        return x - goal if terminal else ca.SX.zeros(n, 1)

    def inequalities(x, u, theta, terminal):
        margin = ca.sum1(theta) if with_theta else 0.0
        centers, radii, pos, pole_tip = geometry(x[:4])
        pieces = [x[2] - theta_lim, -x[2] - theta_lim]
        for center, radius in zip(centers, radii):
            pieces.extend([-center[0] + world_lo[0] + radius, -center[1] + world_lo[1] + radius])
        for center, radius in zip(centers, radii):
            pieces.extend([center[0] - world_hi[0] + radius, center[1] - world_hi[1] + radius])
        for obs_center, obs_radius in obs:
            for center, radius in zip(centers, radii):
                dx = center[0] - obs_center[0]
                dy = center[1] - obs_center[1]
                pieces.append(-(dx * dx + dy * dy - (obs_radius + radius + margin) ** 2))
            seg = pole_tip - pos
            seg_norm_sq = ca.dot(seg, seg) + 1e-12
            tproj = ca.dot(obs_center - pos, seg) / seg_norm_sq
            tproj_clipped = ca.fmin(1.0, ca.fmax(0.0, tproj))
            closest = pos + tproj_clipped * seg
            delta_pole = closest - obs_center
            pieces.append(
                -(ca.dot(delta_pole, delta_pole) - (obs_radius + margin) ** 2)
            )
        if terminal:
            pieces.extend([-1.0, -1.0, -1.0, -1.0])
        else:
            pieces.extend([ctrl_lo[0] - u[0], ctrl_lo[1] - u[1]])
            pieces.extend([u[0] - ctrl_hi[0], u[1] - ctrl_hi[1]])
        return ca.vertcat(*pieces)

    x0 = np.concatenate((pos_0, np.zeros(4)))
    theta_init = np.array([0.0]) if with_theta else np.zeros(0)
    return ProblemData(
        name=(
            "trajax_l4dc/quadpendulum_theta"
            if with_theta
            else "trajax_l4dc/quadpendulum"
        ),
        T=T,
        n=n,
        m=m,
        theta_dim=1 if with_theta else 0,
        c_dim=n,
        g_dim=50,
        x0=x0,
        X_init=np.linspace(x0, goal, T + 1),
        U_init=np.tile(u_hover, (T, 1)),
        theta_init=theta_init,
        max_iterations=2000,
        settings_override_cpp=(
            """
  settings.penalty.initial_penalty_parameter = 10.0;
  settings.penalty.penalty_parameter_increase_factor = 1.3;
  settings.penalty.penalty_parameter_decrease_factor = 0.9;
  settings.barrier.initial_mu = 1e-2;
  settings.barrier.mu_update_factor = 0.6;
"""
            if with_theta
            else ""
        ),
        cost=cost,
        dynamics=dynamics,
        equalities=equalities,
        inequalities=inequalities,
    )

