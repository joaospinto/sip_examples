import casadi as ca
import numpy as np

from problem_definitions.casadi_problems.codegen_common import (
    ProblemData,
    main as codegen_main,
)
from problem_definitions.casadi_problems.dymos.common import control_bounds, rk4_step


DEG_TO_RAD = np.pi / 180.0
RAD_TO_DEG = 180.0 / np.pi

H = 0
GAMMA = 1
PHI = 2
PSI = 3
THETA = 4
V = 5

ALPHA = 0
BETA = 1

STATE_REFS = np.array([300000.0, 1.0, 1.0, 1.0, 1.0, 25000.0])
TIME_REF = 2000.0


def _physical_state(x):
    return ca.vertcat(*[x[i] * STATE_REFS[i] for i in range(6)])


def _atmosphere(h):
    h_r = 23800.0
    rho_0 = 0.002378
    return rho_0 * ca.exp(-h / h_r)


def _aerodynamics(alpha_rad, v, rho):
    alpha_deg = alpha_rad * RAD_TO_DEG
    a_0 = -0.20704
    a_1 = 0.029244
    b_0 = 0.07854
    b_1 = -0.61592e-2
    b_2 = 0.621408e-3
    s_ref = 2690.0
    c_l = a_0 + a_1 * alpha_deg
    c_d = b_0 + b_1 * alpha_deg + b_2 * alpha_deg**2
    drag = 0.5 * c_d * s_ref * rho * v**2
    lift = 0.5 * c_l * s_ref * rho * v**2
    return lift, drag


def _heating(alpha_rad, v, rho):
    alpha_deg = alpha_rad * RAD_TO_DEG
    c_0 = 1.0672181
    c_1 = -0.19213774e-1
    c_2 = 0.21286289e-3
    c_3 = -0.10117249e-5
    q_r = 17700.0 * ca.sqrt(rho) * (0.0001 * v) ** 3.07
    q_a = c_0 + c_1 * alpha_deg + c_2 * alpha_deg**2 + c_3 * alpha_deg**3
    return q_r * q_a


def _rates_physical(x_phys, u):
    h = x_phys[H]
    gamma = x_phys[GAMMA]
    phi = x_phys[PHI]
    psi = x_phys[PSI]
    theta = x_phys[THETA]
    v = x_phys[V]
    del phi

    rho = _atmosphere(h)
    lift, drag = _aerodynamics(u[ALPHA], v, rho)

    g_0 = 32.174
    weight = 203000.0
    r_e = 20902900.0
    mu = 0.14076539e17
    mass = weight / g_0
    r = r_e + h
    g = mu / r**2

    s_beta = ca.sin(u[BETA])
    c_beta = ca.cos(u[BETA])
    s_gamma = ca.sin(gamma)
    c_gamma = ca.cos(gamma)
    s_psi = ca.sin(psi)
    c_psi = ca.cos(psi)
    c_theta = ca.cos(theta)
    s_theta = ca.sin(theta)

    hdot = v * s_gamma
    gammadot = lift / (mass * v) * c_beta + c_gamma * (v / r - g / v)
    phidot = v / r * c_gamma * s_psi / c_theta
    psidot = (
        lift * s_beta / (mass * v * c_gamma)
        + v * c_gamma * s_psi * s_theta / (r * c_theta)
    )
    thetadot = c_gamma * c_psi * v / r
    vdot = -drag / mass - g * s_gamma
    return ca.vertcat(hdot, gammadot, phidot, psidot, thetadot, vdot)


def _ode(x, u, theta):
    duration = theta[0] * TIME_REF
    del duration
    rates = _rates_physical(_physical_state(x), u)
    return ca.vertcat(*[rates[i] / STATE_REFS[i] for i in range(6)])


def make_problem(
    constrained: bool = False,
    name: str = "dymos/shuttle_reentry_unconstrained",
) -> ProblemData:
    horizon = 50

    x0_phys = np.array(
        [
            260000.0,
            -1.0 * DEG_TO_RAD,
            0.0,
            90.0 * DEG_TO_RAD,
            0.0,
            25600.0,
        ]
    )
    xf_phys = np.array(
        [
            80000.0,
            -5.0 * DEG_TO_RAD,
            75.0 * DEG_TO_RAD,
            10.0 * DEG_TO_RAD,
            25.0 * DEG_TO_RAD,
            2500.0,
        ]
    )
    x0 = x0_phys / STATE_REFS
    xf_guess = xf_phys / STATE_REFS

    x_init = np.column_stack([np.linspace(x0[i], xf_guess[i], horizon + 1) for i in range(6)])
    u_init = np.column_stack(
        [
            np.linspace(17.4 * DEG_TO_RAD, 17.4 * DEG_TO_RAD, horizon),
            np.linspace(-75.0 * DEG_TO_RAD, 0.0, horizon),
        ]
    )
    theta_init = np.array([2000.0 / TIME_REF])

    def dynamics(x, u, theta):
        dt = theta[0] * TIME_REF / horizon
        return rk4_step(_ode, x, u, theta, dt)

    def cost(x, u, theta, terminal):
        del u, theta
        if terminal:
            return -x[THETA] * STATE_REFS[THETA] / 0.01
        return ca.SX(0.0)

    def equalities(x, u, theta, terminal):
        del u, theta
        if not terminal:
            return ca.SX.zeros(3, 1)
        return ca.vertcat(
            x[H] - 80000.0 / STATE_REFS[H],
            x[GAMMA] - (-5.0 * DEG_TO_RAD) / STATE_REFS[GAMMA],
            x[V] - 2500.0 / STATE_REFS[V],
        )

    def inequalities(x, u, theta, terminal):
        del terminal
        x_phys = _physical_state(x)
        h = x_phys[H]
        gamma = x_phys[GAMMA]
        phi = x_phys[PHI]
        psi = x_phys[PSI]
        theta_angle = x_phys[THETA]
        v = x_phys[V]
        rho = _atmosphere(h)
        pieces = [
            -h / STATE_REFS[H],
            (-89.0 * DEG_TO_RAD - gamma) / STATE_REFS[GAMMA],
            (gamma - 89.0 * DEG_TO_RAD) / STATE_REFS[GAMMA],
            -phi / STATE_REFS[PHI],
            (phi - 89.0 * DEG_TO_RAD) / STATE_REFS[PHI],
            -psi / STATE_REFS[PSI],
            (psi - 90.0 * DEG_TO_RAD) / STATE_REFS[PSI],
            (-89.0 * DEG_TO_RAD - theta_angle) / STATE_REFS[THETA],
            (theta_angle - 89.0 * DEG_TO_RAD) / STATE_REFS[THETA],
            (500.0 - v) / STATE_REFS[V],
            control_bounds(
                u,
                [-np.pi / 2.0, -89.0 * DEG_TO_RAD],
                [np.pi / 2.0, 1.0 * DEG_TO_RAD],
            ),
            -theta[0],
        ]
        if constrained:
            q = _heating(u[ALPHA], v, rho)
            pieces.append(-q / 70.0)
            pieces.append((q - 70.0) / 70.0)
        return ca.vertcat(*pieces)

    return ProblemData(
        name=name,
        T=horizon,
        n=6,
        m=2,
        theta_dim=1,
        c_dim=3,
        g_dim=17 if constrained else 15,
        x0=x0,
        X_init=x_init,
        U_init=u_init,
        theta_init=theta_init,
        max_iterations=1000,
        cost=cost,
        dynamics=dynamics,
        equalities=equalities,
        inequalities=inequalities,
        settings_override_cpp="""
  settings.penalty.initial_penalty_parameter = 100000.0;
  settings.penalty.penalty_parameter_increase_factor = 1.5;
  settings.barrier.initial_mu = 1e-3;
  settings.regularization.initial = 1e-3;
  settings.regularization.maximum = 1e12;
  settings.line_search.skip_line_search = false;
  settings.line_search.start_ls_with_alpha_s_max = true;
""",
    )


if __name__ == "__main__":
    codegen_main(make_problem)
