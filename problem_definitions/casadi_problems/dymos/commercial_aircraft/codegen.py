import casadi as ca
import numpy as np
from scipy.optimize import root

from problem_definitions.casadi_problems.codegen_common import (
    GraphEdge,
    GraphProblemData,
    main as codegen_main,
)
from problem_definitions.casadi_problems.dymos.atmosphere_1976 import (
    atmosphere_properties_casadi,
)
from problem_definitions.casadi_problems.dymos.commercial_aircraft_data import (
    ALPHA_GRID_DEG,
    ALTITUDE_GRID_FT,
    DRAG_COEFFICIENT,
    ETA_GRID_DEG,
    LIFT_COEFFICIENT,
    MACH_GRID,
    MOMENT_COEFFICIENT,
)
from problem_definitions.casadi_problems.dymos.common import control_bounds, rk4_step


DEG_TO_RAD = np.pi / 180.0
METERS_PER_NAUTICAL_MILE = 1852.0
METERS_PER_KILOFOOT = 304.8
METERS_PER_SECOND_PER_FOOT_PER_MINUTE = 0.00508
KG_PER_LBM = 0.45359237
GRAVITY = 9.80665
MACH = 0.8
REFERENCE_AREA = 427.8
EMPTY_MASS = 0.15e6
PAYLOAD_MASS = 84.02869 * 400.0
SEA_LEVEL_TSFC = 2.0 * 8.951e-6 * GRAVITY
SEA_LEVEL_MAX_THRUST = 1.02e6
STATE_SCALES = np.array(
    [
        1.0e-3 * METERS_PER_NAUTICAL_MILE,
        1.0e2 * KG_PER_LBM,
        1.0e-3 * METERS_PER_KILOFOOT,
    ]
)
CONTROL_SCALES = np.array(
    [METERS_PER_SECOND_PER_FOOT_PER_MINUTE, DEG_TO_RAD, DEG_TO_RAD]
)
THETA_SCALES = np.array([1000.0])
_MACH_INDEX = int(np.flatnonzero(MACH_GRID == MACH)[0])


def _normalized_grid(grid):
    center = 0.5 * (grid[0] + grid[-1])
    scale = 0.5 * (grid[-1] - grid[0])
    return (grid - center) / scale, center, scale


_ALPHA_NORMALIZED, _ALPHA_CENTER, _ALPHA_SCALE = _normalized_grid(ALPHA_GRID_DEG)
_ALTITUDE_NORMALIZED, _ALTITUDE_CENTER, _ALTITUDE_SCALE = _normalized_grid(
    ALTITUDE_GRID_FT
)
_ETA_NORMALIZED, _ETA_CENTER, _ETA_SCALE = _normalized_grid(ETA_GRID_DEG)
_ALPHA_INVERSE = np.linalg.inv(np.polynomial.polynomial.polyvander(_ALPHA_NORMALIZED, 3))
_ALTITUDE_INVERSE = np.linalg.inv(
    np.polynomial.polynomial.polyvander(_ALTITUDE_NORMALIZED, 3)
)
_ETA_INVERSE = np.linalg.inv(np.polynomial.polynomial.polyvander(_ETA_NORMALIZED, 3))


def _aero_polynomial_coefficients(data):
    return np.einsum(
        "ia,jb,kc,abc->ijk",
        _ALPHA_INVERSE,
        _ALTITUDE_INVERSE,
        _ETA_INVERSE,
        data[_MACH_INDEX],
    )


_LIFT_POLYNOMIAL = _aero_polynomial_coefficients(LIFT_COEFFICIENT)
_DRAG_POLYNOMIAL = _aero_polynomial_coefficients(DRAG_COEFFICIENT)
_MOMENT_POLYNOMIAL = _aero_polynomial_coefficients(MOMENT_COEFFICIENT)


def _aero_coefficient(coefficients, alpha_rad, altitude_m, eta_rad):
    alpha = (alpha_rad / DEG_TO_RAD - _ALPHA_CENTER) / _ALPHA_SCALE
    altitude = (altitude_m * 3.28 - _ALTITUDE_CENTER) / _ALTITUDE_SCALE
    eta = (eta_rad / DEG_TO_RAD - _ETA_CENTER) / _ETA_SCALE
    alpha_polynomial = 0.0
    for alpha_power in reversed(range(4)):
        altitude_polynomial = 0.0
        for altitude_power in reversed(range(4)):
            eta_polynomial = 0.0
            for eta_power in reversed(range(4)):
                eta_polynomial = (
                    eta_polynomial * eta
                    + coefficients[alpha_power, altitude_power, eta_power]
                )
            altitude_polynomial = altitude_polynomial * altitude + eta_polynomial
        alpha_polynomial = alpha_polynomial * alpha + altitude_polynomial
    return alpha_polynomial


def _flight_quantities(x, control):
    fuel_mass = x[1]
    altitude = x[2]
    climb_rate, alpha, eta = control[0], control[1], control[2]
    pressure, density, speed_of_sound = atmosphere_properties_casadi(altitude)
    true_airspeed = MACH * speed_of_sound
    gamma = ca.asin(ca.fmin(1.0, ca.fmax(-1.0, climb_rate / true_airspeed)))
    dynamic_pressure = 0.5 * density * true_airspeed**2
    q_area = dynamic_pressure * REFERENCE_AREA
    total_weight = GRAVITY * (fuel_mass + EMPTY_MASS + PAYLOAD_MASS)

    lift_coefficient = _aero_coefficient(_LIFT_POLYNOMIAL, alpha, altitude, eta)
    drag_coefficient = _aero_coefficient(_DRAG_POLYNOMIAL, alpha, altitude, eta) + 0.015
    moment_coefficient = _aero_coefficient(_MOMENT_POLYNOMIAL, alpha, altitude, eta)
    thrust_coefficient = (
        total_weight * ca.sin(gamma) / (ca.cos(alpha) * q_area)
        + drag_coefficient / ca.cos(alpha)
    )
    required_lift_coefficient = (
        total_weight * ca.cos(gamma) / q_area
        - thrust_coefficient * ca.sin(alpha)
    )
    thrust = thrust_coefficient * q_area
    maximum_thrust = SEA_LEVEL_MAX_THRUST * pressure / 101325.0
    throttle = thrust / maximum_thrust
    tsfc = SEA_LEVEL_TSFC - 1.0e-6 * (1.5e-4 * GRAVITY) * altitude
    fuel_rate = -tsfc * thrust / GRAVITY
    range_rate = true_airspeed * ca.cos(gamma)
    return (
        range_rate,
        fuel_rate,
        required_lift_coefficient - lift_coefficient,
        moment_coefficient,
        throttle,
    )


def _equilibrium_controls(states):
    x = ca.SX.sym("equilibrium_x", 3)
    control = ca.SX.sym("equilibrium_control", 3)
    quantities = _flight_quantities(x, control)
    residual = ca.Function(
        "commercial_aircraft_equilibrium",
        [x, control],
        [ca.vertcat(quantities[2], quantities[3])],
    )

    controls = []
    alpha_eta_guess = np.array([0.01, 0.01])
    for state in states:
        def evaluate(alpha_eta):
            value = residual(state, np.concatenate(([0.0], alpha_eta)))
            return np.asarray(value).reshape(-1)

        solution = root(evaluate, alpha_eta_guess)
        if not solution.success:
            raise RuntimeError(
                "failed to initialize commercial-aircraft equilibrium controls: "
                f"{solution.message}"
            )
        alpha_eta_guess = solution.x
        controls.append(np.concatenate(([0.0], alpha_eta_guess)))
    return controls


def make_problem() -> GraphProblemData:
    num_steps = 15
    theta_init = np.array([2000.0])
    times = np.linspace(0.0, 1.0, num_steps + 1)
    starts = np.array([0.0, 30000.0 * KG_PER_LBM, 10.0 * METERS_PER_KILOFOOT])
    finishes = np.array(
        [
            724.0 * METERS_PER_NAUTICAL_MILE,
            1.0e-3 * KG_PER_LBM,
            10.0 * METERS_PER_KILOFOOT,
        ]
    )
    x_init = [starts + time * (finishes - starts) for time in times]
    controls = _equilibrium_controls(x_init[:-1])

    def ode(x, u, theta):
        del theta
        range_rate, fuel_rate, _, _, _ = _flight_quantities(x, u)
        return ca.vertcat(range_rate, fuel_rate, u[0])

    def dynamics(x, u, theta):
        return rk4_step(ode, x, u, theta, theta[0] / num_steps)

    edges = [GraphEdge(i, i + 1, 3, dynamics) for i in range(num_steps)]
    terminal = num_steps

    def root_residual(x, theta):
        del theta
        return starts - x

    def cost(node, x, theta):
        del theta
        return -x[0] / (100.0 * METERS_PER_NAUTICAL_MILE) if node == terminal else ca.SX(0.0)

    def equalities(node, x, theta, outgoing_controls):
        del theta
        if outgoing_controls:
            _, _, lift_residual, moment_residual, _ = _flight_quantities(
                x, outgoing_controls[0]
            )
            return ca.vertcat(lift_residual, moment_residual)
        if node == terminal:
            return ca.vertcat(
                x[1] - 1.0e-3 * KG_PER_LBM,
                (x[2] - 10.0 * METERS_PER_KILOFOOT) / METERS_PER_KILOFOOT,
            )
        return ca.SX.zeros(0, 1)

    def inequalities(node, x, theta, outgoing_controls):
        pieces = [
            -x[0] / (2000.0 * METERS_PER_NAUTICAL_MILE),
            x[0] / (2000.0 * METERS_PER_NAUTICAL_MILE) - 1.0,
            -x[1] / (1.5e5 * KG_PER_LBM),
            x[1] / (1.5e5 * KG_PER_LBM) - 1.0,
            -x[2] / (60.0 * METERS_PER_KILOFOOT),
            x[2] / (60.0 * METERS_PER_KILOFOOT) - 1.0,
        ]
        if outgoing_controls:
            control = outgoing_controls[0]
            pieces.append(
                control_bounds(
                    control,
                    [
                        -3000.0 * METERS_PER_SECOND_PER_FOOT_PER_MINUTE,
                        -20.0 * DEG_TO_RAD,
                        -30.0 * DEG_TO_RAD,
                    ],
                    [
                        3000.0 * METERS_PER_SECOND_PER_FOOT_PER_MINUTE,
                        30.0 * DEG_TO_RAD,
                        30.0 * DEG_TO_RAD,
                    ],
                )
            )
            _, _, _, _, throttle = _flight_quantities(x, control)
            pieces.extend([0.01 - throttle, throttle - 2.0])
        if node == 0:
            pieces.extend([(300.0 - theta[0]) / 5600.0, (theta[0] - 10000.0) / 5600.0])
        return ca.vertcat(*pieces)

    c_dims = [2 for _ in range(num_steps + 1)]
    g_dims = [14 for _ in range(num_steps)] + [6]
    g_dims[0] = 16

    return GraphProblemData(
        name="dymos/commercial_aircraft",
        state_dims=[3 for _ in range(num_steps + 1)],
        edges=edges,
        theta_dim=1,
        c_dims=c_dims,
        g_dims=g_dims,
        X_init=x_init,
        U_init=controls,
        theta_init=theta_init,
        max_iterations=1000,
        root_residual=root_residual,
        cost=cost,
        equalities=equalities,
        inequalities=inequalities,
        state_scales=[STATE_SCALES.copy() for _ in range(num_steps + 1)],
        control_scales=[CONTROL_SCALES.copy() for _ in range(num_steps)],
        theta_scales=THETA_SCALES.copy(),
    )


if __name__ == "__main__":
    codegen_main(make_problem)
