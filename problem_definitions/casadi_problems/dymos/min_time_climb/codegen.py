import casadi as ca
import numpy as np

from problem_definitions.casadi_problems.codegen_common import (
    GraphEdge,
    GraphProblemData,
    main as codegen_main,
)
from problem_definitions.casadi_problems.dymos.atmosphere_1976 import (
    atmosphere_properties_casadi,
)
from problem_definitions.casadi_problems.dymos.common import control_bounds, rk4_step
from problem_definitions.casadi_problems.dymos.min_time_climb_data import (
    ALTITUDE_GRID_FT,
    MACH_GRID,
    MAX_THRUST_LBF,
)


DEG_TO_RAD = np.pi / 180.0
FT_PER_M = 1.0 / 0.3048
N_PER_LBF = 4.4482216
GRAVITY = 9.80665
REFERENCE_AREA = 49.2386
ISP = 1600.0
STATE_SCALES = np.array([1.0e3, 2.0e4, 1.0e2, 1.0, 1.0e4])
CONTROL_SCALES = np.array([DEG_TO_RAD])
THETA_SCALES = np.array([100.0])


def _natural_second_derivatives(grid, values):
    count = len(grid)
    second = [0.0 for _ in range(count)]
    temporary = [0.0 for _ in range(count)]
    for index in range(1, count - 1):
        mu = (grid[index] - grid[index - 1]) / (grid[index + 1] - grid[index - 1])
        left_slope = (values[index] - values[index - 1]) / (grid[index] - grid[index - 1])
        right_slope = (values[index + 1] - values[index]) / (grid[index + 1] - grid[index])
        rhs = 6.0 * (right_slope - left_slope) / (grid[index + 1] - grid[index - 1])
        partial = 1.0 / (mu * second[index - 1] + 2.0)
        second[index] = (mu - 1.0) * partial
        temporary[index] = (rhs - mu * temporary[index - 1]) * partial
    for index in range(count - 2, 0, -1):
        second[index] = second[index] * second[index + 1] + temporary[index]
    return second


def _cubic_power_coefficients(grid, values):
    values = np.asarray(values)
    second = np.asarray(_natural_second_derivatives(grid, values))
    coefficients = np.empty((len(grid) - 1, 4))
    for index in range(len(grid) - 1):
        step = grid[index + 1] - grid[index]
        coefficients[index] = [
            values[index],
            (values[index + 1] - values[index]) / step
            - step * (2.0 * second[index] + second[index + 1]) / 6.0,
            second[index] / 2.0,
            (second[index + 1] - second[index]) / (6.0 * step),
        ]
    return coefficients


def _thrust_surface_coefficients():
    mach_coefficients = np.empty(
        (len(ALTITUDE_GRID_FT), len(MACH_GRID) - 1, 4)
    )
    for altitude_index, values in enumerate(MAX_THRUST_LBF):
        mach_coefficients[altitude_index] = _cubic_power_coefficients(
            MACH_GRID, values
        )

    coefficients = np.empty(
        (len(ALTITUDE_GRID_FT) - 1, len(MACH_GRID) - 1, 4, 4)
    )
    for mach_interval in range(len(MACH_GRID) - 1):
        for mach_power in range(4):
            coefficients[:, mach_interval, :, mach_power] = (
                _cubic_power_coefficients(
                    ALTITUDE_GRID_FT,
                    mach_coefficients[:, mach_interval, mach_power],
                )
            )
    return coefficients


_THRUST_COEFFICIENTS = _thrust_surface_coefficients()


def _interval_origin(point, grid):
    return ca.pw_const(
        -point,
        ca.DM(-np.asarray(grid[-2:0:-1])),
        ca.DM(np.asarray(grid[:-1])[::-1]),
    )


def _thrust_coefficient(altitude_ft, mach, altitude_power, mach_power):
    by_altitude = []
    for altitude_interval in range(len(ALTITUDE_GRID_FT) - 1):
        by_altitude.append(
            ca.pw_const(
                -mach,
                ca.DM(-MACH_GRID[-2:0:-1]),
                ca.DM(
                    _THRUST_COEFFICIENTS[
                        altitude_interval, ::-1, altitude_power, mach_power
                    ]
                ),
            )
        )
    return ca.pw_const(
        -altitude_ft,
        ca.DM(-ALTITUDE_GRID_FT[-2:0:-1]),
        ca.vertcat(*reversed(by_altitude)),
    )


def _maximum_thrust(altitude_m, mach):
    altitude_ft = altitude_m * FT_PER_M
    altitude_delta = altitude_ft - _interval_origin(altitude_ft, ALTITUDE_GRID_FT)
    mach_delta = mach - _interval_origin(mach, MACH_GRID)
    altitude_polynomial = 0.0
    for altitude_power in reversed(range(4)):
        mach_polynomial = 0.0
        for mach_power in reversed(range(4)):
            mach_polynomial = (
                mach_polynomial * mach_delta
                + _thrust_coefficient(
                    altitude_ft, mach, altitude_power, mach_power
                )
            )
        altitude_polynomial = altitude_polynomial * altitude_delta + mach_polynomial
    return altitude_polynomial * N_PER_LBF


def _aerodynamics(mach, alpha, density, velocity):
    low = mach < 1.15
    cla_low = 3.44 + 1.0 / ca.cosh((mach - 1.0) / 0.06) ** 2
    cla_high = 3.44 + 1.0 / np.cosh(0.15 / 0.06) ** 2 - 0.96 / 0.63 * (mach - 1.15)
    cd0_low = 0.013 + 0.0144 * (1.0 + ca.tanh((mach - 0.98) / 0.06))
    cd0_high = 0.013 + 0.0144 * (1.0 + np.tanh(0.17 / 0.06)) - 0.011 * (mach - 1.15)
    kappa_low = 0.54 + 0.15 * (1.0 + ca.tanh((mach - 0.9) / 0.06))
    kappa_high = 0.54 + 0.15 * (1.0 + np.tanh(0.25 / 0.06)) + 0.14 * (mach - 1.15)
    cla = ca.if_else(low, cla_low, cla_high)
    cd0 = ca.if_else(low, cd0_low, cd0_high)
    kappa = ca.if_else(low, kappa_low, kappa_high)
    lift_coefficient = cla * alpha
    drag_coefficient = cd0 + cla * kappa * alpha**2
    dynamic_pressure_area = 0.5 * density * velocity**2 * REFERENCE_AREA
    return dynamic_pressure_area * lift_coefficient, dynamic_pressure_area * drag_coefficient


def _mach(altitude, velocity):
    _, _, speed_of_sound = atmosphere_properties_casadi(altitude, geodetic=True)
    return velocity / speed_of_sound


def make_problem() -> GraphProblemData:
    num_steps = 15
    duration = np.array([350.0])
    times = np.linspace(0.0, 1.0, num_steps + 1)
    starts = np.array([0.0, 100.0, 135.964, 0.0, 19030.468])
    finishes = np.array([111319.54, 20000.0, 283.159, 0.0, 16841.431])
    x_init = [starts + time * (finishes - starts) for time in times]
    controls = [np.zeros(1) for _ in range(num_steps)]

    def ode(x, u, theta):
        del theta
        _, density, speed_of_sound = atmosphere_properties_casadi(x[1], geodetic=True)
        mach = x[2] / speed_of_sound
        thrust = _maximum_thrust(x[1], mach)
        lift, drag = _aerodynamics(mach, u[0], density, x[2])
        return ca.vertcat(
            x[2] * ca.cos(x[3]),
            x[2] * ca.sin(x[3]),
            (thrust * ca.cos(u[0]) - drag) / x[4] - GRAVITY * ca.sin(x[3]),
            (thrust * ca.sin(u[0]) + lift) / (x[4] * x[2])
            - GRAVITY * ca.cos(x[3]) / x[2],
            -thrust / (GRAVITY * ISP),
        )

    def dynamics(x, u, theta):
        return rk4_step(ode, x, u, theta, theta[0] / num_steps)

    edges = [GraphEdge(i, i + 1, 1, dynamics) for i in range(num_steps)]
    terminal = num_steps

    def root_residual(x, theta):
        del theta
        return starts - x

    def cost(node, x, theta):
        del x
        return theta[0] if node == terminal else ca.SX(0.0)

    def equalities(node, x, theta, outgoing_controls):
        del theta, outgoing_controls
        if node == terminal:
            return ca.vertcat(
                (x[1] - 20000.0) / 20000.0,
                _mach(x[1], x[2]) - 1.0,
                x[3],
            )
        return ca.SX.zeros(0, 1)

    def inequalities(node, x, theta, outgoing_controls):
        mach = _mach(x[1], x[2])
        pieces = [
            -x[0] / 1.0e6,
            (x[0] - 1.0e6) / 1.0e6,
            (100.0 - x[1]) / 20000.0,
            (x[1] - 20000.0) / 20000.0,
            (10.0 - x[2]) / 100.0,
            -1.5 - x[3],
            x[3] - 1.5,
            (10.0 - x[4]) / 10000.0,
            (x[4] - 1.0e5) / 10000.0,
            0.1 - mach,
            mach - 1.8,
        ]
        if outgoing_controls:
            pieces.append(
                control_bounds(outgoing_controls[0] / DEG_TO_RAD, [-8.0], [8.0])
            )
        if node == 0:
            pieces.extend([(50.0 - theta[0]) / 100.0, (theta[0] - 400.0) / 100.0])
        return ca.vertcat(*pieces)

    c_dims = [0 for _ in range(num_steps + 1)]
    c_dims[terminal] = 3
    g_dims = [13 for _ in range(num_steps)] + [11]
    g_dims[0] = 15

    return GraphProblemData(
        name="dymos/min_time_climb",
        state_dims=[5 for _ in range(num_steps + 1)],
        edges=edges,
        theta_dim=1,
        c_dims=c_dims,
        g_dims=g_dims,
        X_init=x_init,
        U_init=controls,
        theta_init=duration,
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
