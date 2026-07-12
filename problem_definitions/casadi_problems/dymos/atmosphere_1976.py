import casadi as ca
import numpy as np

from problem_definitions.casadi_problems.dymos.atmosphere_1976_data import (
    AKIMA_DENSITY,
    AKIMA_PRESSURE,
    AKIMA_TEMPERATURE,
    ALT_FT,
    DENSITY_SLUG_FT3,
)


FT_PER_M = 1.0 / 0.3048
PA_PER_PSI = 6894.757293168361
KG_M3_PER_SLUG_FT3 = 515.3788184
M_S_PER_FT_S = 0.3048
GAS_SPEED_OF_SOUND_FACTOR = 1.4 * 1716.49
GEOPOTENTIAL_RADIUS_FT = 6_356_766.0 * FT_PER_M
_AKIMA_ORIGINS_FT = np.concatenate(([ALT_FT[0]], ALT_FT[:-1], [ALT_FT[-1]]))
_LINEAR_DENSITY = ca.interpolant(
    "dymos_linear_density",
    "linear",
    [ALT_FT * 0.3048],
    DENSITY_SLUG_FT3 * KG_M3_PER_SLUG_FT3,
)


def _akima_numpy(altitude_ft, coefficients):
    index = np.searchsorted(ALT_FT, altitude_ft, side="left")
    left = ALT_FT[0] if index == 0 else ALT_FT[index - 1]
    dx = altitude_ft - left
    coeff = coefficients[index]
    return coeff[0] + dx * (coeff[1] + dx * (coeff[2] + dx * coeff[3]))


def _akima_casadi(altitude_ft, coefficients):
    transitions = ca.DM(-ALT_FT[::-1])

    def select(values):
        return ca.pw_const(-altitude_ft, transitions, ca.DM(values[::-1]))

    dx = altitude_ft - select(_AKIMA_ORIGINS_FT)
    c0 = select(coefficients[:, 0])
    c1 = select(coefficients[:, 1])
    c2 = select(coefficients[:, 2])
    c3 = select(coefficients[:, 3])
    return c0 + dx * (c1 + dx * (c2 + dx * c3))


def _altitude_ft(altitude_m, geodetic):
    altitude_ft = altitude_m * FT_PER_M
    if geodetic:
        altitude_ft = altitude_ft * GEOPOTENTIAL_RADIUS_FT / (
            GEOPOTENTIAL_RADIUS_FT + altitude_ft
        )
    return altitude_ft


def atmosphere_numpy(altitude_m, geodetic=False):
    altitude_ft = _altitude_ft(altitude_m, geodetic)
    pressure = _akima_numpy(altitude_ft, AKIMA_PRESSURE) * PA_PER_PSI
    density = _akima_numpy(altitude_ft, AKIMA_DENSITY) * KG_M3_PER_SLUG_FT3
    return pressure, density


def atmosphere_casadi(altitude_m, geodetic=False):
    altitude_ft = _altitude_ft(altitude_m, geodetic)
    pressure = _akima_casadi(altitude_ft, AKIMA_PRESSURE) * PA_PER_PSI
    density = _akima_casadi(altitude_ft, AKIMA_DENSITY) * KG_M3_PER_SLUG_FT3
    return pressure, density


def atmosphere_properties_numpy(altitude_m, geodetic=False):
    altitude_ft = _altitude_ft(altitude_m, geodetic)
    pressure = _akima_numpy(altitude_ft, AKIMA_PRESSURE) * PA_PER_PSI
    density = _akima_numpy(altitude_ft, AKIMA_DENSITY) * KG_M3_PER_SLUG_FT3
    temperature = _akima_numpy(altitude_ft, AKIMA_TEMPERATURE)
    speed_of_sound = np.sqrt(GAS_SPEED_OF_SOUND_FACTOR * temperature) * M_S_PER_FT_S
    return pressure, density, speed_of_sound


def atmosphere_properties_casadi(altitude_m, geodetic=False):
    altitude_ft = _altitude_ft(altitude_m, geodetic)
    pressure = _akima_casadi(altitude_ft, AKIMA_PRESSURE) * PA_PER_PSI
    density = _akima_casadi(altitude_ft, AKIMA_DENSITY) * KG_M3_PER_SLUG_FT3
    temperature = _akima_casadi(altitude_ft, AKIMA_TEMPERATURE)
    speed_of_sound = ca.sqrt(GAS_SPEED_OF_SOUND_FACTOR * temperature) * M_S_PER_FT_S
    return pressure, density, speed_of_sound


def linear_density_numpy(altitude_m):
    altitude_ft = np.asarray(altitude_m) * FT_PER_M
    index = np.searchsorted(ALT_FT, altitude_ft, side="right") - 1
    index = np.clip(index, 0, len(ALT_FT) - 2)
    slope = (DENSITY_SLUG_FT3[index + 1] - DENSITY_SLUG_FT3[index]) / (
        ALT_FT[index + 1] - ALT_FT[index]
    )
    density = DENSITY_SLUG_FT3[index] + slope * (altitude_ft - ALT_FT[index])
    return density * KG_M3_PER_SLUG_FT3


def linear_density_casadi(altitude_m):
    return _LINEAR_DENSITY(altitude_m)
