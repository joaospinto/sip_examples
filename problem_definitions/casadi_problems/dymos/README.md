# Dymos problem corpus

These problems are mathematically equivalent ports of the distinct optimization
examples in Dymos 1.15.1 (`8d6d953afbfcaa6f976c66b1c75a397c030dcbf7`).
They use explicit RK4 rather than Dymos's original transcription. Mesh choices,
initial guesses, scaling, and solver settings are therefore not expected to
match the upstream examples.

The differential equations, fixed parameters, design-variable bounds,
objectives, and constraints do match upstream. The only formulation-level
changes are nonnegative duration bounds where an upstream free-time problem
otherwise admits invalid negative time, and algebraically equivalent folding of
downstream or linked phases into the graph representation.

| Port | Upstream example family |
| --- | --- |
| `balanced_field` | `balanced_field` |
| `ballistic_spacecraft` | `ballistic_spacecraft` |
| `battery_multibranch` | `battery_multibranch` / `multibranch_trajectory` |
| `brachistochrone` | `brachistochrone` |
| `brachistochrone_tandem_phases` | `brachistochrone_tandem_phases` |
| `breakwell` | `double_integrator/test/test_double_integrator_breakwell.py` |
| `bryson_denham` | `bryson_denham` |
| `cannonball_implicit_duration` | `cannonball_implicit_duration` |
| `cart_pole` | `cart_pole` |
| `commercial_aircraft` | `commercial_aircraft` / `aircraft_steady_flight` |
| `double_integrator` | `double_integrator` |
| `finite_burn_orbit_raise` | `finite_burn_orbit_raise` |
| `flying_robot` | `flying_robot` |
| `goddard_rocket` | `goddard_rocket_problem` |
| `hull` | `hull_problem` |
| `hypersensitive` | `hyper_sensitive` |
| `length_constrained_brachistochrone` | `length_constrained_brachistochrone` |
| `low_thrust_spiral` | `low_thrust_spiral` |
| `min_time_climb` | `min_time_climb` |
| `moon_landing` | `moon_landing` |
| `mountain_car` | `mountain_car` |
| `multi_phase_cannonball` | `multi_phase_cannonball` / two-phase cannonball |
| `racecar` | `racecar` |
| `robot_arm` | `robot_arm` |
| `shuttle_reentry_constrained` | constrained `shuttle_reentry` |
| `shuttle_reentry_unconstrained` | unconstrained `shuttle_reentry` |
| `ssto_earth` | `ssto_earth` |
| `ssto_linear_tangent` | `ssto_moon_linear_tangent` |
| `vanderpol` | `vanderpol` |
| `water_rocket_height` | height-objective `water_rocket` |
| `water_rocket_range` | range-objective `water_rocket` |

The following are intentionally not separate ports:

- `oscillator` and `robertson_problem` were explicitly excluded from this
  benchmark corpus.
- The first-order polynomial-control lunar SSTO has the same affine
  `tan(theta)` guidance law and feasible set as `ssto_linear_tangent`.
- The implicit-duration brachistochrone variants change how Dymos/OpenMDAO
  solves for duration, but not the mathematical optimization problem represented
  by `brachistochrone`.
- Upstream-state, vector-state, matrix-state, transcription, linkage-API,
  simulation, and component-implementation variants are test fixtures for Dymos
  features rather than distinct benchmark formulations.

Atmosphere, thrust, and CRM aerodynamic tables are derived from the corresponding
Dymos 1.15.1 data files. Dymos is distributed under the Apache License 2.0.
