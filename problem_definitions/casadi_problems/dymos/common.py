import casadi as ca


def rk4_step(ode, x, u, theta, dt):
    k1 = ode(x, u, theta)
    k2 = ode(x + 0.5 * dt * k1, u, theta)
    k3 = ode(x + 0.5 * dt * k2, u, theta)
    k4 = ode(x + dt * k3, u, theta)
    return x + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


def control_bounds(u, lower, upper):
    pieces = []
    for i in range(len(lower)):
        pieces.append(lower[i] - u[i])
        pieces.append(u[i] - upper[i])
    return ca.vertcat(*pieces)
