import argparse
import os
from dataclasses import dataclass

import casadi as ca
import numpy as np
import scipy as sp
from slacg.internal.common import build_sparse_LT
from slacg.kkt_codegen import kkt_codegen


@dataclass(frozen=True)
class ProblemData:
    name: str
    T: int
    n: int
    m: int
    theta_dim: int
    c_dim: int
    g_dim: int
    x0: np.ndarray
    X_init: np.ndarray
    U_init: np.ndarray
    theta_init: np.ndarray
    max_iterations: int
    cost: object
    dynamics: object
    equalities: object
    inequalities: object
    settings_override_cpp: str = ""


def _c_array(name, values, c_type="int", line_width=12):
    vals = list(values)
    if not vals:
        return f"static constexpr {c_type} {name}[] = {{0}};\n"
    chunks = [vals[i : i + line_width] for i in range(0, len(vals), line_width)]
    rendered = []
    for chunk in chunks:
        rendered.append("  " + ", ".join(str(v) for v in chunk))
    body = ",\n".join(rendered) if rendered else ""
    return f"static constexpr {c_type} {name}[] = {{\n{body}\n}};\n"


def _cpp_double_array(name, values):
    vals = [f"{float(v):.17g}" for v in np.asarray(values).reshape(-1)]
    return _c_array(name, vals, "double", 6)


def _sp_from_sparsity(sparsity):
    rows = np.asarray(sparsity.row(), dtype=int)
    indptr = np.asarray(sparsity.colind(), dtype=int)
    data = np.ones(rows.size)
    return sp.sparse.csc_matrix((data, rows, indptr), shape=sparsity.shape)


def _upper_with_diagonal(raw_upper):
    return (raw_upper + sp.sparse.eye(raw_upper.shape[0], format="csc")).astype(bool).astype(float)


def _full_symmetric_from_upper(upper):
    return (upper + upper.T - sp.sparse.diags(upper.diagonal(), format="csc")).astype(bool).astype(float)


def _kkt_pattern(H_full, C, G):
    x_dim = H_full.shape[0]
    y_dim = C.shape[0]
    z_dim = G.shape[0]
    I_y = sp.sparse.eye(y_dim, format="csc")
    I_z = sp.sparse.eye(z_dim, format="csc")
    Z_yz = sp.sparse.csc_matrix((y_dim, z_dim))
    Z_zy = sp.sparse.csc_matrix((z_dim, y_dim))
    return sp.sparse.bmat(
        [[H_full, C.T, G.T], [C, I_y, Z_yz], [G, Z_zy, I_z]], format="csc"
    ).astype(bool).astype(float)


def _amd_order(K):
    coo = K.tocoo()
    sparsity = ca.Sparsity.triplet(K.shape[0], K.shape[1], coo.row.tolist(), coo.col.tolist())
    return np.asarray(sparsity.amd(), dtype=int)


def _matrix_nonzero_map(src, dst):
    src = src.tocsc()
    dst = dst.tocsc()
    src_map = {}
    for j in range(src.shape[1]):
        for k in range(src.indptr[j], src.indptr[j + 1]):
            src_map[(int(src.indices[k]), j)] = k
    out = []
    for j in range(dst.shape[1]):
        for k in range(dst.indptr[j], dst.indptr[j + 1]):
            coord = (int(dst.indices[k]), j)
            out.append(src_map.get(coord, -1))
    return np.asarray(out, dtype=int)


def _dense(expr):
    return ca.reshape(ca.densify(expr), -1, 1)


def _flat_initial(problem):
    return np.concatenate(
        [
            np.asarray(problem.X_init).reshape(-1),
            np.asarray(problem.U_init).reshape(-1),
            np.asarray(problem.theta_init).reshape(-1),
        ]
    )


def _ocp_initial(problem):
    pieces = []
    for i in range(problem.T):
        pieces.append(problem.X_init[i])
        pieces.append(problem.U_init[i])
    pieces.append(problem.X_init[problem.T])
    pieces.append(problem.theta_init)
    return np.concatenate([np.asarray(p).reshape(-1) for p in pieces])


def _build_flat_function(problem):
    T, n, m, td = problem.T, problem.n, problem.m, problem.theta_dim
    nx = n * (T + 1)
    nu = m * T
    x_dim = nx + nu + td
    X = ca.SX.sym("X", n, T + 1)
    U = ca.SX.sym("U", m, T)
    Theta_sym = ca.SX.sym("Theta", max(td, 1))
    theta = Theta_sym[:td] if td > 0 else ca.SX.zeros(0, 1)
    z_pieces = [ca.reshape(X, -1, 1), ca.reshape(U, -1, 1)]
    if td > 0:
        z_pieces.append(theta)
    z = ca.vertcat(*z_pieces)
    f_total = ca.SX(0.0)
    c_pieces = [X[:, 0] - problem.x0]
    g_pieces = []
    for t in range(T + 1):
        u_t = U[:, t] if t < T else ca.SX.zeros(m, 1)
        terminal = t == T
        f_total += problem.cost(X[:, t], u_t, theta, terminal)
        if t < T:
            c_pieces.append(X[:, t + 1] - problem.dynamics(X[:, t], u_t, theta))
        if problem.c_dim > 0:
            c_pieces.append(problem.equalities(X[:, t], u_t, theta, terminal))
        if problem.g_dim > 0:
            g_pieces.append(problem.inequalities(X[:, t], u_t, theta, terminal))
    c_expr = ca.vertcat(*c_pieces) if c_pieces else ca.SX.zeros(0, 1)
    g_expr = ca.vertcat(*g_pieces) if g_pieces else ca.SX.zeros(0, 1)
    y = ca.SX.sym("Y", c_expr.numel()) if c_expr.numel() > 0 else ca.SX.zeros(0, 1)
    lam_g = ca.SX.sym("Z", g_expr.numel()) if g_expr.numel() > 0 else ca.SX.zeros(0, 1)
    lag = f_total
    if c_expr.numel() > 0:
        lag += ca.dot(y, c_expr)
    if g_expr.numel() > 0:
        lag += ca.dot(lam_g, g_expr)
    H, _ = ca.hessian(lag, z)
    H_raw = ca.triu(H)
    C = ca.jacobian(c_expr, z) if c_expr.numel() > 0 else ca.SX.zeros(0, x_dim)
    G = ca.jacobian(g_expr, z) if g_expr.numel() > 0 else ca.SX.zeros(0, x_dim)
    fun = ca.Function(
        "flat_eval",
        [z, y, lam_g],
        [
            f_total,
            ca.gradient(f_total, z),
            c_expr,
            g_expr,
            H_raw,
            C,
            G,
            C.T,
            G.T,
        ],
    )
    return fun, H_raw.sparsity(), C.sparsity(), G.sparsity()


def _build_flat_stage_functions(problem):
    n, m, td, c_dim, g_dim = (
        problem.n,
        problem.m,
        problem.theta_dim,
        problem.c_dim,
        problem.g_dim,
    )
    x = ca.SX.sym("x", n)
    xp = ca.SX.sym("xp", n)
    u = ca.SX.sym("u", m)
    theta_sym = ca.SX.sym("theta", max(td, 1))
    theta = theta_sym[:td] if td > 0 else ca.SX.zeros(0, 1)
    dyn_mult = ca.SX.sym("dyn_mult", n)
    eq_mult = ca.SX.sym("eq_mult", c_dim) if c_dim > 0 else ca.SX.zeros(0, 1)
    ineq_mult = ca.SX.sym("ineq_mult", g_dim) if g_dim > 0 else ca.SX.zeros(0, 1)

    f = problem.cost(x, u, theta, False)
    dyn = xp - problem.dynamics(x, u, theta)
    eq = problem.equalities(x, u, theta, False) if c_dim > 0 else ca.SX.zeros(0, 1)
    g = problem.inequalities(x, u, theta, False) if g_dim > 0 else ca.SX.zeros(0, 1)
    c = ca.vertcat(dyn, eq)
    xloc = ca.vertcat(x, xp, u, theta)
    lag = f + ca.dot(dyn_mult, dyn)
    if c_dim > 0:
        lag += ca.dot(eq_mult, eq)
    if g_dim > 0:
        lag += ca.dot(ineq_mult, g)
    H, _ = ca.hessian(lag, xloc)
    H = ca.triu(H)
    C = ca.jacobian(c, xloc)
    G = ca.jacobian(g, xloc) if g_dim > 0 else ca.SX.zeros(0, xloc.numel())
    inner = ca.Function(
        "flat_inner_eval",
        [x, xp, u, theta, dyn_mult, eq_mult, ineq_mult],
        [f, ca.gradient(f, xloc), c, g, H, C, G],
    )

    xt = ca.SX.sym("xt", n)
    terminal_u = ca.SX.zeros(m, 1)
    terminal_f = problem.cost(xt, terminal_u, theta, True)
    terminal_eq = (
        problem.equalities(xt, terminal_u, theta, True)
        if c_dim > 0
        else ca.SX.zeros(0, 1)
    )
    terminal_g = (
        problem.inequalities(xt, terminal_u, theta, True)
        if g_dim > 0
        else ca.SX.zeros(0, 1)
    )
    terminal_xloc = ca.vertcat(xt, theta)
    terminal_lag = terminal_f
    if c_dim > 0:
        terminal_lag += ca.dot(eq_mult, terminal_eq)
    if g_dim > 0:
        terminal_lag += ca.dot(ineq_mult, terminal_g)
    terminal_H, _ = ca.hessian(terminal_lag, terminal_xloc)
    terminal_H = ca.triu(terminal_H)
    terminal_C = (
        ca.jacobian(terminal_eq, terminal_xloc)
        if c_dim > 0
        else ca.SX.zeros(0, terminal_xloc.numel())
    )
    terminal_G = (
        ca.jacobian(terminal_g, terminal_xloc)
        if g_dim > 0
        else ca.SX.zeros(0, terminal_xloc.numel())
    )
    terminal = ca.Function(
        "flat_terminal_eval",
        [xt, theta, eq_mult, ineq_mult],
        [
            terminal_f,
            ca.gradient(terminal_f, terminal_xloc),
            terminal_eq,
            terminal_g,
            terminal_H,
            terminal_C,
            terminal_G,
        ],
    )
    return inner, terminal


def _csc_from_entries(shape, entries):
    if not entries:
        return sp.sparse.csc_matrix(shape)
    rows, cols = zip(*sorted(set((int(r), int(c)) for r, c in entries)))
    data = np.ones(len(rows))
    return sp.sparse.coo_matrix((data, (rows, cols)), shape=shape).tocsc()


def _flat_offsets(problem):
    state_offset = 0
    control_offset = problem.n * (problem.T + 1)
    theta_offset = control_offset + problem.m * problem.T
    return state_offset, control_offset, theta_offset


def _flat_inner_col_map(problem, t):
    _, control_offset, theta_offset = _flat_offsets(problem)
    cols = []
    cols.extend(range(t * problem.n, (t + 1) * problem.n))
    cols.extend(range((t + 1) * problem.n, (t + 2) * problem.n))
    cols.extend(range(control_offset + t * problem.m, control_offset + (t + 1) * problem.m))
    cols.extend(range(theta_offset, theta_offset + problem.theta_dim))
    return np.asarray(cols, dtype=int)


def _flat_terminal_col_map(problem):
    _, _, theta_offset = _flat_offsets(problem)
    cols = list(range(problem.T * problem.n, (problem.T + 1) * problem.n))
    cols.extend(range(theta_offset, theta_offset + problem.theta_dim))
    return np.asarray(cols, dtype=int)


def _flat_inner_c_row_map(problem, t):
    base = problem.n + t * (problem.n + problem.c_dim)
    rows = list(range(base, base + problem.n + problem.c_dim))
    return np.asarray(rows, dtype=int)


def _flat_terminal_c_row_map(problem):
    base = problem.n + problem.T * (problem.n + problem.c_dim)
    return np.asarray(range(base, base + problem.c_dim), dtype=int)


def _flat_g_row_map(problem, t):
    base = t * problem.g_dim
    return np.asarray(range(base, base + problem.g_dim), dtype=int)


def _mapped_entries(local_matrix, row_map, col_map):
    entries = []
    local = local_matrix.tocsc()
    for j in range(local.shape[1]):
        for k in range(local.indptr[j], local.indptr[j + 1]):
            entries.append((int(row_map[local.indices[k]]), int(col_map[j])))
    return entries


def _scatter_to_global(local_matrix, row_map, col_map, global_matrix):
    local = local_matrix.tocsc()
    global_matrix = global_matrix.tocsc()
    global_map = {}
    for j in range(global_matrix.shape[1]):
        for k in range(global_matrix.indptr[j], global_matrix.indptr[j + 1]):
            global_map[(int(global_matrix.indices[k]), j)] = k
    scatter = []
    for j in range(local.shape[1]):
        for k in range(local.indptr[j], local.indptr[j + 1]):
            coord = (int(row_map[local.indices[k]]), int(col_map[j]))
            scatter.append(global_map[coord])
    return np.asarray(scatter, dtype=int)


def _transpose_map(matrix):
    matrix = matrix.tocsc()
    transpose = matrix.T.tocsc()
    transpose_lookup = {}
    for j in range(transpose.shape[1]):
        for k in range(transpose.indptr[j], transpose.indptr[j + 1]):
            transpose_lookup[(int(transpose.indices[k]), j)] = k
    out = []
    for j in range(matrix.shape[1]):
        for k in range(matrix.indptr[j], matrix.indptr[j + 1]):
            out.append(transpose_lookup[(j, int(matrix.indices[k]))])
    return np.asarray(out, dtype=int)


def _flat_stage_metadata(problem, inner, terminal):
    T, n, m, td = problem.T, problem.n, problem.m, problem.theta_dim
    x_dim = n * (T + 1) + m * T + td
    y_dim = n * (T + 1) + problem.c_dim * (T + 1)
    s_dim = problem.g_dim * (T + 1)
    inner_H = sp.sparse.triu(_sp_from_sparsity(inner.sparsity_out(4)), format="csc")
    inner_C = _sp_from_sparsity(inner.sparsity_out(5))
    inner_G = _sp_from_sparsity(inner.sparsity_out(6))
    terminal_H = sp.sparse.triu(_sp_from_sparsity(terminal.sparsity_out(4)), format="csc")
    terminal_C = _sp_from_sparsity(terminal.sparsity_out(5))
    terminal_G = _sp_from_sparsity(terminal.sparsity_out(6))

    h_entries = []
    c_entries = [(i, i) for i in range(n)]
    g_entries = []
    for t in range(T):
        col_map = _flat_inner_col_map(problem, t)
        c_row_map = _flat_inner_c_row_map(problem, t)
        g_row_map = _flat_g_row_map(problem, t)
        h_entries.extend(_mapped_entries(inner_H, col_map, col_map))
        c_entries.extend(_mapped_entries(inner_C, c_row_map, col_map))
        g_entries.extend(_mapped_entries(inner_G, g_row_map, col_map))
    terminal_col_map = _flat_terminal_col_map(problem)
    terminal_c_row_map = _flat_terminal_c_row_map(problem)
    terminal_g_row_map = _flat_g_row_map(problem, T)
    h_entries.extend(_mapped_entries(terminal_H, terminal_col_map, terminal_col_map))
    c_entries.extend(_mapped_entries(terminal_C, terminal_c_row_map, terminal_col_map))
    g_entries.extend(_mapped_entries(terminal_G, terminal_g_row_map, terminal_col_map))

    h_raw = _csc_from_entries((x_dim, x_dim), h_entries)
    h_sup = _upper_with_diagonal(h_raw)
    h_full = _full_symmetric_from_upper(h_sup)
    C = _csc_from_entries((y_dim, x_dim), c_entries)
    G = _csc_from_entries((s_dim, x_dim), g_entries)
    CT = C.T.tocsc()
    GT = G.T.tocsc()
    K = _kkt_pattern(h_full, C, G)
    P = _amd_order(K)
    P_inv = np.zeros_like(P)
    P_inv[P] = np.arange(P.size)

    inner_grad_scatter = []
    inner_c_value_scatter = []
    inner_g_value_scatter = []
    inner_h_scatter = []
    inner_c_scatter = []
    inner_g_scatter = []
    for t in range(T):
        col_map = _flat_inner_col_map(problem, t)
        c_row_map = _flat_inner_c_row_map(problem, t)
        g_row_map = _flat_g_row_map(problem, t)
        inner_grad_scatter.extend(col_map.tolist())
        inner_c_value_scatter.extend(c_row_map.tolist())
        inner_g_value_scatter.extend(g_row_map.tolist())
        inner_h_scatter.extend(_scatter_to_global(inner_H, col_map, col_map, h_sup).tolist())
        inner_c_scatter.extend(_scatter_to_global(inner_C, c_row_map, col_map, C).tolist())
        inner_g_scatter.extend(_scatter_to_global(inner_G, g_row_map, col_map, G).tolist())

    terminal_grad_scatter = terminal_col_map
    terminal_c_value_scatter = terminal_c_row_map
    terminal_g_value_scatter = terminal_g_row_map
    terminal_h_scatter = _scatter_to_global(terminal_H, terminal_col_map, terminal_col_map, h_sup)
    terminal_c_scatter = _scatter_to_global(terminal_C, terminal_c_row_map, terminal_col_map, C)
    terminal_g_scatter = _scatter_to_global(terminal_G, terminal_g_row_map, terminal_col_map, G)

    initial_c_scatter = _scatter_to_global(
        sp.sparse.eye(n, format="csc"),
        np.asarray(range(n), dtype=int),
        np.asarray(range(n), dtype=int),
        C,
    )

    return {
        "x_dim": x_dim,
        "y_dim": y_dim,
        "s_dim": s_dim,
        "h_raw": h_raw,
        "h_sup": h_sup,
        "C": C,
        "G": G,
        "CT": CT,
        "GT": GT,
        "P": P,
        "P_inv": P_inv,
        "kkt_dim": K.shape[0],
        "kkt_nnz": h_sup.nnz + C.nnz + G.nnz + C.shape[0] + G.shape[0],
        "kkt_L_nnz": build_sparse_LT(K, P=P).nnz,
        "inner_local_dim": 2 * n + m + td,
        "terminal_local_dim": n + td,
        "inner_c_dim": n + problem.c_dim,
        "inner_h_nnz": inner_H.nnz,
        "inner_c_nnz": inner_C.nnz,
        "inner_g_nnz": inner_G.nnz,
        "terminal_h_nnz": terminal_H.nnz,
        "terminal_c_nnz": terminal_C.nnz,
        "terminal_g_nnz": terminal_G.nnz,
        "initial_c_scatter": initial_c_scatter,
        "inner_grad_scatter": np.asarray(inner_grad_scatter, dtype=int),
        "inner_c_value_scatter": np.asarray(inner_c_value_scatter, dtype=int),
        "inner_g_value_scatter": np.asarray(inner_g_value_scatter, dtype=int),
        "inner_h_scatter": np.asarray(inner_h_scatter, dtype=int),
        "inner_c_scatter": np.asarray(inner_c_scatter, dtype=int),
        "inner_g_scatter": np.asarray(inner_g_scatter, dtype=int),
        "terminal_grad_scatter": np.asarray(terminal_grad_scatter, dtype=int),
        "terminal_c_value_scatter": np.asarray(terminal_c_value_scatter, dtype=int),
        "terminal_g_value_scatter": np.asarray(terminal_g_value_scatter, dtype=int),
        "terminal_h_scatter": terminal_h_scatter,
        "terminal_c_scatter": terminal_c_scatter,
        "terminal_g_scatter": terminal_g_scatter,
        "c_to_ct": _transpose_map(C),
        "g_to_gt": _transpose_map(G),
    }


def _build_ocp_functions(problem):
    n, m, td, c_dim, g_dim = (
        problem.n,
        problem.m,
        problem.theta_dim,
        problem.c_dim,
        problem.g_dim,
    )
    x = ca.SX.sym("x", n)
    u = ca.SX.sym("u", m)
    theta_sym = ca.SX.sym("theta", max(td, 1))
    theta = theta_sym[:td] if td > 0 else ca.SX.zeros(0, 1)
    dyn_mult = ca.SX.sym("dyn_mult", n)
    eq_mult = ca.SX.sym("eq_mult", c_dim) if c_dim > 0 else ca.SX.zeros(0, 1)
    ineq_mult = ca.SX.sym("ineq_mult", g_dim) if g_dim > 0 else ca.SX.zeros(0, 1)

    def stage_outputs(terminal):
        u_arg = ca.SX.zeros(m, 1) if terminal else u
        f = problem.cost(x, u_arg, theta, terminal)
        dyn = problem.dynamics(x, u_arg, theta) if not terminal else ca.SX.zeros(n, 1)
        c = problem.equalities(x, u_arg, theta, terminal) if c_dim > 0 else ca.SX.zeros(0, 1)
        g = problem.inequalities(x, u_arg, theta, terminal) if g_dim > 0 else ca.SX.zeros(0, 1)
        xut = ca.vertcat(x, u_arg, theta) if not terminal else ca.vertcat(x, theta)
        lag = f
        if not terminal:
            lag += ca.dot(dyn_mult, dyn)
        if c_dim > 0:
            lag += ca.dot(eq_mult, c)
        if g_dim > 0:
            lag += ca.dot(ineq_mult, g)
        H, _ = ca.hessian(lag, xut)
        grad_f = ca.gradient(f, xut)
        dyn_jac = ca.jacobian(dyn, xut) if not terminal else ca.SX.zeros(0, xut.numel())
        c_jac = ca.jacobian(c, xut) if c_dim > 0 else ca.SX.zeros(0, xut.numel())
        g_jac = ca.jacobian(g, xut) if g_dim > 0 else ca.SX.zeros(0, xut.numel())
        if not terminal:
            return [
                f,
                grad_f[:n],
                grad_f[n : n + m],
                grad_f[n + m :],
                dyn,
                dyn_jac[:, :n],
                dyn_jac[:, n : n + m],
                dyn_jac[:, n + m :],
                c,
                c_jac[:, :n],
                c_jac[:, n : n + m],
                c_jac[:, n + m :],
                g,
                g_jac[:, :n],
                g_jac[:, n : n + m],
                g_jac[:, n + m :],
                H[:n, :n],
                H[:n, n : n + m],
                H[n : n + m, n : n + m],
                H[:n, n + m :],
                H[n : n + m, n + m :],
                H[n + m :, n + m :],
            ]
        return [
            f,
            grad_f[:n],
            grad_f[n:],
            c,
            c_jac[:, :n],
            c_jac[:, n:],
            g,
            g_jac[:, :n],
            g_jac[:, n:],
            H[:n, :n],
            H[:n, n:],
            H[n:, n:],
        ]

    inner_raw = stage_outputs(False)
    terminal_raw = stage_outputs(True)
    inner = ca.Function(
        "ocp_inner_eval",
        [x, u, theta, dyn_mult, eq_mult, ineq_mult],
        [_dense(v) if not v.is_scalar() else v for v in inner_raw],
    )
    terminal = ca.Function(
        "ocp_terminal_eval",
        [x, theta, eq_mult, ineq_mult],
        [_dense(v) if not v.is_scalar() else v for v in terminal_raw],
    )
    return inner, terminal


def _settings_cpp(problem):
    cpp = f"""
sip::Settings Problem::settings() {{
  auto settings = default_casadi_problem_settings({problem.max_iterations});
"""
    cpp += problem.settings_override_cpp
    cpp += """
  return settings;
}
"""
    return cpp


def _emit_flat_cpp(problem, metadata, out_dir):
    h_sup = metadata["h_sup"]
    C = metadata["C"]
    G = metadata["G"]
    CT = metadata["CT"]
    GT = metadata["GT"]
    P_inv = metadata["P_inv"]
    initial_flat = _flat_initial(problem)

    arrays = ""
    arrays += _c_array("kHInd", h_sup.indices)
    arrays += _c_array("kHIndptr", h_sup.indptr)
    arrays += _c_array("kCInd", C.indices)
    arrays += _c_array("kCIndptr", C.indptr)
    arrays += _c_array("kGInd", G.indices)
    arrays += _c_array("kGIndptr", G.indptr)
    arrays += _c_array("kCTInd", CT.indices)
    arrays += _c_array("kCTIndptr", CT.indptr)
    arrays += _c_array("kGTInd", GT.indices)
    arrays += _c_array("kGTIndptr", GT.indptr)
    arrays += _c_array("kKktPinv", P_inv)
    arrays += _c_array("kInitialCScatter", metadata["initial_c_scatter"])
    arrays += _c_array("kInnerGradScatter", metadata["inner_grad_scatter"])
    arrays += _c_array("kInnerCValueScatter", metadata["inner_c_value_scatter"])
    arrays += _c_array("kInnerGValueScatter", metadata["inner_g_value_scatter"])
    arrays += _c_array("kInnerHScatter", metadata["inner_h_scatter"])
    arrays += _c_array("kInnerCScatter", metadata["inner_c_scatter"])
    arrays += _c_array("kInnerGScatter", metadata["inner_g_scatter"])
    arrays += _c_array("kTerminalGradScatter", metadata["terminal_grad_scatter"])
    arrays += _c_array("kTerminalCValueScatter", metadata["terminal_c_value_scatter"])
    arrays += _c_array("kTerminalGValueScatter", metadata["terminal_g_value_scatter"])
    arrays += _c_array("kTerminalHScatter", metadata["terminal_h_scatter"])
    arrays += _c_array("kTerminalCScatter", metadata["terminal_c_scatter"])
    arrays += _c_array("kTerminalGScatter", metadata["terminal_g_scatter"])
    arrays += _c_array("kCToCT", metadata["c_to_ct"])
    arrays += _c_array("kGToGT", metadata["g_to_gt"])
    arrays += _cpp_double_array("kInitialFlatX", initial_flat)
    arrays += _cpp_double_array("kInitialState", problem.x0)

    header = f"""#pragma once

#include "problem_definitions/casadi_problems/common/problem.hpp"

namespace sip_examples::problem_definitions::casadi_problems::generated_problem {{

struct Problem {{
  static const FlatProblemSpec &flat_spec();
  static sip::Settings settings();
  static CasadiWork make_flat_work();
  static void eval_flat_qdldl(const double *x, const double *y, const double *z,
                              double *f, double *grad_f, double *c, double *g,
                              double *H, double *C, double *G, double *CT,
                              double *GT, CasadiWork &work);
  static void eval_flat_slacg(const double *x, const double *y, const double *z,
                              double *f, double *grad_f, double *c, double *g,
                              double *H, double *C, double *G, double *CT,
                              double *GT, CasadiWork &work);
}};

}} // namespace sip_examples::problem_definitions::casadi_problems::generated_problem
"""
    with open(os.path.join(out_dir, "generated_flat.hpp"), "w") as f:
        f.write(header)

    cpp = f"""
#include "problem_definitions/casadi_problems/{problem.name}/generated_flat.hpp"

#include "problem_definitions/casadi_problems/{problem.name}/generated_flat_casadi.h"

#include <algorithm>

namespace sip_examples::problem_definitions::casadi_problems::generated_problem {{
namespace {{

{arrays}

constexpr int kNumStages = {problem.T};
constexpr int kStateDim = {problem.n};
constexpr int kControlDim = {problem.m};
constexpr int kThetaDim = {problem.theta_dim};
constexpr int kUserEqDim = {problem.c_dim};
constexpr int kIneqDim = {problem.g_dim};
constexpr int kStateBlockDim = kStateDim * (kNumStages + 1);
constexpr int kControlBlockOffset = kStateBlockDim;
constexpr int kThetaOffset = kControlBlockOffset + kControlDim * kNumStages;
constexpr int kInnerLocalDim = {metadata["inner_local_dim"]};
constexpr int kTerminalLocalDim = {metadata["terminal_local_dim"]};
constexpr int kInnerCDim = {metadata["inner_c_dim"]};
constexpr int kInnerHNnz = {metadata["inner_h_nnz"]};
constexpr int kInnerCNnz = {metadata["inner_c_nnz"]};
constexpr int kInnerGNnz = {metadata["inner_g_nnz"]};
constexpr int kTerminalHNnz = {metadata["terminal_h_nnz"]};
constexpr int kTerminalCNnz = {metadata["terminal_c_nnz"]};
constexpr int kTerminalGNnz = {metadata["terminal_g_nnz"]};
constexpr int kMaxGradDim = kInnerLocalDim > kTerminalLocalDim ? kInnerLocalDim : kTerminalLocalDim;
constexpr int kMaxCDim = kInnerCDim > kUserEqDim ? kInnerCDim : kUserEqDim;
constexpr int kMaxHNnz = kInnerHNnz > kTerminalHNnz ? kInnerHNnz : kTerminalHNnz;
constexpr int kMaxCNnz = kInnerCNnz > kTerminalCNnz ? kInnerCNnz : kTerminalCNnz;
constexpr int kMaxGNnz = kInnerGNnz > kTerminalGNnz ? kInnerGNnz : kTerminalGNnz;
constexpr int kFlatIw = flat_inner_eval_SZ_IW > flat_terminal_eval_SZ_IW
                            ? flat_inner_eval_SZ_IW
                            : flat_terminal_eval_SZ_IW;
constexpr int kFlatW = flat_inner_eval_SZ_W > flat_terminal_eval_SZ_W
                           ? flat_inner_eval_SZ_W
                           : flat_terminal_eval_SZ_W;
constexpr int kScratchSize = kMaxGradDim + kMaxCDim + kIneqDim + kMaxHNnz + kMaxCNnz + kMaxGNnz;
constexpr double kDummy = 0.0;

}} // namespace

const FlatProblemSpec &Problem::flat_spec() {{
  static constexpr FlatProblemSpec spec{{
      .name = "{problem.name}",
      .x_dim = {metadata["x_dim"]},
      .s_dim = {metadata["s_dim"]},
      .y_dim = {metadata["y_dim"]},
      .upper_hessian_lagrangian_nnz = {h_sup.nnz},
      .raw_upper_hessian_lagrangian_nnz = {metadata["h_raw"].nnz},
      .jacobian_c_nnz = {C.nnz},
      .jacobian_g_nnz = {G.nnz},
      .jacobian_c_transpose_nnz = {CT.nnz},
      .jacobian_g_transpose_nnz = {GT.nnz},
      .kkt_dim = {metadata["kkt_dim"]},
      .kkt_nnz = {metadata["kkt_nnz"]},
      .kkt_L_nnz = {metadata["kkt_L_nnz"]},
      .upper_hessian_lagrangian_ind = kHInd,
      .upper_hessian_lagrangian_indptr = kHIndptr,
      .jacobian_c_ind = kCInd,
      .jacobian_c_indptr = kCIndptr,
      .jacobian_g_ind = kGInd,
      .jacobian_g_indptr = kGIndptr,
      .jacobian_c_transpose_ind = kCTInd,
      .jacobian_c_transpose_indptr = kCTIndptr,
      .jacobian_g_transpose_ind = kGTInd,
      .jacobian_g_transpose_indptr = kGTIndptr,
      .kkt_pinv = kKktPinv,
      .initial_x = kInitialFlatX,
  }};
  return spec;
}}

"""
    cpp += _settings_cpp(problem)
    cpp += """
CasadiWork Problem::make_flat_work() {
  CasadiWork work;
  work.resize(kFlatIw, kFlatW, kScratchSize);
  return work;
}

void Problem::eval_flat_qdldl(const double *x, const double *y, const double *z,
                              double *f, double *grad_f, double *c, double *g,
                              double *H, double *C, double *G, double *CT,
                              double *GT, CasadiWork &work) {
  eval_flat_slacg(x, y, z, f, grad_f, c, g, H, C, G, CT, GT, work);
}

void Problem::eval_flat_slacg(const double *x, const double *y, const double *z,
                              double *f, double *grad_f, double *c, double *g,
                              double *H, double *C, double *G, double *CT,
                              double *GT, CasadiWork &work) {
  const auto &spec = flat_spec();
  *f = 0.0;
  std::fill_n(grad_f, spec.x_dim, 0.0);
  std::fill_n(c, spec.y_dim, 0.0);
  std::fill_n(g, spec.s_dim, 0.0);
  std::fill_n(H, spec.upper_hessian_lagrangian_nnz, 0.0);
  std::fill_n(C, spec.jacobian_c_nnz, 0.0);
  std::fill_n(G, spec.jacobian_g_nnz, 0.0);

  for (int i = 0; i < kStateDim; ++i) {
    c[i] = x[i] - kInitialState[i];
    C[kInitialCScatter[i]] = 1.0;
  }

  double *local_grad = work.scratch.data();
  double *local_c = local_grad + kMaxGradDim;
  double *local_g = local_c + kMaxCDim;
  double *local_H = local_g + kIneqDim;
  double *local_C = local_H + kMaxHNnz;
  double *local_G = local_C + kMaxCNnz;

  const double *theta = kThetaDim > 0 ? x + kThetaOffset : &kDummy;
  for (int stage = 0; stage < kNumStages; ++stage) {
    double stage_f = 0.0;
    const double *arg[] = {
        x + stage * kStateDim,
        x + (stage + 1) * kStateDim,
        x + kControlBlockOffset + stage * kControlDim,
        theta,
        y + kStateDim + stage * (kStateDim + kUserEqDim),
        kUserEqDim > 0 ? y + kStateDim + stage * (kStateDim + kUserEqDim) + kStateDim : &kDummy,
        kIneqDim > 0 ? z + stage * kIneqDim : &kDummy,
    };
    double *res[] = {
        &stage_f,
        local_grad,
        local_c,
        local_g,
        local_H,
        local_C,
        local_G,
    };
    flat_inner_eval(arg, res, work.iw.data(), work.w.data(), 0);
    *f += stage_f;

    const int grad_offset = stage * kInnerLocalDim;
    for (int i = 0; i < kInnerLocalDim; ++i) {
      grad_f[kInnerGradScatter[grad_offset + i]] += local_grad[i];
    }
    const int c_value_offset = stage * kInnerCDim;
    for (int i = 0; i < kInnerCDim; ++i) {
      c[kInnerCValueScatter[c_value_offset + i]] = local_c[i];
    }
    const int g_value_offset = stage * kIneqDim;
    for (int i = 0; i < kIneqDim; ++i) {
      g[kInnerGValueScatter[g_value_offset + i]] = local_g[i];
    }
    const int h_offset = stage * kInnerHNnz;
    for (int i = 0; i < kInnerHNnz; ++i) {
      H[kInnerHScatter[h_offset + i]] += local_H[i];
    }
    const int c_offset = stage * kInnerCNnz;
    for (int i = 0; i < kInnerCNnz; ++i) {
      C[kInnerCScatter[c_offset + i]] = local_C[i];
    }
    const int g_offset = stage * kInnerGNnz;
    for (int i = 0; i < kInnerGNnz; ++i) {
      G[kInnerGScatter[g_offset + i]] = local_G[i];
    }
  }

  double terminal_f = 0.0;
  const double *terminal_arg[] = {
      x + kNumStages * kStateDim,
      theta,
      kUserEqDim > 0 ? y + kStateDim + kNumStages * (kStateDim + kUserEqDim) : &kDummy,
      kIneqDim > 0 ? z + kNumStages * kIneqDim : &kDummy,
  };
  double *terminal_res[] = {
      &terminal_f,
      local_grad,
      local_c,
      local_g,
      local_H,
      local_C,
      local_G,
  };
  flat_terminal_eval(terminal_arg, terminal_res, work.iw.data(), work.w.data(), 0);
  *f += terminal_f;
  for (int i = 0; i < kTerminalLocalDim; ++i) {
    grad_f[kTerminalGradScatter[i]] += local_grad[i];
  }
  for (int i = 0; i < kUserEqDim; ++i) {
    c[kTerminalCValueScatter[i]] = local_c[i];
  }
  for (int i = 0; i < kIneqDim; ++i) {
    g[kTerminalGValueScatter[i]] = local_g[i];
  }
  for (int i = 0; i < kTerminalHNnz; ++i) {
    H[kTerminalHScatter[i]] += local_H[i];
  }
  for (int i = 0; i < kTerminalCNnz; ++i) {
    C[kTerminalCScatter[i]] = local_C[i];
  }
  for (int i = 0; i < kTerminalGNnz; ++i) {
    G[kTerminalGScatter[i]] = local_G[i];
  }

  for (int i = 0; i < spec.jacobian_c_nnz; ++i) {
    CT[kCToCT[i]] = C[i];
  }
  for (int i = 0; i < spec.jacobian_g_nnz; ++i) {
    GT[kGToGT[i]] = G[i];
  }
}
"""
    cpp += """
} // namespace sip_examples::problem_definitions::casadi_problems::generated_problem
"""
    with open(os.path.join(out_dir, "generated_flat.cpp"), "w") as f:
        f.write(cpp)


def _emit_ocp_cpp(problem, out_dir):
    initial_ocp = _ocp_initial(problem)
    initial_state = np.asarray(problem.x0).reshape(-1)
    arrays = ""
    arrays += _cpp_double_array("kInitialOcpX", initial_ocp)
    arrays += _cpp_double_array("kInitialState", initial_state)

    header = f"""#pragma once

#include "problem_definitions/casadi_problems/common/problem.hpp"
#include "sip_optimal_control/types.hpp"

namespace sip_examples::problem_definitions::casadi_problems::generated_problem {{

struct Problem {{
  static const OcpProblemSpec &ocp_spec();
  static sip::Settings settings();
  static CasadiWork make_ocp_work();
  static void eval_ocp(const ::sip::optimal_control::ModelCallbackInput &mci,
                       ::sip::optimal_control::ModelCallbackOutput &mco,
                       CasadiWork &work);
}};

}} // namespace sip_examples::problem_definitions::casadi_problems::generated_problem
"""
    with open(os.path.join(out_dir, "generated_ocp.hpp"), "w") as f:
        f.write(header)

    cpp = f"""
#include "problem_definitions/casadi_problems/{problem.name}/generated_ocp.hpp"

#include "problem_definitions/casadi_problems/{problem.name}/generated_ocp_casadi.h"

#include <algorithm>

namespace sip_examples::problem_definitions::casadi_problems::generated_problem {{
namespace {{

{arrays}

constexpr int kOcpIw = ocp_inner_eval_SZ_IW > ocp_terminal_eval_SZ_IW
                           ? ocp_inner_eval_SZ_IW
                           : ocp_terminal_eval_SZ_IW;
constexpr int kOcpW = ocp_inner_eval_SZ_W > ocp_terminal_eval_SZ_W
                          ? ocp_inner_eval_SZ_W
                          : ocp_terminal_eval_SZ_W;
constexpr double kDummy = 0.0;

}} // namespace

const OcpProblemSpec &Problem::ocp_spec() {{
  static constexpr OcpProblemSpec spec{{
      .name = "{problem.name}",
      .num_stages = {problem.T},
      .state_dim = {problem.n},
      .control_dim = {problem.m},
      .c_dim = {problem.c_dim},
      .g_dim = {problem.g_dim},
      .theta_dim = {problem.theta_dim},
      .initial_x = kInitialOcpX,
  }};
  return spec;
}}
"""
    cpp += _settings_cpp(problem)
    cpp += """
CasadiWork Problem::make_ocp_work() {
  const auto &spec = ocp_spec();
  CasadiWork work;
  work.resize(kOcpIw, kOcpW, spec.theta_dim + spec.theta_dim * spec.theta_dim);
  return work;
}

void Problem::eval_ocp(const ::sip::optimal_control::ModelCallbackInput &mci,
                       ::sip::optimal_control::ModelCallbackOutput &mco,
                       CasadiWork &work) {
  const auto &spec = ocp_spec();
  mco.f = 0.0;
  std::fill_n(mco.df_dtheta, spec.theta_dim, 0.0);
  std::fill_n(mco.d2L_dtheta2, spec.theta_dim * spec.theta_dim, 0.0);
  double *stage_df_dtheta = spec.theta_dim > 0 ? work.scratch.data() : nullptr;
  double *stage_d2L_dtheta2 =
      spec.theta_dim > 0 ? work.scratch.data() + spec.theta_dim : nullptr;
  for (int j = 0; j < spec.state_dim; ++j) {
    mco.dyn_res[0][j] = kInitialState[j] - mci.states[0][j];
  }
  const double *theta = spec.theta_dim > 0 ? mci.theta : &kDummy;
  for (int i = 0; i < spec.num_stages; ++i) {
    double f = 0.0;
    std::fill_n(mco.df_dx[i], spec.state_dim, 0.0);
    std::fill_n(mco.df_du[i], spec.control_dim, 0.0);
    std::fill_n(mco.dyn_res[i + 1], spec.state_dim, 0.0);
    std::fill_n(mco.ddyn_dx[i], spec.state_dim * spec.state_dim, 0.0);
    std::fill_n(mco.ddyn_du[i], spec.state_dim * spec.control_dim, 0.0);
    if (spec.theta_dim > 0) {
      std::fill_n(stage_df_dtheta, spec.theta_dim, 0.0);
      std::fill_n(mco.ddyn_dtheta[i], spec.state_dim * spec.theta_dim, 0.0);
      std::fill_n(stage_d2L_dtheta2, spec.theta_dim * spec.theta_dim, 0.0);
    }
    if (spec.c_dim > 0) {
      std::fill_n(mco.c[i], spec.c_dim, 0.0);
      std::fill_n(mco.dc_dx[i], spec.c_dim * spec.state_dim, 0.0);
      std::fill_n(mco.dc_du[i], spec.c_dim * spec.control_dim, 0.0);
      if (spec.theta_dim > 0) {
        std::fill_n(mco.dc_dtheta[i], spec.c_dim * spec.theta_dim, 0.0);
      }
    }
    if (spec.g_dim > 0) {
      std::fill_n(mco.g[i], spec.g_dim, 0.0);
      std::fill_n(mco.dg_dx[i], spec.g_dim * spec.state_dim, 0.0);
      std::fill_n(mco.dg_du[i], spec.g_dim * spec.control_dim, 0.0);
      if (spec.theta_dim > 0) {
        std::fill_n(mco.dg_dtheta[i], spec.g_dim * spec.theta_dim, 0.0);
      }
    }
    std::fill_n(mco.d2L_dx2[i], spec.state_dim * spec.state_dim, 0.0);
    std::fill_n(mco.d2L_dxdu[i], spec.state_dim * spec.control_dim, 0.0);
    std::fill_n(mco.d2L_du2[i], spec.control_dim * spec.control_dim, 0.0);
    if (spec.theta_dim > 0) {
      std::fill_n(mco.d2L_dxdtheta[i], spec.state_dim * spec.theta_dim, 0.0);
      std::fill_n(mco.d2L_dudtheta[i], spec.control_dim * spec.theta_dim, 0.0);
    }
    double *res[] = {
        &f,
        mco.df_dx[i],
        mco.df_du[i],
        stage_df_dtheta,
        mco.dyn_res[i + 1],
        mco.ddyn_dx[i],
        mco.ddyn_du[i],
        spec.theta_dim > 0 ? mco.ddyn_dtheta[i] : nullptr,
        spec.c_dim > 0 ? mco.c[i] : nullptr,
        spec.c_dim > 0 ? mco.dc_dx[i] : nullptr,
        spec.c_dim > 0 ? mco.dc_du[i] : nullptr,
        spec.theta_dim > 0 && spec.c_dim > 0 ? mco.dc_dtheta[i] : nullptr,
        spec.g_dim > 0 ? mco.g[i] : nullptr,
        spec.g_dim > 0 ? mco.dg_dx[i] : nullptr,
        spec.g_dim > 0 ? mco.dg_du[i] : nullptr,
        spec.theta_dim > 0 && spec.g_dim > 0 ? mco.dg_dtheta[i] : nullptr,
        mco.d2L_dx2[i],
        mco.d2L_dxdu[i],
        mco.d2L_du2[i],
        spec.theta_dim > 0 ? mco.d2L_dxdtheta[i] : nullptr,
        spec.theta_dim > 0 ? mco.d2L_dudtheta[i] : nullptr,
        stage_d2L_dtheta2,
    };
    const double *arg[] = {
        mci.states[i],
        mci.controls[i],
        theta,
        mci.costates[i + 1],
        spec.c_dim > 0 ? mci.equality_constraint_multipliers[i] : &kDummy,
        spec.g_dim > 0 ? mci.inequality_constraint_multipliers[i] : &kDummy,
    };
    ocp_inner_eval(arg, res, work.iw.data(), work.w.data(), 0);
    mco.f += f;
    for (int j = 0; j < spec.theta_dim; ++j) {
      mco.df_dtheta[j] += stage_df_dtheta[j];
    }
    for (int j = 0; j < spec.theta_dim * spec.theta_dim; ++j) {
      mco.d2L_dtheta2[j] += stage_d2L_dtheta2[j];
    }
    for (int j = 0; j < spec.state_dim; ++j) {
      mco.dyn_res[i + 1][j] -= mci.states[i + 1][j];
    }
  }

  double f = 0.0;
  std::fill_n(mco.df_dx[spec.num_stages], spec.state_dim, 0.0);
  if (spec.theta_dim > 0) {
    std::fill_n(stage_df_dtheta, spec.theta_dim, 0.0);
    std::fill_n(stage_d2L_dtheta2, spec.theta_dim * spec.theta_dim, 0.0);
  }
  if (spec.c_dim > 0) {
    std::fill_n(mco.c[spec.num_stages], spec.c_dim, 0.0);
    std::fill_n(mco.dc_dx[spec.num_stages], spec.c_dim * spec.state_dim, 0.0);
    if (spec.theta_dim > 0) {
      std::fill_n(mco.dc_dtheta[spec.num_stages], spec.c_dim * spec.theta_dim, 0.0);
    }
  }
  if (spec.g_dim > 0) {
    std::fill_n(mco.g[spec.num_stages], spec.g_dim, 0.0);
    std::fill_n(mco.dg_dx[spec.num_stages], spec.g_dim * spec.state_dim, 0.0);
    if (spec.theta_dim > 0) {
      std::fill_n(mco.dg_dtheta[spec.num_stages], spec.g_dim * spec.theta_dim, 0.0);
    }
  }
  std::fill_n(mco.d2L_dx2[spec.num_stages], spec.state_dim * spec.state_dim, 0.0);
  if (spec.theta_dim > 0) {
    std::fill_n(mco.d2L_dxdtheta[spec.num_stages], spec.state_dim * spec.theta_dim, 0.0);
  }
  double *res[] = {
      &f,
      mco.df_dx[spec.num_stages],
      stage_df_dtheta,
      spec.c_dim > 0 ? mco.c[spec.num_stages] : nullptr,
      spec.c_dim > 0 ? mco.dc_dx[spec.num_stages] : nullptr,
      spec.theta_dim > 0 && spec.c_dim > 0 ? mco.dc_dtheta[spec.num_stages] : nullptr,
      spec.g_dim > 0 ? mco.g[spec.num_stages] : nullptr,
      spec.g_dim > 0 ? mco.dg_dx[spec.num_stages] : nullptr,
      spec.theta_dim > 0 && spec.g_dim > 0 ? mco.dg_dtheta[spec.num_stages] : nullptr,
      mco.d2L_dx2[spec.num_stages],
      spec.theta_dim > 0 ? mco.d2L_dxdtheta[spec.num_stages] : nullptr,
      stage_d2L_dtheta2,
  };
  const double *arg[] = {
      mci.states[spec.num_stages],
      theta,
      spec.c_dim > 0 ? mci.equality_constraint_multipliers[spec.num_stages] : &kDummy,
      spec.g_dim > 0 ? mci.inequality_constraint_multipliers[spec.num_stages] : &kDummy,
  };
  ocp_terminal_eval(arg, res, work.iw.data(), work.w.data(), 0);
  mco.f += f;
  for (int j = 0; j < spec.theta_dim; ++j) {
    mco.df_dtheta[j] += stage_df_dtheta[j];
  }
  for (int j = 0; j < spec.theta_dim * spec.theta_dim; ++j) {
    mco.d2L_dtheta2[j] += stage_d2L_dtheta2[j];
  }
}

} // namespace sip_examples::problem_definitions::casadi_problems::generated_problem
"""
    with open(os.path.join(out_dir, "generated_ocp.cpp"), "w") as f:
        f.write(cpp)


def generate_flat(problem, out_dir, emit_kkt_code):
    os.makedirs(out_dir, exist_ok=True)
    flat_inner, flat_terminal = _build_flat_stage_functions(problem)
    metadata = _flat_stage_metadata(problem, flat_inner, flat_terminal)

    old_cwd = os.getcwd()
    try:
        os.chdir(out_dir)
        cg = ca.CodeGenerator("generated_flat_casadi.c", {"with_header": True})
        cg.add(flat_inner)
        cg.add(flat_terminal)
        cg.generate()
    finally:
        os.chdir(old_cwd)

    if emit_kkt_code:
        cpp_header_code, cpp_impl_code = kkt_codegen(
            H=_full_symmetric_from_upper(metadata["h_sup"]),
            C=metadata["C"],
            G=metadata["G"],
            P=metadata["P"],
            namespace="sip_examples::problem_definitions::casadi_problems::generated_problem",
            header_name="kkt_codegen",
        )
        with open(os.path.join(out_dir, "kkt_codegen.hpp"), "w") as f:
            f.write(cpp_header_code)
        with open(os.path.join(out_dir, "kkt_codegen.cpp"), "w") as f:
            f.write(cpp_impl_code)

    _emit_flat_cpp(problem, metadata, out_dir)


def generate_ocp(problem, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    ocp_inner, ocp_terminal = _build_ocp_functions(problem)
    old_cwd = os.getcwd()
    try:
        os.chdir(out_dir)
        cg = ca.CodeGenerator("generated_ocp_casadi.c", {"with_header": True})
        cg.add(ocp_inner)
        cg.add(ocp_terminal)
        cg.generate()
    finally:
        os.chdir(old_cwd)
    _emit_ocp_cpp(problem, out_dir)



def main(problem_factory):
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", required=True, choices=["flat", "ocp"])
    parser.add_argument("--emit-kkt-code", action="store_true")
    parser.add_argument("out_dir")
    args = parser.parse_args()
    problem = problem_factory()
    if args.mode == "flat":
        generate_flat(problem, args.out_dir, args.emit_kkt_code)
    else:
        generate_ocp(problem, args.out_dir)
