import argparse
import os
from dataclasses import dataclass

import casadi as ca
import numpy as np


def _kkt_codegen(*args, **kwargs):
    from slacg.kkt_codegen import kkt_codegen

    return kkt_codegen(*args, **kwargs)


@dataclass(frozen=True)
class CscPattern:
    shape: tuple
    indices: np.ndarray
    indptr: np.ndarray

    @property
    def nnz(self):
        return int(self.indices.size)

    def tocsc(self):
        return self

    @property
    def T(self):
        rows, cols = self.entries()
        return CscPattern.from_entries((self.shape[1], self.shape[0]), zip(cols, rows))

    @classmethod
    def from_entries(cls, shape, entries):
        cols_to_rows = [[] for _ in range(shape[1])]
        for row, col in entries:
            row = int(row)
            col = int(col)
            if row < 0 or row >= shape[0] or col < 0 or col >= shape[1]:
                raise ValueError(f"entry {(row, col)} outside sparse shape {shape}")
            cols_to_rows[col].append(row)
        indices = []
        indptr = [0]
        for rows in cols_to_rows:
            indices.extend(sorted(set(rows)))
            indptr.append(len(indices))
        return cls(
            shape=tuple(shape),
            indices=np.asarray(indices, dtype=int),
            indptr=np.asarray(indptr, dtype=int),
        )

    @classmethod
    def from_sparsity(cls, sparsity, upper_only=False):
        rows = np.asarray(sparsity.row(), dtype=int)
        indptr = np.asarray(sparsity.colind(), dtype=int)
        entries = []
        for col in range(sparsity.shape[1]):
            for k in range(indptr[col], indptr[col + 1]):
                row = int(rows[k])
                if not upper_only or row <= col:
                    entries.append((row, col))
        return cls.from_entries(sparsity.shape, entries)

    @classmethod
    def identity(cls, n):
        return cls.from_entries((n, n), ((i, i) for i in range(n)))

    def entries(self):
        rows = []
        cols = []
        for col in range(self.shape[1]):
            for k in range(self.indptr[col], self.indptr[col + 1]):
                rows.append(int(self.indices[k]))
                cols.append(col)
        return rows, cols


def _to_scipy(pattern):
    import scipy as sp

    data = np.ones(pattern.nnz)
    return sp.sparse.csc_matrix(
        (data, pattern.indices, pattern.indptr), shape=pattern.shape
    )


def _flatten_groups(groups):
    values = []
    offsets = []
    for group in groups:
        offsets.append(len(values))
        values.extend(np.asarray(group).reshape(-1))
    return values, offsets


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


@dataclass(frozen=True)
class GraphEdge:
    parent: int
    child: int
    control_dim: int
    parameters: np.ndarray
    dynamics: object


@dataclass(frozen=True)
class GraphProblemData:
    name: str
    state_dims: list
    edges: list
    theta_dim: int
    c_dims: list
    g_dims: list
    X_init: list
    U_init: list
    theta_init: np.ndarray
    max_iterations: int
    root_residual: object
    cost: object
    equalities: object
    inequalities: object
    settings_override_cpp: str = ""
    state_scales: list = None
    control_scales: list = None
    theta_scales: np.ndarray = None
    objective_scale: float = 1.0

    @property
    def T(self):
        return len(self.edges)

    @property
    def control_dims(self):
        return [edge.control_dim for edge in self.edges]

    @property
    def parameter_dims(self):
        return [np.asarray(edge.parameters).size for edge in self.edges]

    @property
    def root(self):
        children = {edge.child for edge in self.edges}
        roots = [node for node in range(len(self.state_dims)) if node not in children]
        if len(roots) != 1:
            raise ValueError(f"{self.name} must have exactly one root")
        return roots[0]


def _graph_vector_scales(problem, dimensions, specified_scales, name, owners):
    if specified_scales is None:
        return [np.ones(dimension) for dimension in dimensions]
    if len(specified_scales) != len(dimensions):
        raise ValueError(f"{problem.name} must provide one {name} scale per {owners}")

    scales = []
    for index, (dimension, specified_scale) in enumerate(
        zip(dimensions, specified_scales)
    ):
        scale = np.asarray(specified_scale, dtype=float).reshape(-1)
        if scale.size != dimension:
            raise ValueError(
                f"{problem.name} {name} scale {index} has size {scale.size}; "
                f"expected {dimension}"
            )
        if not np.all(np.isfinite(scale)) or np.any(scale <= 0.0):
            raise ValueError(
                f"{problem.name} {name} scale {index} must be finite and positive"
            )
        scales.append(scale)
    return scales


def _graph_scales(problem):
    state_scales = _graph_vector_scales(
        problem, problem.state_dims, problem.state_scales, "state", "graph node"
    )
    control_scales = _graph_vector_scales(
        problem,
        problem.control_dims,
        problem.control_scales,
        "control",
        "graph edge",
    )
    specified_theta_scales = (
        None if problem.theta_scales is None else [problem.theta_scales]
    )
    theta_scale = _graph_vector_scales(
        problem,
        [problem.theta_dim],
        specified_theta_scales,
        "theta",
        "problem",
    )[0]
    return state_scales, control_scales, theta_scale


def _graph_physical_vector(vector, scale):
    return ca.diag(ca.DM(scale)) @ vector


def _graph_scaled_vector(vector, scale):
    return ca.diag(ca.DM(1.0 / scale)) @ vector


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
    return CscPattern.from_sparsity(sparsity)


def _triu_from_sparsity(sparsity):
    return CscPattern.from_sparsity(sparsity, upper_only=True)


def _upper_with_diagonal(raw_upper):
    rows, cols = raw_upper.entries()
    entries = list(zip(rows, cols))
    entries.extend((i, i) for i in range(raw_upper.shape[0]))
    return CscPattern.from_entries(raw_upper.shape, entries)


def _full_symmetric_from_upper(upper):
    rows, cols = upper.entries()
    entries = list(zip(rows, cols))
    entries.extend((col, row) for row, col in zip(rows, cols))
    return CscPattern.from_entries(upper.shape, entries)


def _kkt_pattern(H_full, C, G):
    x_dim = H_full.shape[0]
    y_dim = C.shape[0]
    z_dim = G.shape[0]
    entries = []
    h_rows, h_cols = H_full.entries()
    entries.extend(zip(h_rows, h_cols))
    c_rows, c_cols = C.entries()
    for row, col in zip(c_rows, c_cols):
        entries.append((x_dim + row, col))
        entries.append((col, x_dim + row))
    g_rows, g_cols = G.entries()
    for row, col in zip(g_rows, g_cols):
        entries.append((x_dim + y_dim + row, col))
        entries.append((col, x_dim + y_dim + row))
    entries.extend((x_dim + i, x_dim + i) for i in range(y_dim))
    entries.extend((x_dim + y_dim + i, x_dim + y_dim + i) for i in range(z_dim))
    dim = x_dim + y_dim + z_dim
    return CscPattern.from_entries((dim, dim), entries)


def _amd_order(K):
    rows, cols = K.entries()
    sparsity = ca.Sparsity.triplet(K.shape[0], K.shape[1], rows, cols)
    return np.asarray(sparsity.amd(), dtype=int)


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
    inner_values = ca.Function(
        "flat_inner_values",
        [x, xp, u, theta],
        [f, c, g],
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
    terminal_values = ca.Function(
        "flat_terminal_values",
        [xt, theta],
        [terminal_f, terminal_eq, terminal_g],
    )
    return inner, terminal, inner_values, terminal_values


def _csc_from_entries(shape, entries):
    return CscPattern.from_entries(shape, entries)


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
    cols.extend(
        range(control_offset + t * problem.m, control_offset + (t + 1) * problem.m)
    )
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
    inner_H = _triu_from_sparsity(inner.sparsity_out(4))
    inner_C = _sp_from_sparsity(inner.sparsity_out(5))
    inner_G = _sp_from_sparsity(inner.sparsity_out(6))
    terminal_H = _triu_from_sparsity(terminal.sparsity_out(4))
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
        inner_h_scatter.extend(
            _scatter_to_global(inner_H, col_map, col_map, h_sup).tolist()
        )
        inner_c_scatter.extend(
            _scatter_to_global(inner_C, c_row_map, col_map, C).tolist()
        )
        inner_g_scatter.extend(
            _scatter_to_global(inner_G, g_row_map, col_map, G).tolist()
        )

    terminal_grad_scatter = terminal_col_map
    terminal_c_value_scatter = terminal_c_row_map
    terminal_g_value_scatter = terminal_g_row_map
    terminal_h_scatter = _scatter_to_global(
        terminal_H, terminal_col_map, terminal_col_map, h_sup
    )
    terminal_c_scatter = _scatter_to_global(
        terminal_C, terminal_c_row_map, terminal_col_map, C
    )
    terminal_g_scatter = _scatter_to_global(
        terminal_G, terminal_g_row_map, terminal_col_map, G
    )

    initial_c_scatter = _scatter_to_global(
        CscPattern.identity(n),
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
        "kkt_L_nnz": K.shape[0] * (K.shape[0] + 1) // 2,
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
        c = (
            problem.equalities(x, u_arg, theta, terminal)
            if c_dim > 0
            else ca.SX.zeros(0, 1)
        )
        g = (
            problem.inequalities(x, u_arg, theta, terminal)
            if g_dim > 0
            else ca.SX.zeros(0, 1)
        )
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

    header = """#pragma once

#include "problem_definitions/casadi_problems/common/problem.hpp"

namespace sip_examples::problem_definitions::casadi_problems::generated_problem {

struct Problem {
  static const FlatProblemSpec &flat_spec();
  static sip::Settings settings();
  static CasadiWork make_flat_work();
  static void eval_flat_values(const double *x, double *f, double *c, double *g,
                               CasadiWork &work);
  static void eval_flat_qdldl(const double *x, const double *y, const double *z,
                              double *f, double *grad_f, double *c, double *g,
                              double *H, double *C, double *G, double *CT,
                              double *GT, CasadiWork &work);
  static void eval_flat_slacg(const double *x, const double *y, const double *z,
                              double *f, double *grad_f, double *c, double *g,
                              double *H, double *C, double *G, double *CT,
                              double *GT, CasadiWork &work);
};

} // namespace sip_examples::problem_definitions::casadi_problems::generated_problem
"""
    with open(os.path.join(out_dir, "generated_flat.hpp"), "w") as f:
        f.write(header)

    cpp = f"""
#include "problem_definitions/casadi_problems/{problem.name}/generated_flat.hpp"

#include "problem_definitions/casadi_problems/{problem.name}/generated_flat_casadi.h"
#include "problem_definitions/casadi_problems/{problem.name}/generated_flat_values_casadi.h"

#include <algorithm>

namespace sip_examples::problem_definitions::casadi_problems::generated_problem {{
namespace {{

{arrays}

constexpr int kNumEdges = {problem.T};
constexpr int kStateDim = {problem.n};
constexpr int kControlDim = {problem.m};
constexpr int kThetaDim = {problem.theta_dim};
constexpr int kUserEqDim = {problem.c_dim};
constexpr int kIneqDim = {problem.g_dim};
constexpr int kStateBlockDim = kStateDim * (kNumEdges + 1);
constexpr int kControlBlockOffset = kStateBlockDim;
constexpr int kThetaOffset = kControlBlockOffset + kControlDim * kNumEdges;
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
constexpr int kFlatEvalIw = flat_inner_eval_SZ_IW > flat_terminal_eval_SZ_IW
                                ? flat_inner_eval_SZ_IW
                                : flat_terminal_eval_SZ_IW;
constexpr int kFlatValuesIw =
    flat_inner_values_SZ_IW > flat_terminal_values_SZ_IW
        ? flat_inner_values_SZ_IW
        : flat_terminal_values_SZ_IW;
constexpr int kFlatIw = kFlatEvalIw > kFlatValuesIw ? kFlatEvalIw
                                                     : kFlatValuesIw;
constexpr int kFlatEvalW = flat_inner_eval_SZ_W > flat_terminal_eval_SZ_W
                               ? flat_inner_eval_SZ_W
                               : flat_terminal_eval_SZ_W;
constexpr int kFlatValuesW = flat_inner_values_SZ_W > flat_terminal_values_SZ_W
                                 ? flat_inner_values_SZ_W
                                 : flat_terminal_values_SZ_W;
constexpr int kFlatW = kFlatEvalW > kFlatValuesW ? kFlatEvalW : kFlatValuesW;
constexpr int kFlatEvalArg = flat_inner_eval_SZ_ARG > flat_terminal_eval_SZ_ARG
                                 ? flat_inner_eval_SZ_ARG
                                 : flat_terminal_eval_SZ_ARG;
constexpr int kFlatValuesArg =
    flat_inner_values_SZ_ARG > flat_terminal_values_SZ_ARG
        ? flat_inner_values_SZ_ARG
        : flat_terminal_values_SZ_ARG;
constexpr int kFlatArg = kFlatEvalArg > kFlatValuesArg ? kFlatEvalArg
                                                        : kFlatValuesArg;
constexpr int kFlatEvalRes = flat_inner_eval_SZ_RES > flat_terminal_eval_SZ_RES
                                 ? flat_inner_eval_SZ_RES
                                 : flat_terminal_eval_SZ_RES;
constexpr int kFlatValuesRes =
    flat_inner_values_SZ_RES > flat_terminal_values_SZ_RES
        ? flat_inner_values_SZ_RES
        : flat_terminal_values_SZ_RES;
constexpr int kFlatRes = kFlatEvalRes > kFlatValuesRes ? kFlatEvalRes
                                                        : kFlatValuesRes;
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

void Problem::eval_flat_values(const double *x, double *f, double *c, double *g,
                               CasadiWork &work) {
  const auto &spec = flat_spec();
  *f = 0.0;
  std::fill_n(c, spec.y_dim, 0.0);
  std::fill_n(g, spec.s_dim, 0.0);

  for (int i = 0; i < kStateDim; ++i) {
    c[i] = x[i] - kInitialState[i];
  }

  double *local_c = work.scratch.data();
  double *local_g = local_c + kMaxCDim;
  const double *theta = kThetaDim > 0 ? x + kThetaOffset : &kDummy;
  for (int stage = 0; stage < kNumEdges; ++stage) {
    double stage_f = 0.0;
    const double *arg[kFlatArg] = {
        x + stage * kStateDim,
        x + (stage + 1) * kStateDim,
        x + kControlBlockOffset + stage * kControlDim,
        theta,
    };
    double *res[kFlatRes] = {&stage_f, local_c, local_g};
    flat_inner_values(arg, res, work.iw.data(), work.w.data(), 0);
    *f += stage_f;

    const int c_value_offset = stage * kInnerCDim;
    for (int i = 0; i < kInnerCDim; ++i) {
      c[kInnerCValueScatter[c_value_offset + i]] = local_c[i];
    }
    const int g_value_offset = stage * kIneqDim;
    for (int i = 0; i < kIneqDim; ++i) {
      g[kInnerGValueScatter[g_value_offset + i]] = local_g[i];
    }
  }

  double terminal_f = 0.0;
  const double *terminal_arg[kFlatArg] = {
      x + kNumEdges * kStateDim,
      theta,
  };
  double *terminal_res[kFlatRes] = {&terminal_f, local_c, local_g};
  flat_terminal_values(terminal_arg, terminal_res, work.iw.data(), work.w.data(),
                       0);
  *f += terminal_f;
  for (int i = 0; i < kUserEqDim; ++i) {
    c[kTerminalCValueScatter[i]] = local_c[i];
  }
  for (int i = 0; i < kIneqDim; ++i) {
    g[kTerminalGValueScatter[i]] = local_g[i];
  }
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
  for (int stage = 0; stage < kNumEdges; ++stage) {
    double stage_f = 0.0;
    const double *arg[kFlatArg] = {
        x + stage * kStateDim,
        x + (stage + 1) * kStateDim,
        x + kControlBlockOffset + stage * kControlDim,
        theta,
        y + kStateDim + stage * (kStateDim + kUserEqDim),
        kUserEqDim > 0 ? y + kStateDim + stage * (kStateDim + kUserEqDim) + kStateDim : &kDummy,
        kIneqDim > 0 ? z + stage * kIneqDim : &kDummy,
    };
    double *res[kFlatRes] = {
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
  const double *terminal_arg[kFlatArg] = {
      x + kNumEdges * kStateDim,
      theta,
      kUserEqDim > 0 ? y + kStateDim + kNumEdges * (kStateDim + kUserEqDim) : &kDummy,
      kIneqDim > 0 ? z + kNumEdges * kIneqDim : &kDummy,
  };
  double *terminal_res[kFlatRes] = {
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

    header = """#pragma once

#include "problem_definitions/casadi_problems/common/ocp_problem.hpp"
#include "problem_definitions/casadi_problems/common/problem.hpp"
#include "sip_optimal_control/types.hpp"

namespace sip_examples::problem_definitions::casadi_problems::generated_problem {

struct Problem {
  static const OcpProblemSpec &ocp_spec();
  static sip::Settings settings();
  static CasadiWork make_ocp_work();
  static void eval_ocp(const ::sip::optimal_control::ModelCallbackInput &mci,
                       ::sip::optimal_control::ModelCallbackOutput &mco,
                       CasadiWork &work);
};

} // namespace sip_examples::problem_definitions::casadi_problems::generated_problem
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

constexpr int kStateDims[] = {{{", ".join([str(problem.n)] * (problem.T + 1))}}};
constexpr int kControlDims[] = {{{", ".join([str(problem.m)] * problem.T)}}};
constexpr int kCDims[] = {{{", ".join([str(problem.c_dim)] * (problem.T + 1))}}};
constexpr int kGDims[] = {{{", ".join([str(problem.g_dim)] * (problem.T + 1))}}};
constexpr int kEdgeParents[] = {{{", ".join(str(i) for i in range(problem.T))}}};
constexpr int kEdgeChildren[] = {{{", ".join(str(i + 1) for i in range(problem.T))}}};

constexpr int kOcpIw = ocp_inner_eval_SZ_IW > ocp_terminal_eval_SZ_IW
                           ? ocp_inner_eval_SZ_IW
                           : ocp_terminal_eval_SZ_IW;
constexpr int kOcpW = ocp_inner_eval_SZ_W > ocp_terminal_eval_SZ_W
                          ? ocp_inner_eval_SZ_W
                          : ocp_terminal_eval_SZ_W;
constexpr int kOcpArg = ocp_inner_eval_SZ_ARG > ocp_terminal_eval_SZ_ARG
                            ? ocp_inner_eval_SZ_ARG
                            : ocp_terminal_eval_SZ_ARG;
constexpr int kOcpRes = ocp_inner_eval_SZ_RES > ocp_terminal_eval_SZ_RES
                            ? ocp_inner_eval_SZ_RES
                            : ocp_terminal_eval_SZ_RES;
constexpr double kDummy = 0.0;

}} // namespace

const OcpProblemSpec &Problem::ocp_spec() {{
  static constexpr OcpProblemSpec spec{{
      .name = "{problem.name}",
      .num_edges = {problem.T},
      .dimensions = ::sip::optimal_control::Dimensions{{
          {problem.theta_dim}, kStateDims, kControlDims, kCDims, kGDims}},
      .topology_root = 0,
      .edge_parents = kEdgeParents,
      .edge_children = kEdgeChildren,
      .initial_x = kInitialOcpX,
  }};
  return spec;
}}
"""
    cpp += _settings_cpp(problem)
    cpp += """
CasadiWork Problem::make_ocp_work() {
  const auto &spec = ocp_spec();
  const int theta_dim = spec.dimensions.get_schur_dim();
  CasadiWork work;
  work.resize(kOcpIw, kOcpW, theta_dim + theta_dim * theta_dim);
  return work;
}

void Problem::eval_ocp(const ::sip::optimal_control::ModelCallbackInput &mci,
                       ::sip::optimal_control::ModelCallbackOutput &mco,
                       CasadiWork &work) {
  const auto &spec = ocp_spec();
  const auto &dimensions = spec.dimensions;
  const int num_edges = spec.num_edges;
  const int state_dim = dimensions.get_state_dim(0);
  const int control_dim = dimensions.get_control_dim(0);
  const int c_dim = dimensions.get_c_dim(0);
  const int g_dim = dimensions.get_g_dim(0);
  const int theta_dim = dimensions.get_schur_dim();
  mco.f = 0.0;
  std::fill_n(mco.df_dtheta, theta_dim, 0.0);
  std::fill_n(mco.d2L_dtheta2, theta_dim * theta_dim, 0.0);
  double *stage_df_dtheta = theta_dim > 0 ? work.scratch.data() : nullptr;
  double *stage_d2L_dtheta2 =
      theta_dim > 0 ? work.scratch.data() + theta_dim : nullptr;
  for (int j = 0; j < state_dim; ++j) {
    mco.dyn_res[0][j] = kInitialState[j] - mci.states[0][j];
  }
  const double *theta = theta_dim > 0 ? mci.theta : &kDummy;
  for (int i = 0; i < num_edges; ++i) {
    double f = 0.0;
    std::fill_n(mco.df_dx[i], state_dim, 0.0);
    std::fill_n(mco.df_du[i], control_dim, 0.0);
    std::fill_n(mco.dyn_res[i + 1], state_dim, 0.0);
    std::fill_n(mco.ddyn_dx[i], state_dim * state_dim, 0.0);
    std::fill_n(mco.ddyn_du[i], state_dim * control_dim, 0.0);
    if (theta_dim > 0) {
      std::fill_n(stage_df_dtheta, theta_dim, 0.0);
      std::fill_n(mco.ddyn_dtheta[i], state_dim * theta_dim, 0.0);
      std::fill_n(stage_d2L_dtheta2, theta_dim * theta_dim, 0.0);
    }
    if (c_dim > 0) {
      std::fill_n(mco.c[i], c_dim, 0.0);
      std::fill_n(mco.dc_dx[i], c_dim * state_dim, 0.0);
      std::fill_n(mco.dc_du[i], c_dim * control_dim, 0.0);
      if (theta_dim > 0) {
        std::fill_n(mco.dc_dtheta[i], c_dim * theta_dim, 0.0);
      }
    }
    if (g_dim > 0) {
      std::fill_n(mco.g[i], g_dim, 0.0);
      std::fill_n(mco.dg_dx[i], g_dim * state_dim, 0.0);
      std::fill_n(mco.dg_du[i], g_dim * control_dim, 0.0);
      if (theta_dim > 0) {
        std::fill_n(mco.dg_dtheta[i], g_dim * theta_dim, 0.0);
      }
    }
    std::fill_n(mco.d2L_dx2[i], state_dim * state_dim, 0.0);
    std::fill_n(mco.d2L_dxdu[i], state_dim * control_dim, 0.0);
    std::fill_n(mco.d2L_du2[i], control_dim * control_dim, 0.0);
    if (theta_dim > 0) {
      std::fill_n(mco.d2L_dxdtheta[i], state_dim * theta_dim, 0.0);
      std::fill_n(mco.d2L_dudtheta[i], control_dim * theta_dim, 0.0);
    }
    double *res[kOcpRes] = {
        &f,
        mco.df_dx[i],
        mco.df_du[i],
        stage_df_dtheta,
        mco.dyn_res[i + 1],
        mco.ddyn_dx[i],
        mco.ddyn_du[i],
        theta_dim > 0 ? mco.ddyn_dtheta[i] : nullptr,
        c_dim > 0 ? mco.c[i] : nullptr,
        c_dim > 0 ? mco.dc_dx[i] : nullptr,
        c_dim > 0 ? mco.dc_du[i] : nullptr,
        theta_dim > 0 && c_dim > 0 ? mco.dc_dtheta[i] : nullptr,
        g_dim > 0 ? mco.g[i] : nullptr,
        g_dim > 0 ? mco.dg_dx[i] : nullptr,
        g_dim > 0 ? mco.dg_du[i] : nullptr,
        theta_dim > 0 && g_dim > 0 ? mco.dg_dtheta[i] : nullptr,
        mco.d2L_dx2[i],
        mco.d2L_dxdu[i],
        mco.d2L_du2[i],
        theta_dim > 0 ? mco.d2L_dxdtheta[i] : nullptr,
        theta_dim > 0 ? mco.d2L_dudtheta[i] : nullptr,
        stage_d2L_dtheta2,
    };
    const double *arg[kOcpArg] = {
        mci.states[i],
        mci.controls[i],
        theta,
        mci.costates[i + 1],
        c_dim > 0 ? mci.equality_constraint_multipliers[i] : &kDummy,
        g_dim > 0 ? mci.inequality_constraint_multipliers[i] : &kDummy,
    };
    ocp_inner_eval(arg, res, work.iw.data(), work.w.data(), 0);
    mco.f += f;
    for (int j = 0; j < theta_dim; ++j) {
      mco.df_dtheta[j] += stage_df_dtheta[j];
    }
    for (int j = 0; j < theta_dim * theta_dim; ++j) {
      mco.d2L_dtheta2[j] += stage_d2L_dtheta2[j];
    }
    for (int j = 0; j < state_dim; ++j) {
      mco.dyn_res[i + 1][j] -= mci.states[i + 1][j];
    }
  }

  double f = 0.0;
  std::fill_n(mco.df_dx[num_edges], state_dim, 0.0);
  if (theta_dim > 0) {
    std::fill_n(stage_df_dtheta, theta_dim, 0.0);
    std::fill_n(stage_d2L_dtheta2, theta_dim * theta_dim, 0.0);
  }
  if (c_dim > 0) {
    std::fill_n(mco.c[num_edges], c_dim, 0.0);
    std::fill_n(mco.dc_dx[num_edges], c_dim * state_dim, 0.0);
    if (theta_dim > 0) {
      std::fill_n(mco.dc_dtheta[num_edges], c_dim * theta_dim, 0.0);
    }
  }
  if (g_dim > 0) {
    std::fill_n(mco.g[num_edges], g_dim, 0.0);
    std::fill_n(mco.dg_dx[num_edges], g_dim * state_dim, 0.0);
    if (theta_dim > 0) {
      std::fill_n(mco.dg_dtheta[num_edges], g_dim * theta_dim, 0.0);
    }
  }
  std::fill_n(mco.d2L_dx2[num_edges], state_dim * state_dim, 0.0);
  if (theta_dim > 0) {
    std::fill_n(mco.d2L_dxdtheta[num_edges], state_dim * theta_dim, 0.0);
  }
  double *res[kOcpRes] = {
      &f,
      mco.df_dx[num_edges],
      stage_df_dtheta,
      c_dim > 0 ? mco.c[num_edges] : nullptr,
      c_dim > 0 ? mco.dc_dx[num_edges] : nullptr,
      theta_dim > 0 && c_dim > 0 ? mco.dc_dtheta[num_edges] : nullptr,
      g_dim > 0 ? mco.g[num_edges] : nullptr,
      g_dim > 0 ? mco.dg_dx[num_edges] : nullptr,
      theta_dim > 0 && g_dim > 0 ? mco.dg_dtheta[num_edges] : nullptr,
      mco.d2L_dx2[num_edges],
      theta_dim > 0 ? mco.d2L_dxdtheta[num_edges] : nullptr,
      stage_d2L_dtheta2,
  };
  const double *arg[kOcpArg] = {
      mci.states[num_edges],
      theta,
      c_dim > 0 ? mci.equality_constraint_multipliers[num_edges] : &kDummy,
      g_dim > 0 ? mci.inequality_constraint_multipliers[num_edges] : &kDummy,
  };
  ocp_terminal_eval(arg, res, work.iw.data(), work.w.data(), 0);
  mco.f += f;
  for (int j = 0; j < theta_dim; ++j) {
    mco.df_dtheta[j] += stage_df_dtheta[j];
  }
  for (int j = 0; j < theta_dim * theta_dim; ++j) {
    mco.d2L_dtheta2[j] += stage_d2L_dtheta2[j];
  }
}

} // namespace sip_examples::problem_definitions::casadi_problems::generated_problem
"""
    with open(os.path.join(out_dir, "generated_ocp.cpp"), "w") as f:
        f.write(cpp)


def generate_flat(problem, out_dir, emit_kkt_code):
    os.makedirs(out_dir, exist_ok=True)
    flat_inner, flat_terminal, flat_inner_values, flat_terminal_values = (
        _build_flat_stage_functions(problem)
    )
    metadata = _flat_stage_metadata(problem, flat_inner, flat_terminal)

    old_cwd = os.getcwd()
    try:
        os.chdir(out_dir)
        cg = ca.CodeGenerator("generated_flat_casadi.c", {"with_header": True})
        cg.add(flat_inner)
        cg.add(flat_terminal)
        cg.generate()
        values_cg = ca.CodeGenerator(
            "generated_flat_values_casadi.c", {"with_header": True}
        )
        values_cg.add(flat_inner_values)
        values_cg.add(flat_terminal_values)
        values_cg.generate()
    finally:
        os.chdir(old_cwd)

    if emit_kkt_code:
        cpp_header_code, cpp_impl_code = _kkt_codegen(
            H=_to_scipy(_full_symmetric_from_upper(metadata["h_sup"])),
            C=_to_scipy(metadata["C"]),
            G=_to_scipy(metadata["G"]),
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


def _graph_offsets(problem):
    state_offsets = []
    control_offsets = []
    offset = 0
    for edge in range(problem.T):
        state_offsets.append(offset)
        offset += problem.state_dims[edge]
        control_offsets.append(offset)
        offset += problem.control_dims[edge]
    state_offsets.append(offset)
    offset += problem.state_dims[problem.T]
    theta_offset = offset
    x_dim = theta_offset + problem.theta_dim
    return state_offsets, control_offsets, theta_offset, x_dim


def _graph_y_offsets(problem):
    dyn_offsets = []
    c_offsets = []
    offset = 0
    for node, n in enumerate(problem.state_dims):
        dyn_offsets.append(offset)
        offset += n
        c_offsets.append(offset)
        offset += problem.c_dims[node]
    return dyn_offsets, c_offsets, offset


def _graph_z_offsets(problem):
    offsets = []
    offset = 0
    for g_dim in problem.g_dims:
        offsets.append(offset)
        offset += g_dim
    return offsets, offset


def _graph_initial(problem):
    state_scales, control_scales, theta_scale = _graph_scales(problem)
    pieces = []
    for edge in range(problem.T):
        pieces.append(np.asarray(problem.X_init[edge]) / state_scales[edge])
        pieces.append(np.asarray(problem.U_init[edge]) / control_scales[edge])
    pieces.append(np.asarray(problem.X_init[problem.T]) / state_scales[problem.T])
    pieces.append(np.asarray(problem.theta_init) / theta_scale)
    return np.concatenate([np.asarray(p).reshape(-1) for p in pieces])


def _graph_connectivity(problem):
    incoming = {edge.child: edge_index for edge_index, edge in enumerate(problem.edges)}
    outgoing = {node: [] for node in range(problem.T + 1)}
    for edge_index, edge in enumerate(problem.edges):
        outgoing[edge.parent].append(edge_index)
    return incoming, outgoing


def _graph_edge_slices(values, edge_indices, dimensions):
    pieces = []
    offset = 0
    for edge_index in edge_indices:
        dim = dimensions[edge_index]
        pieces.append(values[offset : offset + dim])
        offset += dim
    return pieces


def _build_graph_flat_stage_functions(
    problem, prefix="graph_flat", build_value_functions=True
):
    _, outgoing = _graph_connectivity(problem)
    state_scales, control_scales, theta_scale = _graph_scales(problem)
    objective_scale = float(problem.objective_scale)
    if not np.isfinite(objective_scale) or objective_scale <= 0.0:
        raise ValueError(f"{problem.name} objective scale must be finite and positive")
    td = problem.theta_dim
    theta_sym = ca.SX.sym("theta", max(td, 1))
    theta_variable = theta_sym[:td] if td > 0 else ca.SX.zeros(0, 1)
    theta = _graph_physical_vector(theta_variable, theta_scale)

    root = problem.root
    root_n = problem.state_dims[root]
    root_x = ca.SX.sym("root_x", root_n)
    root_mult = ca.SX.sym("root_mult", root_n)
    root_physical_x = _graph_physical_vector(root_x, state_scales[root])
    root_dyn = _graph_scaled_vector(
        problem.root_residual(root_physical_x, theta), state_scales[root]
    )
    root_xloc = ca.vertcat(root_x, theta_variable)
    root_lag = ca.dot(root_mult, root_dyn)
    root_H, _ = ca.hessian(root_lag, root_xloc)
    root_H = ca.triu(root_H)
    root_C = (
        ca.jacobian(root_dyn, root_xloc)
        if root_n > 0
        else ca.SX.zeros(0, root_xloc.numel())
    )
    root_fun = ca.Function(
        f"{prefix}_root_eval",
        [root_x, theta_variable, root_mult],
        ca.cse([root_dyn, root_H, root_C]),
    )
    root_values = None
    if build_value_functions:
        root_values = ca.Function(
            f"{prefix}_root_values",
            [root_x, theta_variable],
            [root_dyn],
        )

    edge_funs = []
    edge_value_funs = []
    edge_function_cache = {}
    for edge_index, edge in enumerate(problem.edges):
        parent_n = problem.state_dims[edge.parent]
        child_n = problem.state_dims[edge.child]
        control_dim = edge.control_dim
        parameter_dim = problem.parameter_dims[edge_index]
        cache_key = (
            id(edge.dynamics),
            parent_n,
            child_n,
            control_dim,
            parameter_dim,
            tuple(state_scales[edge.parent]),
            tuple(state_scales[edge.child]),
            tuple(control_scales[edge_index]),
            tuple(theta_scale),
        )
        if cache_key in edge_function_cache:
            edge_fun, edge_value_fun = edge_function_cache[cache_key]
            edge_funs.append(edge_fun)
            if build_value_functions:
                edge_value_funs.append(edge_value_fun)
            continue

        parent_x = ca.SX.sym("edge_parent_x", parent_n)
        child_x = ca.SX.sym("edge_child_x", child_n)
        control_sym = ca.SX.sym("edge_u", max(control_dim, 1))
        control_variable = (
            control_sym[:control_dim] if control_dim > 0 else ca.SX.zeros(0, 1)
        )
        parameter_sym = ca.SX.sym("edge_parameters", max(parameter_dim, 1))
        parameters = (
            parameter_sym[:parameter_dim] if parameter_dim > 0 else ca.SX.zeros(0, 1)
        )
        control = _graph_physical_vector(control_variable, control_scales[edge_index])
        dyn_mult = ca.SX.sym("edge_dyn_mult", child_n)
        parent_physical_x = _graph_physical_vector(parent_x, state_scales[edge.parent])
        next_physical_x = edge.dynamics(parent_physical_x, control, theta, parameters)
        dyn = _graph_scaled_vector(next_physical_x, state_scales[edge.child]) - child_x
        xloc = ca.vertcat(parent_x, child_x, control_variable, theta_variable)
        lag = ca.dot(dyn_mult, dyn)
        H, _ = ca.hessian(lag, xloc)
        H = ca.triu(H)
        C = ca.jacobian(dyn, xloc) if child_n > 0 else ca.SX.zeros(0, xloc.numel())
        model_index = len(edge_function_cache)
        edge_fun = ca.Function(
            f"{prefix}_edge_model_{model_index}_eval",
            [
                parent_x,
                child_x,
                control_variable,
                theta_variable,
                parameters,
                dyn_mult,
            ],
            ca.cse([dyn, H, C]),
        )
        edge_funs.append(edge_fun)
        edge_value_fun = None
        if build_value_functions:
            edge_value_fun = ca.Function(
                f"{prefix}_edge_model_{model_index}_values",
                [
                    parent_x,
                    child_x,
                    control_variable,
                    theta_variable,
                    parameters,
                ],
                [dyn],
            )
            edge_value_funs.append(edge_value_fun)
        edge_function_cache[cache_key] = (edge_fun, edge_value_fun)

    node_funs = []
    node_value_funs = []
    node_function_cache = {}
    for node in range(problem.T + 1):
        state_dim = problem.state_dims[node]
        out_edges = outgoing[node]
        out_control_dim = sum(problem.control_dims[edge] for edge in out_edges)
        out_parameter_dim = sum(problem.parameter_dims[edge] for edge in out_edges)
        c_dim = problem.c_dims[node]
        g_dim = problem.g_dims[node]
        x_node = ca.SX.sym("node_x", state_dim)
        physical_x_node = _graph_physical_vector(x_node, state_scales[node])
        u_out_sym = ca.SX.sym("node_u_out", max(out_control_dim, 1))
        u_out_variable = (
            u_out_sym[:out_control_dim] if out_control_dim > 0 else ca.SX.zeros(0, 1)
        )
        outgoing_control_variables = _graph_edge_slices(
            u_out_variable, out_edges, problem.control_dims
        )
        p_out_sym = ca.SX.sym("node_p_out", max(out_parameter_dim, 1))
        p_out_variable = (
            p_out_sym[:out_parameter_dim]
            if out_parameter_dim > 0
            else ca.SX.zeros(0, 1)
        )
        outgoing_parameters = _graph_edge_slices(
            p_out_variable, out_edges, problem.parameter_dims
        )
        outgoing_controls = [
            _graph_physical_vector(control, control_scales[edge_index])
            for edge_index, control in zip(out_edges, outgoing_control_variables)
        ]
        eq_mult = ca.SX.sym("node_eq_mult", c_dim) if c_dim > 0 else ca.SX.zeros(0, 1)
        ineq_mult = (
            ca.SX.sym("node_ineq_mult", g_dim) if g_dim > 0 else ca.SX.zeros(0, 1)
        )
        f = problem.cost(node, physical_x_node, theta) / objective_scale
        eq = (
            problem.equalities(
                node,
                physical_x_node,
                theta,
                outgoing_controls,
                outgoing_parameters,
            )
            if c_dim > 0
            else ca.SX.zeros(0, 1)
        )
        g = (
            problem.inequalities(
                node,
                physical_x_node,
                theta,
                outgoing_controls,
                outgoing_parameters,
            )
            if g_dim > 0
            else ca.SX.zeros(0, 1)
        )
        xloc = ca.vertcat(x_node, u_out_variable, theta_variable)
        lag = f
        if c_dim > 0:
            lag += ca.dot(eq_mult, eq)
        if g_dim > 0:
            lag += ca.dot(ineq_mult, g)
        H, _ = ca.hessian(lag, xloc)
        H = ca.triu(H)
        C = ca.jacobian(eq, xloc) if c_dim > 0 else ca.SX.zeros(0, xloc.numel())
        G = ca.jacobian(g, xloc) if g_dim > 0 else ca.SX.zeros(0, xloc.numel())
        outputs = ca.cse([f, ca.gradient(f, xloc), eq, g, H, C, G])
        cache_key = (
            state_dim,
            out_control_dim,
            out_parameter_dim,
            c_dim,
            g_dim,
            problem.theta_dim,
            *(str(output) for output in outputs),
        )
        if cache_key in node_function_cache:
            node_fun, node_value_fun = node_function_cache[cache_key]
            node_funs.append(node_fun)
            if build_value_functions:
                node_value_funs.append(node_value_fun)
            continue

        model_index = len(node_function_cache)
        node_fun = ca.Function(
            f"{prefix}_node_model_{model_index}_eval",
            [
                x_node,
                theta_variable,
                u_out_variable,
                p_out_variable,
                eq_mult,
                ineq_mult,
            ],
            outputs,
        )
        node_funs.append(node_fun)
        node_value_fun = None
        if build_value_functions:
            node_value_fun = ca.Function(
                f"{prefix}_node_model_{model_index}_values",
                [x_node, theta_variable, u_out_variable, p_out_variable],
                [f, eq, g],
            )
            node_value_funs.append(node_value_fun)
        node_function_cache[cache_key] = (node_fun, node_value_fun)

    return (
        root_fun,
        edge_funs,
        node_funs,
        root_values,
        edge_value_funs,
        node_value_funs,
        outgoing,
    )


def _graph_root_col_map(problem):
    state_offsets, _, theta_offset, _ = _graph_offsets(problem)
    root = problem.root
    cols = list(
        range(state_offsets[root], state_offsets[root] + problem.state_dims[root])
    )
    cols.extend(range(theta_offset, theta_offset + problem.theta_dim))
    return np.asarray(cols, dtype=int)


def _graph_edge_col_map(problem, edge_index):
    state_offsets, control_offsets, theta_offset, _ = _graph_offsets(problem)
    edge = problem.edges[edge_index]
    cols = list(
        range(
            state_offsets[edge.parent],
            state_offsets[edge.parent] + problem.state_dims[edge.parent],
        )
    )
    cols.extend(
        range(
            state_offsets[edge.child],
            state_offsets[edge.child] + problem.state_dims[edge.child],
        )
    )
    cols.extend(
        range(
            control_offsets[edge_index], control_offsets[edge_index] + edge.control_dim
        )
    )
    cols.extend(range(theta_offset, theta_offset + problem.theta_dim))
    return np.asarray(cols, dtype=int)


def _graph_node_col_map(problem, node, outgoing):
    state_offsets, control_offsets, theta_offset, _ = _graph_offsets(problem)
    cols = list(
        range(state_offsets[node], state_offsets[node] + problem.state_dims[node])
    )
    for edge_index in outgoing[node]:
        cols.extend(
            range(
                control_offsets[edge_index],
                control_offsets[edge_index] + problem.control_dims[edge_index],
            )
        )
    cols.extend(range(theta_offset, theta_offset + problem.theta_dim))
    return np.asarray(cols, dtype=int)


def _graph_local_metadata_summary(
    problem, root_meta, edge_meta, node_meta, outgoing, functions
):
    outgoing_control_dims = [
        sum(problem.control_dims[edge] for edge in outgoing[node])
        for node in range(problem.T + 1)
    ]
    outgoing_parameter_dims = [
        sum(problem.parameter_dims[edge] for edge in outgoing[node])
        for node in range(problem.T + 1)
    ]
    return {
        "root": root_meta,
        "edges": edge_meta,
        "nodes": node_meta,
        "outgoing": outgoing,
        "max_node_out_control_dim": max(outgoing_control_dims + [0]),
        "max_node_out_parameter_dim": max(outgoing_parameter_dims + [0]),
        "max_grad_dim": max([item["local_dim"] for item in node_meta] + [0]),
        "max_c_dim": max(problem.state_dims + problem.c_dims + [0]),
        "max_g_dim": max(problem.g_dims + [0]),
        "max_h_nnz": max(
            [root_meta["H"].nnz]
            + [item["H"].nnz for item in edge_meta]
            + [item["H"].nnz for item in node_meta]
            + [0]
        ),
        "max_c_nnz": max(
            [root_meta["C"].nnz]
            + [item["C"].nnz for item in edge_meta]
            + [item["C"].nnz for item in node_meta]
            + [0]
        ),
        "max_g_nnz": max([item["G"].nnz for item in node_meta] + [0]),
        "max_iw": max(fun.sz_iw() for fun in functions),
        "max_w": max(fun.sz_w() for fun in functions),
        "max_arg": max(fun.sz_arg() for fun in functions),
        "max_res": max(fun.sz_res() for fun in functions),
    }


def _graph_flat_stage_metadata(
    problem,
    root_fun,
    edge_funs,
    node_funs,
    root_values,
    edge_value_funs,
    node_value_funs,
    outgoing,
):
    _, _, _, x_dim = _graph_offsets(problem)
    dyn_offsets, c_offsets, y_dim = _graph_y_offsets(problem)
    z_offsets, z_dim = _graph_z_offsets(problem)

    h_entries = []
    c_entries = []
    g_entries = []

    root_cols = _graph_root_col_map(problem)
    root_rows = np.asarray(
        range(
            dyn_offsets[problem.root],
            dyn_offsets[problem.root] + problem.state_dims[problem.root],
        ),
        dtype=int,
    )
    root_H = _triu_from_sparsity(root_fun.sparsity_out(1))
    root_C = _sp_from_sparsity(root_fun.sparsity_out(2))
    h_entries.extend(_mapped_entries(root_H, root_cols, root_cols))
    c_entries.extend(_mapped_entries(root_C, root_rows, root_cols))

    edge_meta = []
    for edge_index, edge_fun in enumerate(edge_funs):
        edge = problem.edges[edge_index]
        cols = _graph_edge_col_map(problem, edge_index)
        rows = np.asarray(
            range(
                dyn_offsets[edge.child],
                dyn_offsets[edge.child] + problem.state_dims[edge.child],
            ),
            dtype=int,
        )
        H = _triu_from_sparsity(edge_fun.sparsity_out(1))
        C = _sp_from_sparsity(edge_fun.sparsity_out(2))
        h_entries.extend(_mapped_entries(H, cols, cols))
        c_entries.extend(_mapped_entries(C, rows, cols))
        edge_meta.append(
            {
                "cols": cols,
                "rows": rows,
                "H": H,
                "C": C,
                "eval_name": edge_fun.name(),
                "value_name": (
                    edge_value_funs[edge_index].name() if edge_value_funs else None
                ),
            }
        )

    node_meta = []
    for node, node_fun in enumerate(node_funs):
        cols = _graph_node_col_map(problem, node, outgoing)
        c_rows = np.asarray(
            range(c_offsets[node], c_offsets[node] + problem.c_dims[node]), dtype=int
        )
        g_rows = np.asarray(
            range(z_offsets[node], z_offsets[node] + problem.g_dims[node]), dtype=int
        )
        H = _triu_from_sparsity(node_fun.sparsity_out(4))
        C = _sp_from_sparsity(node_fun.sparsity_out(5))
        G = _sp_from_sparsity(node_fun.sparsity_out(6))
        h_entries.extend(_mapped_entries(H, cols, cols))
        c_entries.extend(_mapped_entries(C, c_rows, cols))
        g_entries.extend(_mapped_entries(G, g_rows, cols))
        node_meta.append(
            {
                "cols": cols,
                "local_dim": len(cols),
                "c_rows": c_rows,
                "g_rows": g_rows,
                "H": H,
                "C": C,
                "G": G,
                "eval_name": node_fun.name(),
                "value_name": (
                    node_value_funs[node].name() if node_value_funs else None
                ),
            }
        )

    h_raw = _csc_from_entries((x_dim, x_dim), h_entries)
    h_sup = _upper_with_diagonal(h_raw)
    h_full = _full_symmetric_from_upper(h_sup)
    C = _csc_from_entries((y_dim, x_dim), c_entries)
    G = _csc_from_entries((z_dim, x_dim), g_entries)
    CT = C.T.tocsc()
    GT = G.T.tocsc()
    K = _kkt_pattern(h_full, C, G)
    P = _amd_order(K)
    P_inv = np.zeros_like(P)
    P_inv[P] = np.arange(P.size)

    root_meta = {
        "cols": root_cols,
        "rows": root_rows,
        "H": root_H,
        "C": root_C,
        "h_scatter": _scatter_to_global(root_H, root_cols, root_cols, h_sup),
        "c_scatter": _scatter_to_global(root_C, root_rows, root_cols, C),
    }
    for item in edge_meta:
        item["h_scatter"] = _scatter_to_global(
            item["H"], item["cols"], item["cols"], h_sup
        )
        item["c_scatter"] = _scatter_to_global(item["C"], item["rows"], item["cols"], C)
    for item in node_meta:
        item["h_scatter"] = _scatter_to_global(
            item["H"], item["cols"], item["cols"], h_sup
        )
        item["c_scatter"] = _scatter_to_global(
            item["C"], item["c_rows"], item["cols"], C
        )
        item["g_scatter"] = _scatter_to_global(
            item["G"], item["g_rows"], item["cols"], G
        )

    all_functions = [root_fun] + edge_funs + node_funs
    if root_values is not None:
        all_functions += [root_values] + edge_value_funs + node_value_funs
    local_metadata = _graph_local_metadata_summary(
        problem, root_meta, edge_meta, node_meta, outgoing, all_functions
    )

    return {
        "x_dim": x_dim,
        "y_dim": y_dim,
        "s_dim": z_dim,
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
        "kkt_L_nnz": K.shape[0] * (K.shape[0] + 1) // 2,
        **local_metadata,
        "c_to_ct": _transpose_map(C),
        "g_to_gt": _transpose_map(G),
    }


def _graph_ocp_stage_metadata(
    problem,
    root_fun,
    edge_funs,
    node_funs,
    outgoing,
):
    root_H = _triu_from_sparsity(root_fun.sparsity_out(1))
    root_C = _sp_from_sparsity(root_fun.sparsity_out(2))
    root_meta = {
        "H": root_H,
        "C": root_C,
    }

    edge_meta = []
    for edge_fun in edge_funs:
        edge_meta.append(
            {
                "H": _triu_from_sparsity(edge_fun.sparsity_out(1)),
                "C": _sp_from_sparsity(edge_fun.sparsity_out(2)),
                "eval_name": edge_fun.name(),
            }
        )

    node_meta = []
    for node, node_fun in enumerate(node_funs):
        out_control_dim = sum(problem.control_dims[edge] for edge in outgoing[node])
        node_meta.append(
            {
                "local_dim": (
                    problem.state_dims[node] + out_control_dim + problem.theta_dim
                ),
                "H": _triu_from_sparsity(node_fun.sparsity_out(4)),
                "C": _sp_from_sparsity(node_fun.sparsity_out(5)),
                "G": _sp_from_sparsity(node_fun.sparsity_out(6)),
                "eval_name": node_fun.name(),
            }
        )

    all_functions = [root_fun] + edge_funs + node_funs
    return _graph_local_metadata_summary(
        problem, root_meta, edge_meta, node_meta, outgoing, all_functions
    )


def _emit_graph_flat_split_cpp(problem, metadata, out_dir):
    state_offsets, control_offsets, theta_offset, _ = _graph_offsets(problem)
    dyn_offsets, c_offsets, _ = _graph_y_offsets(problem)
    z_offsets, _ = _graph_z_offsets(problem)
    h_sup = metadata["h_sup"]
    C = metadata["C"]
    G = metadata["G"]
    CT = metadata["CT"]
    GT = metadata["GT"]

    edge_parameters, edge_parameter_offsets = _flatten_groups(
        edge.parameters for edge in problem.edges
    )
    edge_h_scatter, edge_h_scatter_offsets = _flatten_groups(
        item["h_scatter"] for item in metadata["edges"]
    )
    edge_c_scatter, edge_c_scatter_offsets = _flatten_groups(
        item["c_scatter"] for item in metadata["edges"]
    )
    node_outgoing, node_outgoing_offsets = _flatten_groups(
        metadata["outgoing"][node] for node in range(problem.T + 1)
    )
    node_grad_scatter, node_grad_scatter_offsets = _flatten_groups(
        item["cols"] for item in metadata["nodes"]
    )
    node_h_scatter, node_h_scatter_offsets = _flatten_groups(
        item["h_scatter"] for item in metadata["nodes"]
    )
    node_c_scatter, node_c_scatter_offsets = _flatten_groups(
        item["c_scatter"] for item in metadata["nodes"]
    )
    node_g_scatter, node_g_scatter_offsets = _flatten_groups(
        item["g_scatter"] for item in metadata["nodes"]
    )

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
    arrays += _c_array("kKktPinv", metadata["P_inv"])
    arrays += _c_array("kCToCT", metadata["c_to_ct"])
    arrays += _c_array("kGToGT", metadata["g_to_gt"])
    arrays += _cpp_double_array("kInitialFlatX", _graph_initial(problem))
    arrays += _c_array("kRootHScatter", metadata["root"]["h_scatter"])
    arrays += _c_array("kRootCScatter", metadata["root"]["c_scatter"])
    arrays += _cpp_double_array("kEdgeParameters", edge_parameters)
    arrays += _c_array("kEdgeHScatter", edge_h_scatter)
    arrays += _c_array("kEdgeCScatter", edge_c_scatter)
    arrays += _c_array("kNodeOutgoing", node_outgoing)
    arrays += _c_array("kNodeGradScatter", node_grad_scatter)
    arrays += _c_array("kNodeHScatter", node_h_scatter)
    arrays += _c_array("kNodeCScatter", node_c_scatter)
    arrays += _c_array("kNodeGScatter", node_g_scatter)

    header = """#pragma once

#include "problem_definitions/casadi_problems/common/problem.hpp"

namespace sip_examples::problem_definitions::casadi_problems::generated_problem {

struct Problem {
  static const FlatProblemSpec &flat_spec();
  static sip::Settings settings();
  static CasadiWork make_flat_work();
  static void eval_flat_values(const double *x, double *f, double *c, double *g,
                               CasadiWork &work);
  static void eval_flat_qdldl(const double *x, const double *y, const double *z,
                              double *f, double *grad_f, double *c, double *g,
                              double *H, double *C, double *G, double *CT,
                              double *GT, CasadiWork &work);
  static void eval_flat_slacg(const double *x, const double *y, const double *z,
                              double *f, double *grad_f, double *c, double *g,
                              double *H, double *C, double *G, double *CT,
                              double *GT, CasadiWork &work);
};

} // namespace sip_examples::problem_definitions::casadi_problems::generated_problem
"""
    with open(os.path.join(out_dir, "generated_flat.hpp"), "w") as f:
        f.write(header)

    def ptr(base, offset, dim):
        return f"{base} + {offset}" if dim > 0 else "&kDummy"

    root = problem.root
    root_n = problem.state_dims[root]
    root_meta = metadata["root"]

    root_body = []
    root_value_body = []
    root_body.append(
        f"""
  {{
    const double *arg[kMaxArg] = {{
        {ptr("x", state_offsets[root], root_n)},
        theta,
        {ptr("y", dyn_offsets[root], root_n)},
    }};
    double *res[kMaxRes] = {{local_c, local_H, local_C}};
    graph_flat_root_eval(arg, res, work.iw.data(), work.w.data(), 0);
"""
    )
    for i, row in enumerate(root_meta["rows"]):
        root_body.append(f"    c[{int(row)}] = local_c[{i}];\n")
    for i in range(root_meta["H"].nnz):
        root_body.append(f"    H[kRootHScatter[{i}]] += local_H[{i}];\n")
    for i in range(root_meta["C"].nnz):
        root_body.append(f"    C[kRootCScatter[{i}]] = local_C[{i}];\n")
    root_body.append("  }\n")
    root_value_body.append(
        f"""
  {{
    const double *arg[kMaxArg] = {{
        {ptr("x", state_offsets[root], root_n)},
        theta,
    }};
    double *res[kMaxRes] = {{local_c}};
    graph_flat_root_values(arg, res, work.iw.data(), work.w.data(), 0);
"""
    )
    for i, row in enumerate(root_meta["rows"]):
        root_value_body.append(f"    c[{int(row)}] = local_c[{i}];\n")
    root_value_body.append("  }\n")

    edge_records = []
    for edge_index, (edge, item) in enumerate(zip(problem.edges, metadata["edges"])):
        edge_records.append(
            "  {"
            + ", ".join(
                str(value)
                for value in (
                    state_offsets[edge.parent],
                    problem.state_dims[edge.parent],
                    state_offsets[edge.child],
                    problem.state_dims[edge.child],
                    control_offsets[edge_index],
                    edge.control_dim,
                    edge_parameter_offsets[edge_index],
                    problem.parameter_dims[edge_index],
                    dyn_offsets[edge.child],
                    edge_h_scatter_offsets[edge_index],
                    item["H"].nnz,
                    edge_c_scatter_offsets[edge_index],
                    item["C"].nnz,
                )
            )
            + f", &{item['eval_name']}, &{item['value_name']}}},"
        )

    node_records = []
    for node, item in enumerate(metadata["nodes"]):
        node_records.append(
            "  {"
            + ", ".join(
                str(value)
                for value in (
                    state_offsets[node],
                    problem.state_dims[node],
                    c_offsets[node],
                    problem.c_dims[node],
                    z_offsets[node],
                    problem.g_dims[node],
                    node_outgoing_offsets[node],
                    len(metadata["outgoing"][node]),
                    node_grad_scatter_offsets[node],
                    item["local_dim"],
                    node_h_scatter_offsets[node],
                    item["H"].nnz,
                    node_c_scatter_offsets[node],
                    item["C"].nnz,
                    node_g_scatter_offsets[node],
                    item["G"].nnz,
                )
            )
            + f", &{item['eval_name']}, &{item['value_name']}}},"
        )

    tables = f"""
using CasadiFunction = decltype(&graph_flat_root_eval);

struct EdgeData {{
  int parent_x_offset;
  int parent_dim;
  int child_x_offset;
  int child_dim;
  int control_offset;
  int control_dim;
  int parameter_offset;
  int parameter_dim;
  int dynamics_offset;
  int h_scatter_offset;
  int h_nnz;
  int c_scatter_offset;
  int c_nnz;
  CasadiFunction eval;
  CasadiFunction values;
}};

static constexpr EdgeData kEdges[] = {{
{os.linesep.join(edge_records)}
}};

struct NodeData {{
  int state_offset;
  int state_dim;
  int c_offset;
  int c_dim;
  int g_offset;
  int g_dim;
  int outgoing_offset;
  int outgoing_count;
  int grad_scatter_offset;
  int grad_nnz;
  int h_scatter_offset;
  int h_nnz;
  int c_scatter_offset;
  int c_nnz;
  int g_scatter_offset;
  int g_nnz;
  CasadiFunction eval;
  CasadiFunction values;
}};

static constexpr NodeData kNodes[] = {{
{os.linesep.join(node_records)}
}};
"""

    cpp = f"""
#include "problem_definitions/casadi_problems/{problem.name}/generated_flat.hpp"

#include "problem_definitions/casadi_problems/{problem.name}/generated_graph_flat_casadi.h"
#include "problem_definitions/casadi_problems/{problem.name}/generated_graph_flat_values_casadi.h"

#include <algorithm>

namespace sip_examples::problem_definitions::casadi_problems::generated_problem {{
namespace {{

{arrays}
{tables}

constexpr int kThetaDim = {problem.theta_dim};
constexpr int kThetaOffset = {theta_offset};
constexpr int kFlatIw = {metadata["max_iw"]};
constexpr int kFlatW = {metadata["max_w"]};
constexpr int kMaxArg = {metadata["max_arg"]};
constexpr int kMaxRes = {metadata["max_res"]};
constexpr int kMaxGradDim = {metadata["max_grad_dim"]};
constexpr int kMaxCDim = {metadata["max_c_dim"]};
constexpr int kMaxGDim = {metadata["max_g_dim"]};
constexpr int kMaxHNnz = {metadata["max_h_nnz"]};
constexpr int kMaxCNnz = {metadata["max_c_nnz"]};
constexpr int kMaxGNnz = {metadata["max_g_nnz"]};
constexpr int kMaxNodeOutControlDim = {metadata["max_node_out_control_dim"]};
constexpr int kMaxNodeOutParameterDim = {metadata["max_node_out_parameter_dim"]};
constexpr int kScratchSize = kMaxGradDim + kMaxCDim + kMaxGDim + kMaxHNnz +
                             kMaxCNnz + kMaxGNnz + kMaxNodeOutControlDim +
                             kMaxNodeOutParameterDim;
constexpr double kDummy = 0.0;

const double *pointer_or_dummy(const double *values, int offset, int size) {{
  return size > 0 ? values + offset : &kDummy;
}}

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
    cpp += f"""
CasadiWork Problem::make_flat_work() {{
  CasadiWork work;
  work.resize(kFlatIw, kFlatW, kScratchSize);
  return work;
}}

void Problem::eval_flat_values(const double *x, double *f, double *c, double *g,
                               CasadiWork &work) {{
  const auto &spec = flat_spec();
  *f = 0.0;
  std::fill_n(c, spec.y_dim, 0.0);
  std::fill_n(g, spec.s_dim, 0.0);

  double *local_c = work.scratch.data();
  double *local_g = local_c + kMaxCDim;
  double *local_u = local_g + kMaxGDim;
  double *local_p = local_u + kMaxNodeOutControlDim;
  const double *theta = kThetaDim > 0 ? x + kThetaOffset : &kDummy;

{"".join(root_value_body)}
  for (const EdgeData &edge : kEdges) {{
    const double *arg[kMaxArg] = {{
        pointer_or_dummy(x, edge.parent_x_offset, edge.parent_dim),
        pointer_or_dummy(x, edge.child_x_offset, edge.child_dim),
        pointer_or_dummy(x, edge.control_offset, edge.control_dim),
        theta,
        pointer_or_dummy(kEdgeParameters, edge.parameter_offset,
                         edge.parameter_dim),
    }};
    double *res[kMaxRes] = {{local_c}};
    edge.values(arg, res, work.iw.data(), work.w.data(), 0);
    std::copy_n(local_c, edge.child_dim, c + edge.dynamics_offset);
  }}

  for (const NodeData &node : kNodes) {{
    int control_cursor = 0;
    int parameter_cursor = 0;
    for (int i = 0; i < node.outgoing_count; ++i) {{
      const EdgeData &edge = kEdges[kNodeOutgoing[node.outgoing_offset + i]];
      std::copy_n(x + edge.control_offset, edge.control_dim,
                  local_u + control_cursor);
      std::copy_n(kEdgeParameters + edge.parameter_offset, edge.parameter_dim,
                  local_p + parameter_cursor);
      control_cursor += edge.control_dim;
      parameter_cursor += edge.parameter_dim;
    }}
    double node_f = 0.0;
    const double *arg[kMaxArg] = {{
        pointer_or_dummy(x, node.state_offset, node.state_dim),
        theta,
        control_cursor > 0 ? local_u : &kDummy,
        parameter_cursor > 0 ? local_p : &kDummy,
    }};
    double *res[kMaxRes] = {{&node_f, local_c, local_g}};
    node.values(arg, res, work.iw.data(), work.w.data(), 0);
    *f += node_f;
    std::copy_n(local_c, node.c_dim, c + node.c_offset);
    std::copy_n(local_g, node.g_dim, g + node.g_offset);
  }}
}}

void Problem::eval_flat_qdldl(const double *x, const double *y, const double *z,
                              double *f, double *grad_f, double *c, double *g,
                              double *H, double *C, double *G, double *CT,
                              double *GT, CasadiWork &work) {{
  eval_flat_slacg(x, y, z, f, grad_f, c, g, H, C, G, CT, GT, work);
}}

void Problem::eval_flat_slacg(const double *x, const double *y, const double *z,
                              double *f, double *grad_f, double *c, double *g,
                              double *H, double *C, double *G, double *CT,
                              double *GT, CasadiWork &work) {{
  const auto &spec = flat_spec();
  *f = 0.0;
  std::fill_n(grad_f, spec.x_dim, 0.0);
  std::fill_n(c, spec.y_dim, 0.0);
  std::fill_n(g, spec.s_dim, 0.0);
  std::fill_n(H, spec.upper_hessian_lagrangian_nnz, 0.0);
  std::fill_n(C, spec.jacobian_c_nnz, 0.0);
  std::fill_n(G, spec.jacobian_g_nnz, 0.0);

  double *local_grad = work.scratch.data();
  double *local_c = local_grad + kMaxGradDim;
  double *local_g = local_c + kMaxCDim;
  double *local_H = local_g + kMaxGDim;
  double *local_C = local_H + kMaxHNnz;
  double *local_G = local_C + kMaxCNnz;
  double *local_u = local_G + kMaxGNnz;
  double *local_p = local_u + kMaxNodeOutControlDim;
  const double *theta = kThetaDim > 0 ? x + kThetaOffset : &kDummy;

{"".join(root_body)}
  for (const EdgeData &edge : kEdges) {{
    const double *arg[kMaxArg] = {{
        pointer_or_dummy(x, edge.parent_x_offset, edge.parent_dim),
        pointer_or_dummy(x, edge.child_x_offset, edge.child_dim),
        pointer_or_dummy(x, edge.control_offset, edge.control_dim),
        theta,
        pointer_or_dummy(kEdgeParameters, edge.parameter_offset,
                         edge.parameter_dim),
        pointer_or_dummy(y, edge.dynamics_offset, edge.child_dim),
    }};
    double *res[kMaxRes] = {{local_c, local_H, local_C}};
    edge.eval(arg, res, work.iw.data(), work.w.data(), 0);
    std::copy_n(local_c, edge.child_dim, c + edge.dynamics_offset);
    for (int i = 0; i < edge.h_nnz; ++i) {{
      H[kEdgeHScatter[edge.h_scatter_offset + i]] += local_H[i];
    }}
    for (int i = 0; i < edge.c_nnz; ++i) {{
      C[kEdgeCScatter[edge.c_scatter_offset + i]] = local_C[i];
    }}
  }}

  for (const NodeData &node : kNodes) {{
    int control_cursor = 0;
    int parameter_cursor = 0;
    for (int i = 0; i < node.outgoing_count; ++i) {{
      const EdgeData &edge = kEdges[kNodeOutgoing[node.outgoing_offset + i]];
      std::copy_n(x + edge.control_offset, edge.control_dim,
                  local_u + control_cursor);
      std::copy_n(kEdgeParameters + edge.parameter_offset, edge.parameter_dim,
                  local_p + parameter_cursor);
      control_cursor += edge.control_dim;
      parameter_cursor += edge.parameter_dim;
    }}
    double node_f = 0.0;
    const double *arg[kMaxArg] = {{
        pointer_or_dummy(x, node.state_offset, node.state_dim),
        theta,
        control_cursor > 0 ? local_u : &kDummy,
        parameter_cursor > 0 ? local_p : &kDummy,
        pointer_or_dummy(y, node.c_offset, node.c_dim),
        pointer_or_dummy(z, node.g_offset, node.g_dim),
    }};
    double *res[kMaxRes] = {{&node_f, local_grad, local_c, local_g,
                            local_H, local_C, local_G}};
    node.eval(arg, res, work.iw.data(), work.w.data(), 0);
    *f += node_f;
    for (int i = 0; i < node.grad_nnz; ++i) {{
      grad_f[kNodeGradScatter[node.grad_scatter_offset + i]] += local_grad[i];
    }}
    std::copy_n(local_c, node.c_dim, c + node.c_offset);
    std::copy_n(local_g, node.g_dim, g + node.g_offset);
    for (int i = 0; i < node.h_nnz; ++i) {{
      H[kNodeHScatter[node.h_scatter_offset + i]] += local_H[i];
    }}
    for (int i = 0; i < node.c_nnz; ++i) {{
      C[kNodeCScatter[node.c_scatter_offset + i]] = local_C[i];
    }}
    for (int i = 0; i < node.g_nnz; ++i) {{
      G[kNodeGScatter[node.g_scatter_offset + i]] = local_G[i];
    }}
  }}
  for (int i = 0; i < spec.jacobian_c_nnz; ++i) {{
    CT[kCToCT[i]] = C[i];
  }}
  for (int i = 0; i < spec.jacobian_g_nnz; ++i) {{
    GT[kGToGT[i]] = G[i];
  }}
}}

}} // namespace sip_examples::problem_definitions::casadi_problems::generated_problem
"""
    with open(os.path.join(out_dir, "generated_flat.cpp"), "w") as f:
        f.write(cpp)


def _emit_graph_ocp_split_cpp(problem, metadata, out_dir):
    edge_parameters, edge_parameter_offsets = _flatten_groups(
        edge.parameters for edge in problem.edges
    )
    node_outgoing, node_outgoing_offsets = _flatten_groups(
        metadata["outgoing"][node] for node in range(problem.T + 1)
    )

    arrays = ""
    arrays += _cpp_double_array("kInitialOcpX", _graph_initial(problem))
    arrays += _c_array("kStateDims", problem.state_dims)
    arrays += _c_array("kControlDims", problem.control_dims)
    arrays += _c_array("kCDims", problem.c_dims)
    arrays += _c_array("kGDims", problem.g_dims)
    arrays += _c_array("kEdgeParents", [edge.parent for edge in problem.edges])
    arrays += _c_array("kEdgeChildren", [edge.child for edge in problem.edges])
    arrays += _cpp_double_array("kEdgeParameters", edge_parameters)
    arrays += _c_array("kNodeOutgoing", node_outgoing)

    header = """#pragma once

#include "problem_definitions/casadi_problems/common/ocp_problem.hpp"
#include "problem_definitions/casadi_problems/common/problem.hpp"
#include "sip_optimal_control/types.hpp"

namespace sip_examples::problem_definitions::casadi_problems::generated_problem {

struct Problem {
  static const OcpProblemSpec &ocp_spec();
  static sip::Settings settings();
  static CasadiWork make_ocp_work();
  static void eval_ocp(const ::sip::optimal_control::ModelCallbackInput &mci,
                       ::sip::optimal_control::ModelCallbackOutput &mco,
                       CasadiWork &work);
};

} // namespace sip_examples::problem_definitions::casadi_problems::generated_problem
"""
    with open(os.path.join(out_dir, "generated_ocp.hpp"), "w") as f:
        f.write(header)

    def local_kind(node_control_map, index):
        if index < node_control_map["state_dim"]:
            return ("x", None, index)
        index -= node_control_map["state_dim"]
        if index < len(node_control_map["controls"]):
            return ("u",) + node_control_map["controls"][index]
        index -= len(node_control_map["controls"])
        return ("theta", None, index)

    def add_symmetric_block_assignment(
        body, target, rows, row, col, value, indent="    "
    ):
        body.append(f"{indent}{target}[{row} + {col} * {rows}] += {value};\n")
        if row != col:
            body.append(f"{indent}{target}[{col} + {row} * {rows}] += {value};\n")

    def scatter_root_hessian(body, item):
        root = problem.root
        n = problem.state_dims[root]
        for value_index, (row, col) in enumerate(zip(*item["H"].entries())):
            value = f"local_H[{value_index}]"
            if row < n and col < n:
                add_symmetric_block_assignment(
                    body, "mco.d2L_dx2[kRoot]", n, row, col, value
                )
            elif row < n and col >= n:
                theta_col = col - n
                body.append(
                    f"    mco.d2L_dxdtheta[kRoot][{row} + {theta_col} * {n}] += {value};\n"
                )
            elif row >= n and col >= n:
                theta_row = row - n
                theta_col = col - n
                add_symmetric_block_assignment(
                    body, "mco.d2L_dtheta2", "kThetaDim", theta_row, theta_col, value
                )
            else:
                raise ValueError("unexpected root Hessian sparsity")

    def scatter_edge_hessian(body, edge_index, item):
        edge = problem.edges[edge_index]
        parent_n = problem.state_dims[edge.parent]
        child_n = problem.state_dims[edge.child]
        control_dim = edge.control_dim
        parent_end = parent_n
        child_end = parent_end + child_n
        control_end = child_end + control_dim
        for value_index, (row, col) in enumerate(zip(*item["H"].entries())):
            value = f"local_H[{value_index}]"
            if row < parent_end and col < parent_end:
                add_symmetric_block_assignment(
                    body, "mco.d2L_dx2[edge.parent]", parent_n, row, col, value
                )
            elif row < parent_end and child_end <= col < control_end:
                u_col = col - child_end
                body.append(
                    f"  mco.d2L_dxdu[edge_index][{row} + {u_col} * {parent_n}] += {value};\n"
                )
            elif child_end <= row < control_end and child_end <= col < control_end:
                u_row = row - child_end
                u_col = col - child_end
                add_symmetric_block_assignment(
                    body, "mco.d2L_du2[edge_index]", control_dim, u_row, u_col, value
                )
            elif row < parent_end and col >= control_end:
                theta_col = col - control_end
                body.append(
                    f"  mco.d2L_dxdtheta[edge.parent][{row} + {theta_col} * {parent_n}] += {value};\n"
                )
            elif child_end <= row < control_end and col >= control_end:
                u_row = row - child_end
                theta_col = col - control_end
                body.append(
                    f"  mco.d2L_dudtheta[edge_index][{u_row} + {theta_col} * {control_dim}] += {value};\n"
                )
            elif row >= control_end and col >= control_end:
                theta_row = row - control_end
                theta_col = col - control_end
                add_symmetric_block_assignment(
                    body, "mco.d2L_dtheta2", "kThetaDim", theta_row, theta_col, value
                )
            elif parent_end <= row < child_end or parent_end <= col < child_end:
                raise ValueError(
                    f"{problem.name} edge {edge_index} has nonlinear child-state Hessian entries"
                )
            else:
                raise ValueError(f"unexpected edge Hessian sparsity in {problem.name}")

    def node_control_map(node):
        controls = []
        control_dims = []
        for out_slot, edge_index in enumerate(metadata["outgoing"][node]):
            control_dims.append(problem.control_dims[edge_index])
            for local_u in range(problem.control_dims[edge_index]):
                controls.append((out_slot, local_u))
        return {
            "state_dim": problem.state_dims[node],
            "controls": controls,
            "control_dims": control_dims,
        }

    def scatter_node_hessian(body, node, item, control_map):
        n = problem.state_dims[node]
        for value_index, (row, col) in enumerate(zip(*item["H"].entries())):
            value = f"local_H[{value_index}]"
            row_kind, row_edge, row_local = local_kind(control_map, row)
            col_kind, col_edge, col_local = local_kind(control_map, col)
            if row_kind == "x" and col_kind == "x":
                add_symmetric_block_assignment(
                    body, "mco.d2L_dx2[node_index]", n, row_local, col_local, value
                )
            elif row_kind == "x" and col_kind == "u":
                body.append(
                    f"  mco.d2L_dxdu[out_edges[{col_edge}]][{row_local} + {col_local} * {n}] += {value};\n"
                )
            elif row_kind == "u" and col_kind == "u":
                if row_edge != col_edge:
                    raise ValueError(
                        f"{problem.name} node {node} has cross-edge control Hessian entries"
                    )
                control_dim = control_map["control_dims"][row_edge]
                add_symmetric_block_assignment(
                    body,
                    f"mco.d2L_du2[out_edges[{row_edge}]]",
                    control_dim,
                    row_local,
                    col_local,
                    value,
                )
            elif row_kind == "x" and col_kind == "theta":
                body.append(
                    f"  mco.d2L_dxdtheta[node_index][{row_local} + {col_local} * {n}] += {value};\n"
                )
            elif row_kind == "u" and col_kind == "theta":
                control_dim = control_map["control_dims"][row_edge]
                body.append(
                    f"  mco.d2L_dudtheta[out_edges[{row_edge}]][{row_local} + {col_local} * {control_dim}] += {value};\n"
                )
            elif row_kind == "theta" and col_kind == "theta":
                add_symmetric_block_assignment(
                    body, "mco.d2L_dtheta2", "kThetaDim", row_local, col_local, value
                )
            else:
                raise ValueError(f"unexpected node Hessian sparsity in {problem.name}")

    def scatter_edge_jacobian(body, edge_index, item):
        edge = problem.edges[edge_index]
        parent_n = problem.state_dims[edge.parent]
        child_n = problem.state_dims[edge.child]
        control_dim = edge.control_dim
        parent_end = parent_n
        child_end = parent_end + child_n
        control_end = child_end + control_dim
        for value_index, (row, col) in enumerate(zip(*item["C"].entries())):
            value = f"local_C[{value_index}]"
            if col < parent_end:
                body.append(
                    f"  mco.ddyn_dx[edge_index][{row} + {col} * {child_n}] = {value};\n"
                )
            elif child_end <= col < control_end:
                u_col = col - child_end
                body.append(
                    f"  mco.ddyn_du[edge_index][{row} + {u_col} * {child_n}] = {value};\n"
                )
            elif col >= control_end:
                theta_col = col - control_end
                body.append(
                    f"  mco.ddyn_dtheta[edge_index][{row} + {theta_col} * {child_n}] = {value};\n"
                )

    def scatter_node_jacobian(
        body, node, item, control_map, matrix_name, target_prefix
    ):
        row_dim = problem.c_dims[node] if target_prefix == "c" else problem.g_dims[node]
        matrix = item["C"] if target_prefix == "c" else item["G"]
        for value_index, (row, col) in enumerate(zip(*matrix.entries())):
            value = f"{matrix_name}[{value_index}]"
            kind, edge_index, local_index = local_kind(control_map, col)
            if kind == "x":
                body.append(
                    f"  mco.d{target_prefix}_dx[node_index][{row} + {local_index} * {row_dim}] = {value};\n"
                )
            elif kind == "u":
                body.append(
                    f"  mco.d{target_prefix}_du[out_edges[{edge_index}]][{row} + {local_index} * {row_dim}] = {value};\n"
                )
            else:
                body.append(
                    f"  mco.d{target_prefix}_dtheta[node_index][{row} + {local_index} * {row_dim}] = {value};\n"
                )

    body = [
        """  mco.f = 0.0;
  std::fill_n(mco.df_dtheta, kThetaDim, 0.0);
  std::fill_n(mco.d2L_dtheta2, kThetaDim * kThetaDim, 0.0);
  for (int node = 0; node < kNumNodes; ++node) {
    const int n = kStateDims[node];
    const int c_dim = kCDims[node];
    const int g_dim = kGDims[node];
    std::fill_n(mco.df_dx[node], n, 0.0);
    std::fill_n(mco.dyn_res[node], n, 0.0);
    std::fill_n(mco.c[node], c_dim, 0.0);
    std::fill_n(mco.g[node], g_dim, 0.0);
    std::fill_n(mco.dc_dx[node], c_dim * n, 0.0);
    std::fill_n(mco.dg_dx[node], g_dim * n, 0.0);
    std::fill_n(mco.dc_dtheta[node], c_dim * kThetaDim, 0.0);
    std::fill_n(mco.dg_dtheta[node], g_dim * kThetaDim, 0.0);
    std::fill_n(mco.d2L_dx2[node], n * n, 0.0);
    std::fill_n(mco.d2L_dxdtheta[node], n * kThetaDim, 0.0);
  }
  for (int edge_index = 0; edge_index < kNumEdges; ++edge_index) {
    const int parent = kEdgeParents[edge_index];
    const int child = kEdgeChildren[edge_index];
    const int parent_n = kStateDims[parent];
    const int child_n = kStateDims[child];
    const int m = kControlDims[edge_index];
    std::fill_n(mco.df_du[edge_index], m, 0.0);
    std::fill_n(mco.ddyn_dx[edge_index], child_n * parent_n, 0.0);
    std::fill_n(mco.ddyn_du[edge_index], child_n * m, 0.0);
    std::fill_n(mco.ddyn_dtheta[edge_index], child_n * kThetaDim, 0.0);
    std::fill_n(mco.dc_du[edge_index], kCDims[parent] * m, 0.0);
    std::fill_n(mco.dg_du[edge_index], kGDims[parent] * m, 0.0);
    std::fill_n(mco.d2L_dxdu[edge_index], parent_n * m, 0.0);
    std::fill_n(mco.d2L_du2[edge_index], m * m, 0.0);
    std::fill_n(mco.d2L_dudtheta[edge_index], m * kThetaDim, 0.0);
  }
"""
    ]

    root = problem.root
    root_n = problem.state_dims[root]
    root_meta = metadata["root"]
    body.append(
        """
  {
    const double *arg[kMaxArg] = {
        mci.states[kRoot],
        theta,
        mci.costates[kRoot],
    };
    double *res[kMaxRes] = {local_c, local_H, local_C};
    graph_ocp_root_eval(arg, res, work.iw.data(), work.w.data(), 0);
"""
    )
    for i in range(root_n):
        body.append(f"    mco.dyn_res[kRoot][{i}] = local_c[{i}];\n")
    scatter_root_hessian(body, root_meta)
    body.append("  }\n")

    edge_pattern_indices = []
    edge_patterns = {}
    edge_helpers = []
    for edge_index, edge in enumerate(problem.edges):
        item = metadata["edges"][edge_index]
        pattern_key = (
            problem.state_dims[edge.parent],
            problem.state_dims[edge.child],
            edge.control_dim,
            tuple(zip(*item["H"].entries())),
            tuple(zip(*item["C"].entries())),
        )
        pattern_index = edge_patterns.get(pattern_key)
        if pattern_index is None:
            pattern_index = len(edge_patterns)
            edge_patterns[pattern_key] = pattern_index
            helper = [
                f"""void eval_edge_pattern_{pattern_index}(
    int edge_index, const EdgeData &edge,
    const ::sip::optimal_control::ModelCallbackInput &mci,
    ::sip::optimal_control::ModelCallbackOutput &mco, const double *theta,
    double *local_c, double *local_H, double *local_C, CasadiWork &work) {{
  const double *arg[kMaxArg] = {{
      mci.states[edge.parent],
      mci.states[edge.child],
      edge.control_dim > 0 ? mci.controls[edge_index] : &kDummy,
      theta,
      edge.parameter_dim > 0 ? kEdgeParameters + edge.parameter_offset
                             : &kDummy,
      mci.costates[edge.child],
  }};
  double *res[kMaxRes] = {{local_c, local_H, local_C}};
  edge.eval(arg, res, work.iw.data(), work.w.data(), 0);
  std::copy_n(local_c, kStateDims[edge.child], mco.dyn_res[edge.child]);
"""
            ]
            scatter_edge_hessian(helper, edge_index, item)
            scatter_edge_jacobian(helper, edge_index, item)
            helper.append("}\n")
            edge_helpers.append("".join(helper))
        edge_pattern_indices.append(pattern_index)

    edge_records = []
    for edge_index, (edge, item, pattern_index) in enumerate(
        zip(problem.edges, metadata["edges"], edge_pattern_indices)
    ):
        edge_records.append(
            f"  {{{edge.parent}, {edge.child}, {edge.control_dim}, "
            f"{edge_parameter_offsets[edge_index]}, "
            f"{problem.parameter_dims[edge_index]}, &{item['eval_name']}, "
            f"&eval_edge_pattern_{pattern_index}}},"
        )

    body.append(
        """  for (int edge_index = 0; edge_index < kNumEdges; ++edge_index) {
    const EdgeData &edge = kEdges[edge_index];
    edge.evaluate(edge_index, edge, mci, mco, theta, local_c, local_H,
                  local_C, work);
  }
"""
    )

    node_pattern_indices = []
    node_patterns = {}
    node_helpers = []
    for node in range(problem.T + 1):
        item = metadata["nodes"][node]
        control_map = node_control_map(node)
        pattern_key = (
            problem.state_dims[node],
            tuple(control_map["control_dims"]),
            problem.c_dims[node],
            problem.g_dims[node],
            item["local_dim"],
            tuple(zip(*item["H"].entries())),
            tuple(zip(*item["C"].entries())),
            tuple(zip(*item["G"].entries())),
        )
        pattern_index = node_patterns.get(pattern_key)
        if pattern_index is None:
            pattern_index = len(node_patterns)
            node_patterns[pattern_key] = pattern_index
            helper = [
                f"""void eval_node_pattern_{pattern_index}(
    int node_index, const NodeData &node,
    const ::sip::optimal_control::ModelCallbackInput &mci,
    ::sip::optimal_control::ModelCallbackOutput &mco, const double *theta,
    double *local_grad, double *local_c, double *local_g, double *local_H,
    double *local_C, double *local_G, double *local_u, double *local_p,
    CasadiWork &work) {{
  const int *out_edges = kNodeOutgoing + node.outgoing_offset;
  int control_cursor = 0;
  int parameter_cursor = 0;
  for (int i = 0; i < node.outgoing_count; ++i) {{
    const EdgeData &edge = kEdges[out_edges[i]];
    if (edge.control_dim > 0) {{
      std::copy_n(mci.controls[out_edges[i]], edge.control_dim,
                  local_u + control_cursor);
    }}
    if (edge.parameter_dim > 0) {{
      std::copy_n(kEdgeParameters + edge.parameter_offset, edge.parameter_dim,
                  local_p + parameter_cursor);
    }}
    control_cursor += edge.control_dim;
    parameter_cursor += edge.parameter_dim;
  }}
  double node_f = 0.0;
  const double *arg[kMaxArg] = {{
      mci.states[node_index],
      theta,
      control_cursor > 0 ? local_u : &kDummy,
      parameter_cursor > 0 ? local_p : &kDummy,
      kCDims[node_index] > 0 ? mci.equality_constraint_multipliers[node_index]
                             : &kDummy,
      kGDims[node_index] > 0 ? mci.inequality_constraint_multipliers[node_index]
                             : &kDummy,
  }};
  double *res[kMaxRes] = {{&node_f, local_grad, local_c, local_g,
                          local_H, local_C, local_G}};
  node.eval(arg, res, work.iw.data(), work.w.data(), 0);
  mco.f += node_f;
"""
            ]
            for value_index in range(item["local_dim"]):
                kind, out_slot, local_index = local_kind(control_map, value_index)
                if kind == "x":
                    helper.append(
                        f"  mco.df_dx[node_index][{local_index}] += "
                        f"local_grad[{value_index}];\n"
                    )
                elif kind == "u":
                    helper.append(
                        f"  mco.df_du[out_edges[{out_slot}]][{local_index}] += "
                        f"local_grad[{value_index}];\n"
                    )
                else:
                    helper.append(
                        f"  mco.df_dtheta[{local_index}] += "
                        f"local_grad[{value_index}];\n"
                    )
            helper.append(
                "  std::copy_n(local_c, kCDims[node_index], mco.c[node_index]);\n"
            )
            helper.append(
                "  std::copy_n(local_g, kGDims[node_index], mco.g[node_index]);\n"
            )
            scatter_node_hessian(helper, node, item, control_map)
            scatter_node_jacobian(helper, node, item, control_map, "local_C", "c")
            scatter_node_jacobian(helper, node, item, control_map, "local_G", "g")
            helper.append("}\n")
            node_helpers.append("".join(helper))
        node_pattern_indices.append(pattern_index)

    node_records = []
    for node, (item, pattern_index) in enumerate(
        zip(metadata["nodes"], node_pattern_indices)
    ):
        node_records.append(
            f"  {{{node_outgoing_offsets[node]}, "
            f"{len(metadata['outgoing'][node])}, &{item['eval_name']}, "
            f"&eval_node_pattern_{pattern_index}}},"
        )

    body.append(
        """  for (int node_index = 0; node_index < kNumNodes; ++node_index) {
    const NodeData &node = kNodes[node_index];
    node.evaluate(node_index, node, mci, mco, theta, local_grad, local_c,
                  local_g, local_H, local_C, local_G, local_u, local_p, work);
  }
"""
    )

    tables = f"""
using CasadiFunction = decltype(&graph_ocp_root_eval);

struct EdgeData;
using EdgeEvaluator = void (*)(
    int, const EdgeData &,
    const ::sip::optimal_control::ModelCallbackInput &,
    ::sip::optimal_control::ModelCallbackOutput &, const double *, double *,
    double *, double *, CasadiWork &);

struct EdgeData {{
  int parent;
  int child;
  int control_dim;
  int parameter_offset;
  int parameter_dim;
  CasadiFunction eval;
  EdgeEvaluator evaluate;
}};

struct NodeData;
using NodeEvaluator = void (*)(
    int, const NodeData &,
    const ::sip::optimal_control::ModelCallbackInput &,
    ::sip::optimal_control::ModelCallbackOutput &, const double *, double *,
    double *, double *, double *, double *, double *, double *, double *,
    CasadiWork &);

struct NodeData {{
  int outgoing_offset;
  int outgoing_count;
  CasadiFunction eval;
  NodeEvaluator evaluate;
}};

{"".join(edge_helpers)}

static constexpr EdgeData kEdges[] = {{
{os.linesep.join(edge_records)}
}};

{"".join(node_helpers)}

static constexpr NodeData kNodes[] = {{
{os.linesep.join(node_records)}
}};
"""

    cpp = f"""
#include "problem_definitions/casadi_problems/{problem.name}/generated_ocp.hpp"

#include "problem_definitions/casadi_problems/{problem.name}/generated_graph_ocp_casadi.h"

#include <algorithm>

namespace sip_examples::problem_definitions::casadi_problems::generated_problem {{
namespace {{

{arrays}

constexpr int kRoot = {problem.root};
constexpr int kNumEdges = {problem.T};
constexpr int kNumNodes = kNumEdges + 1;
constexpr int kThetaDim = {problem.theta_dim};
constexpr int kOcpIw = {metadata["max_iw"]};
constexpr int kOcpW = {metadata["max_w"]};
constexpr int kMaxArg = {metadata["max_arg"]};
constexpr int kMaxRes = {metadata["max_res"]};
constexpr int kMaxGradDim = {metadata["max_grad_dim"]};
constexpr int kMaxCDim = {metadata["max_c_dim"]};
constexpr int kMaxGDim = {metadata["max_g_dim"]};
constexpr int kMaxHNnz = {metadata["max_h_nnz"]};
constexpr int kMaxCNnz = {metadata["max_c_nnz"]};
constexpr int kMaxGNnz = {metadata["max_g_nnz"]};
constexpr int kMaxNodeOutControlDim = {metadata["max_node_out_control_dim"]};
constexpr int kMaxNodeOutParameterDim = {metadata["max_node_out_parameter_dim"]};
constexpr int kScratchSize = kMaxGradDim + kMaxCDim + kMaxGDim + kMaxHNnz +
                             kMaxCNnz + kMaxGNnz + kMaxNodeOutControlDim +
                             kMaxNodeOutParameterDim;
constexpr double kDummy = 0.0;

{tables}

}} // namespace

const OcpProblemSpec &Problem::ocp_spec() {{
  static constexpr OcpProblemSpec spec{{
      .name = "{problem.name}",
      .num_edges = {problem.T},
      .dimensions = ::sip::optimal_control::Dimensions{{
          {problem.theta_dim}, kStateDims, kControlDims, kCDims, kGDims}},
      .topology_root = {problem.root},
      .edge_parents = kEdgeParents,
      .edge_children = kEdgeChildren,
      .initial_x = kInitialOcpX,
  }};
  return spec;
}}
"""
    cpp += _settings_cpp(problem)
    cpp += f"""
CasadiWork Problem::make_ocp_work() {{
  CasadiWork work;
  work.resize(kOcpIw, kOcpW, kScratchSize);
  return work;
}}

void Problem::eval_ocp(const ::sip::optimal_control::ModelCallbackInput &mci,
                       ::sip::optimal_control::ModelCallbackOutput &mco,
                       CasadiWork &work) {{
  double *local_grad = work.scratch.data();
  double *local_c = local_grad + kMaxGradDim;
  double *local_g = local_c + kMaxCDim;
  double *local_H = local_g + kMaxGDim;
  double *local_C = local_H + kMaxHNnz;
  double *local_G = local_C + kMaxCNnz;
  double *local_u = local_G + kMaxGNnz;
  double *local_p = local_u + kMaxNodeOutControlDim;
  const double *theta = kThetaDim > 0 ? mci.theta : &kDummy;

{"".join(body)}
}}

}} // namespace sip_examples::problem_definitions::casadi_problems::generated_problem
"""
    with open(os.path.join(out_dir, "generated_ocp.cpp"), "w") as f:
        f.write(cpp)


def _unique_functions(functions):
    return {function.name(): function for function in functions}.values()


def generate_graph_flat(problem, out_dir, emit_kkt_code):
    os.makedirs(out_dir, exist_ok=True)
    (
        root_fun,
        edge_funs,
        node_funs,
        root_values,
        edge_value_funs,
        node_value_funs,
        outgoing,
    ) = _build_graph_flat_stage_functions(problem)
    metadata = _graph_flat_stage_metadata(
        problem,
        root_fun,
        edge_funs,
        node_funs,
        root_values,
        edge_value_funs,
        node_value_funs,
        outgoing,
    )
    old_cwd = os.getcwd()
    try:
        os.chdir(out_dir)
        cg = ca.CodeGenerator("generated_graph_flat_casadi.c", {"with_header": True})
        cg.add(root_fun)
        for fun in _unique_functions(edge_funs):
            cg.add(fun)
        for fun in _unique_functions(node_funs):
            cg.add(fun)
        cg.generate()
        values_cg = ca.CodeGenerator(
            "generated_graph_flat_values_casadi.c", {"with_header": True}
        )
        values_cg.add(root_values)
        for fun in _unique_functions(edge_value_funs):
            values_cg.add(fun)
        for fun in _unique_functions(node_value_funs):
            values_cg.add(fun)
        values_cg.generate()
    finally:
        os.chdir(old_cwd)
    if emit_kkt_code:
        cpp_header_code, cpp_impl_code = _kkt_codegen(
            H=_to_scipy(_full_symmetric_from_upper(metadata["h_sup"])),
            C=_to_scipy(metadata["C"]),
            G=_to_scipy(metadata["G"]),
            P=metadata["P"],
            namespace="sip_examples::problem_definitions::casadi_problems::generated_problem",
            header_name="kkt_codegen",
        )
        with open(os.path.join(out_dir, "kkt_codegen.hpp"), "w") as f:
            f.write(cpp_header_code)
        with open(os.path.join(out_dir, "kkt_codegen.cpp"), "w") as f:
            f.write(cpp_impl_code)
    _emit_graph_flat_split_cpp(problem, metadata, out_dir)


def generate_graph_ocp(problem, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    (
        root_fun,
        edge_funs,
        node_funs,
        root_values,
        edge_value_funs,
        node_value_funs,
        outgoing,
    ) = _build_graph_flat_stage_functions(
        problem, prefix="graph_ocp", build_value_functions=False
    )
    metadata = _graph_ocp_stage_metadata(
        problem,
        root_fun,
        edge_funs,
        node_funs,
        outgoing,
    )
    old_cwd = os.getcwd()
    try:
        os.chdir(out_dir)
        cg = ca.CodeGenerator("generated_graph_ocp_casadi.c", {"with_header": True})
        cg.add(root_fun)
        for fun in _unique_functions(edge_funs):
            cg.add(fun)
        for fun in _unique_functions(node_funs):
            cg.add(fun)
        cg.generate()
    finally:
        os.chdir(old_cwd)
    _emit_graph_ocp_split_cpp(problem, metadata, out_dir)


def main(problem_factory):
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", required=True, choices=["flat", "ocp"])
    parser.add_argument("--emit-kkt-code", action="store_true")
    parser.add_argument("out_dir")
    args = parser.parse_args()
    problem = problem_factory()
    if isinstance(problem, GraphProblemData):
        if args.mode == "flat":
            generate_graph_flat(problem, args.out_dir, args.emit_kkt_code)
        else:
            generate_graph_ocp(problem, args.out_dir)
    elif args.mode == "flat":
        generate_flat(problem, args.out_dir, args.emit_kkt_code)
    else:
        generate_ocp(problem, args.out_dir)
