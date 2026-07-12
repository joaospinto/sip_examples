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
    return sp.sparse.csc_matrix((data, pattern.indices, pattern.indptr), shape=pattern.shape)


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
    dyn_refs: object = None
    settings_override_cpp: str = ""

    @property
    def T(self):
        return len(self.edges)

    @property
    def control_dims(self):
        return [edge.control_dim for edge in self.edges]

    @property
    def root(self):
        children = {edge.child for edge in self.edges}
        roots = [node for node in range(len(self.state_dims)) if node not in children]
        if len(roots) != 1:
            raise ValueError(f"{self.name} must have exactly one root")
        return roots[0]


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
  for (int stage = 0; stage < kNumEdges; ++stage) {
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
      x + kNumEdges * kStateDim,
      theta,
      kUserEqDim > 0 ? y + kStateDim + kNumEdges * (kStateDim + kUserEqDim) : &kDummy,
      kIneqDim > 0 ? z + kNumEdges * kIneqDim : &kDummy,
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

#include "problem_definitions/casadi_problems/common/ocp_problem.hpp"
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
    double *res[] = {
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
    const double *arg[] = {
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
  double *res[] = {
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
  const double *arg[] = {
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


def _graph_dyn_refs(problem, node):
    state_dim = problem.state_dims[node]
    if getattr(problem, "dyn_refs", None) is None:
        return np.ones(state_dim)
    refs = np.asarray(problem.dyn_refs[node], dtype=float).reshape(-1)
    if refs.size != state_dim:
        raise ValueError(
            f"{problem.name} dyn_refs[{node}] has size {refs.size}, expected {state_dim}"
        )
    if np.any(refs <= 0.0):
        raise ValueError(f"{problem.name} dyn_refs[{node}] must be positive")
    return refs


def _scale_graph_dyn(problem, node, dyn):
    refs = _graph_dyn_refs(problem, node)
    if refs.size == 0 or np.all(refs == 1.0):
        return dyn
    return ca.vertcat(*[dyn[i] / refs[i] for i in range(refs.size)])


def _graph_initial(problem):
    pieces = []
    for edge in range(problem.T):
        pieces.append(problem.X_init[edge])
        pieces.append(problem.U_init[edge])
    pieces.append(problem.X_init[problem.T])
    pieces.append(problem.theta_init)
    return np.concatenate([np.asarray(p).reshape(-1) for p in pieces])


def _graph_connectivity(problem):
    incoming = {edge.child: edge_index for edge_index, edge in enumerate(problem.edges)}
    outgoing = {node: [] for node in range(problem.T + 1)}
    for edge_index, edge in enumerate(problem.edges):
        outgoing[edge.parent].append(edge_index)
    return incoming, outgoing


def _graph_control_slice(controls, edge_indices, control_dims):
    pieces = []
    offset = 0
    for edge_index in edge_indices:
        dim = control_dims[edge_index]
        pieces.append(controls[offset : offset + dim])
        offset += dim
    return pieces


def _build_graph_flat_stage_functions(problem, prefix="graph_flat"):
    _, outgoing = _graph_connectivity(problem)
    td = problem.theta_dim
    theta_sym = ca.SX.sym("theta", max(td, 1))
    theta = theta_sym[:td] if td > 0 else ca.SX.zeros(0, 1)

    root = problem.root
    root_n = problem.state_dims[root]
    root_x = ca.SX.sym("root_x", root_n)
    root_mult = ca.SX.sym("root_mult", root_n)
    root_dyn = _scale_graph_dyn(problem, root, problem.root_residual(root_x, theta))
    root_xloc = ca.vertcat(root_x, theta)
    root_lag = ca.dot(root_mult, root_dyn)
    root_H, _ = ca.hessian(root_lag, root_xloc)
    root_H = ca.triu(root_H)
    root_C = ca.jacobian(root_dyn, root_xloc) if root_n > 0 else ca.SX.zeros(0, root_xloc.numel())
    root_fun = ca.Function(
        f"{prefix}_root_eval",
        [root_x, theta, root_mult],
        ca.cse([root_dyn, root_H, root_C]),
    )

    edge_funs = []
    for edge_index, edge in enumerate(problem.edges):
        parent_n = problem.state_dims[edge.parent]
        child_n = problem.state_dims[edge.child]
        control_dim = edge.control_dim
        parent_x = ca.SX.sym(f"edge_{edge_index}_parent_x", parent_n)
        child_x = ca.SX.sym(f"edge_{edge_index}_child_x", child_n)
        control_sym = ca.SX.sym(f"edge_{edge_index}_u", max(control_dim, 1))
        control = control_sym[:control_dim] if control_dim > 0 else ca.SX.zeros(0, 1)
        dyn_mult = ca.SX.sym(f"edge_{edge_index}_dyn_mult", child_n)
        dyn = _scale_graph_dyn(
            problem, edge.child, edge.dynamics(parent_x, control, theta) - child_x
        )
        xloc = ca.vertcat(parent_x, child_x, control, theta)
        lag = ca.dot(dyn_mult, dyn)
        H, _ = ca.hessian(lag, xloc)
        H = ca.triu(H)
        C = ca.jacobian(dyn, xloc) if child_n > 0 else ca.SX.zeros(0, xloc.numel())
        edge_funs.append(
            ca.Function(
                f"{prefix}_edge_{edge_index}_eval",
                [parent_x, child_x, control, theta, dyn_mult],
                ca.cse([dyn, H, C]),
            )
        )

    node_funs = []
    for node in range(problem.T + 1):
        state_dim = problem.state_dims[node]
        out_edges = outgoing[node]
        out_control_dim = sum(problem.control_dims[edge] for edge in out_edges)
        c_dim = problem.c_dims[node]
        g_dim = problem.g_dims[node]
        x_node = ca.SX.sym(f"node_{node}_x", state_dim)
        u_out_sym = ca.SX.sym(f"node_{node}_u_out", max(out_control_dim, 1))
        u_out = u_out_sym[:out_control_dim] if out_control_dim > 0 else ca.SX.zeros(0, 1)
        outgoing_controls = _graph_control_slice(u_out, out_edges, problem.control_dims)
        eq_mult = ca.SX.sym(f"node_{node}_eq_mult", c_dim) if c_dim > 0 else ca.SX.zeros(0, 1)
        ineq_mult = ca.SX.sym(f"node_{node}_ineq_mult", g_dim) if g_dim > 0 else ca.SX.zeros(0, 1)
        f = problem.cost(node, x_node, theta)
        eq = (
            problem.equalities(node, x_node, theta, outgoing_controls)
            if c_dim > 0
            else ca.SX.zeros(0, 1)
        )
        g = (
            problem.inequalities(node, x_node, theta, outgoing_controls)
            if g_dim > 0
            else ca.SX.zeros(0, 1)
        )
        xloc = ca.vertcat(x_node, u_out, theta)
        lag = f
        if c_dim > 0:
            lag += ca.dot(eq_mult, eq)
        if g_dim > 0:
            lag += ca.dot(ineq_mult, g)
        H, _ = ca.hessian(lag, xloc)
        H = ca.triu(H)
        C = ca.jacobian(eq, xloc) if c_dim > 0 else ca.SX.zeros(0, xloc.numel())
        G = ca.jacobian(g, xloc) if g_dim > 0 else ca.SX.zeros(0, xloc.numel())
        node_funs.append(
            ca.Function(
                f"{prefix}_node_{node}_eval",
                [x_node, theta, u_out, eq_mult, ineq_mult],
                ca.cse([f, ca.gradient(f, xloc), eq, g, H, C, G]),
            )
        )

    return root_fun, edge_funs, node_funs, outgoing


def _graph_root_col_map(problem):
    state_offsets, _, theta_offset, _ = _graph_offsets(problem)
    root = problem.root
    cols = list(range(state_offsets[root], state_offsets[root] + problem.state_dims[root]))
    cols.extend(range(theta_offset, theta_offset + problem.theta_dim))
    return np.asarray(cols, dtype=int)


def _graph_edge_col_map(problem, edge_index):
    state_offsets, control_offsets, theta_offset, _ = _graph_offsets(problem)
    edge = problem.edges[edge_index]
    cols = list(range(state_offsets[edge.parent], state_offsets[edge.parent] + problem.state_dims[edge.parent]))
    cols.extend(range(state_offsets[edge.child], state_offsets[edge.child] + problem.state_dims[edge.child]))
    cols.extend(range(control_offsets[edge_index], control_offsets[edge_index] + edge.control_dim))
    cols.extend(range(theta_offset, theta_offset + problem.theta_dim))
    return np.asarray(cols, dtype=int)


def _graph_node_col_map(problem, node, outgoing):
    state_offsets, control_offsets, theta_offset, _ = _graph_offsets(problem)
    cols = list(range(state_offsets[node], state_offsets[node] + problem.state_dims[node]))
    for edge_index in outgoing[node]:
        cols.extend(
            range(
                control_offsets[edge_index],
                control_offsets[edge_index] + problem.control_dims[edge_index],
            )
        )
    cols.extend(range(theta_offset, theta_offset + problem.theta_dim))
    return np.asarray(cols, dtype=int)


def _graph_flat_stage_metadata(problem, root_fun, edge_funs, node_funs, outgoing):
    _, _, _, x_dim = _graph_offsets(problem)
    dyn_offsets, c_offsets, y_dim = _graph_y_offsets(problem)
    z_offsets, z_dim = _graph_z_offsets(problem)

    h_entries = []
    c_entries = []
    g_entries = []

    root_cols = _graph_root_col_map(problem)
    root_rows = np.asarray(
        range(dyn_offsets[problem.root], dyn_offsets[problem.root] + problem.state_dims[problem.root]),
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
            range(dyn_offsets[edge.child], dyn_offsets[edge.child] + problem.state_dims[edge.child]),
            dtype=int,
        )
        H = _triu_from_sparsity(edge_fun.sparsity_out(1))
        C = _sp_from_sparsity(edge_fun.sparsity_out(2))
        h_entries.extend(_mapped_entries(H, cols, cols))
        c_entries.extend(_mapped_entries(C, rows, cols))
        edge_meta.append({"cols": cols, "rows": rows, "H": H, "C": C})

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
        node_meta.append({"cols": cols, "c_rows": c_rows, "g_rows": g_rows, "H": H, "C": C, "G": G})

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
        item["h_scatter"] = _scatter_to_global(item["H"], item["cols"], item["cols"], h_sup)
        item["c_scatter"] = _scatter_to_global(item["C"], item["rows"], item["cols"], C)
    for item in node_meta:
        item["h_scatter"] = _scatter_to_global(item["H"], item["cols"], item["cols"], h_sup)
        item["c_scatter"] = _scatter_to_global(item["C"], item["c_rows"], item["cols"], C)
        item["g_scatter"] = _scatter_to_global(item["G"], item["g_rows"], item["cols"], G)

    max_node_out_control_dim = max(
        [sum(problem.control_dims[edge] for edge in outgoing[node]) for node in range(problem.T + 1)]
        + [0]
    )
    max_grad_dim = max([len(item["cols"]) for item in node_meta] + [0])
    max_c_dim = max(problem.state_dims + problem.c_dims + [0])
    max_g_dim = max(problem.g_dims + [0])
    max_h_nnz = max(
        [root_H.nnz]
        + [item["H"].nnz for item in edge_meta]
        + [item["H"].nnz for item in node_meta]
        + [0]
    )
    max_c_nnz = max(
        [root_C.nnz]
        + [item["C"].nnz for item in edge_meta]
        + [item["C"].nnz for item in node_meta]
        + [0]
    )
    max_g_nnz = max([item["G"].nnz for item in node_meta] + [0])
    max_iw = max([root_fun.sz_iw()] + [fun.sz_iw() for fun in edge_funs] + [fun.sz_iw() for fun in node_funs])
    max_w = max([root_fun.sz_w()] + [fun.sz_w() for fun in edge_funs] + [fun.sz_w() for fun in node_funs])

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
        "root": root_meta,
        "edges": edge_meta,
        "nodes": node_meta,
        "outgoing": outgoing,
        "max_node_out_control_dim": max_node_out_control_dim,
        "max_grad_dim": max_grad_dim,
        "max_c_dim": max_c_dim,
        "max_g_dim": max_g_dim,
        "max_h_nnz": max_h_nnz,
        "max_c_nnz": max_c_nnz,
        "max_g_nnz": max_g_nnz,
        "max_iw": max_iw,
        "max_w": max_w,
        "c_to_ct": _transpose_map(C),
        "g_to_gt": _transpose_map(G),
    }


def _emit_graph_flat_split_cpp(problem, metadata, out_dir):
    state_offsets, control_offsets, theta_offset, _ = _graph_offsets(problem)
    dyn_offsets, c_offsets, _ = _graph_y_offsets(problem)
    z_offsets, _ = _graph_z_offsets(problem)
    h_sup = metadata["h_sup"]
    C = metadata["C"]
    G = metadata["G"]
    CT = metadata["CT"]
    GT = metadata["GT"]

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
    for edge_index, item in enumerate(metadata["edges"]):
        arrays += _c_array(f"kEdge{edge_index}HScatter", item["h_scatter"])
        arrays += _c_array(f"kEdge{edge_index}CScatter", item["c_scatter"])
    for node, item in enumerate(metadata["nodes"]):
        arrays += _c_array(f"kNode{node}GradScatter", item["cols"])
        arrays += _c_array(f"kNode{node}HScatter", item["h_scatter"])
        arrays += _c_array(f"kNode{node}CScatter", item["c_scatter"])
        arrays += _c_array(f"kNode{node}GScatter", item["g_scatter"])

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

    def ptr(base, offset, dim):
        return f"{base} + {offset}" if dim > 0 else "&kDummy"

    root = problem.root
    root_n = problem.state_dims[root]
    root_meta = metadata["root"]

    body = []
    body.append(
        f"""
  {{
    const double *arg[] = {{
        {ptr("x", state_offsets[root], root_n)},
        theta,
        {ptr("y", dyn_offsets[root], root_n)},
    }};
    double *res[] = {{local_c, local_H, local_C}};
    graph_flat_root_eval(arg, res, work.iw.data(), work.w.data(), 0);
"""
    )
    for i, row in enumerate(root_meta["rows"]):
        body.append(f"    c[{int(row)}] = local_c[{i}];\n")
    for i in range(root_meta["H"].nnz):
        body.append(f"    H[kRootHScatter[{i}]] += local_H[{i}];\n")
    for i in range(root_meta["C"].nnz):
        body.append(f"    C[kRootCScatter[{i}]] = local_C[{i}];\n")
    body.append("  }\n")

    for edge_index, edge in enumerate(problem.edges):
        item = metadata["edges"][edge_index]
        parent_n = problem.state_dims[edge.parent]
        child_n = problem.state_dims[edge.child]
        control_dim = edge.control_dim
        body.append(
            f"""
  {{
    const double *arg[] = {{
        {ptr("x", state_offsets[edge.parent], parent_n)},
        {ptr("x", state_offsets[edge.child], child_n)},
        {ptr("x", control_offsets[edge_index], control_dim)},
        theta,
        {ptr("y", dyn_offsets[edge.child], child_n)},
    }};
    double *res[] = {{local_c, local_H, local_C}};
    graph_flat_edge_{edge_index}_eval(arg, res, work.iw.data(), work.w.data(), 0);
"""
        )
        for i, row in enumerate(item["rows"]):
            body.append(f"    c[{int(row)}] = local_c[{i}];\n")
        for i in range(item["H"].nnz):
            body.append(f"    H[kEdge{edge_index}HScatter[{i}]] += local_H[{i}];\n")
        for i in range(item["C"].nnz):
            body.append(f"    C[kEdge{edge_index}CScatter[{i}]] = local_C[{i}];\n")
        body.append("  }\n")

    for node in range(problem.T + 1):
        item = metadata["nodes"][node]
        state_dim = problem.state_dims[node]
        c_dim = problem.c_dims[node]
        g_dim = problem.g_dims[node]
        out_edges = metadata["outgoing"][node]
        body.append("  {\n")
        cursor = 0
        for edge_index in out_edges:
            control_dim = problem.control_dims[edge_index]
            if control_dim > 0:
                body.append(
                    f"    std::copy_n(x + {control_offsets[edge_index]}, {control_dim}, local_u + {cursor});\n"
                )
            cursor += control_dim
        body.append(
            f"""    double node_f = 0.0;
    const double *arg[] = {{
        {ptr("x", state_offsets[node], state_dim)},
        theta,
        {("local_u" if cursor > 0 else "&kDummy")},
        {ptr("y", c_offsets[node], c_dim)},
        {ptr("z", z_offsets[node], g_dim)},
    }};
    double *res[] = {{&node_f, local_grad, local_c, local_g, local_H, local_C, local_G}};
    graph_flat_node_{node}_eval(arg, res, work.iw.data(), work.w.data(), 0);
    *f += node_f;
"""
        )
        for i in range(len(item["cols"])):
            body.append(f"    grad_f[kNode{node}GradScatter[{i}]] += local_grad[{i}];\n")
        for i, row in enumerate(item["c_rows"]):
            body.append(f"    c[{int(row)}] = local_c[{i}];\n")
        for i, row in enumerate(item["g_rows"]):
            body.append(f"    g[{int(row)}] = local_g[{i}];\n")
        for i in range(item["H"].nnz):
            body.append(f"    H[kNode{node}HScatter[{i}]] += local_H[{i}];\n")
        for i in range(item["C"].nnz):
            body.append(f"    C[kNode{node}CScatter[{i}]] = local_C[{i}];\n")
        for i in range(item["G"].nnz):
            body.append(f"    G[kNode{node}GScatter[{i}]] = local_G[{i}];\n")
        body.append("  }\n")

    cpp = f"""
#include "problem_definitions/casadi_problems/{problem.name}/generated_flat.hpp"

#include "problem_definitions/casadi_problems/{problem.name}/generated_graph_flat_casadi.h"

#include <algorithm>

namespace sip_examples::problem_definitions::casadi_problems::generated_problem {{
namespace {{

{arrays}

constexpr int kThetaDim = {problem.theta_dim};
constexpr int kThetaOffset = {theta_offset};
constexpr int kFlatIw = {metadata["max_iw"]};
constexpr int kFlatW = {metadata["max_w"]};
constexpr int kMaxGradDim = {metadata["max_grad_dim"]};
constexpr int kMaxCDim = {metadata["max_c_dim"]};
constexpr int kMaxGDim = {metadata["max_g_dim"]};
constexpr int kMaxHNnz = {metadata["max_h_nnz"]};
constexpr int kMaxCNnz = {metadata["max_c_nnz"]};
constexpr int kMaxGNnz = {metadata["max_g_nnz"]};
constexpr int kMaxNodeOutControlDim = {metadata["max_node_out_control_dim"]};
constexpr int kScratchSize = kMaxGradDim + kMaxCDim + kMaxGDim + kMaxHNnz +
                             kMaxCNnz + kMaxGNnz + kMaxNodeOutControlDim;
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
    cpp += f"""
CasadiWork Problem::make_flat_work() {{
  CasadiWork work;
  work.resize(kFlatIw, kFlatW, kScratchSize);
  return work;
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
  const double *theta = kThetaDim > 0 ? x + kThetaOffset : &kDummy;

{"".join(body)}
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
    state_offsets, control_offsets, theta_offset, _ = _graph_offsets(problem)
    dyn_offsets, c_offsets, _ = _graph_y_offsets(problem)
    z_offsets, _ = _graph_z_offsets(problem)

    arrays = ""
    arrays += _cpp_double_array("kInitialOcpX", _graph_initial(problem))
    arrays += _c_array("kStateDims", problem.state_dims)
    arrays += _c_array("kControlDims", problem.control_dims)
    arrays += _c_array("kCDims", problem.c_dims)
    arrays += _c_array("kGDims", problem.g_dims)
    arrays += _c_array("kEdgeParents", [edge.parent for edge in problem.edges])
    arrays += _c_array("kEdgeChildren", [edge.child for edge in problem.edges])

    header = f"""#pragma once

#include "problem_definitions/casadi_problems/common/ocp_problem.hpp"
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

    def ptr(base, offset, dim):
        return f"{base} + {offset}" if dim > 0 else "&kDummy"

    def local_kind(node_control_map, index):
        if index < node_control_map["state_dim"]:
            return ("x", None, index)
        index -= node_control_map["state_dim"]
        if index < len(node_control_map["controls"]):
            return ("u",) + node_control_map["controls"][index]
        index -= len(node_control_map["controls"])
        return ("theta", None, index)

    def add_symmetric_block_assignment(body, target, rows, row, col, value, indent="    "):
        body.append(f"{indent}{target}[{row} + {col} * {rows}] += {value};\n")
        if row != col:
            body.append(f"{indent}{target}[{col} + {row} * {rows}] += {value};\n")

    def scatter_root_hessian(body, item):
        root = problem.root
        n = problem.state_dims[root]
        for value_index, (row, col) in enumerate(zip(*item["H"].entries())):
            value = f"local_H[{value_index}]"
            if row < n and col < n:
                add_symmetric_block_assignment(body, "mco.d2L_dx2[kRoot]", n, row, col, value)
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
                    body, f"mco.d2L_dx2[{edge.parent}]", parent_n, row, col, value
                )
            elif row < parent_end and child_end <= col < control_end:
                u_col = col - child_end
                body.append(
                    f"    mco.d2L_dxdu[{edge_index}][{row} + {u_col} * {parent_n}] += {value};\n"
                )
            elif child_end <= row < control_end and child_end <= col < control_end:
                u_row = row - child_end
                u_col = col - child_end
                add_symmetric_block_assignment(
                    body, f"mco.d2L_du2[{edge_index}]", control_dim, u_row, u_col, value
                )
            elif row < parent_end and col >= control_end:
                theta_col = col - control_end
                body.append(
                    f"    mco.d2L_dxdtheta[{edge.parent}][{row} + {theta_col} * {parent_n}] += {value};\n"
                )
            elif child_end <= row < control_end and col >= control_end:
                u_row = row - child_end
                theta_col = col - control_end
                body.append(
                    f"    mco.d2L_dudtheta[{edge_index}][{u_row} + {theta_col} * {control_dim}] += {value};\n"
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
        for edge_index in metadata["outgoing"][node]:
            for local_u in range(problem.control_dims[edge_index]):
                controls.append((edge_index, local_u))
        return {
            "state_dim": problem.state_dims[node],
            "controls": controls,
        }

    def scatter_node_hessian(body, node, item, control_map):
        n = problem.state_dims[node]
        for value_index, (row, col) in enumerate(zip(*item["H"].entries())):
            value = f"local_H[{value_index}]"
            row_kind, row_edge, row_local = local_kind(control_map, row)
            col_kind, col_edge, col_local = local_kind(control_map, col)
            if row_kind == "x" and col_kind == "x":
                add_symmetric_block_assignment(body, f"mco.d2L_dx2[{node}]", n, row_local, col_local, value)
            elif row_kind == "x" and col_kind == "u":
                body.append(
                    f"    mco.d2L_dxdu[{col_edge}][{row_local} + {col_local} * {n}] += {value};\n"
                )
            elif row_kind == "u" and col_kind == "u":
                if row_edge != col_edge:
                    raise ValueError(
                        f"{problem.name} node {node} has cross-edge control Hessian entries"
                    )
                control_dim = problem.control_dims[row_edge]
                add_symmetric_block_assignment(
                    body, f"mco.d2L_du2[{row_edge}]", control_dim, row_local, col_local, value
                )
            elif row_kind == "x" and col_kind == "theta":
                body.append(
                    f"    mco.d2L_dxdtheta[{node}][{row_local} + {col_local} * {n}] += {value};\n"
                )
            elif row_kind == "u" and col_kind == "theta":
                control_dim = problem.control_dims[row_edge]
                body.append(
                    f"    mco.d2L_dudtheta[{row_edge}][{row_local} + {col_local} * {control_dim}] += {value};\n"
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
                    f"    mco.ddyn_dx[{edge_index}][{row} + {col} * {child_n}] = {value};\n"
                )
            elif child_end <= col < control_end:
                u_col = col - child_end
                body.append(
                    f"    mco.ddyn_du[{edge_index}][{row} + {u_col} * {child_n}] = {value};\n"
                )
            elif col >= control_end:
                theta_col = col - control_end
                body.append(
                    f"    mco.ddyn_dtheta[{edge_index}][{row} + {theta_col} * {child_n}] = {value};\n"
                )

    def scatter_node_jacobian(body, node, item, control_map, matrix_name, target_prefix):
        row_dim = problem.c_dims[node] if target_prefix == "c" else problem.g_dims[node]
        matrix = item["C"] if target_prefix == "c" else item["G"]
        for value_index, (row, col) in enumerate(zip(*matrix.entries())):
            value = f"{matrix_name}[{value_index}]"
            kind, edge_index, local_index = local_kind(control_map, col)
            if kind == "x":
                body.append(
                    f"    mco.d{target_prefix}_dx[{node}][{row} + {local_index} * {row_dim}] = {value};\n"
                )
            elif kind == "u":
                body.append(
                    f"    mco.d{target_prefix}_du[{edge_index}][{row} + {local_index} * {row_dim}] = {value};\n"
                )
            else:
                body.append(
                    f"    mco.d{target_prefix}_dtheta[{node}][{row} + {local_index} * {row_dim}] = {value};\n"
                )

    body = []
    body.append("  mco.f = 0.0;\n")
    body.append("  std::fill_n(mco.df_dtheta, kThetaDim, 0.0);\n")
    body.append("  std::fill_n(mco.d2L_dtheta2, kThetaDim * kThetaDim, 0.0);\n")
    for node in range(problem.T + 1):
        n = problem.state_dims[node]
        c_dim = problem.c_dims[node]
        g_dim = problem.g_dims[node]
        body.append(f"  std::fill_n(mco.df_dx[{node}], {n}, 0.0);\n")
        body.append(f"  std::fill_n(mco.dyn_res[{node}], {n}, 0.0);\n")
        body.append(f"  std::fill_n(mco.c[{node}], {c_dim}, 0.0);\n")
        body.append(f"  std::fill_n(mco.g[{node}], {g_dim}, 0.0);\n")
        body.append(f"  std::fill_n(mco.dc_dx[{node}], {c_dim * n}, 0.0);\n")
        body.append(f"  std::fill_n(mco.dg_dx[{node}], {g_dim * n}, 0.0);\n")
        body.append(f"  std::fill_n(mco.dc_dtheta[{node}], {c_dim} * kThetaDim, 0.0);\n")
        body.append(f"  std::fill_n(mco.dg_dtheta[{node}], {g_dim} * kThetaDim, 0.0);\n")
        body.append(f"  std::fill_n(mco.d2L_dx2[{node}], {n * n}, 0.0);\n")
        body.append(f"  std::fill_n(mco.d2L_dxdtheta[{node}], {n} * kThetaDim, 0.0);\n")
    for edge_index, edge in enumerate(problem.edges):
        parent = edge.parent
        child = edge.child
        parent_n = problem.state_dims[parent]
        child_n = problem.state_dims[child]
        m = edge.control_dim
        c_parent = problem.c_dims[parent]
        g_parent = problem.g_dims[parent]
        body.append(f"  std::fill_n(mco.df_du[{edge_index}], {m}, 0.0);\n")
        body.append(f"  std::fill_n(mco.ddyn_dx[{edge_index}], {child_n * parent_n}, 0.0);\n")
        body.append(f"  std::fill_n(mco.ddyn_du[{edge_index}], {child_n * m}, 0.0);\n")
        body.append(f"  std::fill_n(mco.ddyn_dtheta[{edge_index}], {child_n} * kThetaDim, 0.0);\n")
        body.append(f"  std::fill_n(mco.dc_du[{edge_index}], {c_parent * m}, 0.0);\n")
        body.append(f"  std::fill_n(mco.dg_du[{edge_index}], {g_parent * m}, 0.0);\n")
        body.append(f"  std::fill_n(mco.d2L_dxdu[{edge_index}], {parent_n * m}, 0.0);\n")
        body.append(f"  std::fill_n(mco.d2L_du2[{edge_index}], {m * m}, 0.0);\n")
        body.append(f"  std::fill_n(mco.d2L_dudtheta[{edge_index}], {m} * kThetaDim, 0.0);\n")

    root = problem.root
    root_n = problem.state_dims[root]
    root_meta = metadata["root"]
    body.append(
        f"""
  {{
    const double *arg[] = {{
        mci.states[kRoot],
        theta,
        mci.costates[kRoot],
    }};
    double *res[] = {{local_c, local_H, local_C}};
    graph_ocp_root_eval(arg, res, work.iw.data(), work.w.data(), 0);
"""
    )
    for i in range(root_n):
        body.append(f"    mco.dyn_res[kRoot][{i}] = local_c[{i}];\n")
    scatter_root_hessian(body, root_meta)
    body.append("  }\n")

    for edge_index, edge in enumerate(problem.edges):
        item = metadata["edges"][edge_index]
        parent_n = problem.state_dims[edge.parent]
        child_n = problem.state_dims[edge.child]
        control_dim = edge.control_dim
        body.append(
            f"""
  {{
    const double *arg[] = {{
        mci.states[{edge.parent}],
        mci.states[{edge.child}],
        {("mci.controls[" + str(edge_index) + "]") if control_dim > 0 else "&kDummy"},
        theta,
        mci.costates[{edge.child}],
    }};
    double *res[] = {{local_c, local_H, local_C}};
    graph_ocp_edge_{edge_index}_eval(arg, res, work.iw.data(), work.w.data(), 0);
"""
        )
        for i in range(child_n):
            body.append(f"    mco.dyn_res[{edge.child}][{i}] = local_c[{i}];\n")
        scatter_edge_hessian(body, edge_index, item)
        scatter_edge_jacobian(body, edge_index, item)
        body.append("  }\n")

    for node in range(problem.T + 1):
        item = metadata["nodes"][node]
        state_dim = problem.state_dims[node]
        c_dim = problem.c_dims[node]
        g_dim = problem.g_dims[node]
        out_edges = metadata["outgoing"][node]
        control_map = node_control_map(node)
        body.append("  {\n")
        cursor = 0
        for edge_index in out_edges:
            control_dim = problem.control_dims[edge_index]
            if control_dim > 0:
                body.append(
                    f"    std::copy_n(mci.controls[{edge_index}], {control_dim}, local_u + {cursor});\n"
                )
            cursor += control_dim
        body.append(
            f"""    double node_f = 0.0;
    const double *arg[] = {{
        mci.states[{node}],
        theta,
        {("local_u" if cursor > 0 else "&kDummy")},
        {("mci.equality_constraint_multipliers[" + str(node) + "]") if c_dim > 0 else "&kDummy"},
        {("mci.inequality_constraint_multipliers[" + str(node) + "]") if g_dim > 0 else "&kDummy"},
    }};
    double *res[] = {{&node_f, local_grad, local_c, local_g, local_H, local_C, local_G}};
    graph_ocp_node_{node}_eval(arg, res, work.iw.data(), work.w.data(), 0);
    mco.f += node_f;
"""
        )
        for i, col in enumerate(item["cols"]):
            if col in range(state_offsets[node], state_offsets[node] + state_dim):
                body.append(f"    mco.df_dx[{node}][{col - state_offsets[node]}] += local_grad[{i}];\n")
            elif theta_offset <= col < theta_offset + problem.theta_dim:
                body.append(f"    mco.df_dtheta[{col - theta_offset}] += local_grad[{i}];\n")
            else:
                for edge_index in out_edges:
                    start = control_offsets[edge_index]
                    stop = start + problem.control_dims[edge_index]
                    if start <= col < stop:
                        body.append(f"    mco.df_du[{edge_index}][{col - start}] += local_grad[{i}];\n")
                        break
                else:
                    raise ValueError(f"unexpected node gradient column in {problem.name}")
        for i in range(c_dim):
            body.append(f"    mco.c[{node}][{i}] = local_c[{i}];\n")
        for i in range(g_dim):
            body.append(f"    mco.g[{node}][{i}] = local_g[{i}];\n")
        scatter_node_hessian(body, node, item, control_map)
        scatter_node_jacobian(body, node, item, control_map, "local_C", "c")
        scatter_node_jacobian(body, node, item, control_map, "local_G", "g")
        body.append("  }\n")

    cpp = f"""
#include "problem_definitions/casadi_problems/{problem.name}/generated_ocp.hpp"

#include "problem_definitions/casadi_problems/{problem.name}/generated_graph_ocp_casadi.h"

#include <algorithm>

namespace sip_examples::problem_definitions::casadi_problems::generated_problem {{
namespace {{

{arrays}

constexpr int kRoot = {problem.root};
constexpr int kNumEdges = {problem.T};
constexpr int kThetaDim = {problem.theta_dim};
constexpr int kOcpIw = {metadata["max_iw"]};
constexpr int kOcpW = {metadata["max_w"]};
constexpr int kMaxGradDim = {metadata["max_grad_dim"]};
constexpr int kMaxCDim = {metadata["max_c_dim"]};
constexpr int kMaxGDim = {metadata["max_g_dim"]};
constexpr int kMaxHNnz = {metadata["max_h_nnz"]};
constexpr int kMaxCNnz = {metadata["max_c_nnz"]};
constexpr int kMaxGNnz = {metadata["max_g_nnz"]};
constexpr int kMaxNodeOutControlDim = {metadata["max_node_out_control_dim"]};
constexpr int kScratchSize = kMaxGradDim + kMaxCDim + kMaxGDim + kMaxHNnz +
                             kMaxCNnz + kMaxGNnz + kMaxNodeOutControlDim;
constexpr double kDummy = 0.0;

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
  const double *theta = kThetaDim > 0 ? mci.theta : &kDummy;

{"".join(body)}
}}

}} // namespace sip_examples::problem_definitions::casadi_problems::generated_problem
"""
    with open(os.path.join(out_dir, "generated_ocp.cpp"), "w") as f:
        f.write(cpp)


def generate_graph_flat(problem, out_dir, emit_kkt_code):
    os.makedirs(out_dir, exist_ok=True)
    root_fun, edge_funs, node_funs, outgoing = _build_graph_flat_stage_functions(problem)
    metadata = _graph_flat_stage_metadata(problem, root_fun, edge_funs, node_funs, outgoing)
    old_cwd = os.getcwd()
    try:
        os.chdir(out_dir)
        cg = ca.CodeGenerator("generated_graph_flat_casadi.c", {"with_header": True})
        cg.add(root_fun)
        for fun in edge_funs:
            cg.add(fun)
        for fun in node_funs:
            cg.add(fun)
        cg.generate()
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
    root_fun, edge_funs, node_funs, outgoing = _build_graph_flat_stage_functions(
        problem, prefix="graph_ocp"
    )
    metadata = _graph_flat_stage_metadata(problem, root_fun, edge_funs, node_funs, outgoing)
    old_cwd = os.getcwd()
    try:
        os.chdir(out_dir)
        cg = ca.CodeGenerator("generated_graph_ocp_casadi.c", {"with_header": True})
        cg.add(root_fun)
        for fun in edge_funs:
            cg.add(fun)
        for fun in node_funs:
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
