load("@sip_examples_pip//:requirements.bzl", "requirement")
load("//problem_definitions/casadi_problems:problem.bzl", _casadi_problem = "casadi_problem")


def dymos_casadi_problem(
        graph = False,
        codegen_deps = None,
        codegen_library = None,
        extra_codegen_deps = [],
        emit_kkt = True,
        num_factor_chunks = 8,
        num_solve_chunks = 8,
        num_product_chunks = 8):
    """Defines code-generation and C++ libraries for one Dymos problem."""
    default_codegen_deps = [
        "//problem_definitions/casadi_problems:codegen_common",
        "//problem_definitions/casadi_problems/dymos:common",
        requirement("casadi"),
        requirement("numpy"),
    ]
    deps = (default_codegen_deps if codegen_deps == None else codegen_deps) + extra_codegen_deps
    _casadi_problem(
        codegen_deps = deps,
        graph = graph,
        codegen_library = codegen_library,
        emit_kkt = emit_kkt,
        num_factor_chunks = num_factor_chunks,
        num_solve_chunks = num_solve_chunks,
        num_product_chunks = num_product_chunks,
    )
