load("@rules_cc//cc:defs.bzl", "cc_library")
load("@rules_python//python:defs.bzl", "py_binary", "py_library")
load("@sip_examples_pip//:requirements.bzl", "requirement")


def dymos_casadi_problem(
        graph = False,
        codegen_deps = None,
        codegen_library = None,
        extra_codegen_deps = [],
        emit_kkt = True):
    """Defines code-generation and C++ libraries for one Dymos problem."""
    default_codegen_deps = [
        "//problem_definitions/casadi_problems:codegen_common",
        "//problem_definitions/casadi_problems/dymos:common",
        requirement("casadi"),
        requirement("numpy"),
    ]
    deps = (default_codegen_deps if codegen_deps == None else codegen_deps) + extra_codegen_deps

    if codegen_library != None:
        py_library(
            name = codegen_library,
            srcs = ["codegen.py"],
            deps = deps,
            visibility = ["//visibility:public"],
        )
        binary_deps = [":" + codegen_library]
    else:
        binary_deps = deps

    py_binary(
        name = "codegen",
        srcs = ["codegen.py"],
        deps = binary_deps,
    )

    casadi_stem = "generated_graph_{}_casadi" if graph else "generated_{}_casadi"
    flat_outs = [
        "generated_flat.hpp",
        "generated_flat.cpp",
        casadi_stem.format("flat") + ".h",
        casadi_stem.format("flat") + ".c",
        casadi_stem.format("flat_values") + ".h",
        casadi_stem.format("flat_values") + ".c",
    ]
    flat_command = "$(location :codegen) --mode flat $(RULEDIR)"
    if emit_kkt and not graph:
        flat_outs.extend(["kkt_codegen.hpp", "kkt_codegen.cpp"])
        flat_command = "$(location :codegen) --mode flat --emit-kkt-code $(RULEDIR)"

    native.genrule(
        name = "flat_codegen_genrule",
        outs = flat_outs,
        cmd = flat_command,
        tools = [":codegen"],
    )

    native.genrule(
        name = "ocp_codegen_genrule",
        outs = [
            "generated_ocp.hpp",
            "generated_ocp.cpp",
            casadi_stem.format("ocp") + ".h",
            casadi_stem.format("ocp") + ".c",
        ],
        cmd = "$(location :codegen) --mode ocp $(RULEDIR)",
        tools = [":codegen"],
    )

    cc_library(
        name = "flat_generated_problem",
        srcs = [
            "generated_flat.cpp",
            casadi_stem.format("flat") + ".c",
            casadi_stem.format("flat_values") + ".c",
        ],
        hdrs = [
            "generated_flat.hpp",
            casadi_stem.format("flat") + ".h",
            casadi_stem.format("flat_values") + ".h",
        ],
        copts = [
            "-O0",
            "-Wno-unused-variable",
        ],
        deps = [
            "//problem_definitions/casadi_problems/common",
            "@sip//sip",
        ],
        visibility = ["//visibility:public"],
    )

    cc_library(
        name = "ocp_generated_problem",
        srcs = [
            "generated_ocp.cpp",
            casadi_stem.format("ocp") + ".c",
        ],
        hdrs = [
            "generated_ocp.hpp",
            casadi_stem.format("ocp") + ".h",
        ],
        copts = [
            "-O0",
            "-Wno-unused-variable",
        ],
        deps = [
            "//problem_definitions/casadi_problems/common",
            "@sip//sip",
            "@sip_optimal_control//sip_optimal_control",
        ],
        visibility = ["//visibility:public"],
    )

    if emit_kkt and not graph:
        cc_library(
            name = "kkt_codegen",
            srcs = ["kkt_codegen.cpp"],
            hdrs = ["kkt_codegen.hpp"],
            copts = ["-Wno-unused-parameter"],
            visibility = ["//visibility:public"],
        )
