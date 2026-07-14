load("@rules_cc//cc:defs.bzl", "cc_library")
load("@rules_python//python:defs.bzl", "py_binary", "py_library")
load("@slacg//:kkt_codegen.bzl", "kkt_codegen_files", "kkt_codegen_sources")


def casadi_problem(
        codegen_deps,
        graph = False,
        codegen_library = None,
        emit_kkt = True,
        num_factor_chunks = 8,
        num_solve_chunks = 8,
        num_product_chunks = 8):
    """Defines code-generation and C++ libraries for one CasADi problem."""
    if codegen_library != None:
        py_library(
            name = codegen_library,
            srcs = ["codegen.py"],
            deps = codegen_deps,
            visibility = ["//visibility:public"],
        )
        binary_deps = [":" + codegen_library]
    else:
        binary_deps = codegen_deps

    py_binary(
        name = "codegen",
        srcs = ["codegen.py"],
        deps = binary_deps,
    )

    if emit_kkt:
        py_binary(
            name = "kkt_codegen_tool",
            srcs = ["codegen.py"],
            main = "codegen.py",
            deps = binary_deps + ["@slacg//slacg:kkt_codegen"],
        )

    casadi_stem = "generated_graph_{}_casadi" if graph else "generated_{}_casadi"
    native.genrule(
        name = "flat_codegen_genrule",
        outs = [
            "generated_flat.hpp",
            "generated_flat.cpp",
            casadi_stem.format("flat") + ".h",
            casadi_stem.format("flat") + ".c",
            casadi_stem.format("flat_values") + ".h",
            casadi_stem.format("flat_values") + ".c",
        ],
        cmd = "$(location :codegen) --mode flat $(RULEDIR)",
        tools = [":codegen"],
    )

    if emit_kkt:
        native.genrule(
            name = "kkt_codegen_genrule",
            outs = kkt_codegen_files(
                "kkt_codegen",
                num_factor_chunks,
                num_solve_chunks,
                num_product_chunks,
            ),
            cmd = "$(location :kkt_codegen_tool) --mode kkt --num-factor-chunks {} --num-solve-chunks {} --num-product-chunks {} $(RULEDIR)".format(
                num_factor_chunks,
                num_solve_chunks,
                num_product_chunks,
            ),
            tools = [":kkt_codegen_tool"],
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
            "-Wno-unused-variable",
        ],
        deps = [
            "//problem_definitions/casadi_problems/common",
            "@sip//sip",
            "@sip_optimal_control//sip_optimal_control",
        ],
        visibility = ["//visibility:public"],
    )

    if emit_kkt:
        cc_library(
            name = "kkt_codegen",
            srcs = kkt_codegen_sources(
                "kkt_codegen",
                num_factor_chunks,
                num_solve_chunks,
                num_product_chunks,
            ),
            hdrs = ["kkt_codegen.hpp"],
            copts = ["-Wno-unused-parameter"],
            visibility = ["//visibility:public"],
        )
