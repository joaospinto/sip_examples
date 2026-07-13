load("@rules_cc//cc:defs.bzl", "cc_binary", "cc_test")
load("@rules_python//python:defs.bzl", "py_test")
load("@sip_examples_pip//:requirements.bzl", "requirement")


def sip_optimal_control_test(problem, corpus, tags = [], size = "medium"):
    problem_path = "problem_definitions/casadi_problems/{}/{}".format(corpus, problem)
    cc_test(
        name = problem,
        size = size,
        srcs = ["//problem_definitions/casadi_problems/common:ocp_test.cpp"],
        copts = ["-DGENERATED_HEADER=\\\"{}/generated_ocp.hpp\\\"".format(problem_path)],
        tags = tags,
        deps = [
            "//problem_definitions/casadi_problems/common:ocp_common",
            "//{}:ocp_generated_problem".format(problem_path),
            "@googletest//:gtest",
            "@googletest//:gtest_main",
            "@sip//sip",
            "@sip_optimal_control//sip_optimal_control",
        ],
    )


def sip_qdldl_test(problem, corpus, tags = [], size = "medium"):
    problem_path = "problem_definitions/casadi_problems/{}/{}".format(corpus, problem)
    cc_test(
        name = problem,
        size = size,
        srcs = ["//problem_definitions/casadi_problems/common:flat_qdldl_test.cpp"],
        copts = ["-DGENERATED_HEADER=\\\"{}/generated_flat.hpp\\\"".format(problem_path)],
        tags = tags,
        deps = [
            "//problem_definitions/casadi_problems/common:flat_qdldl_common",
            "//{}:flat_generated_problem".format(problem_path),
            "@googletest//:gtest",
            "@googletest//:gtest_main",
            "@qdldl//:qdldl",
            "@sip//sip",
            "@sip_qdldl//sip_qdldl",
        ],
    )


def ipopt_test(problem, corpus, tags = [], size = "medium"):
    problem_path = "problem_definitions/casadi_problems/{}/{}".format(corpus, problem)
    native_name = problem + "_native"
    cc_binary(
        name = native_name,
        srcs = ["//problem_definitions/casadi_problems/common:flat_ipopt_main.cpp"],
        copts = ["-DGENERATED_HEADER=\\\"{}/generated_flat.hpp\\\"".format(problem_path)],
        deps = [
            "//problem_definitions/casadi_problems/common:flat_ipopt_model",
            "//problem_definitions/ipopt",
            "//{}:flat_generated_problem".format(problem_path),
        ],
    )
    py_test(
        name = problem,
        size = size,
        srcs = ["//problem_definitions/casadi_problems/common:ipopt_runner.py"],
        main = "//problem_definitions/casadi_problems/common:ipopt_runner.py",
        args = ["$(rootpath :{})".format(native_name)],
        data = [":" + native_name],
        deps = [requirement("casadi")],
        tags = tags,
    )


def model_equivalence_test(problem, corpus, tags = [], size = "medium"):
    problem_path = "problem_definitions/casadi_problems/{}/{}".format(corpus, problem)
    flat_binary = "{}_flat_model_signature".format(problem)
    ocp_binary = "{}_ocp_model_signature".format(problem)
    cc_binary(
        name = flat_binary,
        srcs = ["//problem_definitions/casadi_problems/common:flat_model_signature_main.cpp"],
        copts = ["-DGENERATED_HEADER=\\\"{}/generated_flat.hpp\\\"".format(problem_path)],
        deps = [
            "//problem_definitions/casadi_problems/common:model_signature",
            "//{}:flat_generated_problem".format(problem_path),
        ],
    )
    cc_binary(
        name = ocp_binary,
        srcs = ["//problem_definitions/casadi_problems/common:ocp_model_signature_main.cpp"],
        copts = ["-DGENERATED_HEADER=\\\"{}/generated_ocp.hpp\\\"".format(problem_path)],
        deps = [
            "//problem_definitions/casadi_problems/common:model_signature",
            "//problem_definitions/casadi_problems/common:ocp_common",
            "//{}:ocp_generated_problem".format(problem_path),
            "@sip_optimal_control//sip_optimal_control:helpers",
        ],
    )
    py_test(
        name = "{}_model_equivalence".format(problem),
        size = size,
        srcs = ["//problem_definitions/casadi_problems/common:model_signature_test.py"],
        main = "//problem_definitions/casadi_problems/common:model_signature_test.py",
        args = [
            "$(rootpath :{})".format(flat_binary),
            "$(rootpath :{})".format(ocp_binary),
        ],
        data = [
            ":" + flat_binary,
            ":" + ocp_binary,
        ],
        tags = tags,
    )
