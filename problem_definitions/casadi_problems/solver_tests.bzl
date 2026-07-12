load("@rules_cc//cc:defs.bzl", "cc_test")


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
