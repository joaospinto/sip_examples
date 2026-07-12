"""Defines IPOPT comparison tests for generated CUTEst repositories."""

_RUNNER = "@//problem_definitions/cutest_problems:ipopt_runner"


def ipopt_test(problem, use_qp_settings):
    outsdif = "//:_cutest/{}/OUTSDIF.d".format(problem)
    problem_library = "//:_cutest/{}/libproblem.so".format(problem)
    native.sh_test(
        name = problem,
        srcs = ["@cutest_tools//:run_test.sh"],
        args = [
            "$(rootpath {})".format(_RUNNER),
            "$(rootpath @cutest_tools//:runtime)",
            "$(rootpath {})".format(problem_library),
            "$(rootpath {})".format(outsdif),
            "1" if use_qp_settings else "0",
        ],
        data = [
            outsdif,
            problem_library,
            _RUNNER,
            "@cutest_tools//:runtime",
        ],
        tags = ["manual"],
    )


def ipopt_tests(problems, use_qp_settings):
    for problem in problems:
        ipopt_test(problem, use_qp_settings)
