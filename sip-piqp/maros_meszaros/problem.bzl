"""Defines manual PIQP comparison tests for Maros-Meszaros problems."""

_RUNNER = "//problem_definitions/cutest_problems:piqp_runner"


def piqp_test(problem):
    outsdif = "@maros_meszaros//:_cutest/{}/OUTSDIF.d".format(problem)
    problem_library = "@maros_meszaros//:_cutest/{}/libproblem.so".format(problem)
    native.sh_test(
        name = problem.lower(),
        srcs = ["@cutest_tools//:run_test.sh"],
        args = [
            "$(rootpath {})".format(_RUNNER),
            "$(rootpath @cutest_tools//:runtime)",
            "$(rootpath {})".format(problem_library),
            "$(rootpath {})".format(outsdif),
            "1",
        ],
        data = [
            outsdif,
            problem_library,
            _RUNNER,
            "@cutest_tools//:runtime",
        ],
        tags = ["manual"],
    )


def piqp_tests(problems):
    for problem in problems:
        piqp_test(problem)
