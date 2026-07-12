from problem_definitions.casadi_problems.codegen_common import main as codegen_main
from problem_definitions.casadi_problems.dymos.shuttle_reentry_unconstrained.codegen import (
    make_problem as make_shuttle_reentry_problem,
)


def make_problem():
    return make_shuttle_reentry_problem(
        constrained=True,
        name="dymos/shuttle_reentry_constrained",
    )


if __name__ == "__main__":
    codegen_main(make_problem)
