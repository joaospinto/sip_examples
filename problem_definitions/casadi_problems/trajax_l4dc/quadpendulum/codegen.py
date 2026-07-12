from problem_definitions.casadi_problems.codegen_common import main as codegen_main
from problem_definitions.casadi_problems.trajax_l4dc.quadpendulum_problem import (
    make_quadpendulum_problem,
)


def make_problem():
    return make_quadpendulum_problem(False)


if __name__ == "__main__":
    codegen_main(make_problem)
