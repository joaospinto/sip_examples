import numpy as np

from problem_definitions.casadi_problems.codegen_common import main as codegen_main
from problem_definitions.casadi_problems.dymos.water_rocket_range.codegen import (
    DEG_TO_RAD,
    make_water_rocket_problem,
)


def make_problem():
    return make_water_rocket_problem(
        "dymos/water_rocket_height",
        "height",
        np.array(
            [
                0.1050,
                2.8950,
                3.7095,
                85.0 * DEG_TO_RAD,
                0.98,
                0.144,
            ]
        ),
    )


if __name__ == "__main__":
    codegen_main(make_problem)
