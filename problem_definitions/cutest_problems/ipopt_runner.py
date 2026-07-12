import os
from pathlib import Path
import platform
import sys

import casadi
from python.runfiles import runfiles


def main() -> None:
    native_runner = runfiles.Create().Rlocation(
        "_main/problem_definitions/cutest_problems/ipopt_native_runner"
    )
    if native_runner is None:
        raise RuntimeError("could not locate the native IPOPT runner")

    casadi_directory = Path(casadi.__file__).parent
    candidates = (
        casadi_directory / "libipopt.dylib",
        casadi_directory / "libipopt.so",
        casadi_directory / "ipopt.dll",
    )
    ipopt_library = next((path for path in candidates if path.exists()), None)
    if ipopt_library is None:
        raise RuntimeError("the CasADi package does not contain IPOPT")

    runtime_directory = os.environ.get("SIP_CUTEST_FORTRAN_RUNTIME_DIR")
    if runtime_directory is not None:
        loader_variable = (
            "DYLD_LIBRARY_PATH" if platform.system() == "Darwin" else "LD_LIBRARY_PATH"
        )
        existing = os.environ.get(loader_variable)
        os.environ[loader_variable] = (
            runtime_directory
            if existing is None
            else f"{runtime_directory}:{existing}"
        )

    os.execv(
        native_runner,
        [native_runner, str(ipopt_library), *sys.argv[1:]],
    )


if __name__ == "__main__":
    main()
