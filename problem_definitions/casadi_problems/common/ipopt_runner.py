import os
from pathlib import Path
import sys

import casadi


def main() -> None:
    if len(sys.argv) != 2:
        raise RuntimeError("usage: ipopt_runner NATIVE_RUNNER")

    casadi_directory = Path(casadi.__file__).parent
    candidates = (
        casadi_directory / "libipopt.dylib",
        casadi_directory / "libipopt.so",
        casadi_directory / "ipopt.dll",
    )
    ipopt_library = next((path for path in candidates if path.exists()), None)
    if ipopt_library is None:
        raise RuntimeError("the CasADi package does not contain IPOPT")

    native_runner = os.path.abspath(sys.argv[1])
    os.execv(native_runner, [native_runner, str(ipopt_library)])


if __name__ == "__main__":
    main()
