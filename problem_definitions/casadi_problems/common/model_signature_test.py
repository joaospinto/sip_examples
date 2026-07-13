import math
import subprocess
import sys


def read_signature(executable):
    output = subprocess.run(
        [executable], check=True, capture_output=True, text=True
    ).stdout
    signature = {}
    for line in output.splitlines():
        fields = line.split()
        if not fields:
            continue
        name = fields[0]
        size = int(fields[1])
        values = [float(value) for value in fields[2:]]
        if len(values) != size:
            raise RuntimeError(
                f"{name} reported {size} values but emitted {len(values)}"
            )
        signature[name] = values
    return signature


def main():
    flat = read_signature(sys.argv[1])
    ocp = read_signature(sys.argv[2])
    if flat.keys() != ocp.keys():
        raise RuntimeError(f"signature fields differ: {flat.keys()} != {ocp.keys()}")

    failed = False
    for name in flat:
        if len(flat[name]) != len(ocp[name]):
            print(f"{name}: lengths differ", file=sys.stderr)
            failed = True
            continue
        max_error = 0.0
        for index, (flat_value, ocp_value) in enumerate(zip(flat[name], ocp[name])):
            scale = max(1.0, abs(flat_value), abs(ocp_value))
            error = abs(flat_value - ocp_value) / scale
            max_error = max(max_error, error)
            if not math.isfinite(error) or error > 1.0e-10:
                print(
                    f"{name}[{index}]: flat={flat_value:.17g} "
                    f"ocp={ocp_value:.17g} relative_error={error:.3e}",
                    file=sys.stderr,
                )
                failed = True
        print(f"{name}: max_relative_error={max_error:.3e}")
    return int(failed)


if __name__ == "__main__":
    raise SystemExit(main())
