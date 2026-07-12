#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 6 ]]; then
  echo "usage: compile_problem WRAPPER OUTPUT ELFUN GROUP RANGE EXTER" >&2
  exit 2
fi

wrapper="$1"
output="$2"
shift 2
sources=("$1" "$2" "$3" "$4")

flags=(-shared -fPIC)
if [[ "$(uname -s)" == "Darwin" ]]; then
  flags+=(-Wl,-undefined,dynamic_lookup)
fi
"$wrapper" "${flags[@]}" "${sources[@]}" -o "$output"
