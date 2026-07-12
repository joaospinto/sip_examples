#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 9 ]]; then
  echo "usage: decode_problem WRAPPER DECODER SIF NAME OUTSDIF ELFUN GROUP RANGE EXTER" >&2
  exit 2
fi

execroot="$PWD"
absolute_path() {
  if [[ "$1" = /* ]]; then
    printf '%s\n' "$1"
  else
    printf '%s/%s\n' "$execroot" "$1"
  fi
}

wrapper="$(absolute_path "$1")"
decoder="$(absolute_path "$2")"
sif="$(absolute_path "$3")"
name="$4"
outsdif="$(absolute_path "$5")"
elfun="$(absolute_path "$6")"
group="$(absolute_path "$7")"
range="$(absolute_path "$8")"
extern="$(absolute_path "$9")"

scratch="$(mktemp -d "${TMPDIR:-/tmp}/sip-cutest-decode.XXXXXX")"
trap 'rm -rf "$scratch"' EXIT
cd "$scratch"
"$wrapper" "$decoder" -dp -suffix "$sif"

cp "OUTSDIF_${name}.d" "$outsdif"
cp "ELFUN_${name}.f" "$elfun"
cp "GROUP_${name}.f" "$group"
cp "RANGE_${name}.f" "$range"
if [[ -f "EXTER_${name}.f" ]]; then
  cp "EXTER_${name}.f" "$extern"
else
  : >"$extern"
fi
