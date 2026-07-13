#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 || $# -gt 4 ]]; then
  echo "usage: classify_corpus REPOSITORY OUTPUT_DIRECTORY [BATCH_SIZE] [JOBS]" >&2
  exit 2
fi

repository="$1"
output_directory="$2"
batch_size="${3:-20}"
jobs="${4:-2}"
shard_count="${SIP_CORPUS_SHARD_COUNT:-1}"
shard_index="${SIP_CORPUS_SHARD_INDEX:-0}"
problems_file="${SIP_CORPUS_PROBLEMS_FILE:-}"
results="$output_directory/results.tsv"
failures="$output_directory/failures"

if ! [[ "$shard_count" =~ ^[1-9][0-9]*$ ]] ||
    ! [[ "$shard_index" =~ ^[0-9]+$ ]] ||
    ((shard_index >= shard_count)); then
  echo "invalid corpus shard ${shard_index}/${shard_count}" >&2
  exit 2
fi
if [[ -n "$problems_file" && ! -f "$problems_file" ]]; then
  echo "problem list does not exist: $problems_file" >&2
  exit 2
fi

mkdir -p "$failures"
if [[ ! -f "$results" ]]; then
  printf 'problem\tstatus\titerations\tls_iterations\tduration_ms\n' >"$results"
fi

targets=()
target_index=0
if [[ "$repository" == //* || "$repository" == @*//* ]]; then
  query_pattern="${repository}:*"
else
  query_pattern="@${repository}//:*"
fi
while IFS= read -r target; do
  problem="${target##*:}"
  if [[ -n "$problems_file" ]] && ! grep -Fqx "$problem" "$problems_file"; then
    continue
  fi
  if ((target_index % shard_count == shard_index)); then
    targets+=("$target")
  fi
  target_index="$((target_index + 1))"
done < <(bazel query "kind(\".*_test\", ${query_pattern})" --output=label)

test_env_args=()
if [[ -n "${SIP_CUTEST_FILTER_MIN_LS:-}" ]]; then
  test_env_args+=("--test_env=SIP_CUTEST_FILTER_MIN_LS=${SIP_CUTEST_FILTER_MIN_LS}")
fi
if [[ -n "${SIP_CUTEST_ITERATIVE_REFINEMENT_STEPS:-}" ]]; then
  test_env_args+=("--test_env=SIP_CUTEST_ITERATIVE_REFINEMENT_STEPS=${SIP_CUTEST_ITERATIVE_REFINEMENT_STEPS}")
fi
if [[ -n "${PIQP_ABLATION:-}" ]]; then
  test_env_args+=("--test_env=PIQP_ABLATION=${PIQP_ABLATION}")
fi
if [[ -n "${IPOPT_ABLATION:-}" ]]; then
  test_env_args+=("--test_env=IPOPT_ABLATION=${IPOPT_ABLATION}")
fi
if [[ -n "${SIP_CUTEST_PREDICTOR_CORRECTOR:-}" ]]; then
  test_env_args+=("--test_env=SIP_CUTEST_PREDICTOR_CORRECTOR=${SIP_CUTEST_PREDICTOR_CORRECTOR}")
fi
if [[ -n "${SIP_CUTEST_INITIAL_PENALTY:-}" ]]; then
  test_env_args+=("--test_env=SIP_CUTEST_INITIAL_PENALTY=${SIP_CUTEST_INITIAL_PENALTY}")
fi
if [[ -n "${SIP_CUTEST_REGULARIZATION_DECREASE_FACTOR:-}" ]]; then
  test_env_args+=("--test_env=SIP_CUTEST_REGULARIZATION_DECREASE_FACTOR=${SIP_CUTEST_REGULARIZATION_DECREASE_FACTOR}")
fi

pending=()
for target in "${targets[@]}"; do
  problem="${target##*:}"
  if ! awk -F '\t' -v problem="$problem" 'NR > 1 && $1 == problem { found = 1 } END { exit !found }' "$results"; then
    pending+=("$target")
  fi
done

total="${#targets[@]}"
completed="$((total - ${#pending[@]}))"
printf 'repository=%s shard=%d/%d corpus_total=%d total=%d completed=%d pending=%d\n' \
  "$repository" "$shard_index" "$shard_count" "$target_index" "$total" \
  "$completed" "${#pending[@]}"

for ((begin = 0; begin < ${#pending[@]}; begin += batch_size)); do
  batch=("${pending[@]:begin:batch_size}")
  batch_number="$((begin / batch_size + 1))"
  bep="$output_directory/batch-${batch_number}.bep.json"
  batch_log="$output_directory/batch-${batch_number}.log"

  set +e
  bazel test "${batch[@]}" \
    ${test_env_args[@]+"${test_env_args[@]}"} \
    --build_event_json_file="$bep" \
    --cache_test_results=no \
    --jobs="$jobs" \
    --keep_going \
    --local_test_jobs="$jobs" \
    --test_output=errors \
    --test_timeout=120 >"$batch_log" 2>&1
  bazel_status="$?"
  set -e

  observed="$output_directory/batch-${batch_number}.observed"
  : >"$observed"
  while IFS=$'\t' read -r label status duration uri; do
    if [[ -z "$status" ]]; then
      continue
    fi
    problem="${label##*:}"
    log_path="${uri#file://}"
    iterations=""
    ls_iterations=""
    if [[ -f "$log_path" ]]; then
      stats="$(sed -n 's/.*iterations=\([0-9][0-9]*\) ls_iterations=\([0-9][0-9]*\).*/\1\t\2/p' "$log_path" | tail -n 1)"
      if [[ -n "$stats" ]]; then
        iterations="${stats%%$'\t'*}"
        ls_iterations="${stats##*$'\t'}"
      fi
      if [[ "$status" != "PASSED" ]]; then
        cp "$log_path" "$failures/${problem}.log"
      fi
    fi
    printf '%s\t%s\t%s\t%s\t%s\n' \
      "$problem" "$status" "$iterations" "$ls_iterations" "$duration" >>"$results"
    printf '%s\n' "$problem" >>"$observed"
  done < <(
    jq -r '
      select(.id.testResult) |
      [
        .id.testResult.label,
        .testResult.status,
        (.testResult.testAttemptDurationMillis // ""),
        ([.testResult.testActionOutput[]? | select(.name == "test.log") | .uri][0] // "")
      ] | @tsv
    ' "$bep"
  )

  for target in "${batch[@]}"; do
    problem="${target##*:}"
    if ! grep -Fqx "$problem" "$observed"; then
      printf '%s\tBUILD_FAILED\t\t\t\n' "$problem" >>"$results"
      printf 'See %s\n' "$batch_log" >"$failures/${problem}.build.log"
    fi
  done

  completed="$((completed + ${#batch[@]}))"
  pass_count="$(awk -F '\t' '$2 == "PASSED" { count++ } END { print count + 0 }' "$results")"
  fail_count="$((completed - pass_count))"
  printf 'batch=%d bazel_status=%d completed=%d/%d passed=%d failed=%d\n' \
    "$batch_number" "$bazel_status" "$completed" "$total" "$pass_count" "$fail_count"

  rm -f "$bep" "$observed"
  if [[ "$bazel_status" -eq 0 ]]; then
    rm -f "$batch_log"
  fi
  available_kib="$(df -Pk . | awk 'NR == 2 { print $4 }')"
  if ((available_kib < 2 * 1024 * 1024)); then
    bazel clean >/dev/null
  fi
done

sorted="$output_directory/results.sorted.tsv"
head -n 1 "$results" >"$sorted"
tail -n +2 "$results" | LC_ALL=C sort -t $'\t' -k1,1 >>"$sorted"
mv "$sorted" "$results"

build_failure_count="$(awk -F '\t' '$2 == "BUILD_FAILED" { count++ } END { print count + 0 }' "$results")"
if ((build_failure_count > 0)); then
  printf 'build_failures=%d\n' "$build_failure_count" >&2
  exit 1
fi
