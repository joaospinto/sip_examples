"""Pinned CUTEst, SIFDecode, and SIF corpus repositories."""

load("//problem_definitions/cutest_problems:classification.bzl", "CUTEST_ENABLED_PROBLEMS")

_CUTEST_VERSION = "2.7.1"
_SIFDECODE_VERSION = "3.1.1"
_SIF_COMMIT = "ba729aec456b4a38f02917a305d8719b26b4c9fc"

_PLATFORM_ASSETS = {
    ("mac os x", "aarch64"): (
        "aarch64-apple-darwin",
        "36d941094b09a55172bcaf4f017446c8513858f6433d4c41b73a1648cf87475e",
        "d0ca2c9a6a7230c2a072704ad9e45435fcb7f48154002e869cace0ecc3f18681",
    ),
    ("mac os x", "x86_64"): (
        "x86_64-apple-darwin",
        "343ea8f59254231c554b5a3f2a15460ea0b73ae19e13dff2bfe4157a4f703c34",
        "b95d7e138e444fd1c968874eb1ce290e6057ad5263f37ff547b87c97dddc958c",
    ),
    ("linux", "aarch64"): (
        "aarch64-linux-gnu",
        "a789c850bb5bbc5a545c28b1d490a8384539ccf6a2d9780e46f72d936465ea5e",
        "148892bf0bc0036c8f65060009bac8924300fccb30b83a596f73efd2cdd12ffa",
    ),
    ("linux", "x86_64"): (
        "x86_64-linux-gnu",
        "64e5a5655b5612cf220300733347aa4b57b3f4df8532a73b48c09b31c0d77f15",
        "f04a628ddbde72e7763f5d684203186a055b538953e3ff5d8e923f314d5d8b3c",
    ),
}


def _host_key(repository_ctx):
    os_name = repository_ctx.os.name.lower()
    arch = repository_ctx.os.arch.lower()
    if arch == "arm64":
        arch = "aarch64"
    key = (os_name, arch)
    if key not in _PLATFORM_ASSETS:
        fail("CUTEst tools are not configured for host %s/%s" % key)
    return key


def _runtime_library_name(os_name):
    if os_name == "mac os x":
        return "libcutest_double.dylib"
    if os_name == "linux":
        return "libcutest_double.so"
    fail("CUTEst runtime is not configured for host OS %s" % os_name)


def _gfortran_runtime_name(os_name):
    if os_name == "mac os x":
        return "libgfortran.5.dylib"
    if os_name == "linux":
        return "libgfortran.so.5"
    fail("gfortran runtime is not configured for host OS %s" % os_name)


def _cutest_tools_repository_impl(repository_ctx):
    key = _host_key(repository_ctx)
    target, cutest_sha256, sifdecode_sha256 = _PLATFORM_ASSETS[key]

    repository_ctx.download_and_extract(
        url = "https://github.com/ralna/CUTEst/releases/download/v%s/CUTEst_binaries.v%s.%s-libgfortran5.tar.gz" % (
            _CUTEST_VERSION,
            _CUTEST_VERSION,
            target,
        ),
        output = "cutest",
        sha256 = cutest_sha256,
    )
    repository_ctx.download_and_extract(
        url = "https://github.com/ralna/SIFDecode/releases/download/v%s/SIFDecode_binaries.v%s.%s-libgfortran5.tar.gz" % (
            _SIFDECODE_VERSION,
            _SIFDECODE_VERSION,
            target,
        ),
        output = "sifdecode",
        sha256 = sifdecode_sha256,
    )

    gfortran = repository_ctx.which("gfortran")
    if gfortran == None:
        fail("CUTEst problem generation requires gfortran on PATH")

    gfortran_runtime = _gfortran_runtime_name(key[0])
    result = repository_ctx.execute([
        str(gfortran),
        "-print-file-name=%s" % gfortran_runtime,
    ])
    if result.return_code != 0:
        fail("gfortran failed to locate %s: %s" % (gfortran_runtime, result.stderr))
    runtime_path = result.stdout.strip()
    if not runtime_path or runtime_path == gfortran_runtime:
        fail("gfortran could not locate %s" % gfortran_runtime)
    runtime_dir = str(repository_ctx.path(runtime_path).dirname)

    if key[0] == "mac os x":
        loader_variable = "DYLD_LIBRARY_PATH"
    else:
        loader_variable = "LD_LIBRARY_PATH"

    repository_ctx.file(
        "sifdecoder.sh",
        content = """#!/usr/bin/env bash
set -euo pipefail
decoder="$1"
shift
export {loader_variable}="{runtime_dir}${{{loader_variable}:+:${loader_variable}}}"
exec "$decoder" "$@"
""".format(
            loader_variable = loader_variable,
            runtime_dir = runtime_dir,
        ),
        executable = True,
    )
    repository_ctx.file(
        "gfortran.sh",
        content = """#!/usr/bin/env bash
set -euo pipefail
exec "{gfortran}" "$@"
""".format(gfortran = str(gfortran)),
        executable = True,
    )
    repository_ctx.file(
        "run_test.sh",
        content = """#!/usr/bin/env bash
set -euo pipefail
resolve_runfile() {{
  if [[ "$1" = /* ]]; then
    printf '%s\\n' "$1"
  else
    printf '%s/%s/%s\\n' "$TEST_SRCDIR" "$TEST_WORKSPACE" "$1"
  fi
}}
runner="$(resolve_runfile "$1")"
runtime="$(resolve_runfile "$2")"
problem_library="$(resolve_runfile "$3")"
outsdif="$(resolve_runfile "$4")"
scratch="$(mktemp -d "${{TMPDIR:-/tmp}}/sip-cutest-run.XXXXXX")"
trap 'rm -rf "$scratch"' EXIT
cp "$runtime" "$scratch/runtime"
cp "$problem_library" "$scratch/problem"
cp "$outsdif" "$scratch/OUTSDIF.d"
export {loader_variable}="{runtime_dir}${{{loader_variable}:+:${loader_variable}}}"
"$runner" "$scratch/runtime" "$scratch/problem" "$scratch/OUTSDIF.d"
""".format(
            loader_variable = loader_variable,
            runtime_dir = runtime_dir,
        ),
        executable = True,
    )

    runtime_library = _runtime_library_name(key[0])
    repository_ctx.file(
        "BUILD.bazel",
        content = """package(default_visibility = ["//visibility:public"])

exports_files([
    "gfortran.sh",
    "run_test.sh",
    "sifdecoder.sh",
    "sifdecode/bin/sifdecoder",
])

cc_library(
    name = "headers",
    hdrs = glob(["cutest/include/*.h"]),
    includes = ["cutest/include"],
)

filegroup(
    name = "runtime",
    srcs = ["cutest/lib/{runtime_library}"],
)
""".format(runtime_library = runtime_library),
    )


cutest_tools_repository = repository_rule(
    implementation = _cutest_tools_repository_impl,
    local = True,
)


def _sif_problem_repository_impl(repository_ctx):
    repository_ctx.download_and_extract(
        url = repository_ctx.attr.url,
        sha256 = repository_ctx.attr.sha256,
        stripPrefix = repository_ctx.attr.strip_prefix,
    )
    problems = []
    for path in repository_ctx.path(".").readdir():
        if path.basename.endswith(".SIF"):
            problems.append(path.basename[:-4])
    problems = sorted(problems)
    targets = []
    for problem in problems:
        tags = "[]" if problem in repository_ctx.attr.enabled_problems else '["disabled", "manual"]'
        targets.append("""cutest_problem(
    name = "{problem}",
    sif = ":{problem}.SIF",
    tags = {tags},
)
""".format(problem = problem, tags = tags))
    repository_ctx.file(
        "BUILD.bazel",
        content = """load("@//sip-qdldl/cutest_problems:problem.bzl", "cutest_problem")

package(default_visibility = ["//visibility:public"])

exports_files(glob(["*.SIF"]))

{targets}
""".format(targets = "\n".join(targets)),
    )


sif_problem_repository = repository_rule(
    implementation = _sif_problem_repository_impl,
    attrs = {
        "enabled_problems": attr.string_list(),
        "sha256": attr.string(mandatory = True),
        "strip_prefix": attr.string(mandatory = True),
        "url": attr.string(mandatory = True),
    },
)


def _cutest_repositories_impl(module_ctx):
    cutest_tools_repository(name = "cutest_tools")
    sif_problem_repository(
        name = "cutest_sif",
        enabled_problems = CUTEST_ENABLED_PROBLEMS,
        url = "https://github.com/ralna/SIF/archive/%s.tar.gz" % _SIF_COMMIT,
        sha256 = "a8876b264b4cf5c95fb08f292e7a9aeab4f85d54cd0799be10869e2bf11875af",
        strip_prefix = "SIF-%s" % _SIF_COMMIT,
    )


cutest_repositories = module_extension(
    implementation = _cutest_repositories_impl,
)
