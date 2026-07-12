"""Builds a SIF problem and runs it through the shared CUTEst adapter."""

_COMPILE_PROBLEM = Label("//tools/cutest:compile_problem.sh")
_DECODE_PROBLEM = Label("//tools/cutest:decode_problem.sh")
_RUNNER = Label("//problem_definitions/cutest_problems:runner")


def cutest_problem(name, sif, tags = None):
    if tags == None:
        tags = ["disabled", "manual"]

    output_dir = "_cutest/" + name
    outsdif = output_dir + "/OUTSDIF.d"
    elfun = output_dir + "/ELFUN.f"
    group = output_dir + "/GROUP.f"
    range_file = output_dir + "/RANGE.f"
    extern = output_dir + "/EXTER.f"
    decoded = [outsdif, elfun, group, range_file, extern]

    native.genrule(
        name = name + "_decode",
        srcs = [sif],
        outs = decoded,
        cmd = " ".join([
            "$(location %s)" % _DECODE_PROBLEM,
            "$(location @cutest_tools//:sifdecoder.sh)",
            "$(location @cutest_tools//:sifdecode/bin/sifdecoder)",
            "$(location %s)" % sif,
            name,
            "$(location :%s)" % outsdif,
            "$(location :%s)" % elfun,
            "$(location :%s)" % group,
            "$(location :%s)" % range_file,
            "$(location :%s)" % extern,
        ]),
        tags = tags,
        tools = [
            _DECODE_PROBLEM,
            "@cutest_tools//:sifdecode/bin/sifdecoder",
            "@cutest_tools//:sifdecoder.sh",
        ],
    )

    problem_library = output_dir + "/libproblem.so"
    native.genrule(
        name = name + "_library",
        srcs = decoded,
        outs = [problem_library],
        cmd = " ".join([
            "$(location %s)" % _COMPILE_PROBLEM,
            "$(location @cutest_tools//:gfortran.sh)",
            "$(location :%s)" % problem_library,
            "$(location :%s)" % elfun,
            "$(location :%s)" % group,
            "$(location :%s)" % range_file,
            "$(location :%s)" % extern,
        ]),
        tags = tags,
        tools = [
            _COMPILE_PROBLEM,
            "@cutest_tools//:gfortran.sh",
        ],
    )

    native.sh_test(
        name = name,
        srcs = ["@cutest_tools//:run_test.sh"],
        args = [
            "$(rootpath %s)" % _RUNNER,
            "$(rootpath @cutest_tools//:runtime)",
            "$(rootpath :%s)" % problem_library,
            "$(rootpath :%s)" % outsdif,
        ],
        data = [
            ":" + outsdif,
            ":" + problem_library,
            _RUNNER,
            "@cutest_tools//:runtime",
        ],
        tags = tags,
    )
