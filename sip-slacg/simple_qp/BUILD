load("@rules_python//python:defs.bzl", "py_binary")

py_binary(
  name = "codegen",
  srcs = ["codegen.py"],
  deps = [
    "@slacg//slacg:kkt_codegen",
  ]
)

genrule(
  name="codegen_genrule",
  outs = [
    "kkt_codegen.hpp",
    "kkt_codegen.cpp",
  ],
  cmd = "$(location :codegen) $(RULEDIR) > /dev/null",
  tools = [":codegen"],
)

cc_library(
  name = "codegen_output",
  srcs = [
    "kkt_codegen.cpp",
  ],
  hdrs = [
    "kkt_codegen.hpp",
  ],
)

cc_test(
  name = "simple_qp",
  size = "small",
  srcs = ["test.cpp"],
  deps = [
    "@googletest//:gtest",
    "@googletest//:gtest_main",
    "@sip//sip",
    ":codegen_output",
  ],
)
