load("@rules_python//python:defs.bzl", "py_binary")

py_binary(
  name = "codegen",
  srcs = ["codegen.py"],
  deps = [
    "@slacg//slacg:kkt_codegen",
    "@slacg//slacg:mat_vec_mult_codegen",
  ]
)

genrule(
  name="codegen_genrule",
  outs = [
    "kkt_codegen.hpp",
    "kkt_codegen.cpp",
    "H_ops.hpp",
    "H_ops.cpp",
    "C_ops.hpp",
    "C_ops.cpp",
    "G_ops.hpp",
    "G_ops.cpp",
  ],
  cmd = "$(location :codegen) $(RULEDIR) > /dev/null",
  tools = [":codegen"],
)

cc_library(
  name = "codegen_output",
  srcs = [
    "kkt_codegen.cpp",
    "H_ops.cpp",
    "C_ops.cpp",
    "G_ops.cpp",
  ],
  hdrs = [
    "kkt_codegen.hpp",
    "H_ops.hpp",
    "C_ops.hpp",
    "G_ops.hpp",
  ],
)

cc_test(
  name = "simple_nlp",
  size = "small",
  srcs = ["test.cpp"],
  deps = [
    "@googletest//:gtest",
    "@googletest//:gtest_main",
    "@sip//sip",
    ":codegen_output",
  ],
)
