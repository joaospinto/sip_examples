# CUTEst problem corpus

This integration imports the authoritative CUTEst SIF corpus without
redefining its problems. Bazel downloads and caches these pinned upstream
artifacts on demand:

- CUTEst 2.7.1
- SIFDecode 3.1.1
- `ralna/SIF` commit `ba729aec456b4a38f02917a305d8719b26b4c9fc`

The SIF repository contains 1,539 authoritative root problem definitions.
Categorized subdirectories duplicate those files and are intentionally not
treated as additional problems. Parameterized SIF files use their upstream
default parameter values.

Each generated target decodes its original SIF file, compiles the resulting
Fortran routines into a problem-specific shared library, and loads that
library through the pinned CUTEst runtime. The common C++ adapter passes the
following mathematically equivalent NLP to SIP-QDLDL:

- fixed variables become equality constraints;
- finite variable bounds become one-sided inequalities;
- CUTEst equality constraints remain equalities;
- lower, upper, and ranged constraints become one or two inequalities as
  appropriate;
- transformed multipliers are mapped back to the original CUTEst constraint
  multipliers before evaluating the exact Lagrangian Hessian.

Objective gradients, constraint Jacobians, and Lagrangian Hessians use
CUTEst's sparse interfaces. Each problem's KKT sparsity is ordered with AMD,
and the QDLDL workspace size is obtained by symbolic factorization of that
permuted pattern.

The complete corpus was classified with a 120-second per-problem timeout. Of
the 1,539 problems, 755 solved locally. Twenty-five of those were kept disabled
because their runtime or iteration count was close enough to a classification
limit to make them marginal. The remaining 730 are enabled; all failures,
timeouts, and marginal cases retain the `manual` and `disabled` tags.

Every problem remains individually runnable:

```sh
bazel test @cutest_sif//:ROSENBR --test_output=all
```

The enabled set can be run together with:

```sh
bazel test @cutest_sif//...
```

The current generation pipeline supports macOS and Linux on arm64 and x86-64
and requires `gfortran` with the libgfortran 5 ABI on `PATH`. The upstream
CUTEst and SIFDecode licenses are included in their downloaded release
archives.
