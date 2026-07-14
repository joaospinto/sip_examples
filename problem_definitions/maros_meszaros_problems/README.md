# Maros-Meszaros QP corpus

This integration imports the complete 138-problem Maros-Meszaros convex
quadratic programming test set without redefining its problems. Bazel downloads
and caches the SIF/QPS archive at Optrove commit `9adfb5707b1e`, the same pinned
artifact used by QPSReader.jl.

The original problems have the form

```text
minimize  c0 + c' x + 0.5 x' Q x
subject to A x = b
           l <= x <= u
```

where `Q` is symmetric positive semidefinite. QPS is a subset of SIF, so each
original file is decoded directly by SIFDecode and evaluated through CUTEst.
The shared CUTEst adapter then supplies the exact sparse objective, constraints,
bounds, and Hessian to SIP-QDLDL. Its KKT system is ordered with AMD before
symbolic sizing and numerical factorization to control fill-in.

The complete corpus was classified with a 120-second per-problem timeout. Of
the 138 problems, 126 solve locally and are enabled. The remaining 12 failures
and timeouts retain the `manual` and `disabled` tags, so they remain available
for focused solver development without making the default corpus test fail.

Every problem remains available from the external repository by its canonical
name:

```sh
bazel test @maros_meszaros//:QAFIRO --test_output=all
```

The enabled set can be run together with:

```sh
bazel test @maros_meszaros//...
```

The enabled set also runs in CI on Ubuntu.

The archive README attributes the collection to Istvan Maros and Csaba
Meszaros and references their 1999 Optimization Methods and Software paper,
"A Repository of Convex Quadratic Programming Problems."
