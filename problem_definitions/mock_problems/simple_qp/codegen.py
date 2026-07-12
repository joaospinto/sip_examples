import sys

import numpy as np
import scipy as sp
from slacg.kkt_codegen import kkt_codegen

x_dim = 2
s_dim = 4
y_dim = 1
dim = x_dim + y_dim + s_dim

H = sp.sparse.csc_matrix(np.ones([x_dim, x_dim]))
C = sp.sparse.csc_matrix([[1.0, 1.0]])
G = sp.sparse.csc_matrix(
    [
        [1.0, 0.0],
        [-1.0, 0.0],
        [0.0, 1.0],
        [0.0, -1.0],
    ]
)
P = np.arange(dim - 1, -1, -1)

output_prefix = sys.argv[-1]
cpp_header_code, cpp_impl_code = kkt_codegen(
    H=H, C=C, G=G, P=P, namespace="sip_examples", header_name="kkt_codegen"
)

with open(f"{output_prefix}/kkt_codegen.hpp", "w") as f:
    f.write(cpp_header_code)

with open(f"{output_prefix}/kkt_codegen.cpp", "w") as f:
    f.write(cpp_impl_code)
