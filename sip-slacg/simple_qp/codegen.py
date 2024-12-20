import numpy as np

import sys

import slacg

from slacg.kkt_codegen import kkt_codegen
from slacg.mat_vec_mult_codegen import mat_vec_mult_codegen

x_dim = 2
s_dim = 4
y_dim = 1

dim = x_dim + y_dim + s_dim

H = np.ones([x_dim, x_dim])
C = np.array([[1., 1.]])
G = np.array([[ 1.,  0.],
              [-1.,  0.],
              [ 0.,  1.],
              [ 0., -1.]])

P = np.arange(dim - 1, -1, -1)

output_prefix = sys.argv[-1]

cpp_header_code, cpp_impl_code = kkt_codegen(
    H=H, C=C, G=G, P=P, namespace="sip_examples", header_name="kkt_codegen"
)

with open(f"{output_prefix}/kkt_codegen.hpp", "w") as f:
    f.write(cpp_header_code)

with open(f"{output_prefix}/kkt_codegen.cpp", "w") as f:
    f.write(cpp_impl_code)

cpp_header_code, cpp_impl_code = mat_vec_mult_codegen(
    M=H, namespace="sip_examples::H_ops", header_name="H_ops"
)

with open(f"{output_prefix}/H_ops.hpp", "w") as f:
    f.write(cpp_header_code)

with open(f"{output_prefix}/H_ops.cpp", "w") as f:
    f.write(cpp_impl_code)

cpp_header_code, cpp_impl_code = mat_vec_mult_codegen(
    M=C, namespace="sip_examples::C_ops", header_name="C_ops"
)

with open(f"{output_prefix}/C_ops.hpp", "w") as f:
    f.write(cpp_header_code)

with open(f"{output_prefix}/C_ops.cpp", "w") as f:
    f.write(cpp_impl_code)

cpp_header_code, cpp_impl_code = mat_vec_mult_codegen(
    M=G, namespace="sip_examples::G_ops", header_name="G_ops"
)

with open(f"{output_prefix}/G_ops.hpp", "w") as f:
    f.write(cpp_header_code)

with open(f"{output_prefix}/G_ops.cpp", "w") as f:
    f.write(cpp_impl_code)
