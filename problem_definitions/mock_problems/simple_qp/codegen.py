import sys

import numpy as np
import scipy as sp
from slacg.kkt_codegen import kkt_codegen, write_generated_files

x_dim = 2
s_dim = 0
y_dim = 1
dim = x_dim + y_dim + s_dim

H = sp.sparse.csc_matrix(np.ones([x_dim, x_dim]))
C = sp.sparse.csc_matrix([[1.0, 1.0]])
G = sp.sparse.csc_matrix((s_dim, x_dim))
P = np.arange(dim - 1, -1, -1)

output_prefix = sys.argv[-1]
write_generated_files(
    output_prefix,
    kkt_codegen(
        H=H,
        C=C,
        G=G,
        P=P,
        namespace="sip_examples",
        header_name="kkt_codegen",
    ),
)
