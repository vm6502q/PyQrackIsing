import math
import networkx as nx
import numpy as np
import os
from numba import njit, prange


class OpenCLContext:
    def __init__(self, a, c, q, k):
        self.IS_OPENCL_AVAILABLE = a
        self.ctx = c
        self.queue = q
        self.maxcut_hamming_cdf_kernel = k

IS_OPENCL_AVAILABLE = True
ctx = None
queue = None
try:
    import pyopencl as cl
    
    # Pick a device (GPU if available)
    ctx = cl.create_some_context()
    queue = cl.CommandQueue(ctx)

    # Load and build OpenCL kernels
    kernel_src = open(os.path.dirname(os.path.abspath(__file__)) + "/kernels.cl").read()
    program = cl.Program(ctx, kernel_src).build()
    maxcut_hamming_cdf_kernel = program.maxcut_hamming_cdf
except ImportError:
    IS_OPENCL_AVAILABLE = False

opencl_context = OpenCLContext(IS_OPENCL_AVAILABLE, ctx, queue, maxcut_hamming_cdf_kernel)
