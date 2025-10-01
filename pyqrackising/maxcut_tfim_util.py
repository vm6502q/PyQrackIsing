import math
import networkx as nx
import numpy as np
import os
from numba import njit, prange


@njit
def get_cut(solution, nodes):
    bit_string = ""
    l, r = [], []
    for i in range(len(solution)):
        if solution[i]:
            bit_string += "1"
            r.append(nodes[i])
        else:
            bit_string += "0"
            l.append(nodes[i])

    return bit_string, l, r


@njit(parallel=True)
def init_theta(h_mult, n_qubits, J_eff, degrees, dtype):
    theta = np.empty(n_qubits, dtype=dtype)
    h_mult = abs(h_mult)
    for q in prange(n_qubits):
        J = J_eff[q]
        z = degrees[q]
        theta[q] = np.arcsin(
            max(
                -1.0,
                min(
                    1.0,
                    np.sign(J) if np.isclose(abs(z * J), 0.0) else (h_mult / (z * J)),
                ),
            )
        )

    return theta


@njit
def init_thresholds(n_qubits, dtype):
    n_bias = n_qubits - 1
    thresholds = np.empty(n_bias, dtype=dtype)
    tot_prob = 0
    p = 1.0
    if n_qubits & 1:
        q = n_qubits // 2
        thresholds[q - 1] = p
        tot_prob = p
        p /= 2
    for q in range(1, n_qubits // 2):
        thresholds[q - 1] = p
        thresholds[n_bias - q] = p
        tot_prob += 2 * p
        p /= 2
    thresholds /= tot_prob

    return thresholds


@njit(parallel=True)
def maxcut_hamming_cdf(n_qubits, J_func, degrees, quality, hamming_prob, dtype):
    if n_qubits < 2:
        hamming_prob.fill(0.0)
        return

    n_steps = 1 << quality
    delta_t = 1.0 / n_steps
    tot_t = 2.0 * n_steps * delta_t
    h_mult = 2.0 / tot_t
    n_bias = n_qubits - 1

    theta = init_theta(h_mult, n_qubits, J_func, degrees, dtype)

    for qc in prange(n_qubits, n_steps * n_qubits):
        step = qc // n_qubits
        q = qc % n_qubits
        J_eff = J_func[q]
        if np.isclose(abs(J_eff), 0.0):
            continue
        z = degrees[q]
        theta_eff = theta[q]
        t = step * delta_t
        tm1 = (step - 1) * delta_t
        h_t = h_mult * (tot_t - t)
        bias = probability_by_hamming_weight(J_eff, h_t, z, theta_eff, t, n_qubits, dtype)
        last_bias = probability_by_hamming_weight(J_eff, h_t, z, theta_eff, tm1, n_qubits, dtype)
        for i in range(n_bias):
            hamming_prob[i] += bias[i] - last_bias[i]

    tot_prob = hamming_prob.sum()
    hamming_prob /= tot_prob

    tot_prob = 0.0
    for i in range(n_bias):
        tot_prob += hamming_prob[i]
        hamming_prob[i] = tot_prob
    hamming_prob[-1] = 2.0


@njit
def fix_cdf(hamming_prob):
    hamming_prob /= hamming_prob.sum()
    tot_prob = 0.0
    for i in range(len(hamming_prob)):
        tot_prob += hamming_prob[i]
        hamming_prob[i] = tot_prob
    hamming_prob[-1] = 2.0


@njit
def probability_by_hamming_weight(J, h, z, theta, t, n_qubits, dtype):
    # critical angle
    theta_c = np.arcsin(max(-1.0, min(1.0, abs(h) / (z * J))))

    p = (
        pow(2.0, abs(J / h) - 1.0)
        * (1.0 + np.sin(theta - theta_c) * np.cos(1.5 * np.pi * J * t + theta) / (1.0 + np.sqrt(t)))
        - 0.5
    )

    if (p * n_qubits) >= 1024:
        return np.zeros(n_qubits - 1, dtype=dtype)

    bias = np.empty(n_qubits - 1, dtype=dtype)
    tot_n = 1.0 + 1.0 / pow(2.0, p * n_qubits)
    factor = pow(2.0, -p)
    n = 1.0
    for q in range(1, n_qubits):
        n *= factor
        bias[q - 1] = n
        tot_n += n

    if np.isnan(tot_n) or np.isinf(tot_n):
        return np.zeros(n_qubits - 1, dtype=dtype)

    bias /= tot_n

    if J > 0.0:
        return bias[::-1]

    return bias


class OpenCLContext:
    def __init__(self, p, a, w, d, r, c, q, i, m, b, s, x, y, k, l):
        self.MAX_GPU_PROC_ELEM = p
        self.IS_OPENCL_AVAILABLE = a
        self.work_group_size = w
        self.dtype = d
        self.max_alloc = r
        self.ctx = c
        self.queue = q
        self.init_theta_kernel = i
        self.maxcut_hamming_cdf_kernel = m
        self.bootstrap_kernel = b
        self.bootstrap_sparse_kernel = s
        self.bootstrap_segmented_kernel = x
        self.bootstrap_sparse_segmented_kernel = y
        self.sample_for_solution_best_bitset_kernel = k
        self.sample_for_solution_best_bitset_sparse_kernel = l
        self.G_m_buf = None
        self.G_data_buf = None
        self.G_rows_buf = None
        self.G_cols_buf = None

IS_OPENCL_AVAILABLE = True
ctx = None
queue = None
compute_units = None
dtype = np.float32
work_group_size = 32
max_alloc = 0xFFFFFFFFFFFFFFFF
init_theta_kernel = None
maxcut_hamming_cdf_kernel = None
bootstrap_kernel = None
bootstrap_sparse_kernel = None
bootstrap_segmented_kernel = None
bootstrap_sparse_segmented_kernel = None
sample_for_solution_best_bitset_kernel = None
sample_for_solution_best_bitset_sparse_kernel = None

dtype_bits = int(os.getenv('PYQRACKISING_FPPOW', '5'))
kernel_src = ''
if dtype_bits <= 4:
    dtype = np.float16
    kernel_src += "#pragma OPENCL EXTENSION cl_khr_fp16 : enable\n"
    kernel_src += "#define FP16 1\n"
    kernel_src += "#define real1 half\n"
    kernel_src += "#define qint short\n"
    kernel_src += "#define EPSILON ((half)0.00097656f)\n"
    kernel_src += "#define ZERO_R1 ((half)0.0f)\n"
    kernel_src += "#define ONE_R1 ((half)1.0f)\n"
    kernel_src += "#define TWO_R1 ((half)2.0f)\n"
elif dtype_bits >= 6:
    dtype = np.float64
    kernel_src += "#pragma OPENCL EXTENSION cl_khr_fp64 : enable\n"
    kernel_src += "#define real1 double\n"
    kernel_src += "#define qint long\n"
    kernel_src += "#define EPSILON DBL_EPSILON\n"
    kernel_src += "#define ZERO_R1 0.0\n"
    kernel_src += "#define ONE_R1 1.0\n"
    kernel_src += "#define TWO_R1 2.0\n"
else:
    dtype = np.float32
    kernel_src += "#define real1 float\n"
    kernel_src += "#define qint int\n"
    kernel_src += "#define EPSILON FLT_EPSILON\n"
    kernel_src += "#define ZERO_R1 0.0f\n"
    kernel_src += "#define ONE_R1 1.0f\n"
    kernel_src += "#define TWO_R1 2.0f\n"

try:
    import pyopencl as cl
    import warnings

    warnings.simplefilter("ignore", cl.CompilerWarning)
    
    # Pick a device (GPU if available)
    ctx = cl.create_some_context()
    queue = cl.CommandQueue(ctx)
    compute_units = int(os.getenv('PYQRACKISING_MAX_GPU_PROC_ELEM', str(ctx.devices[0].get_info(cl.device_info.MAX_COMPUTE_UNITS))))

    # Load and build OpenCL kernels
    kernel_src += f"#define MAX_PROC_ELEM {compute_units}\n"
    kernel_src += f"#define TOP_N {os.getenv('PYQRACKISING_GPU_TOP_N', '32')}\n"
    kernel_src += open(os.path.dirname(os.path.abspath(__file__)) + "/kernels.cl").read()
    program = cl.Program(ctx, kernel_src).build()
    init_theta_kernel = program.init_theta
    maxcut_hamming_cdf_kernel = program.maxcut_hamming_cdf
    bootstrap_kernel = program.bootstrap
    bootstrap_sparse_kernel = program.bootstrap_sparse
    bootstrap_segmented_kernel = program.bootstrap_segmented
    bootstrap_sparse_segmented_kernel = program.bootstrap_sparse_segmented
    sample_for_solution_best_bitset_kernel = program.sample_for_solution_best_bitset
    sample_for_solution_best_bitset_sparse_kernel = program.sample_for_solution_best_bitset_sparse

    work_group_size = bootstrap_kernel.get_work_group_info(
        cl.kernel_work_group_info.PREFERRED_WORK_GROUP_SIZE_MULTIPLE,
        ctx.devices[0]
    )

    max_alloc = ctx.devices[0].get_info(cl.device_info.MAX_MEM_ALLOC_SIZE)
except ImportError:
    IS_OPENCL_AVAILABLE = False
    print("PyOpenCL not installed. (If you have any OpenCL accelerator devices with available ICDs, you might want to optionally install pyopencl.)")

opencl_context = OpenCLContext(compute_units, IS_OPENCL_AVAILABLE, work_group_size, dtype, max_alloc, ctx, queue, init_theta_kernel, maxcut_hamming_cdf_kernel, bootstrap_kernel, bootstrap_sparse_kernel, bootstrap_segmented_kernel, bootstrap_sparse_segmented_kernel, sample_for_solution_best_bitset_kernel, sample_for_solution_best_bitset_sparse_kernel)
