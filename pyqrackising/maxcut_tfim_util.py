import math
import networkx as nx
import numpy as np
import os
from numba import njit, prange
from scipy.sparse import lil_matrix


class OpenCLContext:
    def __init__(self, p, a, w, d, e, r, c, q, b, s, x, y):
        self.MAX_GPU_PROC_ELEM = p
        self.IS_OPENCL_AVAILABLE = a
        self.work_group_size = w
        self.dtype = d
        self.epsilon = e
        self.max_alloc = r
        self.ctx = c
        self.queue = q
        self.bootstrap_kernel = b
        self.bootstrap_sparse_kernel = s
        self.bootstrap_segmented_kernel = x
        self.bootstrap_sparse_segmented_kernel = y
        self.G_m_buf = None
        self.G_data_buf = None
        self.G_rows_buf = None
        self.G_cols_buf = None

IS_OPENCL_AVAILABLE = True
ctx = None
queue = None
compute_units = None
dtype = np.float32
epsilon = 2 ** -23
work_group_size = 32
max_alloc = 0xFFFFFFFFFFFFFFFF
bootstrap_kernel = None
bootstrap_sparse_kernel = None
bootstrap_segmented_kernel = None
bootstrap_sparse_segmented_kernel = None

dtype_bits = int(os.getenv('PYQRACKISING_FPPOW', '5'))
kernel_src = ''
if dtype_bits <= 4:
    dtype = np.float16
    epsilon = 2 ** -10
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
    epsilon = 2 ** -52
    kernel_src += "#pragma OPENCL EXTENSION cl_khr_fp64 : enable\n"
    kernel_src += "#define real1 double\n"
    kernel_src += "#define qint long\n"
    kernel_src += "#define EPSILON DBL_EPSILON\n"
    kernel_src += "#define ZERO_R1 0.0\n"
    kernel_src += "#define ONE_R1 1.0\n"
    kernel_src += "#define TWO_R1 2.0\n"
else:
    dtype = np.float32
    epsilon = 2 ** -23
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
    bootstrap_kernel = program.bootstrap
    bootstrap_sparse_kernel = program.bootstrap_sparse
    bootstrap_segmented_kernel = program.bootstrap_segmented
    bootstrap_sparse_segmented_kernel = program.bootstrap_sparse_segmented

    work_group_size = bootstrap_kernel.get_work_group_info(
        cl.kernel_work_group_info.PREFERRED_WORK_GROUP_SIZE_MULTIPLE,
        ctx.devices[0]
    )

    max_alloc = ctx.devices[0].get_info(cl.device_info.MAX_MEM_ALLOC_SIZE)
except ImportError:
    IS_OPENCL_AVAILABLE = False
    print("PyOpenCL not installed. (If you have any OpenCL accelerator devices with available ICDs, you might want to optionally install pyopencl.)")

opencl_context = OpenCLContext(compute_units, IS_OPENCL_AVAILABLE, work_group_size, dtype, epsilon, max_alloc, ctx, queue, bootstrap_kernel, bootstrap_sparse_kernel, bootstrap_segmented_kernel, bootstrap_sparse_segmented_kernel)


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


@njit
def binary_search(l, t):
    left = 0
    right = len(l) - 1

    while left <= right:
        mid = (left + right) >> 1

        if l[mid] == t:
            return mid

        if l[mid] < t:
            left = mid + 1
        else:
            right = mid - 1

    return len(l)


def to_scipy_sparse_upper_triangular(G, nodes, n_nodes):
    lil = lil_matrix((n_nodes, n_nodes), dtype=dtype)
    for u in range(n_nodes):
        u_node = nodes[u]
        for v in range(u + 1, n_nodes):
            v_node = nodes[v]
            if G.has_edge(u_node, v_node):
                lil[u, v] = G[u_node][v_node].get('weight', 1.0)

    return lil.tocsr()


@njit(parallel=True)
def init_theta(h_mult, n_qubits, J_eff, degrees):
    theta = np.empty(n_qubits, dtype=dtype)
    h_mult = abs(h_mult)
    for q in prange(n_qubits):
        J = J_eff[q]
        z = degrees[q]
        abs_zJ = abs(z * J)
        theta[q] = (np.pi if J > 0 else -np.pi) if abs_zJ < epsilon else np.arcsin(max(-1.0, min(1.0, h_mult / (z * J))))

    return theta


@njit
def init_thresholds(n_qubits):
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


@njit
def probability_by_hamming_weight(J, h, z, theta, t, n_qubits):
    if abs(J) < epsilon:
        return np.full((n_qubits - 1,), 1.0 / (n_qubits - 1), dtype=dtype)

    ratio = max(1.0, min(-1.0, abs(h) / (z * J)))
    theta_c = np.arcsin(ratio)

    p = (
        pow(2.0, abs(J / h) - 1.0)
        * (1.0 + np.sin(theta - theta_c) * np.cos(1.5 * np.pi * J * t + theta) / (1.0 + np.sqrt(t)))
        - 0.5
    )

    numerator = pow(2.0, (n_qubits + 2) * p) - 1.0
    denominator = pow(2.0, p) - 1.0

    bias = np.empty(n_qubits - 1, dtype=dtype)
    for q in range(n_qubits - 1):
        result = numerator * pow(2.0, -((n_qubits + 1) * p) - p * q) / denominator
        bias[q] = 0.0 if np.isnan(result) or np.isinf(result) else result

    if J > 0.0:
        return bias[::-1]

    return bias


@njit(parallel=True)
def maxcut_hamming_cdf(n_qubits, J_func, degrees, quality, tot_t, h_mult):
    if n_qubits < 2:
        return np.full((n_qubits,), 1.0 / n_qubits, dtype=dtype)

    hamming_prob = init_thresholds(n_qubits)

    n_steps = 1 << quality
    delta_t = 1.0 / n_steps
    n_bias = n_qubits - 1

    theta = init_theta(h_mult, n_qubits, J_func, degrees)

    for qc in prange(n_qubits, n_steps * n_qubits):
        step = qc // n_qubits
        q = qc % n_qubits
        J_eff = J_func[q]
        z = degrees[q]
        theta_eff = theta[q]
        t = step * delta_t
        tm1 = (step - 1) * delta_t
        h_t = h_mult * (tot_t - t)
        bias = probability_by_hamming_weight(J_eff, h_t, z, theta_eff, t, n_qubits)
        last_bias = probability_by_hamming_weight(J_eff, h_t, z, theta_eff, tm1, n_qubits)
        for i in range(n_bias):
            hamming_prob[i] += bias[i] - last_bias[i]

    fix_cdf(hamming_prob)

    return hamming_prob

@njit
def fix_cdf(hamming_prob):
    hamming_prob /= hamming_prob.sum()
    tot_prob = 0.0
    for i in range(len(hamming_prob)):
        tot_prob += hamming_prob[i]
        hamming_prob[i] = tot_prob
    hamming_prob[-1] = 2.0

