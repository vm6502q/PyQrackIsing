import math
import networkx as nx
import numpy as np
import os
from numba import njit, prange
from scipy.sparse import lil_matrix


class OpenCLContext:
    def __init__(self, p, a, w, d, e, r, c, q, b, s, x, y, i, j, k, l):
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
        self.calculate_cut_kernel = i
        self.calculate_cut_sparse_kernel = j
        self.calculate_cut_segmented_kernel = k
        self.calculate_cut_sparse_segmented_kernel = l

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
calculate_cut_kernel = None
calculate_cut_sparse_kernel = None
calculate_cut_segmented_kernel = None
calculate_cut_sparse_segmented_kernel = None

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
    kernel_src += open(os.path.dirname(os.path.abspath(__file__)) + "/kernels.cl").read()
    program = cl.Program(ctx, kernel_src).build()
    bootstrap_kernel = program.bootstrap
    bootstrap_sparse_kernel = program.bootstrap_sparse
    bootstrap_segmented_kernel = program.bootstrap_segmented
    bootstrap_sparse_segmented_kernel = program.bootstrap_sparse_segmented
    calculate_cut_kernel = program.calculate_cut
    calculate_cut_sparse_kernel = program.calculate_cut_sparse
    calculate_cut_segmented_kernel = program.calculate_cut_segmented
    calculate_cut_sparse_segmented_kernel = program.calculate_cut_sparse_segmented

    work_group_size = bootstrap_kernel.get_work_group_info(
        cl.kernel_work_group_info.PREFERRED_WORK_GROUP_SIZE_MULTIPLE,
        ctx.devices[0]
    )

    max_alloc = ctx.devices[0].get_info(cl.device_info.MAX_MEM_ALLOC_SIZE)
except ImportError:
    IS_OPENCL_AVAILABLE = False
    print("PyOpenCL not installed. (If you have any OpenCL accelerator devices with available ICDs, you might want to optionally install pyopencl.)")

opencl_context = OpenCLContext(compute_units, IS_OPENCL_AVAILABLE, work_group_size, dtype, epsilon, max_alloc, ctx, queue, bootstrap_kernel, bootstrap_sparse_kernel, bootstrap_segmented_kernel, bootstrap_sparse_segmented_kernel, calculate_cut_kernel, calculate_cut_sparse_kernel, calculate_cut_segmented_kernel, calculate_cut_sparse_segmented_kernel)


def setup_opencl(l, g, args_np):
    ctx = opencl_context.ctx
    queue = opencl_context.queue
    dtype = opencl_context.dtype
    wgs = opencl_context.work_group_size

    # Buffers
    mf = cl.mem_flags
    args_buf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=args_np)

    # Group sizes
    local_size = min(wgs, l)
    max_global_size = ((opencl_context.MAX_GPU_PROC_ELEM + local_size - 1) // local_size) * local_size  # corresponds to MAX_PROC_ELEM macro in OpenCL kernel program
    global_size = min(((g + local_size - 1) // local_size) * local_size, max_global_size)

    # Local memory allocation (1 float per work item)
    local_energy_buf = cl.LocalMemory(np.dtype(dtype).itemsize * local_size)
    local_index_buf = cl.LocalMemory(np.dtype(np.int32).itemsize * local_size)

    # Allocate max_energy and max_index result buffers per workgroup
    max_energy_host = np.empty(global_size // local_size, dtype=dtype)
    max_index_host = np.empty(global_size // local_size, dtype=np.int32)

    max_energy_buf = cl.Buffer(ctx, mf.WRITE_ONLY, max_energy_host.nbytes)
    max_index_buf = cl.Buffer(ctx, mf.WRITE_ONLY, max_index_host.nbytes)

    return local_size, global_size, args_buf, local_energy_buf, local_index_buf, max_energy_host, max_index_host, max_energy_buf, max_index_buf


def make_G_m_buf(G_m, is_segmented, segment_size):
    mf = cl.mem_flags
    ctx = opencl_context.ctx
    if is_segmented:
        o_shape = G_m.shape[0] * G_m.shape[1]
        n_shape = segment_size << 2
        _G_m = np.reshape(G_m, (o_shape,))
        if n_shape != o_shape:
            np.resize(_G_m, (n_shape,))
        G_m_segments = np.split(_G_m, 4)
        G_m_buf = [
            cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=seg)
            for seg in G_m_segments
        ]
    else:
        G_m_buf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=G_m)

    return G_m_buf


def make_G_m_csr_buf(G_m, is_segmented, segment_size):
    mf = cl.mem_flags
    ctx = opencl_context.ctx
    G_rows_buf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=G_m.indptr)
    G_cols_buf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=G_m.indices)
    if is_segmented:
        o_shape = G_m.data.shape[0]
        n_shape = segment_size << 2
        _G_data = np.reshape(G_m.data, (o_shape,))
        if n_shape != o_shape:
            np.resize(_G_data, (n_shape,))
        G_data_segments = np.split(_G_data, 4)
        G_data_buf = [
            cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=seg)
            for seg in G_data_segments
        ]
        _G_data = None
        _G_data_segments = None
    else:
        G_data_buf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=G_m.data)

    return G_data_buf, G_rows_buf, G_cols_buf


def make_theta_buf(theta, is_segmented, shots, n):
    mf = cl.mem_flags
    ctx = opencl_context.ctx
    if is_segmented:
        n_shape = (((shots + 3) >> 2) << 2) * ((n + 31) >> 5)
        theta = np.reshape(theta, (n_shape,))
        theta_segments = np.split(theta, 4)
        theta_buf = [
            cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=seg)
            for seg in theta_segments
        ]
    else:
        theta_buf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=theta)

    return theta_buf


@njit(parallel=True)
def convert_bool_to_uint(samples):
    shots = samples.shape[0]
    n = samples.shape[1]
    n32 = (n + 31) >> 5
    theta = np.zeros(shots * n32, dtype=np.uint32)
    for i in prange(shots):
        i_offset = i * n32
        for j in range(n):
            if samples[i, j]:
                theta[i_offset + (j >> 5)] |= 1 << (j & 31)

    return theta


@njit
def get_cut(solution, nodes, n):
    bit_string = ""
    l, r = [], []
    for i in range(n):
        if solution[i]:
            bit_string += "1"
            r.append(nodes[i])
        else:
            bit_string += "0"
            l.append(nodes[i])

    return bit_string, l, r


@njit
def get_cut_base(solution, n):
    l, r = [], []
    for i in range(n):
        if solution[i]:
            r.append(i)
        else:
            l.append(i)

    return l, r


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
    theta = np.empty(n_qubits, dtype=np.float64)
    h_mult = abs(h_mult)
    for q in prange(n_qubits):
        J = J_eff[q]
        z = degrees[q]
        zJ = z * J
        theta[q] = ((np.pi if J > 0 else -np.pi) / 2) if abs(zJ) < epsilon else np.arcsin(max(-1.0, min(1.0, h_mult / zJ)))

    return theta


@njit
def init_thresholds(n_qubits):
    n_bias = n_qubits - 1
    thresholds = np.empty(n_bias, dtype=np.float64)
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
def probability_by_hamming_weight(J, h, z, theta, t, n_bias):
    zJ = z * J
    theta_c = ((np.pi if J > 0 else -np.pi) / 2) if abs(zJ) < epsilon else np.arcsin(max(-1.0, min(1.0, h / zJ)))

    p = (
        2.0 ** (abs(J / h) - 1.0)
        * (1.0 + np.sin(theta - theta_c) * np.cos(1.5 * np.pi * J * t + theta) / (1.0 + np.sqrt(t)))
        - 0.5
    )


    bias = np.empty(n_bias, dtype=np.float64)
    factor = 2.0 ** -(p / n_bias)
    result = 1.0
    for q in range(n_bias):
        result *= factor
        bias[q] = result

    if (result == 0.0) or np.isnan(result) or np.isinf(result):
        print("[WARN]: probability_by_hamming_weight() went below maximum precision.")

    if J > 0.0:
        return bias[::-1]

    return bias


@njit(parallel=True)
def maxcut_hamming_cdf(n_qubits, J_func, degrees, quality, tot_t, h_mult):
    hamming_prob = init_thresholds(n_qubits)

    n_steps = 1 << quality
    delta_t = tot_t / n_steps
    n_bias = n_qubits + 1

    theta = init_theta(h_mult, n_qubits, J_func, degrees)

    for qc in prange(n_qubits, n_steps * n_qubits):
        step = qc // n_qubits
        q = qc % n_qubits
        J_eff = J_func[q]
        if abs(J_eff) < epsilon:
            continue
        z = degrees[q]
        theta_eff = theta[q]
        t = step * delta_t
        tm1 = (step - 1) * delta_t
        h_t = h_mult * (tot_t - t)
        bias = probability_by_hamming_weight(J_eff, h_t, z, theta_eff, t, n_bias)
        last_bias = probability_by_hamming_weight(J_eff, h_t, z, theta_eff, tm1, n_bias)
        for i in range(n_bias):
            hamming_prob[i] += bias[i] - last_bias[i]

    hamming_prob /= hamming_prob.sum() - (hamming_prob[0] + hamming_prob[-1])
    tot_prob = 0.0
    n_bias -= 2
    cum_prob = np.empty(n_bias, dtype=np.float64)
    for i in range(n_bias):
        tot_prob += hamming_prob[i + 1]
        cum_prob[i] = tot_prob
    cum_prob[-1] = 1.0

    return cum_prob


@njit
def sample_mag(cum_prob):
    p = np.random.random()
    m = 0
    left = 0
    right = len(cum_prob) - 1
    while True:
        m = (left + right) >> 1

        if (cum_prob[m] >= p) and ((m == 0) or cum_prob[m - 1] < p):
            break

        if cum_prob[m] < p:
            left = m + 1
        else:
            right = m - 1

    return m + 1

@njit
def init_bit_pick(weights, p, n):
    p *= np.random.rand()
    cum = 0.0
    node = 0
    for i in range(n):
        cum += weights[i]
        if p < cum:
            node = i
            break

    return node

@njit
def bit_pick(weights, used, n):
    # Count available
    p = 0.0
    for i in range(n):
        if used[i]:
            continue
        p += weights[i]

    # Normalize & sample
    p *= np.random.rand()
    cum = 0.0
    node = 0
    for i in range(n):
        if used[i]:
            continue
        cum += weights[i]
        if p < cum:
            node = i
            break

    return node
