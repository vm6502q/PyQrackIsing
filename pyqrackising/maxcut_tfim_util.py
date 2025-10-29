import math
import networkx as nx
import numpy as np
import os
from numba import njit, prange
from scipy.sparse import lil_matrix


class OpenCLContext:
    def __init__(self, a, b, g, d, e, f, c, q, i, j, k, l, m, n, o, p, x, y, z, w):
        self.MAX_GPU_PROC_ELEM = a
        self.IS_OPENCL_AVAILABLE = b
        self.work_group_size = g
        self.dtype = d
        self.epsilon = e
        self.max_alloc = f
        self.ctx = c
        self.queue = q
        self.calculate_cut_kernel = i
        self.calculate_cut_sparse_kernel = j
        self.calculate_cut_segmented_kernel = k
        self.calculate_cut_sparse_segmented_kernel = l
        self.single_bit_flips_kernel = m
        self.single_bit_flips_sparse_kernel = n
        self.single_bit_flips_segmented_kernel = o
        self.single_bit_flips_sparse_segmented_kernel = p
        self.double_bit_flips_kernel = x
        self.double_bit_flips_sparse_kernel = y
        self.double_bit_flips_segmented_kernel = z
        self.double_bit_flips_sparse_segmented_kernel = w

IS_OPENCL_AVAILABLE = True
ctx = None
queue = None
compute_units = None
dtype = np.float32
epsilon = 2 ** -23
work_group_size = 32
max_alloc = 0xFFFFFFFFFFFFFFFF
calculate_cut_kernel = None
calculate_cut_sparse_kernel = None
calculate_cut_segmented_kernel = None
calculate_cut_sparse_segmented_kernel = None
single_bit_flips_kernel = None
single_bit_flips_sparse_kernel = None
single_bit_flips_segmented_kernel = None
single_bit_flips_sparse_segmented_kernel = None
double_bit_flips_kernel = None
double_bit_flips_sparse_kernel = None
double_bit_flips_segmented_kernel = None
double_bit_flips_sparse_segmented_kernel = None

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
    calculate_cut_kernel = program.calculate_cut
    calculate_cut_sparse_kernel = program.calculate_cut_sparse
    calculate_cut_segmented_kernel = program.calculate_cut_segmented
    calculate_cut_sparse_segmented_kernel = program.calculate_cut_sparse_segmented
    single_bit_flips_kernel = program.single_bit_flips
    single_bit_flips_sparse_kernel = program.single_bit_flips_sparse
    single_bit_flips_segmented_kernel = program.single_bit_flips_segmented
    single_bit_flips_sparse_segmented_kernel = program.single_bit_flips_sparse_segmented
    double_bit_flips_kernel = program.double_bit_flips
    double_bit_flips_sparse_kernel = program.double_bit_flips_sparse
    double_bit_flips_segmented_kernel = program.double_bit_flips_segmented
    double_bit_flips_sparse_segmented_kernel = program.double_bit_flips_sparse_segmented

    work_group_size = calculate_cut_kernel.get_work_group_info(
        cl.kernel_work_group_info.PREFERRED_WORK_GROUP_SIZE_MULTIPLE,
        ctx.devices[0]
    )

    max_alloc = ctx.devices[0].get_info(cl.device_info.MAX_MEM_ALLOC_SIZE)
except ImportError:
    IS_OPENCL_AVAILABLE = False
    print("PyOpenCL not installed. (If you have any OpenCL accelerator devices with available ICDs, you might want to optionally install pyopencl.)")

opencl_context = OpenCLContext(compute_units, IS_OPENCL_AVAILABLE, work_group_size, dtype, epsilon, max_alloc, ctx, queue, calculate_cut_kernel, calculate_cut_sparse_kernel, calculate_cut_segmented_kernel, calculate_cut_sparse_segmented_kernel, single_bit_flips_kernel, single_bit_flips_sparse_kernel, single_bit_flips_segmented_kernel, single_bit_flips_sparse_segmented_kernel, double_bit_flips_kernel, double_bit_flips_sparse_kernel, double_bit_flips_segmented_kernel, double_bit_flips_sparse_segmented_kernel)


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


def make_best_theta_buf(theta):
    n = theta.shape[0]
    n32 = (n + 31) >> 5
    theta_np = np.zeros(n32, dtype=np.uint32)
    for i in range(n):
        if theta[i]:
            theta_np[(i >> 5)] |= 1 << (i & 31)

    mf = cl.mem_flags
    theta_buf = cl.Buffer(opencl_context.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=theta_np)

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
def compute_energy(sample, G_m, n_qubits):
    energy = 0
    for u in range(n_qubits):
        for v in range(u + 1, n_qubits):
            val = G_m[u, v]
            energy += val if sample[u] == sample[v] else -val

    return -energy


@njit
def compute_cut(sample, G_m, n_qubits):
    l, r = get_cut_base(sample, n_qubits)
    cut = 0
    for u in l:
        for v in r:
            cut += G_m[u, v]

    return cut


@njit
def compute_energy_sparse(sample, G_data, G_rows, G_cols, n_qubits):
    energy = 0
    for u in range(n_qubits):
        u_bit = sample[u]
        for col in range(G_rows[u], G_rows[u + 1]):
            val = G_data[col]
            energy += val if u_bit == sample[G_cols[col]] else -val

    return -energy


@njit
def compute_cut_sparse(sample, G_data, G_rows, G_cols, n_qubits):
    l, r = get_cut_base(sample, n_qubits)
    s = l if len(l) < len(r) else r
    cut = 0
    for u in s:
        u_bit = sample[u]
        for col in range(G_rows[u], G_rows[u + 1]):
            if u_bit != sample[G_cols[col]]:
                cut += G_data[col]

    return cut


@njit
def compute_energy_streaming(sample, G_func, nodes, n_qubits):
    energy = 0
    for u in range(n_qubits):
        u_bit = sample[u]
        for v in range(u + 1, n_qubits):
            val = G_func(nodes[u], nodes[v])
            energy += val if u_bit == sample[v] else -val

    return -energy


@njit
def compute_cut_streaming(sample, G_func, nodes, n_qubits):
    l, r = get_cut_base(sample, n_qubits)
    cut = 0
    for u in l:
        for v in r:
            cut += G_func(nodes[u], nodes[v])

    return cut


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


# By Gemini (Google Search AI)
def int_to_bitstring(integer, length):
    return (bin(integer)[2:].zfill(length))[::-1]


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


# From Google Search AI
@njit
def factorial(num):
    """Calculates the factorial of a non-negative integer."""
    if num == 0:
        return 1

    result = 1
    for i in range(1, num + 1):
        result *= i

    return result


# From Google Search AI
@njit
def comb(n, k):
    """
    Calculates the number of combinations (n choose k) from scratch.
    n: The total number of items.
    k: The number of items to choose.
    """
    # Optimize by choosing the smaller of k and (n-k)
    # This reduces the number of multiplications in the factorial calculation
    k = min(k, n - k)

    # Calculate the numerator: n * (n-1) * ... * (n-k+1)
    numerator = 1
    for i in range(k):
        numerator *= (n - i)

    # Calculate the denominator: k!
    denominator = factorial(k)

    return numerator // denominator


@njit
def init_thresholds(n_qubits):
    n_bias = n_qubits - 1
    thresholds = np.empty(n_bias, dtype=np.float64)
    tot_prob = 0
    p = n_qubits
    for q in range(1, n_qubits // 2):
        thresholds[q - 1] = p
        thresholds[n_bias - q] = p
        tot_prob += 2 * p
        p = comb(n_qubits, q + 1)
    if n_qubits & 1:
        thresholds[q - 1] = p
        tot_prob += p
    thresholds /= tot_prob

    return thresholds


@njit
def probability_by_hamming_weight(J, h, z, theta, t, n_bias, normalized=True):
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

    if normalized:
        bias /= bias.sum()

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
        bias = probability_by_hamming_weight(J_eff, h_t, z, theta_eff, t, n_bias, False)
        last_bias = probability_by_hamming_weight(J_eff, h_t, z, theta_eff, tm1, n_bias, False)
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

        if (cum_prob[m] >= p) and ((m == 0) or (cum_prob[m - 1] < p)):
            break

        if cum_prob[m] < p:
            left = m + 1
        else:
            right = m - 1

    return m


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


@njit
def gray_code_next(state, curr_idx, offset):
    prev = curr_idx
    curr = curr_idx + 1
    prev = prev ^ (prev >> 1)
    curr = curr ^ (curr >> 1)
    diff = prev ^ curr
    flip_bit = int(np.log2(diff))
    state[offset + flip_bit] = not state[offset + flip_bit]


@njit
def gray_mutation(index, seed_bits, offset):
    """Apply Gray-code-indexed bit flips to a seed bitstring."""
    n = seed_bits.shape[0]
    gray = index ^ (index >> 1)
    bits = seed_bits.copy()
    for i in range(n):
        bits[n - offset - 1 - i] ^= (gray >> i) & 1
    return bits
