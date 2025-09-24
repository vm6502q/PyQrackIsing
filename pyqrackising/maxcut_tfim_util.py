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
def init_theta(delta_t, tot_t, h_mult, n_qubits, J_eff, degrees):
    theta = np.empty(n_qubits, dtype=np.float32)
    for q in prange(n_qubits):
        J = J_eff[q]
        z = degrees[q]
        theta[q] = np.arcsin(
            max(
                -1.0,
                min(
                    1.0,
                    (1.0 if J > 0.0 else -1.0) if np.isclose(abs(z * J), 0.0) else (abs(h_mult) / (z * J)),
                ),
            )
        )

    return theta


@njit
def init_thresholds(n_qubits):
    n_bias = n_qubits - 1
    thresholds = np.empty(n_bias, dtype=np.float32)
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
def low_width(G_m, nodes, n_qubits):
    if n_qubits == 0:
        return "", 0, ([], [])

    if n_qubits == 1:
        return "0", 0, (nodes, [])

    if n_qubits == 2:
        weight = G_m[0, 1]
        if weight < 0.0:
            return "00", 0, (nodes, [])

        return "01", weight, ([nodes[0]], [nodes[1]])


@njit(parallel=True)
def maxcut_hamming_cdf(n_qubits, J_func, degrees, quality, hamming_prob):
    if n_qubits < 2:
        hamming_prob.fill(0.0)
        return

    n_steps = 1 << quality
    delta_t = 1.0 / n_steps
    tot_t = 2.0 * n_steps * delta_t
    h_mult = 2.0 / tot_t
    n_bias = n_qubits - 1

    theta = np.empty(n_qubits, dtype=np.float32)
    for q in range(n_qubits):
        J = J_func[q]
        z = degrees[q]
        theta[q] = np.arcsin(
            max(
                -1.0,
                min(
                    1.0,
                    (1.0 if J > 0.0 else -1.0) if np.isclose(abs(z * J), 0.0) else (abs(h_mult) / (z * J)),
                ),
            )
        )

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
        bias = probability_by_hamming_weight(J_eff, h_t, z, theta_eff, t, n_qubits)
        last_bias = probability_by_hamming_weight(J_eff, h_t, z, theta_eff, tm1, n_qubits)
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
def probability_by_hamming_weight(J, h, z, theta, t, n_qubits):
    bias = np.empty(n_qubits - 1, dtype=np.float32)

    # critical angle
    theta_c = np.arcsin(max(-1.0, min(1.0, abs(h) / (z * J))))

    p = (
        pow(2.0, abs(J / h) - 1.0)
        * (1.0 + np.sin(theta - theta_c) * np.cos(1.5 * np.pi * J * t + theta) / (1.0 + np.sqrt(t)))
        - 0.5
    )

    if (p * n_qubits) >= 1024:
        return bias

    tot_n = 1.0 + 1.0 / pow(2.0, p * n_qubits)
    factor = pow(2.0, -p)
    n = 1.0
    for q in range(1, n_qubits):
        n *= factor
        bias[q - 1] = n
        tot_n += n
    bias /= tot_n

    if J > 0.0:
        return bias[::-1]

    return bias


class OpenCLContext:
    def __init__(self, a, c, q, k):
        self.IS_OPENCL_AVAILABLE = a
        self.ctx = c
        self.queue = q
        self.maxcut_hamming_cdf_kernel = k

IS_OPENCL_AVAILABLE = True
ctx = None
queue = None
maxcut_hamming_cdf_kernel = None
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
    print("PyOpenCL not installed. (If you have any OpenCL accelerator devices with available ICDs, you might want to optionally install pyopencl.)")

opencl_context = OpenCLContext(IS_OPENCL_AVAILABLE, ctx, queue, maxcut_hamming_cdf_kernel)
