from .maxcut_tfim_util import init_thresholds, probability_by_hamming_weight, sample_mag, opencl_context
import math
from numba import njit
import numpy as np
import sys


epsilon = opencl_context.epsilon


def hadamard(dist):
    """Krawtchouk/MacWilliams transform -- the bitwise Hadamard transform
    restricted to Hamming-weight-symmetric distributions. Built via the
    stable 3-term Krawtchouk recurrence rather than the closed-form
    alternating sum of binomial coefficients, which suffers catastrophic
    cancellation for larger n_qubits.

    Provided by (Anthropic) Claude."""
    n = len(dist) - 1
    k_arr = np.arange(n + 1)
    K_prev = np.ones(n + 1)                 # K_0(x) = 1
    result = np.empty(n + 1)
    result[0] = np.dot(dist, K_prev)
    if n >= 1:
        K_curr = n - 2 * k_arr              # K_1(x) = n - 2x
        result[1] = np.dot(dist, K_curr)
        for j in range(1, n):
            K_next = ((n - 2 * k_arr) * K_curr - (n - j + 1) * K_prev) / (j + 1)
            result[j + 1] = np.dot(dist, K_next)
            K_prev, K_curr = K_curr, K_next
    tmp = result / (2 ** n)
    tmp /= tmp.sum()

    return tmp


def get_otoc_hamming_distribution(J=-1.0, h=2.0, z=4, theta=0.0, t=5, n_qubits=65, pauli_strings=["X" + "I" * 64]):
    n_bias = n_qubits + 1
    if (abs(h) <= epsilon) or (abs(t) <= epsilon):
        bias = np.empty(n_bias, dtype=np.float64)
        bias[0] = 1.0
        return bias

    # The removed items are identity operation, round-trip.
    pauli_strings = [item for item in pauli_strings if item.count("I") != n_qubits]

    signal_frac_x = 0
    signal_frac_z = 0

    for pauli_string in pauli_strings:
        pauli_string = list(pauli_string)
        if len(pauli_string) != n_qubits:
            raise ValueError("OTOCS pauli_string must be same length as n_qubits! (Use 'I' for qubits that aren't changed.)")

        count_y = pauli_string.count("Y")
        signal_frac_x += pauli_string.count("X") + count_y
        signal_frac_z += pauli_string.count("Z") + count_y

    x_basis = init_thresholds(n_qubits, theta)
    if signal_frac_x:
        phi = theta + np.pi / 2
        fwd_x = probability_by_hamming_weight(-h, -J, z, phi, t, n_qubits + 1)
        rev = probability_by_hamming_weight(h, J, z, phi - np.pi, t, n_qubits + 1)
        diff_x = rev - fwd_x
        diff_x -= diff_x.mean()
        signal_frac_x /= n_qubits
        fwd_x += signal_frac_x * diff_x
        signal_frac_x /= len(pauli_strings)
        x_basis = (1.0 - signal_frac_x) * x_basis + signal_frac_x * fwd_x
        x_min = x_basis.min()
        if x_min < 0:
            x_basis -= x_min
            x_basis /= x_basis.sum()

    z_basis = hadamard(x_basis)
    if signal_frac_z:
        fwd_z = probability_by_hamming_weight(J, h, z, theta, t, n_qubits + 1)
        rev = probability_by_hamming_weight(-J, -h, z, theta + np.pi, t, n_qubits + 1)
        diff_z = rev - fwd_z
        diff_z -= diff_z.mean()
        signal_frac_z /= n_qubits
        fwd_z += signal_frac_z * diff_z
        signal_frac_z /= len(pauli_strings)
        z_basis = (1.0 - signal_frac_z) * z_basis + signal_frac_z * fwd_z
        z_min = z_basis.min()
        if z_min < 0:
            z_basis -= z_min
            z_basis /= z_basis.sum()

    return z_basis


@njit
def fix_cdf(hamming_prob):
    tot_prob = 0.0
    n_bias = len(hamming_prob)
    cum_prob = np.empty(n_bias, dtype=np.float64)
    for i in range(n_bias):
        tot_prob += hamming_prob[i]
        cum_prob[i] = tot_prob
    cum_prob[-1] = 2.0

    return cum_prob


@njit
def factor_width(width):
    col_len = int(np.floor(np.sqrt(width)))
    while ((width // col_len) * col_len) != width:
        col_len -= 1
    row_len = width // col_len

    return row_len, col_len


# Provided by Google search AI
def find_all_bit_flips(main_string):
    indices = []
    start_index = 0
    while True:
        index = main_string.find("X", start_index)
        if index == -1:
            break
        indices.append(index)
        start_index = index + 1  # Start searching after the found occurrence
    start_index = 0
    while True:
        index = main_string.find("Y", start_index)
        if index == -1:
            break
        indices.append(index)
        start_index = index + 1  # Start searching after the found occurrence

    return indices


def take_sample(n_qubits, sample, m, inv_dist):
    indices = [i for i in range(n_qubits)]
    tot_inv_dist = 0.0
    for i in range(n_qubits):
        tot_inv_dist += inv_dist[i]
    selected = []
    for i in range(m):
        r = tot_inv_dist * np.random.random()
        p = inv_dist[indices[0]]
        idx = 0
        while p < r:
            idx += 1
            if idx >= len(indices):
                idx = len(indices) - 1
                break
            p += inv_dist[indices[idx]]
        i = indices[idx]
        del indices[idx]
        selected.append(i)
        tot_inv_dist -= inv_dist[i]
    for i in selected:
        sample |= 1 << i

    return sample


def get_willow_inv_dist(butterfly_idx_x, n_qubits, row_len, col_len, t):
    inv_dist = np.zeros(n_qubits, dtype=np.float64)
    for idx in butterfly_idx_x:
        b_row, b_col = divmod(idx, row_len)
        for q in range(n_qubits):
            q_row, q_col = divmod(q, row_len)
            inv_dist[q] -= abs(q_row - b_row) + abs(q_col - b_col)
    inv_dist /= t

    return inv_dist


def get_inv_dist(butterfly_idx_x, n_qubits, row_len, col_len, t):
    inv_dist = np.zeros(n_qubits, dtype=np.float64)
    half_row = row_len >> 1
    half_col = col_len >> 1
    for idx in butterfly_idx_x:
        b_row, b_col = divmod(idx, row_len)
        for q in range(n_qubits):
            q_row, q_col = divmod(q, row_len)
            row_d = abs(q_row - b_row)
            if row_d > half_row:
                row_d = row_len - row_d
            col_d = abs(q_col - b_col)
            if col_d > half_col:
                col_d = col_len - col_d
            inv_dist[q] -= row_d + col_d
    inv_dist /= t

    return inv_dist


def generate_otoc_samples(
    J=-1.0,
    h=2.0,
    z=4,
    theta=0.0,
    t=5,
    n_qubits=65,
    pauli_strings=["X" + "I" * 64],
    shots=100,
    is_orbifold=True,
):
    thresholds = fix_cdf(get_otoc_hamming_distribution(J, h, z, theta, t, n_qubits, pauli_strings))
    row_len, col_len = factor_width(n_qubits)
    inv_dist = np.zeros(n_qubits, dtype=np.float64)
    for i, pauli_string in enumerate(pauli_strings):
        if pauli_string.count("I") == n_qubits:
            continue
        butterfly_idx_x = find_all_bit_flips(pauli_string)
        if is_orbifold:
            inv_dist = 0.5 * inv_dist + get_inv_dist(butterfly_idx_x, n_qubits, row_len, col_len, t)
        else:
            inv_dist = 0.5 * inv_dist + get_willow_inv_dist(butterfly_idx_x, n_qubits, row_len, col_len, t)
    inv_dist = 2 ** inv_dist

    qubit_pows = [1 << q for q in range(n_qubits)]
    samples = []
    for _ in range(shots):
        # First dimension: Hamming weight
        m = sample_mag(thresholds)
        if m == 0:
            samples.append(0)
            continue
        if m >= n_qubits:
            samples.append((1 << n_qubits) - 1)
            continue

        # Second dimension: permutation within Hamming weight
        samples.append(take_sample(n_qubits, 0, m, inv_dist))

    return samples
