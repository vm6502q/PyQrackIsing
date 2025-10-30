from .maxcut_tfim_util import probability_by_hamming_weight, sample_mag, opencl_context
import math
from numba import njit
import numpy as np
import sys


epsilon = opencl_context.epsilon


def get_otoc_hamming_distribution(J=-1.0, h=2.0, z=4, theta=0.0, t=5, n_qubits=65, pauli_strings = ['X' + 'I' * 64]):
    n_bias = n_qubits + 1
    if h <= epsilon:
        bias = np.empty(n_bias, dtype=np.float64)
        bias[0] = 1.0
        return bias

    diff_x = np.empty(n_bias, dtype=np.float64)
    tot_prob = 0
    p = 1.0
    for q in range(n_qubits >> 1):
        diff_x[q] = p
        diff_x[n_bias - (q + 1)] = p
        tot_prob += 2 * p
        p = math.comb(n_qubits, q + 1)
    if n_qubits & 1:
        diff_x[n_qubits >> 1] = p
        tot_prob += p
    diff_x *= n_qubits / tot_prob

    signal_frac = 0.0
    diff_z = np.zeros(n_bias, dtype=np.float64)
    diff_z[0] = n_qubits
    for pauli_string in pauli_strings:
        pauli_string = list(pauli_string)
        if len(pauli_string) != n_qubits:
            raise ValueError("OTOCS pauli_string must be same length as n_qubits! (Use 'I' for qubits that aren't changed.)")

        term_signal = 0.5 * pauli_string.count('X') + pauli_string.count('Z') + 1.5 * pauli_string.count('Y')
        if term_signal == 0:
            continue

        signal_frac -= term_signal

        fwd = probability_by_hamming_weight(J, h, z, theta, t, n_qubits + 1)
        rev = probability_by_hamming_weight(-J, -h, z, theta + np.pi, t, n_qubits + 1)
        diff_theta = rev - fwd

        phi = theta + np.pi / 2
        fwd = probability_by_hamming_weight(-h, -J, z, phi, t, n_qubits + 1)
        rev = probability_by_hamming_weight(h, J, z, phi - np.pi, t, n_qubits + 1)
        diff_phi = rev - fwd

        for b in pauli_string:
            match b:
                case 'X':
                    diff_z += diff_theta
                case 'Z':
                    diff_x += diff_phi
                case 'Y':
                    diff_z += diff_theta
                    diff_x += diff_phi
                case _:
                    pass

    # Normalize:
    diff_z /= diff_z.sum()
    diff_x /= diff_x.sum()

    signal_frac = 2 ** signal_frac
    diff_z = signal_frac * diff_z + (1 - signal_frac) * diff_x

    # Normalize:
    diff_z /= diff_z.sum()

    return diff_z


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
def find_all_str_occurrences(main_string, sub_string):
    indices = []
    start_index = 0
    while True:
        index = main_string.find(sub_string, start_index)
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


def get_willow_inv_dist(butterfly_idx_x, butterfly_idx_z, n_qubits, row_len, col_len, t):
    inv_dist = np.zeros(n_qubits, dtype=np.float64)
    for idx in butterfly_idx_x:
        b_row, b_col = divmod(idx, row_len)
        for q in range(n_qubits):
            q_row, q_col = divmod(q, row_len)
            inv_dist[q] += abs(q_row - b_row) + abs(q_col - b_col)
    for idx in butterfly_idx_z:
        b_row, b_col = divmod(idx, row_len)
        for q in range(n_qubits):
            q_row, q_col = divmod(q, row_len)
            inv_dist[q] -= abs(q_row - b_row) + abs(q_col - b_col)
    inv_dist = 2 ** (inv_dist / t)

    return inv_dist


def get_inv_dist(butterfly_idx_x, butterfly_idx_z, n_qubits, row_len, col_len, t):
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
            inv_dist[q] += row_d + col_d
    for idx in butterfly_idx_z:
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
    inv_dist = 2 ** (inv_dist / t)

    return inv_dist


def generate_otoc_samples(J=-1.0, h=2.0, z=4, theta=0.0, t=5, n_qubits=65, pauli_strings = ['X' + 'I' * 64], shots=100, is_orbifold=True):
    thresholds = fix_cdf(get_otoc_hamming_distribution(J, h, z, theta, t, n_qubits, pauli_strings))

    row_len, col_len = factor_width(n_qubits)
    inv_dist = np.zeros(n_qubits, dtype=np.float64)
    for pauli_string in pauli_strings:
        if (pauli_string.count('X') + pauli_string.count('Y') + pauli_string.count('Z')) == 0:
            continue
        butterfly_idx_x = find_all_str_occurrences(pauli_string, 'X')
        butterfly_idx_z = find_all_str_occurrences(pauli_string, 'Z')
        if is_orbifold:
            inv_dist += get_inv_dist(butterfly_idx_x, butterfly_idx_z, n_qubits, row_len, col_len, t)
        else:
            inv_dist += get_willow_inv_dist(butterfly_idx_x, butterfly_idx_z, n_qubits, row_len, col_len, t)
        inv_dist /= 2.0

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
