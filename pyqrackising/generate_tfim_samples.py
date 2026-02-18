from .maxcut_tfim_util import probability_by_hamming_weight, sample_mag, opencl_context
import itertools
import math
import sys
import numpy as np
from numba import njit

from collections import Counter


epsilon = opencl_context.epsilon


@njit
def factor_width(width, is_transpose=False):
    col_len = math.floor(math.sqrt(width))
    while ((width // col_len) * col_len) != width:
        col_len -= 1
    row_len = width // col_len

    return (col_len, row_len) if is_transpose else (row_len, col_len)


# By Gemini (Google Search AI)
def int_to_bitstring(integer, length):
    return bin(integer)[2:].zfill(length)


# Drafted by Elara (OpenAI custom GPT), improved by Dan Strano
def closeness_like_bits(perm, n_rows, n_cols):
    """
    Compute closeness-of-like-bits metric C(state) for a given bitstring on an LxL toroidal grid.

    Parameters:
        perm: integer representing basis state, bit-length n_rows * n_cols
        n_rows: row count of torus
        n_cols: column count of torus

    Returns:
        normalized_closeness: float, in [-1, +1]
            +1 means all neighbors are like-like, -1 means all neighbors are unlike
    """
    # reshape the bitstring into LxL grid
    bitstring = list(int_to_bitstring(perm, n_rows * n_cols))
    grid = np.array(bitstring).reshape((n_rows, n_cols))
    total_edges = 0
    like_count = 0

    # iterate over each site, count neighbors (right and down to avoid double-count)
    for i in range(n_rows):
        for j in range(n_cols):
            s = grid[i, j]

            # right neighbor (wrap around)
            s_right = grid[i, (j + 1) % n_cols]
            like_count += 1 if s == s_right else -1
            total_edges += 1

            # down neighbor (wrap around)
            s_down = grid[(i + 1) % n_rows, j]
            like_count += 1 if s == s_down else -1
            total_edges += 1

    # normalize
    normalized_closeness = like_count / total_edges
    return normalized_closeness


@njit
def comb(n, k):
    if (k < 0) or (k > n):
        return 0
    if (k == 0) or (k == n):
        return 1
    res = 1
    for i in range(1, k + 1):
        res = res * (n - k + i) // i
    return res


@njit
def expected_closeness_weight(n_rows, n_cols, hamming_weight):
    L = n_rows * n_cols
    same_pairs = comb(hamming_weight, 2) + comb(L - hamming_weight, 2)
    total_pairs = comb(L, 2)
    mu_k = same_pairs / total_pairs
    return 2.0 * mu_k - 1.0


@njit
def sample_hamming_weight(thresholds, shots):
    hamming_samples = np.zeros(shots, dtype=np.int32)
    for s in range(shots):
        hamming_samples[s] = sample_mag(thresholds)

    return hamming_samples

@njit
def fix_cdf(hamming_prob):
    tot_prob = 0.0
    n_bias = len(hamming_prob)
    cum_prob = np.empty(n_bias, dtype=np.float64)
    for i in range(n_bias):
        tot_prob += hamming_prob[i]
        cum_prob[i] = tot_prob
    cum_prob[-1] = 1.0

    return cum_prob


@njit
def get_tfim_hamming_distribution(J=-1.0, h=2.0, z=4, theta=0.174532925199432957, t=5, n_qubits=56, omega = 1.5 * np.pi):
    if abs(t) <= epsilon:
        p = (1.0 - np.cos(theta)) / 2.0
        bias = np.empty(n_qubits + 1, dtype=np.float64)
        for k in range(n_qubits + 1):
            bias[k] = comb(n_qubits, k) * (p ** k) * ((1.0 - p) ** (n_qubits - k))

        return bias / bias.sum()

    if abs(h) <= epsilon:
        bias = np.empty(n_qubits + 1, dtype=np.float64)
        if J > 0:
            bias[-1] = 1.0
        else:
            bias[0] = 1.0
        return bias

    bias = probability_by_hamming_weight(J, h, z, theta, t, n_qubits + 1, normalized=True, omega = omega)

    return bias / bias.sum()


def generate_tfim_samples(
    J=-1.0, h=2.0, z=4, theta=0.174532925199432957, t=5, n_qubits=56, shots=100, omega = 1.5 * np.pi
):
    samples = []

    if abs(t) <= epsilon:
        prob = (1.0 - np.cos(theta)) / 2.0
        for s in range(shots):
            sample = 0
            for q in range(n_qubits):
                sample <<= 1
                if np.random.random() < prob:
                    sample |= 1
            samples.append(sample)

        return samples

    n_rows, n_cols = factor_width(n_qubits)

    # First dimension: Hamming weight
    bias = get_tfim_hamming_distribution(J=J, h=h, z=z, theta=theta, t=t, n_qubits=n_qubits, omega=omega)
    thresholds = fix_cdf(bias)
    hamming_samples = dict(Counter(sample_hamming_weight(thresholds, shots)))

    for h_weight, count in hamming_samples.items():
        if h_weight == 0:
            samples += count * [0]
            continue

        if h_weight == n_qubits:
            samples += count * [(1 << n_qubits) - 1]
            continue

        rands = [np.random.random() for _ in range(count)]
        rands.sort()
        state_int = 0
        tot_prob = 0
        s = 0
        # How closely grouped are "like" bits to "like"?
        expected_closeness = expected_closeness_weight(n_rows, n_cols, h_weight)
        h_weight_combos = math.comb(n_qubits, h_weight)
        for combo in itertools.combinations(range(n_qubits), h_weight):
            state_int = sum(1 << pos for pos in combo)
            # When we add all "closeness" possibilities for the particular Hamming weight, we should maintain the (n+1) mean probability dimensions.
            normed_closeness = (1 + closeness_like_bits(state_int, n_rows, n_cols)) / (1 + expected_closeness)
            # Use a normalized weighted average that favors the (n+1)-dimensional model at later times.
            # The (n+1)-dimensional marginal probability is the product of a function of Hamming weight and "closeness," split among all basis states with that specific Hamming weight.
            tot_prob += normed_closeness / h_weight_combos
            while (rands[s] <= tot_prob):
                samples.append(state_int)
                s += 1
                if s == count:
                    break
            if s == count:
                break
        if s < count:
            samples += (count - s) * [state_int]


    np.random.shuffle(samples)

    return samples

@njit
def get_fh_hamming_distribution(J=-1.0, h=2.0, z=4, theta=0.174532925199432957, t=5, n_qubits=56, omega = 0.5 * np.pi):
    bias_z = get_tfim_hamming_distribution(J=J, h=h, z=z, theta=theta+np.pi/2, t=t, n_qubits=n_qubits, omega=omega)
    bias_x = get_tfim_hamming_distribution(J=-h, h=-J, z=z, theta=theta+np.pi, t=t, n_qubits=n_qubits, omega=omega)
    return [(z + x) / 2 for z, x in zip(bias_z, bias_x)]

def generate_fh_samples(
    J=-1.0, h=2.0, z=4, theta=0.174532925199432957, t=5, n_qubits=56, shots=100, omega = 1.5 * np.pi
):
    samples = []

    if abs(t) <= epsilon:
        prob = (1.0 - np.cos(theta)) / 2.0
        for s in range(shots):
            sample = 0
            for q in range(n_qubits):
                sample <<= 1
                if np.random.random() < prob:
                    sample |= 1
            samples.append(sample)

        return samples

    n_rows, n_cols = factor_width(n_qubits)

    # First dimension: Hamming weight
    bias = get_fh_hamming_distribution(J=J, h=h, z=z, theta=theta, t=t, n_qubits=n_qubits, omega=omega)
    thresholds = fix_cdf(bias)
    hamming_samples = dict(Counter(sample_hamming_weight(thresholds, shots)))

    for h_weight, count in hamming_samples.items():
        if h_weight == 0:
            samples += count * [0]
            continue

        if h_weight == n_qubits:
            samples += count * [(1 << n_qubits) - 1]
            continue

        rands = [np.random.random() for _ in range(count)]
        rands.sort()
        state_int = 0
        tot_prob = 0
        s = 0
        # How closely grouped are "like" bits to "like"?
        expected_closeness = expected_closeness_weight(n_rows, n_cols, h_weight)
        h_weight_combos = math.comb(n_qubits, h_weight)
        for combo in itertools.combinations(range(n_qubits), h_weight):
            state_int = sum(1 << pos for pos in combo)
            # When we add all "closeness" possibilities for the particular Hamming weight, we should maintain the (n+1) mean probability dimensions.
            normed_closeness = (1 + closeness_like_bits(state_int, n_rows, n_cols)) / (1 + expected_closeness)
            # Use a normalized weighted average that favors the (n+1)-dimensional model at later times.
            # The (n+1)-dimensional marginal probability is the product of a function of Hamming weight and "closeness," split among all basis states with that specific Hamming weight.
            tot_prob += normed_closeness / h_weight_combos
            while (rands[s] <= tot_prob):
                samples.append(state_int)
                s += 1
                if s == count:
                    break
            if s == count:
                break
        if s < count:
            samples += (count - s) * [state_int]


    np.random.shuffle(samples)

    return samples
        
