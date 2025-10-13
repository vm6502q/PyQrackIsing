from .maxcut_tfim_util import probability_by_hamming_weight, sample_mag
import itertools
import math
import numpy as np
from numba import njit


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
    hamming_prob /= hamming_prob.sum()
    tot_prob = 0.0
    n_bias = len(hamming_prob)
    cum_prob = np.empty(n_bias, dtype=np.float64)
    for i in range(n_bias):
        tot_prob += hamming_prob[i]
        cum_prob[i] = tot_prob
    cum_prob[-1] = 2.0

    return cum_prob


def generate_tfim_samples(
    J=-1.0, h=2.0, z=4, theta=0.174532925199432957, t=5, n_qubits=56, shots=100
):
    n_rows, n_cols = factor_width(n_qubits)

    # First dimension: Hamming weight
    thresholds = fix_cdf(probability_by_hamming_weight(J, h, z, theta, t, n_qubits + 1))
    hamming_samples = sample_hamming_weight(thresholds, shots)

    samples = []
    for m in range(len(hamming_samples)):
        # Second dimension: permutation within Hamming weight
        # (Written with help from Elara, the custom OpenAI GPT)
        hs = hamming_samples[m]
        rands = []
        for s in range(hs):
            rands.append(np.random.random())
        rands.sort()
        state_int = 0
        tot_prob = 0
        s = 0
        for combo in itertools.combinations(range(n_qubits), m):
            state_int = sum(1 << pos for pos in combo)
            tot_prob += (1.0 + closeness_like_bits(state_int, n_rows, n_cols)) / (
                1.0 + expected_closeness_weight(n_rows, n_cols, m)
            )
            while (s < hs) and (rands[s] <= tot_prob):
                samples.append(state_int)
                s += 1
            if s == hs:
                break
        for r in range(hs - s):
            samples.append(state_int)

    np.random.shuffle(samples)

    return samples
