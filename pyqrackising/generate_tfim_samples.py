from .maxcut_tfim_util import probability_by_hamming_weight, sample_mag, opencl_context
import itertools
import math
import random
import sys
import numpy as np
from numba import njit

from collections import Counter


epsilon = opencl_context.epsilon


# Fixed-Hamming-weight Kawasaki swaps with acceptance probability, by Elara
def random_fixed_hamming_state(n_qubits, h):
    bits = np.zeros(n_qubits, dtype=np.bool_)
    bits[:h] = True
    np.random.shuffle(bits)
    state = 0
    for b in bits:
        state <<= 1
        if b:
            state |= 1

    return state


def build_neighbors(n_rows, n_cols):
    L = n_rows * n_cols
    neighbors = [[] for _ in range(L)]

    for i in range(n_rows):
        for j in range(n_cols):
            idx = i * n_cols + j

            right = i * n_cols + ((j + 1) % n_cols)
            left = i * n_cols + ((j - 1) % n_cols)
            down = ((i + 1) % n_rows) * n_cols + j
            up = ((i - 1) % n_rows) * n_cols + j

            neighbors[idx] = [right, left, down, up]

    return neighbors


def count_like_edges(state, neighbors):
    like = 0
    visited = set()

    for i, nbrs in enumerate(neighbors):
        si = (state >> i) & 1
        for j in nbrs:
            if (j, i) in visited:
                continue
            sj = (state >> j) & 1
            if si == sj:
                like += 1
            visited.add((i, j))

    return like


def delta_like_edges(state, i, j, neighbors):
    si = (state >> i) & 1
    sj = (state >> j) & 1

    delta = 0

    for idx in [i, j]:
        s_old = (state >> idx) & 1

        for nbr in neighbors[idx]:
            if nbr == i or nbr == j:
                continue

            s_nbr = (state >> nbr) & 1

            old_like = s_old == s_nbr
            new_spin = sj if idx == i else si
            new_like = new_spin == s_nbr

            delta += int(new_like) - int(old_like)

    # handle edge between i and j if neighbors
    if j in neighbors[i]:
        old_like = si == sj
        new_like = sj == si
        delta += int(new_like) - int(old_like)

    return delta


def sample_fixed_hamming_weight(h_weight, count, n_rows, n_cols, burnin=10):
    if count == 0:
        return []

    n_qubits = n_rows * n_cols
    neighbors = build_neighbors(n_rows, n_cols)

    like_edges = 0
    while like_edges == 0:
        state = random_fixed_hamming_state(n_qubits, h_weight)
        like_edges = count_like_edges(state, neighbors)

    samples = []

    burn = burnin * n_qubits
    thinning = n_qubits

    ones = set([i for i in range(n_qubits) if (state >> i) & 1])
    zeros = set([i for i in range(n_qubits) if not (state >> i) & 1])

    for step in range(burn + thinning * count):
        # choose random 1-bit and 0-bit

        i = random.choice(tuple(ones))
        j = random.choice(tuple(zeros))

        delta = delta_like_edges(state, i, j, neighbors)
        new_like = like_edges + delta

        if new_like > 0:
            accept_prob = new_like / like_edges
            if np.random.random() < accept_prob:
                state ^= 1 << i
                ones.remove(i)
                zeros.add(i)

                state ^= 1 << j
                zeros.remove(i)
                ones.add(i)

                like_edges = new_like

        if step >= burn and (step - burn) % thinning == 0:
            samples.append(state)

    return samples


@njit
def factor_width(width, is_transpose=False):
    col_len = math.floor(math.sqrt(width))
    while ((width // col_len) * col_len) != width:
        col_len -= 1
    row_len = width // col_len

    return (col_len, row_len) if is_transpose else (row_len, col_len)


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
def get_tfim_hamming_distribution(J=-1.0, h=2.0, z=4, theta=0.174532925199432957, t=5, n_qubits=56, omega=1.5 * np.pi):
    if abs(t) <= epsilon:
        p = (1.0 - np.cos(theta)) / 2.0
        bias = np.empty(n_qubits + 1, dtype=np.float64)
        for k in range(n_qubits + 1):
            bias[k] = comb(n_qubits, k) * (p**k) * ((1.0 - p) ** (n_qubits - k))

        return bias / bias.sum()

    if abs(h) <= epsilon:
        bias = np.empty(n_qubits + 1, dtype=np.float64)
        if J > 0:
            bias[-1] = 1.0
        else:
            bias[0] = 1.0
        return bias

    bias = probability_by_hamming_weight(J, h, z, theta, t, n_qubits + 1, normalized=True, omega=omega)

    return bias / bias.sum()


def generate_tfim_samples(J=-1.0, h=2.0, z=4, theta=0.174532925199432957, t=5, n_qubits=56, shots=100):
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
    bias = get_tfim_hamming_distribution(J=J, h=h, z=z, theta=theta, t=t, n_qubits=n_qubits)
    counts = np.random.multinomial(shots, bias)

    samples += [0] * counts[0]
    samples += [(1 << n_qubits) - 1] * counts[-1]
    if n_qubits > 1:
        samples += [int(1 << np.random.randint(n_qubits)) for _ in range(counts[1])]
    if n_qubits > 2:
        mask = (1 << n_qubits) - 1
        samples += [(mask ^ int(1 << np.random.randint(n_qubits))) for _ in range(counts[-2])]

    # Second dimension: magnetic localization
    for h_weight in range(2, len(bias) - 2):
        samples += sample_fixed_hamming_weight(h_weight, counts[h_weight], n_rows, n_cols)

    np.random.shuffle(samples)

    return samples


def generate_fermi_hubbard_samples(J=-1.0, h=2.0, z=4, theta=0.174532925199432957, t=5, n_qubits=56, shots=100, omega=1.5 * np.pi):
    shots_x, shots_y, shots_z = np.random.multinomial(shots, [1 / 3, 1 / 3, 1 / 3])
    return (
        generate_tfim_samples(J=J, h=h, z=z, theta=theta, t=t, n_qubits=n_qubits, shots=shots_z)
        + generate_tfim_samples(J=-h, h=-J, z=z, theta=theta + np.pi / 2, t=t, n_qubits=n_qubits, shots=shots_x)
        + generate_tfim_samples(J=J, h=h, z=z, theta=theta + np.pi / 2, t=t, n_qubits=n_qubits, shots=shots_y)
    )
