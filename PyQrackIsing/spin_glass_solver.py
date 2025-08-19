from .maxcut_tfim import maxcut_tfim
import itertools
import numpy as np
from numba import njit, prange
import os


def evaluate_cut_edges(state, edge_keys, edge_values):
    cut_edges = []
    cut_value = 0
    for i in range(len(edge_values)):
        k = i << 1
        u, v = edge_keys[k], edge_keys[k + 1]
        if ((state >> u) & 1) != ((state >> v) & 1):
            cut_edges.append((u, v))
            cut_value += edge_values[i]

    return float(cut_value), cut_edges


@njit
def compute_energy(theta_bits, edge_keys, edge_values):
    energy = 0
    for i in range(len(edge_values)):
        k = i << 1
        u, v = edge_keys[k], edge_keys[k + 1]
        spin_u = 1 if theta_bits[u] else -1
        spin_v = 1 if theta_bits[v] else -1
        energy += edge_values[i] * spin_u * spin_v

    return energy


@njit
def bootstrap_worker(theta, edge_keys, edge_values, indices):
    local_theta = theta.copy()
    for i in indices:
        local_theta[i] = not local_theta[i]
    energy = compute_energy(local_theta, edge_keys, edge_values)

    return energy


@njit(parallel=True)
def bootstrap(theta, edge_keys, edge_values, k, indices_array):
    n = len(indices_array) // k
    energies = np.empty(n)
    j = 0
    for i in prange(n):
        energies[i] = bootstrap_worker(theta, edge_keys, edge_values, indices_array[j:j+k])
        j += k

    return energies


# By Gemini (Google Search AI)
def int_to_bitstring(integer, length):
    return (bin(integer)[2:].zfill(length))[::-1]


def spin_glass_solver(G, quality=2, best_guess=None):
    bitstring = ''
    if isinstance(best_guess, str):
        bitstring = best_guess
    elif isinstance(best_guess, int):
        bitstring = int_to_bitstring(best_guess)
    elif isinstance(best_guess, list):
        bitstring = "".join(["1" if b else "0" for b in best_guess])
    else:
        cut_value, bitstring, cut_edges = maxcut_tfim(G, quality=0)
    best_theta = [ b == '1' for b in list(bitstring)]
    n_qubits = len(best_theta)

    edge_keys = []
    edge_values = []
    for u, v, data in G.edges(data=True):
        edge_keys.append(u)
        edge_keys.append(v)
        edge_values.append(data.get("weight", 1.0))

    min_energy = compute_energy(best_theta, edge_keys, edge_values)
    improved = True
    while improved:
        improved = False
        for k in range(1, quality + 1):
            if n_qubits < k:
                break

            theta = best_theta.copy()

            combos = list(item for sublist in itertools.combinations(range(n_qubits), k) for item in sublist)
            energies = bootstrap(theta, edge_keys, edge_values, k, combos)

            energy = min(energies)
            if energy < min_energy:
                index_match = np.random.choice(np.where(energies == energy)[0])
                indices = combos[(index_match * k):((index_match + 1) * k)]
                min_energy = energy
                for i in indices:
                    best_theta[i] = not best_theta[i]
                improved = True
                break

    sample = 0
    bitstring = ""
    for i in range(len(best_theta)):
        b = best_theta[i]
        bitstring += '1' if b else '0'
        if b:
            sample |= 1 << i

    cut_value, cut_edges = evaluate_cut_edges(sample, edge_keys, edge_values)

    return cut_value, bitstring, cut_edges, min_energy
