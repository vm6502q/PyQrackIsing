from .maxcut_tfim import maxcut_tfim
import itertools
import networkx as nx
import numpy as np
from numba import njit, prange
import os


def evaluate_cut_edges(state, edge_keys, edge_values):
    cut_value = 0
    for i in range(len(edge_values)):
        k = i << 1
        u, v = edge_keys[k], edge_keys[k + 1]
        if ((state >> u) & 1) != ((state >> v) & 1):
            cut_value += edge_values[i]

    return float(cut_value)


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
    j = [i * k for i in range(n)]
    for i in prange(n):
        energies[i] = bootstrap_worker(theta, edge_keys, edge_values, indices_array[j[i] : j[i] + k])

    return energies


# By Gemini (Google Search AI)
def int_to_bitstring(integer, length):
    return (bin(integer)[2:].zfill(length))[::-1]


def spin_glass_solver(G, quality=None, shots=None, correction_quality=None, best_guess=None):
    nodes = None
    n_qubits = 0
    G_m = None
    if isinstance(G, nx.Graph):
        nodes = list(G.nodes())
        n_qubits = len(nodes)
        G_m = nx.to_numpy_array(G, weight='weight', nonedge=0.0)
    else:
        n_qubits = len(G)
        nodes = list(range(n_qubits))
        G_m = G

    if n_qubits == 0:
        return "", 0, ([], []), 0

    if n_qubits == 1:
        return "0", 0, (nodes, []), 0

    if n_qubits == 2:
        weight = G_m[0, 1]
        if weight < 0.0:
            return "00", 0, (nodes, []), weight

        return "01", weight, ([nodes[0]], [nodes[1]]), -weight

    if correction_quality is None:
        # maxcut_tfim(G) scales roughly like n^4,
        # so its match order of overhead.
        correction_quality = 4

    bitstring = ""
    if isinstance(best_guess, str):
        bitstring = best_guess
    elif isinstance(best_guess, int):
        bitstring = int_to_bitstring(best_guess, n_qubits)
    elif isinstance(best_guess, list):
        bitstring = "".join(["1" if b else "0" for b in best_guess])
    else:
        bitstring, _, _ = maxcut_tfim(G_m, quality=quality, shots=shots)
    best_theta = [b == "1" for b in list(bitstring)]

    edge_keys = []
    edge_values = []
    for i in range(n_qubits):
        for j in range(i + 1, n_qubits):
            weight = G_m[i, j]
            if weight == 0.0:
                continue
            edge_keys.append(i)
            edge_keys.append(j)
            edge_values.append(weight)

    min_energy = compute_energy(best_theta, edge_keys, edge_values)
    improved = True
    while improved:
        improved = False
        for k in range(1, max(1, correction_quality + 1)):
            if n_qubits < k:
                break

            theta = best_theta.copy()

            combos = list(
                item for sublist in itertools.combinations(range(n_qubits), k) for item in sublist
            )
            energies = bootstrap(theta, edge_keys, edge_values, k, combos)

            energy = min(energies)
            if energy < min_energy:
                index_match = np.random.choice(np.where(energies == energy)[0])
                indices = combos[(index_match * k) : ((index_match + 1) * k)]
                min_energy = energy
                for i in indices:
                    best_theta[i] = not best_theta[i]
                improved = True
                break

    sample = 0
    bitstring = ""
    l, r = [], []
    for i in range(len(best_theta)):
        b = best_theta[i]
        if b:
            bitstring += "1"
            r.append(nodes[i])
            sample |= 1 << i
        else:
            bitstring += "0"
            l.append(nodes[i])

    cut_value = evaluate_cut_edges(sample, edge_keys, edge_values)

    return bitstring, float(cut_value), (l, r), float(min_energy)
