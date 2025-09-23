from .maxcut_tfim_sparse import maxcut_tfim_sparse
import itertools
import networkx as nx
import numpy as np
from numba import njit, prange
import os
from scipy.sparse import lil_matrix, csr_matrix


@njit
def evaluate_cut_edges(theta_bits, G_data, G_rows, G_cols):
    n = G_rows.shape[0] - 1
    cut = 0.0
    for u in range(n):
        for col in range(G_rows[u], G_rows[u + 1]):
            v = G_cols[col]
            if theta_bits[u] != theta_bits[v]:
                cut += G_data[col]

    return cut


@njit
def compute_energy(sample, G_data, G_rows, G_cols):
    n_qubits = G_rows.shape[0] - 1
    energy = 0
    for u in range(n_qubits):
        for col in range(G_rows[u], G_rows[u + 1]):
            v = G_cols[col]
            energy += G_data[col] * (1 if sample[u] == sample[v] else -1)

    return energy


@njit
def bootstrap_worker(theta, G_data, G_rows, G_cols, indices):
    local_theta = theta.copy()
    for i in indices:
        local_theta[i] = not local_theta[i]
    energy = compute_energy(local_theta, G_data, G_rows, G_cols)

    return energy


@njit(parallel=True)
def bootstrap(theta, G_data, G_rows, G_cols, k, indices_array):
    n = theta.shape[0]
    energies = np.empty(n, dtype=np.float32)
    for i in prange(n):
        j = i * k
        energies[i] = bootstrap_worker(theta, G_data, G_rows, G_cols, indices_array[j : j + k])

    return energies


def to_scipy_sparse_upper_triangular(G, nodes, n_nodes):
    lil = lil_matrix((n_nodes, n_nodes), dtype=np.float64)
    for u in range(n_nodes):
        u_node = nodes[u]
        for v in range(u + 1, n_nodes):
            v_node = nodes[v]
            if G.has_edge(u_node, v_node):
                lil[u, v] = G[u_node][v_node].get('weight', 1.0)

    return lil.tocsr()


# By Gemini (Google Search AI)
def int_to_bitstring(integer, length):
    return (bin(integer)[2:].zfill(length))[::-1]


def spin_glass_solver_sparse(G, quality=None, shots=None, best_guess=None):
    nodes = None
    n_qubits = 0
    G_m = None
    if isinstance(G, nx.Graph):
        nodes = list(G.nodes())
        n_qubits = len(nodes)
        G_m = to_scipy_sparse_upper_triangular(G, nodes, n_qubits)
    else:
        n_qubits = G.shape[0]
        nodes = list(range(n_qubits))
        G_m = G

    if n_qubits < 3:
        if n_qubits == 0:
            return "", 0, ([], []), 0

        if n_qubits == 1:
            return "0", 0, (nodes, []), 0

        if n_qubits == 2:
            weight = G_m[0, 1]
            if weight < 0.0:
                return "00", 0, (nodes, []), weight

            return "01", weight, ([nodes[0]], [nodes[1]]), -weight

    if quality is None:
        quuality = 6

    bitstring = ""
    if isinstance(best_guess, str):
        bitstring = best_guess
    elif isinstance(best_guess, int):
        bitstring = int_to_bitstring(best_guess, n_qubits)
    elif isinstance(best_guess, list):
        bitstring = "".join(["1" if b else "0" for b in best_guess])
    else:
        bitstring, _, _ = maxcut_tfim_sparse(G_m, quality=quality, shots=shots)
    best_theta = np.array([b == "1" for b in list(bitstring)], dtype=np.bool_)

    min_energy = compute_energy(best_theta, G_m.data, G_m.indptr, G_m.indices)
    improved = True
    correction_quality = 1
    while improved:
        improved = False
        k = 1
        while k <= correction_quality:
            if n_qubits < k:
                break

            theta = best_theta.copy()

            combos = list(
                item for sublist in itertools.combinations(range(n_qubits), k) for item in sublist
            )
            energies = bootstrap(theta, G_m.data, G_m.indptr, G_m.indices, k, combos)

            energy = energies.min()
            if energy < min_energy:
                index_match = np.random.choice(np.where(energies == energy)[0])
                indices = combos[(index_match * k) : ((index_match + 1) * k)]
                min_energy = energy
                for i in indices:
                    best_theta[i] = not best_theta[i]
                improved = True
                if correction_quality < (k + 1):
                    correction_quality = k + 1
                break

            k = k + 1

    bitstring = ""
    l, r = [], []
    for i in range(len(best_theta)):
        b = best_theta[i]
        if b:
            bitstring += "1"
            r.append(nodes[i])
        else:
            bitstring += "0"
            l.append(nodes[i])

    cut_value = evaluate_cut_edges(best_theta, G_m.data, G_m.indptr, G_m.indices)

    return bitstring, float(cut_value), (l, r), float(min_energy)
