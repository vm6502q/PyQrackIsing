from .maxcut_tfim import maxcut_tfim
import itertools
import multiprocessing
import os


def evaluate_cut_edges(state, edges):
    cut_edges = []
    cut_value = 0
    for key, weight in edges.items():
        if ((state >> key[0]) & 1) != ((state >> key[1]) & 1):
            cut_edges.append(key)
            cut_value += weight

    return float(cut_value), cut_edges


def compute_energy(theta_bits, edges):
    spins = {i: 1 if theta_bits[i] else -1 for i in range(len(theta_bits))}
    energy = sum(value * spins[key[0]] * spins[key[1]] for key, value in edges.items())

    return energy


# Parallelization by Elara (OpenAI custom GPT):
def bootstrap_worker(args):
    theta, edges, indices = args
    local_theta = theta.copy()
    flipped = []
    for i in indices:
        local_theta[i] = not local_theta[i]
        flipped.append(local_theta[i])
    energy = compute_energy(local_theta, edges)

    return indices, energy, flipped


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

    edges = {}
    for u, v, data in G.edges(data=True):
        edges[(u, v)] = data.get("weight", 1.0)

    min_energy = compute_energy(best_theta, edges)
    improved = True
    while improved:
        improved = False
        for k in range(1, quality + 1):
            if n_qubits < k:
                break

            theta = best_theta.copy()
            args = []

            for combo in itertools.combinations(range(n_qubits), k):
                args.append((theta, edges, combo))

            with multiprocessing.Pool(processes=os.cpu_count()) as pool:
                results = pool.map(bootstrap_worker, args)

            results.sort(key=lambda r: r[1])
            indices, energy, flipped = results[0]
            if energy < min_energy:
                min_energy = energy
                for i in range(len(indices)):
                    best_theta[indices[i]] = flipped[i]
                improved = True
                break

    sample = 0
    bitstring = ""
    for i in range(len(best_theta)):
        b = best_theta[i]
        bitstring += '1' if b else '0'
        if b:
            sample |= 1 << i

    cut_value, cut_edges = evaluate_cut_edges(sample, edges)

    return cut_value, bitstring, cut_edges, min_energy
