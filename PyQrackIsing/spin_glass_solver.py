from .maxcut_tfim import maxcut_tfim
import itertools
import multiprocessing
import os


def evaluate_cut_edges(state, G):
    cut_edges = []
    cut_value = 0
    for u, v, data in G.edges(data=True):
        if ((state >> u) & 1) != ((state >> v) & 1):
            cut_edges.append((u, v))
            cut_value += data.get("weight", 1.0)

    return float(cut_value), cut_edges


def compute_energy(theta_bits, G):
    # Reconstruct Ising energy (note: MAXCUT flips sign!)
    spins = {i: 1 if theta_bits[i] else -1 for i in range(len(theta_bits))}
    energy = sum(G[u][v].get("weight", 1) * spins[u] * spins[v] for u, v in G.edges())

    return energy

# Parallelization by Elara (OpenAI custom GPT):
def bootstrap_worker(args):
    theta, G, indices = args
    local_theta = theta.copy()
    flipped = []
    for i in indices:
        local_theta[i] = not local_theta[i]
        flipped.append(local_theta[i])
    energy = compute_energy(local_theta, G)

    return indices, energy, flipped

def spin_glass_solver(G, quality=2):
    cut_value, bitstring, cut_edges = maxcut_tfim(G, quality=0)
    best_theta = [ b == '1' for b in list(bitstring)]
    n_qubits = len(best_theta)
    min_energy = compute_energy(best_theta, G)
    improved = True
    while improved:
        improved = False
        for k in range(1, quality + 1):
            if n_qubits < k:
                break

            theta = best_theta.copy()
            args = []

            for combo in itertools.combinations(range(n_qubits), k):
                args.append((theta, G, combo))

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

    cut_value, cut_edges = evaluate_cut_edges(sample, G)

    return cut_value, bitstring, cut_edges, min_energy
