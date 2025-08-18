from .maxcut_tfim import maxcut_tfim
import multiprocessing
import os


def evaluate_cut_edges(state, flat_edges):
    cut_edges = []
    for i in range(len(flat_edges) // 2):
        i2 = i << 1
        u, v = flat_edges[i2], flat_edges[i2 + 1]
        if ((state >> u) & 1) != ((state >> v) & 1):
            cut_edges.append((u, v))
    cut_size = len(cut_edges)

    return cut_size, cut_edges


def compute_energy(theta_bits, G):
    # Reconstruct Ising energy (note: MAXCUT flips sign!)
    spins = {i: 1 if theta_bits[i] else -1 for i in range(len(theta_bits))}
    energy = -sum(G[u][v].get("weight", 1) * spins[u] * spins[v] for u, v in G.edges())

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

def spin_glass_solver(G):
    cut_value, bitstring, cut_edges = maxcut_tfim(G, quality=0)
    best_theta = [ b == '1' for b in list(bitstring)]
    min_energy = compute_energy(best_theta, G)
    n_qubits = len(best_theta)
    iter_count = 0
    improved = True
    while improved:
        improved = False
        improved_1qb = True
        while improved_1qb:
            improved_1qb = False
            theta = best_theta.copy()

            with multiprocessing.Pool(processes=os.cpu_count()) as pool:
                args = []
                for i in range(n_qubits):
                    args.append((theta, G, (i,)))
                results = pool.map(bootstrap_worker, args)

            results.sort(key=lambda r: r[1])
            indices, energy, flipped = results[0]
            if energy < min_energy:
                min_energy = energy
                for i in range(len(indices)):
                    best_theta[indices[i]] = flipped[i]
                improved_1qb = True

            iter_count += 1

        if n_qubits < 2:
            break

        theta = best_theta.copy()

        with multiprocessing.Pool(processes=os.cpu_count()) as pool:
            args = []
            for i in range(n_qubits):
                for j in range(i + 1, n_qubits):
                    args.append((theta, G, (i, j)))
            results = pool.map(bootstrap_worker, args)

        results.sort(key=lambda r: r[1])
        indices, energy, flipped = results[0]
        if energy < min_energy:
            min_energy = energy
            for i in range(len(indices)):
                best_theta[indices[i]] = flipped[i]
            improved = True

        iter_count += 1

    sample = 0
    bitstring = ""
    for i in range(len(best_theta)):
        b = best_theta[i]
        bitstring += '1' if b else '0'
        if b:
            sample |= 1 << i

    cut_value, cut_edges = evaluate_cut_edges(sample, [int(item) for tup in G.edges() for item in tup])

    return cut_value, bitstring, cut_edges, min_energy
