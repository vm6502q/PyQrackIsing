# MAXCUT
# Produced by Dan Strano, Elara (the OpenAI custom GPT), and Gemini (Google Search AI)

# We reduce transverse field Ising model for globally uniform J and h parameters from a 2^n-dimensional problem to an (n+1)-dimensional approximation that suffers from no Trotter error. Upon noticing most time steps for Quantinuum's parameters had roughly a quarter to a third (or thereabouts) of their marginal probability in |0> state, it became obvious that transition to and from |0> state should dominate the mechanics. Further, the first transition tends to be to or from any state with Hamming weight of 1 (in other words, 1 bit set to 1 and the rest reset 0, or n bits set for Hamming weight of n). Further, on a torus, probability of all states with Hamming weight of 1 tends to be exactly symmetric. Assuming approximate symmetry in every respective Hamming weight, the requirement for the overall probability to converge to 1.0 or 100% in the limit of an infinite-dimensional Hilbert space suggests that Hamming weight marginal probability could be distributed like a geometric series. A small correction to exact symmetry should be made to favor closeness of "like" bits to "like" bits (that is, geometric closeness on the torus of "1" bits to "1" bits and "0" bits to "0" bits), but this does not affect average global magnetization. Adding an oscillation component with angular frequency proportional to J, we find excellent agreement with Trotterization approaching the limit of infinitesimal time step, for R^2 (coefficient of determination) of normalized marginal probability distribution of ideal Trotterized simulation as described by the (n+1)-dimensional approximate model, as well as for R^2 and RMSE (root-mean-square error) of global magnetization curve values.

# After tackling the case where parameters are uniform and independent of time, we generalize the model by averaging per-qubit behavior as if the static case and per-time-step behavior as finite difference. This provides the basis of a novel physics-inspired (adiabatic TFIM) MAXCUT approximate solver that often gives optimal or exact answers on a wide selection of graph types.

from PyQrackIsing import maxcut_tfim
import networkx as nx
import numpy as np
import multiprocessing
import os


# NP-complete spin glass
def generate_spin_glass_graph(n_nodes=16, degree=3, seed=None):
    if not (seed is None):
        np.random.seed(seed)
    G = nx.random_regular_graph(d=degree, n=n_nodes, seed=seed)
    for u, v in G.edges():
        G[u][v]['weight'] = np.random.choice([-1, 1])  # spin glass couplings
    return G


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

def multiprocessing_bootstrap(G):
    cut_value, bitstring, cut_edges = maxcut_tfim(G)
    print((cut_value, bitstring, cut_edges))
    best_theta = np.array([1 if b == '1' else 0 for b in list(bitstring)])
    min_energy = compute_energy(best_theta, G)
    n_qubits = len(best_theta)
    iter_count = 0
    improved = True
    while improved:
        improved = False
        improved_1qb = True
        while improved_1qb:
            improved_1qb = False
            print(f"\nBootstrap Iteration {iter_count + 1}:")
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
                print(f"  Qubit {indices[0]} flip accepted. New energy: {min_energy}")
            else:
                print("  Qubit flips all rejected.")
            print(f"  {best_theta}")

            iter_count += 1

        if n_qubits < 2:
            break

        print(f"\nBootstrap Iteration {iter_count + 1}:")
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
            print(f"  Qubits {indices} flip accepted. New energy: {min_energy}")
        else:
            print("  Qubit flips all rejected.")
        print(f"  {best_theta}")

        iter_count += 1

    return best_theta, min_energy


if __name__ == "__main__":
    # NP-complete spin glass
    G = generate_spin_glass_graph(n_nodes=16, seed=42)

    # Run threaded bootstrap
    theta, min_energy = multiprocessing_bootstrap(G)

    print(f"\nFinal Bootstrap Ground State Energy: {min_energy}")
    print("Final Bootstrap Parameters:")
    print(theta)
