import networkx as nx
import numpy as np
import time
import pandas as pd

from networkx.algorithms.approximation import maxcut as nx_maxcut

from PyQrackIsing import spin_glass_solver

# Try to import cvxpy for GW SDP implementation
try:
    import cvxpy as cp

    CVXPY_AVAILABLE = True
except ImportError:
    CVXPY_AVAILABLE = False


# --- 1. Erdős–Rényi ---
def erdos_renyi_graph(n=128, p=0.5, seed=None):
    return nx.erdos_renyi_graph(n, p, seed=seed)


# --- 2. Planted-partition ---
def planted_partition_graph(n=128, p_in=0.2, p_out=0.8, seed=None):
    # Split into 2 equal communities
    sizes = [n // 2, n - n // 2]
    probs = [[p_in, p_out], [p_out, p_in]]
    return nx.stochastic_block_model(sizes, probs, seed=seed)


# --- 3. Hard instances (regular bipartite expander as example) ---
def hard_instance_graph(n=128, d=10, seed=None):
    # d-regular bipartite graph
    left = list(range(n // 2))
    right = list(range(n // 2, n))
    return nx.random_regular_graph(d, n, seed=seed)


def evaluate_cut_value(G, partition):
    """Compute cut value directly from graph and bitstring."""
    cut = 0
    for u, v, data in G.edges(data=True):
        w = data.get("weight", 1.0)
        if (u in partition[0]) == (v in partition[1]):  # different sides of partition
            cut += w
    return cut


def safe_cholesky(X, eps=1e-8):
    # Symmetrize just in case of numerical asymmetry
    X = 0.5 * (X + X.T)

    # Eigen-decomposition
    eigvals, eigvecs = np.linalg.eigh(X)

    # Clip tiny negatives to zero
    eigvals = np.clip(eigvals, 0, None)

    # Reconstruct PSD matrix
    X_psd = eigvecs @ np.diag(eigvals) @ eigvecs.T

    # Now do Cholesky on the repaired PSD matrix
    return np.linalg.cholesky(X_psd + eps * np.eye(X.shape[0]))


def gw_sdp_maxcut(G):
    """
    Goemans-Williamson SDP relaxation for MaxCut, solved via CVXPY.
    Returns cut value and partition (approximate).
    """
    n = len(G.nodes)
    W = nx.to_numpy_array(G, weight="weight", nonedge=0.0)

    # Define SDP variable
    X = cp.Variable((n, n), symmetric=True)
    constraints = [X >> 0, cp.diag(X) == 1]

    # Objective: maximize 1/4 sum_ij W_ij (1 - X_ij)
    obj = cp.Maximize(0.25 * cp.sum(cp.multiply(W, (1 - X))))
    prob = cp.Problem(obj, constraints)
    prob.solve(solver=cp.SCS, verbose=False)

    # Extract randomized rounding solution
    U = safe_cholesky(X.value)
    r = np.random.randn(U.shape[1])
    x = np.sign(U @ r)

    cut_value = 0.0
    for i, j in G.edges():
        if x[i] != x[j]:
            cut_value += G[i][j].get("weight", 1.0)

    partition = (set(np.where(x > 0)[0]), set(np.where(x <= 0)[0]))
    return cut_value, partition


def benchmark_maxcut(generators, sizes=[64, 128, 256], seed=42, trials=5):
    results = {}
    for n in sizes:
        results[n] = {}
        for key, value in generators.items():
            results[n][key] = []
            for t in range(trials):
                # Generate random graph
                G = value[0](n=n, **(value[1]), seed=seed + t)

                results_dict = {}

                # --- GW SDP (if available) ---
                if CVXPY_AVAILABLE:
                    start = time.time()
                    cut_value, partition = gw_sdp_maxcut(G)
                    verified = evaluate_cut_value(G, partition)
                    assert np.isclose(cut_value, verified)
                    results_dict["GW_SDP"] = (cut_value, time.time() - start)
                else:
                    # --- Greedy local improvement ---
                    start = time.time()
                    cut_value, partition = nx_maxcut.one_exchange(G)
                    verified = evaluate_cut_value(G, partition)
                    assert np.isclose(cut_value, verified)
                    results["Greedy"] = (cut_value, time.time() - start)

                # --- Qrack solver (placeholder) ---
                # Replace with your function call locally
                start = time.time()
                _, cut_value, partition, _ = spin_glass_solver(G)
                verified = evaluate_cut_value(G, partition)
                assert np.isclose(cut_value, verified)
                results_dict["Qrack"] = (cut_value, time.time() - start)

                results[n][key].append(results_dict)

    return results


if __name__ == "__main__":
    # Example runs
    print(
        benchmark_maxcut(
            {
                "Erdős–Rényi": (erdos_renyi_graph, {"p": 0.5}),
                "Planted-partition": (planted_partition_graph, {"p_in": 0.2, "p_out": 0.8}),
                "Hard (bipartite expander)": (hard_instance_graph, {"d": 10}),
            }
        )
    )
