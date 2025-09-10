import networkx as nx
import numpy as np
import time

from networkx.algorithms.approximation import maxcut as nx_maxcut

from pyqrackising import spin_glass_solver

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


# --- 4. Canonical worst-case (Khot-Vishnoi, UGC)
def khot_vishnoi_graph(n, epsilon=0.1, seed=None):
    """
    Generate a Khot-Vishnoi style hard instance for MAXCUT (approximation hardness for GW).

    Parameters:
        n (int): number of vertices, should be a power of 2 (since construction uses hypercube structure).
        epsilon (float): bias parameter controlling edge weights, typically small (e.g., 0.1).
        seed (int or None): random seed for reproducibility.

    Returns:
        G (networkx.Graph): weighted graph instance.
    """
    if seed is not None:
        np.random.seed(seed)

    # Ensure n is a power of 2 for hypercube embedding
    if not (n and ((n & (n - 1)) == 0)):
        raise ValueError("n must be a power of 2 for Khot-Vishnoi construction")

    # Vertices correspond to n-bit vectors in {0,1}^k with k = log2(n)
    k = int(np.log2(n))
    G = nx.Graph()
    G.add_nodes_from(range(n))

    # Construct adjacency: edges weighted by noise-perturbed inner product
    for i in range(n):
        for j in range(i + 1, n):
            # Hamming distance between binary representations
            vi = np.array(list(map(int, np.binary_repr(i, width=k))))
            vj = np.array(list(map(int, np.binary_repr(j, width=k))))
            hamming = np.sum(vi != vj)

            # Weight depends on epsilon and parity of distance
            weight = (1 - 2 * epsilon) ** hamming
            if weight > 0:
                G.add_edge(i, j, weight=weight)

    return G


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

    # Solve with CVXOPT (higher accuracy than SCS)
    prob.solve(solver=cp.CVXOPT, verbose=False)

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


# By Elara (the custom OpenAI GPT):
def ratio_confidence_interval(qrack_vals, gw_vals, confidence=0.99):
    import scipy.stats as st
    import numpy as np

    nQ, nG = len(qrack_vals), len(gw_vals)
    mean_Q, mean_G = np.mean(qrack_vals), np.mean(gw_vals)
    se_Q, se_G = st.sem(qrack_vals), st.sem(gw_vals)

    R = mean_Q / mean_G
    se_R = R * np.sqrt((se_Q / mean_Q)**2 + (se_G / mean_G)**2)

    df = min(nQ-1, nG-1)  # conservative choice
    t_val = st.t.ppf((1 + confidence) / 2., df)

    ci_R = (R - t_val * se_R, R + t_val * se_R)
    return R, ci_R


def benchmark_maxcut(generator, n=64, seed=None, trials=10, **kwargs):
    # Generate random graph
    G = generator(n=n, **kwargs, seed=seed)

    gw = []
    qrack = []
    gw_time = 0
    qrack_time = 0
    for t in range(trials):
        start = time.perf_counter()
        if CVXPY_AVAILABLE:
            # --- GW SDP (if available) ---
            cut_value, partition = gw_sdp_maxcut(G)
            verified = evaluate_cut_value(G, partition)
            assert np.isclose(cut_value, verified)
            gw.append(cut_value)
        else:
            # --- Greedy local improvement ---
            cut_value, partition = nx_maxcut.one_exchange(G)
            verified = evaluate_cut_value(G, partition)
            assert np.isclose(cut_value, verified)
            gw.append(cut_value)
        gw_time += time.perf_counter() - start

        # --- Qrack solver ---
        start = time.perf_counter()
        _, cut_value, partition, _ = spin_glass_solver(G)
        verified = evaluate_cut_value(G, partition)
        assert np.isclose(cut_value, verified)
        qrack.append(cut_value)
        qrack_time += time.perf_counter() - start

    gw_time /= trials
    qrack_time /= trials

    return ratio_confidence_interval(qrack, gw), qrack_time, gw_time, max(qrack), max(gw)


if __name__ == "__main__":
    # Example runs
    if CVXPY_AVAILABLE:
        print("Qrack to Goemans-Williamson cut value ratio (99% CI):")
    else:
        print("Qrack to greedy local cut value ratio (99% CI):")
    ci, qrack_time, gw_time, best_qrack, best_gw = benchmark_maxcut(hard_instance_graph, d=10)
    print(f"Mean={ci[0]}")
    print(f"Range={ci[1]}")
    best_ratio = best_qrack / best_gw
    if CVXPY_AVAILABLE:
        print(f"Qrack to Goemans-Williamson best cut value ratio: {best_ratio}")
    else:
        print(f"Qrack to greedy local cut value ratio: {best_ratio}")
    print(f"Qrack average seconds per trial: {qrack_time}")
    print(f"Goemans-Williamson average seconds per trial: {gw_time}")
    print(f"Best possible GW approximation ratio: {1.0 / best_ratio}")
    print(f"Best possible GW approximation ratio if P=/=NP: {0.941 / best_ratio}")
