import networkx as nx
import numpy as np

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


def benchmark_maxcut(generator, n=64, seed=42, trials=10, **kwargs):
    # Generate random graph
    G = generator(n=n, **kwargs, seed=seed)

    gw = []
    qrack = []
    for t in range(trials):
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
            gw += cut_value

        # --- Qrack solver ---
        _, cut_value, partition, _ = spin_glass_solver(G)
        verified = evaluate_cut_value(G, partition)
        assert np.isclose(cut_value, verified)
        qrack.append(cut_value)

    return ratio_confidence_interval(qrack, gw)


if __name__ == "__main__":
    # Example runs
    if CVXPY_AVAILABLE:
        print("Qrack to Goemans-Williamson cut value ratio (99% CI):")
    else:
        print("Qrack to greedy local cut value ratio (99% CI):")
    ci = benchmark_maxcut(hard_instance_graph, d=10)
    print(f"Mean={ci[0]}")
    print(f"Range={ci[1]}")
