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

def benchmark_maxcut(n=50, p=0.3, seed=None, quality=None):
    # Generate random graph
    G = nx.erdos_renyi_graph(n, p, seed=seed)
    
    results = {}
    
    # --- Greedy local improvement ---
    start = time.time()
    cut_value, partition = nx_maxcut.one_exchange(G)
    results["Greedy"] = (cut_value, time.time() - start)
    
    # --- GW SDP (if available) ---
    if CVXPY_AVAILABLE:
        start = time.time()
        cut_value, partition = gw_sdp_maxcut(G)
        results["GW_SDP"] = (cut_value, time.time() - start)
    
    # --- Qrack solver (placeholder) ---
    # Replace with your function call locally
    start = time.time()
    bitstring, cut_value, _, _ = spin_glass_solver(G, quality=quality)
    results["Qrack"] = (cut_value, time.time() - start)
    
    return results

# Example run (small n for sanity check, if cvxpy is present)
print(benchmark_maxcut(n=50, p=0.5, seed=42))

