# Provided by Elara, the custom OpenAI GPT
from collections import defaultdict
import itertools
import numpy as np


def convert_tensor_network_to_tsp(tensors, index_dims, dtype=np.float32):
    """
    Converts a tensor network to a symmetric TSP distance matrix based on pairwise contraction costs.

    Args:
        tensors: List of dictionaries, each with keys:
                 - 'id': Unique tensor identifier (str)
                 - 'indices': Set of indices (Set[str])
        index_dims: Dictionary mapping index name to its dimension (e.g., {'i': 2, 'j': 4})

    Returns:
        A tuple containing:
        - A normalized 2D numpy array representing the symmetric TSP distance matrix.
        - A list of tensor IDs corresponding to the node order in the distance matrix.
    """
    # Step 1: Build inverse map: index -> tensors using it
    index_to_tensors = defaultdict(set)
    for tensor in tensors:
        for idx in tensor["indices"]:
            index_to_tensors[idx].add(tensor["id"])

    # Step 2: Build graph of neighboring tensors
    graph = defaultdict(set)
    for idx, tensor_ids in index_to_tensors.items():
        for t1, t2 in itertools.combinations(tensor_ids, 2):
            graph[t1].add(t2)
            graph[t2].add(t1)

    # Step 3: Estimate pairwise contraction costs
    def estimate_cost(t1, t2):
        common = t1["indices"] & t2["indices"]
        if not common:
            return float("inf")
        all_indices = t1["indices"] | t2["indices"]
        return sum(index_dims.get(i, 1) for i in all_indices)

    max_cost = float("-inf")
    edge_costs = {}
    for t1, neighbors in graph.items():
        for t2 in neighbors:
            key = tuple(sorted([t1, t2]))
            if key not in edge_costs:
                t1_data = next(t for t in tensors if t["id"] == t1)
                t2_data = next(t for t in tensors if t["id"] == t2)
                cost = estimate_cost(t1_data, t2_data)
                edge_costs[key] = cost
                if (cost < float("inf")) and (cost > max_cost):
                    max_cost = cost

    # Step 4: Build distance matrix
    tensor_ids = [t["id"] for t in tensors]
    id_to_node = {tid: i for i, tid in enumerate(tensor_ids)}
    n = len(tensor_ids)
    dist_matrix = np.full((n, n), np.sqrt(np.finfo(dtype).max), dtype=dtype)

    for (t1, t2), cost in edge_costs.items():
        if cost == float("inf"):
            continue
        i, j = id_to_node[t1], id_to_node[t2]
        dist_matrix[i, j] = dist_matrix[j, i] = cost / max_cost

    return dist_matrix, tensor_ids



def convert_quimb_tree_to_tsp(tn, dtype=np.float32):
    """
    Converts a quimb TensorNetwork to a TSP distance matrix for contraction optimization.

    Args:
        tn: A quimb.tensor.TensorNetwork instance.

    Returns:
        - Normalized symmetric distance matrix (2D numpy array of floats)
        - List of tensor IDs (tags) corresponding to the matrix order
    """
    tensors = []
    index_dims = {}

    for tensor in tn.tensors:
        # Use a unique tag or fallback to name
        tensor_id = frozenset(tensor.tags) if tensor.tags else str(tensor.name)
        tensors.append({
            "id": tensor_id,
            "indices": set(tensor.inds)
        })

        for ind, dim in zip(tensor.inds, tensor.shape):
            index_dims[ind] = dim  # capture dimension size

    return convert_tensor_network_to_tsp(tensors, index_dims, dtype)
