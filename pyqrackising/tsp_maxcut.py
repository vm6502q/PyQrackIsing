from .tsp import tsp_symmetric

def tsp_to_maxcut_bipartition(tsp_path, weights):
    n = len(tsp_path)
    best_cut_value = -float('inf')
    best_partition = None
    direction = 0

    for offset in [-1, 0, 1]:
        mid = n // 2 + offset
        A = tsp_path[:mid]
        B = tsp_path[mid:]
        cut_value = sum(
            weights[u, v]
            for u in A
            for v in B
        )
        if cut_value > best_cut_value:
            best_cut_value = cut_value
            best_partition = (A, B)
            direction = offset

    if direction == 0:
        return best_partition, best_cut_value

    improved = True
    best_offset = direction
    while improved:
        improved = False
        offset = best_offset + direction
        mid = n // 2 + offset
        A = tsp_path[:mid]
        B = tsp_path[mid:]
        cut_value = sum(
            weights[u, v]
            for u in A
            for v in B
        )
        if cut_value > best_cut_value:
            best_cut_value = cut_value
            best_partition = (A, B)
            best_offset = offset
            improved = True

    return best_partition, best_cut_value

def tsp_maxcut(G, k_neighbors=20):
    path, length = tsp_symmetric(G=G, is_cyclic=False, monte_carlo=True, k_neighbors=k_neighbors)
    return tsp_to_maxcut_bipartition(path, G)
