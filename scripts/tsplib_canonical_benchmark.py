# TSPLIB benchmark: PyQrackIsing vs. mathematically-proven optimal tours.
#
# Data source: the canonical TSPLIB symmetric-TSP mirror at
#   https://github.com/mastqe/tsplib
# which hosts the original .tsp instance files from Heidelberg's TSPLIB95
# (http://comopt.ifi.uni-heidelberg.de/software/TSPLIB95/) along with a
# `solutions` file of known-optimal tour lengths, all proven optimal via
# Concorde (see TSPLIB FAQ).
#
# Supports the two coordinate-based TSPLIB edge-weight types that cover the large
# majority of well-known instances:
#   - EUC_2D: standard Euclidean distance
#   - ATT:    the "pseudo-Euclidean" distance used by att48/att532, per the TSPLIB spec
# Instances using EDGE_WEIGHT_TYPE EXPLICIT (a pre-given distance matrix, no coordinates,
# e.g. bayg29, gr17) are deliberately skipped rather than mishandled -- silently guessing
# at those would risk exactly the kind of unflagged error this script exists to avoid.
#
# Script created by Anthropic Claude;
# PyQrackIsing is by Daniel Strano, with LLM assistance where and as credited.

import math
import os
import time
import urllib.request
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd
from pyqrackising import tsp_symmetric

TSPLIB_RAW_BASE = "https://raw.githubusercontent.com/mastqe/tsplib/master"
OUTPUT_CSV = os.environ.get("TSP_OUTPUT_CSV", "pyqrackising_tsplib_results.csv")

# A representative spread of well-known, coordinate-based (non-EXPLICIT) instances,
# small to large. Add/remove names freely -- run_instance() will skip anything that
# turns out to be EXPLICIT-format or otherwise unsupported, and say so explicitly.
DEFAULT_INSTANCES = [
    "burma14",
    "bayg29",  # EXPLICIT-format instance, included on purpose to show the skip path working
    "berlin52", "att48", "eil51", "eil76", "st70", "eil101", "ch130", "ch150",
    "att532", "a280",
]


def fetch_text(url):
    with urllib.request.urlopen(url, timeout=30) as resp:
        return resp.read().decode("utf-8")


def fetch_solutions():
    """Parse the canonical 'solutions' file into {instance_name: known_optimal_length}."""
    raw = fetch_text(f"{TSPLIB_RAW_BASE}/solutions")
    solutions = {}
    for line in raw.splitlines():
        line = line.strip()
        if not line or ":" not in line:
            continue
        name, value = line.split(":", 1)
        name = name.strip()
        value = value.strip().split()[0]  # drop trailing annotations like "(CEIL_2D)"
        try:
            solutions[name] = float(value)
        except ValueError:
            continue
    return solutions


def parse_tsp_file(raw_text):
    """
    Minimal TSPLIB .tsp parser for EUC_2D and ATT instances.
    Returns (edge_weight_type, {node_id: (x, y)}) or raises ValueError for
    unsupported formats (e.g. EXPLICIT) so callers can skip cleanly.
    """
    lines = raw_text.splitlines()
    edge_weight_type = None
    coords = {}
    in_coord_section = False

    for line in lines:
        stripped = line.strip()
        if stripped.startswith("EDGE_WEIGHT_TYPE"):
            edge_weight_type = stripped.split(":", 1)[1].strip()
        elif stripped.startswith("NODE_COORD_SECTION"):
            in_coord_section = True
            continue
        elif stripped.startswith("EOF"):
            break
        elif in_coord_section and stripped:
            parts = stripped.split()
            node_id = int(parts[0])
            x, y = float(parts[1]), float(parts[2])
            coords[node_id] = (x, y)

    if edge_weight_type not in ("EUC_2D", "ATT"):
        raise ValueError(
            f"Unsupported EDGE_WEIGHT_TYPE={edge_weight_type!r}; "
            f"this script only handles EUC_2D and ATT (coordinate-based) instances."
        )
    if not coords:
        raise ValueError("No NODE_COORD_SECTION found or it was empty.")

    return edge_weight_type, coords


def att_distance(p1, p2):
    """TSPLIB ATT pseudo-Euclidean distance (used by att48, att532, etc.)."""
    xd = p1[0] - p2[0]
    yd = p1[1] - p2[1]
    rij = math.sqrt((xd * xd + yd * yd) / 10.0)
    tij = round(rij)
    return tij + 1 if tij < rij else tij


def build_distance_matrix(edge_weight_type, coords):
    node_ids = sorted(coords.keys())
    n = len(node_ids)
    pts = np.array([coords[i] for i in node_ids], dtype=np.float64)

    if edge_weight_type == "EUC_2D":
        diff_x = pts[:, 0][:, None] - pts[:, 0][None, :]
        diff_y = pts[:, 1][:, None] - pts[:, 1][None, :]
        dist = np.sqrt(diff_x ** 2 + diff_y ** 2)
    elif edge_weight_type == "ATT":
        dist = np.zeros((n, n), dtype=np.float64)
        for i in range(n):
            for j in range(n):
                if i != j:
                    dist[i, j] = att_distance(pts[i], pts[j])
    else:
        raise ValueError(f"build_distance_matrix: unhandled type {edge_weight_type!r}")

    return dist


def nearest_neighbor_tour(dist_matrix):
    """Cheap, standard baseline: greedy nearest-neighbor heuristic, starting at city 0."""
    n = dist_matrix.shape[0]
    unvisited = set(range(1, n))
    tour = [0]
    current = 0
    while unvisited:
        nxt = min(unvisited, key=lambda j: dist_matrix[current, j])
        tour.append(nxt)
        unvisited.remove(nxt)
        current = nxt
    length = sum(dist_matrix[tour[i], tour[(i + 1) % n]] for i in range(n))
    return tour, length


def _single_solve(dist_matrix):
    """One independent top-level solve. Module-level so it's picklable for ProcessPoolExecutor."""
    return tsp_symmetric(dist_matrix, monte_carlo=True, is_cyclic=True)


def best_of_n_solve(dist_matrix, n_runs=8, n_workers=None):
    """
    Run tsp_symmetric n_runs independent times and keep the best result.
    This is just calling the function repeatedly at the top level and taking the
    minimum -- tsp_symmetric has real run-to-run stochasticity (confirmed: a few
    percent spread across repeated calls on the same input), so this is a genuine,
    non-placebo lever, not a no-op.

    Runs in parallel across n_workers processes when more than one CPU is available
    (defaults to os.cpu_count()).
    """
    if n_workers is None:
        n_workers = os.cpu_count() or 1
    n_workers = max(1, min(n_workers, n_runs))

    best_path, best_weight = None, float("inf")

    if n_workers == 1:
        for _ in range(n_runs):
            path, weight = _single_solve(dist_matrix)
            if weight < best_weight:
                best_path, best_weight = path, weight
    else:
        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            futures = [
                executor.submit(_single_solve, dist_matrix)
                for _ in range(n_runs)
            ]
            for f in futures:
                path, weight = f.result()
                if weight < best_weight:
                    best_path, best_weight = path, weight

    return best_path, best_weight


def run_instance(name, known_solutions, n_runs=8, n_workers=None):
    raw = fetch_text(f"{TSPLIB_RAW_BASE}/{name}.tsp")
    edge_weight_type, coords = parse_tsp_file(raw)  # raises ValueError -> caller skips
    n = len(coords)
    dist_matrix = build_distance_matrix(edge_weight_type, coords)

    known_best = known_solutions.get(name)
    if known_best is None:
        raise ValueError(f"No known-optimal length found in solutions file for {name!r}.")

    t0 = time.perf_counter()
    path, weight = best_of_n_solve(
        dist_matrix, n_runs=n_runs, n_workers=n_workers
    )
    elapsed = time.perf_counter() - t0

    _, nn_length = nearest_neighbor_tour(dist_matrix)

    ratio = weight / known_best if known_best > 0 else float("nan")
    nn_ratio = nn_length / known_best if known_best > 0 else float("nan")

    return {
        "instance": name,
        "num_cities": n,
        "edge_weight_type": edge_weight_type,
        "known_best_length": known_best,
        "pyqrackising_length": weight,
        "ratio_to_known_best": ratio,  # 1.0 = matched the proven optimum
        "nearest_neighbor_length": nn_length,
        "nearest_neighbor_ratio_to_known_best": nn_ratio,  # cheap baseline, for context only
        "seconds": elapsed,
        "path": path,
    }


def main(instance_names=None, n_runs=8, n_workers=None):
    if instance_names is None:
        instance_names = DEFAULT_INSTANCES

    known_solutions = fetch_solutions()

    results = []
    for name in instance_names:
        try:
            res = run_instance(
                name, known_solutions, n_runs=n_runs, n_workers=n_workers,
            )
            print(
                f"{res['instance']:>10s}  n={res['num_cities']:<5d}  "
                f"type={res['edge_weight_type']:<6s}  "
                f"known={res['known_best_length']:.1f}  "
                f"pyqrackising={res['pyqrackising_length']:.1f} (ratio={res['ratio_to_known_best']:.4f})  "
                f"nearest_neighbor={res['nearest_neighbor_length']:.1f} (ratio={res['nearest_neighbor_ratio_to_known_best']:.4f})  "
                f"time={res['seconds']:.2f}s"
            )
            results.append(res)
        except ValueError as e:
            print(f"{name:>10s}  SKIPPED: {e}")
        except Exception as e:
            print(f"{name:>10s}  FAILED: {e}")

    out = pd.DataFrame(results)
    out.to_csv(OUTPUT_CSV, index=False)

    if len(out):
        print("\n--- summary ---")
        print(f"instances run:                  {len(out)}")
        print(f"PyQrackIsing mean ratio:         {out['ratio_to_known_best'].mean():.4f}")
        print(f"PyQrackIsing median ratio:       {out['ratio_to_known_best'].median():.4f}")
        print(f"PyQrackIsing worst ratio:        {out['ratio_to_known_best'].max():.4f}")
        print(f"Nearest-neighbor mean ratio:     {out['nearest_neighbor_ratio_to_known_best'].mean():.4f}  (cheap baseline, for context)")
        print(f"Total wall time (s):             {out['seconds'].sum():.2f}")
    print(f"\nFull results written to {OUTPUT_CSV}")
    return out


if __name__ == "__main__":
    main(n_runs=os.cpu_count())
