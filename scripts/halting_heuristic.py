import math
import networkx as nx
import numpy as np
import time

from collections import Counter

from PyQrackIsing import spin_glass_solver

def build_program_graph(halts=True, size=7):
    """
    Encode a toy program into a graph:
    - If halts=True, create a finite chain with a sink (halts).
    - If halts=False, create a cycle (simulating infinite loop).
    """
    G = nx.Graph()
    nodes = list(range(size))
    for i in range(size - 1):
        G.add_edge(nodes[i], nodes[i + 1], weight=1)
    if halts:
        # add terminal sink node
        G.add_node("HALT")
        G.add_edge(nodes[-1], "HALT", weight=2)
    else:
        # close cycle to simulate infinite loop
        G.add_edge(nodes[-1], nodes[0], weight=2)
    return G

def halting_entropy(G, trials=10):
    cuts = []
    for i in range(trials):
        _, _, cut, _ = spin_glass_solver(G, quality=3, correction_quality=3)
        cut = (frozenset(cut[0]), frozenset(cut[1])) if 0 in cut[0] else (frozenset(cut[1]), frozenset(cut[0]))
        cuts.append(cut)

    counts = Counter(cuts)
    probs = [c / trials for c in counts.values()]
    entropy = -sum(p * math.log2(p) for p in probs)

    return entropy, counts

print("Graph that actually halts:")
halts = build_program_graph(halts=True)
entropy, counts = halting_entropy(halts)
print(f"Entropy: {entropy}")
print(f"Counts: {counts}")
if np.isclose(entropy, 0):
    print("Probably halts.")
else:
    print("Probably does not halt.")

print()

print("Graph that actually doesn't halt:")
loops = build_program_graph(halts=False)
entropy, counts = halting_entropy(loops)
print(f"Entropy: {entropy}")
print(f"Counts: {counts}")
if np.isclose(entropy, 0):
    print("Probably halts.")
else:
    print("Probably does not halt.")
