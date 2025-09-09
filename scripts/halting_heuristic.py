import networkx as nx
import numpy as np
import time

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

def test_halting(G):
    _, _, cut, _ = spin_glass_solver(G)
    print(cut)
    cut = (set(cut[0]), set(cut[1]))
    is_halting = True
    for i in range(3):
        _, _, n_cut, _ = spin_glass_solver(G, quality=3, correction_quality=3)
        print(n_cut)
        n_cut = (set(n_cut[0]), set(n_cut[1]))
        if cut[0] != n_cut[0] and cut[0] != n_cut[1]:
            is_halting = False
            break

    return is_halting

print("Graph that actually halts:")
halts = build_program_graph(halts=True)
if test_halting(halts):
    print("Probably halts.")
else:
    print("Probably does not halt.")

print()

print("Graph that actually doesn't halt:")
loops = build_program_graph(halts=False)
if test_halting(loops):
    print("Probably halts.")
else:
    print("Probably does not halt.")
