from pyqrackising import solve_maxcut_bnb, solve_maxcut_bnb_sparse, solve_maxcut_bnb_streaming
import networkx as nx

G = nx.petersen_graph()
print(solve_maxcut_bnb(G))
