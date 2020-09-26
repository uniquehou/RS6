import networkx as nx
import matplotlib.pyplot as plt

G = nx.DiGraph()
edges = [("A", "B"), ("A", "C"), ("A", "D"), ("B", "A"), ("B", "D"), ("C", "A"), ("D", "B"), ("D", "C")]
for edge in edges:
    G.add_edge(edge[0], edge[1])

layout = nx.spring_layout(G)
nx.draw(G, pos=layout, with_labels=True)
plt.show()

pr = nx.pagerank(G, alpha=1)
random_pr = nx.pagerank(G, alpha=0.8)