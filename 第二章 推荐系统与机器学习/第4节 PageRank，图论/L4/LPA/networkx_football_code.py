import networkx as nx
from networkx.algorithms import community
import matplotlib.pyplot as plt

G = nx.read_gml('football.gml')
nx.draw(G, with_labels=True)
plt.show()
communities = list(community.label_propagation_communities(G))
print(communities)
print(len(communities))
