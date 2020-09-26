import networkx as nx
import matplotlib.pyplot as plt

G = nx.read_gml('football.gml')
nx.draw(G, with_labels=True)
plt.show()
path_1 = nx.shortest_path(G, source='Buffalo', target='Kent')
path_2 = nx.shortest_path(G, source='Buffalo', target='Rice')

# Dijkstra
path_3 = nx.single_source_dijkstra_path(G, source='Buffalo')
path_4 = nx.multi_source_dijkstra_path(G, {'Buffalo', 'Rice'})
# flody
path_5 = nx.floyd_warshall(G, weight='weight')