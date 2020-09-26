import pandas as pd
import networkx as nx
import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt

emails = pd.read_csv('./Emails.csv')
file = pd.read_csv('./Aliases.csv')
alises = {}
for index, row in file.iterrows():
    alises[row['Alias']] = row['PersonId']
file = pd.read_csv("./Persons.csv")
persons = {}
for index, row in file.iterrows():
    persons[row['Id']] = row['Name']

# 别名转换
def unify_name(name):
    name = str(name).lower().replace(',','').split('@')[0]
    if name in alises.keys():
        return persons[alises[name]]
    return name

# 画网络图
def show_graph(graph, type='spring_layout'):
    if type=='spring_layout':
        positions = nx.spring_layout(graph)    # 中心放射状
    if type=='circular_layout':
        positions = nx.circular_layout(graph)    # 圆环分布

    # 图中的节点大小
    nodesize = [x['pagerank']*20000 for v,x in graph.nodes(data=True)]
    edgesize = [np.sqrt(e[2]['weight']) for e in graph.edges(data=True)]
    nx.draw_networkx_nodes(graph, positions, node_size=nodesize, alpha=0.4)
    nx.draw_networkx_edges(graph, positions, alpha=0.2)
    nx.draw_networkx_labels(graph, positions, font_size=10)
    plt.show()

if __name__ == '__main__':
    emails.MetadataFrom = emails.MetadataFrom.apply(unify_name)
    emails.metaDataTo = emails.MetadataTo.apply(unify_name)
    edges_weights_temp = defaultdict(list)
    for row in zip(emails.MetadataFrom, emails.MetadataTo, emails.RawText):
        temp = (row[0], row[1])
        if temp not in edges_weights_temp:
            edges_weights_temp[temp] = 1
        else:
            edges_weights_temp[temp] += 1

    edges_weights = [(key[0], key[1], val) for key, val in edges_weights_temp.items()]
    # 创建有向图
    graph = nx.DiGraph()
    graph.add_weighted_edges_from(edges_weights)
    pagerank = nx.pagerank(graph)
    pagerank_list = {node: rank for node, rank in pagerank.items()}

    nx.set_node_attributes(graph, name='pagerank', values=pagerank)
    show_graph(graph)

    pagerank_threshold = 0.0015
    small_graph = graph.copy()
    for n, p_rank in graph.nodes(data=True):
        if p_rank['pagerank'] < pagerank_threshold:
            small_graph.remove_node(n)
    show_graph(small_graph, 'circular_layout')

