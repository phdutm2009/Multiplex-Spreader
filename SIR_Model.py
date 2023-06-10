import networkx as nx
import numpy as np
import pickle
beta = 0.1
landa = 0.1
repeati = 100
Top_L = 24
datasetName = 'fao_trade_multiplex'

inputDataset = np.loadtxt('./datasets/'+datasetName+'.txt')
inputDataset= np.delete(inputDataset, [-1], axis=1)
edgeList = {}

for i in inputDataset:
    edgeList.setdefault(i[0],[]).append((int(i[1]-1),int(i[2])-1))
edgeList = edgeList.values()
graph_list = []
for i in edgeList:
    G = nx.Graph()
    G.add_edges_from(i)
    graph_list.append(G)


tempNodes = []
for i in graph_list:
    tempNodes= tempNodes + list(i.nodes)
Nodes = list(set(tempNodes))
N = (len(Nodes))
rankDict = {}
for i in Nodes:
    rankDict[i] = 0


def SIR_Model_Node(nid, tgraph):
    labels_node = []
    nodes_label = {}
    for i in tgraph.nodes:
        nodes_label[i] = 's'
    seed = nid
    nodes_label[seed] = 'i'
    infected_node_set.add(seed)
    node_scale = 1
    n = 5
    while n >= 0:
        for x in nodes_label.keys():
            if x != -1:
                nbr = []
                if nodes_label[x] == 'i':
                    nbr = list(tgraph.neighbors(x))
                    for i in nbr:
                        if nodes_label[i] == 's':
                            rnd1 = np.random.random_sample()
                            if beta >= rnd1:
                                nodes_label[i] = 'i'
                                node_scale = node_scale + 1
                                infected_node_set.add(i)

            if nodes_label[x] == 'i':
                rnd = np.random.random_sample()
                if landa >= rnd:
                    nodes_label[x] = 'r'
        n = n - 1

    inode = node_scale
    return inode
sample = Top_L
while (repeati > 0):
    print('iteration: ', repeati)
    infected_list = []
    for i in Nodes:
        infected_node_set = set()
        for G in graph_list:
            if G.has_node(i):
                SIR_Model_Node(i, G)
        rankDict[i] = rankDict[i] + len(infected_node_set)
    repeati = repeati - 1
rankDict = sorted(rankDict.items(), key=lambda x:x[1],reverse=True)
final_list = []
for i in range(sample):
    final_list.append(rankDict[i][0])
with open('./results/'+datasetName+'_SIR_'+str(beta)+'.pickle', 'wb') as f:
    pickle.dump(final_list, f)
