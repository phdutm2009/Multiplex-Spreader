import networkx as nx
import numpy as np
import pickle
#--------------------- Read dataset ----------------------------
SIR_beta = 0.1 # Beta parameter for examined SIR
Top_L = 24
datasetName = 'fao_trade_multiplex' #name of dataset

inputDataset = np.loadtxt('./datasets/'+datasetName+'.txt')
inputDataset= np.delete(inputDataset, [-1], axis=1)
edgeList = {}
for i in inputDataset:
    edgeList.setdefault(i[0],[]).append((int(i[1])-1,int(i[2])-1))
edgeList = edgeList.values()
graph_list = []
for i in edgeList:
    G = nx.Graph()
    G.add_edges_from(i)
    graph_list.append(G)
# --------------------Number of Active nodes and edges (NAN , NAE)--------------------------
layer_NAN_NAE_ANCPC_NIICCL_Influence = {} # normalized active nodes of each layer
nodes_sum = 0
layerNumber = 0
nodes_degree_sum = 0
degrees_sum = 0
for i in graph_list:
    b_i = len(i.nodes) # number of active nodes
    nodes_sum = nodes_sum + b_i
    nodes_degree_sum = sum(list(map(lambda x: x[0], i.degree(i.nodes)))) # number of active edges
    degrees_sum = degrees_sum + nodes_degree_sum
    layer_NAN_NAE_ANCPC_NIICCL_Influence[layerNumber] = [b_i,nodes_degree_sum]
    layerNumber += 1
temp = 0
for i in layer_NAN_NAE_ANCPC_NIICCL_Influence.keys(): # normalized active nodes and active edges
    W = layer_NAN_NAE_ANCPC_NIICCL_Influence[i][1]/degrees_sum
    layer_NAN_NAE_ANCPC_NIICCL_Influence[i].append([layer_NAN_NAE_ANCPC_NIICCL_Influence[i][0] / nodes_sum,W])
    temp = temp + W
for i in layer_NAN_NAE_ANCPC_NIICCL_Influence.keys():
    layer_NAN_NAE_ANCPC_NIICCL_Influence[i][2][1]= layer_NAN_NAE_ANCPC_NIICCL_Influence[i][2][1] / temp

# --------------- Ratio of Active nodes and their connection (ANCPC) -------------------
tempNodes = []
for i in graph_list:
    tempNodes= tempNodes + list(i.nodes)
Nodes = list(set(tempNodes))
N = (len(Nodes))

for i in layer_NAN_NAE_ANCPC_NIICCL_Influence.keys():
    ANCPC = (2*(layer_NAN_NAE_ANCPC_NIICCL_Influence[i][0]+layer_NAN_NAE_ANCPC_NIICCL_Influence[i][1])) / (N*(N-1))
    layer_NAN_NAE_ANCPC_NIICCL_Influence[i].append(ANCPC)
# -------------------  Intersection of layers (NIICCL) ------------------------------
total_sum = 0
for i in range(len(graph_list)):
    tempInstersection = 0
    for j in range(len(graph_list)):
        if i!=j:
            tempInstersection =tempInstersection + len(set(graph_list[i].edges()).intersection(graph_list[j].edges()))
    layer_NAN_NAE_ANCPC_NIICCL_Influence[i].append(tempInstersection)
    total_sum = total_sum + tempInstersection

# Normalized intersection
for i in layer_NAN_NAE_ANCPC_NIICCL_Influence.keys():
    layer_NAN_NAE_ANCPC_NIICCL_Influence[i][4] = layer_NAN_NAE_ANCPC_NIICCL_Influence[i][4] / total_sum
#--------------------------Centrality vector of nodes-----------------------------
ClosenessVector = {}
BetweennessVector = {}
PageRankVector = {}
EigenVector = {}
KatsVector = {}
HarmonicVector = {}
M_CentralityVector = {}

layerCounter = 0
for G in graph_list:
    tempCentrality = {}
    tempCentrality= nx.closeness_centrality(G) # Closeness centrality
    for i in tempCentrality.keys():
        ClosenessVector.setdefault(i, []).append((layerCounter,tempCentrality[i]))

    tempCentrality = {}
    tempCentrality = nx.betweenness_centrality(G)  # Betweenness centrality
    for i in tempCentrality.keys():
        BetweennessVector.setdefault(i, []).append((layerCounter,tempCentrality[i]))
    tempCentrality = {}
    tempCentrality = nx.pagerank(G, alpha=0.7)  # PageRank centrality
    for i in tempCentrality.keys():
        PageRankVector.setdefault(i, []).append((layerCounter,tempCentrality[i]))
    tempCentrality = {}
    tempCentrality = nx.eigenvector_centrality_numpy(G) # EigenVector centrality
    for i in tempCentrality.keys():
        EigenVector.setdefault(i, []).append((layerCounter,tempCentrality[i]))
    tempCentrality = {}
    tempCentrality = nx.katz_centrality_numpy(G) # Kats centrality
    for i in tempCentrality.keys():
        KatsVector.setdefault(i, []).append((layerCounter,tempCentrality[i]))

    tempCentrality = {}
    tempCentrality = nx.harmonic_centrality(G) # harmonic centrality
    for i in tempCentrality.keys():
        HarmonicVector.setdefault(i, []).append((layerCounter,tempCentrality[i]))

    tempCentrality = {} #M-Centrality
    G.remove_edges_from(nx.selfloop_edges(G))
    k_shell = nx.core_number(G)
    for node in G.nodes:
        neighbors = []
        neighbors = G.neighbors(node)
        neighborsDegreeSum = 0
        degreeVariation = 0
        for j in neighbors:
            neighborsDegreeSum += G.degree(j)

        for j in neighbors:
            degreeVariation += G.degree(node) * ((abs(G.degree(node) - G.degree(j))) / neighborsDegreeSum)
        tempCentrality[node] = (0.25*k_shell[node]) + ((1-0.25)*degreeVariation)
    for i in tempCentrality.keys():
        M_CentralityVector.setdefault(i, []).append((layerCounter,tempCentrality[i]))

    layerCounter += 1
#------------------------- -Layer influence ------------------------------
for i in layer_NAN_NAE_ANCPC_NIICCL_Influence.keys():
    Mi = layer_NAN_NAE_ANCPC_NIICCL_Influence[i][2][1] + layer_NAN_NAE_ANCPC_NIICCL_Influence[i][3]
    layer_NAN_NAE_ANCPC_NIICCL_Influence[i].append(Mi)
nodesInfluence_Closeness = {}
nodesInfluence_Betweenness = {}
nodesInfluence_PageRank = {}
nodesInfluence_EigenVector = {}
nodesInfluence_Kats = {}
nodesInfluence_Harmonic = {}
nodesInfluence_MCentrality = {}
nodesInfluence_ClosenessMean = {}
nodesInfluence_ClosenessVariance = {}
nodesInfluence_BetweennessMean = {}
nodesInfluence_BetweennessVariance = {}
nodesInfluence_PageRankMean = {}
nodesInfluence_PageRankVariance = {}
nodesInfluence_EigenVectorMean = {}
nodesInfluence_EigenVectorVariance = {}
nodesInfluence_KatsMean = {}
nodesInfluence_KatsVariance = {}
nodesInfluence_HarmonicMean = {}
nodesInfluence_HarmonicVariance = {}
nodesInfluence_MCentralityMean = {}
nodesInfluence_MCentralityVariance = {}

for i in Nodes:

    tempSum = 0
    tempList = []
    for j in ClosenessVector[i]: # Closencess centrality rank
        tempSum = tempSum + (j[1] * layer_NAN_NAE_ANCPC_NIICCL_Influence[j[0]][5])
        tempList.append(j[1])
    nodesInfluence_Closeness[i] = tempSum
    nodesInfluence_ClosenessMean[i] = sum(tempList) / len(tempList)
    nodesInfluence_ClosenessVariance[i] = (np.var(tempList))

    tempSum = 0
    tempList = []
    for j in BetweennessVector[i]: # Betweenness centrality rank
        tempSum = tempSum + (j[1] * layer_NAN_NAE_ANCPC_NIICCL_Influence[j[0]][5])
        tempList.append(j[1])
    nodesInfluence_Betweenness[i] = tempSum
    nodesInfluence_BetweennessMean[i] = sum(tempList) / len(tempList)
    nodesInfluence_BetweennessVariance[i] = (np.var(tempList))

    tempSum = 0
    tempList = []
    for j in PageRankVector[i]: # PageRankVector centrality rank
        tempSum = tempSum + (j[1] * layer_NAN_NAE_ANCPC_NIICCL_Influence[j[0]][5])
        tempList.append(j[1])
    nodesInfluence_PageRank[i] = tempSum
    nodesInfluence_PageRankMean[i] = sum(tempList) / len(tempList)
    nodesInfluence_PageRankVariance[i] = (np.var(tempList))

    tempSum = 0
    tempList = []
    for j in EigenVector[i]: # EigenVector centrality rank
        tempSum = tempSum + (j[1] * layer_NAN_NAE_ANCPC_NIICCL_Influence[j[0]][5])
        tempList.append(j[1])
    nodesInfluence_EigenVector[i] = tempSum
    nodesInfluence_EigenVectorMean[i] = sum(tempList) / len(tempList)
    nodesInfluence_EigenVectorVariance[i] = (np.var(tempList))

    tempSum = 0
    tempList = []
    for j in KatsVector[i]: # KatsVector centrality rank
        tempSum = tempSum + (j[1] * layer_NAN_NAE_ANCPC_NIICCL_Influence[j[0]][5])
        tempList.append(j[1])
    nodesInfluence_Kats[i] = tempSum
    nodesInfluence_KatsMean[i] = sum(tempList) / len(tempList)
    nodesInfluence_KatsVariance[i] = (np.var(tempList))

    tempSum = 0
    tempList = []
    for j in HarmonicVector[i]: # HarmonicVector centrality rank
        tempSum = tempSum + (j[1] * layer_NAN_NAE_ANCPC_NIICCL_Influence[j[0]][5])
        tempList.append(j[1])
    nodesInfluence_Harmonic[i] = tempSum
    nodesInfluence_HarmonicMean[i] = sum(tempList) / len(tempList)
    nodesInfluence_HarmonicVariance[i] = (np.var(tempList))

    tempSum = 0
    tempList = []
    for j in M_CentralityVector[i]: # M-centrality Vector  rank
        tempSum = tempSum + (j[1] * layer_NAN_NAE_ANCPC_NIICCL_Influence[j[0]][5])
        tempList.append(j[1])
    nodesInfluence_MCentrality[i] = tempSum
    nodesInfluence_MCentralityMean[i] = sum(tempList) / len(tempList)
    nodesInfluence_MCentralityVariance[i] = (np.var(tempList))

nodesInfluence_Closeness = sorted(nodesInfluence_Closeness.items(), key=lambda x:x[1],reverse=True)
nodesInfluence_Betweenness = sorted(nodesInfluence_Betweenness.items(), key=lambda x:x[1],reverse=True)
nodesInfluence_PageRank = sorted(nodesInfluence_PageRank.items(), key=lambda x:x[1],reverse=True)
nodesInfluence_EigenVector = sorted(nodesInfluence_EigenVector.items(), key=lambda x:x[1],reverse=True)
nodesInfluence_Kats = sorted(nodesInfluence_Kats.items(), key=lambda x:x[1],reverse=True)
nodesInfluence_Harmonic = sorted(nodesInfluence_Harmonic.items(), key=lambda x:x[1],reverse=True)
nodesInfluence_MCentrality = sorted(nodesInfluence_MCentrality.items(), key=lambda x:x[1],reverse=True)
nodesInfluence_ClosenessMean = sorted(nodesInfluence_ClosenessMean.items(), key=lambda x:x[1],reverse=True)
nodesInfluence_ClosenessVariance = sorted(nodesInfluence_ClosenessVariance.items(), key=lambda x:x[1],reverse=True)
nodesInfluence_BetweennessMean = sorted(nodesInfluence_BetweennessMean.items(), key=lambda x:x[1],reverse=True)
nodesInfluence_BetweennessVariance = sorted(nodesInfluence_BetweennessVariance.items(), key=lambda x:x[1],reverse=True)
nodesInfluence_PageRankMean = sorted(nodesInfluence_PageRankMean.items(), key=lambda x:x[1],reverse=True)
nodesInfluence_PageRankVariance = sorted(nodesInfluence_PageRankVariance.items(), key=lambda x:x[1],reverse=True)
nodesInfluence_EigenVectorMean = sorted(nodesInfluence_EigenVectorMean.items(), key=lambda x:x[1],reverse=True)
nodesInfluence_EigenVectorVariance = sorted(nodesInfluence_EigenVectorVariance.items(), key=lambda x:x[1],reverse=True)
nodesInfluence_KatsMean = sorted(nodesInfluence_KatsMean.items(), key=lambda x:x[1],reverse=True)
nodesInfluence_KatsVariance = sorted(nodesInfluence_KatsVariance.items(), key=lambda x:x[1],reverse=True)
nodesInfluence_HarmonicMean = sorted(nodesInfluence_HarmonicMean.items(), key=lambda x:x[1],reverse=True)
nodesInfluence_HarmonicVariance = sorted(nodesInfluence_HarmonicVariance.items(), key=lambda x:x[1],reverse=True)
nodesInfluence_MCentralityMean = sorted(nodesInfluence_MCentralityMean.items(), key=lambda x:x[1],reverse=True)
nodesInfluence_MCentralityVariance = sorted(nodesInfluence_MCentralityVariance.items(), key=lambda x:x[1],reverse=True)

with open('./results/'+datasetName+'_SIR_'+str(SIR_beta)+'.pickle', 'rb') as f: # Load result of SIR model for a dataset
    SIR_Rank = pickle.load(f)

def differenceOfIndex (list1,list2):
    differenceSum = 0
    for i in list1:
        position1 = list1.index(i)
        if i in SIR_Rank:
            position2 = list2.index(i)
            differenceSum = differenceSum + abs(position1-position2)
        else:
            differenceSum = differenceSum +(Top_L)
    return differenceSum

temp1 = []
temp2 = []
temp3 = []
for i in range(0,Top_L):
    temp1.append(nodesInfluence_Closeness[i][0])
    temp2.append(nodesInfluence_ClosenessMean[i][0])
    temp3.append(nodesInfluence_ClosenessVariance[i][0])
similarityPercentageP = ((len(set(temp1) & set(SIR_Rank))) * 100) / Top_L
similarityPercentageM = ((len(set(temp2) & set(SIR_Rank))) * 100) / Top_L
similarityPercentageV = ((len(set(temp3) & set(SIR_Rank))) * 100) / Top_L
print("Closeness metric Similarity:")
print("Proposed Method: ", similarityPercentageP)
print("Mean: ", similarityPercentageM)
print("Variance: ", similarityPercentageV)
print("    ")
print("***Closeness metric Difference:***")
print("Proposed Method: ",differenceOfIndex(temp1,SIR_Rank))
print("Mean: ",differenceOfIndex(temp2,SIR_Rank))
print("Variance: ",differenceOfIndex(temp3,SIR_Rank))

print("============================================================")

temp1 = []
temp2 = []
temp3 = []
for i in range(0,Top_L):
    temp1.append(nodesInfluence_Betweenness[i][0])
    temp2.append(nodesInfluence_BetweennessMean[i][0])
    temp3.append(nodesInfluence_BetweennessVariance[i][0])
similarityPercentageP = ((len(set(temp1) & set(SIR_Rank))) * 100) / Top_L
similarityPercentageM = ((len(set(temp2) & set(SIR_Rank))) * 100) / Top_L
similarityPercentageV = ((len(set(temp3) & set(SIR_Rank))) * 100) / Top_L
print("Betweenness metric Similarity: ")
print("Proposed Method: ", similarityPercentageP)
print("Mean: ", similarityPercentageM)
print("Variance: ", similarityPercentageV)
print("    ")
print("***Betweenness metric Difference:***")
print("Proposed Method: ",differenceOfIndex(temp1,SIR_Rank))
print("Mean: ",differenceOfIndex(temp2,SIR_Rank))
print("Variance: ",differenceOfIndex(temp3,SIR_Rank))
print("============================================================")
temp1 = []
temp2 = []
temp3 = []
for i in range(0,Top_L):
    temp1.append(nodesInfluence_PageRank[i][0])
    temp2.append(nodesInfluence_PageRankMean[i][0])
    temp3.append(nodesInfluence_PageRankVariance[i][0])

similarityPercentageP = ((len(set(temp1) & set(SIR_Rank))) * 100) / Top_L
similarityPercentageM = ((len(set(temp2) & set(SIR_Rank))) * 100) / Top_L
similarityPercentageV = ((len(set(temp3) & set(SIR_Rank))) * 100) / Top_L
print("PageRank metric Similarity: ")
print("Proposed Method: ", similarityPercentageP)
print("Mean: ", similarityPercentageM)
print("Variance: ", similarityPercentageV)
print("    ")
print("***PageRank metric Difference:***")
print("Proposed Method: ",differenceOfIndex(temp1,SIR_Rank))
print("Mean: ",differenceOfIndex(temp2,SIR_Rank))
print("Variance: ",differenceOfIndex(temp3,SIR_Rank))
print("============================================================")
temp1 = []
temp2 = []
temp3 = []
for i in range(0,Top_L):
    temp1.append(nodesInfluence_EigenVector[i][0])
    temp2.append(nodesInfluence_EigenVectorMean[i][0])
    temp3.append(nodesInfluence_EigenVectorVariance[i][0])
similarityPercentageP = ((len(set(temp1) & set(SIR_Rank))) * 100) / Top_L
similarityPercentageM = ((len(set(temp2) & set(SIR_Rank))) * 100) / Top_L
similarityPercentageV = ((len(set(temp3) & set(SIR_Rank))) * 100) / Top_L
print("EigenVector metric Similarity: ")
print("Proposed Method: ", similarityPercentageP)
print("Mean: ", similarityPercentageM)
print("Variance: ", similarityPercentageV)
print("    ")
print("***EigenVector metric Difference:***")
print("Proposed Method: ",differenceOfIndex(temp1,SIR_Rank))
print("Mean: ",differenceOfIndex(temp2,SIR_Rank))
print("Variance: ",differenceOfIndex(temp3,SIR_Rank))
print("============================================================")

temp1 = []
temp2 = []
temp3 = []
for i in range(0,Top_L):
    temp1.append(nodesInfluence_Kats[i][0])
    temp2.append(nodesInfluence_KatsMean[i][0])
    temp3.append(nodesInfluence_KatsVariance[i][0])
similarityPercentageP = ((len(set(temp1) & set(SIR_Rank))) * 100) / Top_L
similarityPercentageM = ((len(set(temp2) & set(SIR_Rank))) * 100) / Top_L
similarityPercentageV = ((len(set(temp3) & set(SIR_Rank))) * 100) / Top_L
print("Kats metric Similarity: ")
print("Proposed Method: ", similarityPercentageP)
print("Mean: ", similarityPercentageM)
print("Variance: ", similarityPercentageV)
print("    ")
print("***Kats metric Difference:***")
print("Proposed Method: ",differenceOfIndex(temp1,SIR_Rank))
print("Mean: ",differenceOfIndex(temp2,SIR_Rank))
print("Variance: ",differenceOfIndex(temp3,SIR_Rank))
print("============================================================")

temp1 = []
temp2 = []
temp3 = []
for i in range(0,Top_L):
    temp1.append(nodesInfluence_Harmonic[i][0])
    temp2.append(nodesInfluence_HarmonicMean[i][0])
    temp3.append(nodesInfluence_HarmonicVariance[i][0])
similarityPercentageP = ((len(set(temp1) & set(SIR_Rank))) * 100) / Top_L
similarityPercentageM = ((len(set(temp2) & set(SIR_Rank))) * 100) / Top_L
similarityPercentageV = ((len(set(temp3) & set(SIR_Rank))) * 100) / Top_L
print("Harmonic metric Similarity: ")
print("Proposed Method: ", similarityPercentageP)
print("Mean: ", similarityPercentageM)
print("Variance: ", similarityPercentageV)
print("    ")
print("***Harmonic metric Difference:***")
print("Proposed Method: ",differenceOfIndex(temp1,SIR_Rank))
print("Mean: ",differenceOfIndex(temp2,SIR_Rank))
print("Variance: ",differenceOfIndex(temp3,SIR_Rank))
print("============================================================")

temp1 = []
temp2 = []
temp3 = []
for i in range(0,Top_L):
    temp1.append(nodesInfluence_MCentrality[i][0])
    temp2.append(nodesInfluence_MCentralityMean[i][0])
    temp3.append(nodesInfluence_MCentralityVariance[i][0])

similarityPercentageP = ((len(set(temp1) & set(SIR_Rank))) * 100) / Top_L
similarityPercentageM = ((len(set(temp2) & set(SIR_Rank))) * 100) / Top_L
similarityPercentageV = ((len(set(temp3) & set(SIR_Rank))) * 100) / Top_L
print("M-Centrality metric Similarity: ")
print("Proposed Method: ", similarityPercentageP)
print("Mean: ", similarityPercentageM)
print("Variance: ", similarityPercentageV)
print("    ")
print("***M-Centrality metric Difference:***")
print("Proposed Method: ",differenceOfIndex(temp1,SIR_Rank))
print("Mean: ",differenceOfIndex(temp2,SIR_Rank))
print("Variance: ",differenceOfIndex(temp3,SIR_Rank))
print("============================================================")