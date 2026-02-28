import networkx as nx
import numpy as np
import matplotlib.pyplot as plt

# Stochastic block model: nodes grouped into communities 
# High intra-commuinity edge probability, low inter-community

NUM_COMMUNITIES = 3
NODES_PER_COMMUNITY = 15
P_INTRA = 0.7 # edge probabilty within same community
P_INTER = 0.05 # edge probabilty between different communities
NUM_GRAPHS = 5 


sizes = [NODES_PER_COMMUNITY] * NUM_COMMUNITIES

# probabilty matrix
p_matrix = np.full((NUM_COMMUNITIES, NUM_COMMUNITIES), P_INTER)
np.fill_diagonal(p_matrix, P_INTRA) 

for i in range(NUM_GRAPHS): 
    # generate graph
    G = nx.stochastic_block_model(sizes, p_matrix, seed = i) 
    # extract community labels
    communities = [G.nodes[n]['block'] for n in G.nodes()]

    print(f"\n ---- SBM graph {i+1} ----")
    print(f" Nodes : {G.number_of_nodes()}")
    print(f" Edges: {G.number_of_edges()}")
    print(f" Communities: {NUM_COMMUNITIES} (sizes : { sizes })") 
    print(f" Is connected: {nx.is_connected(G)}") 
    print(f" Avg degree: {np.mean([ d for _, d in G.degree()]):.2f}")
    print(f" Community labels: {np.unique(communities, return_counts = True)}") 

    # adjacency matrix
    A = nx.to_numpy_array(G) 
    print(f" Adjacency matrix shape: {A.shape}") 
    print(f" Edge density : {A.sum() / (A.shape[0] * (A.shape[0] - 1)):.3f}")

plt.figure(figsize = (10,5)) 

plt.subplot(1,2,1) 
colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
node_colors  = [ colors[G.nodes[n]['block']] for n in G.nodes()] 
pos = nx.spring_layout(G, seed = 42)
nx.draw(G, pos = pos, node_size = 80, node_color = node_colors , edge_color = 'lightgray', 
            with_labels = False) 
plt.title(f"SBM graph (3 communities * {NODES_PER_COMMUNITY} nodes)") 

plt.subplot(1,2,2)
A = nx.to_numpy_array(G)
# sort by community for block-diagonal structure
order = np.argsort(communities)
A_sorted = A[np.ix_(order, order)] 

plt.imshow(A_sorted, cmap = 'gray' , interpolation = 'nearest') 
plt.title('Adjacency matrix (sorted by community)') 
plt.colorbar(shrink = 0.7)

plt.tight_layout() 
plt.show() 

