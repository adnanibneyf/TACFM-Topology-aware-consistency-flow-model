import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import pickle
import os

def n_community(num_communities, max_nodes, p_inter = 0.05): 
    """ Generate a single community-structured graph.
    
    Args:
        num_communities (int): Number of communities.
        max_nodes (int): Maximum number of nodes per community.
        p_inter (float, optional): Probability of edges between communities. Defaults to 0.05.

    Returns: 
        A single networkx Graph with community structure.
    """
    assert num_communities > 1 

    one_community_size = max_nodes // num_communities
    # c_sizes is a list of community sizes (e.g. [10, 10, 10])   
    c_sizes = [one_community_size] * num_communities 
    total_nodes = one_community_size * num_communities 
    
    # calculate bridge probability
    # This converts p_inter into the probability of creating a bridge between two communities
    # The formula accounts for the fact that there are 2 * (num_communities - 1) * one_community_size possible bridges
    #  and that we want the expected number of bridges to be equal to p_inter * total_nodes
    p_make_a_bridge = p_inter * 2 / ((num_communities-1) * one_community_size) 
    # generate graphs with high intra-community edge probability (0.7)
    graphs = [nx.gnp_random_graph(c_sizes[i], 0.7, seed = i ) for i in range(len(c_sizes))] 

    # Merge all communities into one graph
    G = nx.disjoint_union_all(graphs) 

    # Add sparse inter-community bridges
    communities = list(G.subgraph(c) for c in nx.connected_components(G)) 
    for i in range(len(communities)):
        subG1 = communities[i]
        nodes1= list(subG1.nodes()) 
        for j in range(i+1, len(communities)): 
            subG2 = communities[j]
            nodes2 = list(subG2.nodes())
            has_inter_edge = False
            for n1 in nodes1: 
                for n2 in nodes2: 
                    if np.random.rand() < p_make_a_bridge: 
                        G.add_edge(n1, n2) 
                        has_inter_edge = True 
            if not has_inter_edge: 
                G.add_edge(nodes1[0], nodes2[0]) 
    return G 


def generate_community_dataset(num_graphs = 100, min_nodes = 12, max_nodes = 20, 
                                    num_communities=2, save_path = None): 
    
    """ Generate a dataset of community graphs matching GDSS's Community_small config. 

    Each graph has a random number of nodes between min_nodes and max_nodes, 
    but always has exactly num_communities communities. 

    Args: 
        num_graphs: Number of graphs to generate. Defaults to 100. 
        min_nodes: Minimum number of nodes per graph. Defaults to 12. 
        max_nodes: Maximum number of nodes per graph. Defaults to 20. 
        num_communities: Number of communities per graph. Defaults to 2. 
        save_path: Path to save the dataset. Defaults to None. 

    Returns: 
        A list of networkx Graphs with community structure. 
    """

    graph_list = [] 

    for i in range(num_graphs): 
        n_nodes = np.random.randint(min_nodes, max_nodes + 1)
        G = n_community(num_communities = num_communities, max_nodes = n_nodes) 
        graph_list.append(G) 

        if (i+1)%20 == 0: 
            print(f"Generated {i+1}/{num_graphs} graphs") 

    # stats 
    node_counts = [G.number_of_nodes() for G in graph_list] 
    edge_counts = [G.number_of_edges() for G in graph_list] 
    print(f"\n---Summary-----")
    print(f"Total graphs: {len(graph_list)}") 
    print(f"Nodes range: {min(node_counts)} - {max(node_counts)}") 
    print(f"Edges range: {min(edge_counts)} - {max(edge_counts)}") 

    # save 
    if save_path: 
        os.makedirs(os.path.dirname(save_path), exist_ok = True) 
        with open(save_path, 'wb') as f: 
            pickle.dump(graph_list, f) 
        print(f"\nDataset saved to {save_path}") 

    return graph_list 

def graphs_to_adj_tensors(graph_list, max_node_num=20):
    """
        Convert a list of networkx graphs ---> padded adjacency tensors. 

        This is how the model sees the data: 
        ---> Each graph becomes a (max_node_num * max_node_num) binary matrix
        ---> Smaller graphs are zero-padded to max_node_num

        Args: 
            graph_list: List of networkx graphs. 
            max_node_num: Maximum number of nodes per graph. Defaults to 20. 

        Returns: 
            numpy array of shape ( num_graphs, max_node_num, max_node_num )  
    """ 

    adj_list = [ ] 
    for G in graph_list: 
        adj = nx.to_numpy_array(G) ## shape = n*n where n is the actual node_count.
        padded = np.zeros((max_node_num, max_node_num)) 
        n = adj.shape[0] 
        padded[:n, :n] = adj 
        adj_list.append(padded) 
    return np.array(adj_list) # Shape (num_graphs, 20,20) 


def adj_tensors_to_graphs(adj_tensors, threshold = 0.5): 
    """ 
    Convert generated adjacency list back into networkx graphs. 
    The model outputs continuos values in [0,1]. We threshold at 0.5 to get binary edges.
    Then we remove isolated nodes. 

    Args: 
        adj_tensors: Numpy array of shape (num_graphs, N, N)  
        threshold: Values above this are treated as edges. 

    Returns: 
        List of networkx graphs. 
    """  

    graphs = [] 
    for adj in adj_tensors: 
        adj = (adj > threshold).astype(float)
        adj = np.maximum(adj, adj.T) # make symmetric 
        np.fill_diagonal(adj, 0) # remove self loops 
        G = nx.from_numpy_array(adj) 
        G.remove_nodes_from(list(nx.isolates(G))) 
        graphs.append(G) 
    return graphs 


if __name__ == "__main__": 
    print("------- Generating community_small dataset ( GDSS matching ) ---------") 
    graphs = generate_community_dataset( 
        num_graphs = 100, 
        min_nodes = 12, 
        max_nodes = 20, 
        num_communities = 2, 
        save_path = "data/community_small.pkl" 
    )

    adj_tensors = graphs_to_adj_tensors(graphs, max_node_num=20) 
    print(f"\nAdjac ency tensors shape: {adj_tensors.shape}") 
    print(f" ----> {adj_tensors.shape[0]} graphs, each a {adj_tensors.shape[1]}x{adj_tensors.shape[2]} matrix")

    # Visualize one example.
    fig,axes = plt.subplots(1,2, figsize=(12,5)) 

    # graph viz 
    G = graphs[-1] 
    pos = nx.spring_layout(G, seed = 42) 
    nx.draw(G, pos = pos, ax =axes[0] , node_size = 80, node_color = 'dodgerblue', 
            edge_color= 'lightgray', with_labels = True, font_size = 8) 
    axes[0].set_title(f"Community Graph ({G.number_of_nodes()} nodes)")

    # adjacency matrix viz
    axes[1].imshow(adj_tensors[-1] , cmap = 'Blues', interpolation = 'nearest') 
    axes[1].set_title('Padded Adjacency Matrix') 
    axes[1].set_xlabel("Nodes") 
    axes[1].set_ylabel("Nodes") 
    plt.tight_layout()
    plt.savefig("community_sample.png")  
    plt.show() 
    