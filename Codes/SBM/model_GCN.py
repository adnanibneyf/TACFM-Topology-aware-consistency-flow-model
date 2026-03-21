"""

GCN based TACFM model for graph generation. 

key differences from model.py (MLP): 
MLP: treats each edge independently ---> Cant learn correlations. 
GCN: nodes talk to neighbors ---> learns community structure directly. 

The GCN processes the graph structure during flow matching. 
At each time t, the intermediate adjacency x_t defines a soft graph and the GCN
does message passing on that graph to predict velocities. 
"""

import torch
import torch.nn as nn
import numpy as np

class SinusoidalTimeEmbeddings(nn.Module):
    def __init__(self, dim): 
        super().__init__() 
        self.dim = dim 

    def forward(self, time): 
        device = time.device 
        half_dim = self.dim // 2 
        embeddings = np.log(10000) / (half_dim - 1) 
        embeddings = torch.exp(torch.arange(half_dim, device = device) * -embeddings) 
        embeddings = time * embeddings[None, :] 
        embeddings = torch.cat((embeddings.sin() , embeddings.cos()), dim=-1) 
        return embeddings 

class GCNLayer(nn.Module): 
    """
    Graph convolutional layer: h' = ReLU(D^{-1} A_hat h W + b) 
    What it does: 
    for each node i : 
        1. Collect features from ALL neighbors of i. 
        2. Average them ( normalize by degree ) 
        3. Transform through a linear layer.
        4. Add a skip connection. 
    This is how the model learns "if nodes 3,5,7 are in the same community, their edges
    should be connected."
    """

    def __init__(self, in_dim , out_dim): 
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim) 
        self.skip = nn.Linear(in_dim, out_dim) if in_dim != out_dim else nn.Identity() 
        self.norm = nn.LayerNorm(out_dim) 
        self.act = nn.GELU() 

    def forward(self, h, adj): 
        """
        h: (batch,  N, in_dim ) ---> node features. 
        adj: (batch, N, N) ---> soft adjacency ( edge weights from flow ) 
        """
        N = h.shape[1] 

        # add self-loops: every node always sees itself.
        adj_hat = adj + torch.eye(N, device = adj.device).unsqueeze(0) 

        # Row - normalize : divide by degree so high-degree nodes don't dominate.
        degree = adj_hat.sum(dim = -1, keepdim = True).clamp(min = 1.0) 
        adj_norm = adj_hat / degree # ( batch, N , N )

        # Message passing: aggregate neighbor features.
        h_neighbors = torch.bmm(adj_norm, h) # ( batch, N , in_dim )

        # transform + skip connection + normalize
        h_out = self.linear(h_neighbors) + self.skip(h)
        h_out = self.norm(h_out) 
        h_out = self.act(h_out) 
        return h_out 


class GCN_TACFM(nn.Module):
    """
    Topology aware flow matching with graph convolutional backbone.

    Architecture:
    1. Reshape flat vector (190) --> adjacency matrix (20x20)
    2. Use the adjacency as both features and graph structure.
    3. GCN message passing : nodes exchange info with neighbors. 
    4. Predict edge velocities from pairs of node embeddings.
    5. Project to tangent space of the hypersphere. 

    Why is this better than MLP? 
    MLP sees ---> [edge_12, edge_13, edge_14, ....] (flat with no structure) 
    GCN sees ---> " node 1 connects to nodes 2,3,4" ( graph aware ) 
    """
    def __init__(self, max_nodes = 20, time_dim = 32, node_hidden = 128, num_gcn_layers = 4): 
        super().__init__() 

        self.max_nodes = max_nodes
        self.data_dim = max_nodes * ( max_nodes - 1) // 2 # 190 

        self.time_mlp = nn.Sequential( 
            SinusoidalTimeEmbeddings(time_dim), 
            nn.Linear(time_dim, time_dim), 
            nn.GELU() 
        )

        # each node's features = its row in adjacency matrix + time embedding
        self.node_input = nn.Linear(max_nodes + time_dim, node_hidden) 

        self.gcn_layers = nn.ModuleList([ 
            GCNLayer(node_hidden, node_hidden) for _ in range(num_gcn_layers)
        ])

        self.edge_mlp = nn.Sequential( 
            nn.Linear(2 * node_hidden, 2 * node_hidden), 
            nn.GELU(), 
            nn.Linear(2 * node_hidden, node_hidden), 
            nn.GELU(), 
            nn.Linear(node_hidden, 1) 
        )

    def forward(self , x_flat , t): 
        """
        x_flat = (batch, 190) ---> flattened upper-triangle adjacency on sphere
        t : (batch , 1) ---> time in [0,1] 
        Returns: (batch, 190) ---> predicted velocity vector. 1
        """
        batch_shape = x_flat.shape[0] 
        N = self.max_nodes 

        # x_flat lives on sphere 
        # Reconstruct full symmetric matrix for message passing. 
        adj = self._vec_to_adj(x_flat) 

        # Soft adjacency matrix for GCN : use sigmoid to get [0,1] edge weights. 
        #  This tells the GCN " how connected " each pair of nodes cuurently is.
        adj_soft = torch.sigmoid(adj * 10) # sharpen the soft edges. 

        # Build node features. 
        # Each node's feature = row of the adjacency matrix. 
        node_feat = adj # (batch, 20, 20) --> node i's feature is adj[i, :] 

        # Time embedding --> broadcast to all nodes. 
        t_emb = self.time_mlp(t) # (batch, 32) 
        t_emb = t_emb.unsqueeze(1).expand(-1, N , -1) # ( batch, 20, 32 )

        # Combine ( batch , 20, 20 + 32) 
        h = torch.cat([node_feat, t_emb], dim = -1) 
        h = self.node_input(h) # (batch, 20, 64) 

        # GCN message passing.
        # Nodes exchange information with their neighbors
        # After this, each node knows about its community
        for gcn in self.gcn_layers: 
            h = gcn(h, adj_soft)# (batch, 20, 64 ) 

        # Predict edge velocities
        #  for each edge (i,j) combine the two embeddings
        idx = torch.triu_indices(N, N, offset = 1) 
        h_i = h[: , idx[0], :] # ( batch, 190, 64) --> source node features
        h_j = h[:, idx[1] , :] # ( batch, 190, 64) --> target node features

        edge_feat = torch.cat([h_i, h_j] , dim = -1) # ( batch, 190 , 128) 
        velocity = self.edge_mlp(edge_feat).squeeze(-1) # ( batch , 190 ) 

        return velocity

    def project_to_tangent(self, x, v): 
        """
        Same tangent projection as MLP model --> sphere geometry.
        """
        dot = (v * x).sum(dim=1, keepdim=True)
        return v - dot * x

    def _vec_to_adj(self , vec): 
        """
        helper : flat vector ---> symmetric adjacency matrix.  
        """
        batch_size = vec.shape[0] 
        N = self.max_nodes
        adj = torch.zeros(batch_size, N , N , device = vec.device) 
        idx = torch.triu_indices(N , N , offset = 1) 
        adj[:, idx[0], idx[1]] = vec 
        adj = adj + adj.transpose(1,2) 
        return adj


# Utility functions ( same as model.py )
def flatten_adj_to_vec(adj_batch):
    n = adj_batch.shape[1]
    idx = torch.triu_indices(n, n, offset=1)
    return adj_batch[:, idx[0], idx[1]]
def vect_to_adj(vect_batch, n=20):
    batch_sz = vect_batch.shape[0]
    adj = torch.zeros(batch_sz, n, n, device=vect_batch.device)
    idx = torch.triu_indices(n, n, offset=1)
    adj[:, idx[0], idx[1]] = vect_batch
    adj = adj + adj.transpose(1, 2)
    return adj
def normalize_to_sphere(x):
    return x / (x.norm(dim=1, keepdim=True) + 1e-8)


# Sanity check 

if __name__ == '__main__': 
    print("GCN TACFM architecture test.") 
    model = GCN_TACFM(max_nodes = 20, time_dim = 32, node_hidden = 128, num_gcn_layers=4) 
    param_count = sum(p.numel() for p in model.parameters()) 
    print(f"Parameters: {param_count:,}")

     # Test forward pass
    x = torch.randn(8, 190)
    x = normalize_to_sphere(x)
    t = torch.rand(8, 1)
    v = model(x, t)
    v_tan = model.project_to_tangent(x, v)
    print(f"Input:  x={x.shape}, t={t.shape}")
    print(f"Output: v={v.shape}")
    print(f"Tangent check (should be ~0): {(v_tan * x).sum(dim=1).abs().mean():.6f}")
    # Compare with MLP
    from model import TACFM
    mlp_model = TACFM(data_dim=190)
    mlp_params = sum(p.numel() for p in mlp_model.parameters())
    print(f"\nMLP params:  {mlp_params:,}")
    print(f"GCN params:  {param_count:,}")

