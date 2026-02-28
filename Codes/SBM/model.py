import torch
import torch.nn as nn
import numpy as np


class SinusoidalTimeEmbeddings(nn.Module):
    """
    Encode scalar time t into a rich vector using sin/cos frequencies. 
    Why ? The network needs to understand "how far along the flow am I?"
    A single number t isn't expressive enough. This turns it into a 32-dimensional
    vector with different frequency components.
    """ 
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

class ResidualBlock(nn.Module): 
    """
    x --> Linear --> GELU ---> + x 
    The skip connection (the "+ x ") helps gradients flow during training.
    """ 
    def __init__(self, dim): 
        super().__init__()
        self.fc = nn.Linear(dim, dim) 
        self.act = nn.GELU() 

    def forward(self, x) : 
        return x + self.act(self.fc(x))

#  TACFM model (topology aware ---> uses geodesic flow on S^d ) 
class TACFM(nn.Module):
    """
    Key idea: We treat the flattened upper-triangle of the adjacency matrix as a point on a higher
    dimensional hypersphere S^d. 
    Why a sphere ? 
    1) The adjacency vector is normalized to unit norm --> lives on S^d 
    2) Geodesic on S^d = great circles (shortest curved paths) 
    3) Tangent projection keeps veclocity vectors ON the sphere surface 
    4) This prevents the model from "drifting off" the data manifold.

    The flow goes like this ---> random point on S^d ---> real graph's adjacency (on S^d) 
    along a geodesic ( curved shortest path ) , Not a straight line.
    """
    def __init__(self, data_dim, time_dim = 32, hidden_dim = 256):  
        """
        Args: 
            data_dim: Dimension of flattened upper-triangle of the adjacency vector. 
                      For 20 node graphs, it is 20*19/2 = 190. 
            time_dim: Dimension of the time embedding 
            hidden_dim: Dimension of the hidden layers
        """
        super().__init__() 
        self.data_dim = data_dim

        self.time_mlp = nn.Sequential(
            SinusoidalTimeEmbeddings(time_dim), 
            nn.Linear(time_dim, time_dim), 
            nn.GELU() 
        )

        self.input_projection = nn.Linear(data_dim + time_dim, hidden_dim) 

        # Deeper body for the higher dimensional problem 
        self.body = nn.Sequential( 
            ResidualBlock(hidden_dim), 
            ResidualBlock(hidden_dim), 
            ResidualBlock(hidden_dim), 
            ResidualBlock(hidden_dim), 
            nn.LayerNorm(hidden_dim)
        )

        # Output : veclocity vector ( same dim as input = 190 ) 
        self.output_head = nn.Linear(hidden_dim , data_dim) 


    def forward(self, x, t): 
        """
        Given a point x on the spehere and time t, predict the velocity vector
        Args: 
        x : (batch, data_dim) ---> point on S^d (normalized adjacency vector) 
        t : (batch, 1) ---> time in [0,1] 
        Returns : veclocity ( batch, data_dim) ---> predicted tangent vector at x
        """

        t_emb = self.time_mlp(t) 
        h = torch.cat([x, t_emb], dim=1) 
        h = self.input_projection(h) 
        h = self.body(h)
        velocity = self.output_head(h) 
        return velocity 


    def project_to_tangent(self, x , v): 
        """
        Project velocity v onto the tangent space of S^d at point x. 

        On a sphere, the tangent space at x is the hyperplane perpendicular to x.
        Formula : v_tangent = v - (v.x)*x 

        This is the core of topology-awareness. 
        ---> It ensures the predicted velocity doesn't push points off the sphere. 
        ---> In Euclidean space, this step doesn't exist. ( tangent space = full R^d ) 
        """
        dot = (v * x).sum(dim=1, keepdim=True) # how much v points along x. 
        return v - dot * x # remove that component. 


# Euclidean baseline which is topology-blind ( flat space ) 

class EuclideanFM_GraphModel(nn.Module):
    """
    Standard (Euclidean) flow matching for graph generation. 

    This is the baseline model. It treats the adjacency vector as a flat vector in R^d with
    NO geometric constraints. 

    --> Interpolation: Straight line --> x_t = (1-t)*x_0 + t*x_1 
    --> No tangent projection (velocity can point anywhere) 
    --> No normalization constraint.

    Same arch as TACFM. The only difference is the geometry used in the loss function, not in the 
    model.
    """

    def __init__(self, data_dim, time_dim=32, hidden_dim=256):
        super().__init__()
        
        self.data_dim = data_dim
        
        self.time_mlp = nn.Sequential(
            SinusoidalTimeEmbeddings(time_dim),
            nn.Linear(time_dim, time_dim),
            nn.GELU()
        )
        
        self.input_projection = nn.Linear(data_dim + time_dim, hidden_dim)
        
        # EXACT same body as TACFM — this is crucial for fair comparison
        self.body = nn.Sequential(
            ResidualBlock(hidden_dim),
            ResidualBlock(hidden_dim),
            ResidualBlock(hidden_dim),
            ResidualBlock(hidden_dim),
            nn.LayerNorm(hidden_dim)
        )
        
        self.output_head = nn.Linear(hidden_dim, data_dim)

    def forward(self, x, t):
        t_emb = self.time_mlp(t)
        h = torch.cat([x, t_emb], dim=1)
        h = self.input_projection(h)
        h = self.body(h)
        velocity = self.output_head(h)
        return velocity  # No projection — velocity lives in flat R^d   



# geometric utilities ( used during training, not inside model ) 

def flatten_adj_to_vec(adj_batch): 
    """
    Extract upper-triangle of adjacency matrices ---> flat vectors. 
    A 20x20 symmetric matrix has 190 independent entries (20*19/2)
    We only store the upper triangle (excluding diagonal) to avoid redundancy.

    adj_batch: (batch, 20, 20) ---> returns ( batch, 190 ) 
    """

    batch_sz = adj_batch.shape[0] 
    n = adj_batch.shape[1] 

    idx = torch.triu_indices(n,n, offset = 1)
    return adj_batch[:, idx[0], idx[1]] # (batch, 190)


def vect_to_adj(vect_batch, n=20): 
    """
    Convert flat vectors back to symmetric adjacency matrices. 
    """

    batch_sz = vect_batch.shape[0] 
    adj = torch.zeros(batch_sz, n , n, device = vect_batch.device) 
    idx = torch.triu_indices(n,n, offset = 1) 
    adj[:, idx[0], idx[1]] = vect_batch 
    adj = adj + adj.transpose(1,2) # make symmetric 
    return adj 


def normalize_to_sphere(x): 
    """
    Project vectors onto the unit hypersphere S^d. 
    x : (batch, d ) ---> x / ||x|| for each row  
    """
    return x / (x.norm(dim = 1, keepdim = True) + 1e-8) 


if __name__ == "__main__": 
    print("Model Architecture test:") 
    MAX_NODES = 20
    DATA_DIM = MAX_NODES * (MAX_NODES-1) //2 
    BATCH_SIZE = 8

    print(f"Data dimension ( upper triangle of {MAX_NODES} x {MAX_NODES}) : {DATA_DIM}")

    #  Test TACFM
    tacfm = TACFM(data_dim = DATA_DIM) 
    x = torch.randn(BATCH_SIZE, DATA_DIM) 
    x = normalize_to_sphere(x) 
    t = torch.rand(BATCH_SIZE, 1)

    v = tacfm(x, t) 
    v_tangent = tacfm.project_to_tangent(x,v) 

    print(f"\TACFM Model: ") 
    print(f" Parameters : {sum(p.numel() for p in tacfm.parameters()):,}") 
    print(f" input shape : x = {x.shape}, t = {t.shape}")
    print(f" Output shape: {v.shape}") 
    print(f" After tangent projection:  {v_tangent.shape}")

    dot = (v_tangent * x ).sum(dim = 1) 
    print(f" Tangent check (should be ~0): {dot.abs().mean():.6f}") 

    # Test Euclidean
    euclidean = EuclideanFM_GraphModel(data_dim=DATA_DIM)
    v_euc = euclidean(x, t)
    
    print(f"\nEuclidean Model:")
    print(f"  Parameters: {sum(p.numel() for p in euclidean.parameters()):,}")
    print(f"  Output shape: {v_euc.shape}")
    print(f"  (No tangent projection)")
    
    # Test utility functions
    print(f"\n=== Utility Functions Test ===")
    adj = torch.randint(0, 2, (4, 20, 20)).float()
    adj = (adj + adj.transpose(1, 2)).clamp(0, 1)  # make symmetric
    vec = flatten_adj_to_vec(adj)
    adj_reconstructed = vect_to_adj(vec)
    print(f"  adj → vec: {adj.shape} → {vec.shape}")
    print(f"  vec → adj: {vec.shape} → {adj_reconstructed.shape}")
    print(f"  Reconstruction error: {(adj - adj_reconstructed).abs().max():.6f}")