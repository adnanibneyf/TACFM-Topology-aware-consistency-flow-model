import torch
import torch.nn as nn
import numpy as np

class SinusoidalTimeEmbeddings(nn.Module):
    """
    Standard Time Embedding from the "Attention is All You Need" paper.
    Helps the network understand 'time' much better than a single number.
    """
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        # time shape: (Batch, 1)
        device = time.device
        half_dim = self.dim // 2
        embeddings = np.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings

class ResidualBlock(nn.Module):
    """
    A block that learns the 'difference' rather than the whole function.
    Helps training deep networks.
    """
    def __init__(self, dim):
        super().__init__()
        self.fc = nn.Linear(dim, dim)
        self.act = nn.GELU() # GELU is smoother than ReLU, better for flows

    def forward(self, x):
        return x + self.act(self.fc(x)) # The "Skip Connection" is the 'x +'

class TACFM_Model(nn.Module):
    def __init__(self, data_dim=3, time_dim=32, hidden_dim=128):
        super().__init__()
        
        # 1. Smarter Time Processing
        self.time_mlp = nn.Sequential(
            SinusoidalTimeEmbeddings(time_dim),
            nn.Linear(time_dim, time_dim),
            nn.GELU()
        )

        # 2. Input Embedding (Projects 3 coordinates to 128 hidden features)
        self.input_projection = nn.Linear(data_dim + time_dim, hidden_dim)

        # 3. The "Body" (Residual Blocks)
        # This is deeper and smarter than a simple feed-forward
        self.body = nn.Sequential(
            ResidualBlock(hidden_dim),
            ResidualBlock(hidden_dim),
            ResidualBlock(hidden_dim),
            nn.LayerNorm(hidden_dim) # Stabilizes training
        )

        # 4. Output Projection (Back to 3 coordinates)
        self.output_head = nn.Linear(hidden_dim, data_dim)

    def forward(self, x, t):
        # x: (Batch, 3) location on sphere
        # t: (Batch, 1) time [0, 1]

        # Embed time
        t_emb = self.time_mlp(t) # Turns (Batch, 1) -> (Batch, 32)
        
        # Combine x and t
        x_input = torch.cat([x, t_emb], dim=1)
        
        # Project to hidden dimension
        h = self.input_projection(x_input)
        
        # Pass through residual body
        h = self.body(h)
        
        # Predict velocity
        velocity = self.output_head(h)
        
        return velocity

    def project_to_tangent(self, x, vector):
        """
        Geometric constraint: Remove component perpendicular to surface
        """
        dot_product = (vector * x).sum(dim=1, keepdim=True)
        tangent_velocity = vector - dot_product * x
        return tangent_velocity