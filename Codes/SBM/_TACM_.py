"""
_TACM_.py: Riemannian Consistency Model for Topology-Aware Graph Generation

Approach: Exact Riemannian Exponential Map Parameterization.
Unlike Euclidean interpolation, this script enforces consistency
strictly along the curved surface of the hypersphere using:
  1. Tangent projection of network outputs
  2. The Riemannian Exponential Map for prediction: Exp_x((1-t)*v)
  3. Geodesic distance (arccos) instead of Euclidean MSE for losses.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pickle
import copy
import os
import time

from model_GCN import GCN_TACFM, flatten_adj_to_vec, vect_to_adj, normalize_to_sphere

MAX_NODES = 20
DATA_DIM = MAX_NODES * (MAX_NODES - 1) // 2  # 190
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# ============================================================
#  CONSISTENCY MODEL (Exponential Map)
# ============================================================

class _TACM_(nn.Module):
    def __init__(self, max_nodes=20):
        super().__init__()
        self.net = GCN_TACFM(max_nodes=max_nodes)
        self.data_dim = max_nodes * (max_nodes - 1) // 2

    def forward(self, x, t):
        """
        x: (batch, 190) on sphere
        t: (batch, 1) in [0, 1]
        Returns: predicted destination x_1 on the sphere
        """
        # 1. Raw prediction vector from GCN
        raw = self.net(x, t)
        
        # -------------------------------------------------------------
        # MATHEMATICAL CHANGE 1: Projection & Exponential Map
        # -------------------------------------------------------------
        # Project raw output strictly onto the Tangent Space T_x(M)
        dot = (raw * x).sum(dim=1, keepdim=True)
        v = raw - dot * x
        
        # Scale velocity by remaining time (1 - t)
        v_scaled = (1 - t) * v
        
        # Riemannian Exponential Map: Exp_x(v_scaled)
        # S is the magnitude of the jump (geodesic distance to travel)
        S = torch.norm(v_scaled, p=2, dim=1, keepdim=True)
        
        # Avoid division by zero
        S_clamp = S.clamp(min=1e-7)
        v_norm = v_scaled / S_clamp
        
        # The exact path on a sphere using trigonometry
        out = torch.cos(S) * x + torch.sin(S) * v_norm
        
        # Re-normalize to fix tiny floating point errors
        return normalize_to_sphere(out)


# ============================================================
#  GEODESIC INTERPOLATION 
# ============================================================

def geodesic_interpolation(x0, x1, t):
    """
    Compute x_t on the exact geodesic between x0 and x1 on S^d.
    """
    cos_theta = (x0 * x1).sum(dim=1, keepdim=True).clamp(-1 + 1e-6, 1 - 1e-6)
    theta = torch.acos(cos_theta)
    sin_theta = torch.sin(theta).clamp(min=1e-6)
    
    # Spherical linear interpolation (slerp)
    x_t = (torch.sin((1 - t) * theta) / sin_theta) * x0 + \
          (torch.sin(t * theta) / sin_theta) * x1
    
    return normalize_to_sphere(x_t)


# ============================================================
#  DATA LOADING
# ============================================================

def load_data(pkl_path, train_split=0.8):
    with open(pkl_path, 'rb') as f:
        graph_list = pickle.load(f)
    
    import networkx as nx
    adj_list = []
    for G in graph_list:
        adj = nx.to_numpy_array(G)
        padded = np.zeros((MAX_NODES, MAX_NODES))
        n = adj.shape[0]
        padded[:n, :n] = adj
        adj_list.append(padded)
    
    adj_all = np.array(adj_list, dtype=np.float32)
    n_train = int(len(adj_all) * train_split)
    train_tensor = torch.tensor(adj_all[:n_train])
    return DataLoader(TensorDataset(train_tensor), batch_size=16, shuffle=True)


# ============================================================
#  CONSISTENCY TRAINING 
# ============================================================

def distill(teacher_path=None, epochs=2000, lr=5e-4, ema_rate=0.995):
    results_dir = "results_tacm"
    os.makedirs(results_dir, exist_ok=True)

    student = _TACM_(max_nodes=MAX_NODES).to(DEVICE)
    
    ema_student = copy.deepcopy(student)
    ema_student.eval()
    for p in ema_student.parameters():
        p.requires_grad = False

    print(f"  Student params: {sum(p.numel() for p in student.parameters()):,}")

    train_loader = load_data('data/community_small.pkl')

    optimizer = optim.Adam(student.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    loss_history = []
    best_loss = float('inf')

    print(f"\n  Riemannian Consistency Training for {epochs} epochs...")
    start_time = time.time()

    for epoch in range(epochs):
        epoch_loss = 0

        for (adj_batch,) in train_loader:
            adj_batch = adj_batch.to(DEVICE)
            batch_sz = adj_batch.shape[0]

            x1 = flatten_adj_to_vec(adj_batch)
            x1 = x1 * 2 - 1
            x1 = normalize_to_sphere(x1)

            x0 = torch.randn_like(x1)
            x0 = normalize_to_sphere(x0)

            delta = torch.rand(batch_sz, 1, device=DEVICE) * 0.2 + 0.1
            t = torch.rand(batch_sz, 1, device=DEVICE) * 0.65 + 0.3
            t_minus = (t - delta).clamp(min=0.01)

            x_t = geodesic_interpolation(x0, x1, t)
            x_t_minus = geodesic_interpolation(x0, x1, t_minus)

            pred_t = student(x_t, t)
            with torch.no_grad():
                pred_t_minus = ema_student(x_t_minus, t_minus)

            # -------------------------------------------------------------
            # MATHEMATICAL CHANGE 2: Geodesic Consistency Loss
            # -------------------------------------------------------------
            # Calculate the angle between the two predictions on the sphere
            cos_angle = (pred_t * pred_t_minus).sum(dim=1).clamp(-1 + 1e-6, 1 - 1e-6)
            loss_consistency = torch.mean(torch.acos(cos_angle) ** 2)

            # -------------------------------------------------------------
            # MATHEMATICAL CHANGE 3: Geodesic Denoising Anchor Loss
            # -------------------------------------------------------------
            cos_denoise = (pred_t * x1).sum(dim=1).clamp(-1 + 1e-6, 1 - 1e-6)
            loss_denoise = torch.mean(torch.acos(cos_denoise) ** 2)

            # Combined loss (weight for denoising remains the same)
            loss = loss_consistency + 3.0 * loss_denoise

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(student.parameters(), max_norm=1.0)
            optimizer.step()

            with torch.no_grad():
                for p_ema, p_student in zip(ema_student.parameters(), student.parameters()):
                    p_ema.data.mul_(ema_rate).add_(p_student.data, alpha=1 - ema_rate)

            epoch_loss += loss.item()

        scheduler.step()
        avg_loss = epoch_loss / len(train_loader)
        loss_history.append(avg_loss)

        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(student.state_dict(), os.path.join(results_dir, "_TACM_best.pth"))

        if (epoch + 1) % 50 == 0 or epoch == 0:
            elapsed = time.time() - start_time
            print(f"Epoch {epoch+1:4d}/{epochs} | Loss: {avg_loss:.6f} | "
                  f"Best: {best_loss:.6f} | Time: {elapsed:.0f}s")

    print(f"\nTraining complete. Best loss: {best_loss:.6f}")
    torch.save(student.state_dict(), os.path.join(results_dir, "_TACM_final.pth"))
    return student


# ============================================================
#  DATA GENERATION
# ============================================================

@torch.no_grad()
def generate_one_step(model, num_samples=50):
    model.eval()
    x = torch.randn(num_samples, DATA_DIM, device=DEVICE)
    x = normalize_to_sphere(x)

    t = torch.zeros(num_samples, 1, device=DEVICE)
    out = model(x, t)

    adj = vect_to_adj(out, n=MAX_NODES)
    adj = (adj > 0).float()
    return adj.cpu().numpy()


@torch.no_grad()
def generate_few_steps(model, num_samples=50, num_steps=4):
    model.eval()
    x = torch.randn(num_samples, DATA_DIM, device=DEVICE)
    x = normalize_to_sphere(x)

    time_points = torch.linspace(0.0, 0.8, num_steps)

    for i, t_val in enumerate(time_points):
        t = torch.full((num_samples, 1), t_val.item(), device=DEVICE)
        x = model(x, t)
        
        if i < num_steps - 1:
            noise = torch.randn_like(x) * 0.05
            x = normalize_to_sphere(x + noise)

    adj = vect_to_adj(x, n=MAX_NODES)
    adj = (adj > 0).float()
    return adj.cpu().numpy()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=2000)
    parser.add_argument('--lr', type=float, default=5e-4)
    args = parser.parse_args()

    student = distill(epochs=args.epochs, lr=args.lr)

    print("\n=== 1-Step Generation ===")
    adj_1step = generate_one_step(student, num_samples=20)
    nodes = [np.sum(a.sum(axis=1) > 0) for a in adj_1step]
    edges = [int(a.sum() / 2) for a in adj_1step]
    print(f"Generated {len(adj_1step)} graphs")
    print(f"  Nodes: {np.mean(nodes):.1f} avg (target: 15.6)")
    print(f"  Edges: {np.mean(edges):.1f} avg (target: 36.9)")

    print("\n=== 4-Step Generation ===")
    adj_4step = generate_few_steps(student, num_samples=20, num_steps=4)
    nodes = [np.sum(a.sum(axis=1) > 0) for a in adj_4step]
    edges = [int(a.sum() / 2) for a in adj_4step]
    print(f"Generated {len(adj_4step)} graphs")
    print(f"  Nodes: {np.mean(nodes):.1f} avg (target: 15.6)")
    print(f"  Edges: {np.mean(edges):.1f} avg (target: 36.9)")
