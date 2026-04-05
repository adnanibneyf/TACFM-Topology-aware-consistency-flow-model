"""
TACFM Consistency Model — Topology-Aware Consistency Flow Matching

Approach: Consistency Training (CT) — NO teacher needed.
Since we know the exact geodesic between noise x0 and data x1,
we compute x_t and x_{t-delta} DIRECTLY on the geodesic.

Loss = consistency_loss + reconstruction_loss
  consistency:    ||student(x_t, t) - ema_student(x_{t-d}, t-d)||^2
  reconstruction: ||student(x1, 1) - x1||^2   (anchor to real data)
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
#  CONSISTENCY MODEL
# ============================================================

class _TACFM_(nn.Module):
    """
    Consistency model with time-dependent skip connection.
    
    f(x, t) = t * x  +  (1-t) * network(x, t)
    
    Convention (matching our flow matching):
      t=0: noise, t=1: clean data
    
    Boundary condition: f(x, t=1) = x (identity at clean data)
    At t=0: f(x, 0) = network(x, 0) (fully predicted destination)
    """
    def __init__(self, max_nodes=20):
        super().__init__()
        self.net = GCN_TACFM(max_nodes=max_nodes)
        self.data_dim = max_nodes * (max_nodes - 1) // 2

    def forward(self, x, t):
        """
        x: (batch, 190) on sphere
        t: (batch, 1) in [0, 1]
        Returns: (batch, 190) predicted DESTINATION (x1) on sphere
        """
        raw = self.net(x, t)
        
        # Skip connection:
        #   t=0 (noise): f = 0*x + 1*net = net(x,0)  ← network predicts destination
        #   t=1 (data):  f = 1*x + 0*net = x          ← identity (already clean)
        output = t * x + (1 - t) * raw
        
        return normalize_to_sphere(output)

    def project_to_tangent(self, x, v):
        dot = (v * x).sum(dim=1, keepdim=True)
        return v - dot * x


# ============================================================
#  GEODESIC INTERPOLATION (exact)
# ============================================================

def geodesic_interpolation(x0, x1, t):
    """
    Compute x_t on the exact geodesic between x0 and x1 on S^d.
    x_t = x0 * cos(theta * t) + dir * sin(theta * t)
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
    """
    Consistency Training: learn to map any point on a geodesic
    trajectory to the same destination, using exact geodesics.
    
    No teacher needed — we compute trajectory points directly.
    """
    results_dir = "results_consistency"
    os.makedirs(results_dir, exist_ok=True)

    # Student model
    student = _TACFM_(max_nodes=MAX_NODES).to(DEVICE)
    
    # EMA copy (stable target)
    ema_student = copy.deepcopy(student)
    ema_student.eval()
    for p in ema_student.parameters():
        p.requires_grad = False

    param_count = sum(p.numel() for p in student.parameters())
    print(f"  Student params: {param_count:,}")

    # Load data
    train_loader = load_data('data/community_small.pkl')

    optimizer = optim.Adam(student.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    loss_history = []
    best_loss = float('inf')

    print(f"\n  Consistency Training for {epochs} epochs...")
    start_time = time.time()

    for epoch in range(epochs):
        epoch_loss = 0

        for (adj_batch,) in train_loader:
            adj_batch = adj_batch.to(DEVICE)
            batch_sz = adj_batch.shape[0]

            # === Prepare data ===
            x1 = flatten_adj_to_vec(adj_batch)
            x1 = x1 * 2 - 1  # {0,1} -> {-1,+1}
            x1 = normalize_to_sphere(x1)

            x0 = torch.randn_like(x1)
            x0 = normalize_to_sphere(x0)

            # === Sample two time points: t and t-delta ===
            # t in [0.3, 0.95], delta in [0.1, 0.3]
            delta = torch.rand(batch_sz, 1, device=DEVICE) * 0.2 + 0.1  # [0.1, 0.3]
            t = torch.rand(batch_sz, 1, device=DEVICE) * 0.65 + 0.3     # [0.3, 0.95]
            t_minus = (t - delta).clamp(min=0.01)

            # === Compute x_t and x_{t-delta} on EXACT geodesic ===
            # No teacher needed — we know the geodesic analytically
            x_t = geodesic_interpolation(x0, x1, t)
            x_t_minus = geodesic_interpolation(x0, x1, t_minus)

            # === Consistency loss ===
            # Both points are on the SAME trajectory, so the model
            # should map them to the SAME destination (x1)
            pred_t = student(x_t, t)

            with torch.no_grad():
                pred_t_minus = ema_student(x_t_minus, t_minus)

            loss_consistency = torch.mean((pred_t - pred_t_minus) ** 2)

            # === Denoising loss (direct supervision) ===
            # The model's output should be close to the real data x1
            # This tells the model WHAT the destination is, not just
            # that it should be consistent
            loss_denoise = torch.mean((pred_t - x1) ** 2)

            # === Combined loss ===
            loss = loss_consistency + 3.0 * loss_denoise

            # === Update ===
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(student.parameters(), max_norm=1.0)
            optimizer.step()

            # === Update EMA ===
            with torch.no_grad():
                for p_ema, p_student in zip(ema_student.parameters(), student.parameters()):
                    p_ema.data.mul_(ema_rate).add_(p_student.data, alpha=1 - ema_rate)

            epoch_loss += loss.item()

        scheduler.step()
        avg_loss = epoch_loss / len(train_loader)
        loss_history.append(avg_loss)

        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(student.state_dict(), os.path.join(results_dir, "best_model.pth"))

        if (epoch + 1) % 50 == 0 or epoch == 0:
            elapsed = time.time() - start_time
            print(f"Epoch {epoch+1:4d}/{epochs} | Loss: {avg_loss:.6f} | "
                  f"Best: {best_loss:.6f} | Time: {elapsed:.0f}s")

    print(f"\nTraining complete. Best loss: {best_loss:.6f}")
    torch.save(student.state_dict(), os.path.join(results_dir, "final_model.pth"))
    return student


# ============================================================
#  1-STEP GENERATION
# ============================================================

@torch.no_grad()
def generate_one_step(model, num_samples=50):
    """Generate graphs in ONE forward pass."""
    model.eval()
    x = torch.randn(num_samples, DATA_DIM, device=DEVICE)
    x = normalize_to_sphere(x)

    # ONE forward pass at t=0 (input is pure noise, network fully predicts)
    t = torch.zeros(num_samples, 1, device=DEVICE)
    out = model(x, t)

    adj = vect_to_adj(out, n=MAX_NODES)
    adj = (adj > 0).float()
    return adj.cpu().numpy()


@torch.no_grad()
def generate_few_steps(model, num_samples=50, num_steps=4):
    """
    Multi-step consistency generation.
    Start from noise (t=0), progressively refine.
    """
    model.eval()
    x = torch.randn(num_samples, DATA_DIM, device=DEVICE)
    x = normalize_to_sphere(x)

    # Start at t=0 (noise), step toward t=1 (clean)
    time_points = torch.linspace(0.0, 0.8, num_steps)

    for i, t_val in enumerate(time_points):
        t = torch.full((num_samples, 1), t_val.item(), device=DEVICE)
        x = model(x, t)  # predict destination
        x = normalize_to_sphere(x)
        
        # Between steps, mix prediction with small noise to re-enter trajectory
        if i < num_steps - 1:
            noise = torch.randn_like(x) * 0.05
            x = normalize_to_sphere(x + noise)

    adj = vect_to_adj(x, n=MAX_NODES)
    adj = (adj > 0).float()
    return adj.cpu().numpy()


# ============================================================
#  MAIN
# ============================================================

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--teacher_path', type=str, 
                        default='TACFM_GCN_results/best_model.pth',
                        help='(Unused, kept for compatibility)')
    parser.add_argument('--epochs', type=int, default=2000)
    parser.add_argument('--lr', type=float, default=5e-4)
    args = parser.parse_args()

    student = distill(epochs=args.epochs, lr=args.lr)

    # Test 1-step generation
    print("\n=== 1-Step Generation ===")
    adj_1step = generate_one_step(student, num_samples=20)
    nodes = [np.sum(a.sum(axis=1) > 0) for a in adj_1step]
    edges = [int(a.sum() / 2) for a in adj_1step]
    print(f"Generated {len(adj_1step)} graphs")
    print(f"  Nodes: {np.mean(nodes):.1f} avg (target: 15.6)")
    print(f"  Edges: {np.mean(edges):.1f} avg (target: 36.9)")

    # Test 4-step generation
    print("\n=== 4-Step Generation ===")
    adj_4step = generate_few_steps(student, num_samples=20, num_steps=4)
    nodes = [np.sum(a.sum(axis=1) > 0) for a in adj_4step]
    edges = [int(a.sum() / 2) for a in adj_4step]
    print(f"Generated {len(adj_4step)} graphs")
    print(f"  Nodes: {np.mean(nodes):.1f} avg (target: 15.6)")
    print(f"  Edges: {np.mean(edges):.1f} avg (target: 36.9)")
