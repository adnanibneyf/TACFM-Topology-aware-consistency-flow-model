import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pickle
import argparse
import matplotlib.pyplot as plt
import time  
import os

from model import ( 
    TACFM, EuclideanFM_GraphModel, flatten_adj_to_vec, vect_to_adj, normalize_to_sphere 
)
from model_GCN import GCN_TACFM

# Config 
MAX_NODES = 20
DATA_DIM = MAX_NODES * (MAX_NODES - 1) // 2 
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu') 

# Dataset loading

def load_dataset(pkl_path, train_split = 0.8): 
    """
    Load community graphs from pickle, convert to adjacency tensors, and split into 
    train/test sets. 
    Returns: train_loader, test_adj_tensors ( numpy )
    """

    with open(pkl_path, 'rb') as f: 
        graph_list = pickle.load(f) 

    # Convert networkx graphs ---> padded adjacency matrices. 
    import networkx as nx
    adj_list = [] 
    for G in graph_list:
        adj = nx.to_numpy_array(G) 
        padded = np.zeros((MAX_NODES, MAX_NODES)) 
        n = adj.shape[0] 
        padded[:n, :n] = adj 
        adj_list.append(padded) 
    
    adj_all = np.array(adj_list, dtype = np.float32) 

    n_train = int(len(adj_all) * train_split) 
    train_adj = adj_all[:n_train]    
    test_adj = adj_all[n_train:] 

    train_tensor = torch.tensor(train_adj) 
    train_dataset = TensorDataset(train_tensor) 
    train_loader = DataLoader(train_dataset, batch_size = 16, shuffle = True) 

    print(f"Loaded {len(adj_all)} graphs --> Train : {n_train}, Test: {len(test_adj)}")

    return train_loader, test_adj 


# TACFM Loss ---> Geodesic flow on hypersphere S^d

def compute_TACFM_loss( model, adj_batch ) : 
    """
    This is the core of TACFM: geodesic flow matching on the hypersphere.

    Computed on the 190-dimensional hypersphere.

    The math doesn't change with dimension --> great circles, log maps 
    and tangent projections work the same way. 

    Steps: 
    1) Flatten adjacency ---> vector, normalize onto sphere.
    2) Sample random noise ON the sphere.
    3) Compute geodesic from x0 --> x1 
    4) Pick random time t, find point x_t on the geodesic
    5) Compute true velocity u_t (derivative of geodesic at t)
    6) Model predicts velocity, project to tangent space 
    7) Loss = ||predicted - true||^2 
    """

    batch_size = adj_batch.shape[0] 

    x1 = flatten_adj_to_vec(adj_batch) 
    x1 = x1 * 2 - 1   # {0,1} → {-1,+1} — sharp binary signal
    x1 = normalize_to_sphere(x1) 

    # random noise on the sphere.
    x0 = torch.randn_like(x1)
    x0 = normalize_to_sphere(x0) 

    # random time 
    t = torch.rand(batch_size, 1, device = DEVICE) 

    #  geodesic interpolation 
    #  Angle between x0 and x1
    cos_theta = (x0 * x1).sum(dim = 1, keepdim = True).clamp(-1+1e-6, 1-1e-6) 
    theta = torch.acos(cos_theta) 
    sin_theta = torch.sin(theta) + 1e-6

    # Log map: initial velocity v0 (direction from x0 toward x1) 
    v0 = (x1-x0 * cos_theta) * (theta/ sin_theta) 

    #  Point on geodesic at time t 
    dir_vect = v0 /( theta + 1e-6) 
    x_t = x0 *torch.cos(theta*t) + dir_vect*torch.sin(theta * t) 


    # True velocity at time t (derivative of geodesic) 
    u_t = -x0*theta * torch.sin(theta* t) + dir_vect * theta * torch.cos(theta * t) 

    predicted_v = model(x_t, t) 
    predicted_v = model.project_to_tangent(x_t, predicted_v) 

    loss = torch.mean((predicted_v - u_t)**2) 
    return loss


#  Euclidean loss 
def compute_euclidean_loss(model, adj_batch): 
    """
    Standard flow matching loss. 
    This is the baseline. No geometry, no manifold awareness. Everything 
    happens at flat S^d. 

    The difference from TACFM: 
    x_t = geodesic(x0,  x1, t) ( curved path on sphere )
    x_t = (1-t) * x0 + t * x1 on Euclidean

    u_t = d/dt(geodesic) (changes with curvature) 
    u_t =  x1 - x0 ---> constant velocity everywhere  
    """ 

    batch_size = adj_batch.shape[0] 

    x1 = flatten_adj_to_vec(adj_batch) 
    x1 = x1 * 2 - 1   # {0,1} → {-1,+1} — sharp binary signal 
    x0 = torch.randn_like(x1) 

    # random time 
    t = torch.rand(batch_size, 1, device = DEVICE) 

    # Euclidean interpolation (straight line) 
    x_t = (1-t) * x0 + t * x1 

    # True velocity in flat space 
    u_t = x1 - x0 

    predicted_v = model(x_t, t) 
    loss = torch.mean((predicted_v - u_t)**2) 
    return loss

    
#  generation --> create new graphs from noise.

@torch.no_grad()
def generate_graphs_tacfm(model, num_samples, num_steps = 50): 
    """
    Generate new graphs using the trained TACFM model.

    Start from random noise on S^d. Then walk along the sphere using the 
    learned velocity field. (Euler integration on manifold) 

    The key: after each step, we re-normalize to stay in the sphere.
    This is the Exp map. 
    For spheres, it simplifies to: move + normalize

    Args: 
        num_samples: How many graphs to generate
        num_steps : Number of integration steps ( more = more accurate ) 
    Returns: 
        (batch, 20, 20) numpy array of adjacency matrices 
    """ 

    model.eval() 
    dt = 1.0/ num_steps

    x = torch.randn(num_samples, DATA_DIM, device = DEVICE) 
    x = normalize_to_sphere(x) 

    # Walk along the sphere using learned velocity
    for step in range(num_steps): 
        t_val = step * dt 
        t = torch.full((num_samples, 1), t_val, device = DEVICE) 

        v = model(x, t) 
        v = model.project_to_tangent(x, v) 

        # Euler step on sphere: move + normalize 
        x = x + dt * v 
        x = normalize_to_sphere(x) 

    #  Convert back to adjacency matrices
    adj = vect_to_adj(x, n=MAX_NODES)
    adj = (adj > 0).float()  # positive = edge, negative = no edge
    return adj.cpu().numpy() 

@torch.no_grad()
def generate_graphs_euclidean(model, num_samples, num_steps=50):
    """
    Generate new graphs using trained Euclidean model.
    
    Same Euler integration, but in FLAT space (no normalization step).
    """
    model.eval()
    dt = 1.0 / num_steps
    # Start from Gaussian noise (NOT on sphere)
    x = torch.randn(num_samples, DATA_DIM, device=DEVICE)
    for step in range(num_steps):
        t_val = step * dt
        t = torch.full((num_samples, 1), t_val, device=DEVICE)
        v = model(x, t)
        x = x + v * dt  # plain Euler step, no normalization
    adj = vect_to_adj(x, n=MAX_NODES)
    adj = (adj > 0).float()  # positive = edge, negative = no edge
    return adj.cpu().numpy()
    
    

# Training loop 

def train(args): 

    # Create results directory
    results_dir = f"{args.model.upper()}_{args.arch.upper()}_results"
    os.makedirs(results_dir, exist_ok=True)

    print(f" Training : {args.model.upper()} model (arch: {args.arch})") 
    print(f" Epochs: {args.epochs}") 
    print(f" Device : {DEVICE}") 
    print(f" Results dir: {results_dir}/")

    # load the data

    train_loader, test_adj = load_dataset(args.data_path) 


    if args.model == 'tacfm': 
        if args.arch == 'gcn':
            model = GCN_TACFM(max_nodes=MAX_NODES).to(DEVICE)
        else:
            model = TACFM(DATA_DIM).to(DEVICE) 
        loss_fn = compute_TACFM_loss
        generate_fn = generate_graphs_tacfm
    else: 
        model = EuclideanFM_GraphModel(DATA_DIM).to(DEVICE) 
        loss_fn = compute_euclidean_loss
        generate_fn = generate_graphs_euclidean

    

    param_cnt = sum(p.numel() for p in model.parameters()) 
    print(f" Model parameters: {param_cnt:,}")

    optimizer = optim.Adam(model.parameters(), lr = args.lr) 
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max = args.epochs) 

    loss_history = [] 
    best_loss = float('inf') 


    model.train() 
    start_time = time.time() 

    for epoch in range(args.epochs): 
        epoch_loss = 0

        for (adj_batch,) in train_loader: 
            adj_batch = adj_batch.to(DEVICE) 
            optimizer.zero_grad() 

            loss = loss_fn(model, adj_batch) 
            loss.backward() 
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm = 1.0)

            optimizer.step() 
            epoch_loss += loss.item() 
        scheduler.step()
        avg_loss = epoch_loss / len(train_loader) 
        loss_history.append(avg_loss) 

        if avg_loss < best_loss: 
            best_loss = avg_loss 
            torch.save(model.state_dict(), os.path.join(results_dir, "best_model.pth")) 

        if ( epoch + 1 ) % 50 == 0 or epoch == 0:
            elapsed = time.time() - start_time 
            print(f"Epoch {epoch+1:4d}/{args.epochs} |" 
            f"Loss: {avg_loss:.6f} | "
            f"Best: {best_loss:.6f} | "
            f"Time: {elapsed:.0f}s")
        
    total_time = time.time() - start_time
    print(f"\nTraining completed in {total_time:.1f}s")
    print(f"Best loss: {best_loss:.6f}")

    #  Save the final model
    torch.save(model.state_dict(), os.path.join(results_dir, "final_model.pth"))

    # generate sample graphs
    print(f"Generating 20 sample graphs...")  
    gen_strt = time.time() 
    generate_adj = generate_fn(model, num_samples=20, num_steps=50) 
    gen_time = time.time() - gen_strt 
    print(f"Generated {len(generate_adj)} graphs in {gen_time:.1f}s") 
    np.save(os.path.join(results_dir, "generated_graphs.npy"), generate_adj) 
    

    #  plot training curve 

    plt.figure(figsize=(10,5)) 
    plt.plot(loss_history, label = f"{args.model.upper()} ({args.arch.upper()}) loss")
    plt.xlabel("Epoch") 
    plt.ylabel("Loss(MSE)") 
    plt.title(f"{args.model.upper()} ({args.arch.upper()}) Training progress") 
    plt.legend() 
    plt.grid(True, alpha = 0.3) 
    plt.savefig(os.path.join(results_dir, "training_curve.png"))
    plt.close() 
    print(f"Saved results to {results_dir}/")

    return loss_history


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train TACFM or Euclidean model")
    parser.add_argument('--model', type=str, choices=['tacfm', 'euclidean'],
                        required=True, help='Model type: tacfm or euclidean')
    parser.add_argument('--arch', type=str, choices=['mlp', 'gcn'], default='mlp',
                        help='Architecture: mlp (default) or gcn (graph-aware)')
    parser.add_argument('--epochs', type=int, default=2000,
                        help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='Learning rate')
    parser.add_argument('--data_path', type=str,
                        default='data/community_small.pkl',
                        help='Path to dataset pickle file')
    args = parser.parse_args()
    train(args)
     
    
            
            
            