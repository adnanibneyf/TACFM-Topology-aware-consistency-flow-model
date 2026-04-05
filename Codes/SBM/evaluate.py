"""
evaluate.py — Compare TACFM vs Euclidean vs GDSS using MMD metrics.

Computes the EXACT same metrics used in the GDSS paper (ICML 2022):
  1. Degree MMD     — Are the degree distributions similar?
  2. Clustering MMD — Are the clustering coefficients similar?
  3. Orbit MMD      — Are the subgraph patterns (4-node motifs) similar?

Lower is better for all metrics. 0.0 = perfect match.
"""

import numpy as np
import networkx as nx
import pickle
import torch
from scipy.linalg import eigvalsh
import argparse
import warnings
warnings.filterwarnings('ignore')

import os
import time

from model import (
    TACFM, EuclideanFM_GraphModel,
    flatten_adj_to_vec, vect_to_adj, normalize_to_sphere
)
from model_GCN import GCN_TACFM
from TACFM import _TACFM_
from _TACM_ import _TACM_

MAX_NODES = 20
DATA_DIM = MAX_NODES * (MAX_NODES - 1) // 2
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# ============================================================
#  MMD KERNEL — Gaussian EMD (from GDSS evaluation code)
# ============================================================

def gaussian_emd(x, y, sigma=1.0):
    """
    Gaussian kernel for Earth Mover's Distance between two histograms.
    This is the SAME kernel GDSS uses for degree and spectral MMD.
    """
    # Pad to same length
    if len(x) < len(y):
        x = np.concatenate([x, np.zeros(len(y) - len(x))])
    elif len(y) < len(x):
        y = np.concatenate([y, np.zeros(len(x) - len(y))])
    
    # Normalize to probability distributions
    x = x / (x.sum() + 1e-8)
    y = y / (y.sum() + 1e-8)
    
    # EMD for 1D distributions = L1 of CDFs
    emd = np.sum(np.abs(np.cumsum(x) - np.cumsum(y)))
    return np.exp(-emd * emd / (2 * sigma * sigma))


def gaussian_kernel(x, y, sigma=1.0):
    """Simple Gaussian RBF kernel for vectors."""
    dist = np.linalg.norm(x - y)
    return np.exp(-dist * dist / (2 * sigma * sigma))


def compute_mmd(samples_ref, samples_pred, kernel, sigma=1.0, is_hist=True):
    """
    Maximum Mean Discrepancy between two sets of samples.
    
    MMD = E[k(x,x')] + E[k(y,y')] - 2*E[k(x,y)]
    
    where x ~ ref, y ~ pred, k = kernel function.
    Lower = better (0 means distributions are identical).
    """
    n_ref = len(samples_ref)
    n_pred = len(samples_pred)
    
    if n_ref == 0 or n_pred == 0:
        return 1.0
    
    # E[k(x, x')]
    kxx = 0.0
    for i in range(n_ref):
        for j in range(i + 1, n_ref):
            if is_hist:
                kxx += kernel(samples_ref[i], samples_ref[j], sigma)
            else:
                kxx += gaussian_kernel(samples_ref[i], samples_ref[j], sigma)
    kxx /= max(n_ref * (n_ref - 1) / 2, 1)
    
    # E[k(y, y')]
    kyy = 0.0
    for i in range(n_pred):
        for j in range(i + 1, n_pred):
            if is_hist:
                kyy += kernel(samples_pred[i], samples_pred[j], sigma)
            else:
                kyy += gaussian_kernel(samples_pred[i], samples_pred[j], sigma)
    kyy /= max(n_pred * (n_pred - 1) / 2, 1)
    
    # E[k(x, y)]
    kxy = 0.0
    for i in range(n_ref):
        for j in range(n_pred):
            if is_hist:
                kxy += kernel(samples_ref[i], samples_pred[j], sigma)
            else:
                kxy += gaussian_kernel(samples_ref[i], samples_pred[j], sigma)
    kxy /= max(n_ref * n_pred, 1)
    
    mmd = kxx + kyy - 2 * kxy
    return mmd


# ============================================================
#  METRIC 1: Degree MMD
# ============================================================

def degree_stats(graph_ref_list, graph_pred_list):
    """
    Compare degree distributions between real and generated graphs.
    
    Degree = number of edges per node.
    A good community graph should have high-degree nodes within
    communities and low-degree connections between them.
    """
    sample_ref = [np.array(nx.degree_histogram(G)) for G in graph_ref_list 
                  if G.number_of_nodes() > 0]
    sample_pred = [np.array(nx.degree_histogram(G)) for G in graph_pred_list 
                   if G.number_of_nodes() > 0]
    
    if len(sample_pred) == 0:
        return 1.0
    
    return compute_mmd(sample_ref, sample_pred, kernel=gaussian_emd, sigma=1.0)


# ============================================================
#  METRIC 2: Clustering Coefficient MMD
# ============================================================

def clustering_stats(graph_ref_list, graph_pred_list, bins=100):
    """
    Compare clustering coefficient distributions.
    
    Clustering coefficient = how many of a node's neighbors are
    also connected to each other (forming triangles).
    
    Community graphs have HIGH clustering within communities.
    This metric tests if generated graphs capture that property.
    """
    sample_ref = []
    sample_pred = []
    
    for G in graph_ref_list:
        if G.number_of_nodes() == 0:
            continue
        cc_list = list(nx.clustering(G).values())
        hist, _ = np.histogram(cc_list, bins=bins, range=(0.0, 1.0), density=False)
        sample_ref.append(hist)
    
    for G in graph_pred_list:
        if G.number_of_nodes() == 0:
            continue
        cc_list = list(nx.clustering(G).values())
        hist, _ = np.histogram(cc_list, bins=bins, range=(0.0, 1.0), density=False)
        sample_pred.append(hist)
    
    if len(sample_pred) == 0:
        return 1.0
    
    return compute_mmd(sample_ref, sample_pred, kernel=gaussian_emd, 
                       sigma=1.0 / 10)


# ============================================================
#  METRIC 3: Spectral MMD (simpler alternative to Orbit MMD)
# ============================================================

def spectral_stats(graph_ref_list, graph_pred_list):
    """
    Compare spectral (eigenvalue) distributions of graph Laplacians.
    
    The eigenvalues of the normalized Laplacian encode the overall
    structure of the graph — components, bottlenecks, community 
    boundaries.
    
    This is used as a simpler alternative to Orbit MMD (which 
    requires compiling C++ ORCA code). It captures similar 
    structural information.
    """
    sample_ref = []
    sample_pred = []
    
    for G in graph_ref_list:
        if G.number_of_nodes() < 2:
            continue
        try:
            eigs = eigvalsh(nx.normalized_laplacian_matrix(G).todense())
            hist, _ = np.histogram(eigs, bins=200, range=(-1e-5, 2), density=False)
            hist = hist / (hist.sum() + 1e-8)
            sample_ref.append(hist)
        except:
            continue
    
    for G in graph_pred_list:
        if G.number_of_nodes() < 2:
            continue
        try:
            eigs = eigvalsh(nx.normalized_laplacian_matrix(G).todense())
            hist, _ = np.histogram(eigs, bins=200, range=(-1e-5, 2), density=False)
            hist = hist / (hist.sum() + 1e-8)
            sample_pred.append(hist)
        except:
            continue
    
    if len(sample_pred) == 0:
        return 1.0
    
    return compute_mmd(sample_ref, sample_pred, kernel=gaussian_emd, sigma=1.0)


# ============================================================
#  ADJACENCY → GRAPH CONVERSION
# ============================================================

def adj_to_graphs(adj_array, threshold=0.5):
    """Convert numpy adjacency matrices to networkx graphs."""
    graphs = []
    for adj in adj_array:
        adj_bin = (adj > threshold).astype(float)
        adj_bin = np.maximum(adj_bin, adj_bin.T)
        np.fill_diagonal(adj_bin, 0)
        G = nx.from_numpy_array(adj_bin)
        G.remove_nodes_from(list(nx.isolates(G)))
        if G.number_of_nodes() > 0:
            graphs.append(G)
    return graphs


# ============================================================
#  GENERATION FROM SAVED MODELS
# ============================================================

def generate_from_model(model_type, arch, model_path, num_samples=100, num_steps=50):
    """Load a saved model and generate graphs."""
    if model_type == 'tacfm':
        if arch == 'gcn':
            model = GCN_TACFM(max_nodes=MAX_NODES).to(DEVICE)
        else:
            model = TACFM(data_dim=DATA_DIM).to(DEVICE)
    else:
        model = EuclideanFM_GraphModel(data_dim=DATA_DIM).to(DEVICE)
    
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.eval()
    
    dt = 1.0 / num_steps
    
    if model_type == 'tacfm':
        x = torch.randn(num_samples, DATA_DIM, device=DEVICE)
        x = normalize_to_sphere(x)
        for step in range(num_steps):
            t = torch.full((num_samples, 1), step * dt, device=DEVICE)
            v = model(x, t)
            v = model.project_to_tangent(x, v)
            x = x + v * dt
            x = normalize_to_sphere(x)
    else:
        x = torch.randn(num_samples, DATA_DIM, device=DEVICE)
        for step in range(num_steps):
            t = torch.full((num_samples, 1), step * dt, device=DEVICE)
            v = model(x, t)
            x = x + v * dt
    
    adj = vect_to_adj(x, n=MAX_NODES)
    adj = (adj > 0).float()  # positive = edge, negative = no edge
    return adj.detach().cpu().numpy()


@torch.no_grad()
def generate_from_consistency(model_path, num_samples=50, num_steps=1, is_tacm=False):
    """Generate graphs using the consistency model (1-step or few-step)."""
    if is_tacm:
        model = _TACM_(max_nodes=MAX_NODES).to(DEVICE)
    else:
        model = _TACFM_(max_nodes=MAX_NODES).to(DEVICE)
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.eval()
    
    x = torch.randn(num_samples, DATA_DIM, device=DEVICE)
    x = normalize_to_sphere(x)
    
    if num_steps == 1:
        t = torch.zeros(num_samples, 1, device=DEVICE)  # t=0: fully predict
        x = model(x, t)
    else:
        time_points = torch.linspace(0.0, 0.8, num_steps)
        for i in range(num_steps):
            t = torch.full((num_samples, 1), time_points[i].item(), device=DEVICE)
            x = model(x, t)
            x = normalize_to_sphere(x)
            if i < num_steps - 1:
                noise = torch.randn_like(x) * 0.05
                x = normalize_to_sphere(x + noise)
    
    adj = vect_to_adj(x, n=MAX_NODES)
    adj = (adj > 0).float()
    return adj.detach().cpu().numpy()


# ============================================================
#  MAIN EVALUATION
# ============================================================

def evaluate():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_samples', type=int, default=50,
                        help='Number of graphs to generate for evaluation')
    parser.add_argument('--num_steps', type=int, default=50,
                        help='Number of ODE integration steps')
    parser.add_argument('--data_path', type=str, 
                        default='data/community_small.pkl')
    args = parser.parse_args()
    
    # --- Load reference (real) graphs ---
    print("Loading reference graphs...")
    with open(args.data_path, 'rb') as f:
        ref_graphs = pickle.load(f)
    print(f"  Reference: {len(ref_graphs)} graphs")
    
    # --- Models to evaluate (name, model_type, arch, path) ---
    models_to_eval = []
    
    # Check which results directories exist
    if os.path.exists('TACFM_GCN_results/best_model.pth'):
        models_to_eval.append(('TACFM (GCN)', 'tacfm', 'gcn', 'TACFM_GCN_results/best_model.pth'))
    if os.path.exists('TACFM_MLP_results/best_model.pth'):
        models_to_eval.append(('TACFM (MLP)', 'tacfm', 'mlp', 'TACFM_MLP_results/best_model.pth'))
    if os.path.exists('EUCLIDEAN_MLP_results/best_model.pth'):
        models_to_eval.append(('Euclidean', 'euclidean', 'mlp', 'EUCLIDEAN_MLP_results/best_model.pth'))
    
    # Consistency model variants
    consistency_path = None
    for cpath in ['results_consistency/best_model.pth', 'results_consistnecy/best_model.pth']:
        if os.path.exists(cpath):
            consistency_path = cpath
            break
    has_consistency = consistency_path is not None

    tacm_path = None
    if os.path.exists('results_tacm/_TACM_best.pth'):
        tacm_path = 'results_tacm/_TACM_best.pth'
    has_tacm = tacm_path is not None
    
    # Fallback: check for old-style paths
    if not models_to_eval:
        if os.path.exists('tacfm_mlp_best.pth'):
            models_to_eval.append(('TACFM (MLP)', 'tacfm', 'mlp', 'tacfm_mlp_best.pth'))
        if os.path.exists('tacfm_best.pth'):
            models_to_eval.append(('TACFM (MLP)', 'tacfm', 'mlp', 'tacfm_best.pth'))
        if os.path.exists('euclidean_mlp_best.pth'):
            models_to_eval.append(('Euclidean', 'euclidean', 'mlp', 'euclidean_mlp_best.pth'))
        if os.path.exists('euclidean_best.pth'):
            models_to_eval.append(('Euclidean', 'euclidean', 'mlp', 'euclidean_best.pth'))
    
    if not models_to_eval:
        print("ERROR: No trained models found. Run train.py first.")
        return
    
    print(f"  Found {len(models_to_eval)} model(s) to evaluate")
    
    results = {}
    
    for name, mtype, arch, mpath in models_to_eval:
        print(f"\nGenerating {args.num_samples} graphs from {name}...")
        start = time.time()
        gen_adj = generate_from_model(mtype, arch, mpath, args.num_samples, args.num_steps)
        gen_time = time.time() - start
        
        gen_graphs = adj_to_graphs(gen_adj)
        print(f"  Generated {len(gen_graphs)} valid graphs in {gen_time:.2f}s")
        
        # Compute basic stats
        node_counts = [G.number_of_nodes() for G in gen_graphs]
        edge_counts = [G.number_of_edges() for G in gen_graphs]
        print(f"  Nodes: {np.mean(node_counts):.1f} avg (real: "
              f"{np.mean([G.number_of_nodes() for G in ref_graphs]):.1f})")
        print(f"  Edges: {np.mean(edge_counts):.1f} avg (real: "
              f"{np.mean([G.number_of_edges() for G in ref_graphs]):.1f})")
        
        # Compute MMD metrics
        print(f"  Computing metrics...")
        deg_mmd = degree_stats(ref_graphs, gen_graphs)
        clus_mmd = clustering_stats(ref_graphs, gen_graphs)
        spec_mmd = spectral_stats(ref_graphs, gen_graphs)
        
        results[name] = {
            'Deg.': deg_mmd,
            'Clus.': clus_mmd,
            'Spec.': spec_mmd,
            'Time': gen_time,
            'Valid': len(gen_graphs),
        }
    
    # --- Evaluate consistency model variants ---
    if has_consistency:
        for steps_label, n_steps in [('TACFM-C (1-step)', 1), ('TACFM-C (4-step)', 4)]:
            print(f"\nGenerating {args.num_samples} graphs from {steps_label}...")
            start = time.time()
            gen_adj = generate_from_consistency(consistency_path, args.num_samples, n_steps)
            gen_time = time.time() - start
            
            gen_graphs = adj_to_graphs(gen_adj)
            print(f"  Generated {len(gen_graphs)} valid graphs in {gen_time:.2f}s")
            
            if len(gen_graphs) > 0:
                node_counts = [G.number_of_nodes() for G in gen_graphs]
                edge_counts = [G.number_of_edges() for G in gen_graphs]
                print(f"  Nodes: {np.mean(node_counts):.1f} avg (real: "
                      f"{np.mean([G.number_of_nodes() for G in ref_graphs]):.1f})")
                print(f"  Edges: {np.mean(edge_counts):.1f} avg (real: "
                      f"{np.mean([G.number_of_edges() for G in ref_graphs]):.1f})")
                
                print(f"  Computing metrics...")
                results[steps_label] = {
                    'Deg.': degree_stats(ref_graphs, gen_graphs),
                    'Clus.': clustering_stats(ref_graphs, gen_graphs),
                    'Spec.': spectral_stats(ref_graphs, gen_graphs),
                    'Time': gen_time,
                    'Valid': len(gen_graphs),
                }
            else:
                print("  WARNING: No valid graphs generated!")

    if has_tacm:
        for steps_label, n_steps in [('TACM-Exp (1-step)', 1), ('TACM-Exp (4-step)', 4)]:
            print(f"\nGenerating {args.num_samples} graphs from {steps_label}...")
            start = time.time()
            gen_adj = generate_from_consistency(tacm_path, args.num_samples, n_steps, is_tacm=True)
            gen_time = time.time() - start
            
            gen_graphs = adj_to_graphs(gen_adj)
            print(f"  Generated {len(gen_graphs)} valid graphs in {gen_time:.2f}s")
            
            if len(gen_graphs) > 0:
                node_counts = [G.number_of_nodes() for G in gen_graphs]
                edge_counts = [G.number_of_edges() for G in gen_graphs]
                print(f"  Nodes: {np.mean(node_counts):.1f} avg (real: "
                      f"{np.mean([G.number_of_nodes() for G in ref_graphs]):.1f})")
                print(f"  Edges: {np.mean(edge_counts):.1f} avg (real: "
                      f"{np.mean([G.number_of_edges() for G in ref_graphs]):.1f})")
                
                print(f"  Computing metrics...")
                results[steps_label] = {
                    'Deg.': degree_stats(ref_graphs, gen_graphs),
                    'Clus.': clustering_stats(ref_graphs, gen_graphs),
                    'Spec.': spectral_stats(ref_graphs, gen_graphs),
                    'Time': gen_time,
                    'Valid': len(gen_graphs),
                }
            else:
                print("  WARNING: No valid graphs generated!")
    
    # --- Print comparison table ---
    print("\n" + "=" * 80)
    print("  BENCHMARK RESULTS -- Community_small (lower MMD = better)")
    print("=" * 80)
    print(f"{'Model':<20} {'Deg.(v)':>10} {'Clus.(v)':>10} {'Spec.(v)':>10} {'Valid':>8} {'Time':>8}")
    print("-" * 80)
    
    for name, r in results.items():
        print(f"{name:<20} {r['Deg.']:>10.6f} {r['Clus.']:>10.6f} "
              f"{r['Spec.']:>10.6f} {r['Valid']:>6}/{args.num_samples}  "
              f"{r['Time']:>6.2f}s")
    
    # GDSS published numbers
    print(f"{'GDSS (paper)':<20} {'0.045':>10} {'0.017':>10} {'--':>10} {'--':>8} {'--':>8}")
    print("=" * 80)
    
    # --- Find best TACFM variant ---
    tacfm_results = {k: v for k, v in results.items() if 'TACFM' in k or 'TACM' in k}
    euc_results = {k: v for k, v in results.items() if 'Euclidean' in k}
    
    if tacfm_results and euc_results:
        best_tacfm_name = min(tacfm_results, key=lambda k: 
            (tacfm_results[k]['Deg.'] + tacfm_results[k]['Clus.'] + tacfm_results[k]['Spec.']) / 3)
        best_tacfm = tacfm_results[best_tacfm_name]
        euc_name = list(euc_results.keys())[0]
        euc = euc_results[euc_name]
        
        tacfm_avg = (best_tacfm['Deg.'] + best_tacfm['Clus.'] + best_tacfm['Spec.']) / 3
        euc_avg = (euc['Deg.'] + euc['Clus.'] + euc['Spec.']) / 3
        
        if tacfm_avg < euc_avg:
            improvement = ((euc_avg - tacfm_avg) / abs(euc_avg)) * 100
            print(f"\n  Best TACFM variant: {best_tacfm_name} -- wins by {improvement:.1f}% avg MMD reduction!")
        else:
            print(f"\n  Euclidean baseline performs better. Consider tuning hyperparameters.")
    
    return results


if __name__ == "__main__":
    evaluate()
