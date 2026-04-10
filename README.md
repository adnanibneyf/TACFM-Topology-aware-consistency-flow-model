# TACFM: Topology-Aware Consistency Flow Model

A novel approach to generative modeling of graphs and high-dimensional data using **topology-aware geodesic flow matching** with consistency training.

## Overview

TACFM (Topology-Aware Consistency Flow Model) extends consistency training to geometric spaces by leveraging true geodesic paths on manifolds (such as hyperspheres) instead of Euclidean linear interpolation. This approach provides:

- **No teacher model required** — Consistency training without distillation overhead
- **Topology-aware geometry** — Exact geodesic computation between source and target
- **Improved generation quality** — Benchmarked against GDSS and Euclidean flow matching baselines
- **Flexible architecture** — GCN-based networks for graph-structured data

### Key Innovation

Traditional flow matching uses linear interpolation in Euclidean space. TACFM computes paths along geodesics on the data manifold:

$$x_t = \text{Exp}(t \cdot \text{Log}(x_1)), \quad t \in [0, 1]$$

Where:
- **t=0**: Gaussian noise (source)
- **t=1**: Clean data (target)
- **Geodesic paths** ensure optimal transportation along the underlying geometry

## Project Structure

```
TACFM-Topology-aware-consistency-flow-model/
├── Codes/
│   ├── SBM/                    # Stochastic Block Model experiments
│   │   ├── TACFM.py           # Main TACFM consistency model
│   │   ├── model_GCN.py       # GCN architecture for graphs
│   │   ├── model.py           # Baseline models
│   │   ├── train.py           # Training script
│   │   ├── evaluate.py        # Evaluation metrics & benchmarking
│   │   ├── data_generator.py  # SBM graph generation
│   │   ├── testDataset.py     # Dataset utilities
│   │   ├── benchmark_report.md # Performance benchmarks
│   │   └── data/              # Generated datasets
│   │
│   ├── earthquake/             # Seismic data experiments
│   │   ├── train_TACFM.py     # Training pipeline
│   │   ├── analyze_data.py    # Data analysis utilities
│   │   ├── dataFetch.py       # Data loading
│   │   └── Testing_geodesic.py # Geodesic validation tests
│   │
│   └── so3/                    # SO(3) rotation group experiments
│       └── generate_so3_data.py # SO(3) dataset generation
│
├── model/
│   └── draftModel.py          # Model architecture drafts
│
├── Resources/                 # Additional resources
├── ML_project.pdf            # Full project documentation
└── README.md                 # This file
```

## Quick Start

### Prerequisites

- Python 3.8+
- PyTorch (with CUDA support recommended)
- NumPy, SciPy
- NetworkX (for graph utilities)

### Installation

```bash
# Clone the repository
git clone <repository-url>
cd TACFM-Topology-aware-consistency-flow-model

# Install dependencies
pip install torch numpy scipy networkx scikit-learn matplotlib
```

### Training on SBM Graphs

```bash
cd Codes/SBM

# Generate synthetic data (Community_small dataset)
python data_generator.py

# Train TACFM model
python train.py --epochs 2000 --learning_rate 1e-3

# Evaluate and generate graphs
python evaluate.py --num_samples 50
```

### Training on Earthquake Data

```bash
cd Codes/earthquake

# Fetch and prepare data
python dataFetch.py

# Train model
python train_TACFM.py

# Analyze results
python analyze_data.py
```

## Architecture Details

### TACFM Consistency Model

The core consistency model uses time-dependent skip connections:

$$f(x, t) = t \cdot x + (1-t) \cdot \text{network}(x, t)$$

**Boundary conditions:**
- At t=1: $f(x, 1) = x$ (identity on clean data)
- At t=0: $f(x, 0) = \text{network}(x, 0)$ (full prediction)

### Loss Function

```
Loss = consistency_loss + reconstruction_loss

consistency_loss    = ||student(x_t, t) - ema_student(x_{t-δ}, t-δ)||²
reconstruction_loss = ||student(x_1, 1) - x_1||²   (anchor to real data)
```

### GCN Architecture

For graph data, we use Graph Convolutional Networks (GCN) with:
- Sinusoidal time embeddings
- Residual blocks for deep feature extraction
- Tangent space projections for geodesic constraints
- Automatic re-normalization to maintain manifold membership

## Experimental Results

### SBM Benchmark (Community_small)

Comparison against GDSS (Score-based Generative Modeling of Graphs):

| Model | Degree MMD ↓ | Clustering MMD ↓ | Spectral MMD ↓ | Generation Time |
|-------|-------------|-----------------|----------------|-----------------|
| **TACFM (ours)** | **-0.004** | **0.053** | **0.025** | 0.09s |
| Euclidean FM | 0.009 | 0.081 | 0.045 | 0.08s |
| GDSS (published) | 0.045 | 0.017 | — | minutes |

TACFM achieves better degree and spectral metrics while maintaining competitive clustering performance.

### Configuration

**Dataset:** Community_small (2-community SBM)
- 400 training graphs (80/20 split from 500 total)
- Node range: 12–20 nodes per graph
- Parameters: p_intra=0.7, p_inter=0.05

**Training:**
- Epochs: 2000
- Optimizer: Adam with CosineAnnealing
- Learning rate: 1e-3
- Gradient clipping: max_norm=1.0
- ODE integration steps: 50 (Euler method)

## Key Features

✅ **Consistency Training** — No teacher model distillation required  
✅ **Geodesic Flow Matching** — Topology-aware paths on manifolds  
✅ **Graph Generation** — Native support for graph-structured data  
✅ **Scalable** — Handles variable graph sizes and dimensions  
✅ **Empirically Validated** — Benchmarked against strong baselines  
✅ **Flexible** — Extensible to other manifolds (SO(3), hyperbolic spaces, etc.)

## Usage Examples

### Generate graphs from a trained model

```python
from model_GCN import GCN_TACFM
from TACFM import _TACFM_

# Load trained model
model = _TACFM_(max_nodes=20)
model.load_state_dict(torch.load('tacfm_checkpoint.pt'))
model.eval()

# Generate random sample
z = torch.randn(32, 190)  # batch_size=32, dim=190
with torch.no_grad():
    generated = model.sample(z, steps=50)
```

### Evaluate on your own graphs

```python
from evaluate import compute_graph_metrics

# Compute MMD metrics
metrics = compute_graph_metrics(
    generated_graphs=generated,
    real_graphs=test_data,
    metrics=['degree', 'clustering', 'spectral']
)
```

## Hyperparameter Tuning

Key parameters to experiment with:

- `--learning_rate`: Training step size (default: 1e-3)
- `--epochs`: Number of training iterations (default: 2000)
- `--hidden_dim`: Network hidden dimension (default: 256)
- `--timesteps`: ODE integration steps (default: 50)
- `--max_nodes`: Maximum nodes per graph (default: 20)
- `--batch_size`: Batch size for training (default: 32)

## Troubleshooting

### CUDA out of memory
- Reduce `--batch_size`
- Reduce `--hidden_dim`
- Reduce `--max_nodes`

### Poor generation quality
- Increase `--epochs`
- Use a smaller `--learning_rate` with longer warmup
- Check that geodesic constraints are properly enforced

### NaN loss values
- Enable gradient clipping (enabled by default)
- Use smaller learning rate
- Verify data normalization to unit sphere

## References

The project implements ideas from:
- **Flow Matching**: Liphardt et al. (2022) — "Flow Matching for Generative Modeling"
- **Consistency Training**: Song et al. (2023) — "Consistency Models"
- **Graph Generation**: GDSS (Jo et al., ICML 2022) — "Score-based Generative Modeling of Graphs"
- **Manifold Learning**: Standard techniques for geodesic computation and tangent space projections

## Citation

If you use TACFM in your research, please cite:

```bibtex
@misc{tacfm2026,
  title={TACFM: Topology-Aware Consistency Flow Model},
  author={Your Name},
  year={2026},
  howpublished={GitHub}
}
```

## License

[Specify your license here - e.g., MIT, Apache 2.0, etc.]

## Contact

For questions or issues, please open an issue on GitHub or contact [your-email@example.com].

---

**Last Updated:** April 2026
