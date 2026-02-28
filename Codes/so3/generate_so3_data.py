import numpy as np
import torch
from scipy.spatial.transform import Rotation as R
import matplotlib.pyplot as plt

# --- CONFIGURATION ---
NUM_SAMPLES = 10000
NUM_MODES = 4        # We will create 4 distinct clusters of rotations
SIGMA = 0.2          # How "spread out" each cluster is (Noise level)
OUTPUT_FILE = "../dataset/so3_synthetic_dataset.npy"

def generate_so3_data():
    print(f"--- GENERATING SO(3) DATASET ({NUM_SAMPLES} samples) ---")
    
    # 1. Define Centers (The "Truth")
    # We pick 4 random distinct rotations to be the centers of our clusters
    center_rots = R.random(NUM_MODES).as_matrix() # Shape: (4, 3, 3)
    
    data = []
    labels = []
    
    # 2. Generate Noisy Samples around Centers
    samples_per_mode = NUM_SAMPLES // NUM_MODES
    
    for i in range(NUM_MODES):
        center = center_rots[i]
        
        # Generate random "tangent" vectors (small perturbations)
        # In SO(3), the tangent space is skew-symmetric matrices (lie algebra so3)
        # We simulate this by generating random axis-angle vectors
        noise_vecs = np.random.randn(samples_per_mode, 3) * SIGMA
        
        # Convert noise vectors to rotations (Exponential Map)
        noise_rots = R.from_rotvec(noise_vecs).as_matrix()
        
        # Apply noise to the center (Matrix Multiplication)
        # new_rot = center * noise
        cluster_samples = np.matmul(center, noise_rots)
        
        data.append(cluster_samples)
        labels.extend([i] * samples_per_mode)
        
    # Concatenate all clusters
    all_data = np.concatenate(data, axis=0) # Shape: (N, 3, 3)
    
    # Shuffle
    indices = np.random.permutation(len(all_data))
    all_data = all_data[indices]
    
    # 3. Save
    np.save(OUTPUT_FILE, all_data)
    print(f"Saved dataset to '{OUTPUT_FILE}'")
    print(f"Shape: {all_data.shape}")
    
    # 4. Visualization (Project to 3D for sanity check)
    # We visualize by converting to Rotation Vectors (Axis-Angle)
    # This maps the 9D matrix down to a 3D ball so we can see the clusters.
    r_obj = R.from_matrix(all_data[:2000]) # Plot first 2000
    rot_vecs = r_obj.as_rotvec()
    
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Color by cluster
    colors = ['red', 'blue', 'green', 'purple']
    # Note: We lost the labels during shuffle, so just plotting all as one color
    # to show the 'manifold structure'. 
    # If you want colored clusters, plot before shuffle.
    
    ax.scatter(rot_vecs[:,0], rot_vecs[:,1], rot_vecs[:,2], s=1, alpha=0.5)
    ax.set_title(f"Visualizing SO(3) Data\n(Projected to 3D Axis-Angle Space)")
    ax.set_xlabel("X (Roll)")
    ax.set_ylabel("Y (Pitch)")
    ax.set_zlabel("Z (Yaw)")
    plt.savefig("so3_data_viz.png")
    plt.show()

if __name__ == "__main__":
    generate_so3_data()