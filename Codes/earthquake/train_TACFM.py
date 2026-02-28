import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# This imports the class you uploaded in 'draftModel.py'
from draftModel import TACFM_Model

# --- CONFIGURATION ---
BATCH_SIZE = 16          
LEARNING_RATE = 1e-3     # Standard starting rate
EPOCHS = 200             # Enough to learn the flow
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
CSV_FILE = 'noaa_earthquakes_2000_2025.csv' 

class EarthquakeDataset(Dataset):
    def __init__(self, csv_file):
        try:
            df = pd.read_csv(csv_file)
            df = df.head(200) 
            
            lats = df['latitude'].values
            lons = df['longitude'].values
            
            # Convert Lat/Lon to 3D Cartesian (x,y,z) on Sphere
            self.data = self.latlon_to_xyz(lats, lons)
            self.data = torch.tensor(self.data, dtype=torch.float32)
            print(f"Loaded {len(self.data)} earthquake events.")
            
        except Exception as e:
            print(f"Error loading CSV: {e}")
            self.data = torch.randn(10, 3) # Fallback if file fails

    def latlon_to_xyz(self, lats, lons):
        # Degrees -> Radians
        lat_rad = np.deg2rad(lats)
        lon_rad = np.deg2rad(lons)
        # Spherical -> Cartesian
        x = np.cos(lat_rad) * np.cos(lon_rad)
        y = np.cos(lat_rad) * np.sin(lon_rad)
        z = np.sin(lat_rad)
        return np.stack([x, y, z], axis=1)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

# --- 2. THE GEOMETRIC LOSS FUNCTION ---
def compute_riemannian_loss(model, x1):
    """
    This part calculates the 'Geodesic Flow' and forces the model to match it.
    
    x1: The Real Earthquake Data (Target)
    """
    batch_size = x1.shape[0]
    
    # A. Sample Random Noise (x0) on Sphere
    x0 = torch.randn_like(x1)
    x0 = x0 / x0.norm(dim=1, keepdim=True)
    
    # B. Sample Random Time t [0, 1]
    # Shape needs to be (Batch, 1) for the model
    t = torch.rand(batch_size, 1).to(DEVICE)
    
    # --- MATH: SPHERICAL GEODESICS ---
    # We compute the path from x0 -> x1 manually here.
    
    # 1. Calculate Angle (Theta) between Noise and Data
    # Dot product clamped to handle numerical errors
    cos_theta = (x0 * x1).sum(dim=1, keepdim=True).clamp(-1 + 1e-6, 1 - 1e-6)
    theta = torch.acos(cos_theta)
    sin_theta = torch.sqrt(1 - cos_theta**2) + 1e-6 # Avoid div by zero
    
    # 2. Calculate Initial Velocity (v0)
    # This is the "Log Map" (Direction to shoot)
    # Formula: v0 = (x1 - x0 * cos_theta) * (theta / sin_theta)
    v0 = (x1 - x0 * cos_theta) * (theta / sin_theta)
    
    # 3. Calculate Point at time t (x_t)
    # Formula: Geodesic interpolation
    # direction_vector = v0 / theta
    dir_vector = v0 / (theta + 1e-6)
    x_t = x0 * torch.cos(theta * t) + dir_vector * torch.sin(theta * t)
    
    # 4. Calculate TARGET Velocity at time t (u_t)
    # Formula: Derivative of the Geodesic (Parallel Transport of v0)
    u_t = -x0 * theta * torch.sin(theta * t) + dir_vector * theta * torch.cos(theta * t)
    
    # --- MODEL PREDICTION ---
    # The model tries to guess u_t knowing only x_t and t
    predicted_velocity = model(x_t, t)
    
    # Constraint: Project prediction to Tangent Space
    # (Use the function inside your model class)
    pred_tangent = model.project_to_tangent(x_t, predicted_velocity)
    
    # --- LOSS ---
    # Mean Squared Error between Prediction and Geometric Truth
    loss = torch.mean((pred_tangent - u_t)**2)
    
    return loss

# --- 3. TRAINING LOOP ---
def train():
    # Setup
    dataset = EarthquakeDataset(CSV_FILE)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    model = TACFM_Model(data_dim=3, time_dim=32, hidden_dim=128).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    loss_history = []
    
    print("\nStarting Training Loop...")
    model.train()
    
    for epoch in range(EPOCHS):
        total_loss = 0
        for x1 in dataloader:
            x1 = x1.to(DEVICE)
            
            optimizer.zero_grad()
            
            # Calculate Loss (The Math)
            loss = compute_riemannian_loss(model, x1)
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        avg_loss = total_loss / len(dataloader)
        loss_history.append(avg_loss)
        
        if (epoch+1) % 20 == 0:
            print(f"Epoch {epoch+1}/{EPOCHS} | Loss: {avg_loss:.6f}")

    # Save Results
    torch.save(model.state_dict(), "tacfm_earthquake_model.pth")
    print("\nTRAINING COMPLETE.")
    print("Model saved as 'tacfm_earthquake_model.pth'")
    
    # Plot Training Curve
    plt.figure(figsize=(8,5))
    plt.plot(loss_history, label='Flow Matching Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss (MSE)')
    plt.title('TACFM Training Progress\n(Learning to flow from Noise -> Earthquakes)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('training_loss_curve.png')
    plt.show()

if __name__ == "__main__":
    train()