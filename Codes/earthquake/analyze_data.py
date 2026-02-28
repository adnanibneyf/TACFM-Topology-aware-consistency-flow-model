import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# --- CONFIGURATION ---
CSV_FILE = 'noaa_earthquakes_2000_2025.csv'


def convert_to_xyz(lats, lons): 
    lat_rad = np.deg2rad(lats)
    lon_rad = np.deg2rad(lons)

    x = np.cos(lat_rad) * np.cos(lon_rad)
    y = np.cos(lat_rad) * np.sin(lon_rad)  
    z = np.sin(lat_rad)

    return np.stack([x,y,z], axis=1)

def analyze():
    print(f"Loading data from {CSV_FILE}...")

    try:
        df = pd.read_csv(CSV_FILE)
        print(f"Total rows: {len(df)}")
        print("Sample data:")
        print(df.head())
    except FileNotFoundError:
        print(f"Error: File '{CSV_FILE}' not found. Please run dataFetch.py first to download the data.")
        return
    
    print("\n--- DATA QUALITY CHECK ---")
    # Check for missing values
    missing = df.isnull().sum()
    print("Missing Values per Column:")
    print(missing[missing > 0])

    print("\n--- GEOMETRIC VALIDITY CHECK ---")

    if 'latitude' in df.columns and 'longitude' in df.columns:
        lats = df['latitude'].values
        lons = df['longitude'].values

        data_xyz = convert_to_xyz(lats, lons)
        norms = np.linalg.norm(data_xyz, axis=1)

        print(f"Mean Radius: {np.mean(norms):.4f} (Should be close to 1.0)")
        print(f"Min Radius: {np.min(norms):.4f}, Max Radius: {np.max(norms):.4f}")

        invalid_count = np.sum(np.abs(norms - 1.0) > 0.01)
        if invalid_count == 0 : 
            print("All points are valid on the unit sphere.")
        else:
            print(f"Warning: {invalid_count} points deviate significantly from the unit sphere.")

    else: 
        print("Error: 'latitude' and 'longitude' columns not found in the data.")
        return
    
    print("\n--- VISUALIZATIONS ---")

    plt.figure(figsize=(12,6))
    plt.scatter(df['longitude'], df['latitude'], alpha=0.5, s=10)
    plt.title(f"Earthquake Distribution (2D Projection) - Total: {len(df)}")
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.grid(True, alpha=0.3)
    plt.axhline(0, color='black', linewidth=0.5) # Equator
    plt.axvline(0, color='black', linewidth=0.5) # Prime Meridian

    plt.savefig("analysis_2d_map.png")

    # 3D Scatter Plot on Sphere
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Draw wireframe Earth
    u, v = np.mgrid[0:2*np.pi:30j, 0:np.pi:15j]
    earth_x = np.cos(u)*np.sin(v)
    earth_y = np.sin(u)*np.sin(v)
    earth_z = np.cos(v)
    ax.plot_wireframe(earth_x, earth_y, earth_z, color="gray", alpha=0.1)
    
    # Plot Data
    ax.scatter(data_xyz[:,0], data_xyz[:,1], data_xyz[:,2], s=20, c='red', alpha=0.7)
    ax.set_title("Manifold Distribution (S2)")
    
    # Save the plot
    plt.savefig("analysis_3d_sphere.png")
    print("Saved 3D plot to 'analysis_3d_sphere.png'")
    
    plt.show()


if __name__ == "__main__":
    analyze()