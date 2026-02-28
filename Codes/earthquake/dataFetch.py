import requests
import pandas as pd
import io

# This is the API endpoint from your screenshot
# Base URL: www.ngdc.noaa.gov/hazel
# Endpoint: /hazard-service/api/v1/earthquakes
API_URL = "https://www.ngdc.noaa.gov/hazel/hazard-service/api/v1/earthquakes"

print("Contacting NOAA Database...")

# We request data. Let's limit it to recent history to keep it fast initially.
# You can change minYear to 1900 later for the full dataset.
params = {
    'minYear': 2000,
    'maxYear': 2025
}

response = requests.get(API_URL, params=params)

if response.status_code == 200:
    print("Data received! Processing...")
    
    # The data comes back as JSON (a dictionary)
    data_json = response.json()
    
    # The actual list of quakes is usually under a key like 'items'
    if 'items' in data_json:
        quakes = data_json['items']
    else:
        # Sometimes it's just the list directly
        quakes = data_json
        
    # Convert to a Pandas DataFrame
    df = pd.DataFrame(quakes)
    
    # We only care about Location (Lat/Lon) and maybe Magnitude
    # The API usually names them 'latitude' and 'longitude'
    if 'latitude' in df.columns and 'longitude' in df.columns:
        clean_df = df[['id', 'year', 'latitude', 'longitude', 'eqMagnitude']]
        
        # Filter out rows with missing location data
        clean_df = clean_df.dropna(subset=['latitude', 'longitude'])
        
        # Save to CSV
        output_filename = 'noaa_earthquakes_2000_2025.csv'
        clean_df.to_csv(output_filename, index=False)
        print(f"Success! Saved {len(clean_df)} earthquakes to '{output_filename}'")
        print(clean_df.head())
    else:
        print("Error: Could not find 'latitude'/'longitude' columns in the response.")
        print("Available columns:", df.columns)
        
else:
    print(f"Failed to download. Status Code: {response.status_code}")