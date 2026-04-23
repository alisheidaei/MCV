import numpy as np
import pandas as pd
from scipy.interpolate import Rbf

def simulate_spatial_data(grid_size=20, seed=1368):
    np.random.seed(seed)
    
    # 1. Create X, Y Coordinates
    x = np.linspace(0, 10, grid_size)
    y = np.linspace(0, 10, grid_size)
    xv, yv = np.meshgrid(x, y)
    df = pd.DataFrame({'X': xv.flatten(), 'Y': yv.flatten()})
    
    # Helper to create smooth spatial noise (Simulating spatial autocorrelation)
    def generate_spatial_field(scale=1.0):
        points = np.random.uniform(0, 10, (10, 2))
        values = np.random.normal(0, scale, 10)
        rbf = Rbf(points[:,0], points[:,1], values, function='gaussian')
        return rbf(df['X'], df['Y'])

    # 2. Simulate Gridmet-like variables (Temp, Precipitation)
    df['temp'] = 20 + 5 * np.sin(df['X']/2) + generate_spatial_field(2)
    df['prcp'] = np.maximum(0, 2 + generate_spatial_field(3))
    
    # 3. Simulate NDVI (influenced by temp/prcp + spatial patterns)
    df['NDVI'] = 0.5 + 0.1 * df['temp'] / 20 + 0.2 * np.tanh(df['prcp']) + generate_spatial_field(0.1)
    df['NDVI'] = df['NDVI'].clip(0, 1) # NDVI must be between 0 and 1
    
    # 4. Simulate PM2.5 (Target variable)
    # PM2.5 often higher with lower NDVI (less filtration) and specific weather
    df['PM2_5'] = 15 - (5 * df['NDVI']) + (0.5 * df['temp']) + np.random.normal(0, 1, len(df))
    
    return df

if __name__ == "__main__":
    spatial_df = simulate_spatial_data(grid_size=30)
    print("Test Run:")
    print(spatial_df.head())