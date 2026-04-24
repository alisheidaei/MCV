import numpy as np
import pandas as pd
from scipy.interpolate import Rbf

def simulate_spatial_data(grid_size=20, seed=1368):
    """Generates a synthetic spatial dataset with autocorrelated variables.

    This function creates a grid of coordinates and simulates weather-like 
    features (Temp, Prcp) and environmental indicators (NDVI, PM2.5) using 
    Radial Basis Functions (RBF) to ensure spatial dependency. 

    Args:
        grid_size (int): The number of points along one axis of the square grid. 
            The total number of rows will be grid_size^2. Defaults to 20.
        seed (int): Random seed for reproducibility of the spatial fields. Defaults to 1368.

    Returns:
        pd.DataFrame: A dataset containing 'X', 'Y', 'temp', 'prcp', 'NDVI', and 'PM2_5'.
    """
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