import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

def add_spatial_id(df, x_col='X', y_col='Y', id_name='station_id'):
    """Assigns a unique ID to each distinct pair of X and Y coordinates.

    Args:
        df (pd.DataFrame): The input spatial dataset.
        x_col (str): Name of the column containing X coordinates. Defaults to 'X'.
        y_col (str): Name of the column containing Y coordinates. Defaults to 'Y'.
        id_name (str): The name for the newly created ID column. Defaults to 'station_id'.

    Returns:
        pd.DataFrame: The dataframe including the new spatial ID column.
    """
    # Convert the zip to a pandas Index to avoid the FutureWarning
    spatial_index = pd.Index(zip(df[x_col], df[y_col]))
    df[id_name] = pd.factorize(spatial_index)[0]
    return df

def split_spatial_data(df, train_size=0.8, val_size=0.1, test_size=0.1, seed=1368):
    """Splits a dataset by station_id to mitigate spatial data leakage.

    Args:
        df (pd.DataFrame): Input dataframe containing 'X' and 'Y' or 'station_id'.
        train_size (float): Proportion of stations for training. Defaults to 0.8.
        val_size (float): Proportion for validation. Defaults to 0.1.
        test_size (float): Proportion for testing. Defaults to 0.1.
        seed (int): Random seed for reproducibility. Defaults to 1368.

    Returns:
        tuple: (train_df, val_df, test_df) as partitioned DataFrames.
    """

    if 'station_id' not in df.columns:
        df = add_spatial_id(df)
        
    np.random.seed(seed)
    unique_stations = df['station_id'].unique()
    np.random.shuffle(unique_stations)
    
    n_stations = len(unique_stations)
    train_end = int(n_stations * train_size)
    val_end = train_end + int(n_stations * val_size)
    
    train_df = df[df['station_id'].isin(unique_stations[:train_end])].copy()
    val_df = df[df['station_id'].isin(unique_stations[train_end:val_end])].copy()
    test_df = df[df['station_id'].isin(unique_stations[val_end:])].copy()
    
    return train_df, val_df, test_df



def scale_spatial_data(train_df, val_df, test_df, features):
    """Scales specified features using StandardScaler fitted on the training set.

    Args:
        train_df (pd.DataFrame): Training set used to fit the scaler.
        val_df (pd.DataFrame): Validation set to be transformed.
        test_df (pd.DataFrame): Test set to be transformed.
        features (list): List of column names to be scaled.

    Returns:
        tuple: (train_scaled, val_scaled, test_scaled, fitted_scaler).
    """
    scaler = StandardScaler()
    
    # Fit ONLY on training data
    scaler.fit(train_df[features])
    
    # Transform all sets
    train_df[features] = scaler.transform(train_df[features])
    val_df[features] = scaler.transform(val_df[features])
    test_df[features] = scaler.transform(test_df[features])
    
    return train_df, val_df, test_df, scaler