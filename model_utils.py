import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

def build_simple_dnn(input_shape):
    """
    Defines a simple Deep Neural Network for regression.
    """
    model = models.Sequential([
        layers.Input(shape=(input_shape,)),
        layers.Dense(64, activation='relu'),
        layers.Dense(32, activation='relu'),
        layers.Dense(1)  # Output layer for PM2.5 (Regression)
    ])
    
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model

def calculate_metrics(y_true, y_pred):
    """
    Calculates R2, RMSE, and MAE for the results.
    Works with both numpy arrays and tensors.
    """
    # Ensure inputs are flat numpy arrays
    y_true = np.array(y_true).flatten()
    y_pred = np.array(y_pred).flatten()
    
    r2 = r2_score(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    
    return {
        "R2": round(r2, 4),
        "RMSE": round(rmse, 4),
        "MAE": round(mae, 4)
    }

def plot_validation_results(metrics_list):
    """
    Plots the trajectory of metrics (R2, RMSE, MAE) over the simulation iterations.
    """
    df_metrics = pd.DataFrame(metrics_list)
    epochs = range(1, len(df_metrics) + 1)
    
    plt.figure(figsize=(12, 4))
    
    for i, metric in enumerate(['R2', 'RMSE', 'MAE']):
        plt.subplot(1, 3, i+1)
        plt.plot(epochs, df_metrics[metric], marker='o', linestyle='-', color='teal')
        plt.title(f'{metric} Trajectory')
        plt.xlabel('Iteration')
        plt.ylabel('Value')
        plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

def report_final_metrics(metrics_list):
    """
    Calculates and prints the mean and standard deviation of the validation metrics.
    """
    df_metrics = pd.DataFrame(metrics_list)
    summary = df_metrics.agg(['mean', 'std']).T
    
    print("\n" + "="*30)
    print("MONTE CARLO VALIDATION RESULTS")
    print("="*30)
    print(summary)
    print("="*30)
    return summary