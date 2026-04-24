import os
import pandas as pd

# 1. Suppress TensorFlow initialization logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

from simulation import simulate_spatial_data
from preprocessing import split_spatial_data, scale_spatial_data
from model_utils import build_simple_dnn, calculate_metrics, plot_validation_results, report_final_metrics

def run_monte_carlo_pipeline(iterations=10):
    """
    Executes the Monte Carlo validation approach for the spatial DNN.
    """
    # Configuration
    feature_cols = ['temp', 'prcp', 'NDVI']
    target_col = 'PM2_5'
    all_cols = feature_cols + [target_col]
    
    # List to store metrics from each iteration
    metrics_history = []
    
    print(f"--- Starting Monte Carlo Validation ({iterations} iterations) ---")
    
    # 1. Generate the base dataset
    # We generate the data once, but we will reshuffle the splits in the loop
    raw_data = simulate_spatial_data(grid_size=30, seed=42)
    
    for i in range(iterations):
        print(f"Iteration {i+1}/{iterations}...", end=" ", flush=True)
        
        # 2. Split data (using i as a seed to ensure different splits each time)
        train, val, test = split_spatial_data(raw_data, seed=i)
        
        # 3. Scale the features
        train_s, val_s, test_s, _ = scale_spatial_data(train, val, test, all_cols)
        
        # 4. Build the DNN Architecture
        model = build_simple_dnn(input_shape=len(feature_cols))
        
        # 5. Fit the model
        # Tuning parameters (epochs/batch_size) can be adjusted here
        model.fit(
            train_s[feature_cols], train_s[target_col],
            validation_data=(val_s[feature_cols], val_s[target_col]),
            epochs=50,
            batch_size=32,
            verbose=0
        )
        
        # 6. Evaluate on Test Set
        predictions = model.predict(test_s[feature_cols], verbose=0)
        metrics = calculate_metrics(test_s[target_col], predictions)
        
        # 7. Store metrics
        metrics_history.append(metrics)
        print(f"Done. (R2: {metrics['R2']:.3f})")

    # 8. Report Final Results (Mean and SD)
    summary_stats = report_final_metrics(metrics_history)
    
    # 9. Plot Trajectories
    plot_validation_results(metrics_history)

if __name__ == "__main__":
    # You can change the number of iterations here
    run_monte_carlo_pipeline(iterations=10)