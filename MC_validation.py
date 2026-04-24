from simulation import simulate_spatial_data
from preprocessing import split_spatial_data, scale_spatial_data
from model_utils import build_simple_dnn, calculate_metrics, plot_validation_results, report_final_metrics

def run_monte_carlo_pipeline(raw_data=None, cmodel=None, iterations=10, feature_cols=['temp', 'prcp', 'NDVI'], target_col='PM2_5', summary_plot=False, scale_data=True):
    """Executes a Monte Carlo validation routine for spatial regression models.

    This function repeatedly shuffles, splits, and evaluates a model to assess 
    performance stability and prevent spatial data leakage.

    Args:
        raw_data (pd.DataFrame, optional): Input dataset. If None, synthetic data is generated.
        cmodel (tf.keras.Model, optional): A compiled Keras model. If None, a default DNN is used.
        iterations (int): Number of random sub-sampling iterations. Defaults to 10.
        feature_cols (list): Names of predictor variables. Defaults to weather/NDVI features.
        target_col (str): Name of the dependent variable. Defaults to 'PM2_5'.
        summary_plot (bool): If True, renders trajectory plots of metrics. Defaults to False.
        scale_data (bool): Whether to apply feature scaling within each iteration. Defaults to True.
    """
    # Configuration
    all_cols = feature_cols + [target_col]    
    metrics_history = []
    
    print(f"--- Starting Monte Carlo Validation ({iterations} iterations) ---")
    
    # Generate the base dataset
    if raw_data is None:
        print("No input data detected; simulated default data has been generated.\n")
        raw_data = simulate_spatial_data(grid_size=30, seed=42)

    if cmodel is None::
            # Build the DNN Architecture
            print("No input model found; a simple DNN has been substituted.\n")
            model = build_simple_dnn(input_shape=len(feature_cols))
    
    for i in range(iterations):
        print(f"Iteration {i+1}/{iterations}...", end=" ", flush=True)
        
        # Split data (using i as a seed to ensure different splits each time)
        train, val, test = split_spatial_data(raw_data, seed=i)
        
        if scale_data==True:
            # Scale the features
            train_s, val_s, test_s, _ = scale_spatial_data(train, val, test, all_cols)              
        else:
            train_s, val_s, test_s = train, val, test
        
        # Fit the model
        # Tuning parameters (epochs/batch_size) can be adjusted here
        model.fit(
            train_s[feature_cols], train_s[target_col],
            validation_data=(val_s[feature_cols], val_s[target_col]),
            epochs=50,
            batch_size=32,
            verbose=0
        )
        
        # Evaluate on Test Set
        predictions = model.predict(test_s[feature_cols], verbose=0)
        metrics = calculate_metrics(test_s[target_col], predictions)
        
        # Store metrics
        metrics_history.append(metrics)
        print(f"Done. (R2: {metrics['R2']:.3f})")

    # Report Final Results (Mean and SD)
    summary_stats = report_final_metrics(metrics_history)
    
    # Plot Trajectories
    if summary_plot==True:
        plot_validation_results(metrics_history)