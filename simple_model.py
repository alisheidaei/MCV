import os

# 1. Optional: Suppress TensorFlow boilerplate logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

from simulation import simulate_spatial_data
from preprocessing import split_spatial_data, scale_spatial_data
from model_utils import build_simple_dnn, calculate_metrics

def main():
    # 2. Setup configuration
    feature_cols = ['temp', 'prcp', 'NDVI']
    target_col = 'PM2_5'
    all_cols = feature_cols + [target_col]

    print("--- Starting Spatial Data Pipeline ---")

    # 3. Generate Simulated Data
    # Creates a dataset with X, Y, temp, prcp, NDVI, and PM2.5
    raw_data = simulate_spatial_data(grid_size=30, seed=42)
    print(f"Data generated. Shape: {raw_data.shape}")

    # 4. Split Data (Spatial Split by Station ID)
    # This prevents spatial leakage by ensuring stations are unique to each set
    train, val, test = split_spatial_data(raw_data, train_size=0.7, val_size=0.15, test_size=0.15)

    # 5. Scale the Data
    # We fit the scaler ONLY on the training set and apply it to others
    train_s, val_s, test_s, scaler = scale_spatial_data(train, val, test, all_cols)
    print("Data split and scaled successfully.")

    # 6. Build the TensorFlow DNN
    model = build_simple_dnn(input_shape=len(feature_cols))

    # 7. Train the Model
    print("Training model...")
    history = model.fit(
        train_s[feature_cols], train_s[target_col],
        validation_data=(val_s[feature_cols], val_s[target_col]),
        epochs=100,
        batch_size=32,
        verbose=1  # Set to 1 to see progress per epoch
    )

    # 8. Evaluation
    print("Evaluating on test set...")
    predictions = model.predict(test_s[feature_cols], verbose=0)
    
    # Calculate R2, RMSE, and MAE
    metrics = calculate_metrics(test_s[target_col], predictions)

    print("\n--- Final Performance Metrics ---")
    for metric, value in metrics.items():
        print(f"{metric}: {value}")

if __name__ == "__main__":
    main()