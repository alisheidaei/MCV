from simulation import simulate_spatial_data
from preprocessing import split_spatial_data, scale_spatial_data

# 1. Get data
df = simulate_spatial_data()

# 2. Split data
train, val, test = split_spatial_data(df)

# 3. Scale specific features
features_to_scale = ['temp', 'prcp', 'NDVI', 'PM2_5']
train_s, val_s, test_s, model_scaler = scale_spatial_data(train, val, test, features_to_scale)

print("Mean of scaled training temp:", train_s['temp'].mean()) # Should be close to 0