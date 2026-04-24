import os
import pandas as pd

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

from simulation import simulate_spatial_data
from preprocessing import split_spatial_data, scale_spatial_data
from model_utils import build_simple_dnn, calculate_metrics, plot_validation_results, report_final_metrics
from MC_validation import run_monte_carlo_pipeline


if __name__ == "__main__":
    run_monte_carlo_pipeline(iterations=10)