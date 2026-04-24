# Spatial Monte Carlo Validation (SMCV) Framework

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## 🎯 Overview
The **Spatial Monte Carlo Validation (SMCV)** framework is a methodology-focused tool designed to rigorously evaluate the performance and stability of spatial models. The primary objective of this project is to provide a robust validation pipeline that mitigates **spatial data leakage**—a common pitfall in spatial data science where autocorrelation leads to overoptimistic performance metrics.

While this repository includes a Deep Neural Network (DNN) and synthetic environmental datasets (X, Y, NDVI, Temp, etc.), these serve as a **proof-of-concept** to demonstrate the framework's effectiveness in a controlled, reproducible environment.

## 🧠 Methodology
The core of this framework relies on a three-stage validation process:

1.  **Coordinate-to-ID Mapping:** Spatial coordinates $(X, Y)$ are mapped to unique **Station IDs**. This ensures that the fundamental unit of independence is the geographic location, not the individual data row.
2.  **Station-Based Partitioning:** Data splitting (Train/Validation/Test) is performed at the station level. By keeping all observations from a single location within the same set, the model is forced to generalize to entirely new geographic areas during testing.
3.  **Monte Carlo Iteration:** The pipeline executes $s$ iterations of random station-based sub-sampling. This provides a distribution of performance metrics ($R^2$, RMSE, MAE), allowing researchers to assess the **robustness** and **variance** of the model rather than relying on a single, potentially biased split.



## 📂 Project Architecture
* `MC_validation.py`: The **core engine** of the framework. A model-agnostic loop that manages iterative splitting and performance aggregation.
* `preprocessing.py`: Implements the station-based splitting logic and feature scaling protocols.
* `model_utils.py`: Provides evaluation metrics and the "testbed" DNN architecture.
* `simulation.py`: Generates spatially autocorrelated synthetic data for framework demonstration.
* `main.py`: The primary entry point for executing a validation run.

## 🛠️ Getting Started

### Installation
1. Clone the repository:
   ```bash
   git clone [https://github.com/alisheidaei/MCV.git](https://github.com/alisheidaei/MCV.git)
   cd MCV