# Spatial Monte Carlo Validation (SMCV) Framework

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## 🎯 Overview
[cite_start]The **Spatial Monte Carlo Validation (SMCV)** framework is a methodology-focused tool designed to rigorously evaluate the performance and stability of spatial models. [cite_start]The primary objective of this project is to provide a robust validation pipeline that mitigates **spatial data leakage**—a common pitfall in spatial data science where autocorrelation leads to overoptimistic performance metrics.



[cite_start]While this repository includes a Deep Neural Network (DNN) and synthetic environmental datasets (X, Y, NDVI, Temp, etc.), these serve as a **proof-of-concept** to demonstrate the framework's effectiveness in a controlled, reproducible environment.

## 🧠 Methodology
[cite_start]The core of this framework relies on a three-stage validation process:

1.  [cite_start]**Coordinate-to-ID Mapping:** Spatial coordinates $(X, Y)$ are mapped to unique **Station IDs**. [cite_start]This ensures that the fundamental unit of independence is the geographic location, not the individual data row.
2.  [cite_start]**Station-Based Partitioning:** Data splitting (Train/Validation/Test) is performed at the station level. [cite_start]By keeping all observations from a single location within the same set, the model is forced to generalize to entirely new geographic areas during testing.
3.  [cite_start]**Monte Carlo Iteration:** The pipeline executes $s$ iterations of random station-based sub-sampling. [cite_start]This provides a distribution of performance metrics ($R^2$, RMSE, MAE), allowing researchers to assess the **robustness** and **variance** of the model rather than relying on a single, potentially biased split.



## 📂 Project Architecture
* `MC_validation.py`: The **core engine** of the framework. [cite_start]A model-agnostic loop that manages iterative splitting and performance aggregation.
* [cite_start]`preprocessing.py`: Implements the station-based splitting logic and feature scaling protocols.
* [cite_start]`model_utils.py`: Provides evaluation metrics and the "testbed" DNN architecture.
* [cite_start]`simulation.py`: Generates spatially autocorrelated synthetic data for framework demonstration.
* [cite_start]`main.py`: The primary entry point for executing a validation run.

## 🛠️ Getting Started

### Installation
1. Clone the repository:
   ```bash
   git clone [https://github.com/alisheidaei/MCV.git](https://github.com/alisheidaei/MCV.git)
   cd MCV