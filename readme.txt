CONVERGENCE-GUARANTEED ALGORITHMS FOR L1/2-REGULARIZED QUADRATIC PROGRAMS 
WITH ASSIGNMENT CONSTRAINTS
LIJUN XIE RAN GU XIN LIN
===============================================================================================

## Purpose of this Repository
This repository contains the source code, and configuration files used to generate the numerical results presented in the paper mentioned above.

The primary purpose of this repository is to ensure **reproducibility** of the computational experiments. Specifically, this code allows readers to:
1. Reproduce all tables and figures shown in the manuscript.
2. Verify the implementation of the proposed  L1/2-regularized quadratic programming ADMM algorithm.

It contains the MATLAB and Python implementations for the algorithms 
described in the paper and is organized into two main directories: 
one for synthetic quadratic experiments (MATLAB) and another for 
real-world deep learning batch selection experiments (Python/MATLAB hybrid).

1. Directory Structure
=======================

The root directory contains two main folders:

- /orthbatch: MATLAB code for synthetic data (quadratic) experiments.
- /bigbatch:  Python & MATLAB code for real-world data (Deep Learning batch selection) experiments.

2. Synthetic Data Experiments (/orthbatch)
===========================================
This directory contains MATLAB scripts for generating synthetic datasets 
and testing the convergence of the algorithms.

Key Files:
- test.m:               Main script. Handles dataset generation and parameter calling.
- matrix_method.m:      Solver interface for the Matrix model.
- vector_method.m:      Solver interface for the Vector model.
- splitadmm2024*.m:     Core algorithms. Implement ADMM solvers for Matrix/Vector models.
- randbatch.m:          Baseline method (Random Batch).


3. Real-World Data Experiments (/bigbatch)
===========================================
This directory contains code for experiments on CIFAR-10 datasets, 
demonstrating the application in Deep Learning batch selection.

Key Subdirectories:
- /opt:                 Contains the core MATLAB optimization scripts (similar to /orthbatch).
                        Includes pre-processed kernel matrices and labels.
- /new_model:           Contains the Python framework for DL training and data processing.
    - /Datasets:        Scripts for dataset conversion (JPG generation, label processing).
    - /models:          Deep Learning training/testing model, training configurations and PYTHON implementation of core MATLAB optimization code. Due to the large scale of the datasets, running the full pipeline purely in MATLAB proved to be quite time-consuming. To improve efficiency, we first generated a reasonable initial point using Python, and then passed this warm-start solution to the MATLAB solver for high-precision refinement. This hybrid approach significantly reduced computation time while maintaining numerical accuracy. 
    - /results:         Output folder containing training logs, accuracy curves, and loss plots.

4. Dependencies & Environment
==============================
To reproduce the results, please ensure the following environments are configured:

- MATLAB: Required to run the core solvers (*.m files) in both /orthbatch and /bigbatch/opt.

5. How to Run
==============
A. For Synthetic Experiments (MATLAB):
   1. Open MATLAB.
   2. Navigate to the '/orthbatch' directory.
   3. Run the script: 'test.m'

B. For Real-World Experiments (Deep Learning Batch Selection):
   1. Data Preparation:
      - Ensure datasets (MNIST/CIFAR-10) are placed in the correct paths as referenced 
        in the Python scripts (e.g., /bigbatch/new_model/Datasets).
      - Use the provided Python scripts (e.g., 'to_jpg.py', 'generate_train_txt.py') 
        to preprocess the data if needed.
   2. Optimization (PYTHON):
      - Run the main scripts in 'big batch/new_model' (e.g. 'our_main.py')

   3. Optimization (MATLAB):
      - Run the main scripts in '/bigbatch/opt' (e.g., 'test.m' or specific model scripts) 
        to generate the optimal batch indices (sample_order files).

   4. Deep Learning Training (Python):
      - Navigate to '/bigbatch/new_model'.
      - Run the training script: 'dl_models.py' 
      - The script will load the pre-computed optimal batches and train the models.

A snapshot of this repository is provided as supplementary material for long-term archival.