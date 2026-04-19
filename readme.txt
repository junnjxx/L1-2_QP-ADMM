CONVERGENCE-GUARANTEED ALGORITHMS FOR L1/2-REGULARIZED QUADRATIC PROGRAMS WITH ASSIGNMENT CONSTRAINTS
===============================================================================================

This repository contains the MATLAB implementations for the algorithms described in the paper. The code is organized into two main directories: one for synthetic data experiments and another for real-world data experiments.

Directory Structure
===================

1. /orthbatch: 
   - Purpose: Contains code for synthetic (quadratic) experiments.
   - Files:
     * test.m:            Main script. Handles dataset generation, parameter calling, and plotting.
     * matrix_method.m:   Solver interface for the Matrix model. Contains parameter initialization.
     * splitadmm20240119.m: Core algorithm. Implements the ADMM solver for the Matrix model.
     * vector_method.m:   Solver interface for the Vector model. Contains parameter initialization.
     * splitadmm20240307.m: Core algorithm. Implements the ADMM solver for the Vector model.

2. /bigbatch: 
   - Purpose: Contains code for real-world (deep learning batch selection) experiments.
   - Structure: The structure and file naming conventions are identical to the /orthbatch folder.
     * Uses the same solver interfaces (matrix_method.m, vector_method.m) and core algorithms 
       adapted for real data input and kernel matrix computation.

