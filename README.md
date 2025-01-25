# Representative Load Periods Identification
This repository contains the models and methods used in the CIRED 2025 paper for benchmarking approaches to identify representative load profiles (RPs) for fast distribution network analysis (DNA). 

## Dataset
The `Datasets` directory contains all the data types used in the paper. To use and cite the datasets, please refer to the following sources:
- [Analyzing electric vehicle, load and photovoltaic generation uncertainty using publicly available datasets](https://arxiv.org/abs/2409.01284)
- [Dataset sorce GitHub page](https://github.com/umar-hashmi/Public-Load-profile-Datasets)

## Clustering methods
- The `Clustering.methods` module implements two commonly used clustering approaches:
  - K-Means
  - K-Medoids
- Each clustering method provides the option to select dissimilarity measurements:
  - Euclidean Distance
  - Dynamic Time Warping (DTW)
- The K-Medoids object includes an additional method to identify extreme periods.
- To select RPs, follow these steps:
  - `cluster_method = Cluster(n_clusters, max_iter, distance_metric='euclidean', w=10, init_method='random', extreme=False)`
    - Cluster -> KMeans or KMedoids
    - n_clusters -> number of RPs to be selected
    - w -> window size if fast-DTW is used.
  - `distance_matrix = cluster_method.dist_matrix_calculation(input_data)`
  - `cluster_method.fit(input_data, dist_matrix=distance_matrix)`
  - `RPs = cluster_method.get_RPs(input_data)`
  - `weights = cluster_method.get_weights()`

## Optimization-based (LDC-opt) method
- The `Optimization.methods` module includes a load duration curve-based optimization approach (LDC-opt) to identify RPs. To select RPs, follow these steps:
    - `opt_method = dc_opt(Nrepr=5)`
    - `opt_problem = opt_method.define_optimization_problem(input_data)`
    - `opt_method.solve_problem(opt_problem)`
    - `RPs = opt_method.get_RPs(input_data)`
    - `weights = opt_method.get_weights()`

## Evaluation
- In `evaluation.py`, a simple example is implemented to demonstrate the workflow of using methods for benchmarking.
- In `utils.py`, all additional functions are provided, including the implementation of evaluation metrics: RMSE, REE_av, and RELE.
