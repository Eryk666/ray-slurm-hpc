# 1. Environment Setup
Create a conda environment and install dependencies (ray[tune], xgboost, scipy, pandas).

# 2. Model and Data Loading (model.py)
Understanding the core XGBoost training function that will be executed in parallel. Loading and preparing the data.

# 3. The Baseline: SLURM Job Arrays
Submit the run_baseline.sh script. Observe how SLURM spawns independent jobs, each running a subset of trials. Inspect the output files and save results.

# 4. Ray Cluster
Submit the run_ray_cluster.sh script. Monitor the output logs to see the head node start and worker nodes connect. Save the results.

# 6. Benchmark Analysis
Compare the total time and result score of the Baseline and Ray Cluster. Report which solution was better and justify the choice based on factors like speed, final model accuracy, and resource management.

# 7. Troubleshooting & Common HPC Pitfalls
A section to help students navigate common errors.
