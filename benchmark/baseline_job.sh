#!/bin/bash -l
#SBATCH --array=1-10
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --time=00:30:00
#SBATCH --partition=plgrid
#SBATCH --account=plglscclass26-cpu
#SBATCH --output=baseline_%A_%a.out

conda activate rayenv

python -u run_baseline.py