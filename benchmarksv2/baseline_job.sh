#!/bin/bash -l
#SBATCH --array=1-4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --time=00:30:00
#SBATCH --partition=plgrid
#SBATCH --account=plglscclass26-cpu
#SBATCH --output=baseline_%A_%a.out

conda activate rayenv

DATA_PATH=""
TOTAL_TRIALS=200
NUM_NODES=$SLURM_ARRAY_TASK_COUNT
SAMPLES_PER_NODE=$((TOTAL_TRIALS / NUM_NODES))
REMAINDER=$((TOTAL_TRIALS % NUM_NODES))
if [ "$SLURM_ARRAY_TASK_ID" -le "$REMAINDER" ]; then
    SAMPLES_PER_NODE=$((SAMPLES_PER_NODE + 1))
fi

python -u run_baseline.py $NUM_NODES $SLURM_ARRAY_TASK_ID $SAMPLES_PER_NODE $DATA_PATH