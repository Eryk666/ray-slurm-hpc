#!/bin/bash

NODE_CONFIGS=(2 3)

for N in "${NODE_CONFIGS[@]}"; do
    sbatch --array=1-$N --nodes=1 baseline_job.sh
    sbatch --nodes=$N ray_job.sh
done