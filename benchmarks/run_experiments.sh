#!/bin/bash

NODE_CONFIGS=(2 4 6 8 10 12)

for N in "${NODE_CONFIGS[@]}"; do
    sbatch --array=1-$N --nodes=1 baseline_job.sh
    sbatch --nodes=$N ray_job.sh
done