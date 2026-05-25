#!/bin/bash -l
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --time=00:30:00
#SBATCH --partition=plgrid
#SBATCH --account=plglscclass26-cpu
#SBATCH --output=ray_cluster_%j.out

conda activate rayenv

export RAY_TMPDIR=/tmp/ray_$SLURM_JOB_ID
mkdir -p $RAY_TMPDIR
export RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO=0

nodes=($(scontrol show hostnames $SLURM_JOB_NODELIST))
head=${nodes[0]}
head_ip=$(srun --nodes=1 --ntasks=1 -w "$head" hostname -I | awk '{print $1}')
port=6379

echo "=== Starting Ray Head Node ==="
srun -N1 -n1 -w "$head" \
    ray start \
    --head \
    --node-ip-address="$head_ip" \
    --port=$port \
    --num-cpus=$SLURM_CPUS_PER_TASK \
    --temp-dir=$RAY_TMPDIR \
    --include-dashboard=false \
    --block &

sleep 15

echo "=== Starting Ray Worker Nodes ==="
for worker in "${nodes[@]:1}"; do
    srun -N1 -n1 -w "$worker" \
        ray start \
        --address="$head_ip:$port" \
        --num-cpus=$SLURM_CPUS_PER_TASK \
        --temp-dir=$RAY_TMPDIR \
        --block &
done

sleep 5

echo "=== Running Ray Tune Experiment ==="
python run_ray_tune.py
EXIT_CODE=$?

ray stop
rm -rf $RAY_TMPDIR
exit $EXIT_CODE