#!/bin/bash -l
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=00:30:00
#SBATCH --partition=plgrid
#SBATCH --account=plglscclass26-cpu
#SBATCH --output=ray_cluster_%j.out

conda activate rayenv

export RAY_TMPDIR=/tmp/ray_$SLURM_JOB_ID
mkdir -p $RAY_TMPDIR
export RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO=0
export DATASET_PATH="$SCRATCH/3year.arff"

nodes=($(scontrol show hostnames $SLURM_JOB_NODELIST))
head=${nodes[0]}

head_ip=$(srun --nodes=1 --ntasks=1 -w "$head" hostname -I | awk '{print $1}')
port=6379

echo "=== Ray Cluster Configuration ==="
echo "HEAD NODE: $head"
echo "HEAD IP: $head_ip"
echo "WORKER NODES: ${nodes[@]:1}"
echo "CPUS PER NODE: $SLURM_CPUS_PER_TASK"
echo "TEMP DIR: $RAY_TMPDIR"
echo "================================="

# Start head node
echo "Starting head node..."
srun -N1 -n1 -w "$head" \
    ray start \
    --head \
    --node-ip-address="$head_ip" \
    --port=$port \
    --num-cpus=$SLURM_CPUS_PER_TASK \
    --temp-dir=$RAY_TMPDIR \
    --include-dashboard=false \
    --block &

sleep 25

for worker in "${nodes[@]:1}"; do
    echo "Starting worker: $worker"
    srun -N1 -n1 -w "$worker" \
        ray start \
        --address="$head_ip:$port" \
        --num-cpus=$SLURM_CPUS_PER_TASK \
        --temp-dir=$RAY_TMPDIR \
        --block &
    sleep 5
done

sleep 15

echo "Verifying cluster..."
python -c "import ray; ray.init(address='auto'); print('Cluster resources:', ray.cluster_resources()); ray.shutdown()" || {
    echo "ERROR: Failed to connect to Ray cluster"
    ray stop
    exit 1
}

echo "Starting Ray Tune experiment..."
python run_ray_tune.py

EXIT_CODE=$?

echo "Stopping Ray cluster..."
ray stop

rm -rf $RAY_TMPDIR

if [ $EXIT_CODE -eq 0 ]; then
    echo "Experiment completed successfully!"
else
    echo "Experiment failed with exit code $EXIT_CODE"
fi

exit $EXIT_CODE