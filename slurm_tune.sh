#!/bin/bash -l
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=00:30:00
#SBATCH --partition=plgrid
#SBATCH --account=plglscclass26-cpu
#SBATCH --output=ray_cluster_%j.out

conda activate rayenv

mkdir -p /tmp/ray_$SLURM_JOB_ID
export TMPDIR=/tmp/ray_$SLURM_JOB_ID

# Get the head node ip
nodes=$(scontrol show hostnames $SLURM_JOB_NODELIST)
nodes_array=($nodes)
head_node=${nodes_array[0]}
head_node_ip=$(srun --nodes=1 --ntasks=1 -w "$head_node" hostname --ip-address)

echo "Starting HEAD at $head_node_ip"

# block flag keeps the process alive
srun --nodes=1 --ntasks=1 -w "$head_node" \
    ray start --head --node-ip-address="$head_node_ip" --port=6379 \
    --temp-dir="$TMPDIR" --num-cpus="$SLURM_CPUS_PER_TASK" --block &

sleep 10

worker_num=$((SLURM_JOB_NUM_NODES - 1))
for ((i = 1; i <= worker_num; i++)); do
    node_i=${nodes_array[$i]}
    echo "Starting WORKER $i at $node_i"
    srun --nodes=1 --ntasks=1 -w "$node_i" \
        ray start --address="$head_node_ip:6379" \
        --temp-dir="$TMPDIR" --num-cpus="$SLURM_CPUS_PER_TASK" --block &
done

sleep 5

python run_ray_tune.py

ray stop

