import random
import json
import os
import gc
import sys
import time
import numpy as np
import psutil
from tqdm import tqdm
from model import train_model, load_and_preprocess_data

# Set seeds for reproducibility
def set_seeds(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

def log_resources():
    """Log current resource utilization without blocking execution"""
    return {
        "cpu_percent": psutil.cpu_percent(interval=None),  # Set to None so it does not inject 1-second delays
        "memory_percent": psutil.virtual_memory().percent,
        "memory_gb": round(psutil.virtual_memory().used / (1024**3), 2)
    }

def run_baseline_trial(num_nodes, trial_id, num_samples=50, seed=None, data_path=None):
    """
    Run baseline random search trial synchronized with the Ray Tune environment.

    Args:
        num_nodes: Total number of nodes in the experiment (used for logging and storage path organization)
        trial_id: Trial identifier (from SLURM array task)
        num_samples: Number of configurations to evaluate (50 per worker x 4 nodes = 200 total)
        seed: Random seed (if None, uses trial_id * 42)
        data_path: Path to the dataset file (if empty or None, generates synthetic data)
    """
    if seed is None:
        seed = trial_id * 42
    set_seeds(seed)

    print("=" * 60)
    print(f"BASELINE WORKER TRIAL {trial_id}")
    print("=" * 60)
    print(f"Random seed: {seed}")
    print(f"Number of samples per node: {num_samples}")
    print(f"Starting at: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60 + "\n")

    # Generate the shared synthetic dataset locally ONCE per Slurm task to prevent I/O bottlenecks
    X, y = load_and_preprocess_data(data_path)

    # Establish dynamic scratch workspace path
    storage_path = os.path.expandvars(f"$SCRATCH/baseline_results/{num_nodes}")
    os.makedirs(storage_path, exist_ok=True)

    results = []
    trial_start_time = time.time()
    best_so_far = 0

    # Progress bar setup
    pbar = tqdm(range(num_samples), desc=f"Worker Node {trial_id}")

    for i in pbar:
        # Construct configuration block mapping the required resources and clean dataset references
        config = {
            "n_estimators": random.randint(100, 1000),
            "max_depth": random.randint(4, 15),
            "learning_rate": 10 ** random.uniform(-4, -1),
            "data_X": X,
            "data_y": y,
            "num_cpus": 16  # Baseline uses all 16 allocated cores completely for XGBoost training
        }

        resources_start = log_resources()
        sample_start = time.time()

        # Train model
        score = train_model(config)

        sample_duration = time.time() - sample_start
        resources_end = log_resources()

        if score > best_so_far:
            best_so_far = score

        # Clean config logs to prevent dumping raw dataset matrices into JSON records
        clean_config = {k: v for k, v in config.items() if not k.startswith("data_")}

        # Store clean result metrics
        result = {
            "sample_id": i + 1,
            "config": clean_config,
            "accuracy": float(score),
            "training_time": sample_duration,
            "cpu_usage": resources_end["cpu_percent"],
            "memory_gb": resources_end["memory_gb"]
        }
        results.append(result)

        # Update progress visual interface
        pbar.set_postfix({
            "acc": f"{score:.4f}",
            "best": f"{best_so_far:.4f}",
            "time": f"{sample_duration:.1f}s"
        })

        # Actively clear iteration space arrays out of RAM
        gc.collect()

    pbar.close()

    total_time = time.time() - trial_start_time

    # Calculate performance statistics
    accuracies = [r["accuracy"] for r in results]
    training_times = [r["training_time"] for r in results]

    best_result = max(results, key=lambda x: x["accuracy"])
    best_index = accuracies.index(best_result["accuracy"])

    summary = {
        "trial_id": trial_id,
        "seed": seed,
        "method": "baseline_random_search",
        "total_samples": num_samples,
        "best_accuracy": float(best_result["accuracy"]),
        "best_config": best_result["config"],
        "best_found_at_iteration": best_index + 1,
        "mean_accuracy": float(np.mean(accuracies)),
        "std_accuracy": float(np.std(accuracies)),
        "median_accuracy": float(np.median(accuracies)),
        "min_accuracy": float(np.min(accuracies)),
        "max_accuracy": float(np.max(accuracies)),
        "total_time_seconds": total_time,
        "total_training_time_seconds": float(np.sum(training_times)),
        "avg_time_per_sample": float(np.mean(training_times)),
        "time_to_best_seconds": float(np.sum(training_times[:best_index + 1])),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "results": results
    }

    # Save detailed data summary log
    output_file = os.path.join(storage_path, f"baseline_results_trial_{trial_id}.json")
    with open(output_file, 'w') as f:
        json.dump(summary, f, indent=2)

    # Print results summary block
    print("\n" + "=" * 60)
    print(f"TRIAL {trial_id} COMPLETE")
    print("=" * 60)
    print(f"Best accuracy: {summary['best_accuracy']:.6f}")
    print(f"Mean accuracy: {summary['mean_accuracy']:.6f} ± {summary['std_accuracy']:.6f}")
    print(f"Median accuracy: {summary['median_accuracy']:.6f}")
    print(f"Best found at iteration: {summary['best_found_at_iteration']}/{num_samples}")
    print(f"\nBest config:")
    for key, value in best_result["config"].items():
        print(f"  {key}: {value}")
    print(f"\nTiming:")
    print(f"  Total time: {total_time:.2f}s ({total_time/60:.2f}m)")
    print(f"  Total training time: {summary['total_training_time_seconds']:.2f}s")
    print(f"  Average per sample: {summary['avg_time_per_sample']:.2f}s")
    print(f"  Time to find best: {summary['time_to_best_seconds']:.2f}s")
    print(f"\nResults saved to: {output_file}")
    print("=" * 60)

    return summary

if __name__ == "__main__":
    num_nodes = int(sys.argv[1])
    trial_id = int(sys.argv[2])
    num_samples = int(sys.argv[3])
    data_path = sys.argv[4] if len(sys.argv) > 4 else None

    run_baseline_trial(num_nodes, trial_id, num_samples=num_samples, data_path=data_path)