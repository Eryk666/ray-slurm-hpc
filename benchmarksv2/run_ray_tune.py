import sys

import ray
from ray import tune
from ray.tune.schedulers import ASHAScheduler
from ray.tune.search.bayesopt import BayesOptSearch
from model import train_model, load_and_preprocess_data
import time
import os
import json
import psutil
import random
import numpy as np

# Set seeds for reproducibility
def set_seeds(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

def log_resources():
    """Log current resource utilization on the executing worker node"""
    return {
        "cpu_percent": psutil.cpu_percent(interval=None), # Non-blocking for production tune loops
        "memory_percent": psutil.virtual_memory().percent,
        "memory_gb": round(psutil.virtual_memory().used / (1024**3), 2)
    }

def objective(config):
    """Training objective with timing, resource tracking, and type fixing for BayesOpt"""
    start = time.time()

    # Create a local copy to modify parameters safely
    trainer_config = config.copy()

    # Cast float domains from BayesOpt back into integers for XGBoost
    trainer_config["n_estimators"] = int(np.round(config["n_estimators"]))
    trainer_config["max_depth"] = int(np.round(config["max_depth"]))

    # Execute training
    score = train_model(trainer_config)

    duration = time.time() - start
    resources_end = log_resources()

    tune.report({
        "accuracy": score,
        "training_time": duration,
        "cpu_usage": resources_end["cpu_percent"],
        "memory_gb": resources_end["memory_gb"]
    })

def main():
    set_seeds(42)
    num_nodes = int(sys.argv[1])
    total_samples = int(sys.argv[2])

    # Initialize Ray
    print("Connecting to Ray cluster...")
    ray.init(address="auto")

    # Log cluster information
    print("\n=== Ray Cluster Status ===")
    print(f"Cluster resources: {ray.cluster_resources()}")
    print(f"Available nodes: {len(ray.nodes())}")
    print("=" * 50)

    # Generate and pin synthetic dataset to shared memory
    print("start data",time.time())
    X, y = load_and_preprocess_data()
    print("end data",time.time())

    X_ref = ray.put(X)
    y_ref = ray.put(y)

    # Define search space (using continuous distributions to keep BayesOpt happy)
    search_space = {
        "n_estimators": tune.uniform(100, 1000),
        "max_depth": tune.uniform(4, 15),
        "learning_rate": tune.loguniform(1e-4, 1e-1),
        "data_X": X_ref,
        "data_y": y_ref,
        "num_cpus": 15  # Match the resource request token below
    }

    # Configure ASHA scheduler
    scheduler = ASHAScheduler(
        metric="accuracy",
        mode="max",
        max_t=100,
        grace_period=10,
        reduction_factor=3,
        brackets=1
    )


    # Create results directory
    storage_path = os.path.expandvars(f"$SCRATCH/ray_results/{num_nodes}")
    os.makedirs(storage_path, exist_ok=True)

    print(f"\nResults will be saved to: {storage_path}")
    print(f"Starting hyperparameter optimization with {total_samples} samples...")

    experiment_start = time.time()

    # Configure and run tuner matching your 4-node benchmark allocation
    tuner = tune.Tuner(
        tune.with_resources(objective, {"cpu": 16}),
        param_space=search_space,
        tune_config=tune.TuneConfig(
            scheduler=scheduler,
            num_samples=total_samples,
            max_concurrent_trials=num_nodes  # 4 concurrent trials * 15 CPUs = 60/64 CPUs utilized
        ),
        run_config=tune.RunConfig(
            name="xgb_hpo",
            storage_path=storage_path,
            verbose=1
        )
    )

    results = tuner.fit()
    experiment_duration = time.time() - experiment_start

    # Fetch best trial outcome
    best = results.get_best_result(metric="accuracy", mode="max")
    df = results.get_dataframe()

    # Clean the dataset references out of configuration logs before export
    clean_best_config = {k: v for k, v in best.config.items() if not k.startswith("data_")}

    # Calculate complete performance statistics
    summary = {
        "method": "ray_tune",
        "total_samples": len(df),
        "best_accuracy": float(best.metrics["accuracy"]),
        "best_config": clean_best_config,
        "mean_accuracy": float(df["accuracy"].mean()),
        "std_accuracy": float(df["accuracy"].std()),
        "median_accuracy": float(df["accuracy"].median()),
        "total_time_seconds": experiment_duration,
        "total_training_time_seconds": float(df["training_time"].sum()),
        "avg_time_per_sample": float(df["training_time"].mean()),
        "time_to_best_seconds": float(df.loc[:df["accuracy"].idxmax(), "training_time"].sum()),
        "iterations_to_best": int(df["accuracy"].idxmax() + 1),
        "cluster_resources": str(ray.cluster_resources()),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
    }

    # Save statistics output file
    summary_file = os.path.join(storage_path, "ray_tune_summary.json")
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)

    # Print results summary block
    print("\n" + "=" * 60)
    print("RAY TUNE OPTIMIZATION COMPLETE")
    print("=" * 60)
    print(f"Total samples evaluated: {summary['total_samples']}")
    print(f"Best accuracy: {summary['best_accuracy']:.6f}")
    print(f"Mean accuracy: {summary['mean_accuracy']:.6f} ± {summary['std_accuracy']:.6f}")
    print(f"Median accuracy: {summary['median_accuracy']:.6f}")
    print(f"\nBest config:")
    for key, value in clean_best_config.items():
        print(f"  {key}: {value}")
    print(f"\nTiming:")
    print(f"  Total experiment time: {experiment_duration:.2f}s ({experiment_duration/60:.2f}m)")
    print(f"  Total training time: {summary['total_training_time_seconds']:.2f}s")
    print(f"  Average time per sample: {summary['avg_time_per_sample']:.2f}s")
    print(f"  Time to find best: {summary['time_to_best_seconds']:.2f}s")
    print(f"  Iterations to best: {summary['iterations_to_best']}")
    print(f"\nResults saved to: {summary_file}")
    print("=" * 60)

    ray.shutdown()
    return 0

if __name__ == "__main__":
    try:
        exit_code = main()
        exit(exit_code)
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
        ray.shutdown()
        exit(1)