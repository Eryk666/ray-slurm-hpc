import random
import json
import os
from model import train_model

def run_baseline_trial(trial_id, num_samples=20):
    results = []

    for i in range(num_samples):
        config = {
            "n_estimators": random.randint(100, 1000),
            "max_depth": random.randint(4, 15),
            "learning_rate": 10 ** random.uniform(-4, -1)
        }

        score = train_model(config)
        results.append({"config": config, "accuracy": score})
        print(f"Trial {trial_id}, Sample {i+1}/{num_samples}: {score:.4f}")

    output_file = f"results/baseline/baseline_results_{trial_id}.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)

    best = max(results, key=lambda x: x["accuracy"])
    print(f"\nTrial {trial_id} Best Accuracy: {best['accuracy']:.4f}")
    print(f"Best Config: {best['config']}")

    return best

if __name__ == "__main__":
    trial_id = int(os.environ.get("SLURM_ARRAY_TASK_ID", 1))
    run_baseline_trial(trial_id, num_samples=20)