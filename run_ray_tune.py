import ray
from ray import tune
from ray.tune.schedulers import ASHAScheduler
import os

ray.init(address="auto")

def objective(config):
    for i in range(10):
        score = config["alpha"] * i + config["beta"]
        tune.report({"mean_accuracy": score})

search_space = {
    "alpha": tune.uniform(0, 1),
    "beta": tune.uniform(0, 10)
}

tuner = tune.Tuner(
    objective,
    param_space=search_space,
    tune_config=tune.TuneConfig(
        metric="mean_accuracy",
        mode="max",
        num_samples=10,
    ),
)

results = tuner.fit()

print("Best hyperparameters found were: ", results.get_best_result().config)

ray.shutdown()

