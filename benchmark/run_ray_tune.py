import ray
from ray import tune
from ray.tune.schedulers import ASHAScheduler
from model import train_model

ray.init(address="auto")

def objective(config):
    score = train_model(config)
    tune.report({"accuracy": score})

search_space = {
    "n_estimators": tune.randint(100, 1000),
    "max_depth": tune.randint(4, 15),
    "learning_rate": tune.loguniform(1e-4, 1e-1)
}

scheduler = ASHAScheduler(
    metric="accuracy",
    mode="max",
    max_t=100,
    grace_period=10 
)

tuner = tune.Tuner(
    tune.with_resources(objective, {"cpu": 4}),
    param_space=search_space,
    tune_config=tune.TuneConfig(
        scheduler=scheduler,
        num_samples=20
    ),
    run_config=tune.RunConfig(
        name="xgb_hpo",
        storage_path="/tmp/ray_results"
    )
)

results = tuner.fit()
best = results.get_best_result(metric="accuracy", mode="max")

print("Best config:", best.config)
print("Best accuracy:", best.metrics["accuracy"])

ray.shutdown()