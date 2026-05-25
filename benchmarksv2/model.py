import os
import ray
from xgboost import XGBClassifier
from sklearn.model_selection import cross_val_score
from sklearn.datasets import make_classification
import numpy as np

def load_and_preprocess_data():
    """Generates a synthetic dataset deterministically on the host machine."""
    X, y = make_classification(
        n_samples=2000, 
        n_features=30, 
        n_informative=20, 
        n_redundant=10, 
        random_state=42
    )
    return X, y

def train_model(config):
    X = config["data_X"]
    y = config["data_y"]

    # Explicitly resolve Ray Object References if passed unresolved
    if isinstance(X, ray.ObjectRef):
        X = ray.get(X)
    if isinstance(y, ray.ObjectRef):
        y = ray.get(y)

    num_cpus = config.get("num_cpus", 16)

    model = XGBClassifier(
        n_estimators=config["n_estimators"],
        max_depth=config["max_depth"],
        learning_rate=config["learning_rate"],
        tree_method="hist",
        n_jobs=num_cpus,  
        verbosity=0,
        random_state=42
    )

    scores = cross_val_score(model, X, y, cv=3, n_jobs=1)
    return np.mean(scores)