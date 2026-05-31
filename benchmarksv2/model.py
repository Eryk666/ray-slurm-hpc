import os
import ray
from scipy.io import arff
from xgboost import XGBClassifier
from sklearn.model_selection import cross_val_score
from sklearn.datasets import make_classification
import numpy as np
import pandas as pd


def load_and_preprocess_data(data_path):
    """Generates a synthetic dataset deterministically on the host machine or loads from a file."""
    if not data_path:
        X, y = make_classification(
            n_samples=2000,
            n_features=30,
            n_informative=20,
            n_redundant=10,
            random_state=42
        )

    else:
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"File not found: {data_path}")

        data, _ = arff.loadarff(data_path)
        df = pd.DataFrame(data)

        for col in df.columns:
            if df[col].dtype == object:
                df[col] = pd.to_numeric(df[col], errors='coerce')

        X = df.iloc[:, :-1].values.astype(np.float32)
        y = df.iloc[:, -1].values.astype(np.int32)

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