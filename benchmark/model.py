import os
import pandas as pd
from scipy.io import arff
from xgboost import XGBClassifier
from sklearn.model_selection import cross_val_score
import numpy as np

_data_cache = None

def get_dataset(path): # Accept path as argument
    global _data_cache
    if _data_cache is None:
        # HARDCODED: Use the literal string from 'echo $SCRATCH'
        # Ensure the filename is exactly what is on disk (.arff)
        path = "/net/afscra/people/plgolejarzeryk/3year.arff"

        # Load ARFF file
        data, meta = arff.loadarff(path)
        df = pd.DataFrame(data)

        # ARFF strings are often byte-encoded (e.g., b'class_name')
        # This decodes them if necessary
        for col in df.columns:
            if df[col].dtype == object:
                df[col] = df[col].str.decode('utf-8')

        # Assume the last column is the target (common in ARFF)
        target_col = meta.names()[-1]
        X = df.drop(target_col, axis=1)
        y = df[target_col]

        # Convert categorical target to numeric if it's not already
        if y.dtype == object or y.dtype.name == 'category':
            y = pd.factorize(y)[0]

        _data_cache = (X, y)
    return _data_cache

def train_model(config):
    X, y = get_dataset(config["data_path"])

    model = XGBClassifier(
        n_estimators=config["n_estimators"],
        max_depth=config["max_depth"],
        learning_rate=config["learning_rate"],
        tree_method="hist",
        n_jobs=1,
        verbosity=0,
        random_state=42
    )
    print("training start\n")

    scores = cross_val_score(model, X_train, y_train, cv=3, n_jobs=1)
    print("training end\n")
    return np.mean(scores)