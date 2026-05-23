from sklearn.datasets import make_classification
from sklearn.model_selection import cross_val_score
from xgboost import XGBClassifier
import numpy as np

_X_train, _y_train = None, None

def get_dataset():
    global _X_train, _y_train
    if _X_train is None:
        from sklearn.model_selection import train_test_split
        X, y = make_classification(
            n_samples=500, n_features=10,
            n_informative=8, n_redundant=2,
            n_classes=2, random_state=42
        )
        _X_train, _, _y_train, _ = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
    print("dataset ready\n")
    return _X_train, _y_train

def train_model(config):
    X_train, y_train = get_dataset()
    
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