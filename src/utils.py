import json
import os
import pickle
import sys

import numpy as np
from sklearn.model_selection import GridSearchCV, GroupKFold

from src.exception import CustomException


def save_object(file_path, obj):
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, "wb") as file:
            pickle.dump(obj, file)
    except Exception as error:
        raise CustomException(error, sys)


def load_object(file_path):
    try:
        with open(file_path, "rb") as file:
            return pickle.load(file)
    except Exception as error:
        raise CustomException(error, sys)


def save_json(file_path, content):
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, "w") as file:
        json.dump(content, file, indent=2)


def load_json(file_path):
    with open(file_path) as file:
        return json.load(file)


def tune_and_compare_models(x_train, y_train, groups, models, param_grids):
    """Tries every model with its small parameter grid. Returns {name: (best_model, cv_r2)}.
    GroupKFold keeps all variants of the same phone together (no leakage)."""
    try:
        results = {}
        for name, model in models.items():
            search = GridSearchCV(model, param_grids[name], cv=GroupKFold(n_splits=3), scoring="r2")
            search.fit(x_train, y_train, groups=groups)
            results[name] = (search.best_estimator_, search.best_score_)
        return results
    except Exception as error:
        raise CustomException(error, sys)
