import os, sys
from dataclasses import dataclass

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score

from src.constants import ARTIFACT_DIR
from src.exception import CustomException
from src.logger import logging
from src.utils import save_json, save_object, tune_and_compare_models


@dataclass
class ModelTrainerConfig:
    model_path: str = os.path.join(ARTIFACT_DIR, "model.pkl")
    info_path: str = os.path.join(ARTIFACT_DIR, "model_info.json")


class ModelTrainer:
    def __init__(self):
        self.config = ModelTrainerConfig()

    def initiate_model_trainer(self, train_array, test_array, train_groups):
        logging.info("Model training started")
        try:
            x_train, y_train = train_array[:, :-1], train_array[:, -1]
            x_test, y_test = test_array[:, :-1], test_array[:, -1]

            models = {"Ridge": Ridge(),
                      "Random Forest": RandomForestRegressor(random_state=0),
                      "Gradient Boosting": HistGradientBoostingRegressor(random_state=0)}
            param_grids = {"Ridge": {"alpha": [0.1, 1, 10]},
                           "Random Forest": {"n_estimators": [150], "max_depth": [None, 14], "min_samples_leaf": [1, 3]},
                           "Gradient Boosting": {"learning_rate": [0.05, 0.1], "max_iter": [200, 400], "max_leaf_nodes": [15, 31]}}

            results = tune_and_compare_models(x_train, y_train, train_groups, models, param_grids)
            for name, (_, cv_score) in results.items():
                logging.info(f"{name}: cross-validated R2 = {cv_score:.3f}")

            best_name = max(results, key=lambda name: results[name][1])      # chosen on training data only
            best_model = results[best_name][0]

            # final check on phones the model has never seen
            predicted_log = best_model.predict(x_test)
            percent_error = np.abs(np.exp(predicted_log) - np.exp(y_test)) / np.exp(y_test)
            range_factor = float(np.exp(np.quantile(np.abs(y_test - predicted_log), 0.80)))   # 80% of real prices fall inside price/factor .. price*factor
            info = {"best_model": best_name,
                    "test_r2": float(r2_score(y_test, predicted_log)),
                    "typical_error_percent": float(np.median(percent_error) * 100),
                    "within_20_percent": float((percent_error <= 0.2).mean() * 100),
                    "range_factor_80": range_factor,
                    "cv_r2_all_models": {name: float(score) for name, (_, score) in results.items()}}

            save_object(self.config.model_path, best_model)
            save_json(self.config.info_path, info)
            logging.info(f"Best model: {best_name} | test R2 = {info['test_r2']:.3f} | typical error = {info['typical_error_percent']:.1f}%")
            return info
        except Exception as error:
            raise CustomException(error, sys)
