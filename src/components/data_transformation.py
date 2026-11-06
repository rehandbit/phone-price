import os, sys
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.constants import ARTIFACT_DIR, CATEGORICAL_COLUMNS, NUMERIC_COLUMNS, TARGET_COLUMN
from src.exception import CustomException
from src.logger import logging
from src.utils import save_object


@dataclass
class DataTransformationConfig:
    preprocessor_path: str = os.path.join(ARTIFACT_DIR, "preprocessor.pkl")


class DataTransformation:
    def __init__(self):
        self.config = DataTransformationConfig()

    def get_preprocessor(self):
        """Numbers: fill empty with median, then scale. Text: fill empty with 'Unknown', then one-hot."""
        number_steps = Pipeline([("fill", SimpleImputer(strategy="median")),
                                 ("scale", StandardScaler())])
        text_steps = Pipeline([("fill", SimpleImputer(strategy="constant", fill_value="Unknown")),
                               ("onehot", OneHotEncoder(handle_unknown="ignore", min_frequency=10, sparse_output=False))])
        return ColumnTransformer([("numbers", number_steps, NUMERIC_COLUMNS),
                                  ("text", text_steps, CATEGORICAL_COLUMNS)])

    def initiate_data_transformation(self, train_path, test_path):
        logging.info("Data transformation started")
        try:
            train_df, test_df = pd.read_csv(train_path), pd.read_csv(test_path)
            preprocessor = self.get_preprocessor()

            x_train = preprocessor.fit_transform(train_df[NUMERIC_COLUMNS + CATEGORICAL_COLUMNS])
            x_test = preprocessor.transform(test_df[NUMERIC_COLUMNS + CATEGORICAL_COLUMNS])

            # predict the LOG of the price: mistakes then become percentages
            train_array = np.c_[x_train, np.log(train_df[TARGET_COLUMN].values)]
            test_array = np.c_[x_test, np.log(test_df[TARGET_COLUMN].values)]

            save_object(self.config.preprocessor_path, preprocessor)
            logging.info("Preprocessor saved")
            return train_array, test_array, train_df["group_id"].values
        except Exception as error:
            raise CustomException(error, sys)
