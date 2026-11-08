import os
import sys
from dataclasses import dataclass

import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

from src.components.data_cleaning import clean_phone_data
from src.constants import ARTIFACT_DIR, RAW_DATA_PATH
from src.exception import CustomException
from src.logger import logging


@dataclass
class DataIngestionConfig:
    cleaned_data_path: str = os.path.join(ARTIFACT_DIR, "cleaned_data.csv")
    catalog_path: str = os.path.join(ARTIFACT_DIR, "phone_catalog.csv")
    train_data_path: str = os.path.join(ARTIFACT_DIR, "train.csv")
    test_data_path: str = os.path.join(ARTIFACT_DIR, "test.csv")


class DataIngestion:
    def __init__(self):
        self.config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        logging.info("Data ingestion started")
        try:
            raw_df = pd.read_csv(RAW_DATA_PATH, dtype=str)
            logging.info(f"Raw data read: {raw_df.shape}")

            clean_df = clean_phone_data(raw_df)
            logging.info(f"Clean data: {clean_df.shape}")

            os.makedirs(ARTIFACT_DIR, exist_ok=True)
            clean_df.to_csv(self.config.cleaned_data_path, index=False)

            # every modern phone, with or without a price: used by the website dropdowns
            clean_phone_data(raw_df, keep_unpriced=True).to_csv(self.config.catalog_path, index=False)

            # all variants of one phone must stay on the same side of the split
            splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
            train_rows, test_rows = next(splitter.split(clean_df, groups=clean_df["group_id"]))
            clean_df.iloc[train_rows].to_csv(self.config.train_data_path, index=False)
            clean_df.iloc[test_rows].to_csv(self.config.test_data_path, index=False)
            logging.info(f"Train rows: {len(train_rows)} | Test rows: {len(test_rows)}")
            return self.config.train_data_path, self.config.test_data_path
        except Exception as error:
            raise CustomException(error, sys)