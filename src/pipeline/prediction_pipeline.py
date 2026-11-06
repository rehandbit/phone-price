import os
import sys

import numpy as np
import pandas as pd

from src.constants import ARTIFACT_DIR, CATEGORICAL_COLUMNS, NUMERIC_COLUMNS
from src.exception import CustomException
from src.utils import load_json, load_object


class CustomData:
    """Holds the values one user types in and turns them into a one-row table."""

    def __init__(self, brand, chipset_maker, ram_gb, storage_gb, battery_mah, screen_inches, refresh_rate_hz,
                 charging_watts, main_camera_mp, weight_grams, process_nm, release_year, has_5g, has_oled_screen):
        self.values = {"brand": str(brand).strip().title(), "chipset_maker": chipset_maker,
                       "ram_gb": ram_gb, "storage_gb": storage_gb, "battery_mah": battery_mah,
                       "screen_inches": screen_inches, "refresh_rate_hz": refresh_rate_hz,
                       "charging_watts": charging_watts, "main_camera_mp": main_camera_mp,
                       "weight_grams": weight_grams, "process_nm": process_nm, "release_year": release_year,
                       "has_5g": has_5g, "has_oled_screen": has_oled_screen}

    def get_data_as_dataframe(self):
        try:
            frame = pd.DataFrame([self.values])
            for column in NUMERIC_COLUMNS:
                frame[column] = pd.to_numeric(frame[column], errors="coerce")   # blank -> empty, filled by the pipeline
            return frame[NUMERIC_COLUMNS + CATEGORICAL_COLUMNS]
        except Exception as error:
            raise CustomException(error, sys)


class PredictPipeline:
    def __init__(self):
        self.preprocessor = load_object(os.path.join(ARTIFACT_DIR, "preprocessor.pkl"))
        self.model = load_object(os.path.join(ARTIFACT_DIR, "model.pkl"))
        self.info = load_json(os.path.join(ARTIFACT_DIR, "model_info.json"))

    def predict(self, input_frame):
        try:
            predicted_log = self.model.predict(self.preprocessor.transform(input_frame))[0]
            price = float(np.exp(predicted_log))
            factor = self.info["range_factor_80"]
            return {"expected_price_inr": round(price, -2),
                    "likely_lowest_inr": round(price / factor, -2),
                    "likely_highest_inr": round(price * factor, -2)}
        except Exception as error:
            raise CustomException(error, sys)
