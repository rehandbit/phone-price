import os

RAW_DATA_PATH = os.path.join( 'data', 'phones_data_20250729_181901.csv')
BDT_TO_INR_RATE = 0.77
MIN_RELEASE_YEAR = 2019

TARGET_COLUMN = 'price_inr'
NUMERIC_COLUMNS = ["ram_gb", "storage_gb", "battery_mah", "screen_inches", "refresh_rate_hz",
                   "charging_watts", "main_camera_mp", "weight_grams", "process_nm",
                   "release_year", "has_5g", "has_oled_screen"]
CATEGORICAL_COLUMNS = ['brand','chipset_maker']
ARTIFACT_DIR = "artifacts"