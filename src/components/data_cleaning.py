import re

import numpy as np
import pandas as pd

from src.constants import (BDT_TO_INR_RATE, CATEGORICAL_COLUMNS, MIN_RELEASE_YEAR,
                           NUMERIC_COLUMNS, TARGET_COLUMN)


def into_number(column, pattern=r"(\d+(?:\.\d+)?)"):
    """'6.7 inch' -> 6.7   '10,499' -> 10499"""
    text = column.astype(str).str.replace(",", "", regex=False)
    return pd.to_numeric(text.str.extract(pattern)[0], errors="coerce")


def into_gb(column):
    """'1TB' -> 1024   '512MB' -> 0.5   '8GB' -> 8"""
    number = into_number(column)
    unit = column.astype(str).str.extract(r"(TB|GB|MB)", flags=re.I)[0].str.upper()
    return number * unit.map({"TB": 1024, "GB": 1, "MB": 1 / 1024})


def biggest_watt(text):
    """'80W wired, 13.5W PD' -> 80"""
    watts = re.findall(r"(\d+(?:\.\d+)?)\s*W", str(text))
    return max(map(float, watts)) if watts else np.nan


def find_chipset_maker(text):
    if not isinstance(text, str):
        return "Unknown"
    text = text.lower()
    patterns = [("Qualcomm", r"qualcomm|snapdragon|msm\d|sdm\d|sm\d{4}"),
                ("MediaTek", r"mediatek|dimensity|helio|\bmt ?\d|mtk"),
                ("Apple", r"apple"), ("Samsung Exynos", r"exynos"),
                ("Unisoc", r"unisoc|spreadtrum|\bsc\d|tiger"),
                ("HiSilicon Kirin", r"kirin|hisilicon"), ("Google Tensor", r"tensor")]
    for maker, pattern in patterns:
        if re.search(pattern, text):
            return maker
    return "Other"


def clean_phone_data(raw_df: pd.DataFrame, keep_unpriced: bool = False) -> pd.DataFrame:
    """Raw scraped table -> clean table with only the columns the model needs.
    keep_unpriced=True also keeps phones that have no price (used for the website's phone list)."""
    df = raw_df[raw_df["brand"] != "Comparisons"].copy()                  # website pages, not phones
    df["brand"] = df["brand"].str.strip().str.title()                     # Doogee / DOOGEE -> Doogee
    is_smartphone = df["device_type"].str.lower().str.replace(" ", "") == "smartphone"
    df = df[is_smartphone].copy()
 
    # prices: Taka -> Rupees. Official price first, unofficial if official is missing.
    official = pd.to_numeric(df["price_official"].str.replace(r"[^\d.]", "", regex=True), errors="coerce")
    unofficial = pd.to_numeric(df["price_unofficial"].str.replace(r"[^\d.]", "", regex=True), errors="coerce")
    df[TARGET_COLUMN] = official.fillna(unofficial) * BDT_TO_INR_RATE
 
    df["release_year"] = into_number(df["release_date"], r"\b((?:19|20)\d{2})\b")
    df["ram_gb"] = into_gb(df["ram"])
    df["storage_gb"] = into_gb(df["internal_storage"])
    df["battery_mah"] = into_number(df["battery_capacity"], r"(\d{3,6})\s*mAh")
    df["screen_inches"] = into_number(df["screen_size"], r"^\s*(\d+(?:\.\d+)?)")
    df["refresh_rate_hz"] = into_number(df["refresh_rate"])
    df["charging_watts"] = df["quick_charging"].map(biggest_watt)
    df["main_camera_mp"] = into_number(df["primary_camera_resolution"])
    df["process_nm"] = into_number(df["fabrication"])
    df["weight_grams"] = into_number(df["weight"].where(df["weight"].str.contains("g", na=False)))
    df["has_5g"] = (df["network"].fillna("") + df["speed"].fillna("")).str.contains("5G").astype(int)
    df["has_oled_screen"] = df["display_type"].fillna("").str.contains("OLED").astype(int)
    df["chipset_maker"] = df["chipset"].map(find_chipset_maker)
 
    # impossible values -> empty (the pipeline fills them later)
    limits = {"ram_gb": (0.06, 64), "storage_gb": (0.004, 4096), "battery_mah": (300, 30000),
              "screen_inches": (1, 10), "refresh_rate_hz": (30, 240), "weight_grams": (30, 1500),
              "main_camera_mp": (0.1, 250), "release_year": (2000, 2026), TARGET_COLUMN: (200, 800000)}
    for column, (low, high) in limits.items():
        df.loc[~df[column].between(low, high), column] = np.nan
 
    # keep modern phones (and, for training, only those that have a price)
    df = df[df["release_year"] >= MIN_RELEASE_YEAR].copy()

    df = df[df["release_year"] >= MIN_RELEASE_YEAR].copy()
    if not keep_unpriced:
        df = df[df[TARGET_COLUMN].notna()].copy()
 
    # variants of one phone (8GB/128GB and 8GB/256GB) share a group id, used for a fair split
    df["group_id"] = df["brand"] + " | " + df["model"].str.replace(r"\s*\(.*\)", "", regex=True)
    return df[["brand", "model", "group_id"] + NUMERIC_COLUMNS + CATEGORICAL_COLUMNS[1:] + [TARGET_COLUMN]].reset_index(drop=True)