# Phone Price Estimator

A machine learning web app that estimates the price of a smartphone in Indian Rupees (₹) from its specs. It also shows whether a phone is available and its listed market price.

Built end to end: data cleaning, training pipeline, prediction pipeline, logging, error handling and a Flask website.

## What it does
- Pick a **brand**, then a **model**. Only that brand's models and chipsets appear, and the specs fill in automatically.
- Or choose "type my own specs" and enter any phone's specs by hand.
- The app shows:
  - the **expected price** with a likely range,
  - the **availability** (Available, Upcoming, Rumored, Out of stock),
  - the **listed market price** (official or unofficial) and how far it is above or below the estimate,
  - the age of that price and the old price/discount when the data has one.

## Data
- Source: a scrape of mobiledokan.com (Bangladesh), collected on 29 July 2025.
- File used: `data/raw/phones_data_20250729_181901.csv` (the version that contains prices).
- Prices are listed in Bangladeshi Taka and converted to rupees at 1 BDT = 0.71 INR (an approximate rate; change `BDT_TO_INR_RATE` in `src/constants.py`).
- Training uses smartphones released in 2019 or later that have a price (about 2,477 phones).

## Project structure
```
phone_price_project/
├── app.py                          # Flask website and JSON API
├── requirements.txt
├── render.yaml                     # Render deployment settings
├── data/raw/                       # the raw CSV file
├── artifacts/                      # created by training (model, preprocessor, train/test data)
├── logs/                           # created automatically, one log file per run
└── src/
    ├── constants.py                # settings: file path, exchange rate, column lists
    ├── logger.py                   # logging to console and to logs/
    ├── exception.py                # custom error with file name and line number
    ├── utils.py                    # save/load helpers, model comparison
    ├── components/
    │   ├── data_cleaning.py        # raw text -> numbers, prices in rupees
    │   ├── data_ingestion.py       # read, clean, train/test split, phone list
    │   ├── data_transformation.py  # fill empty values, scale, one-hot encode
    │   └── model_trainer.py        # compare models, save the best
    └── pipeline/
        ├── train_pipeline.py       # runs all training steps in order
        └── predict_pipeline.py     # turns user input into a price estimate
```

## How it works
1. **Cleaning:** removes junk rows, fixes brand spelling, keeps smartphones, converts text such as "8GB" or "80W wired" into numbers, removes impossible values and converts prices to rupees.
2. **Split:** all variants of the same phone (for example the 8 GB and 12 GB versions) go to the same side, training or test, so the test is fair.
3. **Transformation:** empty numbers are filled with the median and scaled; empty text becomes "Unknown" and is one-hot encoded.
4. **Training:** Ridge, Random Forest and Gradient Boosting are compared with grouped cross-validation. The best one is chosen on training data only and checked once on unseen test phones. The model predicts the log of the price, so its mistakes are percentages.
5. **Prediction:** the saved preprocessor and model turn the user's specs into an expected price and an 80% range.

## Results (test phones the model never saw)
| Measure | Value |
|---|---|
| Best model | Gradient Boosting |
| R² (on log price) | about 0.92 |
| Typical error | about 13% |
| Predictions within 20% of the real price | about 69% |

## Run it on your computer
Needs Python 3.11 or newer.

```
pip install -r requirements.txt
python -m src.pipeline.train_pipeline
python app.py
```
Then open http://127.0.0.1:5000

Run the training command from the project folder (the one that contains `app.py`), not from inside `src`. Run it again whenever you change the data, the settings or your scikit-learn version.

## API
`POST /api/predict` with JSON:
```json
{
  "brand": "Xiaomi", "model": "", "chipset_maker": "MediaTek",
  "ram_gb": 8, "storage_gb": 256, "battery_mah": 5000, "screen_inches": 6.7,
  "refresh_rate_hz": 120, "charging_watts": 67, "main_camera_mp": 50,
  "weight_grams": 190, "process_nm": 6, "release_year": 2025,
  "has_5g": 1, "has_oled_screen": 1
}
```
Returns `expected_price_inr`, `likely_lowest_inr`, `likely_highest_inr` and a `market` block (or `null` when no model is given).

Other endpoints: `GET /api/models?brand=Asus` and `GET /api/phone?brand=Asus&model=ROG Phone 2`.

## Deploy on Render
- **Build command:** `pip install -r requirements.txt && python -m src.pipeline.train_pipeline`
- **Start command:** `gunicorn app:app --bind 0.0.0.0:$PORT --workers 1 --timeout 120`
- **Environment variable:** `PYTHON_VERSION` = a full version, for example `3.11.9`



## Limitations about this dataset
- Prices are from one scrape on 29 July 2025. They are not live prices.
- Some prices are unofficial (grey-market) and lower than official ones.
- Estimates are not exact: the typical error is about 13%, and larger for very expensive phones such as foldables.
- The "Upcoming" status probably means not yet on sale in Bangladesh. This is an interpretation, not something the file states.
- Phones released after July 2025 are not in the data.
- The app does not estimate depreciation or resale value, because the data has no price history.
