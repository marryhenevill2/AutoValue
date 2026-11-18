
import joblib
from pathlib import Path
import pandas as pd

# Locate project directory
# BASE_DIR = Path(__file__).resolve().parent.parent

# Locate models directory
MODEL_DIR = Path(r"C:\Users\HomePC\used_car_valuation\model\models")

# Load trained models
q10_model = joblib.load(MODEL_DIR / "catboost_q10.pkl")
q50_model = joblib.load(MODEL_DIR / "catboost_q50.pkl")
q90_model = joblib.load(MODEL_DIR / "catboost_q90.pkl")

# Load model metadata
metadata = joblib.load(MODEL_DIR / "models")

FEATURE_COLUMNS = metadata["feature_columns"]
CATEGORICAL_FEATURES = metadata["categorical_features"]


def predict_vehicle_price(vehicle_data):

    # Convert user input into DataFrame
    input_df = pd.DataFrame([vehicle_data])

    # Ensure correct feature order
    input_df = input_df[FEATURE_COLUMNS]

    # Generate predictions
    lower_price = q10_model.predict(input_df)[0]
    median_price = q50_model.predict(input_df)[0]
    upper_price = q90_model.predict(input_df)[0]

    return {
        "lower_price": float(lower_price),
        "fair_price": float(median_price),
        "upper_price": float(upper_price)
    }
