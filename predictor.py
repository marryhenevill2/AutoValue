
from pathlib import Path
import joblib
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_DIR = BASE_DIR / "model" / "models"

q10_model = joblib.load(MODEL_DIR / "catboost_q10.pkl")
q50_model = joblib.load(MODEL_DIR / "catboost_q50.pkl")
q90_model = joblib.load(MODEL_DIR / "catboost_q90.pkl")

print("All three CatBoost models loaded successfully!")

# Exact features used during model training
FEATURE_COLUMNS = [
    'fuel type',
    'gear type',
    'Make',
    'Model',
    'Year of manufacture',
    'Colour',
    'Condition',
    'Mileage',
    'Engine Size',
    'Selling Condition',
    'Bought Condition',
    'car',
    'Trim',
    'Drivetrain',
    'Seats',
    'Number of Cylinders',
    'Horse Power',
    'Registered city',
    'Vehicle Age',
    'Mileage per year',
    'Make_Model'
]

# Categorical features
CATEGORICAL_FEATURES = [
    'fuel type',
    'gear type',
    'Make',
    'Model',
    'Colour',
    'Condition',
    'Selling Condition',
    'Bought Condition',
    'car',
    'Trim',
    'Drivetrain',
    'Registered city',
    'Make_Model'
]

# Numeric features
NUMERIC_FEATURES = [
    col for col in FEATURE_COLUMNS
    if col not in CATEGORICAL_FEATURES
]


def predict_vehicle_price(vehicle_data):

    # Check for missing features
    missing_features = [
        col for col in FEATURE_COLUMNS
        if col not in vehicle_data
    ]

    if missing_features:
        raise ValueError(
            f"Missing features: {missing_features}"
        )

    # Arrange input features in training order
    input_df = pd.DataFrame(
        [vehicle_data],
        columns=FEATURE_COLUMNS
    )

    # Convert numerical features
    for col in NUMERIC_FEATURES:
        input_df[col] = pd.to_numeric(
            input_df[col],
            errors="raise"
        )

    # Ensure categorical features are strings
    for col in CATEGORICAL_FEATURES:
        if pd.isna(input_df.at[0, col]):
            raise ValueError(
                f"Missing categorical value: {col}"
            )

        input_df[col] = input_df[col].astype(str)

    # Generate predictions
    q10_prediction = q10_model.predict(input_df)[0]
    q50_prediction = q50_model.predict(input_df)[0]
    q90_prediction = q90_model.predict(input_df)[0]

    return {
        "lower_estimate": float(q10_prediction),
        "median_estimate": float(q50_prediction),
        "upper_estimate": float(q90_prediction)
    }


print("All three models loaded successfully!")
