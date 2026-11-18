
import joblib
from pathlib import Path
import pandas as pd

# Locate project directory
BASE_DIR = Path(__file__).resolve().parent

# Models are directly inside the GitHub repository
MODEL_DIR = BASE_DIR

# Load trained models
q10_model = joblib.load(MODEL_DIR / "catboost_q10.pkl")
q50_model = joblib.load(MODEL_DIR / "catboost_q50.pkl")
q90_model = joblib.load(MODEL_DIR / "catboost_q90.pkl")

# Define model features
FEATURE_COLUMNS = [
    'fuel type', 'gear type', 'Make', 'Model',
    'Year of manufacture', 'Colour', 'Condition', 'Mileage',
    'Engine Size', 'Selling Condition', 'Bought Condition',
    'car', 'Trim', 'Drivetrain', 'Seats',
    'Number of Cylinders', 'Horse Power', 'Registered city',
    'Vehicle Age', 'Mileage per year', 'Make_Model'
]

CATEGORICAL_FEATURES = [
    'fuel type', 'gear type', 'Make', 'Model', 'Colour',
    'Condition', 'Selling Condition', 'Bought Condition',
    'car', 'Trim', 'Drivetrain', 'Registered city', 'Make_Model'
]
