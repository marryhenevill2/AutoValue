
import streamlit as st
import pandas as pd
from pathlib import Path

from predictor import predict_vehicle_price


# ==========================================
# 1. PROJECT CONFIGURATION
# ==========================================

st.set_page_config(
    page_title="AutoValue - Used Car Valuation",
    page_icon="🚗",
    layout="wide"
)

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "vehicle_cleaned.csv"

CURRENT_YEAR = 2026


# ==========================================
# 2. LOAD DATASET
# ==========================================

@st.cache_data
def load_data():
    return pd.read_csv(DATA_PATH)


df = load_data()


# ==========================================
# 3. GET DROPDOWN OPTIONS
# ==========================================

def get_options(column):

    values = (
        df[column]
        .dropna()
        .astype(str)
        .unique()
        .tolist()
    )

    return sorted(values)


# ==========================================
# 4. APPLICATION TITLE
# ==========================================

st.title("🚗 AutoValue - Used Car Valuation System")

st.write(
    "Estimate the market value of a used vehicle "
    "using Machine Learning and CatBoost Quantile Regression."
)

st.divider()

st.subheader("Enter Vehicle Information")


# ==========================================
# 5. VEHICLE INPUT FORM
# ==========================================

with st.form("vehicle_form"):

    col1, col2, col3 = st.columns(3)

    # --------------------------------------
    # COLUMN 1
    # --------------------------------------

    with col1:

        make = st.selectbox(
            "Vehicle Make",
            get_options("Make")
        )

        model_options = (
            df.loc[
                df["Make"].astype(str) == make,
                "Model"
            ]
            .dropna()
            .astype(str)
            .unique()
            .tolist()
        )

        model_options = sorted(model_options)

        model = st.selectbox(
            "Vehicle Model",
            model_options
        )

        year = st.number_input(
            "Year of Manufacture",
            min_value=1900,
            max_value=CURRENT_YEAR,
            value=2020,
            step=1
        )

        colour = st.selectbox(
            "Colour",
            get_options("Colour")
        )

        condition = st.selectbox(
            "Vehicle Condition",
            get_options("Condition")
        )

    # --------------------------------------
    # COLUMN 2
    # --------------------------------------

    with col2:

        fuel_type = st.selectbox(
            "Fuel Type",
            get_options("fuel type")
        )

        gear_type = st.selectbox(
            "Gear Type",
            get_options("gear type")
        )

        mileage = st.number_input(
            "Mileage (km)",
            min_value=0,
            value=50000,
            step=1000
        )

        engine_size = st.number_input(
            "Engine Size",
            min_value=0.0,
            value=2.0,
            step=0.1
        )

        horsepower = st.number_input(
            "Horse Power",
            min_value=0,
            value=150,
            step=5
        )

        cylinders = st.number_input(
            "Number of Cylinders",
            min_value=0,
            value=4,
            step=1
        )

    # --------------------------------------
    # COLUMN 3
    # --------------------------------------

    with col3:

        car_type = st.selectbox(
            "Vehicle Type",
            get_options("car")
        )

        trim = st.selectbox(
            "Vehicle Trim",
            get_options("Trim")
        )

        drivetrain = st.selectbox(
            "Drivetrain",
            get_options("Drivetrain")
        )

        seats = st.number_input(
            "Number of Seats",
            min_value=1,
            value=5,
            step=1
        )

        selling_condition = st.selectbox(
            "Selling Condition",
            get_options("Selling Condition")
        )

        bought_condition = st.selectbox(
            "Bought Condition",
            get_options("Bought Condition")
        )

        registered_city = st.selectbox(
            "Registered City",
            get_options("Registered city")
        )

    st.divider()

    submitted = st.form_submit_button(
        "Predict Vehicle Price",
        use_container_width=True
    )


# ==========================================
# 6. FEATURE ENGINEERING AND PREDICTION
# ==========================================

if submitted:

    vehicle_age = CURRENT_YEAR - year

    mileage_per_year = mileage / max(vehicle_age, 1)

    make_model = make + model

    vehicle_data = {

        "fuel type": fuel_type,

        "gear type": gear_type,

        "Make": make,

        "Model": model,

        "Year of manufacture": year,

        "Colour": colour,

        "Condition": condition,

        "Mileage": mileage,

        "Engine Size": engine_size,

        "Selling Condition": selling_condition,

        "Bought Condition": bought_condition,

        "car": car_type,

        "Trim": trim,

        "Drivetrain": drivetrain,

        "Seats": seats,

        "Number of Cylinders": cylinders,

        "Horse Power": horsepower,

        "Registered city": registered_city,

        "Vehicle Age": vehicle_age,

        "Mileage per year": mileage_per_year,

        "Make_Model": make_model
    }

    try:

        predictions = predict_vehicle_price(vehicle_data)

        st.success(
            "Vehicle valuation completed successfully!"
        )

        st.subheader("Estimated Vehicle Value")

        col1, col2, col3 = st.columns(3)

        with col1:

            st.metric(
                "Lower Estimate (Q10)",
                f"₦{predictions['lower_estimate']:,.0f}"
            )

        with col2:

            st.metric(
                "Fair Market Estimate (Q50)",
                f"₦{predictions['median_estimate']:,.0f}"
            )

        with col3:

            st.metric(
                "Upper Estimate (Q90)",
                f"₦{predictions['upper_estimate']:,.0f}"
            )

        st.caption(
            "These are machine-learning-based estimates, "
            "not guaranteed selling prices."
        )

    except Exception as error:

        st.error(
            f"Prediction failed: {error}"
        )


# ==========================================
# 7. FOOTER
# ==========================================

st.divider()

st.caption(
    "AutoValue | Powered by CatBoost Quantile Regression"
)

