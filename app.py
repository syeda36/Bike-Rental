
import streamlit as st
import pandas as pd
import numpy as np
import pickle
from datetime import date

# --------------------------------------------------
# 1. PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="Bike Rental Demand Prediction",
    page_icon="🚲",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --------------------------------------------------
# 2. CUSTOM CSS
# --------------------------------------------------

st.markdown("""
<style>

.stApp {
    background: linear-gradient(135deg, #edf5f7, #f2f8f7);
}

section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #e5f0f5, #e8f3f2);
    border-right: 1px solid #d4e5e9;
}

section[data-testid="stSidebar"] h3 {
    color: #123e56;
}

section[data-testid="stSidebar"] p {
    color: #29485d;
    line-height: 1.8;
}

.hero-card {
    background: linear-gradient(110deg, #102f49, #087f89);
    padding: 38px 34px;
    border-radius: 22px;
    color: white;
    margin-bottom: 24px;
    box-shadow: 0 8px 22px rgba(16, 47, 73, 0.16);
}

.hero-card h1 {
    color: white !important;
    font-size: 36px;
    font-weight: 800;
    margin: 0 0 12px 0;
}

.hero-card p {
    color: #e8f7fb;
    font-size: 16px;
    line-height: 1.7;
    margin: 0;
}

.welcome-card {
    background: linear-gradient(135deg, #ffffff, #f0f8fb);
    border: 1px solid #d5e6ed;
    padding: 22px;
    border-radius: 16px;
    margin-bottom: 25px;
    box-shadow: 0 5px 16px rgba(20, 70, 90, 0.07);
    color: #29485d;
    line-height: 1.7;
}

h1, h2, h3 {
    color: #153c55 !important;
}

.stSelectbox label,
.stSlider label,
.stDateInput label {
    color: #29485d !important;
    font-weight: 600 !important;
}

.stButton > button {
    width: 100%;
    min-height: 54px;
    border-radius: 12px;
    border: 1px solid #087f89;
    background: linear-gradient(100deg, #123e56, #078b91);
    color: white !important;
    font-size: 17px;
    font-weight: 700;
    box-shadow: 0 5px 14px rgba(8, 127, 137, 0.18);
}

.stButton > button:hover {
    background: linear-gradient(100deg, #0b3048, #06777d);
    color: white !important;
}

.result-card {
    background: linear-gradient(120deg, #103a54, #087f89);
    padding: 34px 20px;
    border-radius: 20px;
    text-align: center;
    border: 1px solid #2d939b;
    box-shadow: 0 9px 25px rgba(16, 58, 84, 0.20);
    margin-top: 10px;
    margin-bottom: 22px;
}

.result-card p {
    color: #e0f7fa;
    font-size: 18px;
    margin: 0;
}

.result-card h2 {
    color: white !important;
    font-size: 58px;
    font-weight: 800;
    margin: 8px 0;
}

div[data-testid="stMetric"] {
    background: linear-gradient(135deg, #ffffff, #e7f3f7);
    padding: 18px;
    border: 1px solid #d4e5ed;
    border-radius: 15px;
    box-shadow: 0 4px 12px rgba(16, 58, 84, 0.07);
}

div[data-testid="stMetricValue"] {
    color: #087f89;
    font-weight: 750;
}

div[data-testid="stExpander"] {
    background: #f7fbfc;
    border: 1px solid #d4e5ed;
    border-radius: 12px;
}

.footer {
    text-align: center;
    color: #617b88;
    font-size: 13px;
    padding: 15px 0;
}

</style>
""", unsafe_allow_html=True)


# --------------------------------------------------
# 3. LOAD TRAINED MODEL
# --------------------------------------------------

@st.cache_resource
def load_model():

    with open("bike_rental_model.pkl", "rb") as file:
        return pickle.load(file)


try:

    model = load_model()

except Exception as e:

    st.error("Unable to load bike_rental_model.pkl")
    st.exception(e)
    st.stop()


# --------------------------------------------------
# 4. SIDEBAR
# --------------------------------------------------

with st.sidebar:

    st.markdown("## 🚲 Bike Demand")

    st.write("Prediction Dashboard")

    st.divider()

    st.markdown("### About this project")

    st.write(
        "This application predicts bike rental demand "
        "from weather, calendar, and time-related features."
    )

    st.markdown("### Model Information")

    st.write("**Target:** Total bike rentals (cnt)")

    st.write("**Task:** Regression")

    st.write(
        "**Algorithms evaluated:** MLR, SVR, "
        "Random Forest and Gradient Boosting"
    )

    st.info(
        "The model was trained using historical bike rental "
        "data from 2011–2012. Predictions are estimates."
    )


# --------------------------------------------------
# 5. MAIN HEADER
# --------------------------------------------------

st.markdown("""
<div class="hero-card">
<h1>🚲 Bike Rental Demand Prediction</h1>
<p>Predict bike rental demand using weather conditions, time, and calendar information.</p>
</div>
""", unsafe_allow_html=True)


# --------------------------------------------------
# 6. WELCOME CARD
# --------------------------------------------------

st.markdown("""
<div class="welcome-card">
<strong>Welcome!</strong><br>
Enter the details below to estimate the expected number
of bike rentals. This application uses a trained machine
learning model to generate predictions.
</div>
""", unsafe_allow_html=True)


# --------------------------------------------------
# 7. DATE AND TIME INPUTS
# --------------------------------------------------

st.subheader("📅 1. Date and Time")

col1, col2 = st.columns(2)

with col1:

    selected_date = st.date_input(
        "Select date",
        value=date(2012, 1, 1),
        min_value=date(2011, 1, 1),
        max_value=date(2026, 12, 31),
        format="YYYY/MM/DD"
    )

with col2:

    selected_hour = st.selectbox(
        "Select hour",
        options=list(range(24)),
        index=8,
        format_func=lambda x: f"{x:02d}:00"
    )


# --------------------------------------------------
# 8. WEATHER INPUTS
# --------------------------------------------------

st.subheader("🌦️ 2. Weather Conditions")

col1, col2, col3 = st.columns(3)

with col1:

    weather = st.selectbox(
        "Weather situation",
        ["Clear", "Mist", "Light Snow", "Heavy Rain"]
    )

    temperature = st.slider(
        "Temperature (°C)",
        0, 41, 25
    )

with col2:

    feels_like = st.slider(
        "Feels-like temperature (°C)",
        0, 50, 28
    )

    humidity = st.slider(
        "Humidity (%)",
        0, 100, 50
    )

with col3:

    windspeed = st.slider(
        "Wind speed",
        0, 67, 10
    )

    holiday = st.selectbox(
        "Is it a holiday?",
        ["No", "Yes"]
    )


# --------------------------------------------------
# 9. AUTOMATIC FEATURE ENGINEERING
# --------------------------------------------------

month = selected_date.month
day = selected_date.day
hour = selected_hour

# Dataset year encoding:
# 2011 = 0
# 2012 = 1
# Years after 2012 are mapped to 1
year_feature = float(selected_date.year)

# Sunday = 0, Monday = 1, ..., Saturday = 6
weekday = (selected_date.weekday() + 1) % 7


# Season

if month in [1, 2, 3]:
    season = "spring"

elif month in [4, 5, 6]:
    season = "summer"

elif month in [7, 8, 9]:
    season = "fall"

else:
    season = "winter"


# Hour period

if 5 <= hour <= 11:
    hour_period = "Morning"

elif 12 <= hour <= 16:
    hour_period = "Afternoon"

elif 17 <= hour <= 20:
    hour_period = "Evening"

else:
    hour_period = "Night"


# Weekend

is_weekend = 1 if weekday in [0, 6] else 0


# Working day

if is_weekend == 1 or holiday == "Yes":
    workingday = "No work"

else:
    workingday = "Working Day"


# Peak hour

is_peak_hour = (
    1 if hour in [7, 8, 9, 17, 18, 19] else 0
)


# Normalize weather inputs

normalized_temp = temperature / 41
normalized_atemp = feels_like / 50
normalized_humidity = humidity / 100
normalized_windspeed = windspeed / 67


# Cyclical features

hour_sin = np.sin(2 * np.pi * hour / 24)
hour_cos = np.cos(2 * np.pi * hour / 24)

month_sin = np.sin(2 * np.pi * (month - 1) / 12)
month_cos = np.cos(2 * np.pi * (month - 1) / 12)

day_of_year = selected_date.timetuple().tm_yday


# --------------------------------------------------
# 10. PREPARE MODEL INPUT
# --------------------------------------------------

input_data = pd.DataFrame([{

    "season": season,
    "yr": year_feature,
    "mnth": int(month),
    "hr": int(hour),
    "holiday": holiday,
    "weekday": int(weekday),
    "workingday": workingday,
    "weathersit": weather,
    "temp": normalized_temp,
    "atemp": normalized_atemp,
    "hum": normalized_humidity,
    "windspeed": normalized_windspeed,
    "day": int(day),
    "day_of_year": int(day_of_year),
    "is_weekend": int(is_weekend),
    "hour_period": hour_period,
    "is_peak_hour": int(is_peak_hour),
    "hour_sin": float(hour_sin),
    "hour_cos": float(hour_cos),
    "month_sin": float(month_sin),
    "month_cos": float(month_cos)

}])


# --------------------------------------------------
# 11. PREDICTION
# --------------------------------------------------

st.write("")

if st.button("🚲 Predict Bike Rental Demand"):

    try:

        prediction = model.predict(input_data)[0]

        prediction = max(0, int(round(prediction)))

        st.subheader("📊 Prediction Result")

        st.markdown(f"""
        <div class="result-card">
        <p>Predicted bike rentals for one hour</p>
        <h2>{prediction:,}</h2>
        <p>Estimated rental count</p>
        </div>
        """, unsafe_allow_html=True)


        # Demand interpretation

        if prediction < 50:

            demand_level = "Low"

            insight = (
                "Expected demand is relatively low. "
                "Fewer bikes may be needed."
            )

        elif prediction < 150:

            demand_level = "Moderate"

            insight = (
                "Expected demand is moderate. "
                "Maintain a reasonable supply of available bikes."
            )

        else:

            demand_level = "High"

            insight = (
                "Expected demand is high. "
                "Consider checking bike availability and capacity."
            )


        col1, col2 = st.columns(2)

        with col1:
            st.metric("Demand Level", demand_level)

        with col2:
            st.metric("Selected Hour", f"{hour:02d}:00")


        st.info(insight)

        st.caption(
            f"Selected date: {selected_date.strftime('%d %B %Y')} | "
            f"Weather: {weather} | Season: {season}"
        )

        # Show the year encoding used by the model
        st.caption(
            f"Selected year: {selected_date.year} | "
            f"Model year input (yr): {year_feature}"
        )

        


        with st.expander("View input features supplied to the model"):

            st.dataframe(
                input_data,
                use_container_width=True
            )


    except Exception as e:

        st.error("An error occurred while predicting demand.")
        st.exception(e)


# --------------------------------------------------
# 12. FOOTER
# --------------------------------------------------

st.divider()

st.markdown("""
<div class="footer">
🚲 Bike Rental Demand Prediction | Machine Learning Project<br>
Predictions are estimates and may differ from actual demand.
</div>
""", unsafe_allow_html=True)
