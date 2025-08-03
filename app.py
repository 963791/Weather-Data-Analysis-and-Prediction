import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import joblib
import os

# Load the dataset
@st.cache_data
def load_data():
    df = pd.read_csv("data/historical_weather_data.csv")
    df['Date'] = pd.to_datetime(df['Date'])  # Corrected column name
    return df

# Load the trained model
def load_model():
    model_path = "model/weather_model.pkl"
    if os.path.exists(model_path):
        return joblib.load(model_path)
    else:
        return None

# Page config
st.set_page_config(page_title="🌤️ Weather Data Analysis", layout="wide")

st.title("🌤️ Weather Data Analysis and Temperature Prediction")

df = load_data()
model = load_model()

# Show raw data
with st.expander("📊 Show Raw Data"):
    st.dataframe(df)

# Line chart: Temperature trend
st.subheader("📈 Temperature Over Time")
fig, ax = plt.subplots()
ax.plot(df['Date'], df['Temperature'], label='Temperature (°C)', color='orange')
ax.set_xlabel("Date")
ax.set_ylabel("Temperature (°C)")
ax.set_title("Temperature Trend Over Time")
ax.legend()
st.pyplot(fig)

# Prediction input
st.subheader("🔮 Predict Temperature")
st.markdown("Provide the weather details to predict the temperature:")

col1, col2, col3, col4 = st.columns(4)
precip = col1.number_input("Precipitation (mm)", min_value=0.0, step=0.1)
humidity = col2.number_input("Humidity (%)", min_value=0.0, max_value=100.0, step=1.0)
wind_speed = col3.number_input("Wind Speed (km/h)", min_value=0.0, step=0.1)
condition = col4.selectbox("Weather Condition", df['WeatherCondition'].unique())

if model:
    input_df = pd.DataFrame({
        'Precipitation': [precip],
        'Humidity': [humidity],
        'WindSpeed': [wind_speed],
        'WeatherCondition': [condition]
    })

    # One-hot encode weather condition
    input_df = pd.get_dummies(input_df)
    model_features = model.feature_names_in_
    for col in model_features:
        if col not in input_df.columns:
            input_df[col] = 0  # Add missing dummy cols

    input_df = input_df[model_features]  # Reorder columns

    prediction = model.predict(input_df)[0]
    st.success(f"🌡️ Predicted Temperature: {prediction:.2f} °C")
else:
    st.error("Model not found. Please run the training script first.")
