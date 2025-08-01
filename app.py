import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import joblib
import datetime

# Title
st.title("🌤️ Weather Data Analysis and Temperature Prediction")

# Sidebar
st.sidebar.header("Navigation")
options = ["Explore Historical Data", "Predict Future Temperature"]
choice = st.sidebar.radio("Go to", options)

# Load data
@st.cache_data
def load_data():
    df = pd.read_csv("data/historical_weather_data.csv")
    df['date'] = pd.to_datetime(df['date'])  # make sure your column name is 'date'
    df = df.sort_values('date')
    return df

# Load model
@st.cache_resource
def load_model():
    return joblib.load("model/weather_model.pkl")

df = load_data()

if choice == "Explore Historical Data":
    st.subheader("📊 Historical Temperature Trends")

    # Show basic data
    st.write("First 5 rows of the dataset:")
    st.dataframe(df.head())

    # Date filter
    start_date = st.date_input("Start date", value=df['date'].min().date())
    end_date = st.date_input("End date", value=df['date'].max().date())

    if start_date > end_date:
        st.error("⚠️ End date must be after start date")
    else:
        filtered_df = df[(df['date'] >= pd.to_datetime(start_date)) & (df['date'] <= pd.to_datetime(end_date))]
        st.line_chart(filtered_df.set_index('date')['temperature'])

elif choice == "Predict Future Temperature":
    st.subheader("📈 Predict Temperature for a Future Date")

    # Load model
    model = load_model()

    # Input date
    future_date = st.date_input("Select a future date", value=datetime.date.today())

    if future_date <= df['date'].max().date():
        st.warning("Please select a future date after the last date in the dataset.")
    else:
        future_date_ordinal = pd.to_datetime(future_date).toordinal()
        predicted_temp = model.predict([[future_date_ordinal]])[0]
        st.success(f"🌡️ Predicted Temperature on {future_date}: **{predicted_temp:.2f}°C**")

        # Plot prediction line
        df['date_ordinal'] = df['date'].map(pd.Timestamp.toordinal)
        future_df = pd.DataFrame({
            'date': pd.date_range(df['date'].min(), future_date, freq='D')
        })
        future_df['date_ordinal'] = future_df['date'].map(pd.Timestamp.toordinal)
        future_df['predicted_temp'] = model.predict(future_df[['date_ordinal']])

        plt.figure(figsize=(10,5))
        plt.plot(future_df['date'], future_df['predicted_temp'], label='Predicted Temperature', color='orange')
        plt.plot(df['date'], df['temperature'], label='Historical Temperature', color='blue')
        plt.axvline(pd.to_datetime(future_date), color='red', linestyle='--', label='Prediction Date')
        plt.title("Temperature Trend with Forecast")
        plt.xlabel("Date")
        plt.ylabel("Temperature (°C)")
        plt.legend()
        st.pyplot(plt)

# Footer
st.markdown("---")
st.caption("Made with ❤️ by [Your Name] – Internship Project")
