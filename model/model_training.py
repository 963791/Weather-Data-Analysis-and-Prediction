# model_training.py

import pandas as pd
from sklearn.linear_model import LinearRegression
import joblib

# Load dataset
df = pd.read_csv("data/historical_weather_data.csv")

# Convert date column to datetime and ordinal
df['date'] = pd.to_datetime(df['date'])
df['date_ordinal'] = df['date'].map(pd.Timestamp.toordinal)

# Features and target
X = df[['date_ordinal']]
y = df['temperature']  # Change this if your column name is different

# Train model
model = LinearRegression()
model.fit(X, y)

# Save model
joblib.dump(model, "model/weather_model.pkl")
print("✅ Model trained and saved as 'model/weather_model.pkl'")
