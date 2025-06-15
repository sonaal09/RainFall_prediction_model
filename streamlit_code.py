import streamlit as st
import pickle
import numpy as np
import pandas as pd

# Load the trained model and feature names from the pickle file
with open('rainfall_prediction_model.pkl', 'rb') as file:
    model_data = pickle.load(file)

model = model_data['model']
feature_names = model_data['features_names']  # Should be 7 features

st.title("🌧️ Rainfall Prediction App")
st.write("Enter today's weather data to predict if it will rain.")

# Show expected features (optional debug)
# st.write("Model expects these features:", feature_names)

# Input fields — match the 7 features exactly and in order
pressure = st.number_input("Pressure (hPa)", min_value=998.5, max_value=1034.6, value=1015.0)
maxtemp = st.number_input("Max Temperature (°C)", min_value=7.1, max_value=36.3, value=25.0)
humidity = st.slider("Humidity (%)", min_value=36, max_value=98, value=70)
cloud = st.slider("Cloud Cover (%)", min_value=0, max_value=100, value=50)
sunshine = st.number_input("Sunshine Hours", min_value=0.0, max_value=12.1, value=5.0)
windspeed = st.number_input("Wind Speed (km/h)", min_value=4.4, max_value=59.5, value=10.0)
dewpoint = st.number_input("Dew Point (°C)", min_value=-0.4, max_value=26.7, value=12.0)

# Combine inputs into a DataFrame in the same order as training
input_data = [pressure, maxtemp, humidity, cloud, sunshine, windspeed, dewpoint]
input_df = pd.DataFrame([input_data], columns=feature_names)

# Predict on button click
if st.button("Predict Rainfall"):
    prediction = model.predict(input_df)
    if prediction[0] == 'yes':
        st.success("🌧️ It will likely rain today.")
    else:
        st.info("☀️ No rain expected today.")
