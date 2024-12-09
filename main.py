import streamlit as st
import pandas as pd
from utils import (
    load_model,
    load_data,
    preprocess_input,
    generate_email,
    generate_strategy,
    cluster_lookup
)
from google.generativeai import configure as genai
pd.set_option('future.no_silent_downcasting', True)

# Streamlit UI
st.title("Marketing Campaign Generator")
st.write("Enter customer details to predict the segment and generate personalized email or campaign strategy.")

# Load the K-Means model and dataset
kmeans = load_model("./models/kmeans_model.pkl")
df = load_data("./data/new_features_data.csv")

# Create two columns for side-by-side inputs
col1, col2 = st.columns(2)

# First column
with col1:
    gender = st.selectbox("Gender", ["Male", "Female"])
    age = st.number_input("Age", min_value=18, max_value=100, step=1)
    recency = st.number_input("Recency (days since last purchase)", min_value=80, max_value=400, step=1)
    churn = st.number_input("Churn (If recency more than 180 days)", min_value=0, max_value=1, step=1)
    cancellation_rate = st.number_input("Cancellation Rate (0-100%)", min_value=0, max_value=100, step=1)

# Second column
with col2:
    loyalty_member = st.selectbox("Loyalty Member", ["Yes", "No"])
    frequency = st.number_input("Frequency (purchases in the last year)", min_value=0, step=1)
    monetary = st.number_input("Monetary (total spend)", min_value=0.0, step=0.01)
    total_orders = st.number_input("Total Orders", min_value=0, step=1)
    addon_frequency = st.number_input("Add-on Frequency (per order)", min_value=0, step=1)

# Prepare input data as a DataFrame
input_data = pd.DataFrame([{
    'Gender': gender,
    'Loyalty Member': loyalty_member,
    'Age': age,
    'Recency': recency,
    'Frequency': frequency,
    'Monetary': monetary,
    'Churn': churn,
    'Total Orders': total_orders,
    'Cancellation Rate': cancellation_rate,
    'Add-on Frequency': addon_frequency
}])

# Process input data using the fitted pipeline
try:
    processed_data = preprocess_input(df, input_data)
except Exception as e:
    st.error(f"Error during preprocessing: {e}")

# Layout for buttons
col1, col2, col3 = st.columns(3)

with col1:
    predict_button = st.button("Predict Segment", use_container_width=True)

with col2:
    email_button = st.button("Generate Email", use_container_width=True)

with col3:
    campaign_button = st.button("Generate Campaign Strategy", use_container_width=True)

if predict_button:
    try:
        segment = kmeans.predict(processed_data)
        st.success(f"The customer belongs to Segment: {segment[0]}")
        cluster_info = cluster_lookup[segment[0]]
        st.write(f"Customer Cluster Behavior: {cluster_info['Behavior']}")
        st.write(f"Campaign Strategy: {cluster_info['Campaign Strategy']}")
    except Exception as e:
        st.error(f"Error during prediction: {e}")

elif email_button:
    try:
        segment = kmeans.predict(processed_data)
        cluster_info = cluster_lookup[segment[0]]

        email = generate_email(input_data.to_dict(orient="records")[0], cluster_info)
        if email:
            st.subheader("Generated Email:")
            st.write(email)
    except Exception as e:
        st.error(f"Error during prediction: {e}")

elif campaign_button:
    try:
        segment = kmeans.predict(processed_data)
        cluster_info = cluster_lookup[segment[0]]

        strategy = generate_strategy(input_data.to_dict(orient="records")[0], cluster_info)
        if strategy:
            st.subheader("Generated Campaign Strategy:")
            st.write(strategy)
    except Exception as e:
        st.error(f"Error during prediction: {e}")