import streamlit as st
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler
from sklearn.compose import ColumnTransformer
import joblib
pd.set_option('future.no_silent_downcasting', True)

# Load the K-Means model
kmeans = joblib.load("./models/kmeans_model.pkl")

# Load the dataset
df = pd.read_csv("./data/new_features_data.csv")
df = df[['Gender', 'Loyalty Member', 'Age', 'Recency', 'Frequency', 'Monetary', 
         'Churn', 'Total Orders', 'Cancellation Rate', 'Add-on Frequency']]

# Streamlit UI
st.title("Customer Segmentation Tool")
st.subheader("Enter customer details to predict the segment")

# Input fields
gender = st.selectbox("Gender", ["Male", "Female"])
loyalty_member = st.selectbox("Loyalty Member", ["Yes", "No"])
age = st.number_input("Age", min_value=18, max_value=100, step=1)
recency = st.number_input("Recency (days since last purchase)", min_value=80, max_value=400, step=1)
frequency = st.number_input("Frequency (purchases in the last year)", min_value=0, step=1)
monetary = st.number_input("Monetary (total spend)", min_value=0.0, step=0.01)
churn = st.number_input("Churn Likeliness", min_value=0.0, max_value=1.0, step=0.5)
total_orders = st.number_input("Total Orders", min_value=0, step=1)
cancellation_rate = st.slider("Cancellation Rate (0-100%)", min_value=0, max_value=100, step=1)
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

# Define binary encoding function
def binary_encode(df):
    return df.replace({'Male': 0, 'Female': 1, 'Yes': 1, 'No': 0})

# Define feature sets
binary_features = ['Gender', 'Loyalty Member', 'Churn']
numerical_features = [
    'Age', 'Recency', 'Frequency', 'Monetary', 
    'Total Orders', 'Cancellation Rate', 'Add-on Frequency'
]

# Define the transformation pipeline
binary_pipeline = Pipeline(steps=[
    ('binary_encoder', FunctionTransformer(binary_encode, validate=False))
])

num_pipeline = Pipeline(steps=[
    ('scaler', StandardScaler())
])

# Apply transformations in the correct order
preprocessor = ColumnTransformer(
    transformers=[
        ('binary', binary_pipeline, binary_features),
        ('numerical', num_pipeline, numerical_features)
    ],
    remainder='drop'
)

# Fit the pipeline on the existing dataset
final_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor)
])

final_pipeline.fit(df)

cluster_lookup = {
    0: {
        "Characteristics": "Older customers (65.4 years), low recency (almost no recent purchases), high monetary value but low frequency.",
        "Behavior": "Low engagement with high churn. Infrequent purchases and high cancellations.",
        "Campaign Strategy": "Focus on retention strategies: offer loyalty rewards, personalized discounts, and reminder emails to reduce churn."
    },
    1: {
        "Characteristics": "Older males (65.2 years), low recency, high purchase volume, and spend.",
        "Behavior": "Occasional high spenders, but high churn with inconsistent engagement.",
        "Campaign Strategy": "Re-engagement with retargeting campaigns, email marketing with personalized content, and exclusive membership benefits."
    },
    2: {
        "Characteristics": "Middle-aged (48.3 years), high frequency and spend, with medium recency.",
        "Behavior": "Active, frequent buyer with diversified interests but no loyalty.",
        "Campaign Strategy": "Use personalized recommendations, reward-based loyalty programs, and exclusive offers to encourage repeat purchases."
    },
    3: {
        "Characteristics": "Middle-aged (49 years), recent buyer, medium frequency.",
        "Behavior": "Frequent buyer with high purchase value, showing potential for loyalty.",
        "Campaign Strategy": "Upsell, cross-sell, and offer VIP benefits to encourage loyalty and repeat purchases."
    },
    4: {
        "Characteristics": "Middle-aged (48.8 years), recent purchases but low frequency.",
        "Behavior": "Occasional but high spenders with a tendency to cancel orders.",
        "Campaign Strategy": "Focus on churn prevention with time-limited discounts, exclusive loyalty offers, and product bundles."
    },
    5: {
        "Characteristics": "Older male (65.4 years), low recency and frequency.",
        "Behavior": "Older demographic, with low churn and occasional purchases.",
        "Campaign Strategy": "Target with senior-friendly promotions, exclusive offers, and gentle re-engagement campaigns."
    },
    6: {
        "Characteristics": "Middle-aged (50.5 years), low recency, medium frequency.",
        "Behavior": "Highly engaged with very low churn. Moderate spending with frequent purchases.",
        "Campaign Strategy": "Convert to loyalty programs with exclusive offers, repeat-purchase discounts, and subscription services."
    },
    7: {
        "Characteristics": "Middle-aged (51.1 years), low recency and frequency.",
        "Behavior": "Frequent and consistent buyer with very low churn.",
        "Campaign Strategy": "Reward consistency with loyalty-based programs, personalized recommendations, and exclusive deals."
    },
    8: {
        "Characteristics": "Young female (33.2 years), recent purchases but infrequent.",
        "Behavior": "Moderate spender with low churn, occasional purchases.",
        "Campaign Strategy": "Engage with social media campaigns and offer bundles, targeted time-limited promotions, and personalized deals."
    },
    9: {
        "Characteristics": "Young female (32.9 years), high frequency but low recency.",
        "Behavior": "One-off shoppers with moderate frequency but high churn.",
        "Campaign Strategy": "Re-engage with discounts, offer exclusive product launches, and provide tailored promotions based on previous purchases."
    },
    10: {
        "Characteristics": "Middle-aged (49.3 years), recent activity but low purchase frequency.",
        "Behavior": "Steady buyer, moderate spender, showing low churn.",
        "Campaign Strategy": "Use loyalty rewards, re-engagement offers, and product recommendations to retain and increase spending."
    },
    11: {
        "Characteristics": "Middle-aged male (49.4 years), both low recency and low frequency.",
        "Behavior": "Infrequent buyer with a very high churn rate.",
        "Campaign Strategy": "Provide first-time buyer discounts, encourage return purchases with exclusive perks or offers."
    },
    12: {
        "Characteristics": "Young male (33.3 years), low recency and frequency.",
        "Behavior": "Moderate buyer with low churn, occasional purchases.",
        "Campaign Strategy": "Target with tailored content based on purchase preferences, offer exclusive time-limited promotions to increase purchase frequency."
    }
}

# Process input data using the fitted pipeline
try:
    processed_data = final_pipeline.transform(input_data)
    processed_df = pd.DataFrame(
        processed_data,
        columns=binary_features + numerical_features
    )
    for col in processed_df.columns:
        processed_df[col] = processed_df[col].astype(float)
except Exception as e:
    st.error(f"Error during preprocessing: {e}")

# Predict and display results
if st.button("Predict Segment"):
    try:
        segment = kmeans.predict(processed_data)
        st.success(f"The customer belongs to Segment: {segment[0]}")
        st.write(f"Behavior: {cluster_lookup[segment[0]]['Behavior']}")
        st.write(f"Campaign Strategy: {cluster_lookup[segment[0]]['Campaign Strategy']}")
    except Exception as e:
        st.error(f"Error during prediction: {e}")