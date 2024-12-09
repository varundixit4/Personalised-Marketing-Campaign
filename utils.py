import os
import pandas as pd
import joblib
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler
from sklearn.compose import ColumnTransformer
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")

# Load the K-Means model
def load_model(model_path):
    return joblib.load(model_path)

# Load dataset
def load_data(data_path):
    df = pd.read_csv(data_path)
    return df[['Gender', 'Loyalty Member', 'Age', 'Recency', 'Frequency', 'Monetary', 
               'Churn', 'Total Orders', 'Cancellation Rate', 'Add-on Frequency']]

# Preprocess input data
def preprocess_input(df, input_data):
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

    processed_data = final_pipeline.transform(input_data)
    return processed_data

# Function to generate email
def generate_email(input_details, cluster_info):
    genai.configure(api_key=api_key)
    prompt = f"""
    You are an expert marketing email copywriter. Create a personalized marketing email based on the following details:
    
    Customer Details:
    {input_details}
    
    Cluster Characteristics:
    {cluster_info["Characteristics"]}
    
    Behavior:
    {cluster_info["Behavior"]}
    
    Campaign Strategy:
    {cluster_info["Campaign Strategy"]}
    
    Make the email engaging, personalized, and designed to maximize ROI. Only generate the email and nothing else.
    """
    try:
        model = genai.GenerativeModel("gemini-1.5-flash")
        response = model.generate_content(f"{prompt}")
        return response.text
    except Exception as e:
        return f"Error during email generation: {e}"

# Function to generate campaign strategy
def generate_strategy(input_details, cluster_info):
    genai.configure(api_key=api_key)
    prompt = f"""
    You are an expert marketing campaign strategist. Create a personalized marketing strategy based on the following details:
    
    Customer Details:
    {input_details}
    
    Cluster Characteristics:
    {cluster_info["Characteristics"]}
    
    Behavior:
    {cluster_info["Behavior"]}
    
    Campaign Strategy:
    {cluster_info["Campaign Strategy"]}
    
    Make a step by step campaign strategy that is designed to maximize ROI. Only generate the strategy and nothing else.
    """
    try:
        model = genai.GenerativeModel("gemini-1.5-flash")
        response = model.generate_content(f"{prompt}")
        return response.text
    except Exception as e:
        return f"Error during strategy generation: {e}"

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