# dashboard/app.py

import streamlit as st
import requests
import json
import pandas as pd

# --- Page Configuration ---
st.set_page_config(
    page_title="Customer Propensity Predictor",
    page_icon="🚀",
    layout="wide"
)

# --- App Title and Description ---
st.title("Customer Purchase Propensity API 🛍️")
st.markdown("""
    This interactive dashboard allows you to get real-time predictions from a machine learning model.
    Enter the behavioral features of a user below to predict their likelihood of making a purchase in the next 7 days.
    The model is served via a REST API deployed on Render.
""")
st.info("Remember: The free Render instance may spin down. The first prediction might take up to 60 seconds.")


# --- API Configuration ---
# This is the URL of your live API on Render
API_URL = "https://propensity-api-himanshu.onrender.com/predict"


# --- Input Form for User Features ---
st.header("Enter User Features")

# Use columns for a cleaner layout
col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("30-Day Activity")
    total_events_30d = st.number_input("Total Events (Last 30 Days)", min_value=0, value=25, step=1)
    num_views_30d = st.number_input("Views (Last 30 Days)", min_value=0, value=20, step=1)
    num_addtocart_30d = st.number_input("Add-to-Carts (Last 30 Days)", min_value=0, value=4, step=1)
    num_unique_items_30d = st.number_input("Unique Items (Last 30 Days)", min_value=0, value=15, step=1)

with col2:
    st.subheader("7-Day Activity")
    total_events_7d = st.number_input("Total Events (Last 7 Days)", min_value=0, value=8, step=1)
    num_views_7d = st.number_input("Views (Last 7 Days)", min_value=0, value=6, step=1)
    num_addtocart_7d = st.number_input("Add-to-Carts (Last 7 Days)", min_value=0, value=2, step=1)
    num_unique_items_7d = st.number_input("Unique Items (Last 7 Days)", min_value=0, value=5, step=1)

with col3:
    st.subheader("1-Day Activity & Recency")
    total_events_1d = st.number_input("Total Events (Last 24 Hours)", min_value=0, value=3, step=1)
    num_views_1d = st.number_input("Views (Last 24 Hours)", min_value=0, value=3, step=1)
    num_addtocart_1d = st.number_input("Add-to-Carts (Last 24 Hours)", min_value=0, value=0, step=1)
    num_unique_items_1d = st.number_input("Unique Items (Last 24 Hours)", min_value=0, value=2, step=1)

# Recency and Ratio features placed below for emphasis
days_since_last_event = st.slider("Days Since Last Event", 0, 90, 1)
add_to_cart_rate_7d = st.number_input("Add-to-Cart Rate (Last 7 Days)", min_value=0.0, max_value=1.0, value=0.33, step=0.01)

# --- Prediction Logic ---
if st.button("Predict Purchase Propensity", type="primary"):
    
    # 1. Collect all features into a dictionary
    features = {
        "total_events_30d": total_events_30d,
        "num_views_30d": num_views_30d,
        "num_addtocart_30d": num_addtocart_30d,
        "num_unique_items_30d": num_unique_items_30d,
        "total_events_7d": total_events_7d,
        "num_views_7d": num_views_7d,
        "num_addtocart_7d": num_addtocart_7d,
        "num_unique_items_7d": num_unique_items_7d,
        "total_events_1d": total_events_1d,
        "num_views_1d": num_views_1d,
        "num_addtocart_1d": num_addtocart_1d,
        "num_unique_items_1d": num_unique_items_1d,
        "days_since_last_event": days_since_last_event,
        "add_to_cart_rate_7d": add_to_cart_rate_7d
    }

    # 2. Structure the payload for the API
    payload = {"features": features}

    # 3. Call the API and handle the response
    with st.spinner("Calling the API and getting prediction..."):
        try:
            response = requests.post(API_URL, json=payload, timeout=60)
            
            if response.status_code == 200:
                prediction = response.json()
                propensity_score = prediction.get("propensity_to_buy", 0)
                
                # Display the result
                st.subheader("Prediction Result")
                
                # Use a metric for a nice visual display
                st.metric(
                    label="Propensity to Buy Score",
                    value=f"{propensity_score:.2%}",
                    delta=f"{'High Likelihood' if propensity_score > 0.5 else 'Low Likelihood'}"
                )
                
                # Add a progress bar for more visual appeal
                st.progress(propensity_score)
                st.success("Prediction successful!")
                
            else:
                st.error(f"API Error: Received status code {response.status_code}")
                st.json(response.text)

        except requests.exceptions.RequestException as e:
            st.error(f"Could not connect to the API: {e}")