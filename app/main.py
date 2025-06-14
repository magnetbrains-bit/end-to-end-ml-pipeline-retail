# app/main.py

from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd
import json
from pathlib import Path  ### NEW ###

# 1. Initialize the FastAPI app
app = FastAPI(title="Propensity to Buy API", version="1.0")

# ### NEW ###: Build paths relative to the current file's location
# This makes the app runnable from anywhere
BASE_DIR = Path(__file__).resolve(strict=True).parent
MODEL_PATH = BASE_DIR / "propensity_to_buy_model_v2.pkl"
FEATURES_PATH = BASE_DIR / "features_v2.json"


# 2. Load the trained model and feature list using the new paths
try:
    model = joblib.load(MODEL_PATH)
    with open(FEATURES_PATH, 'r') as f:
        model_features = json.load(f)
    print("Model and features loaded successfully.")
except Exception as e:
    print(f"Error loading model or features: {e}")
    model = None
    model_features = None

# 3. Define the input data model (No change here)
class UserFeatures(BaseModel):
    features: dict

# 4. Define the root endpoint (No change here)
@app.get("/")
def read_root():
    return {"message": "Welcome to the Propensity to Buy API. Go to /docs for the API documentation."}


# 5. Define the prediction endpoint (No change here)
@app.post("/predict")
def predict_propensity(user_data: UserFeatures):
    """
    Predicts the probability of a user making a transaction.
    
    Accepts a dictionary of user features and returns the prediction probability.
    """
    if not model or not model_features:
        return {"error": "Model not loaded. Please check server logs."}

    try:
        input_df = pd.DataFrame([user_data.features])
        
        for feature in model_features:
            if feature not in input_df.columns:
                input_df[feature] = 0
        
        input_df = input_df[model_features]

        probability = model.predict_proba(input_df)[:, 1]

        return {"visitorid": "unknown", "propensity_to_buy": float(probability[0])}

    except Exception as e:
        return {"error": f"An error occurred during prediction: {e}"}