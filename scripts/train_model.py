# scripts/train_model.py

# ... (keep the import statements at the top the same) ...
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import joblib
import json
import os # Add os import for path handling

def main():
    print("Training model...")
    
    # --- Path Setup ---
    # Make paths robust
    PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    DATA_PATH = os.path.join(PROJECT_ROOT, "data", "training_data.parquet")
    APP_DIR = os.path.join(PROJECT_ROOT, "app")
    
    # Load the processed training data
    training_data = pd.read_parquet(DATA_PATH)
    
    # Separate features and target
    X = training_data.drop(columns=['visitorid', 'target'])
    y = training_data['target']
    
    # Split data for validation
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # --- Model Training ---
    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
    model = xgb.XGBClassifier(
        objective='binary:logistic', eval_metric='logloss', scale_pos_weight=scale_pos_weight,
        learning_rate=0.05, n_estimators=500, max_depth=4, subsample=0.8,
        colsample_bytree=0.8, gamma=0.1, use_label_encoder=False, random_state=42,
        # The early_stopping_rounds parameter moves here, into the constructor
        early_stopping_rounds=50 
    )
    
    # --- CORRECTED .fit() call ---
    # The eval_set is still passed to .fit(), but not early_stopping_rounds
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    
    # --- Evaluate and Save ---
    val_auc = roc_auc_score(y_val, model.predict_proba(X_val)[:, 1])
    print(f"Validation ROC AUC: {val_auc:.4f}")
    
    # Save the model, feature list, and metrics
    os.makedirs(APP_DIR, exist_ok=True) # Ensure app directory exists
    joblib.dump(model, os.path.join(APP_DIR, 'propensity_to_buy_model_v2.pkl'))
    
    features_list = X.columns.tolist()
    with open(os.path.join(APP_DIR, 'features_v2.json'), 'w') as f:
        json.dump(features_list, f)
        
    metrics = {'validation_roc_auc': val_auc}
    with open(os.path.join(APP_DIR, 'metrics.json'), 'w') as f:
        json.dump(metrics, f)
        
    print("Model, features, and metrics saved successfully to the 'app' directory.")

if __name__ == "__main__":
    main()