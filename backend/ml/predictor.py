import os
import joblib
import json
import pandas as pd
from sklearn.preprocessing import LabelEncoder

def _align_features(df: pd.DataFrame, feature_names: list) -> pd.DataFrame:
    for col in feature_names:
        if col not in df.columns:
            df[col] = 0
    return df[feature_names]

def predict(df: pd.DataFrame, model_name: str):
    model_path = f"models/{model_name}_model.pkl"
    scaler_path = f"models/{model_name}_scaler.pkl"
    features_path = f"models/{model_name}_features.json"

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model {model_name} not found")

    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    with open(features_path, "r", encoding="utf-8") as f:
        feature_names = json.load(f)

    mappings_path = f"models/{model_name}_mappings.json"
    category_mappings = {}
    if os.path.exists(mappings_path):
        with open(mappings_path, "r", encoding="utf-8") as f:
            category_mappings = json.load(f)

    # Process format
    df_aligned = _align_features(df, feature_names)
    
    for col, mapping in category_mappings.items():
        if col in df_aligned.columns:
            df_aligned[col] = df_aligned[col].astype(str).map(mapping).fillna(-1)
            
    # Remove imputer and just fill NaN to 0 so scaler does not crash
    df_imputed = df_aligned.fillna(0)
    df_scaled = pd.DataFrame(scaler.transform(df_imputed), columns=df_imputed.columns)

    predictions = model.predict_proba(df_scaled)[:, 1] if hasattr(model, "predict_proba") else [0]*len(df_scaled)
    is_fraud_pred = model.predict(df_scaled)

    out_preds = []
    fraud_count = 0

    for i in range(len(predictions)):
        fraud_val = int(is_fraud_pred[i])
        prob_val = float(predictions[i])
        if fraud_val == 1:
             fraud_count += 1
        
        out_preds.append({
            "fraud_probability": prob_val,
            "is_fraud": fraud_val
        })

    return {
        "predictions": out_preds,
        "summary": {
            "total": len(predictions),
            "fraud_count": fraud_count,
            "clean_count": len(predictions) - fraud_count,
            "fraud_rate": float(fraud_count / len(predictions)) if len(predictions) > 0 else 0.0
        }
    }
