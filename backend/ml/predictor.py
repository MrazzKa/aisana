import os
import joblib
import json
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

from ml.explainer import explain_prediction


def _load_model_artifacts(model_name: str):
    """Load model, scaler, feature names, and category mappings."""
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

    return model, scaler, feature_names, category_mappings


def _align_features(df: pd.DataFrame, feature_names: list) -> pd.DataFrame:
    for col in feature_names:
        if col not in df.columns:
            df[col] = 0
    return df[feature_names]


def _prepare_dataframe(df: pd.DataFrame, feature_names: list, category_mappings: dict, scaler):
    """Align, encode, impute, and scale a DataFrame. Returns scaled DataFrame."""
    df_aligned = _align_features(df, feature_names)

    for col, mapping in category_mappings.items():
        if col in df_aligned.columns:
            df_aligned[col] = df_aligned[col].astype(str).map(mapping).fillna(-1)

    df_imputed = df_aligned.fillna(0)
    df_scaled = pd.DataFrame(scaler.transform(df_imputed), columns=df_imputed.columns)
    return df_scaled


def predict(df: pd.DataFrame, model_name: str, threshold: float = 0.5):
    model, scaler, feature_names, category_mappings = _load_model_artifacts(model_name)
    df_scaled = _prepare_dataframe(df, feature_names, category_mappings, scaler)

    probas = model.predict_proba(df_scaled)[:, 1] if hasattr(model, "predict_proba") else np.zeros(len(df_scaled))
    is_fraud_pred = (probas >= threshold).astype(int)

    out_preds = []
    fraud_count = 0

    for i in range(len(probas)):
        fraud_val = int(is_fraud_pred[i])
        prob_val = float(probas[i])
        if fraud_val == 1:
            fraud_count += 1

        out_preds.append({
            "fraud_probability": prob_val,
            "is_fraud": fraud_val
        })

    return {
        "predictions": out_preds,
        "summary": {
            "total": len(probas),
            "fraud_count": fraud_count,
            "clean_count": len(probas) - fraud_count,
            "fraud_rate": float(fraud_count / len(probas)) if len(probas) > 0 else 0.0
        }
    }


def predict_with_threshold(df: pd.DataFrame, model_name: str, threshold: float = 0.5):
    """Same as predict but with explicit custom threshold."""
    return predict(df, model_name, threshold=threshold)


def predict_single(data: dict, model_name: str, threshold: float = 0.5) -> dict:
    """Predict fraud for a single transaction from a dict of field values.

    Returns:
        {
            fraud_probability: float,
            is_fraud: int,
            explanation: [{feature, value, contribution}, ...]
        }
    """
    model, scaler, feature_names, category_mappings = _load_model_artifacts(model_name)

    # Build a single-row DataFrame from the input data
    row = {}
    for feat in feature_names:
        if feat in data:
            row[feat] = data[feat]
        else:
            row[feat] = 0
    df = pd.DataFrame([row], columns=feature_names)

    df_scaled = _prepare_dataframe(df, feature_names, category_mappings, scaler)

    proba = float(model.predict_proba(df_scaled)[:, 1][0]) if hasattr(model, "predict_proba") else 0.0
    is_fraud = int(proba >= threshold)

    # Generate explanation
    explanation = explain_prediction(model, scaler, feature_names, category_mappings, data)

    return {
        "fraud_probability": proba,
        "is_fraud": is_fraud,
        "explanation": explanation,
    }
