import os
import io
import json
import time
import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, precision_score, recall_score, roc_auc_score, accuracy_score, confusion_matrix, roc_curve
from imblearn.under_sampling import RandomUnderSampler
from scipy.stats import ks_2samp

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier


def _create_model(model_name: str):
    """Factory function that creates a fresh model instance every time."""
    factories = {
        "RandomForest": lambda: RandomForestClassifier(
            n_estimators=100, class_weight="balanced", random_state=42
        ),
        "GradientBoosting": lambda: GradientBoostingClassifier(
            n_estimators=250, learning_rate=0.1, max_depth=6, random_state=42
        ),
        "CatBoost": lambda: CatBoostClassifier(verbose=0, random_state=42),
        "LightGBM": lambda: LGBMClassifier(random_state=42),
        "XGBoost": lambda: XGBClassifier(
            use_label_encoder=False, eval_metric="logloss", random_state=42
        ),
    }
    if model_name not in factories:
        raise ValueError(f"Model {model_name} is not supported. Available: {list(factories.keys())}")
    return factories[model_name]()


SUPPORTED_MODEL_NAMES = ["RandomForest", "GradientBoosting", "CatBoost", "LightGBM", "XGBoost"]


def train_model(df: pd.DataFrame, model_name: str):
    # Validate model name
    model = _create_model(model_name)

    y = df["IsFraud"].map({"Yes": 1, "No": 0})
    # Fallback to binary values if map failed
    if y.isnull().any():
        y = df["IsFraud"].astype(int)

    X = df.drop(columns=["IsFraud"])

    # Categorical encoding
    category_mappings = {}
    for col in X.select_dtypes(include=["object", "category"]).columns:
        unique_vals = X[col].astype(str).unique()
        mapping = {val: int(i) for i, val in enumerate(unique_vals)}
        X[col] = X[col].astype(str).map(mapping)
        category_mappings[col] = mapping

    # Balancing
    rus = RandomUnderSampler(random_state=42)
    X_res, y_res = rus.fit_resample(X, y)

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(X_res, y_res, test_size=0.2, stratify=y_res, random_state=42)

    # Scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    start_time = time.time()
    model.fit(X_train_scaled, y_train)
    train_time = time.time() - start_time

    # Predictions (Test)
    y_pred = model.predict(X_test_scaled)
    y_proba = model.predict_proba(X_test_scaled)[:, 1] if hasattr(model, "predict_proba") else None

    # Predictions (Train)
    y_train_pred = model.predict(X_train_scaled)
    y_train_proba = model.predict_proba(X_train_scaled)[:, 1] if hasattr(model, "predict_proba") else None

    # Test Metrics
    test_metrics = calculate_metrics(y_test, y_pred, y_proba)
    test_metrics["train_time"] = train_time

    # Train Metrics
    train_metrics = calculate_metrics(y_train, y_train_pred, y_train_proba)
    train_metrics["train_time"] = train_time

    # ROC Curve data
    fpr, tpr, _ = roc_curve(y_test, y_proba)

    # Probability Distribution data
    prob_class_0 = y_proba[y_test == 0].tolist()
    prob_class_1 = y_proba[y_test == 1].tolist()

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred).tolist()

    # Feature Importance
    importances = []
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_.tolist()

    # CSV Predictions Base
    X_test_raw = pd.DataFrame(scaler.inverse_transform(X_test_scaled), columns=X.columns)
    results_df = X_test_raw.copy()
    results_df["Предсказанный класс"] = y_pred
    results_df["Вероятность"] = y_proba
    results_df["Истинный IsFraud"] = y_test.values

    # Sample split format
    sample_split = []
    for t in y_train[:5]:
        sample_split.append({"target": int(t), "subset": "train"})
    for t in y_test[:5]:
        sample_split.append({"target": int(t), "subset": "test"})

    # Save to disk
    os.makedirs("models", exist_ok=True)
    joblib.dump(model, f"models/{model_name}_model.pkl")
    joblib.dump(scaler, f"models/{model_name}_scaler.pkl")
    with open(f"models/{model_name}_features.json", "w", encoding="utf-8") as f:
        json.dump(X.columns.tolist(), f)
    with open(f"models/{model_name}_mappings.json", "w", encoding="utf-8") as f:
        json.dump(category_mappings, f, ensure_ascii=False)

    return {
        "model_name": model_name,
        "test_metrics": test_metrics,
        "train_metrics": train_metrics,
        "roc_curve": {"fpr": fpr.tolist(), "tpr": tpr.tolist()},
        "prob_distribution": {"class_0": prob_class_0, "class_1": prob_class_1},
        "confusion_matrix": cm,
        "feature_importance": {"features": X.columns.tolist(), "importances": importances},
        "features_used": X.columns.tolist(),
        "sample_split": sample_split,
        "sample_prediction": int(y_pred[0]),
        "sample_probability": float(y_proba[0]),
        "predictions_csv": results_df.to_csv(index=False)
    }


def calculate_metrics(y_true, y_pred, y_proba):
    acc = float(accuracy_score(y_true, y_pred))
    prec = float(precision_score(y_true, y_pred))
    rec = float(recall_score(y_true, y_pred))
    auc = float(roc_auc_score(y_true, y_proba)) if y_proba is not None else 0.0
    gini = 2 * auc - 1

    ks = 0.0
    if y_proba is not None:
        ks = float(ks_2samp(y_proba[y_true == 1], y_proba[y_true == 0]).statistic)

    return {
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "roc_auc": auc,
        "gini": float(gini),
        "ks": ks
    }
